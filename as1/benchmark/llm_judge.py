# -*- coding: utf-8 -*-
"""
LLM Judge Evaluation System for MCP-Bench.

Ported from mcp-bench project, adapted to use agentscope models.
Provides 6-dimension evaluation of agent task execution:
  1. Task Fulfillment
  2. Grounding
  3. Tool Appropriateness
  4. Parameter Accuracy
  5. Dependency Awareness
  6. Parallelism & Efficiency

Supports multiple judge model providers:
  - Standard OpenAI-compatible (qwen, gpt-5, etc.)
  - vertex_ai.claude-opus-4-6 (Claude via DashScope)
  - vertex_ai.gemini-3.1-pro-preview (Gemini via DashScope)
"""

import asyncio
import json
import logging
import os
import random
import time
from typing import List, Dict, Any, Optional, Set

logger = logging.getLogger(__name__)

# Models that require max_completion_tokens instead of max_tokens
MODELS_WITH_MAX_COMPLETION_TOKENS: Set[str] = {
    "o1-preview", "o1-mini", "o4-mini", "o3-mini", "o3",
    "gpt-4o", "gpt-4o-mini", "gpt-5",
}


def _is_claude_model(model_name: str) -> bool:
    return "claude" in model_name.lower()


def _is_gemini_model(model_name: str) -> bool:
    return "gemini" in model_name.lower()


def _should_use_max_completion_tokens(model_name: str) -> bool:
    m = model_name.lower()
    if model_name in MODELS_WITH_MAX_COMPLETION_TOKENS:
        return True
    if "gpt-5" in m:
        return True
    if any(m.startswith(p) for p in ("o1-", "o3-", "o4-")):
        return True
    return False


def safe_get(item, key, default=None):
    """Safely get a value from a dictionary."""
    if isinstance(item, dict):
        return item.get(key, default)
    return default


class DirectLLMProvider:
    """Direct LLM provider using AsyncOpenAI / raw HTTP.

    Supports Claude, Gemini, and standard OpenAI-compatible models
    via DashScope or any OpenAI-compatible endpoint.
    Ported from mcp-bench's LLMProvider.

    Usage:
        provider = DirectLLMProvider.from_model_name("vertex_ai.claude-opus-4-6")
        text = await provider.get_completion("system", "user", 15000)
    """

    def __init__(self, client, model_name: str):
        self.client = client
        self.model_name = model_name
        self._api_key = getattr(client, "api_key", None)
        self._base_url = str(getattr(client, "base_url", "")).rstrip("/")

    @classmethod
    def from_model_name(cls, model_name: str) -> "DirectLLMProvider":
        """Create a provider from model name, auto-detecting API config."""
        from openai import AsyncOpenAI

        api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("BASE_URL") or os.environ.get(
            "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        if not api_key:
            raise ValueError(
                "No API key found. Set DASHSCOPE_API_KEY or OPENAI_API_KEY."
            )

        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        return cls(client, model_name)

    def _build_params(self, system_prompt: str, user_prompt: str, max_tokens: int) -> dict:
        """Build request params adapted to model's native API format."""
        model = self.model_name

        if _is_claude_model(model):
            return {
                "model": model,
                "messages": [{"role": "user", "content": user_prompt}],
                "max_tokens": max_tokens,
                "extra_body": {"system": system_prompt},
            }

        if _is_gemini_model(model):
            return {
                "model": model,
                "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
                "system_instruction": {"parts": [{"text": system_prompt}]},
                "generationConfig": {"maxOutputTokens": max_tokens},
            }

        # Standard OpenAI
        params = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if _should_use_max_completion_tokens(model):
            params["max_completion_tokens"] = max_tokens
        else:
            params["max_tokens"] = max_tokens
        return params

    async def _raw_post(self, url: str, body: dict) -> dict:
        """Send raw HTTP POST (for Claude/Gemini native response)."""
        import httpx

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=300) as http:
            resp = await http.post(url, json=body, headers=headers)
            if resp.status_code >= 400:
                raise Exception(f"HTTP {resp.status_code}: {resp.text[:500]}")
            return resp.json()

    async def _call_api(self, params: dict) -> str:
        """Call API and return text content."""
        model = self.model_name

        if _is_claude_model(model):
            extra = params.pop("extra_body", {})
            body = {**params, **extra}
            data = await self._raw_post(f"{self._base_url}/chat/completions", body)
            content = ""
            for block in data.get("content", []):
                if block.get("type") == "text":
                    content += block.get("text", "")
            return content

        if _is_gemini_model(model):
            data = await self._raw_post(f"{self._base_url}/chat/completions", params)
            content = ""
            for candidate in data.get("candidates", []):
                for part in candidate.get("content", {}).get("parts", []):
                    if "text" in part and not part.get("thought"):
                        content += part["text"]
            return content

        # Standard OpenAI SDK
        extra_body = params.pop("extra_body", None)
        if extra_body:
            params["extra_body"] = extra_body
        response = await self.client.chat.completions.create(**params)
        return response.choices[0].message.content

    async def get_completion(
        self, system_prompt: str, user_prompt: str, max_tokens: int = 15000
    ) -> str:
        """Get completion with retry."""
        params = self._build_params(system_prompt, user_prompt, max_tokens)
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                content = await self._call_api(params)
                if not content or not content.strip():
                    raise ValueError("Empty response from LLM")
                return content.strip()
            except Exception as e:
                logger.warning(f"Attempt {attempt+1}/{max_attempts} failed for {self.model_name}: {e}")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise

    @staticmethod
    def clean_and_parse_json(raw_json: str) -> Any:
        """Parse JSON from LLM response."""
        text = raw_json.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            parts = text.split("```")
            if len(parts) >= 2:
                text = parts[1].strip()

        text = text.strip()
        if not text.startswith("{") and not text.startswith("["):
            start = text.find("{")
            if start != -1:
                text = text[start:]

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        try:
            import json_repair
            return json_repair.loads(text)
        except (ImportError, Exception):
            pass

        # Last resort
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Failed to parse JSON: {text[:200]}...")


def create_judge_provider(model_name: str):
    """Create appropriate LLM provider for the judge model.

    Supports:
      - vertex_ai.claude-opus-4-6 (Claude via DashScope)
      - vertex_ai.gemini-3.1-pro-preview (Gemini via DashScope)
      - Any OpenAI-compatible model (qwen3-max, gpt-5, etc.)

    For Claude/Gemini models, uses DirectLLMProvider (raw HTTP for native formats).
    For standard models, also uses DirectLLMProvider (AsyncOpenAI SDK).
    """
    return DirectLLMProvider.from_model_name(model_name)


class AgentScopeLLMAdapter:
    """Adapter to wrap agentscope model for LLMJudge use.

    Provides get_completion() and clean_and_parse_json() interfaces
    expected by the evaluator, backed by an agentscope ChatModel.
    """

    def __init__(self, model, formatter):
        self.model = model
        self.formatter = formatter

    async def get_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 15000,
    ) -> str:
        """Get a completion from the agentscope model."""
        from agentscope.message import Msg

        msgs = [
            Msg("system", system_prompt, role="system"),
            Msg("user", user_prompt, role="user"),
        ]
        formatted = await self.formatter.format(msgs=msgs)
        response = await self.model(formatted)

        # Extract text from response
        if hasattr(response, "text"):
            return response.text
        if hasattr(response, "content"):
            if isinstance(response.content, str):
                return response.content
            if isinstance(response.content, list):
                for block in response.content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        return block.get("text", "")
        return str(response)

    @staticmethod
    def clean_and_parse_json(raw_json: str) -> Any:
        """Parse JSON from LLM response, handling markdown fences."""
        text = raw_json.strip()

        # Strip markdown code fences
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first line (```json or ```)
            lines = lines[1:]
            # Remove last line if it's ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try json_repair if available
        try:
            import json_repair
            return json_repair.loads(text)
        except (ImportError, Exception):
            pass

        # Last resort: find JSON object in text
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Failed to parse JSON from response: {text[:200]}...")


class LLMJudge:
    """6-dimension LLM-based evaluation of agent task performance.

    Evaluation Dimensions:
      Task Completion:
        1. Task Fulfillment
        2. Grounding
      Tool Usage:
        3. Tool Appropriateness
        4. Parameter Accuracy
      Planning Effectiveness:
        5. Dependency Awareness
        6. Parallelism & Efficiency
    """

    def __init__(
        self,
        llm_provider,
        enable_judge_stability: bool = False,
    ):
        """Initialize the LLM Judge.

        Args:
            llm_provider: An object with get_completion() and clean_and_parse_json().
                Can be AgentScopeLLMAdapter or mcp-bench LLMProvider.
            enable_judge_stability: Run 5 randomized evaluations and average.
        """
        self.llm = llm_provider
        self.enable_judge_stability = enable_judge_stability

        self.evaluation_dimensions = {
            "Task Completion": {
                "Task Fulfillment": {
                    "1-3": "Perfectly completes 10-30% of requirements.",
                    "4-6": "Perfectly completes 40-60% of requirements.",
                    "7-8": "Perfectly completes 70-80% of requirements.",
                    "9-10": "Perfectly completes 90-100% of requirements.",
                },
                "Grounding": {
                    "1-3": "10-30% of claims are perfectly grounded in tool outputs.",
                    "4-6": "40-60% of claims are perfectly grounded in tool outputs.",
                    "7-8": "70-80% of claims are perfectly grounded in tool outputs.",
                    "9-10": "90-100% of claims are perfectly grounded in tool outputs.",
                },
            },
            "Tool Usage": {
                "Tool Appropriateness": {
                    "1-3": "10-30% of tools were perfectly selected for their subtasks.",
                    "4-6": "40-60% of tools were perfectly selected for their subtasks.",
                    "7-8": "70-80% of tools were perfectly selected for their subtasks.",
                    "9-10": "90-100% of tools were perfectly selected for their subtasks.",
                },
                "Parameter Accuracy": {
                    "1-3": "10-30% of tool calls have perfectly accurate and complete parameters.",
                    "4-6": "40-60% of tool calls have perfectly accurate and complete parameters.",
                    "7-8": "70-80% of tool calls have perfectly accurate and complete parameters.",
                    "9-10": "90-100% of tool calls have perfectly accurate and complete parameters.",
                },
            },
            "Planning Effectiveness and Efficiency": {
                "Dependency Awareness": {
                    "1-3": "10-30% of dependency chains are perfectly executed.",
                    "4-6": "40-60% of dependency chains are perfectly executed.",
                    "7-8": "70-80% of dependency chains are perfectly executed.",
                    "9-10": "90-100% of dependency chains are perfectly executed.",
                },
                "Parallelism and Efficiency": {
                    "1-3": "More than 70% redundant calls OR less than 30% of parallelizable tasks were executed in parallel.",
                    "4-6": "40-60% redundant calls OR 40-60% of parallelizable tasks were executed in parallel.",
                    "7-8": "20-30% redundant calls AND 70-80% of parallelizable tasks were executed in parallel.",
                    "9-10": "Less than 10% redundant calls AND 90-100% of parallelizable tasks were executed in parallel.",
                },
            },
        }

    def _format_available_tools(self, available_tools: Dict[str, Any]) -> str:
        """Format available tools for the evaluation prompt."""
        if not available_tools:
            return "No tools available"

        servers = {}
        for tool_name, tool_info in available_tools.items():
            server = tool_info.get("server", "Unknown")
            if server not in servers:
                servers[server] = []
            desc = tool_info.get("description", "No description available") or "No description available"
            if len(desc) > 500:
                desc = desc[:500] + "..."
            servers[server].append({"name": tool_name, "description": desc})

        lines = []
        for server, tools in sorted(servers.items()):
            lines.append(f"[{server}] ({len(tools)} tools)")
            for tool in tools:
                lines.append(f"  - {tool['name']}: {tool['description']}")
            lines.append("")
        return "\n".join(lines).strip() if lines else "No tools available"

    def _create_execution_summary(
        self,
        execution_results: List[Dict[str, Any]],
        total_rounds: int,
        accumulated_information: str = None,
    ) -> str:
        """Create execution summary for the judge prompt."""
        summary_parts = []

        if not execution_results:
            summary_parts.append("No tools were executed.")
        else:
            successful = [r for r in execution_results if safe_get(r, "success")]
            failed = [r for r in execution_results if not safe_get(r, "success")]

            summary_parts.extend([
                f"Total rounds: {total_rounds}",
                f"Tools executed: {len(execution_results)}",
                f"Successful: {len(successful)}",
                f"Failed: {len(failed)}",
            ])

            if successful:
                names = [safe_get(r, "tool", "unknown") for r in successful]
                summary_parts.append(f"Successful tools: {', '.join(names)}")
            if failed:
                names = [safe_get(r, "tool", "unknown") for r in failed]
                summary_parts.append(f"Failed tools: {', '.join(names)}")

        stats = "; ".join(summary_parts)

        # Append per-call parameter details if available
        has_params = any(safe_get(r, "parameters") for r in (execution_results or []))
        if has_params:
            call_lines = []
            for i, r in enumerate(execution_results, 1):
                tool = safe_get(r, "tool", "unknown")
                params = safe_get(r, "parameters", {})
                success = safe_get(r, "success", False)
                status = "OK" if success else "FAILED"
                params_str = json.dumps(params, ensure_ascii=False)
                if len(params_str) > 500:
                    params_str = params_str[:500] + "..."
                call_lines.append(f"  {i}. [{status}] {tool}({params_str})")
            stats += "\n\n--- TOOL CALL DETAILS (with parameters) ---\n" + "\n".join(call_lines)

        if accumulated_information:
            stats += f"\n\n--- ACCUMULATED INFORMATION FROM EXECUTION ---\n{accumulated_information}"
        return stats

    def _build_eval_prompt(
        self,
        task: str,
        final_solution: str,
        execution_summary: str,
        total_rounds: int,
        available_tools: Dict[str, Any] = None,
        concrete_task_description: str = None,
        dependency_analysis: str = None,
    ) -> str:
        """Build the standard (non-stability) evaluation prompt."""
        # Task section
        if concrete_task_description:
            task_section = f"""
**TASK PRESENTED TO AGENT**: "{task}"

**CONCRETE TASK REFERENCE (For evaluation context only)**:
Note: The agent did NOT see this concrete version. It only saw the task above.
"{concrete_task_description}"
"""
        else:
            task_section = f'**ORIGINAL TASK**: "{task}"'

        dependency_section = ""
        if dependency_analysis:
            dependency_section = f"""
**DEPENDENCY ANALYSIS (Reference Only)**:
{dependency_analysis}
"""

        return f"""You are a STRICT evaluator. Your role is to critically assess performance with HIGH STANDARDS.

IMPORTANT: The average score across all evaluations should be around 4-5, NOT 7-8.

You must assign scores **only based on evidence** from the task, solution, and tool usage.

---

**AVAILABLE TOOLS** ({len(available_tools) if available_tools else 0} tools):
{self._format_available_tools(available_tools)}
{task_section}
**EXECUTION SUMMARY**:
{execution_summary}
**FINAL SOLUTION**: "{final_solution}"
**TOTAL ROUNDS**: {total_rounds}

{dependency_section}

---

### Task Completion Rubric (1-10 per subdimension)

1. **Task Fulfillment and Quality**
- 1-3: Perfectly completes 10-30% of requirements.
- 4-6: Perfectly completes 40-60% of requirements.
- 7-8: Perfectly completes 70-80% of requirements.
- 9-10: Perfectly completes 90-100% of requirements.

2. **Grounding**
- 1-3: 10-30% of claims are perfectly grounded in tool outputs.
- 4-6: 40-60% of claims are perfectly grounded in tool outputs.
- 7-8: 70-80% of claims are perfectly grounded in tool outputs.
- 9-10: 90-100% of claims are perfectly grounded in tool outputs.

---

### Tool Usage Rubric (1-10 per subdimension)

1. **Tool Appropriateness**
- 1-3: 10-30% of tools were perfectly selected for their subtasks.
- 4-6: 40-60% of tools were perfectly selected for their subtasks.
- 7-8: 70-80% of tools were perfectly selected for their subtasks.
- 9-10: 90-100% of tools were perfectly selected for their subtasks.

2. **Parameter Accuracy**
- 1-3: 10-30% of tool calls have perfectly accurate and complete parameters.
- 4-6: 40-60% of tool calls have perfectly accurate and complete parameters.
- 7-8: 70-80% of tool calls have perfectly accurate and complete parameters.
- 9-10: 90-100% of tool calls have perfectly accurate and complete parameters.

---

### Planning Effectiveness and Efficiency (1-10 per subdimension)

1. **Dependency Awareness**
- 1-3: 10-30% of dependency chains are perfectly executed.
- 4-6: 40-60% of dependency chains are perfectly executed.
- 7-8: 70-80% of dependency chains are perfectly executed.
- 9-10: 90-100% of dependency chains are perfectly executed.

2. **Parallelism and Efficiency**
- 1-3: More than 70% redundant calls OR less than 30% parallel.
- 4-6: 40-60% redundant calls OR 40-60% parallel.
- 7-8: 20-30% redundant calls AND 70-80% parallel.
- 9-10: Less than 10% redundant calls AND 90-100% parallel.

---

### SCORING GUIDELINES:
- Default to 4-5 unless you have strong evidence for higher.
- Score based on COMPLETION PERCENTAGES and PROPORTIONAL SUCCESS.
- Most real-world executions should score 4-6. Scores of 8+ should be EXCEPTIONAL.

Return your evaluation in this exact JSON format:
{{
  "task_fulfillment_reasoning": "...",
  "grounding_reasoning": "...",
  "tool_appropriateness_reasoning": "...",
  "parameter_accuracy_reasoning": "...",
  "dependency_awareness_reasoning": "...",
  "parallelism_efficiency_reasoning": "...",

  "task_fulfillment": X,
  "grounding": X,
  "tool_appropriateness": X,
  "parameter_accuracy": X,
  "dependency_awareness": X,
  "parallelism_and_efficiency": X
}}

Return **only** the JSON object."""

    def _calculate_average_scores(self, all_scores: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Average scores from multiple stability evaluations."""
        if not all_scores:
            raise ValueError("No scores to average")

        score_fields = [
            "task_fulfillment", "grounding",
            "tool_appropriateness", "parameter_accuracy",
            "dependency_awareness", "parallelism_and_efficiency",
        ]

        averaged = {}

        # Copy non-score fields from first result
        for key in all_scores[0]:
            if key not in score_fields:
                averaged[key] = all_scores[0][key]

        # Average numeric scores
        for field in score_fields:
            valid = [r[field] for r in all_scores if field in r and isinstance(r[field], (int, float))]
            averaged[field] = sum(valid) / len(valid) if valid else 0

        # Calculate aggregate scores
        averaged["task_completion_score"] = (averaged["task_fulfillment"] + averaged["grounding"]) / 2
        averaged["tool_selection_score"] = (averaged["tool_appropriateness"] + averaged["parameter_accuracy"]) / 2
        averaged["planning_effectiveness_and_efficiency_score"] = (
            averaged["dependency_awareness"] + averaged["parallelism_and_efficiency"]
        ) / 2

        return averaged

    async def evaluate_task_performance(
        self,
        task: str,
        final_solution: str,
        execution_results: List[Dict[str, Any]],
        total_rounds: int,
        available_tools: Dict[str, Any],
        accumulated_information: str = None,
        concrete_task_description: str = None,
        dependency_analysis: str = None,
    ) -> Dict[str, Any]:
        """Evaluate task performance using LLM judge with 6-dimension scoring.

        Args:
            task: Task description as presented to the agent.
            final_solution: The agent's final answer.
            execution_results: List of tool execution result dicts.
            total_rounds: Number of execution rounds/iterations.
            available_tools: Dict of available tools with schemas.
            accumulated_information: Full execution history text.
            concrete_task_description: Original concrete task for reference.
            dependency_analysis: Tool dependency analysis text.

        Returns:
            Dict with 6 dimension scores + 3 aggregate scores + reasoning.
        """
        execution_summary = self._create_execution_summary(
            execution_results, total_rounds, accumulated_information,
        )

        if self.enable_judge_stability:
            return await self._evaluate_with_stability(
                task, final_solution, execution_summary, total_rounds,
                available_tools, concrete_task_description, dependency_analysis,
            )

        # Standard single evaluation
        prompt = self._build_eval_prompt(
            task, final_solution, execution_summary, total_rounds,
            available_tools, concrete_task_description, dependency_analysis,
        )

        start_time = time.time()
        response = await self.llm.get_completion(
            "You are an expert AI task execution evaluator. Score each dimension objectively based on evidence.",
            prompt,
            15000,
        )
        llm_time = time.time() - start_time
        logger.info(f"LLM Judge completed in {llm_time:.2f}s")

        result = self.llm.clean_and_parse_json(response)

        # Extract scores
        tf = result.get("task_fulfillment", 0)
        gr = result.get("grounding", 0)
        ta = result.get("tool_appropriateness", 0)
        pa = result.get("parameter_accuracy", 0)
        da = result.get("dependency_awareness", 0)
        pe = result.get("parallelism_and_efficiency", 0)

        return {
            "task_fulfillment": tf,
            "grounding": gr,
            "tool_appropriateness": ta,
            "parameter_accuracy": pa,
            "dependency_awareness": da,
            "parallelism_and_efficiency": pe,
            "task_completion_score": (tf + gr) / 2,
            "tool_selection_score": (ta + pa) / 2,
            "planning_effectiveness_and_efficiency_score": (da + pe) / 2,
            "task_fulfillment_reasoning": result.get("task_fulfillment_reasoning", ""),
            "grounding_reasoning": result.get("grounding_reasoning", ""),
            "tool_appropriateness_reasoning": result.get("tool_appropriateness_reasoning", ""),
            "parameter_accuracy_reasoning": result.get("parameter_accuracy_reasoning", ""),
            "dependency_awareness_reasoning": result.get("dependency_awareness_reasoning", ""),
            "parallelism_efficiency_reasoning": result.get("parallelism_efficiency_reasoning", ""),
        }

    async def _evaluate_with_stability(
        self, task, final_solution, execution_summary, total_rounds,
        available_tools, concrete_task_description, dependency_analysis,
    ) -> Dict[str, Any]:
        """Run 5 randomized evaluations and average scores."""
        logger.info("Running LLM Judge stability testing (5 randomized evaluations)")
        all_scores = []

        for i in range(5):
            try:
                # Generate randomized prompt (shuffle dimension order)
                prompt = self._generate_randomized_prompt(
                    task, final_solution, execution_summary, total_rounds,
                    available_tools, concrete_task_description, dependency_analysis,
                )
                response = await self.llm.get_completion(
                    "You are an expert AI task execution evaluator. Score each dimension objectively based on evidence.",
                    prompt, 15000,
                )
                parsed = self.llm.clean_and_parse_json(response)
                all_scores.append(parsed)
                logger.debug(f"Stability evaluation {i+1}/5 completed")
            except Exception as e:
                logger.warning(f"Stability evaluation {i+1}/5 failed: {e}")

        if not all_scores:
            raise RuntimeError("All stability evaluations failed")

        return self._calculate_average_scores(all_scores)

    def _generate_randomized_prompt(
        self, task, final_solution, execution_summary, total_rounds,
        available_tools, concrete_task_description, dependency_analysis,
    ) -> str:
        """Generate evaluation prompt with randomized dimension order."""
        # Randomize dimension ordering
        dims_copy = {}
        for main_dim, sub_dims in self.evaluation_dimensions.items():
            dims_copy[main_dim] = {sd: dict(c) for sd, c in sub_dims.items()}

        main_names = list(dims_copy.keys())
        random.shuffle(main_names)

        parts = []
        parts.append("""You are an impartial evaluator judging the quality of an AI agent's multi-server tool-based task execution.
You must assign scores **only based on evidence**.""")

        if concrete_task_description:
            parts.append(f'**TASK PRESENTED TO AGENT**: "{task}"\n**CONCRETE TASK REFERENCE**: "{concrete_task_description}"')
        else:
            parts.append(f'**ORIGINAL TASK**: "{task}"')

        if dependency_analysis:
            parts.append(f"**DEPENDENCY ANALYSIS**:\n{dependency_analysis}")

        parts.append(f"""**FINAL SOLUTION**: "{final_solution}"
**TOTAL ROUNDS**: {total_rounds}
**EXECUTION SUMMARY**:
{execution_summary}
**AVAILABLE TOOLS** ({len(available_tools) if available_tools else 0} tools):
{self._format_available_tools(available_tools)}
---""")

        for main_dim in main_names:
            sub_dims = dims_copy[main_dim]
            sub_names = list(sub_dims.keys())
            random.shuffle(sub_names)
            parts.append(f"\n### {main_dim} Rubric (1-10 per subdimension)\n")
            for i, sub_dim in enumerate(sub_names, 1):
                criteria = sub_dims[sub_dim]
                items = list(criteria.items())
                random.shuffle(items)
                parts.append(f"{i}. **{sub_dim}**")
                for rng, desc in items:
                    parts.append(f"   - {rng}: {desc}")
                parts.append("")

        parts.append("""---
Default to 4-5 unless strong evidence for higher.
Return your evaluation in JSON format:
{
  "task_fulfillment_reasoning": "...",
  "grounding_reasoning": "...",
  "tool_appropriateness_reasoning": "...",
  "parameter_accuracy_reasoning": "...",
  "dependency_awareness_reasoning": "...",
  "parallelism_efficiency_reasoning": "...",
  "task_fulfillment": X,
  "grounding": X,
  "tool_appropriateness": X,
  "parameter_accuracy": X,
  "dependency_awareness": X,
  "parallelism_and_efficiency": X
}
Return **only** the JSON object.""")

        return "\n".join(parts)


class TaskEvaluator:
    """Comprehensive task evaluator combining LLM judge + tool accuracy metrics.

    This is the main entry point for evaluating MCP task execution results.
    """

    def __init__(self, llm_provider, enable_judge_stability: bool = False):
        self.llm_judge = LLMJudge(llm_provider, enable_judge_stability)

    async def evaluate(
        self,
        task: str,
        execution_results: List[Dict[str, Any]],
        final_solution: str,
        total_rounds: int,
        available_tools: Dict[str, Any],
        planning_json_compliance: float = None,
        accumulated_information: str = None,
        concrete_task_description: str = None,
        dependency_analysis: str = None,
    ) -> Optional[Dict[str, Any]]:
        """Comprehensive evaluation of task execution quality.

        Args:
            task: Task description.
            execution_results: List of tool execution results.
            final_solution: Agent's final answer.
            total_rounds: Number of iterations.
            available_tools: Available MCP tools dict.
            planning_json_compliance: Pre-calculated compliance ratio.
            accumulated_information: Execution history context.
            concrete_task_description: Concrete task for reference.
            dependency_analysis: Tool dependency analysis.

        Returns:
            Dict with all evaluation metrics, or None on failure.
        """
        logger.info("Starting comprehensive task evaluation...")
        try:
            llm_scores = await self.llm_judge.evaluate_task_performance(
                task, final_solution, execution_results, total_rounds,
                available_tools, accumulated_information,
                concrete_task_description, dependency_analysis,
            )

            tool_metrics = self._calculate_tool_accuracy_metrics(
                execution_results, available_tools, planning_json_compliance,
            )

            server_metrics = self._calculate_server_utilization_metrics(execution_results)

            evaluation = {
                **llm_scores,
                **tool_metrics,
                "server_utilization_metrics": server_metrics,
            }

            logger.info("Task evaluation completed successfully")
            return evaluation

        except Exception as e:
            logger.error(f"Task evaluation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _calculate_tool_accuracy_metrics(
        self,
        execution_results: List[Dict[str, Any]],
        available_tools: Dict[str, Any],
        planning_json_compliance: float = None,
    ) -> Dict[str, Any]:
        """Calculate tool accuracy metrics."""
        if planning_json_compliance is None:
            planning_json_compliance = 1.0

        if not execution_results:
            return {
                "input_schema_compliance": None,
                "valid_tool_name_rate": None,
                "execution_success_rate": None,
                "valid_call_failure_rate": None,
                "planning_json_compliance": planning_json_compliance,
            }

        valid_tool_calls = 0
        successful_executions = 0
        valid_call_failures = 0
        total_calls = len(execution_results)

        for result in execution_results:
            tool_name = safe_get(result, "tool", "")
            success = safe_get(result, "success", False)
            is_valid = tool_name in (available_tools or {})

            if is_valid:
                valid_tool_calls += 1
                if not success:
                    valid_call_failures += 1
            if success:
                successful_executions += 1

        return {
            "input_schema_compliance": None,  # Cannot validate without full params
            "valid_tool_name_rate": valid_tool_calls / total_calls if total_calls > 0 else 0.0,
            "execution_success_rate": successful_executions / total_calls if total_calls > 0 else 0.0,
            "valid_call_failure_rate": valid_call_failures / valid_tool_calls if valid_tool_calls > 0 else 0.0,
            "planning_json_compliance": planning_json_compliance,
        }

    @staticmethod
    def _calculate_server_utilization_metrics(
        execution_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Calculate server utilization metrics."""
        if not execution_results:
            return {
                "server_count": 0,
                "cross_server_coordination": False,
                "server_distribution": {},
            }

        servers_used = set()
        server_counts = {}
        for result in execution_results:
            server = safe_get(result, "server", "")
            if server:
                servers_used.add(server)
                server_counts[server] = server_counts.get(server, 0) + 1

        return {
            "server_count": len(servers_used),
            "cross_server_coordination": len(servers_used) > 1,
            "server_distribution": server_counts,
        }

    @staticmethod
    def build_execution_results_from_tool_calls(
        tool_calls: Dict[str, int],
        mcp_tools: Dict[str, Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Build execution_results list from MCPWorker's tool_calls dict.

        MCPWorker only tracks {tool_name: call_count}. This builds a
        compatible execution_results list for the evaluator.

        Args:
            tool_calls: Dict of {tool_name: count} from MCPWorker.
            mcp_tools: Optional dict of tool metadata with server info.

        Returns:
            List of execution result dicts compatible with the evaluator.
        """
        results = []
        for tool_name, count in tool_calls.items():
            server = ""
            if mcp_tools and tool_name in mcp_tools:
                server = mcp_tools[tool_name].get("server", "")
            elif "__" in tool_name:
                # MCPWorker uses "ServerName__tool_name" format
                server = tool_name.split("__")[0]

            for _ in range(count):
                results.append({
                    "tool": tool_name,
                    "success": True,  # MCPWorker doesn't track per-call success
                    "server": server,
                    "parameters": {},
                })
        return results
