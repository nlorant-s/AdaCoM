# -*- coding: utf-8 -*-
"""Judge utilities for BrowseComp-Plus."""

import json
import re
import traceback
from agentscope.message import Msg
from .retry import retry_model_call

def parse_judge_response(judge_response: str) -> dict:
    """Parse judge model response (expects JSON format).

    Returns:
        dict with keys: extracted_answer, ground_truth, reasoning, score, parse_error
    """
    result = {
        "extracted_answer": None,
        "ground_truth": None,
        "reasoning": None,
        "score": 0.0,
        "parse_error": False,
    }

    if not judge_response or not judge_response.strip():
        result["parse_error"] = True
        result["reasoning"] = "Empty response from judge model"
        return result

    # Try to extract JSON from the response
    # Sometimes the model might add text before/after the JSON
    json_match = re.search(r"\{.*\}", judge_response, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
    else:
        json_str = judge_response.strip()

    try:
        parsed = json.loads(json_str)

        # Validate required fields
        if "score" not in parsed:
            # Fallback for backward compatibility if model outputs is_correct
            if "is_correct" in parsed:
                 parsed["score"] = 1.0 if parsed["is_correct"] else 0.0
            else:
                result["parse_error"] = True
                result["reasoning"] = "Missing 'score' field in JSON response"
                return result

        # Extract fields
        result["extracted_answer"] = parsed.get("extracted_answer")
        result["ground_truth"] = parsed.get("ground_truth")
        result["reasoning"] = parsed.get("reasoning", "")
        
        # Handle score
        try:
            result["score"] = float(parsed.get("score", 0.0))
        except (ValueError, TypeError):
            result["score"] = 0.0
            result["parse_error"] = True
            result["reasoning"] = "Invalid score format"

        # Additional validation
        if result["extracted_answer"] is None:
            # If no answer was extracted, it should be marked as incorrect
            result["score"] = 0.0

    except json.JSONDecodeError as e:
        result["parse_error"] = True
        result["reasoning"] = f"JSON parsing error: {str(e)}"

        # Fallback: try to extract score with regex
        score_match = re.search(
            r'"score"\s*:\s*([0-9.]+)', judge_response, re.IGNORECASE
        )
        if score_match:
            try:
                result["score"] = float(score_match.group(1))
                result["parse_error"] = False
                result["reasoning"] = "Partial parse via regex"
            except ValueError:
                pass
        else:
             # Fallback for is_correct
             is_correct_match = re.search(
                r'"is_correct"\s*:\s*(true|false)', judge_response, re.IGNORECASE
             )
             if is_correct_match:
                 result["score"] = 1.0 if is_correct_match.group(1).lower() == "true" else 0.0
                 result["parse_error"] = False

    except Exception as e:
        result["parse_error"] = True
        result["reasoning"] = f"Unexpected error: {str(e)}"

    return result


async def judge_result(
    question: str,
    correct_answer: str,
    actual_answer: str,
    judge_model_instance,
    judge_formatter,
    actual_explanation: str | None = None,
    logger=None
) -> dict:
    """Judge whether the agent's answer is correct using LLM-as-judge.

    Uses JSON-based judge template for reliable parsing.

    Args:
        question: The original question
        correct_answer: The ground truth answer
        actual_answer: The agent's final answer
        actual_explanation: The agent's final explanation
        judge_model_instance: The judge model instance
        judge_formatter: The judge formatter
        logger: Logger instance

    Returns:
        Dict with judge results
    """
    # Judge prompt with JSON output format
    explanation_text = actual_explanation or ""
    judge_prompt = f"""You are an expert judge tasked with evaluating whether an agent's response to a question is correct.

**Question**:
<question>
{question}
</question>

**Ground Truth Answer**:
<correct_answer>
{correct_answer}
</correct_answer>

**Agent's Final Answer**:
<final_answer>
{actual_answer}
</final_answer>

**Agent's Explanation**:
<explanation>
{explanation_text}
</explanation>

Your task is to determine if the agent's output is correct based on the ground truth answer. Be strict and precise in your judgment.

**Evaluation Criteria**:
1. Extract the final answer primarily from the agent's final answer field
2. Use the explanation as supporting context that can clarify, refine, or contradict the final answer
3. Compare the agent's overall output with the ground truth answer
4. The agent's answer is correct ONLY if it is semantically equivalent to the ground truth
5. Allow for minor variations in phrasing, but the core information must match exactly
6. For numerical answers, allow small rounding differences (within 1% or 0.1 units)
7. If the final answer and explanation contains additional information that does not contradict the ground truth, it can still be marked as correct
8. If the final answer is ambiguous, contradictory, or contain incorrect information, mark it as incorrect
9. If the agent did not provide a clear final answer, mark it as incorrect

**Output Format**: You MUST respond with a valid JSON object in the following format (no additional text):

{{
  "extracted_answer": "The exact answer extracted from the agent's final answer field, or null if no answer was provided",
  "ground_truth": "{correct_answer}",
  "reasoning": "Brief explanation of why the extracted answer is correct or incorrect compared to the ground truth, taking the explanation into account. If no answer was provided in the agent's output, mark it as incorrect.",
  "score": 1.0 or 0.0
}}

Respond ONLY with the JSON object, no additional text before or after."""

    try:
        # Prepare messages
        messages = [
            Msg("system", "You are a precise evaluator. Always respond with valid JSON format as requested.", "system"),
            Msg("user", judge_prompt, "user")
        ]
        
        # Format messages
        formatted_messages = await judge_formatter.format(messages)
        
        # Call model
        # Note: OpenAIChatModel returns a ModelResponse object
        response = await retry_model_call(
            judge_model_instance,
            formatted_messages,
            stream=False,
            # temperature=0.3,  # Lower temperature for more consistent JSON output
            experiment_logger=logger,
            max_retries=20,
            sleep_seconds=100,
        )

        judge_output = ""
        for block in response.content:
            if isinstance(block, dict) and block.get("type") == "text":
                judge_output += block.get("text", "")

        if logger:
            logger.log_info(
                f"[judge_result] Question: {question[:100]}...\n"
                f"[judge_result] Response: {actual_answer[:100]}...\n"
                f"[judge_result] Ground truth: {correct_answer[:100]}...\n"
                f"[judge_result] Judge output: {judge_output}"
            )

        # Parse judge response using JSON parser
        judge_result_dict = parse_judge_response(judge_output)

        if judge_result_dict.get("parse_error") and logger:
            logger.log_warning(
                f"Could not parse judge response. Treating as incorrect. "
                f"Parse error: {judge_result_dict.get('reasoning')}"
            )

        return judge_result_dict

    except Exception as e:
        if logger:
            logger.log_error(f"Error in judge_result: {e}")
        return {
            "score": 0.0,
            "is_correct": False,
            "reasoning": f"Error executing judge: {str(e)}",
            "parse_error": True
        }


async def judge_task_completion(
    task_description: str,
    actual_answer: str,
    judge_model_instance,
    judge_formatter,
    logger=None
) -> dict:
    """Judge whether the agent completed an open-ended task successfully.

    Used when there is no ground truth answer — evaluates task completion quality
    based on how well the agent followed the instructions in task_description.

    Returns:
        Dict with judge results (score is 0.0 or 1.0)
    """
    judge_prompt = f"""You are an expert judge evaluating whether an AI agent successfully completed a multi-step research task.

**Task Instructions**:
<task>
{task_description}
</task>

**Agent's Response**:
<response>
{actual_answer}
</response>

Evaluate whether the agent completed the task. Consider:
1. Did the agent follow the workflow steps described in the task?
2. Did the agent produce a final output matching the requested format?
3. Is the response substantive (not empty, not just error messages)?
4. Did the agent use the available tools effectively?

A task is "completed" if the agent made a genuine, substantive attempt and produced a meaningful final response that addresses the task. Minor omissions or imperfections are acceptable — the key question is whether the agent engaged with the task and produced useful output.

A task is "not completed" if:
- The response is empty or trivially short
- The agent got stuck in errors and produced no useful output
- The agent ignored the task instructions entirely

**Output Format**: Respond with a valid JSON object only:

{{
  "extracted_answer": "Brief summary of what the agent produced",
  "reasoning": "Why the task is or is not completed",
  "score": 1.0 or 0.0
}}"""

    try:
        messages = [
            Msg("system", "You are a precise evaluator. Always respond with valid JSON format as requested.", "system"),
            Msg("user", judge_prompt, "user")
        ]

        formatted_messages = await judge_formatter.format(messages)

        response = await retry_model_call(
            judge_model_instance,
            formatted_messages,
            stream=False,
            experiment_logger=logger,
            max_retries=20,
            sleep_seconds=100,
        )

        judge_output = ""
        for block in response.content:
            if isinstance(block, dict) and block.get("type") == "text":
                judge_output += block.get("text", "")

        if logger:
            logger.log_info(
                f"[judge_task_completion] Task: {task_description[:100]}...\n"
                f"[judge_task_completion] Response: {actual_answer[:100]}...\n"
                f"[judge_task_completion] Judge output: {judge_output}"
            )

        judge_result_dict = parse_judge_response(judge_output)

        if judge_result_dict.get("parse_error") and logger:
            logger.log_warning(
                f"Could not parse judge response. Treating as incomplete. "
                f"Parse error: {judge_result_dict.get('reasoning')}"
            )

        return judge_result_dict

    except Exception as e:
        if logger:
            logger.log_error(f"Error in judge_task_completion: {e}")
        return {
            "score": 0.0,
            "reasoning": f"Error executing judge: {str(e)}",
            "parse_error": True
        }
