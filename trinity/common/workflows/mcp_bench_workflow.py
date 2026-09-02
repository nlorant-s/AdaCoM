# -*- coding: utf-8 -*-
"""MCP-Bench Workflow for Trinity-RFT using MCPWorker.

This workflow handles training of the Context Manager (MemoryManager) on MCP-Bench tasks,
using multiple MCP servers to provide tools for the agent.
"""

import json
import os
import re
import traceback
import sys
import random
import threading
import time
from typing import List, Optional, Any, Dict
from collections import defaultdict
import openai

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows.workflow import WORKFLOWS, Workflow, Task
from trinity.utils.log import get_logger

logger = get_logger(__name__)

# Try to import AgentScope components
from agentscope.model import (
    OpenAIChatModel,
    AnthropicChatModel,
    DashScopeClaudeChatModel,
    TrinityChatModel,
)
from agentscope.formatter import OpenAIChatFormatter, AnthropicChatFormatter
from agentscope.message import Msg
from agentscope.model import enable_auto_llm_logging, disable_auto_llm_logging

# Try to import MCPWorker and Judge
from asio.agent.mcp_worker import MCPWorker
from asio.logger import ExperimentLogger
from asio.utils.judge import judge_task_completion
from asio.memory.memorymanager import MemoryError
from asio.memory.context_lock import (
    apply_thinking_memory_config,
    resolve_lineage_log_path,
)
import uuid


def _build_rollout_metric_summary(result: Dict[str, Any]) -> Dict[str, float]:
    rewards = result.get("intermediate_rewards", []) or []
    metrics: Dict[str, float] = {}

    end_reason = result.get("end_reason")
    metrics["rollout/end_reason/context_manager_out_of_tokens/count"] = (
        1.0 if end_reason == "context_manager_out_of_tokens" else 0.0
    )
    metrics["rollout/end_reason/context_manager_out_of_tokens/ratio"] = metrics[
        "rollout/end_reason/context_manager_out_of_tokens/count"
    ]

    for reason in (
        "insufficient_budget",
        "deferred_out_of_tokens",
        "repetition_penalty",
        "degenerate_generation",
    ):
        reason_rewards = [item for item in rewards if item.get("reason") == reason]
        metrics[f"reward/{reason}/count"] = float(len(reason_rewards))
        metrics[f"reward/{reason}/task_count"] = 1.0 if reason_rewards else 0.0
        if reason_rewards:
            metrics[f"reward/{reason}/mean_raw"] = sum(
                float(item.get("reward", 0.0)) for item in reason_rewards
            ) / len(reason_rewards)

    repetition_stats = list((result.get("compression_repetition_stats_by_step") or {}).values())
    if repetition_stats:
        max_repeats = [float(item.get("max_repeat", 0.0)) for item in repetition_stats]
        metrics.update({
            "compression/repetition/max_repeat/mean": sum(max_repeats) / len(max_repeats),
            "compression/repetition/max_repeat/max": max(max_repeats),
            "compression/repetition/severe_ratio": sum(
                1.0 for item in repetition_stats if float(item.get("severe_count", 0.0)) > 0
            ) / len(repetition_stats),
        })

    compression_metrics = list((result.get("compression_metrics_by_step") or {}).values())
    if compression_metrics:
        after_tokens = [float(item.get("after_tokens", 0.0)) for item in compression_metrics]
        target_after_tokens = [float(item.get("target_after_tokens", 0.0)) for item in compression_metrics]
        budget_deficits = [float(item.get("budget_deficit", 0.0)) for item in compression_metrics]
        saved_tokens = [float(item.get("saved_tokens", 0.0)) for item in compression_metrics]
        metrics.update({
            "compression/after_tokens/mean": sum(after_tokens) / len(after_tokens),
            "compression/after_tokens/max": max(after_tokens),
            "compression/target_after_tokens/mean": sum(target_after_tokens) / len(target_after_tokens),
            "compression/budget_deficit/mean": sum(budget_deficits) / len(budget_deficits),
            "compression/budget_violation_ratio": sum(1.0 for v in budget_deficits if v > 0) / len(budget_deficits),
            "compression/saved_tokens/mean": sum(saved_tokens) / len(saved_tokens),
        })

    return metrics


def _filter_compression_experiences(
    experiences: List[Experience],
    compression_attempt_types: List[str],
) -> List[Experience]:
    if not experiences or not compression_attempt_types:
        return experiences

    train_attempt_types = [
        attempt_type for attempt_type in compression_attempt_types if attempt_type != "out_of_tokens"
    ]
    if not train_attempt_types:
        return []

    if len(experiences) == len(train_attempt_types):
        return experiences

    if len(experiences) == len(compression_attempt_types):
        return [
            exp
            for exp, attempt_type in zip(experiences, compression_attempt_types)
            if attempt_type != "out_of_tokens"
        ]

    logger.warning(
        "Compression attempt / experience count mismatch: "
        f"experiences={len(experiences)}, "
        f"attempts={len(compression_attempt_types)}, "
        f"train_attempts={len(train_attempt_types)}. "
        "Keeping experiences unchanged to avoid corrupting alignment."
    )
    return experiences


def _is_anthropic_model(model_name: str) -> bool:
    """Check if the model name indicates an Anthropic/Claude model."""
    model_name_lower = model_name.lower()
    return "claude" in model_name_lower or "anthropic" in model_name_lower


def _is_dashscope_url(base_url: str) -> bool:
    """Check if the base_url is a DashScope endpoint."""
    if not base_url:
        return False
    return "dashscope" in base_url.lower()


def create_model_and_formatter(
    model_name: str,
    api_key: str | None = None,
    stream: bool = False,
    base_url: str | None = None,
    **kwargs,
):
    """Create model and formatter instances.

    Automatically selects:
    - DashScopeClaudeChatModel for Claude models via DashScope API
    - AnthropicChatModel for Claude models via native Anthropic API
    - OpenAIChatModel for other models
    """
    is_anthropic = _is_anthropic_model(model_name)
    is_dashscope = _is_dashscope_url(base_url)

    if is_anthropic and is_dashscope:
        api_key = api_key or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
        model = DashScopeClaudeChatModel(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            stream=stream,
            provider="r",
        )
        logger.info(f"[DashScopeClaudeChatModel] Created for model={model_name}, base_url={base_url}")
        formatter = OpenAIChatFormatter()
    elif is_anthropic:
        if not api_key:
            api_key = os.environ.get("ANTHROPIC_API_KEY")

        model_kwargs = {"model_name": model_name, "api_key": api_key, "stream": stream}
        if base_url:
            anthropic_base_url = base_url.rstrip("/")
            if anthropic_base_url.endswith("/v1"):
                anthropic_base_url = anthropic_base_url[:-3]
            model_kwargs["client_args"] = {"base_url": anthropic_base_url}

        model = AnthropicChatModel(**model_kwargs)
        formatter = AnthropicChatFormatter()
    else:
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")

        model_kwargs = {"model_name": model_name, "api_key": api_key, "stream": stream}
        if base_url:
            client_args = {"base_url": base_url}
            model_kwargs["client_args"] = client_args

        model = OpenAIChatModel(**model_kwargs)
        formatter = OpenAIChatFormatter()

    return model, formatter


# Default MCP server configurations for MCP-Bench
DEFAULT_MCP_SERVER_CONFIGS = [
    {
        "name": "Wikipedia",
        "transport": "stdio",
        "command": ["python", "-m", "wikipedia_mcp"],
        "env": {},
        "cwd": None,
    },
    {
        "name": "Unit Converter",
        "transport": "stdio",
        "command": ["python", "-m", "unit_converter_mcp.server"],
        "env": {},
        "cwd": None,
    },
]


@WORKFLOWS.register_module("mcp_bench_workflow")
class MCPBenchWorkflow(Workflow):
    """
    Workflow for MCP-Bench tasks using MCPWorker.

    This workflow leverages MCPWorker to handle multi-server MCP interactions,
    while using Trinity's model wrapper for generation and experience collection.
    """

    can_reset: bool = True
    is_async: bool = True

    # Class-level storage for timing data across tasks in a batch
    batch_timing_data = []
    # Class-level cache for judge model
    _judge_model_cache = None
    _judge_formatter_cache = None

    def __init__(
        self,
        *,
        task: Task,
        model: ModelWrapper,
        auxiliary_models: Optional[List[openai.OpenAI]] = None,
    ):
        super().__init__(
            task=task,
            model=model,
            auxiliary_models=auxiliary_models,
        )
        self.auxiliary_models = auxiliary_models

        # Workflow configuration
        workflow_args = task.workflow_args or {}

        # Agent Model Config
        self.agent_model_name = workflow_args.get("agent_model_name", "qwen3-max")
        self.agent_api_key = workflow_args.get("agent_api_key", os.environ.get("OPENAI_API_KEY"))
        self.agent_base_url = workflow_args.get("agent_base_url", os.environ.get("BASE_URL"))
        self.name = workflow_args.get("name", "mcp_bench")
        self.project = workflow_args.get("project", "Trinity-RFT")
        self.group = workflow_args.get("group", "")
        self.dataset = workflow_args.get("dataset", "mcp_bench")
        self.agent_model_source = workflow_args.get("agent_model_source", "external")

        # Judge Model Config
        self.judge_model_source = workflow_args.get("judge_model_source", "external")
        self.judge_model_name = workflow_args.get("judge_model_name", "gpt-5-2025-08-07")

        # Reward configuration
        self.calculate_reward = workflow_args.get("calculate_reward", True)
        self.reward_config = workflow_args.get("reward_config", {})

        # MCP Config
        self.worker_name = workflow_args.get("worker_name", "assistant")
        self.mcp_server_configs = workflow_args.get("mcp_server_configs", DEFAULT_MCP_SERVER_CONFIGS)
        self.max_iters = int(workflow_args.get("max_iterations", 50))
        self.stop_on_no_tool_use = bool(workflow_args.get("stop_on_no_tool_use", True))
        self.agent_enable_thinking = workflow_args.get("agent_enable_thinking", False)
        self.tokenizer_model = workflow_args.get("tokenizer_model")
        self.tool_cache_path = workflow_args.get("tool_cache_path")
        self.tool_schemas_path = workflow_args.get("tool_schemas_path")
        self.es_fallback_config = workflow_args.get("es_fallback")
        self.skip_live_mcp_servers = workflow_args.get("skip_live_mcp_servers")

        # Memory Config
        self.memory_model_name = workflow_args.get("memory_model")
        if self.memory_model_name:
            self.memory_class = "MemoryManager"
        else:
            self.memory_class = workflow_args.get("memory_class", "InMemory")

        memory_config_raw = workflow_args.get("memory_config", {})
        try:
            from omegaconf import OmegaConf, DictConfig
            if isinstance(memory_config_raw, DictConfig):
                memory_config_raw = OmegaConf.to_container(memory_config_raw, resolve=True)
        except ImportError:
            pass
        self.memory_config = dict(memory_config_raw) if memory_config_raw else {}
        # The manager must know whether the frozen agent runs with extended
        # thinking: that flag turns on the context lock (prompt annotation +
        # apply-side filtering of ops targeting the in-flight tool-use cycle).
        apply_thinking_memory_config(self.memory_config, self.agent_enable_thinking)

        # Logger config
        self.force_create_logger = workflow_args.get("force_create_logger", False)
        self.force_eval_create_logger = workflow_args.get("force_eval_create_logger", False)

        # Components
        self.agent_model = None
        self.agent_formatter = None
        self.memory_model = None
        self.memory_formatter = None
        self.MCPWorker = MCPWorker
        self.ExperimentLogger = ExperimentLogger
        self.judge_model_instance = None
        self.auxiliary_judge_model_instance = None
        self.auxiliary_judge_formatter = None

        # Initialize
        self.reset(task)

    def reset(self, task: Task):
        """Reset the workflow with a new task."""
        self.task = task
        self.workflow_args = task.workflow_args

        # Task info from MCP-Bench format
        raw_task = task.raw_task
        self.task_description = raw_task.get("fuzzy_description", task.task_desc)
        task_id_key = task.format_args.task_id_key if task.format_args else "task_id"
        self.task_id = raw_task.get(task_id_key, str(uuid.uuid4()))

        # Get server names for this task
        self.task_servers = raw_task.get("servers", [])
        self.distraction_servers = raw_task.get("distraction_servers", [])

        # Extract explore step number from batch_id for logging logic
        self.explore_step_num = 0
        if task.is_eval and isinstance(task.batch_id, str):
            parts = task.batch_id.split("/", 1)
            if parts and parts[0].isdigit():
                self.explore_step_num = int(parts[0])
        elif isinstance(task.batch_id, int):
            self.explore_step_num = task.batch_id

    def _get_mcp_configs_for_task(self) -> List[Dict[str, Any]]:
        """Get MCP server configs for the current task.

        Uses the 'servers' field from the task to filter available servers.
        Can also include distraction_servers for more challenging evaluation.
        """
        # For now, filter base configs by server name
        all_server_names = self.task_servers + self.distraction_servers
        if not all_server_names:
            # If no servers specified, use all configured servers
            return self.mcp_server_configs

        filtered_configs = []
        for config in self.mcp_server_configs:
            if config["name"] in all_server_names:
                filtered_configs.append(config)

        return filtered_configs if filtered_configs else self.mcp_server_configs

    async def _initialize_models(self):
        """Initialize models."""
        try:
            # Initialize Agent Model
            if self.agent_model_source == "auxiliary" and self.auxiliary_models and len(self.auxiliary_models) > 1:
                logger.info("Initializing agent model from second auxiliary model (host agent)")
                client = self.auxiliary_models[1]
                model_name = getattr(client, "model_path", "auxiliary_host_agent")
                api_key = getattr(client, "api_key", os.environ.get("OPENAI_API_KEY"))
                base_url = getattr(client, "base_url", os.environ.get("BASE_URL"))
                if base_url:
                    base_url = str(base_url)
                self.agent_model, self.agent_formatter = create_model_and_formatter(
                    model_name,
                    api_key=api_key,
                    base_url=base_url,
                )
            else:
                logger.info(f"Initializing external agent model: {self.agent_model_name}")
                self.agent_model, self.agent_formatter = create_model_and_formatter(
                    self.agent_model_name,
                    self.agent_api_key,
                    base_url=self.agent_base_url,
                )

            # Initialize Memory Model (Trinity model for MemoryManager)
            if self.memory_class == "MemoryManager":
                logger.info("Initializing Trinity model for MemoryManager")
                openai_client = self.model.get_openai_async_client()
                self.memory_model = TrinityChatModel(
                    openai_client,
                    generate_kwargs={
                        "temperature": self.task.rollout_args.temperature,
                        "max_tokens": self.task.rollout_args.max_tokens or 4096,
                    },
                )
                self.memory_formatter = OpenAIChatFormatter()

                self.memory_config.update({
                    "use_trinity_model": True,
                    "_trinity_model": self.memory_model,
                    "_trinity_formatter": self.memory_formatter,
                })
            else:
                logger.info(f"Using {self.memory_class}, skipping memory model initialization.")
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise

    async def run_async(self) -> List[Experience]:
        """Run the workflow."""
        # Clear any existing auto-logging hooks
        disable_auto_llm_logging()

        # Track timing for each phase
        timing_info = {}
        start_time = time.time()

        if not self.MCPWorker:
            logger.error("MCPWorker class not available. Cannot run workflow.")
            return []

        # Phase 1: Model initialization
        model_init_start = time.time()
        if self.agent_model is None:
            await self._initialize_models()
        timing_info["model_initialization"] = time.time() - model_init_start

        # Phase 2: Logger setup
        logger_setup_start = time.time()
        experiment_logger = None
        if self.ExperimentLogger:
            if self.force_create_logger:
                should_create_logger = True
            elif self.force_eval_create_logger and self.task.is_eval:
                should_create_logger = True
            else:
                should_create_logger = False
                if self.explore_step_num == 0 or self.task.is_eval:
                    should_create_logger = hash(str(self.task_id)) % 10 == 1

            if should_create_logger:
                run_id = getattr(self, "current_run_id", None)
                run_suffix = f"/run_{run_id}" if run_id is not None else ""
                logger_base_dir = os.path.join(
                    os.environ.get("BENCHMARK_RESULTS_DIR", "./benchmark_results"),
                    self.project,
                    self.dataset,
                    self.group,
                    self.name,
                )
                experiment_logger = self.ExperimentLogger(
                    base_dir=logger_base_dir,
                    test_mem_config=f"{self.explore_step_num}{run_suffix}",
                )
                enable_auto_llm_logging(logger_instance=experiment_logger)
        timing_info["logger_setup"] = time.time() - logger_setup_start

        # Phase 3: Create MCPWorker
        worker_creation_start = time.time()
        mcp_configs = self._get_mcp_configs_for_task()
        logger.info(f"Creating MCPWorker with {len(mcp_configs)} MCP servers")

        worker = self.MCPWorker(
            name=self.worker_name,
            model=self.agent_model,
            formatter=self.agent_formatter,
            server_configs=mcp_configs,
            max_iters=self.max_iters,
            memory_class="MemoryManager",
            memory_config=self.memory_config,
            experiment_logger=experiment_logger,
            task_id=self.task_id,
            calculate_reward=self.calculate_reward,
            reward_config=self.reward_config,
            stop_on_no_tool_use=self.stop_on_no_tool_use,
            tokenizer_model=self.tokenizer_model,
            tool_cache_path=self.tool_cache_path,
            tool_schemas_path=self.tool_schemas_path,
            es_fallback_config=self.es_fallback_config,
            skip_live_mcp_servers=self.skip_live_mcp_servers,
            enable_thinking=self.agent_enable_thinking,
        )
        timing_info["worker_creation"] = time.time() - worker_creation_start
        logger.info(f"MCPWorker creation took {timing_info['worker_creation']:.2f} seconds")

        # Start logging run if available
        run_dir = None
        if experiment_logger:
            run_dir = experiment_logger.start_run(str(self.task_id), clear_existing=True)
            logger.debug(f"Started ExperimentLogger for task {self.task_id}, dir: {run_dir}")

            if hasattr(worker, "memory") and hasattr(worker.memory, "debug_dir"):
                worker.memory.debug_dir = str(run_dir)

        # Per-step lineage + pre/post context snapshots (drift analysis).
        if hasattr(worker, "memory") and hasattr(worker.memory, "lineage_log_path"):
            lineage_path = resolve_lineage_log_path(
                self.memory_config,
                run_dir=run_dir,
                task_id=self.task_id,
                run_id=getattr(self, "current_run_id", None),
            )
            if lineage_path:
                worker.memory.lineage_log_path = lineage_path
                logger.debug(f"Lineage log for task {self.task_id}: {lineage_path}")

        # Phase 4: Run the actual task
        task_execution_start = time.time()
        logger.info(f"Running MCP task: {self.task_description[:100]}...")

        memory_error_occurred = False
        try:
            result = await worker.run_mcp_task(
                task_description=self.task_description,
            )
        except MemoryError as e:
            logger.warning(f"MemoryError in task execution: {e}")
            memory_error_occurred = True
            result = {
                "answer": "",
                "tool_calls": {},
                "total_iterations": 0,
                "error": str(e),
            }
        except Exception as e:
            logger.error(f"Unhandled error in task execution, discarding run: {e}\n{traceback.format_exc()}")
            disable_auto_llm_logging()
            if experiment_logger:
                try:
                    experiment_logger.save_report(
                        {"error": str(e), "task_id": self.task_id}, filename="result.json"
                    )
                    experiment_logger.end_run()
                except Exception:
                    pass
            self.model.extract_experience_from_history(clear_history=True)
            return []

        timing_info["task_execution"] = time.time() - task_execution_start
        logger.info(f"Task execution took {timing_info['task_execution']:.2f} seconds")

        # Add metadata to result
        result["task_id"] = self.task_id
        result["task_description"] = self.task_description

        answer = result.get("answer")
        if answer is None:
            logger.error(f"MCP Task returned None answer. Error: {result.get('error', 'Unknown error')}")
            answer = ""

        logger.info(
            f"MCP Task completed. Answer length: {len(answer) if answer else 0} "
            f"with {result.get('tool_calls', {})}"
        )

        # Phase 5: Judging
        judging_start = time.time()
        score = 0.0
        if result.get("end_reason") == "context_manager_out_of_tokens":
            logger.warning("Context manager OutOfTokens; skipping judgment and assigning score=0.")
        elif not memory_error_occurred and answer:
            try:
                # Initialize judge model
                if self.judge_model_source == "auxiliary" and self.auxiliary_models and len(self.auxiliary_models) > 0:
                    judge_model = self.auxiliary_models[0]

                    if self.auxiliary_judge_model_instance is None:
                        if not hasattr(judge_model, "reply_nm"):
                            model_name = getattr(judge_model, "model_path", "auxiliary_judge")
                            api_key = getattr(judge_model, "api_key", os.environ.get("OPENAI_API_KEY"))
                            base_url = getattr(judge_model, "base_url", os.environ.get("BASE_URL"))
                            if base_url:
                                base_url = str(base_url)

                            self.auxiliary_judge_model_instance, self.auxiliary_judge_formatter = (
                                create_model_and_formatter(
                                    model_name=model_name,
                                    api_key=api_key,
                                    base_url=base_url,
                                )
                            )
                        else:
                            self.auxiliary_judge_model_instance = judge_model
                            self.auxiliary_judge_formatter = OpenAIChatFormatter()

                    as_judge_model = self.auxiliary_judge_model_instance
                    as_judge_formatter = self.auxiliary_judge_formatter
                else:
                    if self.judge_model_instance is None:
                        logger.info(f"Initializing external judge model: {self.judge_model_name}")
                        self.judge_model_instance, self.judge_formatter = create_model_and_formatter(
                            model_name=self.judge_model_name,
                            stream=False,
                            base_url=os.environ.get("BASE_URL"),
                        )
                    as_judge_model = self.judge_model_instance
                    as_judge_formatter = self.judge_formatter

                judge_output = await judge_task_completion(
                    task_description=self.task_description,
                    actual_answer=answer,
                    judge_model_instance=as_judge_model,
                    judge_formatter=as_judge_formatter,
                    logger=experiment_logger,
                )

                is_correct = float(judge_output.get("score", 0.0))
                score = 1.0 if is_correct else 0.0
                result.update(judge_output)

                logger.info(f"Judge result: {is_correct}")

            except Exception as e:
                logger.error(f"Judging failed: {e}")

        else:
            logger.warning("Memory error occurred or empty answer, skipping judgment.")
        timing_info["judging"] = time.time() - judging_start

        # End logger run
        if experiment_logger:
            try:
                result_file = "result.json"
                experiment_logger.save_report(result, filename=result_file)
                experiment_logger.end_run()
            except Exception as e:
                logger.warning(
                    f"Failed to finalize experiment logger (likely due to concurrent file access): {e}"
                )

        # Phase 6: Extract experiences
        experience_extraction_start = time.time()
        experiences = self.model.extract_experience_from_history(clear_history=True)
        compression_attempt_types = result.get("compression_attempt_types", [])
        experiences = _filter_compression_experiences(experiences, compression_attempt_types)
        timing_info["experience_extraction"] = time.time() - experience_extraction_start

        logger.info(f"Extracted {len(experiences)} experiences from history.")

        # Log timing summary
        timing_info["total_time"] = time.time() - start_time

        # Store timing data for batch aggregation
        self.__class__.batch_timing_data.append(timing_info)

        # Consolidate intermediate rewards by step
        step_reward_map = defaultdict(float)
        step_reward_map_reason = defaultdict(list)
        if "intermediate_rewards" in result:
            for item in result["intermediate_rewards"]:
                if "step" in item and item["step"] is not None:
                    step_reward_map[item["step"]] += item["reward"]
                    step_reward_map_reason[item["step"]].append(item)
        rollout_metric_summary = _build_rollout_metric_summary(result)

        # Annotate experiences with reward and metrics
        for i, exp in enumerate(experiences):
            intermediate_reward = step_reward_map.get(i, 0.0)

            exp.reward = score

            exp.info = exp.info or {}
            exp.info["intermediate_reward"] = intermediate_reward
            exp.info["final_score"] = score
            exp.info["reward_details"] = step_reward_map_reason.get(i, [])

            exp.eid.step = i

            if exp.metrics is None:
                exp.metrics = {}

            exp.metrics.update({
                "score": score,
                "answer_length": len(answer),
                "iterations": result.get("total_iterations", 0),
                "time/model_initialization": timing_info.get("model_initialization", 0),
                "time/logger_setup": timing_info.get("logger_setup", 0),
                "time/worker_creation": timing_info.get("worker_creation", 0),
                "time/task_execution": timing_info.get("task_execution", 0),
                "time/judging": timing_info.get("judging", 0),
                "time/experience_extraction": timing_info.get("experience_extraction", 0),
                "time/total": timing_info.get("total_time", 0),
            })
            exp.metrics.update(rollout_metric_summary)

        if memory_error_occurred and experiences:
            last_exp = experiences[-1]
            last_exp.reward = 0.0
            last_exp.info = last_exp.info or {}
            last_exp.info["intermediate_reward"] = -0.5
            last_exp.info["final_score"] = 0.0
            last_exp.info["memory_error"] = True
            experiences = [last_exp]

        # Clear auto-logging hooks before returning
        disable_auto_llm_logging()

        return experiences
