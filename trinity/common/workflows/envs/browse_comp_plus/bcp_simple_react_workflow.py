# -*- coding: utf-8 -*-
"""BrowseComp-Plus Simple Tool-based ReAct Workflow for Trinity-RFT using BCPWorker."""

import hashlib
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
    GeminiChatModel,
    TrinityChatModel,
)
from agentscope.formatter import OpenAIChatFormatter, AnthropicChatFormatter, GeminiChatFormatter
from agentscope.message import Msg
from agentscope.model import enable_auto_llm_logging, disable_auto_llm_logging
# Try to import BCPWorker and Judge

from asio.agent.bcp_worker import BCPWorker
from asio.logger import ExperimentLogger
from asio.utils.judge import judge_result
from asio.memory.memorymanager import MemoryError
import uuid

def _is_anthropic_model(model_name: str) -> bool:
    """Check if the model name indicates an Anthropic/Claude model."""
    model_name_lower = model_name.lower()
    return "claude" in model_name_lower or "anthropic" in model_name_lower


def _is_dashscope_url(base_url: str) -> bool:
    """Check if the base_url is a DashScope endpoint."""
    if not base_url:
        return False
    return "dashscope" in base_url.lower()


def _is_gemini_model(model_name: str) -> bool:
    """Check if the model name indicates a Gemini model."""
    return "gemini" in model_name.lower()


# Models served by the newapi endpoint, not dashscope
NEWAPI_MODEL_PREFIXES = ("qwen3.5-",)


def _is_newapi_model(model_name: str) -> bool:
    name_lower = model_name.lower()
    return any(name_lower.startswith(p) for p in NEWAPI_MODEL_PREFIXES)


def _filter_compression_experiences(
    experiences: List[Experience],
    compression_attempt_types: List[str],
) -> List[Experience]:
    """Drop retry-only compression attempts from the training experience list.

    After the memory retry update, `compression_attempt_types` keeps at most:
    - one `parsing_error` or `modification_error`
    - the first `out_of_tokens`
    - the final `success`

    Only non-`out_of_tokens` attempts correspond to trainable compression steps.
    Depending on the backend, OOT attempts may or may not produce a recorded
    experience. Handle both cases conservatively.
    """
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


def _filter_terminal_oot_experiences(
    experiences: List[Experience],
    end_reason: str,
) -> List[Experience]:
    """Keep only directly attributable CM experiences for terminal OOT rollouts."""
    if end_reason != "context_manager_out_of_tokens" or not experiences:
        return experiences

    keep_reasons = {
        "deferred_out_of_tokens",
        "parsing_error",
        "modification_error",
        "degenerate_generation",
    }
    kept = []
    for exp in experiences:
        reward_details = (exp.info or {}).get("reward_details", [])
        if any(
            detail.get("reason") in keep_reasons and float(detail.get("reward", 0.0) or 0.0) != 0.0
            for detail in reward_details
        ):
            kept.append(exp)

    return kept


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


def create_model_and_formatter(
    model_name: str,
    api_key: str | None = None,
    stream: bool = False,
    base_url: str | None = None,
    reasoning_effort: str | None = None,
    **kwargs,
):
    """Create model and formatter instances.

    Automatically selects:
    - GeminiChatModel for Gemini models (proxy mode via DashScope)
    - DashScopeClaudeChatModel for Claude models via DashScope API
    - AnthropicChatModel for Claude models via native Anthropic API
    - OpenAIChatModel for other models
    """
    is_anthropic = _is_anthropic_model(model_name)
    is_dashscope = _is_dashscope_url(base_url)

    if _is_gemini_model(model_name):
        # Gemini models: use GeminiChatModel with proxy mode (DashScope)
        api_key = api_key or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
        proxy_url = base_url or os.environ.get("BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        model = GeminiChatModel(
            model_name=model_name,
            api_key=api_key,
            stream=stream,
            client_args={"base_url": proxy_url},
        )
        formatter = GeminiChatFormatter()
        logger.info(f"[GeminiChatModel] Proxy mode: model={model_name}, base_url={proxy_url}")
    elif is_anthropic and is_dashscope:
        # DashScope Claude: Use OpenAI-compatible endpoint with special params
        api_key = api_key or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
        model = DashScopeClaudeChatModel(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            stream=stream,
            provider="r",  # DashScope provider code for Claude relay
        )
        logger.info(f"[DashScopeClaudeChatModel] Created for model={model_name}, base_url={base_url}")
        formatter = OpenAIChatFormatter()
    elif is_anthropic:
        # Native Anthropic API
        if not api_key:
            api_key = os.environ.get("ANTHROPIC_API_KEY")

        model_kwargs = {"model_name": model_name, "api_key": api_key, "stream": stream}
        if base_url:
            # Anthropic SDK auto-appends /v1/messages, so remove trailing /v1 if present
            anthropic_base_url = base_url.rstrip("/")
            if anthropic_base_url.endswith("/v1"):
                anthropic_base_url = anthropic_base_url[:-3]
            model_kwargs["client_args"] = {"base_url": anthropic_base_url}

        model = AnthropicChatModel(**model_kwargs)
        formatter = AnthropicChatFormatter()
    else:
        # OpenAI or compatible models
        if _is_newapi_model(model_name):
            api_key = os.environ.get("NEWAPI_API_KEY") or api_key
            base_url = os.environ.get("NEWAPI_BASE_URL") or base_url
            logger.info(f"[NewAPI] Routing model={model_name} to base_url={base_url}")
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")

        # Default reasoning_effort to "medium" for gpt-oss models
        if reasoning_effort is None and "gpt-oss" in model_name.lower():
            reasoning_effort = "medium"
        model_kwargs = {"model_name": model_name, "api_key": api_key, "stream": stream}
        if base_url:
            client_args = {"base_url": base_url}
            model_kwargs["client_args"] = client_args
        if reasoning_effort:
            model_kwargs["reasoning_effort"] = reasoning_effort

        model = OpenAIChatModel(**model_kwargs)
        formatter = OpenAIChatFormatter()

    return model, formatter


@WORKFLOWS.register_module("bcp_simple_react_workflow")
class BCPSimpleToolReActWorkflow(Workflow):
    """
    ReAct workflow for BrowseComp-Plus using BCPWorker.
    
    This workflow leverages the BCPWorker from asio/agent/bcp_worker.py to handle
    the interaction with the search environment, while using Trinity's model wrapper
    for generation and experience collection.
    """

    can_reset: bool = True
    is_async: bool = True
    
    GOLD_DOCS = None
    KEY_DOCS = None
    _qrels_lock = threading.Lock()
    
    # Class-level storage for timing data across tasks in a batch
    batch_timing_data = []
    # Class-level cache for judge model to avoid re-initialization
    _judge_model_cache = None
    _judge_formatter_cache = None

    @classmethod
    def _ensure_qrels_loaded(cls):
        if cls.GOLD_DOCS is not None and cls.KEY_DOCS is not None:
            return

        with cls._qrels_lock:
            if cls.GOLD_DOCS is not None and cls.KEY_DOCS is not None:
                return

            cls.GOLD_DOCS = defaultdict(set)
            cls.KEY_DOCS = defaultdict(set)
            
            evidence_path = "../BrowseComp-Plus/topics-qrels/qrel_evidence.txt"
            golds_path = "../BrowseComp-Plus/topics-qrels/qrel_golds.txt"
            
            for path, target_dict in [(evidence_path, cls.KEY_DOCS), (golds_path, cls.GOLD_DOCS)]:
                if os.path.exists(path):
                    try:
                        with open(path, 'r') as f:
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) >= 3:
                                    task_id = parts[0]
                                    doc_id = parts[2]
                                    target_dict[task_id].add(doc_id)
                        logger.info(f"Loaded qrels from {path}: {len(target_dict)} tasks found.")
                    except Exception as e:
                        logger.warning(f"Failed to load qrels from {path}: {e}")
                else:
                     logger.warning(f"Qrels file not found at {path}")

    def __init__(
        self,
        *,
        task: Task,
        model: ModelWrapper,
        auxiliary_models: Optional[List[openai.OpenAI]] = None,
    ):
        self._ensure_qrels_loaded()
        super().__init__(
            task=task,
            model=model,
            auxiliary_models=auxiliary_models,
        )
        self.auxiliary_models = auxiliary_models

        # Caches
        self._cached_tokenizer = None
        self._agent_model_cache = {}   # {agent_name: (model, formatter)}
        self.BCPWorker = BCPWorker
        self.ExperimentLogger = ExperimentLogger

        # Model instances (lazy-initialized)
        self.agent_model = None
        self.agent_formatter = None
        self.memory_model = None
        self.memory_formatter = None
        self.judge_model_instance = None
        self.judge_formatter = None
        self.auxiliary_judge_model_instance = None
        self.auxiliary_judge_formatter = None

        self.workflow_args = None  # sentinel: forces _parse_workflow_config on first reset()
        self.reset(task)

        # Pre-load tokenizer once per WorkflowRunner actor (avoids repeated CPFS reads)
        tokenizer_model = self.tokenizer_model or os.environ.get("DEFAULT_TOKENIZER_MODEL")
        if tokenizer_model:
            try:
                from transformers import AutoTokenizer
                self._cached_tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_model, trust_remote_code=True, use_fast=False
                )
                logger.info(f"Pre-loaded tokenizer from {tokenizer_model}")
            except Exception as e:
                logger.warning(f"Failed to pre-load tokenizer: {e}, will load lazily per task")

    def _parse_workflow_config(self, workflow_args: dict):
        """Parse workflow_args into instance attributes. Called only when config object changes."""
        # Agent config
        self.agent_model_name = workflow_args.get("agent_model_name", "qwen3-max")
        self.agent_api_key = workflow_args.get("agent_api_key", os.environ.get("OPENAI_API_KEY"))
        self.agent_base_url = workflow_args.get("agent_base_url", os.environ.get("BASE_URL"))
        self.agent_model_source = workflow_args.get("agent_model_source", "external")
        self.agent_auxiliary_model_index = int(workflow_args.get("agent_auxiliary_model_index", 1))

        # Squad config: "sample" = weighted random per (task, batch), "all" = run_id % N
        self.agent_expand = workflow_args.get("agent_expand", "all")
        agent_models_cfg = workflow_args.get("agent_models")
        if agent_models_cfg:
            self.agent_models_config = agent_models_cfg
            self.agent_model_weights = [m.get("weight", 1.0) for m in agent_models_cfg]
        else:
            self.agent_models_config = None
            self.agent_model_weights = None

        # Searcher config (string or list of {type, weight})
        searcher_raw = workflow_args.get("searcher_type", "bm25")
        if isinstance(searcher_raw, list):
            if searcher_raw and isinstance(searcher_raw[0], dict):
                self.searcher_type_list = [s["type"] for s in searcher_raw]
                self.searcher_type_weights = [s.get("weight", 1.0) for s in searcher_raw]
            else:
                self.searcher_type_list = searcher_raw
                self.searcher_type_weights = None
            self.searcher_type = None
        else:
            self.searcher_type_list = None
            self.searcher_type_weights = None
            self.searcher_type = searcher_raw

        # BCP worker config
        self.worker_name = workflow_args.get("worker_name", "assistant")
        self.browsecomp_path = workflow_args.get("browsecomp_path")
        self.index_path = workflow_args.get("index_path", "indexes/bm25")
        self.dense_index_path = workflow_args.get("dense_index_path", self.index_path)
        self.searcher_model = workflow_args.get("searcher_model", "Qwen/Qwen3-Embedding-8B")
        self.top_k = int(workflow_args.get("top_k", 5))
        self.snippet_max_tokens = int(workflow_args.get("snippet_max_tokens", 512))
        self.doc_max_tokens = int(workflow_args.get("doc_max_tokens", 4096))
        self.include_get_document = bool(workflow_args.get("include_get_document", True))
        self.max_iters = int(workflow_args.get("max_iterations", 50))
        self.stop_on_no_tool_use = bool(workflow_args.get("stop_on_no_tool_use", True))
        self.tokenizer_model = workflow_args.get("tokenizer_model")
        self.log_rollout_time = bool(workflow_args.get("log_rollout_time", False))
        self.agent_enable_thinking = workflow_args.get("agent_enable_thinking", False)
        self.agent_reasoning_effort = workflow_args.get("agent_reasoning_effort", None)
        self.agent_stream = bool(workflow_args.get("agent_stream", False))
        self.agent_temperature = float(workflow_args.get("agent_temperature", 0))

        # Memory config
        self.memory_model_name = workflow_args.get("memory_model")
        self.memory_class = "MemoryManager" if self.memory_model_name else workflow_args.get("memory_class", "InMemory")
        memory_config_raw = workflow_args.get("memory_config", {})
        try:
            from omegaconf import OmegaConf, DictConfig
            if isinstance(memory_config_raw, DictConfig):
                memory_config_raw = OmegaConf.to_container(memory_config_raw, resolve=True)
        except ImportError:
            pass
        self.memory_config = dict(memory_config_raw) if memory_config_raw else {}
        if self.tokenizer_model and "chat_tokenizer_model" not in self.memory_config:
            self.memory_config["chat_tokenizer_model"] = self.tokenizer_model

        # Judge & reward config
        self.judge_model_source = workflow_args.get("judge_model_source", "external")
        self.judge_model_name = workflow_args.get("judge_model_name", "gpt-5-2025-08-07")
        self.calculate_reward = workflow_args.get("calculate_reward", True)
        self.reward_config = workflow_args.get("reward_config", {})

        # Logging config
        self.name = workflow_args.get("name", "bcp")
        self.project = workflow_args.get("project", "Trinity-RFT")
        self.group = workflow_args.get("group", "")
        self.dataset = workflow_args.get("dataset", "default")
        self.force_create_logger = workflow_args.get("force_create_logger", False)
        self.force_eval_create_logger = workflow_args.get("force_eval_create_logger", False)

        # Invalidate config-dependent model caches
        self.judge_model_instance = None
        self.judge_formatter = None
        self.auxiliary_judge_model_instance = None
        self.auxiliary_judge_formatter = None
        self.memory_model = None
        self.memory_formatter = None

    def reset(self, task: Task):
        """Reset the workflow with a new task."""
        new_workflow_args = task.workflow_args or {}
        if new_workflow_args is not self.workflow_args:
            self._parse_workflow_config(new_workflow_args)
        self.task = task
        self.workflow_args = new_workflow_args

        # Task info
        self.question = task.task_desc
        self.ground_truth = task.truth
        task_id_key = task.format_args.task_id_key if task.format_args else 'task_id'
        self.task_id = task.raw_task[task_id_key]

        # Extract explore step number from batch_id
        self.explore_step_num = 0
        if task.is_eval and isinstance(task.batch_id, str):
            parts = task.batch_id.split('/', 1)
            if parts and parts[0].isdigit():
                self.explore_step_num = int(parts[0])
        elif isinstance(task.batch_id, int):
            self.explore_step_num = task.batch_id

        # Agent & searcher assignment
        # Eval: seed without batch_id so same task always gets same agent/searcher across steps
        # Training: seed with batch_id for diversity across batches
        if self.agent_expand != "all":
            self._assign_agent_and_searcher_v6(task)
        elif self.agent_models_config:
            # v7 "all" mode with squad: deferred to run_async (needs current_run_id)
            pass
        else:
            # Single-agent-per-taskset (v8): use defaults + sample searcher
            self._set_default_agent()
            seed_keys = (self.task_id,) if task.is_eval else (self.task_id, task.batch_id)
            rng = self._stable_rng(*seed_keys)
            self._sample_searcher(rng)

    def _stable_rng(self, *keys) -> random.Random:
        """Create a deterministic RNG from string keys (hashlib, not hash())."""
        seed = int(hashlib.sha256("@".join(str(k) for k in keys).encode()).hexdigest(), 16) % (2**32)
        return random.Random(seed)

    def _assign_agent_and_searcher_v6(self, task: Task):
        """v6: sample agent & searcher per (task_id, batch_id). Deterministic within a batch.
        Eval: seed without batch_id so assignment is stable across steps."""
        seed_keys = (self.task_id,) if task.is_eval else (self.task_id, task.batch_id)
        rng = self._stable_rng(*seed_keys)
        if self.agent_models_config:
            sampled = rng.choices(self.agent_models_config, weights=self.agent_model_weights, k=1)[0]
            self._apply_sampled_agent(sampled)
        else:
            self._set_default_agent()
        self._sample_searcher(rng)
        self._load_cached_agent_model()

    def _assign_agent_and_searcher_v7(self):
        """v7: agent by current_run_id % N, searcher by hash(task_id, agent). Called in run_async.
        If fixed_agent_index is set in task.index (from scheduler agent expansion), use it directly."""
        fixed_idx = self.task.index.get("fixed_agent_index") if self.task and self.task.index else None
        if fixed_idx is not None:
            agent_index = fixed_idx
        else:
            raise ValueError("No fixed_agent_index found")
        self._apply_sampled_agent(self.agent_models_config[agent_index])
        rng = self._stable_rng(self.task_id, self.sampled_agent_name)
        self._sample_searcher(rng)
        self._load_cached_agent_model()

    def _apply_sampled_agent(self, sampled: dict):
        """Apply agent config from a squad agent dict."""
        self.sampled_agent_name = sampled.get("name", self.agent_model_name)
        self.sampled_agent_source = sampled.get("source", "external")
        self.sampled_agent_aux_index = int(sampled.get("auxiliary_index", 0))
        self.sampled_agent_base_url = sampled.get("base_url", self.agent_base_url)

    def _set_default_agent(self):
        """Use single-agent defaults (no squad)."""
        self.sampled_agent_name = self.agent_model_name
        self.sampled_agent_source = self.agent_model_source
        self.sampled_agent_aux_index = self.agent_auxiliary_model_index
        self.sampled_agent_base_url = self.agent_base_url

    def _sample_searcher(self, rng: random.Random):
        """Sample searcher type from list using rng, or keep static config."""
        if self.searcher_type_list:
            if self.searcher_type_weights:
                self.searcher_type = rng.choices(self.searcher_type_list, weights=self.searcher_type_weights, k=1)[0]
            else:
                self.searcher_type = rng.choice(self.searcher_type_list)
        self.current_index_path = self.dense_index_path if self.searcher_type == "dense" else self.index_path

    def _load_cached_agent_model(self):
        """Load agent model from cache or mark for lazy init."""
        cached = self._agent_model_cache.get(self.sampled_agent_name)
        if cached:
            self.agent_model, self.agent_formatter = cached
        else:
            self.agent_model = None
            self.agent_formatter = None

    async def _initialize_models(self):
        """Initialize agent model (lazy, cached) and memory model."""
        try:
            # Agent model
            if self.sampled_agent_source == "auxiliary" and \
                    self.auxiliary_models and \
                    len(self.auxiliary_models) > self.sampled_agent_aux_index:
                client = self.auxiliary_models[self.sampled_agent_aux_index]
                model_name = getattr(client, "model_path", self.sampled_agent_name)
                # Update sampled_agent_name from auxiliary model's actual path
                if self.sampled_agent_name == self.agent_model_name and model_name != self.sampled_agent_name:
                    self.sampled_agent_name = os.path.basename(model_name.rstrip("/"))
                api_key = getattr(client, "api_key", os.environ.get("OPENAI_API_KEY"))
                base_url = str(getattr(client, "base_url", os.environ.get("BASE_URL")) or "")
                self.agent_model, self.agent_formatter = create_model_and_formatter(
                    model_name, api_key=api_key, base_url=base_url or None,
                    reasoning_effort=self.agent_reasoning_effort,
                    stream=self.agent_stream,
                )
            else:
                self.agent_model, self.agent_formatter = create_model_and_formatter(
                    self.sampled_agent_name, self.agent_api_key, base_url=self.sampled_agent_base_url,
                    reasoning_effort=self.agent_reasoning_effort,
                    stream=self.agent_stream,
                )
            logger.info(f"Initialized agent model: {self.sampled_agent_name}")
            self._agent_model_cache[self.sampled_agent_name] = (self.agent_model, self.agent_formatter)

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

                # Update memory_config for Trinity model
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
        # Clear any existing auto-logging hooks to prevent stale loggers from previous tasks
        disable_auto_llm_logging()
        
        # Track timing for each phase
        timing_info = {}
        start_time = time.time()
        
        if not self.BCPWorker:
            logger.error("BCPWorker class not available. Cannot run workflow.")
            return []

        # v7 "all" mode: assign agent + searcher now that current_run_id is available
        if self.agent_expand == "all" and self.agent_models_config:
            self._assign_agent_and_searcher_v7()

        # Phase 1: Model initialization
        model_init_start = time.time()
        if self.agent_model is None or (self.memory_class == "MemoryManager" and self.memory_model is None):
            await self._initialize_models()
        timing_info['model_initialization'] = time.time() - model_init_start
        
        # Phase 2: Logger setup
        logger_setup_start = time.time()
        # Create ExperimentLogger based on conditions
        experiment_logger = None
        if self.ExperimentLogger:
            # Check if we should create logger
            if self.force_create_logger:
                # Force creation if configured
                should_create_logger = True
            elif self.force_eval_create_logger and self.task.is_eval:
                # Force creation for all eval tasks if configured
                should_create_logger = True
            else:
                # Use existing logic
                should_create_logger = False
                if self.explore_step_num == 0 or self.task.is_eval:
                    # 50% probability for eval and step-0 train
                    should_create_logger = int(self.task_id) % 2 == 0
                # Training tasks (non-step-0): never create logger
            if should_create_logger:
                run_id = getattr(self, 'current_run_id', None)
                run_suffix = f"/run_{run_id}" if run_id is not None else ""
                # Path format: BENCHMARK_RESULTS_DIR/project/dataset/group/name/{step}/run_{id}/{agent}/tasks/{task_id}/
                # Sanitize agent name for directory (e.g. "meta-llama/Llama-3.1-8B" → "Llama-3.1-8B")
                agent_dir_name = self.sampled_agent_name.rsplit("/", 1)[-1]
                logger_base_dir = os.path.join(
                    os.environ.get("BENCHMARK_RESULTS_DIR", "./benchmark_results"),
                    self.project,
                    self.dataset,
                    self.group,
                    self.name,
                )
                experiment_logger = self.ExperimentLogger(
                    base_dir=logger_base_dir,
                    test_mem_config=f"{self.explore_step_num}{run_suffix}/{agent_dir_name}",
                )
                enable_auto_llm_logging(logger_instance=experiment_logger)
        timing_info['logger_setup'] = time.time() - logger_setup_start

        # Phase 3: Create BCPWorker (including searcher initialization)
        worker_creation_start = time.time()
        logger.info(f"Creating BCPWorker with searcher={self.searcher_type}, index={self.current_index_path}")
        
        # Filter docs for the current task only
        relevant_golds = self.GOLD_DOCS.get(str(self.task_id), set()) if self.GOLD_DOCS else set()
        relevant_keys = self.KEY_DOCS.get(str(self.task_id), set()) if self.KEY_DOCS else set()
        relevant_keys = relevant_keys - relevant_golds
        worker = self.BCPWorker(
            name=self.worker_name,
            model=self.agent_model,
            formatter=self.agent_formatter,
            searcher_type=self.searcher_type,
            index_path=self.current_index_path,
            browsecomp_path=self.browsecomp_path,
            top_k=self.top_k,
            snippet_max_tokens=self.snippet_max_tokens,
            doc_max_tokens=self.doc_max_tokens,
            include_get_document=self.include_get_document,
            max_iters=self.max_iters,
            memory_class=self.memory_class,
            memory_config=self.memory_config,
            experiment_logger=experiment_logger,
            searcher_model_name=self.searcher_model,
            gold_docs=relevant_golds,
            key_docs=relevant_keys,
            task_id=self.task_id,
            calculate_reward=self.calculate_reward,
            reward_config=self.reward_config,
            stop_on_no_tool_use=self.stop_on_no_tool_use,
            tokenizer_model=self.tokenizer_model,
            tokenizer=self._cached_tokenizer,
            enable_thinking=self.agent_enable_thinking,
            **({'agent_temperature': self.agent_temperature} if self.sampled_agent_source == 'auxiliary' else {}),
        )
        timing_info['worker_creation'] = time.time() - worker_creation_start
        logger.info(f"BCPWorker creation took {timing_info['worker_creation']:.2f} seconds")
        
        # Start logging run if available
        if experiment_logger:
            run_dir = experiment_logger.start_run(str(self.task_id), clear_existing=True)
            logger.debug(f"Started ExperimentLogger for task {self.task_id}, dir: {run_dir}")
            
            # Update memory debug_dir if applicable
            if hasattr(worker, 'memory') and hasattr(worker.memory, 'debug_dir'):
                worker.memory.debug_dir = str(run_dir)
        
        # Phase 4: Run the actual task
        task_execution_start = time.time()
        logger.info(f"Running BCP task: {self.question[:100]}...")
        
        memory_error_occurred = False
        try:
            result = await worker.run_search_task(
                question=self.question,
                ground_truth=self.ground_truth
            )
        except MemoryError as e:
            logger.warning(f"MemoryError in task execution: {e}")
            memory_error_occurred = True
            result = {
                "answer": "",
                "tool_calls": {},
                "total_iterations": 0,
                "error": str(e)
            }
        except Exception as e:
            # Unhandled error (e.g. RuntimeError from bug detection) - discard this run
            logger.error(f"Unhandled error in task execution, discarding run: {e}\n{traceback.format_exc()}")
            disable_auto_llm_logging()
            if experiment_logger:
                try:
                    experiment_logger.save_report({"error": str(e), "task_id": self.task_id}, filename="result.json")
                    experiment_logger.end_run()
                except Exception:
                    pass
            self.model.extract_experience_from_history(clear_history=True)
            return []
        
        timing_info['bcp_task_execution'] = time.time() - task_execution_start
        logger.info(f"Task execution took {timing_info['bcp_task_execution']:.2f} seconds")

        # Add metadata to result
        result["task_id"] = self.task_id
        result["question"] = self.question
        result["ground_truth"] = self.ground_truth
        result["agent_model"] = self.sampled_agent_name
        result["searcher_type"] = self.searcher_type
        
        answer = result.get("answer")
        if answer is None:
            logger.error(f"BCP Task returned None answer. Error: {result.get('error', 'Unknown error')}")
            if "traceback" in result: # If worker provides traceback, log it? Worker usually puts tb in 'error' string
                 logger.error(f"Worker Traceback: {result.get('traceback')}")
            # Keep answer as None or set to empty string depending on downstream needs.
            # User asked to KEEP error (implying keep None? or just log error?)
            # Keep the error visible, don't hide the fact it failed.
            # But we must avoid the crash.
            answer = "" # Still need a string for subsequent operations like judging?
            # If answer is None, Judge will likely fail or score 0.
            # Let's set to empty string to allow workflow to proceed to scoring (which will be 0)
            
        logger.info(f"BCP Task completed. Answer length: {len(answer) if answer else 0} with {result.get('tool_calls', {})}")
        
        # Phase 5: Judging
        judging_start = time.time()
        score = 0.0
        if result.get("end_reason") == "context_manager_out_of_tokens":
            logger.warning("Context manager OutOfTokens; skipping judgment and assigning score=0.")
        elif self.ground_truth and not memory_error_occurred:
            try:
                # Initialize judge model based on judge_model_source
                if self.judge_model_source == "auxiliary" and self.auxiliary_models and len(self.auxiliary_models) > 0:
                    judge_model = self.auxiliary_models[0]

                    # Initialize wrapper if not already done
                    if self.auxiliary_judge_model_instance is None:
                         # Wrap raw client into AgentScope model to enable automatic logging
                         if not hasattr(judge_model, "reply_nm"):
                             model_name = getattr(judge_model, "model_path", "auxiliary_judge")
                             api_key = getattr(judge_model, "api_key", os.environ.get("OPENAI_API_KEY"))
                             base_url = getattr(judge_model, "base_url", os.environ.get("BASE_URL"))
                             if base_url: base_url = str(base_url)

                             self.auxiliary_judge_model_instance, self.auxiliary_judge_formatter = create_model_and_formatter(
                                 model_name=model_name,
                                 api_key=api_key,
                                 base_url=base_url
                             )
                         else:
                             self.auxiliary_judge_model_instance = judge_model
                             self.auxiliary_judge_formatter = OpenAIChatFormatter()

                    as_judge_model = self.auxiliary_judge_model_instance
                    as_judge_formatter = self.auxiliary_judge_formatter
                else:
                    # "external" mode: use judge_model_name to create model
                    if self.judge_model_instance is None:
                        logger.info(f"Initializing external judge model: {self.judge_model_name}")
                        self.judge_model_instance, self.judge_formatter = create_model_and_formatter(
                            model_name=self.judge_model_name,
                            stream=False,
                            base_url=os.environ.get("BASE_URL")
                        )
                    as_judge_model = self.judge_model_instance
                    as_judge_formatter = self.judge_formatter

                judge_output = await judge_result(
                    question=self.question,
                    correct_answer=self.ground_truth,
                    actual_answer=answer,
                    judge_model_instance=as_judge_model,
                    judge_formatter=as_judge_formatter,
                    logger=experiment_logger
                )
                is_correct = float(judge_output.get("score", 0.0))
                score = 1.0 if is_correct else 0.0
                result.update(judge_output)
                
                logger.info(f"Judge result: {is_correct}")
                
            except Exception as e:
                logger.error(f"Judging failed: {e}")
                
        else:
            logger.warning("No ground truth provided or memory error occurred, skipping judgment.")
        timing_info['judging'] = time.time() - judging_start
        
        # End logger run
        if experiment_logger:
            try:
                result_file = "result.json"
                experiment_logger.save_report(result, filename=result_file)
                experiment_logger.end_run()
            except Exception as e:
                logger.warning(f"Failed to finalize experiment logger (likely due to concurrent file access): {e}")
        
        # Phase 6: Extract experiences
        experience_extraction_start = time.time()
        # Extract Experience
        # This relies on the fact that if we used TrinityChatModel(self.model),
        # self.model has accumulated the history of the conversation.
        experiences = self.model.extract_experience_from_history(clear_history=True)
        timing_info['experience_extraction'] = time.time() - experience_extraction_start

        compression_attempt_types = result.get("compression_attempt_types", [])
        experiences = _filter_compression_experiences(experiences, compression_attempt_types)
        
        logger.info(f"Extracted {len(experiences)} experiences from history.")
        if len(experiences) == 0:
            tool_calls = result.get("tool_calls", {})
            logger.warning(
                f"Empty experiences: agent answered without invoking MemoryManager "
                f"(tool_calls={tool_calls}, answer_len={len(answer)}). "
                f"Task contributes 0 training signal; returning [] cleanly."
            )
            disable_auto_llm_logging()
            return []
        
        # Log timing summary
        timing_info['total_time'] = time.time() - start_time
        # logger.info("========== TIMING BREAKDOWN ==========")
        # logger.info(f"Model initialization: {timing_info.get('model_initialization', 0):.2f}s")
        # logger.info(f"Logger setup: {timing_info.get('logger_setup', 0):.2f}s")
        # logger.info(f"BCPWorker creation (including searcher init): {timing_info.get('worker_creation', 0):.2f}s")
        # logger.info(f"Task execution: {timing_info.get('bcp_task_execution', 0):.2f}s")
        # logger.info(f"Judging: {timing_info.get('judging', 0):.2f}s")
        # logger.info(f"Experience extraction: {timing_info.get('experience_extraction', 0):.2f}s")
        # logger.info(f"TOTAL TIME: {timing_info['total_time']:.2f}s")
        # logger.info("=======================================")
        
        # Store timing data for batch aggregation
        if self.log_rollout_time:
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
            # Get intermediate reward for this step (already scaled in BCPWorker)
            intermediate_reward = step_reward_map.get(i, 0.0)

            # Initially set reward to just the final score
            # The advantage function will add intermediate rewards during calculation
            exp.reward = score

            # Store intermediate rewards in exp.info
            exp.info = exp.info or {}
            exp.info["intermediate_reward"] = intermediate_reward  # Used by advantage function
            exp.info["final_score"] = score
            exp.info["reward_details"] = step_reward_map_reason.get(i, [])
            
            exp.eid.step = i
            # Encode agent identity into task ID for per-agent GRPO grouping
            # e.g. task_id="abc123" → eid.task="abc123@gpt-4o-mini"
            # so group_by("task") naturally groups by (task, agent)
            exp.eid.task = f"{self.task_id}@{self.sampled_agent_name}"

            if exp.metrics is None:
                exp.metrics = {}
            
            metrics_update = {
                "score": score,
                "intermediate_reward_per_step": intermediate_reward,
                "answer_length": len(answer),
                "iterations": result.get("total_iterations", 0),
                "search_calls": result.get("tool_calls", {}).get("search", 0),
                "get_document_calls": result.get("tool_calls", {}).get("get_document", 0),
                "_agent_model": self.sampled_agent_name,
                "_searcher_type": self.searcher_type,
            }
            if self.log_rollout_time:
                metrics_update.update({
                    "time/model_initialization": timing_info.get('model_initialization', 0),
                    "time/logger_setup": timing_info.get('logger_setup', 0),
                    "time/worker_creation": timing_info.get('worker_creation', 0),
                    "time/bcp_task_execution": timing_info.get('bcp_task_execution', 0),
                    "time/judging": timing_info.get('judging', 0),
                    "time/experience_extraction": timing_info.get('experience_extraction', 0),
                    "time/total": timing_info.get('total_time', 0),
                })
            metrics_update.update(rollout_metric_summary)
            exp.metrics.update(metrics_update)

        experiences = _filter_terminal_oot_experiences(
            experiences,
            result.get("end_reason"),
        )

        if len(experiences) == 0:
            logger.warning(
                "No trainable Context Manager experiences remain after terminal OOT filtering."
            )
            disable_auto_llm_logging()
            return []
            
        if memory_error_occurred and experiences:
            last_exp = experiences[-1]
            last_exp.reward = 0.0
            last_exp.info = last_exp.info or {}
            last_exp.info["intermediate_reward"] = -0.5
            last_exp.info["final_score"] = 0.0
            last_exp.info["memory_error"] = True
            experiences = [last_exp]
            
        # Log all experiences for debugging (remove tensors for serialization)
        debug_exps = []
        for exp in experiences:
            exp_dict = exp.to_dict()
            # Remove tensor keys or other non-serializable fields if present in to_dict output
            # to_dict already converts some structure, but let's be safe and exclude large data
            serializable_exp = {
                "eid": exp_dict.get("eid"),
                "prompt_text": exp_dict.get("prompt_text"),
                "response_text": exp_dict.get("response_text"),
                "reward": exp_dict.get("reward"),
                "info": exp_dict.get("info"),
                "metrics": exp_dict.get("metrics"),
                "messages": exp_dict.get("messages"),
                "tools": exp_dict.get("tools")
            }
            debug_exps.append(serializable_exp)
        
        # logger.info(f"Final Experiences: {json.dumps(debug_exps, indent=2, default=str)}")
            
            # If we are in "Expert Mode" (external agent), these experiences might be empty 
            # unless MemoryManager was used to pipe into self.model. 
            # But currently we don't have that setup explicitly here (SciWorldWorkflow does).
            # If agent_model_name was set, self.model history is likely empty.
            # In that case, we might need to manually construct experience or warn.
        
        # Clear auto-logging hooks before returning to avoid polluting subsequent tasks
        disable_auto_llm_logging()

        return experiences
