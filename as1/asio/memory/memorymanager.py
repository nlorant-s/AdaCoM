from agentscope.memory import MemoryBase
from typing import (
    List,
    Any,
    Optional,
    Callable,
    Literal,
    Union,
    Sequence,
    Iterable,
    AsyncGenerator,
)
import tiktoken
import random
from .memory_config import (
    DEFAULT_MAX_CHAT_HISTORY_LEN,
    DEFAULT_RETURN_CHAT_HISTORY_LEN,
    MAX_CHUNK_SIZE,
    OVERLAP_SIZE,
    ALLOWED_MAX_TOOL_RESULT_LEN,
    DEFAULT_MAX_MEMORY_LEN,
    WRITE_FILE_TOKEN_LIMIT,
)
# from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import json
import zlib
from collections import defaultdict
import re
from ..logger import ExperimentLogger
import uuid
from transformers import AutoTokenizer
import importlib
import traceback
from agentscope.message import (
    Msg,
    TextBlock,
)
import copy
import time
from agentscope.model import ChatModelBase, TrinityChatModel
from agentscope.formatter import FormatterBase
from .utils import (
    format_msgs, output_format,
    has_tool_result, remove_thinking_block, tolerant_loads, check_tool_use_result_paired, has_tool_use
)
from ..utils.retry import retry_model_call
import json_repair
from .context_lock import (
    filter_modifications, format_msgs_with_locks, plan_lineage,
    stamp_produced_uids, snapshot, records_to_json, LOCK_PROMPT_ADDENDUM,
)

# Custom exception for memory parsing errors
class MemoryError(Exception):
    """Custom exception raised when memory parsing fails"""
    pass


class DegenerateGenerationError(MemoryError):
    """Raised when severe repeated generation is detected before applying modifications."""

    def __init__(self, rewards: list[dict]):
        self.rewards = rewards
        severe_rewards = [reward for reward in rewards if reward.get("reason") == "degenerate_generation"]
        detail = severe_rewards[0] if severe_rewards else {}
        super().__init__(
            "degenerate_generation: "
            f"level={detail.get('level', 'severe')}, "
            f"compression_ratio={detail.get('compression_ratio', None)}, "
            f"text_bytes={detail.get('text_bytes', 0)}"
        )

# Custom log levels removed - using ExperimentLogger methods instead

DEFAULT_CM_INPUT_LIMIT = 28672
DEFAULT_TOKEN_WARNING_RATIO = 0.80

TOOL_RESULT_TOO_LONG_MESSAGE = """
Tool result exceeds length limit and has been saved to file.

Extracted information according to the tool use query: {summary}

Full result location: {path}. Only when you need information other than what's in the extracted information, should you use the tool `retrieve_from_memory` to retrieve related info.
"""

REASONING_TOO_LONG_MESSAGE = """
The reasoning process exceeds length limit and has been saved to file.

Summary: {summary}

Full result location: {path}

To search within this file, use `retrieve_from_memory` with the filename parameter. If you need more details, you could search within this file 
"""

READ_LARGE_FILE_HINT= """
This file is too large to read in its entirety at once. Please use the `retrieve_from_memory` tool to search for the specific information you need. The following shows the beginning of the file:
"""

GENERATE_ANSWER_MSG="""
The original user task: {task}\nNow, the task is completed, generate the final answer based on the knowledge below: {knowledge}.
"""
WORKER_MAX_ITER = os.getenv("WORKER_MAX_ITER", 50)

DEFAULT_DEGENERATION_CONFIG = {
    "enabled": True,
    "min_text_bytes_for_severe": 800,
    "severe_compression_ratio": 0.20,
    "severe_reward": -1.0,
}


def detect_repetition_level(
    text: str,
    min_text_bytes: int = 800,
    severe_compression_ratio: float = 0.20,
) -> dict:
    """Detect low-information repeated generation in one new_content text."""
    if not isinstance(text, str) or not text:
        return {"level": "none", "compression_ratio": None, "text_bytes": 0, "compressed_bytes": 0}

    text_bytes = text.encode()
    byte_count = len(text_bytes)
    if byte_count < min_text_bytes:
        return {"level": "none", "compression_ratio": None, "text_bytes": byte_count, "compressed_bytes": 0}

    compressed_bytes = len(zlib.compress(text_bytes))
    compression_ratio = compressed_bytes / byte_count
    if compression_ratio < severe_compression_ratio:
        return {
            "level": "severe",
            "compression_ratio": compression_ratio,
            "text_bytes": byte_count,
            "compressed_bytes": compressed_bytes,
        }

    return {
        "level": "none",
        "compression_ratio": compression_ratio,
        "text_bytes": byte_count,
        "compressed_bytes": compressed_bytes,
    }

def is_tool_use(content_item: Any) -> bool:
    """Check if a content item represents a tool use/call.
    
    Args:
        content_item: A content item from a message (could be dict or other)
        
    Returns:
        True if this represents a tool use/call, False otherwise
    """
    if not isinstance(content_item, dict):
        return False
    
    # Standard tool_use format
    if content_item.get("type") == "tool_use":
        return True
    return False


def is_tool_result(content_item: Any) -> bool:
    """Check if a content item represents a tool result.
    Args:
        content_item: A content item from a message (could be dict or other)
        
    Returns:
        True if this represents a tool result, False otherwise
    """
    if not isinstance(content_item, dict):
        return False
    
    # Standard tool_result format
    if content_item.get("type") == "tool_result":
        return True
    return False


def extract_tool_id(content_item: Any) -> Optional[str]:
    """Extract the ID from a tool use or result item.
    
    For standard formats with explicit IDs, returns the ID.
    For text-based formats (like SciWorld), generates a consistent ID based on content.
    
    Args:
        content_item: A content item from a message
        
    Returns:
        The tool ID if found/generated, None otherwise
    """
    if not isinstance(content_item, dict):
        return None
    
    # Standard format with explicit ID
    if "id" in content_item:
        return content_item["id"]
    
    # For text-based formats without explicit IDs (like SciWorld),
    # we could generate a hash-based ID from the content for matching
    # This ensures tool uses and results can still be paired
    if content_item.get("type") == "text":
        text = content_item.get("text", "")
        if isinstance(text, str):
            # For SciWorld actions and observations, we don't have explicit IDs
            # Return None to indicate no ID-based pairing is possible
            return None
    
    return None


class MemoryManager():
    """
    ReActMemory is a memory manager.
    """

    def __init__(
        self,
        name: str, 
        model: ChatModelBase,
        formatter: FormatterBase,
        sys_prompt: str = "You're a helpful assistant named {name}.",
        inner_prompt: str = "",
        max_iters: int = 10,
        config: dict = None,
        experiment_logger: Optional[ExperimentLogger] = None,
        **kwargs
    ) -> None:
        """
        Initialize the ReActMemory.

        Args:
            config (dict, Optional):
                A dictionary of configuration parameters. Supported keys include:
                - model_config_name (str): the config name of the model to use for updating memory
                - max_chat_history_len (int): the maximum length of chat history to keep
                - embedding_model (Union[str, Callable]): the embedding model to use for embedding the memories
                - emb_tokenizer_model (str): the model name for embedding tokenizer
                - chat_tokenizer_model (str): the model name for chat tokenizer
                - retrieve_type (Literal["source", "processed", "auto"]): the type of retrieval to perform
                - vector_store (Optional[VectorStoreBase]): the vector database to use
                - max_chat_len (int): max chat length for auto retrieval type
                - max_memory_len (int): the maximum length of memory to keep
                - update_memory_prompt (str): the prompt for llm to update memory
                - summarize_model_config_name (str): the model config name for summarizing
                - summary_working_log_prompt (str): the prompt for summarizing working log
                - summary_w_query_prompt (str): the prompt for summarizing with query
                - global_update_allowed (bool): whether global updates are allowed
            experiment_logger (ExperimentLogger, optional):
                Logger for tracking experiments and LLM invocations
            **kwargs:
                Any additional arguments that will override config values
        """

        self._chat_history = []

        # Set default values
        params = {
            "max_chat_history_len": DEFAULT_MAX_CHAT_HISTORY_LEN,
            "embedding_model": "qwen_emb_config_v4",
            "emb_tokenizer_model": "openai-gpt",
            "chat_tokenizer_model": "Qwen/Qwen3-4B-Instruct-2507",
            "retrieve_type": "source",
            "enable_vdb": True,
            "vector_store": None,
            "max_chat_len": DEFAULT_MAX_CHAT_HISTORY_LEN,
            "max_memory_len": DEFAULT_MAX_MEMORY_LEN,
            "update_memory_prompt": "UPDATE_MEMORY_PROMPT_DEFAULT",
            "summarize_model_config_name": "gpt-4o",
            "summary_working_log_prompt": "SUMMARIZE_WORKING_LOG_PROMPT_v2",
            "summary_w_query_prompt": "SUMMARIZE_WORKING_LOG_PROMPT_W_QUERY",
            "global_update_allowed": False,
            "pre_process_func": "direct_add_memory",
            "compress_single": False,
            "compress_prompt": "DIRECT_COMPRESS_PROMPT_v2",
            "post_process_func": "reflect_on_memory",
            "scratchpad_update_prompt": "UPDATE_SCRATCHPAD_PROMPT_v1",
            "init_scratchpad_prompt": "INIT_SCRATCHPAD_PROMPT_v1",
            "manager_init_scratchpad": "MANAGER_INIT_SCRATCHPAD",
            "manager_act_prompt": "MANAGER_ACT_PROMPT",
            "manager_update_prompt": "MANAGER_UPDATE_PROMPT",
            "manager_act_decide_prompt":"MANAGER_ACT_DECIDE_PROMPT",
            "manager_step_1_prompt": "MANAGER_1107_STEP_1_PROMPT",
            "manager_step_2_prompt": "MANAGER_1107_STEP_2_PROMPT_UNRESOLVED",
            "runtime": None,
            "debug_mode": True,
            "debug_dir": None,
            "step_1_prompt": "STEP_1_PROMPT",
            "step_2_prompt": "STEP_2_PROMPT",
            "degeneration": DEFAULT_DEGENERATION_CONFIG.copy(),
        }
        
        # Update with config values if provided
        if config:
            params.update(config)
            
        # Override with provided kwargs
        params.update(kwargs)

        # Assign values from params
        self.max_chat_history_len = params["max_chat_history_len"]
        embedding_model = params.get("embedding_model", None)
        emb_tokenizer_model = params["emb_tokenizer_model"]
        chat_tokenizer_model = params["chat_tokenizer_model"]
        self.memory_retrieve_type = params["retrieve_type"]
        self.vector_store = params["vector_store"]
        self.max_chat_len = params["max_chat_len"]
        self.max_memory_len = params["max_memory_len"]
        self.compress_single = params["compress_single"]
        update_memory_prompt = params["update_memory_prompt"]
        summary_working_log_prompt = params["summary_working_log_prompt"]
        summary_w_query_prompt = params[
            "summary_w_query_prompt"]
        self.global_update_allowed = params["global_update_allowed"]
        self.pre_process_func = params["pre_process_func"]
        self.post_process_func = params["post_process_func"]
        compress_prompt = params["compress_prompt"]
        scratchpad_update_prompt = params["scratchpad_update_prompt"]
        init_scratchpad_prompt = params["init_scratchpad_prompt"]
        self.runtime = params["runtime"]
        self.debug_mode = params["debug_mode"]
        self.debug_dir = params["debug_dir"]
        self.enable_vdb = params["enable_vdb"]

        # Store provided model and formatter
        self.model = model
        self.formatter = formatter
        self.experiment_logger = experiment_logger
        
        # Embedding model (if provided)
        self.embedding_model = embedding_model
        
        # Log initialization if experiment logger is available
        if self.experiment_logger:
            self.experiment_logger.log_debug(f"Initialized MemoryManager with retrieve type: {self.memory_retrieve_type}, max memory len: {self.max_memory_len}")
        # if not self.vector_store:
        #     self.vector_store = Qdrant(
        #         collection_name="react_memory",
        #         embedding_model_dims=1024,
        #         path="/tmp/vector_store",
        #     )
        self.tmp_tool_use_log= []
        self.num_tool_use_wo_result = 0
        self.cur_chat_len, self.cur_memory_len = 0, 0
        self.messages_since_last_context_manager_call = 0
        # self.emb_tokenizer = AutoTokenizer.from_pretrained(
        #     emb_tokenizer_model, trust_remote_code=True
        # )
        
        # Robust tokenizer loading
        if self.experiment_logger:
            self.experiment_logger.log_info(f"chat_tokenizer_model: {chat_tokenizer_model}")

        # Normalize relative local paths to absolute; keep HF repo IDs as-is
        tokenizer_path = chat_tokenizer_model
        if tokenizer_path:
            is_local_path = (
                tokenizer_path.startswith("./") or
                tokenizer_path.startswith("../") or
                tokenizer_path.startswith("/") or
                "\\" in tokenizer_path
            )
            if is_local_path:
                tokenizer_path = os.path.abspath(tokenizer_path)
                if self.experiment_logger:
                    self.experiment_logger.log_info(f"Normalized tokenizer path: {tokenizer_path}")

        # Try AutoTokenizer first (local path or HF repo ID), fallback to tiktoken
        try:
            self.chat_tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, trust_remote_code=True, use_fast=False
            )
        except Exception as e:
            try:
                self.chat_tokenizer = tiktoken.get_encoding(chat_tokenizer_model)
            except Exception as e2:
                raise ValueError(
                    f"Failed to load tokenizer '{chat_tokenizer_model}'. "
                    f"AutoTokenizer error: {e}. tiktoken error: {e2}"
                )

        prompts_module = importlib.import_module(".config.prompts", package=__name__.rsplit(".", 1)[0])
        self.update_memory_prompt = getattr(
            prompts_module, update_memory_prompt
        )
        # Use the same model for summarization (can be overridden in config)
        self.summarize_model = self.model
        # self.text_splitter = RecursiveCharacterTextSplitter(
        #     chunk_size=MAX_CHAT_MODEL_TOKEN_SIZE,
        #     chunk_overlap=OVERLAP_SIZE,
        #     length_function=lambda x: count_words(self.emb_tokenizer, x),
        # )
        self.summary_working_log_prompt_type = summary_working_log_prompt
        self.summary_working_log_prompt = getattr(
            prompts_module, summary_working_log_prompt
        )
        self.summary_w_query_prompt = getattr(
            prompts_module, summary_w_query_prompt
        )
        self.summarize_msg_prompt = getattr(
            prompts_module, "SUMMARIZE_MSG_PROMPT"
        )
        manager_prompt_module = importlib.import_module(
            ".config.manager_prompts", package=__name__.rsplit(".", 1)[0])
        self.compress_prompt = getattr(
            manager_prompt_module, compress_prompt
        )
        self.scratchpad_update_prompt = getattr(
            manager_prompt_module, scratchpad_update_prompt
        )
        self.init_scratchpad_prompt = getattr(
            manager_prompt_module, init_scratchpad_prompt
        )
        self.manager_init_scratchpad = getattr(
            manager_prompt_module, params["manager_init_scratchpad"]
        )
        self.manager_act_prompt = getattr(
            manager_prompt_module, params["manager_act_prompt"]
        )
        self.manager_update_prompt = getattr(
            manager_prompt_module, params["manager_update_prompt"]
        )
        self.manager_act_decide_prompt = getattr(
            manager_prompt_module, params["manager_act_decide_prompt"]
        )
        self.manager_step_1_prompt = getattr(
            manager_prompt_module, params["manager_step_1_prompt"]
        )
        self.manager_step_2_prompt = getattr(
            manager_prompt_module, params["manager_step_2_prompt"]
        )
        self.scratchpad = ""
        self.log_cnt = 0
        if self.experiment_logger:
            self.experiment_logger.log_debug(f"MEMORY_INIT: ReActMemory's configs: {params}")
        self.file_list = []
        self.store_dir = f"/tmp/{uuid.uuid4().hex}/"
        os.makedirs(self.store_dir, exist_ok=True)
        # self.service_toolkit.add(self.retrieve_from_files)
        # self.tools = self.model.format_tools_json_schemas(
        #     self.service_toolkit.json_schemas,
        # )
        self.inner_prompt = inner_prompt
        self.deleted_tool_use_ids = set()  # Track deleted tool_use by their call_id
        self.deleted_tool_result_ids = set()
        
        self.intermediate_rewards = []

        self.task_progress = ""
        self.compression_status = ""
        self.compression_experience = []
        self.exp4modifier = []
        self.num_not_print_history = 0 
        self.num_msg = 0
        self.round_number = 0
        self.previous_agent_attempt = []
        self.previous_token_usage_ratio = 0
        self.previous_compression_actions = []
        self.config = config
        self.last_results = None
        self.last_snapshot = None
        self.potential_snapshot = None
        self.ListOutOfTokens = []
        self.max_model_len = params.get("max_model_len", 32768)
        # Compression frequency: compress_fre for fixed interval, or compress_fre_min/max for random range
        self.compress_fre = params.get("compress_fre", None)
        # --- context lock / lineage (thinking-enabled agents) ---
        self.agent_thinking_enabled = bool(params.get("agent_thinking_enabled", False))
        self.lock_violation_penalty = params.get("lock_violation_penalty", None)  # None => silent drop
        self.lineage_log_path = params.get("lineage_log_path", None)
        self.lock_violations_last = []
        self.lineage_records = []
        self.compress_fre_min = params.get("compress_fre_min", None)
        self.compress_fre_max = params.get("compress_fre_max", None)
        self.max_response_tokens = params.get("max_response_tokens", 4096)
        configured_cm_input_limit = params.get("cm_input_limit", DEFAULT_CM_INPUT_LIMIT)
        try:
            configured_cm_input_limit = int(configured_cm_input_limit)
        except (TypeError, ValueError):
            configured_cm_input_limit = DEFAULT_CM_INPUT_LIMIT
        self.cm_input_limit = min(
            int(self.max_model_len or configured_cm_input_limit),
            configured_cm_input_limit,
        ) if configured_cm_input_limit > 0 else int(self.max_model_len or DEFAULT_CM_INPUT_LIMIT)
        self.token_warning_ratio = float(
            params.get("token_warning_ratio", DEFAULT_TOKEN_WARNING_RATIO)
        )
        self.degeneration_config = DEFAULT_DEGENERATION_CONFIG.copy()
        self.degeneration_config.update(params.get("degeneration") or {})
        self.next_compression_target = 1

    def _format_token_usage_ratio(self, cur_token: int) -> str:
        ratio_denominator = int(self.cm_input_limit or self.max_model_len or 1)
        ratio = cur_token / ratio_denominator if ratio_denominator > 0 else 1.0
        ratio_text = (
            f"{ratio * 100:.2f}% "
            f"(current_tokens={cur_token}, cm_input_limit={ratio_denominator}, "
            f"max_model_len={self.max_model_len})"
        )
        if ratio >= self.token_warning_ratio:
            ratio_text += (
                "\n\nCRITICAL CONTEXT BUDGET WARNING: The memory is already near or above "
                "the safe Context Manager input budget. "
                "You MUST compress aggressively now and target the rewritten memory below "
                "60% of input budget. Remove or heavily summarize old tool results, "
                "repeated reasoning, stale intermediate attempts, and unsupported details. "
                "Preserve only the original task, confirmed evidence, unresolved constraints, "
                "and the most recent essential actions."
            )
        return ratio_text

    def _build_compression_metrics(self, before_tokens: int, after_tokens: int) -> dict:
        tool_budget = 4096
        cm_output_budget = int(self.max_response_tokens or 4096)
        safety_margin = 1024
        required_reserve = tool_budget + cm_output_budget + safety_margin
        max_model_len = int(self.max_model_len or 0)
        target_after_tokens = max(0, max_model_len - required_reserve)
        return {
            "before_tokens": before_tokens,
            "after_tokens": after_tokens,
            "saved_tokens": before_tokens - after_tokens,
            "max_model_len": max_model_len,
            "tool_budget": tool_budget,
            "cm_output_budget": cm_output_budget,
            "safety_margin": safety_margin,
            "required_reserve": required_reserve,
            "target_after_tokens": target_after_tokens,
            "budget_deficit": max(0, after_tokens - target_after_tokens),
        }

    def _detect_modification_degeneration(self, modifications: list[dict]) -> tuple[list[dict], dict]:
        config = self.degeneration_config
        if not config.get("enabled", True):
            return [], {"checked_count": 0, "severe_count": 0, "min_compression_ratio": None}

        rewards = []
        severe_count = 0
        ratios = []

        for item_index, modification in enumerate(modifications):
            new_content = modification.get("new_content", "")
            if not isinstance(new_content, str):
                continue

            result = detect_repetition_level(
                new_content,
                min_text_bytes=int(config.get("min_text_bytes_for_severe", 800)),
                severe_compression_ratio=float(config.get("severe_compression_ratio", 0.20)),
            )
            compression_ratio = result.get("compression_ratio")
            if compression_ratio is not None:
                ratios.append(float(compression_ratio))

            level = result.get("level")
            if level == "severe":
                severe_count += 1
                rewards.append({
                    "reason": "degenerate_generation",
                    "reward": float(config.get("severe_reward", -1.0)),
                    "level": "severe",
                    "compression_ratio": compression_ratio,
                    "text_bytes": result.get("text_bytes", 0),
                    "compressed_bytes": result.get("compressed_bytes", 0),
                    "item_index": item_index,
                })

        stats = {
            "checked_count": len(modifications),
            "severe_count": severe_count,
            "min_compression_ratio": min(ratios) if ratios else None,
        }
        return rewards, stats

    # _truncate_text() has been removed - it referenced undefined self.emb_tokenizer
    # (initialization was already commented out) and is no longer used.
    
    def _recover_msg(self, msg: Msg) -> Msg:
        """Recover a message object. In this implementation, it just returns the message as is.
        
        Args:
            msg: The message to recover
            
        Returns:
            The recovered message
        """
        return msg

    def _save_debug_history(self, msgs:List[Msg], hint:str=None) -> None:
        """Save the chat history to a file for debugging purposes."""
        if not self.debug_dir:
            return
        with open(f"{self.debug_dir}/memory.json", "a", encoding="utf-8") as f:
            if hint:
                f.write(hint+'\n')
            if msgs:
                all_items=[]
                for item in msgs:
                    all_items.append(output_format(item))
                f.write(json.dumps(
                    all_items, indent=2, ensure_ascii=False))
            f.write("\n===========END===========\n")

    async def get_memory(
        self,
        recent_n: Optional[int] = DEFAULT_RETURN_CHAT_HISTORY_LEN,
        filter_func: Optional[Callable[[int, dict], bool]] = None,
        process_returned: Optional[Callable[[List[Msg]], Any]] = None,
        retrieve_type: Optional[Literal["source", "processed", "auto"]] = None,
    ) -> list:
        """Retrieve memory according to the `retrieve_type`, number of memories 
        (`recent_n`) and filter function `filter_func`.

        Args:
            recent_n (Optional[int], default `None`):
                The number of memories to return.
            filter_func (Callable[[int, dict], bool], default to `None`):
                The function to filter memories, which take the index and
                memory unit as input, and return a boolean value.
            process_returned
                (Callable[[List[Msg]], Any], default to `None`):
                The function to process memories before returning the memory
                content, which take a list of memory units as input,
                and return the processed memory.
            retrieve_type
                (Literal["source", "processed", "auto"], default to `None`):
                The type of retrieval to perform.
                "source" to retrieve the exact chat history (`_chat_history`),
                "processed" to retrieve the processed memories (`_memory`),
                "auto" to use the retrieval type according to `max_word_size`,
                or None to use the instance's default.

        Returns:
            list:
                A list of retrieved memories, filtered and processed according
                to `filter_func` and `process_returned`.
        Raises:
            ValueError: the retrieve_type is invalid
        """
        # Use instance's default retrieve_type if None is provided
        if retrieve_type is None:
            retrieve_type = self.memory_retrieve_type
        if recent_n is None or recent_n <= 0:
            if self.experiment_logger:
                self.experiment_logger.log_warning(
                    "The retrieved number of memories is set to None or "
                    "less than or equal to 0, returning "
                    f"{DEFAULT_RETURN_CHAT_HISTORY_LEN} memories by default."
                )
            recent_n = DEFAULT_RETURN_CHAT_HISTORY_LEN
        # extract the recent `recent_n` entries in memories
        if retrieve_type == "source":
            memories = self._chat_history
        elif retrieve_type == "processed":
            memories = self._chat_history

        elif retrieve_type == "auto":
            memories = self._chat_history
            retrieve_type = "source"
        else:
            raise ValueError(
                f"Invalid retrieve_type: {retrieve_type}. "
                "Must be 'source', 'processed', 'auto', or None.",
            )

        if filter_func is not None:
            filtered_memories = [
                _ for i, _ in enumerate(memories) if filter_func(i, _)
            ]
        else:
            filtered_memories = memories
        if process_returned is not None:
            processed_memories = process_returned(filtered_memories)
        else:
            processed_memories = [
                self._recover_msg(msg) for msg in filtered_memories
            ]
        if recent_n < len(processed_memories):
            if self.experiment_logger:
                self.experiment_logger.log_warning(
                    f"The requested number of memories is less"
                    f"than the total number of memories, returning"
                    f"the user request and the last {recent_n-1} memories."
                )
            returned_memories = [processed_memories[0]]  # the user request
            returned_memories.extend(processed_memories[-recent_n + 1 :])
        else:
            returned_memories = processed_memories
            # Different from TemporalMemory, ReActMemory will not raise an 
            # error when the requested number of memories is greater than the 
            # total number of memories.
        # logger.log("GET_MEMORY", f"\n{format_msgs(returned_memories)}")
        return returned_memories

    def direct_add_chat_history(
        self,
        msgs: Union[Sequence[Msg], Msg, None],
    ) -> None:
        """Add the chat messages to the `_chat_history` and
        update the estimated word size.

        Args:
            msgs (Union[Sequence[Msg], Msg, None]):
                the messages to add to the chat history
        """
        if msgs is None:
            return
        if not isinstance(msgs, Sequence):
            msgs = [msgs]
        deep_copied_msgs = copy.deepcopy(msgs)
        self._chat_history.extend(deep_copied_msgs)
        # self.cur_chat_len += sum(
        #     [
        #         count_words(
        #             self.chat_tokenizer, format_msgs(msg)
        #         ) for msg in msgs
        #     ]
        # )
        self._save_debug_history(msgs)

    def _extract_tool_use(self, msgs: Sequence[Msg]) -> None:
        for msg in msgs:
            if isinstance(msg.content, str):
                continue
            for c in msg.content:
                if is_tool_use(c):
                    self.num_tool_use_wo_result += 1

    def remove_solved_tools(self, msgs: Union[Sequence[Msg], Msg]) -> None:
        for msg in msgs:
            if isinstance(msg.content, str):
                continue
            for c in msg.content:
                if is_tool_result(c):
                    self.num_tool_use_wo_result -= 1
                    
    async def add(
        self,
        msgs: Union[Sequence[Msg], Msg, None],
    ) -> None:
        """Add new memory fragment to the memory.

        Args:
            msgs (Union[Sequence[Msg], Msg, None]):
                Messages to be added.
            pre_process_func (Optional[Callable[[List[Msg]], List[dict]]]):
                A function to preprocess the memory before adding it to memory.
        """
        if msgs is None:
            return
        if not isinstance(msgs, Sequence):
            msg_to_record = [msgs]
        else:
            msg_to_record = msgs
        self.direct_add_chat_history(msg_to_record)
        self._extract_tool_use(msg_to_record)
        self.remove_solved_tools(msg_to_record)
        if msg_to_record[0].role == "user":
            return
        for c in msg_to_record[0].content:
            if is_tool_use(c) and c.get("name", "") in ["generate_response", "finish"]:
                return
        self.num_msg += 1
        if not has_tool_result(msg_to_record):
            return
        self.round_number += 1

        # Compression frequency control
        # compress_fre: fixed interval (e.g., compress_fre=3 means compress every 3 rounds)
        # compress_fre_min/max: random interval range
        should_compress = True
        if self.compress_fre is not None:
            # Fixed frequency: compress when round_number is divisible by compress_fre
            if self.round_number % self.compress_fre != 0:
                should_compress = False
        elif self.compress_fre_min is not None and self.compress_fre_max is not None:
            if self.round_number == 1:
                gap = random.randint(self.compress_fre_min, self.compress_fre_max)
                self.next_compression_target = gap
            if self.round_number < self.next_compression_target:
                should_compress = False
            else:
                gap = random.randint(self.compress_fre_min, self.compress_fre_max)
                self.next_compression_target = gap + self.round_number

        # Force compress if token usage exceeds max_model_len - 2*self.max_response_tokens (to leave space for response and avoid OOT)
        precomputed_cur_tokens = None
        if not should_compress:
            cur_token = len(self.chat_tokenizer.encode(format_msgs(self._chat_history)))
            if self.max_model_len and cur_token > int(self.max_model_len) - self.max_response_tokens*2:
                should_compress = True
                precomputed_cur_tokens = cur_token
                if self.experiment_logger:
                    self.experiment_logger.log_warning(
                        f"MEMORY_ADD: force compress triggered! cur_token={cur_token}, "
                        f"max_model_len={self.max_model_len}, threshold={int(self.max_model_len) - self.max_response_tokens*2}"
                    )

        if should_compress:
            result = await self.compress(msg_to_record, precomputed_cur_tokens=precomputed_cur_tokens)
            return result
        return

    async def call_modify(self, msgs: Sequence[Msg], token_usage_ratio):
        """Call LLM once to generate memory modifications.

        LLM call retries are handled by call_llm() internally (retry_model_call).
        Parsing/OutOfTokens retries are handled by compress() at a higher level.

        Returns:
            dict with keys:
            - status: "ok" | "OutOfTokens" | "ParsingFailed"
            - attempt: {"type": "success"|"parsing_error"|"out_of_tokens", "error": str}
            - modifications: list (only if status == "ok")
            - modification_response_text: str (only if status == "ok")
        """
        _serialized = (
            format_msgs_with_locks(self._chat_history, thinking_enabled=True, strip_content_ids=True)
            if self.agent_thinking_enabled
            else format_msgs(self._chat_history, strip_content_ids=True)
        )
        _prompt_tmpl = self.manager_step_2_prompt
        if self.agent_thinking_enabled and "### Locked Messages" not in _prompt_tmpl:
            _prompt_tmpl = _prompt_tmpl.replace("## Your Input", LOCK_PROMPT_ADDENDUM + "\n## Your Input", 1)
        prompt_step_2 = _prompt_tmpl.replace(
            "{{full_memory}}", _serialized
        ).replace(
            "{{token_usage_ratio}}", token_usage_ratio
        ).replace(
            "{{Background}}", self.config.get("background_info", "")
        )

        try:
            modification_response = await self.call_llm(prompt=prompt_step_2, role="user", name="user")
        except Exception as e:
            tb = traceback.format_exc()
            error_msg = str(e)
            if ("Error code: 400" in error_msg
                or ("max_tokens" in error_msg.lower() and "too large" in error_msg.lower())):
                if self.experiment_logger:
                    self.experiment_logger.log_warning(
                        f"MEMORY_WARNING: max_tokens too large error detected: {e}\nTraceback: {tb}"
                    )
                return {"status": "OutOfTokens", "attempt": {"type": "out_of_tokens", "error": str(e)}}
            else:
                # Other LLM errors after internal retries exhausted
                if self.experiment_logger:
                    self.experiment_logger.log_warning(
                        f"MEMORY_WARNING: Failed to call LLM: {e}"
                    )
                raise e

        # Try to parse
        response_text = modification_response.content
        try:
            if isinstance(response_text, list) and len(response_text) > 0:
                response_text = response_text[0].get("text", "")

            if not isinstance(response_text, str):
                raise ValueError(f"Unexpected response content type: {type(response_text)}")

            modification_response_text = remove_thinking_block(response_text)
            decoded_object = json_repair.repair_json(modification_response_text, return_objects=True)

            if not decoded_object:
                raise ValueError("json_repair returned null")

            # Silent recover: top-level is a list instead of {"modifications": [...]}
            if isinstance(decoded_object, list):
                decoded_object = {"modifications": decoded_object}

            if not isinstance(decoded_object, dict):
                raise ValueError(f"decoded_object not a dict: {type(decoded_object)}")

            modifications = decoded_object.get("modifications", None)
            if modifications is None:
                raise ValueError(f"Cannot get modifications from response")

            # Silent recover: single dict instead of list of dicts
            if isinstance(modifications, dict):
                if "ids" in modifications:
                    modifications = [modifications]
                else:
                    raise ValueError(f"modifications is a dict without 'ids': {modifications}")
            elif not isinstance(modifications, list):
                raise ValueError(f"modifications is not a list: {type(modifications)}")

            for item in modifications:
                if not isinstance(item, dict):
                    raise ValueError(f"modification item is not a dict: {type(item)}")
                if "ids" not in item:
                    raise ValueError(f"modification item missing 'ids': {item}")
                if "role" not in item:
                    raise ValueError(f"modification item missing 'role': {item}")
                # Silent recover: missing new_content == "" (delete semantics, matching SFT schema)
                if "new_content" not in item:
                    item["new_content"] = ""

            return {
                "status": "ok",
                "modifications": modifications,
                "modification_response_text": modification_response_text,
                "attempt": {"type": "success"},
            }

        except Exception as e:
            if self.experiment_logger:
                self.experiment_logger.log_warning(
                    f"MEMORY_WARNING: Failed to parse modification: {e}"
                )
            return {"status": "ParsingFailed", "attempt": {"type": "parsing_error", "error": str(e)}}

    async def perform_modifications(self, results):
        """Apply modifications to chat history.

        Note: self.last_results is only set at the end on success,
        so BCPWorker can check it to know if compression succeeded.
        """
        try:
            modifications = results.get("modifications", [])

            # --- lock enforcement (thinking-enabled agent) ---
            if self.agent_thinking_enabled:
                modifications, self.lock_violations_last = filter_modifications(
                    modifications, self._chat_history, thinking_enabled=True
                )
                results["modifications"] = modifications
                results["lock_violations"] = [v.__dict__ for v in self.lock_violations_last]
                if self.lock_violations_last and self.experiment_logger:
                    self.experiment_logger.log_warning(
                        f"MEMORY_LOCK: dropped {len(self.lock_violations_last)} op(s): {results['lock_violations']}"
                    )
            # --- lineage: snapshot before, plan uids ---
            _before_history = list(self._chat_history)
            _pre_snapshot = snapshot(_before_history)
            _lineage_recs, _produced = plan_lineage(modifications, _before_history, step=self.round_number)

            delete_list = []
            for r in modifications:
                ids = r["ids"]
                if not isinstance(ids, list):
                    ids = [ids]
                    r["ids"] = ids
                if not isinstance(ids[0], int):
                    ids = [int(id) for id in ids]
                    r["ids"] = ids
            try:
                modifications.sort(key=lambda item: item['ids'][0]) 
            except Exception as e:
                if self.experiment_logger:
                    self.experiment_logger.log_warning(f"MEMORY_WARNING: modifications failed in sorting due to {e}")

            degeneration_rewards, repetition_stats = self._detect_modification_degeneration(modifications)
            results["degeneration_rewards"] = degeneration_rewards
            results["repetition_stats"] = repetition_stats
            if any(reward.get("reason") == "degenerate_generation" for reward in degeneration_rewards):
                if self.experiment_logger:
                    self.experiment_logger.log_warning(
                        f"MEMORY_WARNING: severe degenerate generation detected: {degeneration_rewards}"
                    )
                raise DegenerateGenerationError(degeneration_rewards)

            # Pre-pass: detect no-change modifications (new_content == original content)
            no_change_count = 0
            for r in modifications:
                ids = r["ids"]
                sub = r["new_content"]
                if isinstance(sub, str) and sub.strip() != "" and len(ids) == 1 and ids[0] < len(self._chat_history):
                    orig = self._chat_history[ids[0]].content
                    orig_text = orig
                    if isinstance(orig, list) and len(orig) == 1 and isinstance(orig[0], dict):
                        orig_text = orig[0].get("text", orig[0].get("output", ""))
                    if isinstance(orig_text, str) and orig_text.strip() == sub.strip():
                        no_change_count += 1
            results["no_change_count"] = no_change_count

            # First pass: identify deleted tool_use messages and collect their call_ids
            for r in modifications:
                ids=r["ids"]
                sub=r["new_content"]
                compression_description = r.get("compression_description", "")
                self.previous_compression_actions.append(compression_description)
                # Collect call_ids from deleted messages (empty new_content) and merged-away messages (ids[1:])
                deleted_msg_indices = list(range(1, len(ids)))  # ids[1:] always deleted
                if isinstance(sub, str) and sub.strip() == "":
                    deleted_msg_indices = list(range(len(ids)))  # all deleted
                for i in deleted_msg_indices:
                    msg = self._chat_history[ids[i]]
                    if (isinstance(msg.content, list) and len(msg.content) > 0 and
                        isinstance(msg.content[0], dict)):
                        for call in msg.content:
                            call_id = call.get("id")
                            if call_id and call.get("type") == "tool_use":
                                self.deleted_tool_use_ids.add(call_id)
                            elif call_id and call.get("type") == "tool_result":
                                self.deleted_tool_result_ids.add(call_id)
            
            # Second pass: apply modifications with tool pairing logic
            for r in modifications:
                ids=r["ids"]
                sub=r["new_content"]
                role=r['role'] if isinstance(r['role'], str) else r['role'][0]
                tmp = self._chat_history[ids[0]].content
                if isinstance(tmp, list) and len(tmp) > 0 and isinstance(tmp[0], dict) and "type" in tmp[0]:
                    if is_tool_use(tmp[0]):
                        if isinstance(sub, str) and sub.strip() == "":
                            delete_list.extend(ids)
                        else:
                            if isinstance(sub, str) and sub.strip() != "":
                                self._chat_history[ids[0]] = Msg(role, [TextBlock(type="text", text=sub)], role=role)
                            else:
                                self._chat_history[ids[0]].content = sub
                            delete_list.extend(ids[1:])
                    elif is_tool_result(tmp[0]):
                        tool_use_id = tmp[0].get("id")
                        if tool_use_id in self.deleted_tool_use_ids: 
                            if isinstance(sub, str) and sub.strip() != "":
                                self._chat_history[ids[0]] = Msg(role, [TextBlock(type="text", text=sub)], role=role)
                            elif isinstance(sub, list) and len(sub) > 0 and isinstance(sub[0], dict) and "type" in sub[0] and sub[0].get("type", "") == "tool_result":
                                output = sub[0]["output"]
                                self._chat_history[ids[0]].content = str(output) if not isinstance(output, str) else output
                        elif isinstance(sub, str) and sub.strip() != "":
                            self._chat_history[ids[0]].content[0]['output'] = sub
                        else:
                            self._chat_history[ids[0]].content = sub
                        if isinstance(sub, str) and sub.strip() == "" or (isinstance(sub, list) and len(sub) == 0): 
                            delete_list.extend(ids)
                        else:
                            delete_list.extend(ids[1:])
                    else:
                        if tmp[0].get("type", "") != "text":
                            if self.experiment_logger:
                                self.experiment_logger.log_warning(f"MEMORY_WARNING: text block found in tool use/result: {tmp[0]}")
                        if isinstance(sub, str) and sub.strip() != "":
                            self._chat_history[ids[0]] = Msg(role, [TextBlock(type="text", text=sub)], role=role)
                            delete_list.extend(ids[1:])
                        else:
                            delete_list.extend(ids)
                else:
                    if isinstance(sub, str) and sub.strip()!="":
                        self._chat_history[ids[0]]= Msg(role, [TextBlock(type="text", text=sub)], role=role)
                        delete_list.extend(ids[1:])
                    elif isinstance(sub, list):
                        self._chat_history[ids[0]].content = sub
                        delete_list.extend(ids[1:])
                    else:
                        delete_list.extend(ids)
            self._chat_history = [
                m for i, m in enumerate(self._chat_history) if i not in delete_list
            ]
            delete_list = []
            # Linear left-to-right pairing: kimi and some providers reuse the
            # same call_id across separate tool calls, so we cannot key by id.
            # Walk the history, pair each tool_result with the nearest earlier
            # unmatched tool_use sharing the same id (FIFO per id).
            tool_use_positions = []     # (call_id, msg_idx, content_idx)
            tool_result_positions = []  # (call_id, msg_idx, content_idx)
            for msg_idx, msg in enumerate(self._chat_history):
                if not isinstance(msg.content, list):
                    continue
                for content_idx, content_item in enumerate(msg.content):
                    if not isinstance(content_item, dict):
                        continue
                    call_id = extract_tool_id(content_item)
                    if not call_id:
                        continue
                    if is_tool_use(content_item):
                        tool_use_positions.append((call_id, msg_idx, content_idx))
                    elif is_tool_result(content_item):
                        tool_result_positions.append((call_id, msg_idx, content_idx))

            # Greedy FIFO pairing within each call_id
            pending_by_id = {}  # call_id -> list of tool_use indices in tool_use_positions
            for tui, (cid, _, _) in enumerate(tool_use_positions):
                pending_by_id.setdefault(cid, []).append(tui)
            matched_tool_use_indices = set()
            orphan_tool_result_positions = []
            for cid, tr_mi, tr_ci in tool_result_positions:
                # Find earliest unmatched tool_use with same id appearing before this tool_result
                chosen = None
                for tui in pending_by_id.get(cid, []):
                    if tui in matched_tool_use_indices:
                        continue
                    _, tu_mi, _ = tool_use_positions[tui]
                    if tu_mi < tr_mi:
                        chosen = tui
                        break
                if chosen is not None:
                    matched_tool_use_indices.add(chosen)
                else:
                    orphan_tool_result_positions.append((tr_mi, tr_ci))
            orphan_tool_use_msg_indices = [
                tool_use_positions[tui][1]
                for tui in range(len(tool_use_positions))
                if tui not in matched_tool_use_indices
            ]

            # Delete unpaired tool_use, convert unpaired tool_result to plain text
            for msg_idx in orphan_tool_use_msg_indices:
                delete_list.append(msg_idx)

            for msg_idx, content_idx in orphan_tool_result_positions:
                msg = self._chat_history[msg_idx]
                if not (isinstance(msg.content, list) and len(msg.content) == 1):
                    if self.experiment_logger:
                        self.experiment_logger.log_warning(f"MEMORY_WARNING: unexpected content format for unpaired tool_result: {msg.content}")
                    continue
                output = msg.content[0].get("output", "")
                to_sub = ""
                if isinstance(output, str):
                    to_sub = output
                elif isinstance(output, list):
                    for o in output:
                        if isinstance(o, dict) and o.get("type") == "text":
                            to_sub += o.get("text", "")
                        else:
                            to_sub += json.dumps(o, ensure_ascii=False)
                msg.content = [TextBlock(type="text", text=to_sub)]
            self._chat_history = [
                m for i, m in enumerate(self._chat_history) if i not in delete_list
            ]
            if modifications:
                self._save_debug_history(self._chat_history, hint=f"Round {self.round_number} Modifications applied:\n{json.dumps(modifications, indent=2, ensure_ascii=False)}\n")
            else:
                self._save_debug_history([], hint=f"No modifications applied")

            # --- lineage: stamp produced uids, persist ---
            stamp_produced_uids(self._chat_history, _before_history, _produced)
            self.lineage_records.extend(_lineage_recs)
            results["lineage"] = records_to_json(_lineage_recs)
            if self.lineage_log_path:
                try:
                    with open(self.lineage_log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "round": self.round_number,
                            "pre": _pre_snapshot,
                            "post": snapshot(self._chat_history),
                            "modifications": modifications,
                            "lock_violations": results.get("lock_violations", []),
                            "lineage": results["lineage"],
                        }, ensure_ascii=False) + "\n")
                except Exception as _e:
                    if self.experiment_logger:
                        self.experiment_logger.log_warning(f"LINEAGE_LOG failed: {_e}")

            # Only set last_results on successful completion
            self.last_results = results
            return
        except Exception as e:
            if isinstance(e, MemoryError):
                raise e
            if self.experiment_logger:
                self.experiment_logger.log_warning(f"Error in perform_modifications: {e}")
            raise MemoryError(f"Error in perform_modifications: {e}") from e 

    async def compress(
        self,
        msgs: Union[Sequence[Msg], Msg, None],
        precomputed_cur_tokens: Optional[int] = None,
    ) -> Optional[dict]:
        
        # if has_tool_use(msgs) and self.round_number == 0:
        #     self.previous_agent_attempt.extend(msgs)
        # elif has_tool_use(msgs) and len(self.previous_agent_attempt) > 0:
        #     if msgs[0].content[0].get("type", "") == "tool_use" and msgs[0].content[0].get("name", "") == "generate_response":
        #         return
        #     ori_prompt = self.manager_step_1_prompt.replace("{{round_number}}", str(self.round_number)).replace("{{previous_agent_attempt}}", format_msgs(self.previous_agent_attempt)).replace("{{previous_token_usage_ratio}}", self.previous_token_usage_ratio).replace("{{previous_compression_actions}}", json.dumps(self.previous_compression_actions)).replace("{{agent_context}}", format_msgs(self._chat_history[:-self.num_msg])).replace("{{current_agent_action}}", format_msgs(self._chat_history[-self.num_msg:], with_id=False)).replace("{{current_compression_experience}}", json.dumps(self.compression_experience)).replace("{{Background}}", self.config.get("background_info", ""))
        #     prompt = copy.deepcopy(ori_prompt)
            
        #     # Retry logic for LLM calls, but not for parsing
        #     max_retries = 5
        #     updated_experience = []
        #     response = None
        #     response_text = None
            
        #     # Retry only for LLM call failures
        #     for retry_attempt in range(max_retries):
        #         try:
        #             response = await self.call_llm(prompt=prompt, role="user", name="user")
        #             break  # Success, exit retry loop
        #         except Exception as e:
        #             tb = traceback.format_exc()
        #             error_msg = str(e)
        #             if ("'max_tokens' or 'max_completion_tokens' is too large" in error_msg
        # or ("max_tokens" in error_msg.lower() and "too large" in error_msg.lower())):
            #             if self.experiment_logger:
            #                 self.experiment_logger.log_warning(f"MEMORY_WARNING: max_tokens too large error detected: {e}\nTraceback: {tb}")
            #             raise e  # Re-raise max_tokens errors immediately
            #         else:
            #             if self.experiment_logger:
            #                 self.experiment_logger.log_warning(f"MEMORY_WARNING: Failed to call LLM on attempt {retry_attempt + 1}/{max_retries}: {e}\nTraceback: {tb}")
            #             if retry_attempt >= max_retries - 1:
            #                 raise e  # Re-raise after all retries exhausted
            
            # # No retry for parsing - raise MemoryError on failure
            # try:
            #     response_text = remove_thinking_block(response.content[0].get("text", ""))
            #     decoded_object = json_repair.repair_json(response_text, return_objects=True)
            #     if not decoded_object:
            #         raise ValueError(f"json_repair return null")
            #     updated_experience = decoded_object.get("updated_experience", [])
            #     if self.experiment_logger:
            #         self.experiment_logger.log_debug(f"MEMORY_INFO: Step 1 response parsed successfully")
            # except Exception as e:
            #     tb = traceback.format_exc()
            #     if self.experiment_logger:
            #         self.experiment_logger.log_warning(f"MEMORY_WARNING: Failed to parse step 1 response: {e} The original output: {response_text}\nTraceback: {tb}")
            #     # Raise MemoryError for parsing failures
            #     raise MemoryError(f"Failed to parse memory compression response: {e} The original output: {response_text}\nTraceback: {tb}")
            # if updated_experience:
            #     self._save_debug_history([], hint=f"Updated experience: {json.dumps(updated_experience, indent=2, ensure_ascii=False)}\n")
            #     # Keep only the 10 most recent experiences if total exceeds 10
            #     if len(updated_experience) > 10:
            #         # Sort by last_updated time (most recent first) and keep top 10
            #         updated_experience.sort(key=lambda x: int(x.get('last_updated', '0')), reverse=True)
            #         updated_experience = updated_experience[:10]
            #     for ex in updated_experience:
            #         ex.pop("updated", None)
            #         ex.pop("reasoning", None)
            #     self.exp4modifier = copy.deepcopy(updated_experience)
            #     for ex in self.exp4modifier:
            #         ex.pop("evidence", None)
            #         ex.pop("last_updated", None)
            #     self.compression_experience=updated_experience
            # self.previous_agent_attempt = []
            # self.previous_agent_attempt.extend(msgs)

        self.previous_agent_attempt.extend(msgs)
        reorganized_msgs = [] # each tool use with the following tool results
        new_msgs = self._chat_history[-self.num_msg:]
        # pair tool use and tool result by id
        tool_use_queue = []
        tool_result_queue = []
        for msg in new_msgs:
            if isinstance(msg.content, str):
                # Preserve plain-text messages (e.g. assistant text without tool use)
                reorganized_msgs.append(copy.deepcopy(msg))
                continue
            if isinstance(msg.content, list) and len(msg.content) > 0 and isinstance(msg.content[0], dict):
                latest_text_block = None
                queue_start = len(tool_use_queue)
                for item in msg.content:
                    if is_tool_use(item):
                        if len(msg.content) > 1:
                            tmp_msg = copy.deepcopy(msg)
                            if latest_text_block:
                                tmp_msg.content = [latest_text_block] + [item]
                                latest_text_block = None
                            else:
                                tmp_msg.content = [item]
                            tool_use_queue.append(tmp_msg)
                        else:
                            tool_use_queue.append(copy.deepcopy(msg))
                    elif item.get("type", "") == "tool_result":
                        tool_result_queue.append(copy.deepcopy(msg)) # always only one
                    elif item.get("type", "") == "text":
                        latest_text_block = copy.deepcopy(item)
                # Attach remaining text block to first tool_use from this msg
                if latest_text_block and queue_start < len(tool_use_queue):
                    first_tu = tool_use_queue[queue_start]
                    first_tu.content = [latest_text_block] + first_tu.content
        converted_orphan_tool_result_count = 0
        # Match by iterating tool_result, find corresponding tool_use; convert unmatched to text in-place
        for tr_msg in tool_result_queue:
            tr_id = tr_msg.content[0].get("id", "")
            matched_tu = None
            for i, tu_msg in enumerate(tool_use_queue):
                for c in tu_msg.content:
                    if c.get("type") == "tool_use" and c.get("id", "") == tr_id:
                        matched_tu = tool_use_queue.pop(i)
                        break
                if matched_tu:
                    break
            if matched_tu:
                reorganized_msgs.append(matched_tu)
                reorganized_msgs.append(tr_msg)
            else:
                # No matching tool_use — convert to plain text (GPT rejects orphan tool role)
                tr_block = tr_msg.content[0]
                output = tr_block.get("output", "")
                name = tr_block.get("name", "tool")
                reorganized_msgs.append(Msg(name="system", content=f"[{name} result]: {output}", role="user"))
                converted_orphan_tool_result_count += 1
        # Drop remaining orphan tool_use (no matching result)
        dropped_orphan_tool_use_count = len(tool_use_queue)
        if dropped_orphan_tool_use_count and self.experiment_logger:
            self.experiment_logger.log_warning(f"MEMORY_COMPRESS: dropped {dropped_orphan_tool_use_count} orphan tool_use messages")
        self._chat_history = self._chat_history[:-self.num_msg] + reorganized_msgs
        if (
            precomputed_cur_tokens is not None
            and len(reorganized_msgs) == len(new_msgs)
            and dropped_orphan_tool_use_count == 0
            and converted_orphan_tool_result_count == 0
        ):
            cur_token = precomputed_cur_tokens
        else:
            cur_token = len(self.chat_tokenizer.encode(format_msgs(self._chat_history)))
        token_usage_ratio = self._format_token_usage_ratio(cur_token)
        self.num_msg = 0
        # Attempt compression once. OutOfTokens is terminal for the rollout;
        # parsing/modification errors abandon this compression event immediately.
        all_attempts = []
        before_tokens = cur_token

        results = await self.call_modify(self._chat_history, token_usage_ratio=token_usage_ratio)

        if results.get("status") == "ok":
            # Try to apply modifications
            self.potential_snapshot = copy.deepcopy(self._chat_history)
            try:
                await self.perform_modifications(results)
            except DegenerateGenerationError as e:
                all_attempts.append({"type": "degenerate_generation", "error": str(e)})
                self._chat_history = copy.deepcopy(self.potential_snapshot)
                self.potential_snapshot = None
                if self.experiment_logger:
                    self.experiment_logger.log_warning(
                        f"DegenerateGeneration; abandoning compression event without applying modifications: {e}"
                    )
                results["status"] = "DegenerateGeneration"
                results["attempts"] = all_attempts
                results["degeneration_rewards"] = e.rewards
                self.last_results = results
                return results
            except MemoryError as e:
                # Modification failed: cancel this compression event without retry.
                all_attempts.append({"type": "modification_error", "error": str(e)})
                self._chat_history = copy.deepcopy(self.potential_snapshot)
                self.potential_snapshot = None
                if self.experiment_logger:
                    self.experiment_logger.log_warning(
                        f"perform_modifications failed; abandoning compression event: {e}"
                    )
                results["status"] = "ModificationFailed"
                results["attempts"] = all_attempts
                self.last_results = results
                return results

            # Modification succeeded - now record the success
            attempt = results.get("attempt")
            if attempt:
                all_attempts.append(attempt)
            results["attempts"] = all_attempts
            if self.potential_snapshot is not None:
                self.last_snapshot = self.potential_snapshot
                self.potential_snapshot = None
            if not results.get("modifications"):
                after_tokens = before_tokens
            else:
                after_tokens = len(self.chat_tokenizer.encode(format_msgs(self._chat_history)))
            results["compression_metrics"] = self._build_compression_metrics(
                before_tokens=before_tokens,
                after_tokens=after_tokens,
            )
            self.last_results = results
            return results

        if results.get("status") == "OutOfTokens":
            attempt = results.get("attempt")
            if attempt:
                all_attempts.append(attempt)
            if self.experiment_logger:
                self.experiment_logger.log_error(
                    "OutOfTokens detected during compression; marking rollout terminal"
                )
            results["status"] = "UnrecoverableOutOfTokens"
            results["attempts"] = all_attempts
            self.last_results = results
            return results

        if results.get("status") == "ParsingFailed":
            attempt = results.get("attempt")
            if attempt:
                all_attempts.append(attempt)
            if self.experiment_logger:
                self.experiment_logger.log_warning(
                    "ParsingFailed; abandoning compression event"
                )
            results["attempts"] = all_attempts
            self.last_results = results
            return results

        unexpected_status = results.get("status")
        if self.experiment_logger:
            self.experiment_logger.log_error(
                f"Unexpected compression status: {unexpected_status}"
            )
        results["status"] = "ModificationFailed"
        results["attempts"] = [{"type": "modification_error", "error": f"Unexpected status: {unexpected_status}"}]
        self.last_results = results
        return results

    async def extract_from_content(self, content: str, query: str) -> str:
        """Extract information from content relevant to a query.

        If the content fits within (max_model_len - 5000) tokens, extract in
        one shot.  Otherwise, split into chunks and process sequentially using
        a rolling reading-note approach.

        Args:
            content: The raw text content (e.g. webpage body).
            query: What information to extract / the user goal.

        Returns:
            str: Extracted information summary.
        """
        max_chunk_tokens = self.max_model_len - 5000

        content_tokens = self.chat_tokenizer.encode(content)
        if self.experiment_logger:
            self.experiment_logger.log_info(
                f"MEMORY_EXTRACT: content_tokens={len(content_tokens)}, max_chunk_tokens={max_chunk_tokens}"
            )
        if len(content_tokens) <= max_chunk_tokens:
            # Single-shot extraction
            chunks = [content]
        else:
            # Split into overlapping chunks
            overlap = min(200, max_chunk_tokens // 10)
            chunks = []
            start = 0
            while start < len(content_tokens):
                end = start + max_chunk_tokens
                chunk_tokens = content_tokens[start:end]
                chunks.append(self.chat_tokenizer.decode(chunk_tokens))
                start = end - overlap

        previous_notes = ""
        tot_len = str(len(chunks))
        for chunk_idx, chunk in enumerate(chunks):
            prompt_text = (
                self.summary_w_query_prompt
                .replace("{{chunk_idx}}", str(chunk_idx + 1))
                .replace("{{total_chunks}}", tot_len)
                .replace("{{chunk}}", chunk)
                .replace("{{existing_notes}}", previous_notes)
                .replace("{{question}}", query)
            )
            response = await self.call_llm(prompt=prompt_text, role="user", name="user")
            # Extract text from response
            resp_content = response.content
            if isinstance(resp_content, list) and len(resp_content) > 0:
                if isinstance(resp_content[0], dict):
                    previous_notes = resp_content[0].get("text", "")
                else:
                    previous_notes = str(resp_content[0])
            elif isinstance(resp_content, str):
                previous_notes = resp_content
            else:
                previous_notes = str(resp_content)
        return previous_notes

    async def call_llm(self, prompt: Optional[str]=None, role: Optional[str]=None, name: Optional[str]=None, append_msgs:
                 Optional[List[Msg]]=None) -> Any:
        msgs = []
        if prompt is not None and role is not None and name is not None:
            msgs=[Msg(name,[TextBlock(type="text", text=prompt)],role=role),]
        if append_msgs is not None:
            msgs.extend(append_msgs)
        if len(msgs) == 0:
            raise ValueError("No messages to call LLM.")
        formatted_prompt = await self.formatter.format(msgs=msgs)
        if self.experiment_logger:
            prompt_len = sum(len(str(m.get("content", ""))) for m in formatted_prompt) if isinstance(formatted_prompt, list) else len(str(formatted_prompt))
            self.experiment_logger.log_info(
                f"MEMORY_LLM: call_llm formatted_prompt char_len={prompt_len}"
            )
        try:
            # Set temperature=0.0 for API models (non-TrinityModel) for deterministic output
            # extra_kwargs = {}
            # if not isinstance(self.model, TrinityChatModel):
            #     extra_kwargs["temperature"] = 0.0

            response = await retry_model_call(
                self.model,
                formatted_prompt,
                max_retries=20,
                sleep_seconds=10,
                experiment_logger=self.experiment_logger,
                # **extra_kwargs
            )
            
            # Handle streaming response if needed
            final_response = None
            if isinstance(response, AsyncGenerator):
                # It's an async generator (streaming)
                final_response = Msg("system", [], "assistant")
                async for chunk in response:
                    final_response.content = chunk.content
            else:
                # Direct response
                final_response = response
            
            # Logging is now handled automatically by the LLM hook system
            
            return final_response
        except Exception as e:
            tb = traceback.format_exc()
            if self.experiment_logger:
                self.experiment_logger.log_error(f"Error calling LLM: {e}\nTraceback: {tb}")
            raise e

    # reply() has been removed - it referenced undefined MAX_CHAT_MODEL_TOKEN_SIZE
    # and is no longer used in the current architecture.


    # Note: _reasoning and _acting methods are not used in MemoryManager
    # The memory manager is not an agent itself, it's a memory component
    # These methods were likely copied from an agent template but are not needed

    async def clear(self) -> None:
        """
        Clear all memory.
        """
        if not self.debug_mode:
            self._chat_history = []
        self.cur_chat_len = 0
        if self.vector_store is not None:
            self.vector_store.reset()

    def size(self) -> int:
        """Returns the number of memory segments in memory."""
        return len(self._chat_history)

    def delete(self, index: Union[Iterable, int]) -> None:
        raise NotImplementedError(
            """
            `Delete` is not supported in ReActMemory, use
            `direct_delete_memory` or `direct_delete_chat_history` instead.
            """
        )


    def direct_delete_chat_history(self, index: Union[uuid.UUID, str]) -> None:
        """
        Delete chat history in self._chat_history.
        """
        found = False
        index = str(index)
        for idx, msg in enumerate(self._chat_history):
            if msg.id == index:
                self._chat_history.pop(idx)
                found = True
                break
        if not found:
            if self.experiment_logger:
                self.experiment_logger.log_warning(
                    f"MEMORY_WARNING: Chat history {index} not found to delete"
                )
