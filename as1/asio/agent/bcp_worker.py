# -*- coding: utf-8 -*-
"""BrowseComp-Plus Worker agent for document search and retrieval tasks.

This worker supports both local search (loading index directly) and 
remote search via MCP (Model Context Protocol) server.
"""

import os
import json
import sys
import re
import traceback
from typing import Optional, Any, Dict, List, Type
import asyncio
import importlib
import uuid

from agentscope.agent import ReActAgent
from agentscope.model import ChatModelBase
from agentscope.formatter import FormatterBase
from agentscope.tool import Toolkit, ToolResponse
from agentscope.memory import MemoryBase
from agentscope.message import Msg, TextBlock, ToolUseBlock, ToolResultBlock
from asio.logger import ExperimentLogger
from asio.utils.retry import (
    retry_model_call, convert_tools_openai_to_anthropic, detect_model_provider,
    get_thinking_kwargs, get_anthropic_sampling_kwargs,
)
from collections import defaultdict
from asio.memory.utils import format_msgs
from asio.agent.background_info import BCP_BACKGROUND_INFO, resolve_background_info
from asio.utils.cost_guard import build_cost_tracker
from asio.agent.memory_reward_utils import (
    build_deferred_out_of_tokens_rewards,
    build_insufficient_budget_reward,
    build_lock_violation_reward,
    build_recent_compression_record,
    build_terminal_out_of_tokens_reward,
    calculate_budget_penalty,
    get_terminal_out_of_tokens_config,
)
try:
    from fastmcp import Client
    from fastmcp.client.transports import SSETransport
except ImportError:
    Client = None
    SSETransport = None

try:
    import anyio
    import httpx
    from mcp.shared._httpx_utils import create_mcp_http_client
except ImportError:
    anyio = None
    httpx = None
    create_mcp_http_client = None


def _make_mcp_http_factory(post_timeout: float = 60.0):
    """Build an httpx factory that forces a longer POST timeout for MCP /messages.

    fastmcp SSETransport has no `timeout` kwarg; mcp sse_client hardcodes 5s,
    which blows up under contention with co-located vLLM/faiss. We intercept the
    factory to override the POST-side timeout while preserving the SSE read window.
    """
    def factory(*, headers=None, auth=None, timeout=None):
        read_timeout = timeout.read if isinstance(timeout, httpx.Timeout) else 300.0
        return create_mcp_http_client(
            headers=headers,
            auth=auth,
            timeout=httpx.Timeout(post_timeout, read=read_timeout),
        )
    return factory


BCP_SYSTEM_PROMPT = """You are a deep research agent. You need to answer the given question by interacting with a search engine, using the search and get_document tools provided. Please perform reasoning and use the tool step by step, in an interleaved manner.

NOTE:
  - You should always call one tool at a time. Use short keyword as the query for the search tool call.
  - You should first provide your reasoning process(your chain of thought) before each tool call step.
  - When you have a definitive answer or cannot progress further, call the 'finish' tool to provide your final answer.
"""


# Kept as a module-level name for compatibility; the variants (and the
# Anthropic tool-use one) live in asio.agent.background_info.
Background_info = BCP_BACKGROUND_INFO


def get_model_call_kwargs(model, enable_thinking: bool = True, temperature: float = 0,
                          thinking_config: dict = None) -> dict:
    """Get all model-specific kwargs (temperature + thinking) for agent calls."""
    model_name = (getattr(model, "model_name", "") or "").lower()
    # Claude rejects temperature+top_p together; gpt-5 doesn't accept temperature at all.
    # With thinking on (or on Claude 4.7+ at all) Claude takes no sampling
    # params whatsoever — get_anthropic_sampling_kwargs holds those rules.
    if "gpt-5" in model_name:
        kwargs = {}
    elif "claude" in model_name or "anthropic" in model_name:
        kwargs = get_anthropic_sampling_kwargs(model_name, temperature, enable_thinking)
    else:
        kwargs = {"temperature": temperature, "top_p": 1}
    thinking = get_thinking_kwargs(model, enable_thinking, thinking_config)
    # Merge extra_body if both have it
    if "extra_body" in kwargs and "extra_body" in thinking:
        kwargs["extra_body"] = {**kwargs["extra_body"], **thinking["extra_body"]}
    else:
        kwargs.update(thinking)
    return kwargs


class BCPWorker(ReActAgent):
    """A worker agent specialized for BrowseComp-Plus document search tasks.

    This worker extends ReActAgent to handle BrowseComp-Plus specific interactions:
    - Search and document retrieval via local indices or MCP server
    - Integration with various searcher backends (BM25, etc.)

    # Class-level searcher cache: shared across all BCPWorker instances within
    # the same process, keyed by (searcher_type, index_path, model_name).
    # Avoids loading heavy embedding models / FAISS indices per worker.
    _shared_searchers: dict = {}
    _shared_searchers_lock = None  # initialized lazily
    - Memory management for search history
    """

    finish_function_name: str = "finish"
    """The function name used to finish replying and return a response to
    the user.
    """
    
    def __init__(
        self,
        name: str,
        model: ChatModelBase,
        formatter: FormatterBase,
        searcher_type: str = "bm25",
        index_path: str = "indexes/bm25",
        toolkit: Optional[Toolkit] = None,
        memory: Optional[MemoryBase] = None,
        max_iters: int = 50,
        memory_class: Optional[str] = None,
        memory_config: Optional[dict] = None,
        experiment_logger: Optional[ExperimentLogger] = None,
        top_k: int = 5,
        snippet_max_tokens: int = 512,
        doc_max_tokens: int = 4096,
        include_get_document: bool = True,
        browsecomp_path: Optional[str] = None,
        searcher_model_name: str = "",
        mcp_url: Optional[str] = None,
        gold_docs: Optional[Dict] = None,
        key_docs: Optional[Dict] = None,
        task_id: Optional[str] = None,
        calculate_reward: bool = False,
        reward_config: Optional[Dict] = None,
        stop_on_no_tool_use: bool = True,
        tokenizer_model: Optional[str] = None,
        tokenizer: Optional[Any] = None,
        enable_thinking: bool = True,
        thinking_config: Optional[dict] = None,
        cost_config: Optional[dict] = None,
        agent_temperature: float = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize the BrowseComp-Plus Worker agent.

        Args:
            name: The name of the agent.
            model: The chat model instance for the agent.
            formatter: The formatter instance to format messages.
            searcher_type: Type of searcher to use (e.g., "bm25", "dense").
            index_path: Path to the search index.
            toolkit: A Toolkit object that contains the tool functions.
            memory: The memory used to store the dialogue history.
            max_iters: Maximum number of iterations for reasoning-acting loops.
            memory_class: The custom memory class name to use.
            memory_config: Configuration for the custom memory class.
            experiment_logger: Logger for tracking experiments.
            top_k: Number of top search results to return.
            snippet_max_tokens: Maximum tokens for document snippets.
            doc_max_tokens: Maximum tokens for full document content.
            include_get_document: Whether to include get_document tool.
            browsecomp_path: Path to BrowseComp-Plus installation.
            searcher_model_name: Name of the model for the searcher (e.g. for embedding searcher).
            mcp_url: URL of the MCP server (e.g. http://127.0.0.1:8080/mcp).
            gold_docs: Dictionary of gold documents for reward calculation.
            key_docs: Dictionary of key documents for reward calculation.
            task_id: Current task ID for reward calculation.
            calculate_reward: Whether to calculate intermediate rewards for compression.
            reward_config: Configuration for reward values (gold_positive, gold_negative, etc.).
            stop_on_no_tool_use: If True (default), stop the task when no tool_use blocks are found,
                treating the last response as the final answer. If False, continue
                to next iteration.
            tokenizer_model: Model name or path for tokenizer. Defaults to DEFAULT_TOKENIZER_MODEL
                environment variable, or "bert-base-uncased" if not set.
            enable_thinking: Enable thinking/reasoning for models that support it (default: True).
            **kwargs: Additional keyword arguments.
        """
        # Store BCP-specific parameters
        self.searcher_type = searcher_type
        self.index_path = index_path
        self.top_k = top_k
        self.snippet_max_tokens = snippet_max_tokens
        self.doc_max_tokens = doc_max_tokens
        self.include_get_document = include_get_document
        self.browsecomp_path = browsecomp_path or os.environ.get("BROWSECOMP_PATH")
        self.searcher_model_name = searcher_model_name
        self.tokenizer_model = tokenizer_model or os.environ.get("DEFAULT_TOKENIZER_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
        self.tokenizer = tokenizer  # Pre-loaded tokenizer (skip loading in _init_searcher if set)
        self.enable_thinking = enable_thinking
        # Provider-specific thinking parameters (effort); see
        # asio.utils.retry.get_anthropic_thinking_kwargs.
        self.thinking_config = thinking_config or {}
        self.agent_temperature = agent_temperature

        # Reward related
        self.gold_docs = gold_docs
        self.key_docs = key_docs
        self.task_id = task_id
        self.calculate_reward = calculate_reward
        self.intermediate_rewards = []
        # Track tool-call signatures from the previous round. A round = one
        # assistant turn, which may emit multiple tool_use blocks. We rotate
        # `current -> last` at the start of each iteration so cross-round
        # duplicates are detected even when the duplicated call is not the
        # last one of the previous round.
        self.last_tool_signatures: set = set()
        self.current_tool_signatures: set = set()
        self.last_compression_text = None
        self.last_compression_step = None
        self.current_iteration = 0
        self.compression_step = 0
        self.compression_attempt_types = []
        self.context_manager_terminal = False
        self.context_manager_end_reason = None
        self.recent_compression_steps = []
        self.compression_metrics_by_step = {}
        self.compression_repetition_stats_by_step = {}
        self.stop_on_no_tool_use = stop_on_no_tool_use

        # Cost guard: per-rollout token / dollar budget for the frozen agent.
        # Prices come from the file named in cost_config.pricing_path.
        self.cost_config = cost_config or {}
        self.cost_tracker = build_cost_tracker(
            self.cost_config,
            model_name=(getattr(model, "model_name", "") or ""),
            experiment_logger=experiment_logger,
            run_id=task_id,
            batch_id=self.cost_config.get("batch_id"),
        )
        self.abort_rollout = False
        self.abort_reason = None

        # Reward configuration with defaults
        self.reward_config = reward_config or {}
        self.reward_values = {
            # Compression doc reward (existing): reward for retaining gold/key docs after compression
            "enable_compression_doc_reward": self.reward_config.get("enable_compression_doc_reward", False),
            "gold_positive": self.reward_config.get("gold_positive", 0.5),  # Increased from 0.2
            "gold_negative": self.reward_config.get("gold_negative", -0.4),  # Changed from 0.3
            "key_positive": self.reward_config.get("key_positive", 0.2),  # Increased from 0.1
            "key_negative": self.reward_config.get("key_negative", -0.2),  # Same as before (but now absolute value)
            "too_many_docs_penalty": self.reward_config.get("too_many_docs_penalty", -0.3),  # Reduced from -0.6
            "parsing_failure_penalty": self.reward_config.get("parsing_failure_penalty", -0.5),
            "duplicate_call_penalty": self.reward_config.get("duplicate_call_penalty", -0.2),  # Reduced from -0.3
            "no_change_penalty": self.reward_config.get("no_change_penalty", -0.5),
            "doc_threshold": self.reward_config.get("doc_threshold", 15),
            # Search hit reward (new): reward for agent finding gold/key docs in search results
            # This rewards the PREVIOUS compression step for enabling good search results
            "enable_search_hit_reward": self.reward_config.get("enable_search_hit_reward", True),
            "search_hit_gold_positive": self.reward_config.get("search_hit_gold_positive", 0.3),
            "search_hit_key_positive": self.reward_config.get("search_hit_key_positive", 0.1),
        }
        self.terminal_out_of_tokens_config = get_terminal_out_of_tokens_config(self.reward_config)
        if memory_config is not None and "degeneration" in self.reward_config and "degeneration" not in memory_config:
            memory_config = {**memory_config, "degeneration": self.reward_config["degeneration"]}

        if "bm25" not in self.searcher_type.lower():
            self.mcp_url = os.environ.get("MCP_SERVER_URL", "MCP_SERVER_URL")
        else:
            self.mcp_url = mcp_url
        
        # Store experiment logger
        self.experiment_logger = experiment_logger
        if self.experiment_logger:
            self.experiment_logger.log_debug(f"searcher_name: {self.searcher_model_name}\n mcp_url: {self.mcp_url}")
        
        # Initialize searcher (will be done lazily)
        self.searcher = None
        self.mcp_client = None
        self._mcp_cm = None
        self._mcp_lock = None
        self.tokenizer = None
        
        # Ensure experiment_logger is passed to parent class
        kwargs['experiment_logger'] = experiment_logger
        
        # Handle custom memory class if specified
        if memory_class and memory is None:
            memory = self._create_custom_memory(memory_class, memory_config)
        
        # Initialize toolkit if not provided
        if toolkit is None:
            toolkit = Toolkit()
        
        # Initialize parent class
        super().__init__(
            name=name,
            sys_prompt=BCP_SYSTEM_PROMPT,
            model=model,
            formatter=formatter,
            toolkit=toolkit,
            memory=memory,
            max_iters=max_iters,
            **kwargs,
        )
        # Register BCP tools
        self._register_tools(toolkit)
        
        # System message for formatting
        self.system_msg = Msg(
            name="system",
            content=[{"type": "text", "text": BCP_SYSTEM_PROMPT}],
            role="system"
        )
        self.previous_found_key_docs = set()
        self.previous_found_gold_docs = set()
        # Track docs found in search results (for search_hit_reward, to avoid duplicate rewards)
        self.previous_search_hit_gold_docs = set()
        self.previous_search_hit_key_docs = set()

    def _get_tool_result_role(self) -> str:
        """Get the appropriate role for tool result messages based on model provider.

        Claude/Anthropic API does not accept role="system" for any message,
        so tool results should use role="user" for Anthropic models.
        """
        if detect_model_provider(self.model) == "anthropic":
            return "user"
        return "system"

    def _calculate_compression_reward(self, text_content=None, step=None):
        if not self.task_id or not self.calculate_reward:
            if self.experiment_logger:
                self.experiment_logger.log_debug(f"Skipping reward calc: task_id={self.task_id}, calculate_reward={self.calculate_reward}")
            return

        # Check if compression doc reward is enabled (default: True for backward compatibility)
        if not self.reward_values.get("enable_compression_doc_reward", True):
            return

        # Direct use of sets passed in __init__
        relevant_golds = self.gold_docs if self.gold_docs else set()
        relevant_keys = self.key_docs if self.key_docs else set()
        
        if not relevant_golds and not relevant_keys:
            if self.experiment_logger:
                self.experiment_logger.log_debug(f"No gold/key docs found for task {self.task_id}")
            return

        # Combine checks
        all_docs = relevant_golds.union(relevant_keys)
        
        # Helper to find docs in history
        def find_docs_in_history(history):
            if not history:
                return set()
            found = set()
            text_content = format_msgs(history, with_id=False)
            
            # Simple regex search for doc IDs
            # Assuming doc IDs are numeric or distinct strings.
            for doc_id in all_docs:
                escaped_id = re.escape(doc_id)
                # Robust pattern: match word boundary OR explicit quotes/escapes
                # This handles cases like "ID", \"ID\", 'ID', etc. where \b might fail
                # or where the user suspects format issues.
                if re.search(r'\b' + escaped_id + r'\b', text_content) or \
                   re.search(r'(?:[\"\'\\])' + escaped_id + r'(?:[\"\'\\])', text_content):
                    found.add(doc_id)
            return found

        docs_before = find_docs_in_history(getattr(self.memory, "last_snapshot", None))
        docs_after = find_docs_in_history(self.memory._chat_history)
        
        if len(docs_after) > self.reward_values["doc_threshold"]:
             excess_count = len(docs_after) - self.reward_values["doc_threshold"]
             reward_entry = {
                "text_content": text_content,
                "reward": self.reward_values["too_many_docs_penalty"] * excess_count,
                "reason": "too_many_docs",
                "count": len(docs_after),
                "excess_count": excess_count,

                "iteration": self.current_iteration,
                "step": step
            }
             self.intermediate_rewards.append(reward_entry)
        # Fixed reward for each new document found (not proportional to recall)
        # Calculate newly found documents
        new_gold_docs = docs_after.intersection(relevant_golds).intersection(docs_before) - self.previous_found_gold_docs
        lost_gold_docs = docs_before.intersection(relevant_golds) - docs_after.intersection(relevant_golds)

        new_key_docs = docs_after.intersection(relevant_keys).intersection(docs_before) - self.previous_found_key_docs
        lost_key_docs = docs_before.intersection(relevant_keys) - docs_after.intersection(relevant_keys)

        # Gold Reward - fixed reward for each new gold doc found
        if new_gold_docs:
            # Give fixed positive reward for finding new gold docs
            reward_val = self.reward_values["gold_positive"]
            reward_entry = {
                "text_content": text_content,
                "reward": reward_val,
                "type": "gold",
                "action": "new_docs_found",
                "new_docs": list(new_gold_docs),
                "count": len(new_gold_docs),

                "iteration": self.current_iteration,
                "step": step
            }
            self.intermediate_rewards.append(reward_entry)
            if self.experiment_logger:
                self.experiment_logger.log_debug(f"Intermediate Reward Recorded: {json.dumps(reward_entry)}")

        if lost_gold_docs:
            # Give fixed negative reward for losing gold docs
            reward_val = self.reward_values["gold_negative"]
            reward_entry = {
                "text_content": text_content,
                "reward": reward_val,
                "type": "gold",
                "action": "docs_lost",
                "lost_docs": list(lost_gold_docs),
                "count": len(lost_gold_docs),

                "iteration": self.current_iteration,
                "step": step
            }
            self.intermediate_rewards.append(reward_entry)
            if self.experiment_logger:
                self.experiment_logger.log_debug(f"Intermediate Reward Recorded: {json.dumps(reward_entry)}")

        # Key Reward - fixed reward for each new key doc found
        if new_key_docs:
            # Give fixed positive reward for finding new key docs
            reward_val = self.reward_values["key_positive"]
            reward_entry = {
                "text_content": text_content,
                "reward": reward_val,
                "type": "key",
                "action": "new_docs_found",
                "new_docs": list(new_key_docs),
                "count": len(new_key_docs),

                "iteration": self.current_iteration,
                "step": step
            }
            self.intermediate_rewards.append(reward_entry)
            if self.experiment_logger:
                self.experiment_logger.log_debug(f"Intermediate Reward Recorded: {json.dumps(reward_entry)}")

        if lost_key_docs:
            # Give fixed negative reward for losing key docs
            reward_val = self.reward_values["key_negative"]
            reward_entry = {
                "text_content": text_content,
                "reward": reward_val,
                "type": "key",
                "action": "docs_lost",
                "lost_docs": list(lost_key_docs),
                "count": len(lost_key_docs),

                "iteration": self.current_iteration,
                "step": step
            }
            self.intermediate_rewards.append(reward_entry)
            if self.experiment_logger:
                self.experiment_logger.log_debug(f"Intermediate Reward Recorded: {json.dumps(reward_entry)}")
        self.previous_found_key_docs = docs_after.intersection(relevant_keys)
        self.previous_found_gold_docs = docs_after.intersection(relevant_golds)

    def _calculate_search_hit_reward(self, msg: Msg) -> None:
        """Calculate reward for agent finding gold/key docs in search results.

        This rewards the PREVIOUS compression step for enabling good search results.
        The logic is: good compression -> agent retains correct context -> generates good query -> finds relevant docs.

        Args:
            msg: The tool_result message containing search results (before compression).
        """
        if not self.task_id or not self.calculate_reward:
            return

        if not self.reward_values.get("enable_search_hit_reward", False):
            return

        # Skip if no previous compression step (first search)
        if self.last_compression_step is None:
            if self.experiment_logger:
                self.experiment_logger.log_debug(
                    "Skipping search_hit_reward: no previous compression step (first search)"
                )
            return

        relevant_golds = self.gold_docs if self.gold_docs else set()
        relevant_keys = self.key_docs if self.key_docs else set()

        if not relevant_golds and not relevant_keys:
            return

        # Extract docids from search result message
        found_docids = self._extract_docids_from_search_result(msg)

        if not found_docids:
            return

        # Find new gold/key docs that haven't been rewarded yet
        found_golds = found_docids.intersection(relevant_golds)
        found_keys = found_docids.intersection(relevant_keys)

        new_gold_hits = found_golds - self.previous_search_hit_gold_docs
        new_key_hits = found_keys - self.previous_search_hit_key_docs

        # Reward for finding new gold docs in search results
        if new_gold_hits:
            reward_val = self.reward_values["search_hit_gold_positive"]
            reward_entry = {
                "reward": reward_val,
                "type": "gold",
                "action": "search_hit",
                "reason": "search_hit_reward",
                "new_docs": list(new_gold_hits),
                "count": len(new_gold_hits),

                "iteration": self.current_iteration,
                "step": self.last_compression_step,  # Attribute to previous compression
            }
            self.intermediate_rewards.append(reward_entry)
            if self.experiment_logger:
                self.experiment_logger.log_debug(
                    f"Search Hit Reward Recorded (gold): {json.dumps(reward_entry)}"
                )

        # Reward for finding new key docs in search results
        if new_key_hits:
            reward_val = self.reward_values["search_hit_key_positive"]
            reward_entry = {
                "reward": reward_val,
                "type": "key",
                "action": "search_hit",
                "reason": "search_hit_reward",
                "new_docs": list(new_key_hits),
                "count": len(new_key_hits),

                "iteration": self.current_iteration,
                "step": self.last_compression_step,  # Attribute to previous compression
            }
            self.intermediate_rewards.append(reward_entry)
            if self.experiment_logger:
                self.experiment_logger.log_debug(
                    f"Search Hit Reward Recorded (key): {json.dumps(reward_entry)}"
                )

        # Update tracking sets
        self.previous_search_hit_gold_docs.update(new_gold_hits)
        self.previous_search_hit_key_docs.update(new_key_hits)

    def _extract_docids_from_search_result(self, msg: Msg) -> set:
        """Extract docids from a search result message.

        Search results are in JSON format: [{"docid": "xxx", "snippet": "..."}, ...]

        Args:
            msg: The tool_result message.

        Returns:
            Set of docids found in the search result.
        """
        docids = set()

        if not isinstance(msg.content, list):
            return docids

        for content_item in msg.content:
            if not isinstance(content_item, dict):
                continue

            # Check if this is a tool_result
            if content_item.get("type") != "tool_result":
                continue

            output = content_item.get("output", "")
            if not output:
                # Try getting content directly
                output = content_item.get("content", "")

            # Handle different output formats
            if isinstance(output, list):
                # Output is a list of content blocks
                for block in output:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "")
                        docids.update(self._parse_docids_from_json(text))
            elif isinstance(output, str):
                docids.update(self._parse_docids_from_json(output))

        return docids

    def _parse_docids_from_json(self, text: str) -> set:
        """Parse docids from JSON text.

        Expected format: [{"docid": "xxx", ...}, ...]

        Args:
            text: JSON string containing search results.

        Returns:
            Set of docids found.
        """
        docids = set()

        try:
            data = json.loads(text)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "docid" in item:
                        docids.add(str(item["docid"]))
        except (json.JSONDecodeError, TypeError):
            # Not valid JSON, try regex fallback
            # Pattern: "docid": "xxx" or "docid":"xxx"
            pattern = r'"docid"\s*:\s*"([^"]+)"'
            matches = re.findall(pattern, text)
            docids.update(matches)

        return docids

    def _create_custom_memory(
        self,
        memory_class: str,
        memory_config: Optional[dict],
    ) -> MemoryBase:
        """Create a custom memory instance based on the given class name.
        
        Args:
            memory_class: The name of the memory class to instantiate.
            memory_config: Configuration for the memory class.
            
        Returns:
            An instance of the specified memory class.
        """
        # Handle InMemory class from agentscope
        if memory_class.lower() == "inmemory":
            from agentscope.memory import InMemoryMemory
            return InMemoryMemory(debug_dir=memory_config.get("debug_dir", None) if memory_config else None)
        
        # Handle MemoryManager from asio
        if memory_class.lower() == "memorymanager":
            try:
                # Import MemoryManager
                memory_module = importlib.import_module("asio.memory.memorymanager")
                MemoryManager = getattr(memory_module, "MemoryManager")
                
                if memory_config and memory_config.get("use_bg_info"):
                    memory_config["background_info"] = resolve_background_info(
                        memory_config, default_key="bcp"
                    )
                if memory_config and memory_config.get("use_trinity_model") and "_trinity_model" in memory_config:
                    # Use Trinity model for training
                    trinity_model = memory_config["_trinity_model"]
                    trinity_formatter = memory_config["_trinity_formatter"]
                    return MemoryManager(
                        name="MemoryManager",
                        model=trinity_model,
                        formatter=trinity_formatter,
                        inner_prompt=memory_config.get("sys_prompt", ""),
                        config=memory_config,
                        experiment_logger=self.experiment_logger,
                        debug_dir=memory_config.get("debug_dir", None),
                    )
                elif memory_config and "_api_model" in memory_config:
                    # Use API model (standalone mode)
                    api_model = memory_config["_api_model"]
                    api_formatter = memory_config["_api_formatter"]
                    
                    return MemoryManager(
                        name="MemoryManager",
                        model=api_model,
                        formatter=api_formatter,
                        inner_prompt=memory_config.get("sys_prompt", ""),
                        config=memory_config,
                        experiment_logger=self.experiment_logger,
                        debug_dir=memory_config.get("debug_dir", None)
                    )
                else:
                    raise ValueError(
                        "MemoryManager requires either Trinity model or API model configuration"
                    )
            except ImportError:
                if self.experiment_logger:
                    self.experiment_logger.log_warning(
                        "MemoryManager not available, using default memory"
                    )
                return None
        
        # Default: return None and let parent class handle it
        return None
    
    async def _init_searcher(self):
        """Initialize the searcher lazily."""
        if self.searcher is not None:
            return
        if self.tokenizer is None:
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_model, trust_remote_code=True, use_fast=False)
                if self.experiment_logger:
                    self.experiment_logger.log_debug(f"Initialized tokenizer from {self.tokenizer_model}")
            except Exception as e:
                raise e
        if self.mcp_url:
            return

        try:
            # Add BrowseComp-Plus path to system path if needed
            if self.browsecomp_path and str(self.browsecomp_path) not in sys.path:
                sys.path.insert(0, str(self.browsecomp_path))
            # Import searcher from BrowseComp-Plus
            from searcher.searchers import SearcherType
            
            # Get searcher class
            searcher_class = SearcherType.get_searcher_class(self.searcher_type)
            
            # Create a minimal args object for searcher
            class SearcherArgs:
                def __init__(self, index_path):
                    self.index_path = index_path
                    self.hf_token = None
                    self.hf_home = None
                    self.model_name = None
            
            args = SearcherArgs(self.index_path)
            if self.searcher_model_name:
                args.model_name = self.searcher_model_name
            self.searcher = searcher_class(args)
            
            if self.experiment_logger:
                self.experiment_logger.log_debug(
                    f"Initialized {self.searcher_type} searcher from {self.index_path}"
                )
        except Exception as e:
            if self.experiment_logger:
                self.experiment_logger.log_error(f"Failed to initialize searcher: {e}")
            raise RuntimeError(
                f"Failed to initialize searcher. Make sure BrowseComp-Plus is in your Python path "
                f"(set BROWSECOMP_PATH env variable) and index exists at {self.index_path}. Traceback: {traceback.format_exc()}"
            ) from e
    
    def _register_tools(self, toolkit: Toolkit):
        """Register BCP-specific tools with the toolkit."""
        toolkit.register_tool_function(self.search)
        if self.include_get_document:
            toolkit.register_tool_function(self.get_document)
        # finish is registered in the parent class
    
    async def search(self, query: str, top_k: Optional[int] = None) -> ToolResponse:
        """Search for documents using the configured searcher. It will return related documents from database.
        
        Args:
            query: The search query string.
            top_k: Number of top results to return (uses self.top_k if not specified).
            
        Returns:
            ToolResponse with search results.
        """
        # Initialize searcher if needed (mostly for local)
        await self._init_searcher()
        
        if top_k is None:
            top_k = self.top_k
        else:
            try:
                top_k = int(top_k)
            except (ValueError, TypeError):
                top_k = self.top_k
        
        try:
            candidates = []
            if self.mcp_url and self.mcp_client:
                # Perform search via MCP (with retry + rebuild on stream-closed)
                res = await self._call_mcp_tool_with_retry("search", {"query": query})
                # Result structure from BCP MCP server is List[Dict]
                candidates = json.loads(res.content[0].text)
                
            elif self.searcher:
                # Perform search locally
                candidates = self.searcher.search(query, top_k)
            else:
                return ToolResponse(
                    content=[TextBlock(type="text", text="Searcher not initialized and MCP client not available.")],
                    metadata={"error": True}
                )
            # Truncate snippets if needed
            if self.snippet_max_tokens and self.snippet_max_tokens > 0 and self.tokenizer:
                for cand in candidates:
                    if "snippet" not in cand:
                        cand["snippet"] = cand.get("text", "")
                    text = cand["snippet"]
                    if text is None: text = ""
                    tokens = self.tokenizer.encode(text, add_special_tokens=False)
                    if len(tokens) > self.snippet_max_tokens:
                        truncated_tokens = tokens[:self.snippet_max_tokens]
                        cand["snippet"] = self.tokenizer.decode(
                            truncated_tokens, skip_special_tokens=True
                        )
                    else:
                        cand["snippet"] = text
            else:
                for cand in candidates:
                    if "snippet" not in cand:
                        cand["snippet"] = cand.get("text", "")
            
            # Format results as JSON list
            results = []
            for cand in candidates:
                if len(results) >= top_k:
                    break
                    
                item = {
                    "docid": cand["docid"],
                    "snippet": cand["snippet"]
                }
                if cand.get("score") is not None:
                    item["score"] = cand["score"]
                results.append(item)
            
            response_text = json.dumps(results, indent=2)
            
            return ToolResponse(
                content=[TextBlock(type="text", text=response_text)],
                metadata={"num_results": len(results)}
            )
            
        except Exception as e:
            tb = traceback.format_exc()
            error_msg = f"Error during search: {str(e)}, traceback: {tb}"
            if self.experiment_logger:
                self.experiment_logger.log_error(error_msg)
            return ToolResponse(
                content=[TextBlock(type="text", text=error_msg)],
                metadata={"error": True}
            )
    
    async def get_document(self, docid: str, doc_max_tokens: Optional[int] = None) -> ToolResponse:
        """Retrieve a specific document by its ID.
        
        Args:
            docid: The document ID to retrieve.
            doc_max_tokens: Maximum tokens for document content (uses self.doc_max_tokens if not specified).
            
        Returns:
            ToolResponse with the document content.
        """
        # Initialize searcher if needed
        await self._init_searcher()
        
        if doc_max_tokens is None:
            doc_max_tokens = self.doc_max_tokens
        else:
            try:
                doc_max_tokens = int(doc_max_tokens)
            except (ValueError, TypeError):
                doc_max_tokens = self.doc_max_tokens
        
        try:
            result = None
            if self.mcp_url and self.mcp_client:
                result = await self._call_mcp_tool_with_retry("get_document", {"docid": docid})
                result = result.content[0].text if result.content else f"No content found for docid '{docid}'."
            elif self.searcher:
                result = self.searcher.get_document(docid)
                if isinstance(result, dict):
                    result = result['text']
            else:
                return ToolResponse(
                    content=[TextBlock(type="text", text="Searcher not initialized and MCP client not available.")],
                    metadata={"error": True}
                )
            
            if result is None:
                return ToolResponse(
                    content=[TextBlock(type="text", text=json.dumps({"error": f"Document with docid '{docid}' not found"}))],
                    metadata={"error": True}
                )
            # Truncate document if needed
            if doc_max_tokens and doc_max_tokens > 0 and self.tokenizer:
                tokens = self.tokenizer.encode(result, add_special_tokens=False)
                if len(tokens) > doc_max_tokens:
                    truncated_tokens = tokens[:doc_max_tokens]
                    result = self.tokenizer.decode(
                        truncated_tokens, skip_special_tokens=True
                    )
            # Format document content as JSON
            response_text = json.dumps(result, indent=2)
            
            return ToolResponse(
                content=[TextBlock(type="text", text=response_text)],
                metadata={"docid": docid}
            )
            
        except Exception as e:
            tb = traceback.format_exc()
            error_msg = f"Error retrieving document: {str(e)}, traceback: {tb}"
            if self.experiment_logger:
                self.experiment_logger.log_error(error_msg)
            return ToolResponse(
                content=[TextBlock(type="text", text=error_msg)],
                metadata={"error": True}
            )
    
    def finish(self, answer: str, explanation: str, confidence: str = None) -> ToolResponse:
        """Return the final result when you have a definitive answer or cannot progress further. Provide a concise answer plus a brief, evidence-grounded explanation.

        Args:
            answer: A succinct, final answer.
            explanation: A brief explanation for your final answer. For this section only, cite evidence documents inline by placing their docids in square brackets at the end of sentences (e.g., [20]). Do not include citations anywhere else.
            confidence: Confidence: your confidence score between 0% and 100% for your answer
        """
        return ToolResponse(
            content=[TextBlock(type="text", text="Task Finished")],
            metadata={"answer": answer, "explanation": explanation, "confidence": confidence}
        )

    async def _run_task_internal(self, question: str, ground_truth: Optional[str] = None, judge_model=None) -> Dict[str, Any]:
        """Internal logic for running search task."""
        try:
            # Clear memory
            await self.memory.clear()
            self.previous_found_key_docs = set()
            self.previous_found_gold_docs = set()
            self.last_tool_signatures = set()
            self.current_tool_signatures = set()
            self.last_compression_text = None
            self.last_compression_step = None
            self.current_iteration = 0
            self.compression_step = 0
            self.compression_attempt_types = []
            self.context_manager_terminal = False
            self.context_manager_end_reason = None
            self.abort_rollout = False
            self.abort_reason = None
            self.recent_compression_steps = []
            self.compression_metrics_by_step = {}
            self.compression_repetition_stats_by_step = {}
            
            # Create initial user message
            user_msg = Msg(
                name="user",
                content=[{"type": "text", "text": f"Question: {question}"}],
                role="user"
            )
            
            # Add to memory
            await self.memory.add(user_msg)
            
            # Track tool usage
            tool_calls = {
                "search": 0,
                "get_document": 0
            }
            final_answer = None
            final_explanation = None
            final_confidence = None
            
            # Run reasoning-acting loop
            for iteration in range(self.max_iters):
                self.current_iteration = iteration
                # Rotate tool-signature tracking: tools issued in the previous
                # round become the baseline for duplicate detection this round.
                self.last_tool_signatures = self.current_tool_signatures
                self.current_tool_signatures = set()
                # Get agent reasoning
                response = await self._reasoning()
                if self.context_manager_terminal or self.abort_rollout:
                    break
                
                # Extract tool calls
                tool_use_blocks = response.get_content_blocks("tool_use")
                
                if not tool_use_blocks:
                    if self.stop_on_no_tool_use:
                        # Treat as finish: extract text from last response as answer
                        if hasattr(response, 'content'):
                            if isinstance(response.content, str):
                                final_answer = response.content
                            elif isinstance(response.content, list):
                                final_answer = ""
                                for block in response.content:
                                    if isinstance(block, dict) and block.get("type") == "text":
                                        final_answer += block.get("text", "")
                                        break
                            else: 
                                final_answer = str(response.content)
                            break
                    continue
                
                # Execute tool calls
                finish_called = False
                for tool_call in tool_use_blocks:
                    tool_name = tool_call.get("name")
                    tool_input = tool_call.get("input", {})
                    
                    if tool_name == "finish":
                        final_answer = tool_input.get("answer", "")
                        final_explanation = tool_input.get("explanation", "")
                        final_confidence = tool_input.get("confidence")
                        finish_called = True
                        break

                    # Track tool usage
                    if tool_name == "search":
                        tool_calls["search"] += 1
                    elif tool_name == "get_document":
                        tool_calls["get_document"] += 1
                    
                    # Execute tool
                    await self._acting(tool_call)
                    if self.context_manager_terminal or self.abort_rollout:
                        break

                if self.context_manager_terminal or self.abort_rollout:
                    break
                
                if finish_called:
                    break
            
            # If no final answer after max iterations, summarize
            if final_answer is None and not self.context_manager_terminal and not self.abort_rollout:
                summary_response = await self._summarizing()
                if hasattr(summary_response, 'content'):
                    if isinstance(summary_response.content, str):
                        final_answer = summary_response.content
                    elif isinstance(summary_response.content, list):
                        for block in summary_response.content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                final_answer = block.get("text", "")
                                break
            
            # Prepare results
            results = {
                "question": question,
                "answer": final_answer or "No answer generated",
                "explanation": final_explanation or "",
                "confidence": final_confidence,
                "ground_truth": ground_truth,
                "tool_calls": tool_calls,
                "total_iterations": iteration + 1,
                "success": (final_answer is not None
                            and not self.context_manager_terminal
                            and not self.abort_rollout),
                "end_reason": self.context_manager_end_reason or self.abort_reason,
                "intermediate_rewards": self.intermediate_rewards,
                "compression_attempt_types": self.compression_attempt_types,
                "compression_metrics_by_step": self.compression_metrics_by_step,
                "compression_repetition_stats_by_step": self.compression_repetition_stats_by_step,
            }
            
            # Per-rollout spend: persisted and folded into the batch total.
            if self.cost_tracker is not None:
                results["cost"] = self.cost_tracker.finalize()

            # Log results
            if self.experiment_logger:
                self.experiment_logger.log_debug(
                    f"Task completed in {results['total_iterations']} iterations with "
                    f"{tool_calls['search']} searches and {tool_calls['get_document']} document retrievals"
                )
            
            return results
            
        except Exception as e:
            tb = traceback.format_exc()
            error_msg = f"Error during search task: {e}\nTraceback: {tb}"
            
            if self.experiment_logger:
                self.experiment_logger.log_error(error_msg)
            
            return {
                "question": question,
                "answer": None,
                "ground_truth": ground_truth,
                "error": str(e),
                "success": False,
                "traceback": tb,
            }

    def _record_agent_cost(self, response) -> None:
        """Count one agent call against the run's token / dollar budget.

        Hitting a cap aborts the rollout rather than raising: the partial
        trajectory is still worth logging, and it is marked so the workflow can
        tell it apart from a completed one.
        """
        if self.cost_tracker is None:
            return
        self.cost_tracker.record_usage(getattr(response, "usage", None))
        if self.cost_tracker.over_budget() and not self.abort_rollout:
            self.abort_rollout = True
            self.abort_reason = "cost_budget_exceeded"
            if self.experiment_logger:
                self.experiment_logger.log_warning(
                    f"COST_GUARD: budget exhausted, aborting rollout: "
                    f"{json.dumps(self.cost_tracker.snapshot())}"
                )

    def _record_lock_violation_penalty(self):
        """Turn dropped lock violations into a format penalty on this step.

        The context lock drops ops that target the in-flight tool-use cycle
        before they are applied; with `memory_config.lock_violation_penalty`
        set, the manager also gets an RL signal for having produced them.
        Default (None) is a silent drop — the right setting for SFT / warm-up.

        Consumed here rather than in the attempt loop below because a round can
        raise after the lock ran (degeneration, invalid context), leaving no
        `last_results`; the violations would otherwise be charged to a later step.
        """
        violations = getattr(self.memory, "lock_violations_last", None) or []
        if not violations:
            return
        try:
            self.memory.lock_violations_last = []  # one penalty per round
        except AttributeError:
            pass
        if not self.calculate_reward:
            return
        reward_entry = build_lock_violation_reward(
            violations,
            getattr(self.memory, "lock_violation_penalty", None),
            iteration=self.current_iteration,
            step=self.compression_step,
        )
        if reward_entry is None:
            return
        self.intermediate_rewards.append(reward_entry)
        if self.experiment_logger:
            self.experiment_logger.log_debug(
                f"Intermediate Reward Recorded (lock_violation): {json.dumps(reward_entry)}"
            )

    def _process_memory_result(self, context: str = ""):
        """Process compression result, record penalties for failed attempts and reward for success.

        All retry logic is handled inside MemoryManager. This method processes the
        `attempts` list to record penalties for each failed attempt, and records
        compression reward for successful completion.

        Args:
            context: Context string for logging (e.g., "reasoning", "acting").
        """
        self._record_lock_violation_penalty()
        add_result = getattr(self.memory, "last_results", None)

        if not add_result:
            return

        if add_result.get("status") == "UnrecoverableOutOfTokens":
            self.context_manager_terminal = bool(
                self.terminal_out_of_tokens_config.get("fail_rollout", True)
            )
            self.context_manager_end_reason = "context_manager_out_of_tokens"

        if not self.calculate_reward:
            self.memory.last_results = None
            return

        # Keep only training-relevant attempts:
        # - parsing/modification: single failure, no retry
        # - out_of_tokens: first failure only
        # - success: keep the final success if present
        raw_attempts = add_result.get("attempts", [])
        attempts = []
        first_out_of_tokens = None
        last_success = None
        for attempt in raw_attempts:
            attempt_type = attempt.get("type", "")
            if attempt_type in ("parsing_error", "modification_error", "degenerate_generation"):
                attempts.append(attempt)
            elif attempt_type == "out_of_tokens" and first_out_of_tokens is None:
                first_out_of_tokens = attempt
            elif attempt_type == "success":
                last_success = attempt
        if first_out_of_tokens is not None:
            attempts.append(first_out_of_tokens)
        if last_success is not None:
            attempts.append(last_success)

        is_terminal_oot = add_result.get("status") == "UnrecoverableOutOfTokens"
        no_change_count = add_result.get("no_change_count", 0)
        status = add_result.get("status", "")
        degeneration_rewards = add_result.get("degeneration_rewards", []) or []
        repetition_stats = add_result.get("repetition_stats") or {}

        self.compression_attempt_types.extend(
            attempt.get("type", "") for attempt in attempts if attempt.get("type")
        )

        # if self.experiment_logger:
        #     self.experiment_logger.log_info(
        #         f"_process_memory_result ({context}): {len(attempts)} attempts, "
        #         f"types=[{', '.join(a.get('type', '?') for a in attempts)}]"
        #     )
        for attempt in attempts:

            attempt_type = attempt.get("type", "")

            if attempt_type in ("parsing_error", "modification_error", "degenerate_generation"):
                degeneration_reward = next(
                    (
                        reward for reward in degeneration_rewards
                        if reward.get("reason") == "degenerate_generation"
                    ),
                    {},
                )
                reward_entry = {
                    "text_content": f"{attempt_type}: {attempt.get('error', '')}",
                    "reward": degeneration_reward.get("reward", self.reward_values["parsing_failure_penalty"]),
                    "reason": attempt_type,
                    "iteration": self.current_iteration,
                    "step": self.compression_step
                }
                if degeneration_reward:
                    reward_entry.update({
                        "level": degeneration_reward.get("level"),
                        "compression_ratio": degeneration_reward.get("compression_ratio"),
                        "text_bytes": degeneration_reward.get("text_bytes"),
                        "compressed_bytes": degeneration_reward.get("compressed_bytes"),
                    })
                self.intermediate_rewards.append(reward_entry)
                if attempt_type == "degenerate_generation" and repetition_stats:
                    self.compression_repetition_stats_by_step[self.compression_step] = repetition_stats
                if self.experiment_logger:
                    self.experiment_logger.log_debug(
                        f"Intermediate Reward Recorded ({attempt_type}): {json.dumps(reward_entry)}"
                    )
                self.compression_step += 1

            elif attempt_type == "out_of_tokens":
                error_text = attempt.get("error", "")
                if is_terminal_oot:
                    self.context_manager_terminal = bool(
                        self.terminal_out_of_tokens_config.get("fail_rollout", True)
                    )
                    self.context_manager_end_reason = "context_manager_out_of_tokens"
                    terminal_reward = build_terminal_out_of_tokens_reward(
                        self.reward_config,
                        iteration=self.current_iteration,
                        step=self.last_compression_step,
                        error_text=error_text,
                    )
                    if terminal_reward is not None:
                        self.intermediate_rewards.append(terminal_reward)
                        if self.experiment_logger:
                            self.experiment_logger.log_debug(
                                f"Intermediate Reward Recorded (terminal_out_of_tokens): {json.dumps(terminal_reward)}"
                            )
                    deferred_rewards = build_deferred_out_of_tokens_rewards(
                        self.recent_compression_steps,
                        self.reward_config,
                        iteration=self.current_iteration,
                        error_text=error_text,
                    )
                    self.intermediate_rewards.extend(deferred_rewards)
                    if self.experiment_logger and deferred_rewards:
                        self.experiment_logger.log_debug(
                            f"Intermediate Rewards Recorded (deferred_out_of_tokens): {json.dumps(deferred_rewards)}"
                        )
                else:
                    if self.experiment_logger:
                        self.experiment_logger.log_warning(
                            f"Non-terminal out_of_tokens ignored: {error_text}"
                        )

            elif attempt_type == "success":
                # Process successful compression reward
                text_content = add_result.get("modification_response_text")
                if text_content:
                    self.last_compression_text = text_content
                    self.last_compression_step = self.compression_step
                    self._calculate_compression_reward(text_content, step=self.compression_step)
                    compression_metrics = add_result.get("compression_metrics")
                    if compression_metrics:
                        self.compression_metrics_by_step[self.compression_step] = compression_metrics
                    if repetition_stats:
                        self.compression_repetition_stats_by_step[self.compression_step] = repetition_stats
                    budget_reward = build_insufficient_budget_reward(
                        compression_metrics,
                        self.reward_config,
                        iteration=self.current_iteration,
                        step=self.compression_step,
                        text_content=text_content,
                    )
                    if budget_reward is not None:
                        self.intermediate_rewards.append(budget_reward)
                        if self.experiment_logger:
                            self.experiment_logger.log_debug(
                                f"Intermediate Reward Recorded (insufficient_budget): {json.dumps(budget_reward)}"
                            )
                    budget_penalty = calculate_budget_penalty(compression_metrics, self.reward_config)
                    self.recent_compression_steps.append(
                        build_recent_compression_record(
                            step=self.compression_step,
                            compression_metrics=compression_metrics,
                            budget_penalty=budget_penalty,
                            no_change_count=no_change_count,
                        )
                    )
                    self.compression_step += 1

        # Penalize no-change modifications (new_content == original content for single-id mods)
        # Skip when compression itself failed — no_change_count may leak from the pre-pass
        if no_change_count > 0 and self.last_compression_step is not None and status not in ("ModificationFailed", "ParsingFailed"):
            reward_entry = {
                "text_content": f"no_change: {no_change_count} modification(s) unchanged",
                "reward": self.reward_values["no_change_penalty"] * no_change_count,
                "reason": "no_change",
                "count": no_change_count,

                "iteration": self.current_iteration,
                "step": self.last_compression_step
            }
            self.intermediate_rewards.append(reward_entry)
            if self.experiment_logger:
                self.experiment_logger.log_debug(
                    f"Intermediate Reward Recorded (no_change): {json.dumps(reward_entry)}"
                )

        self.memory.last_results = None

    async def _reasoning(self) -> Msg:
        """Perform reasoning step to decide next action.
        
        Returns:
            Message with reasoning and potential tool calls.
        """
        try:
            # Get messages from memory
            memory_msgs = await self.memory.get_memory()
            
            # Format messages for the model
            formatted_messages = await self.formatter.format(
                msgs=[self.system_msg, *memory_msgs]
            )
            
            # Get tool schemas
            # Note: AnthropicChatModel internally converts OpenAI format to Anthropic format
            tools = self.toolkit.get_json_schemas()

            # Get response from model with retry
            response = await retry_model_call(
                self.model,
                formatted_messages,
                tools=tools,
                max_retries=20,
                experiment_logger=self.experiment_logger,
                **get_model_call_kwargs(self.model, self.enable_thinking, self.agent_temperature,
                                        getattr(self, 'thinking_config', None))
            )
            
            self._record_agent_cost(response)

            # Response shape (tool_call recovery, channel-marker cleanup,
            # thinking→text, tool_use id → UUID) is handled inside
            # retry_model_call via asio.utils.response_shape.
            if hasattr(response, 'content'):
                msg = Msg(self.name, response.content, "assistant")
            else:
                msg = Msg(self.name, str(response), "assistant")

            # Add to memory - all retry logic is handled inside MemoryManager
            # Compression failures are recorded in memory.last_results["attempts"]
            await self.memory.add(msg)

            # Process compression result (penalties for failures, reward for success)
            self._process_memory_result(context="Reasoning")
            return msg

        except Exception as e:
            tb = traceback.format_exc()
            if self.experiment_logger:
                self.experiment_logger.log_error(
                    f"Error in reasoning: {e}\nTraceback: {tb}"
                )
            raise
    
    async def _acting(self, tool_call: dict) -> None:
        """Execute a tool call and add result to memory.
        
        Args:
            tool_call: Dictionary with tool call information.
        """
        try:
            tool_name = tool_call.get("name")
            tool_input = tool_call.get("input", {})
            tool_id = tool_call.get("id", str(uuid.uuid4()))

            # Check for duplicate tool call (against previous round's signatures)
            if self.calculate_reward:
                # Use sorted JSON string as unique signature for input
                input_signature = json.dumps(tool_input, sort_keys=True)
                tool_signature = (tool_name, input_signature)

                if tool_signature in self.last_tool_signatures:
                    if self.last_compression_step is not None:
                        reward_entry = {
                            "text_content": self.last_compression_text,
                            "reward": self.reward_values["duplicate_call_penalty"],
                            "reason": "duplicated_tool_call",
                            "tool_name": tool_name,

                            "iteration": self.current_iteration,
                            "step": self.last_compression_step
                        }
                        self.intermediate_rewards.append(reward_entry)
                        if self.experiment_logger:
                            self.experiment_logger.log_debug(f"Intermediate Reward Recorded (Duplication): {json.dumps(reward_entry)}")

                self.current_tool_signatures.add(tool_signature)

            # Execute tool function
            if tool_name == "search":
                result = await self.search(
                    query=tool_input.get("query", ""),
                    top_k=tool_input.get("top_k")
                )
            elif tool_name == "get_document":
                result = await self.get_document(
                    docid=tool_input.get("docid", ""),
                    doc_max_tokens=tool_input.get("doc_max_tokens")
                )
            else:
                result = ToolResponse(
                    content=[TextBlock(type="text", text=f"Unknown tool: {tool_name}")],
                    metadata={"error": True}
                )
            
            # Create tool result message
            result_text = ""
            if hasattr(result, 'content'):
                for block in result.content:
                    if hasattr(block, 'text'):
                        result_text += block.text
                    elif isinstance(block, dict) and "text" in block:
                        result_text += block["text"]
            
            tool_result_msg = Msg(
                name="system",
                content=[ToolResultBlock(
                    type="tool_result",
                    id=tool_id,
                    name=tool_name,
                    output=result_text
                )],
                role=self._get_tool_result_role()
            )

            # Calculate search hit reward BEFORE compression
            # This rewards the previous compression for enabling good search results
            if tool_name == "search":
                self._calculate_search_hit_reward(tool_result_msg)

            # Add to memory - all retry logic is handled inside MemoryManager
            # Compression failures are recorded in memory.last_results["attempts"]
            await self.memory.add(tool_result_msg)

        except Exception as e:
            tb = traceback.format_exc()
            if self.experiment_logger:
                self.experiment_logger.log_error(
                    f"Error in acting: {e}\nTraceback: {tb}"
                )

            # Add error result to memory
            error_msg = Msg(
                name="system",
                content=[ToolResultBlock(
                    type="tool_result",
                    id=tool_call.get("id", ""),
                    name=tool_call.get("name", "unknown"),
                    output=f"Tool execution failed: {str(e)}"
                )],
                role=self._get_tool_result_role()
            )
            # Add error msg to memory - all retry logic is handled inside MemoryManager
            await self.memory.add(error_msg)

        # Process compression result (penalties for failures, reward for success)
        self._process_memory_result(context="Acting")
    
    async def _summarizing(self) -> Msg:
        try:
            # Add hint message
            hint_msg = Msg(
                name="user",
                content=[{"type": "text", "text": 
                    "You have reached the maximum number of iterations. "
                    "Please provide your final answer based on the information gathered so far."}],
                role="user"
            )
            
            # Get messages from memory
            memory_msgs = await self.memory.get_memory()
            
            # Format messages for the model
            formatted_messages = await self.formatter.format(
                msgs=[self.system_msg, *memory_msgs, hint_msg]
            )
            
            # Get response from model
            response = await retry_model_call(
                self.model,
                formatted_messages,
                max_retries=20,
                experiment_logger=self.experiment_logger,
                **get_model_call_kwargs(self.model, self.enable_thinking, self.agent_temperature,
                                        getattr(self, 'thinking_config', None))
            )
            
            self._record_agent_cost(response)

            # Convert response to Msg
            if hasattr(response, 'content'):
                msg = Msg(self.name, response.content, "assistant")
            else:
                msg = Msg(self.name, str(response), "assistant")
            
            return msg
            
        except Exception as e:
            tb = traceback.format_exc()
            if self.experiment_logger:
                self.experiment_logger.log_error(
                    f"Error in summarizing: {e}\nTraceback: {tb}"
                )
            
            return Msg(
                self.name,
                [{"type": "text", "text": f"Error generating summary: {str(e)}"}],
                "assistant"
            )

    async def _rebuild_mcp_client(self):
        """Close the current (dead) MCP client and open a fresh one.

        Caller must hold self._mcp_lock.
        """
        if self._mcp_cm is not None:
            try:
                await self._mcp_cm.__aexit__(None, None, None)
            except Exception as e:
                if self.experiment_logger:
                    self.experiment_logger.log_warning(
                        f"Ignored error closing dead MCP client: {e}"
                    )
        transport = SSETransport(
            url=self.mcp_url,
            httpx_client_factory=_make_mcp_http_factory(60.0),
        )
        self._mcp_cm = Client(transport)
        self.mcp_client = await self._mcp_cm.__aenter__()

    async def _call_mcp_tool_with_retry(self, tool_name: str, arguments: dict, retries: int = 1):
        """Call an MCP tool; rebuild the client and retry on stream-closed errors."""
        last_err = None
        for attempt in range(retries + 1):
            try:
                return await self.mcp_client.call_tool(tool_name, arguments=arguments)
            except (anyio.ClosedResourceError, anyio.BrokenResourceError) as e:
                last_err = e
                if attempt >= retries:
                    break
                if self.experiment_logger:
                    self.experiment_logger.log_warning(
                        f"MCP {tool_name} hit {type(e).__name__}, rebuilding client "
                        f"(attempt {attempt + 1}/{retries})"
                    )
                async with self._mcp_lock:
                    await self._rebuild_mcp_client()
        raise last_err

    async def run_search_task(self, question: str, ground_truth: Optional[str] = None, judge_model=None) -> Dict[str, Any]:
        """Run a search task to answer a question.

        Args:
            question: The question to answer.
            ground_truth: Optional ground truth answer for evaluation.
            judge_model: Optional judge model for evaluating the answer.

        Returns:
            Dictionary with task results including answer and metrics.
        """
        if self.mcp_url:
            if Client is None or SSETransport is None:
                raise RuntimeError("fastmcp is required when mcp_url is provided. Please install 'fastmcp'.")

            self._mcp_lock = asyncio.Lock()
            transport = SSETransport(
                url=self.mcp_url,
                httpx_client_factory=_make_mcp_http_factory(60.0),
            )
            self._mcp_cm = Client(transport)
            self.mcp_client = await self._mcp_cm.__aenter__()
            try:
                return await self._run_task_internal(question, ground_truth, judge_model)
            finally:
                if self._mcp_cm is not None:
                    try:
                        await self._mcp_cm.__aexit__(None, None, None)
                    except Exception:
                        pass
                    self._mcp_cm = None
                    self.mcp_client = None
        else:
            self.mcp_client = None
            return await self._run_task_internal(question, ground_truth, judge_model)
