# -*- coding: utf-8 -*-
"""MCP Worker agent for multi-server MCP-based tasks.

This worker supports connecting to multiple MCP servers and executing tools
discovered from them. It integrates with MemoryManager for context compression
during RL training.
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
import hashlib
import threading
import time as _time

from agentscope.agent import ReActAgent
from agentscope.model import ChatModelBase
from agentscope.formatter import FormatterBase
from agentscope.tool import Toolkit, ToolResponse
from agentscope.memory import MemoryBase
from agentscope.message import Msg, TextBlock, ToolUseBlock, ToolResultBlock
from asio.logger import ExperimentLogger
from asio.utils.retry import (
    retry_model_call, detect_model_provider, get_thinking_kwargs,
    get_anthropic_sampling_kwargs,
)
from asio.agent.memory_reward_utils import (
    build_deferred_out_of_tokens_rewards,
    build_insufficient_budget_reward,
    build_recent_compression_record,
    build_terminal_out_of_tokens_reward,
    calculate_budget_penalty,
    get_terminal_out_of_tokens_config,
)
from collections import defaultdict

try:
    from fastmcp import Client
    from fastmcp.client.transports import SSETransport
except ImportError:
    Client = None
    SSETransport = None

try:
    import aiohttp
except ImportError:
    aiohttp = None


MCP_SYSTEM_PROMPT = """You are an AI assistant that can use various tools to complete tasks. You have access to tools from multiple MCP servers. Each tool is prefixed with its server name (e.g., "Wikipedia__search_wikipedia").

NOTE:
  - You should always call tools to find the information you need to complete the task, but no more than five tools at a time.
  - When you have completed the task or cannot progress further, call the 'finish' tool to provide your final answer.
"""


Background_info = """
## Task Background

The agent is responsible for executing complex tasks by leveraging tools across multiple MCP servers.

**Information Retention Strategy:**  
Retain information at a level of detail appropriate to the task at hand.  
- If the task requires generating structured output (e.g., reports, summaries), preserve comprehensive details.  
- Otherwise, retain only key findings and critical clues necessary to complete the task.
"""


class ToolResultCache:
    """JSON-file-backed tool result cache shared across workers.

    - Startup: load JSON into memory dict
    - Lookup: pure in-memory dict lookup
    - Write: append to dict, periodic flush to disk
    """

    _instances: Dict[str, "ToolResultCache"] = {}
    _lock = threading.Lock()

    @classmethod
    def get_or_create(cls, cache_path: str, flush_interval: int = 30) -> "ToolResultCache":
        """Get existing cache for path, or create a new one (singleton per path)."""
        with cls._lock:
            real = os.path.realpath(cache_path)
            if real not in cls._instances:
                cls._instances[real] = cls(cache_path, flush_interval)
            return cls._instances[real]

    def __init__(self, cache_path: str, flush_interval: int = 30) -> None:
        self._path = cache_path
        self._flush_interval = flush_interval
        self._data: Dict[str, str] = {}
        self._dirty = False
        self._rw_lock = threading.Lock()
        self._last_flush = _time.monotonic()
        self._load()

    def _load(self) -> None:
        if os.path.exists(self._path):
            try:
                with open(self._path, "r") as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._data = {}

    @staticmethod
    def _make_key(server: str, tool: str, args: dict) -> str:
        raw = json.dumps({"s": server, "t": tool, "a": args}, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, server: str, tool: str, args: dict) -> Optional[str]:
        key = self._make_key(server, tool, args)
        return self._data.get(key)

    def put(self, server: str, tool: str, args: dict, result: str) -> None:
        key = self._make_key(server, tool, args)
        with self._rw_lock:
            self._data[key] = result
            self._dirty = True
        if _time.monotonic() - self._last_flush >= self._flush_interval:
            self.flush()

    def flush(self) -> None:
        with self._rw_lock:
            if not self._dirty:
                return
            tmp = self._path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self._data, f)
            os.replace(tmp, self._path)
            self._dirty = False
            self._last_flush = _time.monotonic()

    def stats(self) -> Dict[str, int]:
        return {"entries": len(self._data)}


def get_model_call_kwargs(model, enable_thinking: bool = True,
                          thinking_config: dict = None) -> dict:
    """Get all model-specific kwargs (temperature + thinking) for agent calls."""
    model_name = (getattr(model, "model_name", "") or "").lower()
    if "gpt-5" in model_name:
        kwargs = {}
    elif "claude" in model_name or "anthropic" in model_name:
        # Claude rejects temperature+top_p together, and takes no sampling
        # params at all with thinking on / on 4.7+.
        kwargs = get_anthropic_sampling_kwargs(model_name, 0, enable_thinking)
    else:
        kwargs = {"temperature": 0, "top_p": 1}
    thinking = get_thinking_kwargs(model, enable_thinking, thinking_config)
    if "extra_body" in kwargs and "extra_body" in thinking:
        kwargs["extra_body"] = {**kwargs["extra_body"], **thinking["extra_body"]}
    else:
        kwargs.update(thinking)
    return kwargs


class MCPWorker(ReActAgent):
    """A worker agent for MCP-based multi-server tasks.

    This worker extends ReActAgent to handle MCP-specific interactions:
    - Multi-server MCP connections
    - Tool discovery and schema conversion
    - Memory management for context compression
    - Intermediate reward tracking for RL training
    """

    finish_function_name: str = "finish"

    def __init__(
        self,
        name: str,
        model: ChatModelBase,
        formatter: FormatterBase,
        server_configs: List[Dict[str, Any]],
        toolkit: Optional[Toolkit] = None,
        memory: Optional[MemoryBase] = None,
        max_iters: int = 50,
        memory_class: Optional[str] = None,
        memory_config: Optional[dict] = None,
        experiment_logger: Optional[ExperimentLogger] = None,
        task_id: Optional[str] = None,
        calculate_reward: bool = False,
        reward_config: Optional[Dict] = None,
        stop_on_no_tool_use: bool = True,
        tokenizer_model: Optional[str] = None,
        tool_cache_path: Optional[str] = None,
        tool_cache_flush_interval: int = 30,
        es_fallback_config: Optional[Dict[str, Any]] = None,
        skip_live_mcp_servers: Optional[List[str]] = None,
        tool_schemas_path: Optional[str] = None,
        enable_thinking: bool = True,
        thinking_config: Optional[dict] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the MCP Worker agent.

        Args:
            name: The name of the agent.
            model: The chat model instance for the agent (task-solving model).
            formatter: The formatter instance to format messages.
            server_configs: List of MCP server configurations. Each config should have:
                - name: Server name (e.g., "Wikipedia")
                - transport: "sse" or "stdio"
                - url: For SSE transport, the server URL
                - command: For stdio transport, the command list to run
                - env: Optional environment variables
                - cwd: Optional working directory
            toolkit: A Toolkit object that contains the tool functions.
            memory: The memory used to store the dialogue history.
            max_iters: Maximum number of iterations for reasoning-acting loops.
            memory_class: The custom memory class name to use.
            memory_config: Configuration for the custom memory class.
            experiment_logger: Logger for tracking experiments.
            task_id: Current task ID for reward calculation.
            calculate_reward: Whether to calculate intermediate rewards for compression.
            reward_config: Configuration for reward values.
            stop_on_no_tool_use: If True (default), stop the task when no tool_use blocks found.
            tokenizer_model: Model name or path for tokenizer.
            **kwargs: Additional keyword arguments.
        """
        # Store MCP-specific parameters
        self.server_configs = server_configs
        self.enable_thinking = enable_thinking
        # Provider-specific thinking parameters (budget_tokens / effort); see
        # asio.utils.retry.get_anthropic_thinking_kwargs.
        self.thinking_config = thinking_config or {}
        self.tokenizer_model = tokenizer_model or os.environ.get(
            "DEFAULT_TOKENIZER_MODEL", "Qwen/Qwen3-4B-Instruct-2507"
        )

        # Reward related
        self.task_id = task_id
        self.calculate_reward = calculate_reward
        self.intermediate_rewards = []
        self.last_tool_signature = None
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

        # Reward configuration with defaults
        self.reward_config = reward_config or {}
        self.reward_values = {
            "tool_success_reward": self.reward_config.get("tool_success_reward", 0.1),
            "tool_failure_penalty": self.reward_config.get("tool_failure_penalty", -0.2),
            "parsing_failure_penalty": self.reward_config.get("parsing_failure_penalty", -0.5),
            "duplicate_call_penalty": self.reward_config.get("duplicate_call_penalty", -0.2),
            "no_change_penalty": self.reward_config.get("no_change_penalty", -0.5),
        }
        self.terminal_out_of_tokens_config = get_terminal_out_of_tokens_config(self.reward_config)
        if memory_config is not None and "degeneration" in self.reward_config and "degeneration" not in memory_config:
            memory_config = {**memory_config, "degeneration": self.reward_config["degeneration"]}

        # Store experiment logger
        self.experiment_logger = experiment_logger

        # Tool result cache
        self._tool_cache: Optional[ToolResultCache] = None
        if tool_cache_path:
            self._tool_cache = ToolResultCache.get_or_create(
                tool_cache_path, tool_cache_flush_interval,
            )

        # ES fallback for Wikipedia tools (Layer 2 after cache)
        self._es_fallback = None
        if es_fallback_config and es_fallback_config.get("enabled"):
            try:
                from .wikipedia_es_fallback import WikipediaESFallback
                self._es_fallback = WikipediaESFallback(
                    es_url=es_fallback_config.get("es_url"),
                    es_index=es_fallback_config.get("es_index"),
                    miss_log_path=es_fallback_config.get("miss_log_path"),
                )
                if self.experiment_logger:
                    self.experiment_logger.log_info(
                        f"ES fallback enabled: {es_fallback_config.get('es_url', 'default')}"
                    )
            except Exception as e:
                if self.experiment_logger:
                    self.experiment_logger.log_warning(f"ES fallback init failed: {e}")

        # Servers to skip live MCP calls for (use cache + ES only)
        self._skip_live_mcp_servers: set = set(skip_live_mcp_servers or [])

        # Offline tool schemas (pre-dumped JSON, avoids connecting to MCP servers at init)
        self._tool_schemas_path = tool_schemas_path

        # MCP-related state
        self.mcp_clients: Dict[str, Any] = {}
        self.mcp_tools: Dict[str, Dict[str, Any]] = {}  # tool_name -> {server, schema, original_name}
        self.tokenizer = None

        # Ensure experiment_logger is passed to parent class
        kwargs["experiment_logger"] = experiment_logger

        # Handle custom memory class if specified
        if memory_class and memory is None:
            memory = self._create_custom_memory(memory_class, memory_config)

        # Initialize toolkit if not provided
        if toolkit is None:
            toolkit = Toolkit()

        # Initialize parent class
        super().__init__(
            name=name,
            sys_prompt=MCP_SYSTEM_PROMPT,
            model=model,
            formatter=formatter,
            toolkit=toolkit,
            memory=memory,
            max_iters=max_iters,
            **kwargs,
        )

        # NOTE: finish tool is auto-registered by ReActAgent.__init__ via finish_function_name

        # System message for formatting
        self.system_msg = Msg(
            name="system",
            content=[{"type": "text", "text": MCP_SYSTEM_PROMPT}],
            role="system",
        )

    def _get_tool_result_role(self) -> str:
        """Get the appropriate role for tool result messages based on model provider."""
        if detect_model_provider(self.model) == "anthropic":
            return "user"
        return "system"

    def _create_custom_memory(
        self,
        memory_class: str,
        memory_config: Optional[dict],
    ) -> MemoryBase:
        """Create a custom memory instance based on the given class name."""
        if memory_class.lower() == "inmemory":
            from agentscope.memory import InMemoryMemory
            return InMemoryMemory(
                debug_dir=memory_config.get("debug_dir", None) if memory_config else None
            )

        if memory_class.lower() == "memorymanager":
            try:
                memory_module = importlib.import_module("asio.memory.memorymanager")
                MemoryManager = getattr(memory_module, "MemoryManager")

                if memory_config and memory_config.get("use_bg_info"):
                    memory_config["background_info"] = Background_info
                if memory_config and memory_config.get("use_trinity_model") and "_trinity_model" in memory_config:
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
                    api_model = memory_config["_api_model"]
                    api_formatter = memory_config["_api_formatter"]
                    return MemoryManager(
                        name="MemoryManager",
                        model=api_model,
                        formatter=api_formatter,
                        inner_prompt=memory_config.get("sys_prompt", ""),
                        config=memory_config,
                        experiment_logger=self.experiment_logger,
                        debug_dir=memory_config.get("debug_dir", None),
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

        return None

    def _register_finish_tool(self, toolkit: Toolkit):
        """Register the finish tool with the toolkit."""
        toolkit.register_tool_function(self.finish)

    async def _init_tokenizer(self):
        """Initialize the tokenizer lazily."""
        if self.tokenizer is not None:
            return
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_model, trust_remote_code=True
            )
            if self.experiment_logger:
                self.experiment_logger.log_debug(f"Initialized tokenizer from {self.tokenizer_model}")
        except Exception as e:
            if self.experiment_logger:
                self.experiment_logger.log_error(f"Failed to initialize tokenizer: {e}")
            raise

    def _init_offline_server(self, server_name: str, schema_data: dict):
        """Register tools from pre-dumped schema JSON (no MCP connection needed)."""
        tools = schema_data.get("tools", [])
        for tool in tools:
            prefixed_name = self._make_tool_name(server_name, tool["name"])
            tool_obj = type("MCPTool", (), {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "inputSchema": tool.get("inputSchema", {"type": "object", "properties": {}}),
            })()
            self.mcp_tools[prefixed_name] = {
                "server": server_name,
                "original_name": tool["name"],
                "schema": self._convert_mcp_tool_to_openai_schema(tool_obj, server_name),
            }
        # Mark as offline — no live client
        self.mcp_clients[server_name] = {"type": "offline"}
        if self.experiment_logger:
            self.experiment_logger.log_info(
                f"Loaded {len(tools)} tools for {server_name} from offline schema"
            )

    async def _init_mcp_connections(self, max_retries: int = 5, base_delay: float = 3.0):
        """Initialize connections to all configured MCP servers and discover tools.

        Retries each server connection up to max_retries times with exponential backoff.
        """
        # Load offline schemas if available
        offline_schemas = {}
        if self._tool_schemas_path and os.path.exists(self._tool_schemas_path):
            try:
                with open(self._tool_schemas_path, "r") as f:
                    offline_schemas = json.load(f)
                if self.experiment_logger:
                    self.experiment_logger.log_info(
                        f"Loaded offline schemas from {self._tool_schemas_path}: "
                        f"{list(offline_schemas.keys())}"
                    )
            except (json.JSONDecodeError, OSError) as e:
                if self.experiment_logger:
                    self.experiment_logger.log_warning(
                        f"Failed to load offline schemas: {e}"
                    )

        for config in self.server_configs:
            server_name = config["name"]
            transport_type = config.get("transport", "stdio")

            last_error = None
            for attempt in range(max_retries):
                if self.experiment_logger:
                    if attempt == 0:
                        self.experiment_logger.log_info(f"Connecting to MCP server: {server_name} (transport={transport_type})")
                    else:
                        self.experiment_logger.log_info(
                            f"Retrying connection to {server_name} (attempt {attempt + 1}/{max_retries})"
                        )

                try:
                    # Use offline schema if available for this server
                    if server_name in offline_schemas:
                        self._init_offline_server(server_name, offline_schemas[server_name])
                        last_error = None
                        break

                    if transport_type == "remote_http":
                        await self._init_remote_http_server(server_name, config)

                    elif transport_type == "sse":
                        if Client is None or SSETransport is None:
                            raise RuntimeError("fastmcp is required for SSE connections.")
                        url = config.get("url")
                        if not url:
                            raise ValueError(f"SSE transport requires 'url' for server {server_name}")
                        transport = SSETransport(url=url)
                        client = Client(transport)
                        await client.__aenter__()
                        self.mcp_clients[server_name] = client

                        tools_response = await client.session.list_tools()
                        for tool in tools_response.tools:
                            prefixed_name = self._make_tool_name(server_name, tool.name)
                            self.mcp_tools[prefixed_name] = {
                                "server": server_name,
                                "original_name": tool.name,
                                "schema": self._convert_mcp_tool_to_openai_schema(tool, server_name),
                            }

                    else:
                        await self._init_stdio_server(server_name, config)
                        if self.experiment_logger:
                            self.experiment_logger.log_info(
                                f"STDIO server {server_name} configured and tools discovered"
                            )

                    if self.experiment_logger:
                        self.experiment_logger.log_info(
                            f"Connected to {server_name}, discovered {len([t for t in self.mcp_tools if self.mcp_tools[t]['server'] == server_name])} tools"
                        )
                    last_error = None
                    break  # Success

                except Exception as e:
                    last_error = e
                    if self.experiment_logger:
                        self.experiment_logger.log_warning(
                            f"Connection to {server_name} failed (attempt {attempt + 1}/{max_retries}): {type(e).__name__}: {e}"
                        )
                    if attempt < max_retries - 1:
                        delay = min(base_delay * (2 ** attempt), 60)
                        await asyncio.sleep(delay)

            if last_error is not None:
                if self.experiment_logger:
                    self.experiment_logger.log_error(
                        f"Failed to connect to {server_name} after {max_retries} attempts: {type(last_error).__name__}: {last_error}"
                    )
                raise RuntimeError(
                    f"Failed to connect to {server_name} after {max_retries} attempts: {type(last_error).__name__}: {last_error}"
                ) from last_error

    async def _init_remote_http_server(self, server_name: str, config: dict):
        """Initialize a remote HTTP MCP server connection and discover tools.

        Uses JSON-RPC over HTTP to communicate with MCP servers deployed on remote hosts
        (e.g., rack server at 107.174.67.124).
        """
        if aiohttp is None:
            raise RuntimeError("aiohttp is required for remote_http connections. pip install aiohttp")

        host = config.get("host", "localhost")
        port = config.get("port")
        endpoint = config.get("endpoint", "/mcp")
        base_url = f"http://{host}:{port}{endpoint}"

        if self.experiment_logger:
            self.experiment_logger.log_info(f"Connecting to remote HTTP server {server_name} at {base_url}")

        session_id = None

        async def _read_sse_json(resp) -> dict:
            """Read the first JSON data event from an SSE stream incrementally.

            Accumulates chunks and tries to parse data: lines on each chunk.
            Large responses (like tools/list) may span multiple chunks.
            Handles servers that close SSE connections with incomplete chunked encoding.
            """
            content_type = resp.headers.get("content-type", "")
            if "text/event-stream" not in content_type:
                return await resp.json()
            buffer = ""
            try:
                async for chunk in resp.content.iter_any():
                    buffer += chunk.decode("utf-8", errors="replace")
                    for line in buffer.split("\n"):
                        line = line.strip()
                        if line.startswith("data: "):
                            try:
                                return json.loads(line[6:])
                            except json.JSONDecodeError:
                                pass  # JSON incomplete, need more chunks
            except (aiohttp.ClientPayloadError, aiohttp.ClientError):
                pass  # Server closed connection; parse what we have
            # Stream ended or errored - final attempt with accumulated buffer
            for line in buffer.split("\n"):
                line = line.strip()
                if line.startswith("data: "):
                    try:
                        return json.loads(line[6:])
                    except json.JSONDecodeError:
                        pass
            raise RuntimeError(f"SSE stream ended without valid JSON. Buffer length: {len(buffer)}, preview: {buffer[:200]}")

        async with aiohttp.ClientSession() as session:
            # Step 1: Initialize
            init_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "clientInfo": {"name": "trinity-mcp-worker", "version": "1.0.0"},
                },
            }
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            }

            try:
                async with session.post(base_url, json=init_request, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise RuntimeError(f"Initialize failed for {server_name}: HTTP {resp.status}: {error_text}")
                    session_id = resp.headers.get("mcp-session-id")
                    await _read_sse_json(resp)  # Consume the init response
            except (aiohttp.ClientPayloadError, aiohttp.ClientError):
                pass  # SSE stream close may trigger this during context exit

            # Step 2: Send initialized notification
            notif_headers = dict(headers)
            if session_id:
                notif_headers["mcp-session-id"] = session_id
            initialized_notif = {"jsonrpc": "2.0", "method": "notifications/initialized"}
            try:
                async with session.post(base_url, json=initialized_notif, headers=notif_headers, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    pass
            except (asyncio.TimeoutError, aiohttp.ClientPayloadError, aiohttp.ClientError):
                pass  # Notification is fire-and-forget

            # Step 3: Discover tools
            tools_request = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list",
                "params": {},
            }
            tool_headers = dict(headers)
            if session_id:
                tool_headers["mcp-session-id"] = session_id

            result = None
            try:
                async with session.post(base_url, json=tools_request, headers=tool_headers, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise RuntimeError(f"tools/list failed for {server_name}: HTTP {resp.status}: {error_text}")
                    result = await _read_sse_json(resp)
                    if self.experiment_logger:
                        self.experiment_logger.log_info(
                            f"tools/list for {server_name}: result keys={list(result.keys()) if result else 'None'}, "
                            f"tools count={len(result.get('result',{}).get('tools',[])) if result else 0}"
                        )
            except (aiohttp.ClientPayloadError, aiohttp.ClientError) as e:
                if self.experiment_logger:
                    self.experiment_logger.log_warning(
                        f"tools/list ClientPayloadError for {server_name}: result={'set' if result else 'None'}, error={e}"
                    )
                if result is None:
                    raise RuntimeError(f"tools/list failed for {server_name}: {e}") from e

            # Register tools INSIDE the session context to avoid ClientPayloadError on session close
            tools = result.get("result", {}).get("tools", []) if result else []
            for tool in tools:
                prefixed_name = self._make_tool_name(server_name, tool['name'])
                tool_obj = type("MCPTool", (), {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "inputSchema": tool.get("inputSchema", {"type": "object", "properties": {}}),
                })()
                self.mcp_tools[prefixed_name] = {
                    "server": server_name,
                    "original_name": tool["name"],
                    "schema": self._convert_mcp_tool_to_openai_schema(tool_obj, server_name),
                }

            # Store remote_http client info for tool calls
            self.mcp_clients[server_name] = {
                "type": "remote_http",
                "host": config.get("host", "localhost"),
                "port": port,
                "endpoint": endpoint,
                "session_id": session_id,
            }

    async def _init_stdio_server(self, server_name: str, config: dict):
        """Initialize a STDIO MCP server: discover tools at init time.

        Handles the commands file format where 'cmd' is a string and 'cwd' is relative.
        """
        from mcp import ClientSession
        from mcp.client.stdio import stdio_client, StdioServerParameters

        # Build command from config
        cmd = config.get("command") or config.get("cmd", "")
        if isinstance(cmd, str):
            cmd_parts = cmd.split()
        else:
            cmd_parts = list(cmd)

        if not cmd_parts:
            raise ValueError(f"No command specified for STDIO server {server_name}")

        server_params = StdioServerParameters(
            command=cmd_parts[0],
            args=cmd_parts[1:] if len(cmd_parts) > 1 else [],
            env={k: os.environ.get(k, "") for k in config.get("env", [])} if config.get("env") else None,
            cwd=config.get("cwd"),
        )

        if self.experiment_logger:
            self.experiment_logger.log_info(
                f"Starting STDIO server {server_name}: {' '.join(cmd_parts)} (cwd={config.get('cwd')})"
            )

        # Discover tools by connecting briefly
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools_response = await session.list_tools()

                for tool in tools_response.tools:
                    prefixed_name = self._make_tool_name(server_name, tool.name)
                    self.mcp_tools[prefixed_name] = {
                        "server": server_name,
                        "original_name": tool.name,
                        "schema": self._convert_mcp_tool_to_openai_schema(tool, server_name),
                    }

        # Store config for execution
        self.mcp_clients[server_name] = {"config": config, "type": "stdio"}

    @staticmethod
    def _make_tool_name(server_name: str, tool_name: str) -> str:
        """Create a valid OpenAI-compatible tool name from server and tool names.

        OpenAI requires tool names matching ^[a-zA-Z0-9_.-]+$ (no colons or spaces).
        """
        # Replace spaces and special chars in server name
        safe_server = re.sub(r'[^a-zA-Z0-9_.-]', '_', server_name)
        safe_tool = re.sub(r'[^a-zA-Z0-9_.-]', '_', tool_name)
        return f"{safe_server}__{safe_tool}"

    def _convert_mcp_tool_to_openai_schema(self, mcp_tool, server_name: str) -> dict:
        """Convert an MCP tool schema to OpenAI function calling format."""
        prefixed_name = self._make_tool_name(server_name, mcp_tool.name)

        schema = {
            "type": "function",
            "function": {
                "name": prefixed_name,
                "description": mcp_tool.description or f"Tool from {server_name}",
                "parameters": mcp_tool.inputSchema or {"type": "object", "properties": {}},
            },
        }
        return schema

    def _build_tool_schemas(self) -> List[dict]:
        """Build the list of tool schemas for the model."""
        schemas = []

        # Add MCP tools
        for tool_name, tool_info in self.mcp_tools.items():
            schemas.append(tool_info["schema"])

        # Add finish tool
        schemas.append({
            "type": "function",
            "function": {
                "name": "finish",
                "description": "Call this tool when you have completed the task or cannot progress further. Provide your final answer.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "description": "Your final answer to the task.",
                        },
                        "explanation": {
                            "type": "string",
                            "description": "A brief explanation of how you arrived at your answer.",
                        },
                    },
                    "required": ["answer"],
                },
            },
        })

        return schemas

    async def _execute_mcp_tool(self, tool_name: str, tool_input: dict, max_retries: int = 10) -> str:
        """Execute an MCP tool with retry on failure.

        On connection errors, attempts to re-initialize the server session
        (the health loop may have restarted the server with a new session).
        """
        if tool_name not in self.mcp_tools:
            if self.experiment_logger:
                self.experiment_logger.log_error(
                    f"Unknown tool: {tool_name}. Registered tools: {list(self.mcp_tools.keys())[:10]}"
                )
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

        tool_info = self.mcp_tools[tool_name]
        server_name = tool_info["server"]
        original_name = tool_info["original_name"]

        client_info = self.mcp_clients.get(server_name)
        if not client_info:
            return json.dumps({"error": f"No client for server: {server_name}"})

        # Check cache
        if self._tool_cache is not None:
            cached = self._tool_cache.get(server_name, original_name, tool_input)
            if cached is not None:
                if self.experiment_logger:
                    self.experiment_logger.log_debug(
                        f"Cache HIT: {tool_name}({json.dumps(tool_input)[:80]})"
                    )
                return cached

        # ES fallback (Layer 2) for Wikipedia tools
        if self._es_fallback and server_name == "Wikipedia":
            es_result = self._es_fallback.query(original_name, tool_input)
            if es_result is not None:
                if self.experiment_logger:
                    self.experiment_logger.log_info(
                        f"ES fallback HIT: {tool_name}({json.dumps(tool_input)[:80]})"
                    )
                # Write back to cache for future instant hits
                if self._tool_cache is not None:
                    self._tool_cache.put(server_name, original_name, tool_input, es_result)
                return es_result
            else:
                if self.experiment_logger:
                    self.experiment_logger.log_info(
                        f"ES fallback MISS: {tool_name}({json.dumps(tool_input)[:80]})"
                    )

        # Skip live MCP calls for specified servers (training mode: cache + ES only)
        if server_name in self._skip_live_mcp_servers:
            if self.experiment_logger:
                self.experiment_logger.log_warning(
                    f"Skipping live MCP for {tool_name} (skip_live_mcp). Cache+ES miss."
                )
            return json.dumps({"error": f"No cached result available for {tool_name}. Try a different query."})

        last_error = None
        for attempt in range(max_retries):
            try:
                if attempt > 0 and self.experiment_logger:
                    self.experiment_logger.log_info(
                        f"Tool {tool_name} retry attempt {attempt + 1}/{max_retries}"
                    )
                result = await self._execute_mcp_tool_once(client_info, tool_name, original_name, tool_input)
                # Check if result contains an error JSON
                try:
                    parsed = json.loads(result)
                    if isinstance(parsed, dict) and "error" in parsed:
                        last_error = parsed["error"]
                        if self.experiment_logger:
                            self.experiment_logger.log_warning(
                                f"Tool {tool_name} returned error (attempt {attempt + 1}/{max_retries}): {last_error}"
                            )
                        # Re-init session on "Session not found" or HTTP errors
                        if "Session not found" in str(last_error) or "HTTP 4" in str(last_error):
                            if isinstance(client_info, dict) and client_info.get("type") == "remote_http":
                                try:
                                    server_config = next(
                                        (c for c in self.server_configs if c["name"] == server_name), None
                                    )
                                    if server_config:
                                        if self.experiment_logger:
                                            self.experiment_logger.log_info(
                                                f"Re-initializing session for {server_name} after error: {str(last_error)[:60]}"
                                            )
                                        await self._init_remote_http_server(server_name, server_config)
                                        client_info = self.mcp_clients.get(server_name)
                                except Exception as reinit_err:
                                    if self.experiment_logger:
                                        self.experiment_logger.log_warning(
                                            f"Session re-init for {server_name} failed: {reinit_err}"
                                        )
                        await asyncio.sleep(min(3 * (attempt + 1), 15))
                        continue
                except (json.JSONDecodeError, TypeError):
                    pass
                # Cache successful result
                if self._tool_cache is not None:
                    self._tool_cache.put(server_name, original_name, tool_input, result)
                return result

            except Exception as e:
                tb = traceback.format_exc()
                last_error = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
                if self.experiment_logger:
                    self.experiment_logger.log_warning(
                        f"Tool {tool_name} exception (attempt {attempt + 1}/{max_retries}): "
                        f"{type(e).__name__}: {e}\nTraceback: {tb}"
                    )
                if attempt < max_retries - 1:
                    # Try to re-initialize the server session (health loop may have restarted it)
                    if isinstance(client_info, dict) and client_info.get("type") == "remote_http":
                        try:
                            server_config = next(
                                (c for c in self.server_configs if c["name"] == server_name), None
                            )
                            if server_config:
                                if self.experiment_logger:
                                    self.experiment_logger.log_info(
                                        f"Re-initializing session for {server_name} after tool failure..."
                                    )
                                await self._init_remote_http_server(server_name, server_config)
                                client_info = self.mcp_clients.get(server_name)
                        except Exception as reinit_err:
                            if self.experiment_logger:
                                self.experiment_logger.log_warning(
                                    f"Session re-init for {server_name} failed: {type(reinit_err).__name__}: {reinit_err}"
                                )
                    await asyncio.sleep(min(3 * (attempt + 1), 15))

        if self.experiment_logger:
            self.experiment_logger.log_error(
                f"Tool {tool_name} failed after {max_retries} retries. Last error: {last_error}"
            )
        return json.dumps({"error": f"Failed after {max_retries} retries: {last_error}"})

    async def _execute_mcp_tool_once(self, client_info, tool_name: str, original_name: str, tool_input: dict) -> str:
        """Execute an MCP tool call once (no retry)."""
        if isinstance(client_info, dict) and client_info.get("type") == "offline":
            return json.dumps({"error": f"No cached result available for {tool_name}. Try a different query."})
        if isinstance(client_info, dict) and client_info.get("type") == "remote_http":
            return await self._execute_remote_http_tool(client_info, original_name, tool_input)
        elif isinstance(client_info, dict) and client_info.get("type") == "stdio":
            from mcp import ClientSession
            from mcp.client.stdio import stdio_client, StdioServerParameters

            config = client_info["config"]
            # Handle both "command" (list) and "cmd" (string) formats
            cmd = config.get("command") or config.get("cmd", "")
            if isinstance(cmd, str):
                cmd_parts = cmd.split()
            else:
                cmd_parts = list(cmd)

            env_keys = config.get("env", [])
            env_dict = {k: os.environ.get(k, "") for k in env_keys} if isinstance(env_keys, list) and env_keys else (env_keys if isinstance(env_keys, dict) else None)

            server_params = StdioServerParameters(
                command=cmd_parts[0],
                args=cmd_parts[1:] if len(cmd_parts) > 1 else [],
                env=env_dict,
                cwd=config.get("cwd"),
            )

            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(original_name, tool_input)
                    return self._format_mcp_result(result)
        else:
            result = await client_info.call_tool(original_name, arguments=tool_input)
            return self._format_mcp_result(result)

    async def _execute_remote_http_tool(self, client_info: dict, tool_name: str, tool_input: dict) -> str:
        """Execute a tool on a remote HTTP MCP server via JSON-RPC."""
        host = client_info["host"]
        port = client_info["port"]
        endpoint = client_info["endpoint"]
        session_id = client_info.get("session_id")
        base_url = f"http://{host}:{port}{endpoint}"

        tool_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": tool_input,
            },
        }

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if session_id:
            headers["mcp-session-id"] = session_id

        result = None
        buffer = ""
        try:
          async with aiohttp.ClientSession() as session:
            async with session.post(
                base_url,
                json=tool_request,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    return json.dumps({"error": f"HTTP {resp.status}: {error_text}"})

                content_type = resp.headers.get("content-type", "")
                if "text/event-stream" in content_type:
                    try:
                        async for chunk in resp.content.iter_any():
                            buffer += chunk.decode("utf-8", errors="replace")
                            for line in buffer.split("\n"):
                                line = line.strip()
                                if line.startswith("data: "):
                                    try:
                                        result = json.loads(line[6:])
                                        break
                                    except json.JSONDecodeError:
                                        pass
                            if result is not None:
                                break
                    except (aiohttp.ClientPayloadError, aiohttp.ClientError):
                        pass
                    if result is None:
                        for line in buffer.split("\n"):
                            line = line.strip()
                            if line.startswith("data: "):
                                try:
                                    result = json.loads(line[6:])
                                    break
                                except json.JSONDecodeError:
                                    pass
                    if result is None:
                        return json.dumps({"error": f"No valid JSON in SSE response. Buffer len: {len(buffer)}"})
                else:
                    result = await resp.json()

                if "error" in result:
                    return json.dumps({"error": str(result["error"])})

                tool_result = result.get("result", result)
                # Extract text content from MCP result format
                if isinstance(tool_result, dict) and "content" in tool_result:
                    contents = tool_result["content"]
                    if isinstance(contents, list) and len(contents) > 0:
                        first = contents[0]
                        if isinstance(first, dict) and "text" in first:
                            return first["text"]
                return json.dumps(tool_result) if not isinstance(tool_result, str) else tool_result
        except (aiohttp.ClientPayloadError, aiohttp.ClientError):
            # Session/response exit may trigger this; use whatever result we parsed
            if result is not None:
                if "error" in result:
                    return json.dumps({"error": str(result["error"])})
                tool_result = result.get("result", result)
                if isinstance(tool_result, dict) and "content" in tool_result:
                    contents = tool_result["content"]
                    if isinstance(contents, list) and len(contents) > 0:
                        first = contents[0]
                        if isinstance(first, dict) and "text" in first:
                            return first["text"]
                return json.dumps(tool_result) if not isinstance(tool_result, str) else tool_result
            # Try final buffer parse
            for line in buffer.split("\n"):
                line = line.strip()
                if line.startswith("data: "):
                    try:
                        result = json.loads(line[6:])
                        tool_result = result.get("result", result)
                        if isinstance(tool_result, dict) and "content" in tool_result:
                            contents = tool_result["content"]
                            if isinstance(contents, list) and len(contents) > 0:
                                first = contents[0]
                                if isinstance(first, dict) and "text" in first:
                                    return first["text"]
                        return json.dumps(tool_result) if not isinstance(tool_result, str) else tool_result
                    except json.JSONDecodeError:
                        pass
            return json.dumps({"error": "SSE response incomplete after ClientPayloadError"})

    def _format_mcp_result(self, result) -> str:
        """Format MCP result to string."""
        if hasattr(result, "content") and result.content:
            if hasattr(result.content[0], "text"):
                return result.content[0].text
            return str(result.content[0])
        return str(result)

    def finish(self, answer: str, explanation: str = "") -> ToolResponse:
        """Return the final result when task is complete.

        Args:
            answer: The final answer to the task.
            explanation: Brief explanation of the answer.
        """
        return ToolResponse(
            content=[TextBlock(type="text", text="Task Finished")],
            metadata={"answer": answer, "explanation": explanation},
        )

    def _process_memory_result(self, context: str = ""):
        """Process compression result, record penalties for failed attempts and reward for success."""
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

        attempts = add_result.get("attempts", [])
        is_terminal_oot = add_result.get("status") == "UnrecoverableOutOfTokens"
        no_change_count = add_result.get("no_change_count", 0)
        status = add_result.get("status", "")
        degeneration_rewards = add_result.get("degeneration_rewards", []) or []
        repetition_stats = add_result.get("repetition_stats") or {}
        self.compression_attempt_types.extend(
            attempt.get("type", "") for attempt in attempts if attempt.get("type")
        )
        if self.experiment_logger:
            self.experiment_logger.log_info(
                f"_process_memory_result ({context}): {len(attempts)} attempts, "
                f"types=[{', '.join(a.get('type', '?') for a in attempts)}]"
            )

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
                    "step": self.compression_step,
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
                text_content = add_result.get("modification_response_text")
                if text_content:
                    self.last_compression_text = text_content
                    self.last_compression_step = self.compression_step
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
        """Perform reasoning step to decide next action."""
        try:
            memory_msgs = await self.memory.get_memory()
            formatted_messages = await self.formatter.format(msgs=[self.system_msg, *memory_msgs])
            tools = self._build_tool_schemas()

            response = await asyncio.wait_for(
                retry_model_call(
                    self.model,
                    formatted_messages,
                    tools=tools,
                    max_retries=20,
                    experiment_logger=self.experiment_logger,
                    **get_model_call_kwargs(self.model, self.enable_thinking,
                                            getattr(self, 'thinking_config', None)),
                ),
                timeout=180,  # 3 min max per reasoning step
            )

            # Response shape (channel-marker cleanup, tool_use id → UUID,
            # etc.) is handled inside retry_model_call via
            # asio.utils.response_shape.
            if hasattr(response, "content"):
                # MCP-specific business rule: cap tool calls per reasoning
                # step to avoid explosive tool fan-out.
                max_tools = 5
                tool_blocks = [b for b in response.content if (isinstance(b, dict) and b.get("type") == "tool_use") or (hasattr(b, "type") and getattr(b, "type", None) == "tool_use")]
                if len(tool_blocks) > max_tools:
                    if self.experiment_logger:
                        self.experiment_logger.log_warning(
                            f"Model returned {len(tool_blocks)} tool calls, trimming to {max_tools}"
                        )
                    kept_ids = set()
                    for b in tool_blocks[:max_tools]:
                        kept_ids.add(b.get("id") if isinstance(b, dict) else getattr(b, "id", None))
                    response.content = [
                        b for b in response.content
                        if not ((isinstance(b, dict) and b.get("type") == "tool_use") or (hasattr(b, "type") and getattr(b, "type", None) == "tool_use"))
                        or (b.get("id") if isinstance(b, dict) else getattr(b, "id", None)) in kept_ids
                    ]

                msg = Msg(self.name, response.content, "assistant")
            else:
                msg = Msg(self.name, str(response), "assistant")

            await self.memory.add(msg)
            self._process_memory_result(context="Reasoning")
            return msg

        except Exception as e:
            tb = traceback.format_exc()
            if self.experiment_logger:
                self.experiment_logger.log_error(f"Error in reasoning: {e}\nTraceback: {tb}")
            raise

    async def _acting(self, tool_call: dict) -> None:
        """Execute a tool call and add result to memory."""
        try:
            tool_name = tool_call.get("name")
            tool_input = tool_call.get("input", {})
            tool_id = tool_call.get("id", str(uuid.uuid4()))

            # Check for duplicate tool call
            if self.calculate_reward:
                input_signature = json.dumps(tool_input, sort_keys=True)
                tool_signature = (tool_name, input_signature)

                if tool_signature == self.last_tool_signature:
                    if self.last_compression_step:
                        reward_entry = {
                            "text_content": self.last_compression_text,
                            "reward": self.reward_values["duplicate_call_penalty"],
                            "reason": "duplicated_tool_call",
                            "tool_name": tool_name,
        
                            "iteration": self.current_iteration,
                            "step": self.last_compression_step,
                        }
                        self.intermediate_rewards.append(reward_entry)
                        if self.experiment_logger:
                            self.experiment_logger.log_debug(
                                f"Intermediate Reward Recorded (Duplication): {json.dumps(reward_entry)}"
                            )

                self.last_tool_signature = tool_signature

            # Execute tool
            if tool_name == "finish":
                result_text = "Task Finished"
            else:
                result_text = await self._execute_mcp_tool(tool_name, tool_input)

            tool_result_msg = Msg(
                name="system",
                content=[
                    ToolResultBlock(
                        type="tool_result",
                        id=tool_id,
                        name=tool_name,
                        output=result_text,
                    )
                ],
                role=self._get_tool_result_role(),
            )

            await self.memory.add(tool_result_msg)

        except Exception as e:
            tb = traceback.format_exc()
            if self.experiment_logger:
                self.experiment_logger.log_error(f"Error in acting: {e}\nTraceback: {tb}")

            error_msg = Msg(
                name="system",
                content=[
                    ToolResultBlock(
                        type="tool_result",
                        id=tool_call.get("id", str(uuid.uuid4())),
                        name=tool_call.get("name", "unknown"),
                        output=f"Tool execution failed: {str(e)}",
                    )
                ],
                role=self._get_tool_result_role(),
            )
            await self.memory.add(error_msg)

        self._process_memory_result(context="Acting")

    async def _summarizing(self) -> Msg:
        """Generate a summary when max iterations reached."""
        try:
            hint_msg = Msg(
                name="user",
                content=[
                    {
                        "type": "text",
                        "text": "You have reached the maximum number of iterations. "
                        "Please provide your final answer based on the information gathered so far.",
                    }
                ],
                role="user",
            )

            memory_msgs = await self.memory.get_memory()
            formatted_messages = await self.formatter.format(
                msgs=[self.system_msg, *memory_msgs, hint_msg]
            )

            # Provide only the finish tool so the model can properly call it
            finish_tool = {
                "type": "function",
                "function": {
                    "name": "finish",
                    "description": "Call this tool when you have completed the task or cannot progress further. Provide your final answer.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "answer": {
                                "type": "string",
                                "description": "Your final answer to the task.",
                            },
                            "explanation": {
                                "type": "string",
                                "description": "A brief explanation of how you arrived at your answer.",
                            },
                        },
                        "required": ["answer"],
                    },
                },
            }

            response = await retry_model_call(
                self.model,
                formatted_messages,
                tools=[finish_tool],
                tool_choice={"type": "function", "function": {"name": "finish"}},
                max_retries=20,
                experiment_logger=self.experiment_logger,
                **get_model_call_kwargs(self.model, self.enable_thinking,
                                            getattr(self, 'thinking_config', None)),
            )

            # Extract answer from finish tool call if present
            if hasattr(response, "content"):
                if isinstance(response.content, list):
                    for block in response.content:
                        if isinstance(block, dict) and block.get("type") == "tool_use":
                            if block.get("name") == "finish":
                                answer = block.get("input", {}).get("answer", "")
                                msg = Msg(self.name, answer, "assistant")
                                return msg
                msg = Msg(self.name, response.content, "assistant")
            else:
                msg = Msg(self.name, str(response), "assistant")

            return msg

        except Exception as e:
            tb = traceback.format_exc()
            if self.experiment_logger:
                self.experiment_logger.log_error(f"Error in summarizing: {e}\nTraceback: {tb}")

            return Msg(
                self.name,
                [{"type": "text", "text": f"Error generating summary: {str(e)}"}],
                "assistant",
            )

    async def _run_task_internal(
        self, task_description: str
    ) -> Dict[str, Any]:
        """Internal logic for running MCP task."""
        try:
            await self.memory.clear()
            self.last_tool_signature = None
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

            user_msg = Msg(
                name="user",
                content=[{"type": "text", "text": f"Task: {task_description}"}],
                role="user",
            )
            await self.memory.add(user_msg)

            tool_calls_count = defaultdict(int)
            tool_call_details = []  # Per-call records for judge eval
            final_answer = None

            for iteration in range(self.max_iters):
                self.current_iteration = iteration
                try:
                    response = await self._reasoning()
                except (asyncio.TimeoutError, Exception) as e:
                    if self.experiment_logger:
                        self.experiment_logger.log_warning(
                            f"Reasoning failed (iter {iteration}): {type(e).__name__}: {str(e)[:80]}. Retrying..."
                        )
                    continue  # Skip this iteration, try again

                if self.context_manager_terminal:
                    break

                tool_use_blocks = response.get_content_blocks("tool_use")

                if not tool_use_blocks:
                    if self.stop_on_no_tool_use:
                        if hasattr(response, "content"):
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

                finish_called = False
                for tool_call in tool_use_blocks:
                    tool_name = tool_call.get("name")
                    tool_input = tool_call.get("input", {})

                    if tool_name == "finish":
                        final_answer = tool_input.get("answer", "")
                        finish_called = True
                        break

                    tool_calls_count[tool_name] += 1
                    # Record per-call details for judge evaluation
                    server = ""
                    if tool_name in self.mcp_tools:
                        server = self.mcp_tools[tool_name].get("server", "")
                    elif "__" in tool_name:
                        server = tool_name.split("__")[0]
                    tool_call_details.append({
                        "tool": tool_name,
                        "parameters": tool_input,
                        "server": server,
                        "success": True,
                    })
                    await self._acting(tool_call)
                    if self.context_manager_terminal:
                        break

                if self.context_manager_terminal:
                    break

                if finish_called:
                    break

            if final_answer is None and not self.context_manager_terminal:
                summary_response = await self._summarizing()
                if hasattr(summary_response, "content"):
                    if isinstance(summary_response.content, str):
                        final_answer = summary_response.content
                    elif isinstance(summary_response.content, list):
                        for block in summary_response.content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                final_answer = block.get("text", "")
                                break

            results = {
                "task_description": task_description,
                "answer": final_answer or "No answer generated",
                "tool_calls": dict(tool_calls_count),
                "tool_call_details": tool_call_details,
                "total_iterations": iteration + 1,
                "success": final_answer is not None and not self.context_manager_terminal,
                "end_reason": self.context_manager_end_reason,
                "intermediate_rewards": self.intermediate_rewards,
                "compression_attempt_types": self.compression_attempt_types,
                "compression_metrics_by_step": self.compression_metrics_by_step,
                "compression_repetition_stats_by_step": self.compression_repetition_stats_by_step,
            }

            if self.experiment_logger:
                self.experiment_logger.log_debug(
                    f"Task completed in {results['total_iterations']} iterations"
                )

            return results

        except Exception as e:
            tb = traceback.format_exc()
            error_msg = f"Error during MCP task: {e}\nTraceback: {tb}"

            if self.experiment_logger:
                self.experiment_logger.log_error(error_msg)

            return {
                "task_description": task_description,
                "answer": None,
                "error": str(e),
                "success": False,
                "traceback": tb,
            }

    async def run_mcp_task(
        self, task_description: str
    ) -> Dict[str, Any]:
        """Run an MCP task.

        Args:
            task_description: The task description (fuzzy_description from MCP-Bench).

        Returns:
            Dictionary with task results including answer and metrics.
        """
        try:
            # Initialize tokenizer
            await self._init_tokenizer()

            # Initialize MCP connections
            await self._init_mcp_connections()

            # Run the task
            return await self._run_task_internal(task_description)

        finally:
            # Flush cache to disk
            if self._tool_cache is not None:
                self._tool_cache.flush()
            # Clean up MCP connections
            await self._cleanup_mcp_connections()

    async def _cleanup_mcp_connections(self):
        """Clean up MCP client connections."""
        for server_name, client in self.mcp_clients.items():
            if not isinstance(client, dict):  # SSE client
                try:
                    await client.__aexit__(None, None, None)
                except Exception as e:
                    if self.experiment_logger:
                        self.experiment_logger.log_warning(
                            f"Error closing MCP client {server_name}: {e}"
                        )
        self.mcp_clients.clear()
        self.mcp_tools.clear()
