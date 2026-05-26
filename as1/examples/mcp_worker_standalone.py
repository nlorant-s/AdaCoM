#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MCP Worker Standalone for MCP-Bench Evaluation.

This script provides a standalone interface for running MCPWorker against
MCP-Bench tasks. It supports both single task and batch execution modes.

USAGE:
    # Single task from MCP-Bench task file
    python mcp_worker_standalone.py \
        --task-file ../mcp-bench/tasks/mcpbench_tasks_single_runner_format.json \
        --task-id wikipedia_000 \
        --agent-model qwen3-max \
        --enable-memory

    # Batch run from parallel_runner.py
    python mcp_worker_standalone.py \
        --tasks-file /tmp/tasks_xxx.json \
        --agent-model qwen3-max \
        --run-name my_test_run

Located in: as1/examples/
"""

import os
import sys
import json
import argparse
import asyncio
import copy
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentscope.model import OpenAIChatModel, AnthropicChatModel, DashScopeClaudeChatModel, enable_auto_llm_logging
from agentscope.formatter import OpenAIChatFormatter, AnthropicChatFormatter
from agentscope.message import Msg
from agentscope.tool import Toolkit

from asio.logger import ExperimentLogger
from asio.utils.judge import judge_result


# ============================================================================
# Common Utilities
# ============================================================================

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
    api_key: str = None,
    stream: bool = False,
    base_url: str = None,
    **kwargs,
):
    """Create model and formatter instances."""
    is_anthropic = _is_anthropic_model(model_name)
    is_dashscope = _is_dashscope_url(base_url)

    if is_anthropic and is_dashscope:
        api_key = api_key or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Please set DASHSCOPE_API_KEY or OPENAI_API_KEY environment variable for DashScope Claude")

        model = DashScopeClaudeChatModel(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            stream=stream,
            provider="r",
        )
        formatter = OpenAIChatFormatter()
    elif is_anthropic:
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Please set ANTHROPIC_API_KEY environment variable for Claude models")

        client_args = {}
        if base_url:
            client_args["base_url"] = base_url

        model = AnthropicChatModel(
            model_name=model_name,
            stream=stream,
            api_key=api_key,
            client_args=client_args if client_args else None,
        )
        formatter = AnthropicChatFormatter()
    else:
        api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("Please set OPENAI_API_KEY or DASHSCOPE_API_KEY environment variable")

        client_args = {}
        if base_url:
            client_args["base_url"] = base_url

        model = OpenAIChatModel(
            model_name=model_name,
            stream=stream,
            api_key=api_key,
            client_args=client_args,
        )
        formatter = OpenAIChatFormatter()

    return model, formatter


def load_tasks_from_file(tasks_file: Path) -> List[Dict[str, Any]]:
    """Load tasks from JSON file."""
    with open(tasks_file, 'r', encoding='utf-8') as f:
        tasks = json.load(f)
    return tasks


def load_mcp_bench_task(task_file: Path, task_id: str) -> Dict[str, Any]:
    """Load a specific task from MCP-Bench task file."""
    with open(task_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Search for the task in the nested structure
    for server_entry in data.get("server_tasks", []):
        for task in server_entry.get("tasks", []):
            if task.get("task_id") == task_id:
                # Add server info to the task
                task["servers"] = server_entry.get("servers", [server_entry.get("server_name")])
                return task

    raise ValueError(f"Task ID '{task_id}' not found in {task_file}")


# ============================================================================
# Default MCP Server Configurations
# ============================================================================

def get_default_mcp_configs(mcp_bench_path: str = None, commands_file: str = None) -> List[Dict[str, Any]]:
    """Get MCP server configurations.

    If commands_file is provided (e.g., commands_dlc.json), loads configs from it.
    Supports stdio, http, and remote_http transport types.
    Otherwise falls back to hardcoded defaults.
    """
    mcp_bench_path = mcp_bench_path or os.environ.get("MCP_BENCH_PATH", "")

    # Try to load from commands file (e.g., commands_dlc.json)
    if commands_file is None:
        # Auto-detect: prefer commands_dlc.json if on remote
        candidate = os.path.join(mcp_bench_path, "mcp_servers", "commands_dlc.json") if mcp_bench_path else None
        if candidate and os.path.exists(candidate):
            commands_file = candidate

    if commands_file and os.path.exists(commands_file):
        with open(commands_file) as f:
            commands = json.load(f)

        configs = []
        for name, conf in commands.items():
            transport = conf.get("transport", "stdio")
            if transport in ("remote_http", "http"):
                configs.append({
                    "name": name,
                    "transport": transport,
                    "host": conf.get("host", "localhost"),
                    "port": conf.get("port"),
                    "endpoint": conf.get("endpoint", "/mcp"),
                })
            else:
                # stdio
                cmd_str = conf.get("cmd", "")
                command = cmd_str.split() if cmd_str else []
                cwd = conf.get("cwd")
                if cwd and mcp_bench_path:
                    # Resolve relative cwd (e.g., "../unit-converter-mcp" -> absolute)
                    servers_dir = os.path.join(mcp_bench_path, "mcp_servers")
                    cwd = os.path.normpath(os.path.join(servers_dir, cwd))
                configs.append({
                    "name": name,
                    "transport": "stdio",
                    "command": command,
                    "env": {},
                    "cwd": cwd,
                })
        return configs

    # Fallback: hardcoded defaults
    wikipedia_proxy = os.environ.get("WIKIPEDIA_PROXY", "")
    configs = [
        {
            "name": "Wikipedia",
            "transport": "stdio",
            "command": ["python", "-m", "wikipedia_mcp"] + (["--proxy", wikipedia_proxy] if wikipedia_proxy else []),
            "env": {"WIKIPEDIA_PROXY": wikipedia_proxy} if wikipedia_proxy else {},
            "cwd": os.path.join(mcp_bench_path, "mcp_servers/wikipedia-mcp") if mcp_bench_path else None,
        },
        {
            "name": "Unit Converter",
            "transport": "stdio",
            "command": ["python", "-m", "unit_converter_mcp.server"],
            "env": {},
            "cwd": os.path.join(mcp_bench_path, "mcp_servers/unit-converter-mcp") if mcp_bench_path else None,
        },
    ]

    return configs


# ============================================================================
# MCP Worker Task Execution
# ============================================================================

async def run_mcp_task(
    task_description: str,
    ground_truth: Optional[str] = None,
    agent_model: str = "qwen3-max",
    memory_model: Optional[str] = None,
    use_memory_manager: bool = False,
    mcp_server_configs: Optional[List[Dict[str, Any]]] = None,
    max_iters: int = 50,
    output_dir: str = "./outputs",
    task_id: Any = "task_0",
    row_index: int = 0,
    run_name: str = "mcp_standalone",
    dataset: str = "mcp_bench",
    verbose: bool = False,
    max_model_len: Optional[int] = None,
    compress_fre: Optional[int] = None,
    compress_fre_min: Optional[int] = None,
    compress_fre_max: Optional[int] = None,
    calculate_reward: bool = False,
    stop_on_no_tool_use: bool = True,
    judge_model: Optional[str] = None,
    tokenizer_model: Optional[str] = None,
    mcp_bench_path: Optional[str] = None,
    tool_cache_path: Optional[str] = None,
    tool_schemas_path: Optional[str] = None,
    es_fallback_config: Optional[Dict[str, Any]] = None,
    skip_live_mcp_servers: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run a single MCP task."""
    from asio.agent.mcp_worker import MCPWorker

    # Setup logger
    logger = ExperimentLogger(
        base_dir=output_dir,
        test_mem_config=run_name,
        dataset="tasks"
    )
    logger.start_run(str(task_id))
    enable_auto_llm_logging(logger_instance=logger)

    run_output_dir = logger.current_run_dir

    try:
        base_url = os.environ.get("BASE_URL")

        logger.log_info(f"Initializing agent model: {agent_model}")
        agent_model_instance, agent_formatter = create_model_and_formatter(
            model_name=agent_model,
            stream=False,
            base_url=base_url,
        )

        # Initialize memory
        if use_memory_manager and memory_model:
            logger.log_info(f"Using MemoryManager with model: {memory_model}")

            memory_model_instance, memory_formatter = create_model_and_formatter(
                model_name=memory_model,
                stream=False,
                base_url=base_url,
            )

            memory_config = {
                "_api_model": memory_model_instance,
                "_api_formatter": memory_formatter,
                "sys_prompt": """You are a memory manager for a multi-tool agent.
                Your role is to:
                1. Track the tool usage history
                2. Summarize key findings from tool calls
                3. Help the agent avoid redundant tool calls
                4. Maintain context about the task

                Keep your responses concise and focused on helping the agent complete tasks efficiently.""",
                "debug_dir": str(run_output_dir),
                "max_messages": 20,
                "summarize_threshold": 15,
            }
            if max_model_len is not None:
                memory_config["max_model_len"] = max_model_len
            if compress_fre is not None:
                memory_config["compress_fre"] = compress_fre
            if compress_fre_min is not None:
                memory_config["compress_fre_min"] = compress_fre_min
            if compress_fre_max is not None:
                memory_config["compress_fre_max"] = compress_fre_max
            memory_class = "memorymanager"
        else:
            logger.log_info("Using InMemory storage")
            memory_config = {"debug_dir": str(run_output_dir)}
            if max_model_len is not None:
                memory_config["max_model_len"] = max_model_len
            memory_class = "inmemory"

        # Get MCP server configs
        if mcp_server_configs is None:
            mcp_server_configs = get_default_mcp_configs(mcp_bench_path)

        logger.log_info(f"Using {len(mcp_server_configs)} MCP servers")

        # Initialize MCP worker
        logger.log_info("Initializing MCP Worker agent")
        worker = MCPWorker(
            name="assistant",
            model=agent_model_instance,
            formatter=agent_formatter,
            server_configs=mcp_server_configs,
            memory_class=memory_class,
            memory_config=memory_config,
            experiment_logger=logger,
            max_iters=max_iters,
            task_id=str(task_id),
            calculate_reward=calculate_reward,
            stop_on_no_tool_use=stop_on_no_tool_use,
            tokenizer_model=tokenizer_model,
            tool_cache_path=tool_cache_path,
            tool_schemas_path=tool_schemas_path,
            es_fallback_config=es_fallback_config,
            skip_live_mcp_servers=skip_live_mcp_servers,
        )

        # Run the task
        logger.log_info(f"Running MCP task: {task_description[:100]}...")
        start_time = datetime.now()

        result = await worker.run_mcp_task(task_description, ground_truth)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        result["duration_seconds"] = duration
        result["timestamp"] = start_time.isoformat()
        result["task_id"] = task_id
        result["row_index"] = row_index

        # Run judge if ground truth is available
        if ground_truth:
            judge_model_name = judge_model or agent_model
            logger.log_info(f"Running judge evaluation using model: {judge_model_name}")
            judge_model_instance_j, judge_formatter_j = create_model_and_formatter(
                model_name=judge_model_name,
                stream=False,
                base_url=base_url,
            )
            judge_output_dict = await judge_result(
                question=task_description,
                correct_answer=ground_truth,
                actual_answer=result.get("answer", ""),
                judge_model_instance=judge_model_instance_j,
                judge_formatter=judge_formatter_j,
                logger=logger
            )
            result.update(judge_output_dict)
            logger.log_info(f"Judge decision: {'CORRECT' if result.get('score', 0) >= 1.0 else 'INCORRECT'}")

        result["success"] = True
        logger.log_info(f"Task completed in {duration:.2f} seconds")
        logger.log_info(f"Tool usage: {result.get('tool_calls', {})}")

        # Save results
        logger.save_report(result, filename="result.json")

        return result

    except Exception as e:
        tb = traceback.format_exc()
        print(f"Task failed: {e}\nTraceback: {tb}", file=sys.stderr)
        logger.log_error(f"Task failed: {e}\nTraceback: {tb}")
        result = {
            "task_description": task_description,
            "ground_truth": ground_truth,
            "task_id": task_id,
            "row_index": row_index,
            "error": str(e),
            "traceback": tb,
            "success": False,
            "score": 0.0
        }
        logger.save_report(result, filename="result.json")
        return result


# ============================================================================
# Batch Execution
# ============================================================================

async def run_batch_tasks(
    tasks_list: List[Dict[str, Any]],
    output_dir: str = "./outputs",
    run_name: str = None,
    dataset: str = "mcp_bench",
    timeout: Optional[int] = None,
    max_concurrent: int = 1,
    agent_model: str = "qwen3-max",
    memory_model: Optional[str] = None,
    use_memory_manager: bool = False,
    mcp_server_configs: Optional[List[Dict[str, Any]]] = None,
    max_iters: int = 50,
    max_model_len: Optional[int] = None,
    compress_fre: Optional[int] = None,
    compress_fre_min: Optional[int] = None,
    compress_fre_max: Optional[int] = None,
    calculate_reward: bool = False,
    stop_on_no_tool_use: bool = True,
    judge_model: Optional[str] = None,
    tokenizer_model: Optional[str] = None,
    mcp_bench_path: Optional[str] = None,
    tool_cache_path: Optional[str] = None,
    tool_schemas_path: Optional[str] = None,
    es_fallback_config: Optional[Dict[str, Any]] = None,
    skip_live_mcp_servers: Optional[List[str]] = None,
    **kwargs
) -> List[Dict[str, Any]]:
    """Run multiple MCP tasks."""
    if run_name is None:
        run_name = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"Running {len(tasks_list)} tasks with run_name: {run_name}")

    semaphore = asyncio.Semaphore(max_concurrent)

    async def run_with_semaphore(task_idx: int, task: dict) -> Dict[str, Any]:
        async with semaphore:
            print(f"Running task {task_idx + 1}/{len(tasks_list)}")

            # Extract task info from MCP-Bench format
            task_description = (
                task.get("fuzzy_description") or
                task.get("task_description") or
                task.get("question") or
                task.get("query")
            )
            ground_truth = task.get("expected_output") or task.get("answer") or task.get("ground_truth")
            task_id = task.get("task_id", f"task_{task_idx}")
            row_index = task.get("row_index", task_idx)

            try:
                coro = run_mcp_task(
                    task_description=task_description,
                    ground_truth=ground_truth,
                    agent_model=agent_model,
                    memory_model=memory_model,
                    use_memory_manager=use_memory_manager,
                    mcp_server_configs=mcp_server_configs,
                    max_iters=max_iters,
                    output_dir=output_dir,
                    task_id=task_id,
                    row_index=row_index,
                    run_name=run_name,
                    dataset=dataset,
                    max_model_len=max_model_len,
                    compress_fre=compress_fre,
                    compress_fre_min=compress_fre_min,
                    compress_fre_max=compress_fre_max,
                    calculate_reward=calculate_reward,
                    stop_on_no_tool_use=stop_on_no_tool_use,
                    judge_model=judge_model,
                    tokenizer_model=tokenizer_model,
                    mcp_bench_path=mcp_bench_path,
                    tool_cache_path=tool_cache_path,
                    tool_schemas_path=tool_schemas_path,
                    es_fallback_config=es_fallback_config,
                    skip_live_mcp_servers=skip_live_mcp_servers,
                )

                if timeout:
                    result = await asyncio.wait_for(coro, timeout=timeout)
                else:
                    result = await coro

            except asyncio.TimeoutError:
                print(f"Task {task_idx + 1} timed out after {timeout} seconds")
                result = {
                    "task_description": task_description,
                    "ground_truth": ground_truth,
                    "task_id": task_id,
                    "row_index": row_index,
                    "error": f"Task timed out after {timeout} seconds",
                    "success": False,
                    "timeout": True,
                    "score": 0.0
                }
            except Exception as e:
                print(f"Task {task_idx + 1} failed with error: {e}")
                result = {
                    "task_description": task_description,
                    "ground_truth": ground_truth,
                    "task_id": task_id,
                    "row_index": row_index,
                    "error": str(e),
                    "success": False,
                    "score": 0.0
                }

            result["task_index"] = task_idx
            return result

    tasks_futures = [run_with_semaphore(i, task) for i, task in enumerate(tasks_list)]
    results = await asyncio.gather(*tasks_futures)

    return list(results)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="MCP Worker Standalone for MCP-Bench evaluation")

    # Task input modes
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--tasks-file", type=str,
                            help="Path to JSON file containing tasks (batch mode)")
    input_group.add_argument("--task-file", type=str,
                            help="Path to MCP-Bench task file (single task mode)")

    parser.add_argument("--task-id", type=str, default=None,
                        help="Task ID to run (required when using --task-file)")
    parser.add_argument("--dataset", type=str, default="mcp_bench",
                        help="Dataset name for output directory structure")

    # Model configuration
    parser.add_argument("--agent-model", type=str, default="qwen3-max",
                        help="Model name for main agent")
    parser.add_argument("--memory-model", type=str, default=None,
                        help="Model name for memory manager")
    parser.add_argument("--enable-memory", action="store_true",
                        help="Use MemoryManager instead of InMemory")
    parser.add_argument("--judge-model", type=str, default=None,
                        help="Model name for judge (default: uses agent model)")
    parser.add_argument("--tokenizer-model", type=str, default=None,
                        help="Model name/path for tokenizer")

    # MCP configuration
    parser.add_argument("--mcp-bench-path", type=str, default=None,
                        help="Path to MCP-Bench repository (for MCP server working directories)")
    parser.add_argument("--commands-file", type=str, default=None,
                        help="Path to MCP commands JSON file (e.g., commands_dlc.json for remote_http servers)")

    # Memory configuration
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="Max model length for memory_config")
    parser.add_argument("--compress-fre", type=int, default=None,
                        help="Fixed compression frequency")
    parser.add_argument("--compress-fre-min", type=int, default=None,
                        help="Minimum compression frequency")
    parser.add_argument("--compress-fre-max", type=int, default=None,
                        help="Maximum compression frequency")
    parser.add_argument("--calculate-reward", action="store_true",
                        help="Enable reward calculation for memory compression")

    # Execution parameters
    parser.add_argument("--max-iters", type=int, default=50,
                        help="Maximum iterations for agent")
    parser.add_argument("--max-concurrent", type=int, default=1,
                        help="Maximum concurrent tasks (for batch)")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Timeout in seconds for each task")
    parser.add_argument("--con-on-no-tool-use", action="store_true",
                        help="Continue task when no tool_use blocks found instead of stopping")

    # Output configuration
    parser.add_argument("--output-dir", type=str, default="./benchmark_results",
                        help="Directory for output files")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Run name for output directory")

    # Cache configuration
    parser.add_argument("--tool-cache-path", type=str, default=None,
                        help="Path to JSON file for MCP tool result cache (enables cross-run caching)")
    parser.add_argument("--tool-schemas-path", type=str, default=None,
                        help="Path to pre-dumped MCP tool schemas JSON (offline mode, no MCP server needed)")

    # ES fallback configuration
    parser.add_argument("--es-fallback", action="store_true",
                        help="Enable Elasticsearch fallback for Wikipedia tools on cache miss")
    parser.add_argument("--es-url", type=str, default=None,
                        help="Elasticsearch URL (default: env WIKIPEDIA_ES_URL or http://localhost:9200)")
    parser.add_argument("--es-index", type=str, default=None,
                        help="Elasticsearch index name (default: env WIKIPEDIA_ES_INDEX or wikipedia_en)")
    parser.add_argument("--miss-log-path", type=str, default=None,
                        help="Path for cache miss JSONL log (for backfill script)")
    parser.add_argument("--skip-live-mcp", type=str, nargs="*", default=None,
                        help="Server names to skip live MCP calls for (cache+ES only, e.g. Wikipedia)")

    args = parser.parse_args()

    # Validate arguments
    if args.task_file and not args.task_id:
        parser.error("--task-id is required when using --task-file")

    # Set default run name
    if args.run_name is None:
        args.run_name = f"mcp_standalone_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Load tasks
    if args.task_file:
        # Single task mode from MCP-Bench task file
        task = load_mcp_bench_task(Path(args.task_file), args.task_id)
        tasks_list = [task]
    else:
        # Batch mode from tasks file
        tasks_list = load_tasks_from_file(Path(args.tasks_file))

    print(f"Loaded {len(tasks_list)} task(s)")

    if not tasks_list:
        print("No tasks to run.")
        sys.exit(1)

    # Load MCP server configs
    mcp_server_configs = get_default_mcp_configs(args.mcp_bench_path, args.commands_file)

    # Filter configs to only include servers needed by tasks
    all_task_servers = set()
    for t in tasks_list:
        all_task_servers.update(t.get("servers", []))
    if all_task_servers:
        mcp_server_configs = [c for c in mcp_server_configs if c["name"] in all_task_servers]
    print(f"Using {len(mcp_server_configs)} MCP servers: {[c['name'] for c in mcp_server_configs]}")

    # Run tasks
    asyncio.run(run_batch_tasks(
        tasks_list=tasks_list,
        output_dir=args.output_dir,
        run_name=args.run_name,
        dataset=args.dataset,
        timeout=args.timeout,
        max_concurrent=args.max_concurrent,
        agent_model=args.agent_model,
        memory_model=args.memory_model,
        use_memory_manager=args.enable_memory,
        max_iters=args.max_iters,
        max_model_len=args.max_model_len,
        compress_fre=args.compress_fre,
        compress_fre_min=args.compress_fre_min,
        compress_fre_max=args.compress_fre_max,
        calculate_reward=args.calculate_reward,
        stop_on_no_tool_use=not args.con_on_no_tool_use,
        judge_model=args.judge_model,
        tokenizer_model=args.tokenizer_model,
        mcp_bench_path=args.mcp_bench_path,
        mcp_server_configs=mcp_server_configs,
        tool_cache_path=args.tool_cache_path,
        tool_schemas_path=args.tool_schemas_path,
        es_fallback_config={
            "enabled": args.es_fallback,
            "es_url": args.es_url,
            "es_index": args.es_index,
            "miss_log_path": args.miss_log_path,
        } if args.es_fallback else None,
        skip_live_mcp_servers=args.skip_live_mcp,
    ))


if __name__ == "__main__":
    main()
