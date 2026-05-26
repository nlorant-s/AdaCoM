#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified Worker Standalone for BrowseComp-Plus (BCP) and MCP-Bench Evaluation.

Receives task data from parallel_runner.py via a JSON file.

USAGE:
    # Run with tasks file (called by parallel_runner.py)
    python worker_standalone.py --worker-type bcp --tasks-file /tmp/tasks_xxx.json --agent-model qwen3-max

    # The tasks file should contain a JSON array of task objects:
    # [{"question": "...", "answer": "...", "task_id": "query_001", "row_index": 0}, ...]
"""

import os
import sys
import json
import argparse
import asyncio
import copy
import shutil
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentscope.model import OpenAIChatModel, AnthropicChatModel, DashScopeClaudeChatModel, GeminiChatModel, enable_auto_llm_logging
from agentscope.formatter import OpenAIChatFormatter, AnthropicChatFormatter, GeminiChatFormatter
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


def _is_gemini_model(model_name: str) -> bool:
    """Check if the model name indicates a Gemini model."""
    return "gemini" in model_name.lower()


def create_model_and_formatter(
    model_name: str,
    api_key: str = None,
    stream: bool = False,
    base_url: str = None,
    reasoning_effort: str = None,
    enable_thinking: bool = False,
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
        if not api_key:
            raise ValueError("Please set DASHSCOPE_API_KEY or OPENAI_API_KEY environment variable for Gemini models")
        proxy_url = base_url or os.environ.get("BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

        thinking_cfg = None
        if enable_thinking:
            thinking_cfg = {"include_thoughts": True, "thinking_budget": -1}
        model = GeminiChatModel(
            model_name=model_name,
            api_key=api_key,
            stream=stream,
            client_args={"base_url": proxy_url},
            thinking_config=thinking_cfg,
        )
        formatter = GeminiChatFormatter()
    elif is_anthropic and is_dashscope:
        # DashScope Claude: Use OpenAI-compatible endpoint with special params
        api_key = api_key or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Please set DASHSCOPE_API_KEY or OPENAI_API_KEY environment variable for DashScope Claude")

        model = DashScopeClaudeChatModel(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            stream=stream,
            provider="r",  # DashScope provider code for Claude relay
        )
        formatter = OpenAIChatFormatter()
    elif is_anthropic:
        # Native Anthropic API
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("Please set ANTHROPIC_API_KEY environment variable for Claude models")

        client_args = {}
        if base_url:
            # Anthropic SDK appends /v1/messages; strip a trailing /v1 so a single
            # BASE_URL like https://api2.aigcbest.top/v1 works for both SDKs.
            anthropic_base = base_url.rstrip("/")
            if anthropic_base.endswith("/v1"):
                anthropic_base = anthropic_base[:-3]
            client_args["base_url"] = anthropic_base

        model = AnthropicChatModel(
            model_name=model_name,
            stream=stream,
            api_key=api_key,
            client_args=client_args if client_args else None,
        )
        formatter = AnthropicChatFormatter()
    else:
        # OpenAI or compatible models
        api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("Please set OPENAI_API_KEY or DASHSCOPE_API_KEY environment variable")

        client_args = {}
        if base_url:
            client_args["base_url"] = base_url

        # Default reasoning_effort to "medium" for gpt-oss models
        if reasoning_effort is None and "gpt-oss" in model_name.lower():
            reasoning_effort = "medium"

        model = OpenAIChatModel(
            model_name=model_name,
            stream=stream,
            api_key=api_key,
            client_args=client_args,
            reasoning_effort=reasoning_effort,
        )
        formatter = OpenAIChatFormatter()

    return model, formatter


def load_tasks_from_file(tasks_file: Path) -> List[Dict[str, Any]]:
    """Load tasks from JSON file provided by parallel_runner.py.

    Expected format: [{"question": "...", "answer": "...", "task_id": "...", "row_index": 0}, ...]
    """
    with open(tasks_file, 'r', encoding='utf-8') as f:
        tasks = json.load(f)
    return tasks


# ============================================================================
# BCP Worker
# ============================================================================

async def run_bcp_task(
    question: str,
    ground_truth: Optional[str] = None,
    agent_model: str = "gpt-4o-mini",
    memory_model: Optional[str] = None,
    use_memory_manager: bool = False,
    searcher_type: str = "bm25",
    index_path: str = "indexes/bm25",
    browsecomp_path: Optional[str] = None,
    top_k: int = 5,
    snippet_max_tokens: int = 512,
    include_get_document: bool = True,
    max_iters: int = 50,
    output_dir: str = "./outputs",
    task_id: Any = 0,
    row_index: int = 0,
    run_name: str = "bcp_standalone",
    dataset: str = "default",
    verbose: bool = False,
    searcher_model: str = "",
    use_bg_info: bool = False,
    max_model_len: Optional[int] = None,
    compress_fre: Optional[int] = None,
    compress_fre_min: Optional[int] = None,
    compress_fre_max: Optional[int] = None,
    calculate_reward: bool = False,
    stop_on_no_tool_use: bool = True,
    enable_thinking: bool = True,
    judge_model: Optional[str] = None,
    tokenizer_model: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    stream: bool = False,
) -> Dict[str, Any]:
    """Run a single BCP task."""
    from asio.agent.bcp_worker import BCPWorker

    # Use "tasks" as dataset for unified output structure: {output_dir}/{run_name}/tasks/{task_id}/
    # task_id is the query_id from dataset, providing more meaningful folder names
    logger = ExperimentLogger(
        base_dir=output_dir,
        test_mem_config=run_name,
        dataset="tasks"
    )
    logger.start_run(task_id)
    enable_auto_llm_logging(logger_instance=logger)

    run_output_dir = logger.current_run_dir

    try:
        base_url = os.environ.get("BASE_URL")

        logger.log_info(f"Initializing agent model: {agent_model}")
        agent_model_instance, agent_formatter = create_model_and_formatter(
            model_name=agent_model,
            stream=stream,
            base_url=base_url,
            reasoning_effort=reasoning_effort,
            enable_thinking=enable_thinking,
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
                "sys_prompt": """You are a memory manager for a document search agent.
                Your role is to:
                1. Track the search history and document retrievals
                2. Summarize key findings from searches
                3. Help the agent avoid redundant searches
                4. Maintain context about the search task

                Keep your responses concise and focused on helping the agent find relevant information efficiently.""",
                "debug_dir": str(run_output_dir),
                "max_messages": 20,
                "summarize_threshold": 15,
                "use_bg_info": use_bg_info
            }
            if tokenizer_model is not None:
                memory_config["chat_tokenizer_model"] = tokenizer_model
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

        # Initialize BCP worker
        logger.log_info("Initializing BCP Worker agent")
        worker = BCPWorker(
            name="assistant",
            model=agent_model_instance,
            formatter=agent_formatter,
            searcher_type=searcher_type,
            index_path=index_path,
            memory_class=memory_class,
            memory_config=memory_config,
            experiment_logger=logger,
            top_k=top_k,
            snippet_max_tokens=snippet_max_tokens,
            include_get_document=include_get_document,
            browsecomp_path=browsecomp_path,
            max_iters=max_iters,
            searcher_model_name=searcher_model,
            calculate_reward=calculate_reward,
            stop_on_no_tool_use=stop_on_no_tool_use,
            enable_thinking=enable_thinking,
            tokenizer_model=tokenizer_model,
        )

        # Run the search task
        logger.log_info(f"Running search task: {question[:100]}...")
        start_time = datetime.now()

        result = await worker.run_search_task(question, ground_truth)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        result["duration_seconds"] = duration
        result["timestamp"] = start_time.isoformat()
        result["task_id"] = task_id
        result["row_index"] = row_index

        # Run judge if ground truth is available
        if ground_truth:
            judge_model_name = judge_model or os.environ.get("JUDGE_MODEL", "gpt-4o-2024-11-20")
            logger.log_info(f"Running judge evaluation using judge model: {judge_model_name}")
            judge_base_url = os.environ.get("JUDGE_BASE_URL") or base_url
            judge_api_key = os.environ.get("JUDGE_API_KEY")
            judge_model_instance_j, judge_formatter_j = create_model_and_formatter(
                model_name=judge_model_name,
                stream=False,
                api_key=judge_api_key,
                base_url=judge_base_url,
            )
            judge_output_dict = await judge_result(
                question=question,
                correct_answer=ground_truth,
                actual_answer=result.get("answer", ""),
                judge_model_instance=judge_model_instance_j,
                judge_formatter=judge_formatter_j,
                actual_explanation=result.get("explanation", ""),
                logger=logger
            )
            result.update(judge_output_dict)
            logger.log_info(f"Judge decision: {'CORRECT' if result.get('score', 0) >= 1.0 else 'INCORRECT'}")

        result["success"] = "traceback" not in result and "error" not in result
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
            "question": question,
            "ground_truth": ground_truth,
            "task_id": task_id,
            "row_index": row_index,
            "error": str(e),
            "traceback": tb,
            "success": False,
            "score": 0.0
        }
        # Save result.json even on failure for resume tracking
        logger.save_report(result, filename="result.json")
        return result


async def run_mcp_task(
    task: Dict[str, Any],
    agent_model: str = None,
    output_dir: str = "./benchmark_results",
    task_id: Any = 0,
    run_name: str = "mcp_standalone",
    dataset: str = "default",
    max_iters: int = 50,
    judge_model: Optional[str] = None,
    commands_file: Optional[str] = None,
    enable_memory: bool = False,
    memory_model: Optional[str] = None,
    max_model_len: Optional[int] = None,
    compress_fre: Optional[int] = None,
    compress_fre_min: Optional[int] = None,
    compress_fre_max: Optional[int] = None,
    tool_cache_path: Optional[str] = None,
    es_fallback_config: Optional[Dict[str, Any]] = None,
    skip_live_mcp_servers: Optional[List[str]] = None,
    reasoning_effort: Optional[str] = None,
    stream: bool = False,
) -> Dict[str, Any]:
    """Run a single MCP-Bench task."""
    from asio.agent.mcp_worker import MCPWorker
    MODEL_NAME = agent_model or os.getenv("MODEL", "gpt-4o-2024-11-20")
    JUDGE_MODEL = judge_model or os.getenv("JUDGE_MODEL", "gpt-5-2025-08-07")

    logger = ExperimentLogger(
        base_dir=output_dir,
        test_mem_config=run_name,
        dataset="tasks"
    )
    logger.start_run(task_id)
    enable_auto_llm_logging(logger_instance=logger)

    run_output_dir = logger.current_run_dir

    try:
        # Get task info from the raw MCP-Bench task data
        raw_task = task.get("raw", task)
        task_description = raw_task.get("fuzzy_description") or raw_task.get("task_description", "")
        if not task_description:
            task_description = task.get("question", "")
        expected_answer = task.get("answer") or raw_task.get("answer", "")
        servers = task.get("servers", raw_task.get("servers", []))

        print(f"[MCP Task {task_id}] Servers: {servers}")
        print(f"[MCP Task {task_id}] Description: {task_description[:100]}...")

        # Build MCP server configs from commands file
        mcp_server_configs = []
        commands = {}
        commands_base_dir = None
        if commands_file and os.path.exists(commands_file):
            commands_base_dir = os.path.dirname(os.path.abspath(commands_file))
            with open(commands_file, 'r') as f:
                commands = json.load(f)

        # Default STDIO commands for known MCP servers (used when no commands_file)
        default_server_commands = {
            "Wikipedia": ["python", "-m", "wikipedia_mcp"],
            "Unit Converter": ["python", "-m", "unit_converter_mcp.server"],
        }

        for server_name in servers:
            if server_name in commands:
                config = dict(commands[server_name])
                config["name"] = server_name
                # Resolve relative paths following mcp-bench convention:
                # commands file is in mcp_servers/, mcp-bench root is its parent
                if commands_base_dir:
                    mcp_bench_root = os.path.dirname(commands_base_dir)  # parent of mcp_servers/
                    # Resolve cwd: "../xxx" -> mcp_servers/xxx
                    if "cwd" in config:
                        cwd = config["cwd"]
                        if cwd.startswith("../"):
                            config["cwd"] = os.path.join(commands_base_dir, cwd[3:])
                        elif not os.path.isabs(cwd):
                            config["cwd"] = os.path.normpath(os.path.join(commands_base_dir, cwd))
                    # Resolve ".venv/" commands to mcp-bench root's venv
                    cmd_key = "cmd" if "cmd" in config else "command"
                    if cmd_key in config and isinstance(config[cmd_key], str):
                        if config[cmd_key].startswith(".venv/"):
                            config[cmd_key] = os.path.join(mcp_bench_root, config[cmd_key])
                mcp_server_configs.append(config)
            elif server_name in default_server_commands:
                mcp_server_configs.append({
                    "name": server_name,
                    "transport": "stdio",
                    "command": default_server_commands[server_name],
                })
            else:
                print(f"[MCP Task {task_id}] WARNING: Server '{server_name}' not found, skipping")
                continue

        if not mcp_server_configs:
            raise ValueError(f"No MCP servers specified in task {task_id}")

        # Initialize model
        base_url = os.environ.get("BASE_URL")
        model, formatter = create_model_and_formatter(MODEL_NAME, base_url=base_url, reasoning_effort=reasoning_effort, stream=stream)

        # Build memory config
        memory_class = None
        memory_config = None
        if enable_memory:
            # Use MemoryManager with CM compression
            memory_class = "memorymanager"
            mem_model_name = memory_model or MODEL_NAME
            mem_model, mem_formatter = create_model_and_formatter(mem_model_name, base_url=base_url)
            memory_config = {
                "_api_model": mem_model,
                "_api_formatter": mem_formatter,
                "sys_prompt": "",
                "debug_dir": run_output_dir,
            }
            if max_model_len is not None:
                memory_config["max_model_len"] = max_model_len
            if compress_fre is not None:
                memory_config["compress_fre"] = compress_fre
            if compress_fre_min is not None:
                memory_config["compress_fre_min"] = compress_fre_min
            if compress_fre_max is not None:
                memory_config["compress_fre_max"] = compress_fre_max
            print(f"[MCP Task {task_id}] Memory enabled (CM): model={mem_model_name}")
        else:
            # Use InMemoryMemory with debug_dir for memory.json output
            memory_class = "inmemory"
            memory_config = {"debug_dir": run_output_dir}
            print(f"[MCP Task {task_id}] InMemoryMemory with debug logging")

        # Create MCP Worker
        worker = MCPWorker(
            name="mcp_worker",
            model=model,
            formatter=formatter,
            server_configs=mcp_server_configs,
            max_iters=max_iters,
            experiment_logger=logger,
            task_id=task_id,
            memory_class=memory_class,
            memory_config=memory_config,
            tool_cache_path=tool_cache_path,
            es_fallback_config=es_fallback_config,
            skip_live_mcp_servers=skip_live_mcp_servers,
        )

        # Run task
        result = await worker.run_mcp_task(
            task_description=task_description,
        )

        result["score"] = 0.0
        result["task_id"] = task_id
        # Success = no traceback/error in result from MCPWorker
        result["success"] = "traceback" not in result and "error" not in result

        # 6-dimension benchmark evaluation (mcp-bench style, always runs for MCP tasks)
        try:
            from benchmark.llm_judge import TaskEvaluator, create_judge_provider

            bench_judge_model = JUDGE_MODEL
            print(f"[MCP Task {task_id}] 6-dim eval using judge: {bench_judge_model}")
            judge_provider = create_judge_provider(bench_judge_model)
            evaluator = TaskEvaluator(judge_provider)

            tool_calls = result.get("tool_calls", {})
            execution_results = result.get("tool_call_details")
            if not execution_results:
                execution_results = TaskEvaluator.build_execution_results_from_tool_calls(
                    tool_calls, getattr(worker, "mcp_tools", None),
                )
            available_tools = {}
            for tname, tinfo in getattr(worker, "mcp_tools", {}).items():
                available_tools[tname] = {
                    "name": tinfo.get("original_name", tname),
                    "server": tinfo.get("server", ""),
                    "description": tinfo.get("schema", {}).get("description", ""),
                    "input_schema": tinfo.get("schema", {}).get("input_schema", {}),
                }

            concrete_desc = raw_task.get("task_description", "")
            dep_analysis = raw_task.get("dependency_analysis", "")

            evaluation = await evaluator.evaluate(
                task=task_description,
                execution_results=execution_results,
                final_solution=result.get("answer", ""),
                total_rounds=result.get("total_iterations", 0),
                available_tools=available_tools,
                concrete_task_description=concrete_desc or None,
                dependency_analysis=dep_analysis or None,
            )
            if evaluation:
                result["benchmark_evaluation"] = evaluation
                print(
                    f"[MCP Task {task_id}] 6-dim scores: "
                    f"TF={evaluation.get('task_fulfillment', 0):.1f} "
                    f"GR={evaluation.get('grounding', 0):.1f} "
                    f"TA={evaluation.get('tool_appropriateness', 0):.1f} "
                    f"PA={evaluation.get('parameter_accuracy', 0):.1f} "
                    f"DA={evaluation.get('dependency_awareness', 0):.1f} "
                    f"PE={evaluation.get('parallelism_and_efficiency', 0):.1f}"
                )
        except Exception as eval_err:
            print(f"[MCP Task {task_id}] Benchmark eval failed: {eval_err}")
            import traceback as tb_mod
            tb_mod.print_exc()

        logger.save_report(result, filename="result.json")
        logger.end_run()

        return result

    except Exception as e:
        print(f"[MCP Task {task_id}] Error: {e}")
        import traceback
        traceback.print_exc()

        error_result = {
            "task_id": task_id,
            "success": False,
            "error": str(e),
            "score": 0.0,
        }
        logger.save_report(error_result, filename="result.json")
        logger.end_run()
        return error_result



# ============================================================================
# Batch Execution
# ============================================================================

async def run_batch_tasks(
    worker_type: str,
    tasks_list: List[Dict[str, Any]],
    output_dir: str = "./outputs",
    run_name: str = None,
    dataset: str = "tasks",
    timeout: Optional[int] = None,
    max_concurrent: int = 1,
    no_batch_file: bool = True,  # Always True, batch file handled by parallel_runner.py
    # BCP specific
    agent_model: str = "gpt-4o-mini",
    memory_model: Optional[str] = None,
    use_memory_manager: bool = False,
    searcher_type: str = "bm25",
    index_path: str = "indexes/bm25",
    browsecomp_path: Optional[str] = None,
    top_k: int = 5,
    snippet_max_tokens: int = 512,
    include_get_document: bool = True,
    max_iters: int = 50,
    searcher_model: str = "",
    use_bg_info: bool = False,
    stop_on_no_tool_use: bool = True,
    search_engine: str = "nlp_search",
    # Common memory configuration
    max_model_len: Optional[int] = None,
    compress_fre: Optional[int] = None,
    compress_fre_min: Optional[int] = None,
    compress_fre_max: Optional[int] = None,
    calculate_reward: bool = False,
    # Judge model
    judge_model: Optional[str] = None,
    # Tokenizer model
    tokenizer_model: Optional[str] = None,
    # Thinking
    enable_thinking: bool = True,
    reasoning_effort: Optional[str] = None,
    stream: bool = False,
    # MCP specific
    commands_file: Optional[str] = None,
    **kwargs
) -> List[Dict[str, Any]]:
    """Run multiple tasks from a list."""
    if run_name is None:
        run_name = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"Running {len(tasks_list)} tasks with run_name: {run_name}")

    semaphore = asyncio.Semaphore(max_concurrent)

    async def run_with_semaphore(task_idx: int, task: dict) -> Dict[str, Any]:
        async with semaphore:
            print(f"Running task {task_idx + 1}/{len(tasks_list)}")
            question = task.get("question") or task.get("query") or task.get("task_desc") or task.get("problem")
            ground_truth = task.get("answer") or task.get("ground_truth") or task.get("truth")
            task_id = task.get("task_id", task_idx)
            row_index = task.get("row_index", task_idx)

            try:
                if worker_type == "bcp":
                    coro = run_bcp_task(
                        question=question,
                        ground_truth=ground_truth,
                        agent_model=agent_model,
                        memory_model=memory_model,
                        use_memory_manager=use_memory_manager,
                        searcher_type=searcher_type,
                        index_path=index_path,
                        browsecomp_path=browsecomp_path,
                        output_dir=output_dir,
                        task_id=task_id,
                        row_index=row_index,
                        run_name=run_name,
                        dataset=dataset,
                        top_k=top_k,
                        snippet_max_tokens=snippet_max_tokens,
                        include_get_document=include_get_document,
                        max_iters=max_iters,
                        searcher_model=searcher_model,
                        use_bg_info=use_bg_info,
                        max_model_len=max_model_len,
                        compress_fre=compress_fre,
                        compress_fre_min=compress_fre_min,
                        compress_fre_max=compress_fre_max,
                        calculate_reward=calculate_reward,
                        stop_on_no_tool_use=stop_on_no_tool_use,
                        enable_thinking=enable_thinking,
                        judge_model=judge_model,
                        tokenizer_model=tokenizer_model,
                        reasoning_effort=reasoning_effort,
                        stream=stream,
                    )
                else:  # mcp
                    coro = run_mcp_task(
                        task=task,  # Pass full task for MCP (includes servers list)
                        agent_model=agent_model,
                        output_dir=output_dir,
                        task_id=task_id,
                        run_name=run_name,
                        dataset=dataset,
                        max_iters=max_iters,
                        judge_model=judge_model,
                        commands_file=commands_file,
                        enable_memory=use_memory_manager,
                        memory_model=memory_model,
                        max_model_len=max_model_len,
                        compress_fre=compress_fre,
                        compress_fre_min=compress_fre_min,
                        compress_fre_max=compress_fre_max,
                        tool_cache_path=kwargs.get("tool_cache_path"),
                        es_fallback_config=kwargs.get("es_fallback_config"),
                        skip_live_mcp_servers=kwargs.get("skip_live_mcp_servers"),
                        reasoning_effort=reasoning_effort,
                        stream=stream,
                    )

                if timeout:
                    result = await asyncio.wait_for(coro, timeout=timeout)
                else:
                    result = await coro

            except asyncio.TimeoutError:
                print(f"Task {task_idx + 1} timed out after {timeout} seconds")
                result = {
                    "question": question,
                    "ground_truth": ground_truth,
                    "task_id": task_id,
                    "row_index": row_index,
                    "error": f"Task timed out after {timeout} seconds",
                    "success": False,
                    "timeout": True,
                    "score": 0.0
                }
            except Exception as e:
                import traceback
                print(f"Task {task_idx + 1} failed with error: {e}")
                print(f"Task {task_idx + 1} traceback:\n{traceback.format_exc()}")
                result = {
                    "question": question,
                    "ground_truth": ground_truth,
                    "task_id": task_id,
                    "row_index": row_index,
                    "error": str(e),
                    "success": False,
                    "score": 0.0
                }

            result["task_index"] = task_idx
            return result

    # Run all tasks
    tasks_futures = [run_with_semaphore(i, task) for i, task in enumerate(tasks_list)]
    results = await asyncio.gather(*tasks_futures)

    # Note: Batch results and analysis report are generated by parallel_runner.py
    # Individual task results are saved in each task's result.json

    return list(results)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Unified Worker for BCP and MCP evaluation")

    # Worker type
    parser.add_argument("--worker-type", type=str, required=True,
                        choices=["bcp", "mcp"],
                        help="Worker type: bcp or mcp")

    # Task input (from parallel_runner.py)
    parser.add_argument("--tasks-file", type=str, required=True,
                        help="Path to JSON file containing tasks (provided by parallel_runner.py)")
    parser.add_argument("--dataset", type=str, default="tasks",
                        help="Dataset name for output directory structure")

    # Model configuration (common)
    parser.add_argument("--agent-model", type=str, default="gpt-4o-mini",
                        help="Model name for main agent")

    # BCP specific
    parser.add_argument("--memory-model", type=str, default=None,
                        help="Model name for memory manager (BCP)")
    parser.add_argument("--enable-memory", action="store_true",
                        help="Use MemoryManager instead of InMemory")
    parser.add_argument("--search-engine", type=str, default="nlp_search",
                        help="Search engine to use (default: nlp_search)")
    parser.add_argument("--searcher", type=str, default="bm25",
                        choices=["bm25", "faiss"],
                        help="Type of searcher to use (BCP)")
    parser.add_argument("--searcher-model", type=str, default="Qwen/Qwen3-Embedding-8B",
                        help="Model name for searcher (BCP)")
    parser.add_argument("--index-path", type=str, default="indexes/bm25",
                        help="Path to search index (BCP)")
    parser.add_argument("--browsecomp-path", type=str, default=None,
                        help="Path to BrowseComp-Plus installation (BCP)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of top search results (BCP)")
    parser.add_argument("--snippet-max-tokens", type=int, default=512,
                        help="Maximum tokens for document snippets (BCP)")
    parser.add_argument("--no-get-document", action="store_true",
                        help="Disable get_document tool (BCP)")
    parser.add_argument("--use-bg-info", action="store_true",
                        help="Use background information (BCP)")
    parser.add_argument("--tokenizer-model", type=str, default=None,
                        help="Model name/path for tokenizer (BCP, defaults to DEFAULT_TOKENIZER_MODEL env var)")

    # MCP specific
    parser.add_argument("--commands-file", type=str, default=None,
                        help="Path to MCP server commands JSON file (MCP)")
    parser.add_argument("--tool-cache-path", type=str, default=None,
                        help="Path to JSON file for MCP tool result cache")
    parser.add_argument("--es-fallback", action="store_true",
                        help="Enable Elasticsearch fallback for Wikipedia tools")
    parser.add_argument("--es-url", type=str, default=None,
                        help="Elasticsearch URL")
    parser.add_argument("--es-index", type=str, default=None,
                        help="Elasticsearch index name")
    parser.add_argument("--miss-log-path", type=str, default=None,
                        help="Path for cache miss JSONL log")
    parser.add_argument("--skip-live-mcp", type=str, nargs="*", default=None,
                        help="Server names to skip live MCP calls for")

    # Memory configuration (common)
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="Max model length for memory_config (overrides config value)")
    parser.add_argument("--compress-fre", type=int, default=None,
                        help="Fixed compression frequency (compress every N rounds)")
    parser.add_argument("--compress-fre-min", type=int, default=None,
                        help="Minimum compression frequency (rounds between compressions, used with --compress-fre-max)")
    parser.add_argument("--compress-fre-max", type=int, default=None,
                        help="Maximum compression frequency (rounds between compressions, used with --compress-fre-min)")
    parser.add_argument("--calculate-reward", action="store_true",
                        help="Enable reward calculation for memory compression (BCP)")

    # Execution parameters
    parser.add_argument("--max-iters", type=int, default=50,
                        help="Maximum iterations for agent")
    parser.add_argument("--max-concurrent", type=int, default=1,
                        help="Maximum concurrent tasks (for batch)")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Timeout in seconds for each task")
    parser.add_argument("--con-on-no-tool-use", action="store_true",
                        help="Continue task when no tool_use blocks found instead of stopping (default: stop)")
    parser.add_argument("--no-thinking", action="store_true",
                        help="Disable thinking/reasoning for models that support it")
    parser.add_argument("--stream", action="store_true",
                        help="Enable streaming mode for model calls")
    parser.add_argument("--reasoning-effort", type=str, default=None,
                        choices=["low", "medium", "high"],
                        help="Reasoning effort for agent model (e.g. gpt-oss-20b)")
    parser.add_argument("--judge-model", type=str, default="gpt-4o-2024-11-20",
                        help="Model name for judge (default: gpt-4o-2024-11-20)")

    # Output configuration
    parser.add_argument("--output-dir", type=str, default="./benchmark_results",
                        help="Directory for output files")
    parser.add_argument("--run-name", type=str, required=True,
                        help="Run name for output directory")

    args = parser.parse_args()

    # Load tasks from file (provided by parallel_runner.py)
    tasks_file = Path(args.tasks_file)
    if not tasks_file.exists():
        print(f"ERROR: Tasks file not found: {tasks_file}")
        sys.exit(1)

    batch_tasks = load_tasks_from_file(tasks_file)
    print(f"Loaded {len(batch_tasks)} tasks from {tasks_file}")

    if not batch_tasks:
        print("No tasks to run.")
        sys.exit(1)

    # Run batch
    asyncio.run(run_batch_tasks(
        worker_type=args.worker_type,
        tasks_list=batch_tasks,
        output_dir=args.output_dir,
        run_name=args.run_name,
        dataset=args.dataset,
        timeout=args.timeout,
        max_concurrent=args.max_concurrent,
        no_batch_file=True,  # parallel_runner.py handles batch file
        agent_model=args.agent_model,
        memory_model=args.memory_model,
        use_memory_manager=args.enable_memory,
        searcher_type=args.searcher,
        index_path=args.index_path,
        browsecomp_path=args.browsecomp_path,
        top_k=args.top_k,
        snippet_max_tokens=args.snippet_max_tokens,
        include_get_document=not args.no_get_document,
        max_iters=args.max_iters,
        searcher_model=args.searcher_model,
        use_bg_info=args.use_bg_info,
        stop_on_no_tool_use=not args.con_on_no_tool_use,
        enable_thinking=not args.no_thinking,
        reasoning_effort=args.reasoning_effort,
        stream=args.stream,
        search_engine=args.search_engine,
        max_model_len=args.max_model_len,
        compress_fre=args.compress_fre,
        compress_fre_min=args.compress_fre_min,
        compress_fre_max=args.compress_fre_max,
        calculate_reward=args.calculate_reward,
        judge_model=args.judge_model,
        tokenizer_model=args.tokenizer_model,
        commands_file=args.commands_file,
        tool_cache_path=getattr(args, 'tool_cache_path', None),
        es_fallback_config={
            "enabled": True,
            "es_url": args.es_url,
            "es_index": args.es_index,
            "miss_log_path": args.miss_log_path,
        } if getattr(args, 'es_fallback', False) else None,
        skip_live_mcp_servers=getattr(args, 'skip_live_mcp', None),
    ))


if __name__ == "__main__":
    main()
