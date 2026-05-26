#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCP-Bench Benchmark Runner.

Runs MCP-Bench evaluation pipeline with 6-dimension LLM judge scoring.
Ported from the mcp-bench project, uses MCPWorker for task execution.

Pipeline:
  1. Load tasks from MCP-Bench task files
  2. Connect to MCP servers and run MCPWorker per task
  3. Evaluate with 6-dimension LLM judge
  4. Store results in structured format
  5. Aggregate and report metrics

USAGE:
    # Single task
    python run_mcp_benchmark.py \
        --task-file ../../mcp-bench/tasks/mcpbench_tasks_academic_no_bio.json \
        --task-id wikipedia_000 \
        --agent-model qwen3-max

    # Batch (all tasks in a range)
    python run_mcp_benchmark.py \
        --task-file ../../mcp-bench/tasks/mcpbench_tasks_academic_no_bio.json \
        --begin 0 --end 10 \
        --agent-model qwen3-max \
        --parallel 4

    # With memory and custom judge
    python run_mcp_benchmark.py \
        --task-file ../../mcp-bench/tasks/mcpbench_tasks_academic_no_bio.json \
        --begin 0 --end 50 \
        --agent-model qwen3-max \
        --enable-memory --memory-model gpt-5-2025-08-07 \
        --judge-model gpt-5-2025-08-07 \
        --parallel 4

Located in: as1/examples/
"""

import os
import sys
import json
import argparse
import asyncio
import time
import traceback
import subprocess
import threading
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ============================================================================
# Task Loading
# ============================================================================

def load_mcpbench_tasks(task_file: str) -> List[Dict[str, Any]]:
    """Load and flatten tasks from MCP-Bench task file.

    Supports:
      - server_tasks format: {server_tasks: [{server_name, tasks: [...], servers: [...]}]}
      - direct list format
    """
    with open(task_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    tasks = []
    if isinstance(data, dict) and "server_tasks" in data:
        for server_group in data["server_tasks"]:
            servers = server_group.get("servers", [server_group.get("server_name", "")])
            for task in server_group.get("tasks", []):
                task["servers"] = servers
                tasks.append(task)
    elif isinstance(data, list):
        tasks = data
    else:
        tasks = [data]

    return tasks


# ============================================================================
# Single Task Execution + Evaluation
# ============================================================================

async def run_and_evaluate_task(
    task: Dict[str, Any],
    agent_model: str,
    judge_model: str,
    commands_file: Optional[str] = None,
    enable_memory: bool = False,
    memory_model: Optional[str] = None,
    max_iters: int = 50,
    max_model_len: Optional[int] = None,
    compress_fre: Optional[int] = None,
    output_dir: str = "./benchmark_results",
    run_name: str = "mcp_bench",
    use_fuzzy_description: bool = True,
    enable_judge_stability: bool = False,
    tokenizer_model: Optional[str] = None,
    tool_cache_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a single MCP-Bench task and evaluate with 6-dim LLM judge.

    Returns:
        Dict with execution result, evaluation scores, and metadata.
    """
    from asio.agent.mcp_worker import MCPWorker
    from asio.logger import ExperimentLogger
    from agentscope.model import OpenAIChatModel, enable_auto_llm_logging
    from agentscope.formatter import OpenAIChatFormatter
    from benchmark.llm_judge import TaskEvaluator, create_judge_provider

    # Import model creation from worker_standalone
    from worker_standalone import create_model_and_formatter

    task_id = task.get("task_id", "unknown")
    task_description = task.get("fuzzy_description", "") if use_fuzzy_description else ""
    if not task_description:
        task_description = task.get("task_description", "")
    concrete_task_description = task.get("task_description", "")
    expected_output = task.get("expected_output", "") or task.get("answer", "")
    dependency_analysis = task.get("dependency_analysis", "")
    servers = task.get("servers", [])

    # Setup logger
    logger = ExperimentLogger(
        base_dir=output_dir,
        test_mem_config=run_name,
        dataset="benchmark",
    )
    logger.start_run(str(task_id))
    enable_auto_llm_logging(logger_instance=logger)
    run_output_dir = logger.current_run_dir

    result_dict = {
        "task_id": task_id,
        "status": "failed",
        "execution_time": 0,
        "total_rounds": 0,
        "total_tool_calls": 0,
        "evaluation": None,
    }

    try:
        base_url = os.environ.get("BASE_URL")

        # Build MCP server configs
        mcp_server_configs = _build_server_configs(servers, commands_file)
        if not mcp_server_configs:
            raise ValueError(f"No MCP servers for task {task_id}")

        # Initialize agent model
        model, formatter = create_model_and_formatter(agent_model, base_url=base_url)

        # Build memory config
        memory_class = "inmemory"
        memory_config = {"debug_dir": run_output_dir}
        if enable_memory:
            memory_class = "memorymanager"
            mem_model_name = memory_model or agent_model
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

        # Create and run MCPWorker
        worker = MCPWorker(
            name="mcp_worker",
            model=model,
            formatter=formatter,
            server_configs=mcp_server_configs,
            max_iters=max_iters,
            experiment_logger=logger,
            task_id=str(task_id),
            memory_class=memory_class,
            memory_config=memory_config,
            tokenizer_model=tokenizer_model,
            tool_cache_path=tool_cache_path,
        )

        start_time = time.time()
        worker_result = await worker.run_mcp_task(
            task_description=task_description,
            ground_truth=expected_output or None,
        )
        execution_time = time.time() - start_time

        final_answer = worker_result.get("answer", "No answer generated")
        tool_calls = worker_result.get("tool_calls", {})
        total_iters = worker_result.get("total_iterations", 0)

        # Use per-call details (with parameters) if available, else fall back to counts
        execution_results = worker_result.get("tool_call_details")
        if not execution_results:
            execution_results = TaskEvaluator.build_execution_results_from_tool_calls(
                tool_calls, getattr(worker, "mcp_tools", None),
            )

        # Build available_tools dict for evaluator
        available_tools = {}
        for tname, tinfo in getattr(worker, "mcp_tools", {}).items():
            available_tools[tname] = {
                "name": tinfo.get("original_name", tname),
                "server": tinfo.get("server", ""),
                "description": tinfo.get("schema", {}).get("description", ""),
                "input_schema": tinfo.get("schema", {}).get("input_schema", {}),
            }

        result_dict.update({
            "status": "completed",
            "execution_time": execution_time,
            "total_rounds": total_iters,
            "total_tool_calls": sum(tool_calls.values()),
            "final_solution": final_answer,
            "tool_calls": tool_calls,
            "execution_results": execution_results,
            "available_tools_count": len(available_tools),
        })

        # ---- 6-Dimension LLM Judge Evaluation ----
        judge_model_name = judge_model or "gpt-5-2025-08-07"
        print(f"[{task_id}] Running 6-dim LLM judge evaluation (judge: {judge_model_name})...")
        judge_provider = create_judge_provider(judge_model_name)
        evaluator = TaskEvaluator(judge_provider, enable_judge_stability)

        eval_start = time.time()
        evaluation = await evaluator.evaluate(
            task=task_description,
            execution_results=execution_results,
            final_solution=final_answer,
            total_rounds=total_iters,
            available_tools=available_tools,
            concrete_task_description=concrete_task_description if use_fuzzy_description else None,
            dependency_analysis=dependency_analysis or None,
        )
        eval_time = time.time() - eval_start

        if evaluation:
            result_dict["evaluation"] = evaluation
            result_dict["evaluation_time"] = eval_time
            # Print key scores
            print(
                f"[{task_id}] Judge scores: "
                f"TF={evaluation.get('task_fulfillment', 0):.1f} "
                f"GR={evaluation.get('grounding', 0):.1f} "
                f"TA={evaluation.get('tool_appropriateness', 0):.1f} "
                f"PA={evaluation.get('parameter_accuracy', 0):.1f} "
                f"DA={evaluation.get('dependency_awareness', 0):.1f} "
                f"PE={evaluation.get('parallelism_and_efficiency', 0):.1f}"
            )
        else:
            print(f"[{task_id}] Judge evaluation failed")

        # Also run simple correctness judge if expected_output is available
        if expected_output:
            from asio.utils.judge import judge_result
            simple_judge = await judge_result(
                question=task_description,
                predicted_answer=final_answer,
                expected_answer=expected_output,
                judge_model=judge_model_name,
            )
            result_dict["correctness_score"] = float(simple_judge.get("score", 0.0))
            result_dict["correctness_explanation"] = simple_judge.get("explanation", "")

        # Save results
        logger.save_report(result_dict, filename="result.json")
        logger.end_run()

        return result_dict

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[{task_id}] Error: {e}")
        result_dict["error"] = str(e)
        result_dict["traceback"] = tb
        logger.save_report(result_dict, filename="result.json")
        logger.end_run()
        return result_dict


def _build_server_configs(
    servers: List[str], commands_file: Optional[str],
) -> List[Dict[str, Any]]:
    """Build MCP server configs from commands file for the given server list."""
    commands = {}
    commands_base_dir = None

    if commands_file and os.path.exists(commands_file):
        commands_base_dir = os.path.dirname(os.path.abspath(commands_file))
        with open(commands_file, "r") as f:
            commands = json.load(f)

    configs = []
    for server_name in servers:
        if server_name in commands:
            config = dict(commands[server_name])
            config["name"] = server_name
            if commands_base_dir:
                mcp_bench_root = os.path.dirname(commands_base_dir)
                if "cwd" in config:
                    cwd = config["cwd"]
                    if cwd.startswith("../"):
                        config["cwd"] = os.path.join(commands_base_dir, cwd[3:])
                    elif not os.path.isabs(cwd):
                        config["cwd"] = os.path.normpath(os.path.join(commands_base_dir, cwd))
                cmd_key = "cmd" if "cmd" in config else "command"
                if cmd_key in config and isinstance(config[cmd_key], str):
                    if config[cmd_key].startswith(".venv/"):
                        config[cmd_key] = os.path.join(mcp_bench_root, config[cmd_key])
            configs.append(config)
        else:
            print(f"WARNING: Server '{server_name}' not found in commands file")
            configs.append({"name": server_name, "transport": "stdio"})

    return configs


# ============================================================================
# Batch Execution
# ============================================================================

async def run_batch_benchmark(
    tasks: List[Dict[str, Any]],
    args,
) -> List[Dict[str, Any]]:
    """Run benchmark for a batch of tasks sequentially."""
    results = []
    for i, task in enumerate(tasks):
        task_id = task.get("task_id", f"task_{i}")
        print(f"\n[{time.strftime('%H:%M:%S')}] === Task {i+1}/{len(tasks)}: {task_id} ===")

        result = await run_and_evaluate_task(
            task=task,
            agent_model=args.agent_model,
            judge_model=args.judge_model or args.agent_model,
            commands_file=args.commands_file,
            enable_memory=args.enable_memory,
            memory_model=args.memory_model,
            max_iters=args.max_iters,
            max_model_len=args.max_model_len,
            compress_fre=args.compress_fre,
            output_dir=args.output_dir,
            run_name=args.run_name,
            use_fuzzy_description=not args.use_concrete_description,
            enable_judge_stability=args.judge_stability,
            tokenizer_model=args.tokenizer_model,
            tool_cache_path=args.tool_cache_path,
        )
        results.append(result)

    return results


def run_parallel_benchmark(tasks: List[Dict[str, Any]], args):
    """Run benchmark in parallel using subprocess workers."""
    from benchmark.results_aggregator import ResultsAggregator
    from benchmark.results_storage import BenchmarkResultsStorage

    num_groups = min(args.parallel, len(tasks))
    groups = []
    for i in range(num_groups):
        groups.append([])
    for i, task in enumerate(tasks):
        groups[i % num_groups].append(task)
    groups = [g for g in groups if g]

    print(f"[{time.strftime('%H:%M:%S')}] Splitting {len(tasks)} tasks into {len(groups)} parallel groups")

    # Write each group to a temp file and spawn a subprocess
    temp_files = []
    threads = []
    thread_results = {}

    def run_worker(process_id, task_group):
        tf = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", prefix="bench_tasks_", delete=False, encoding="utf-8",
        )
        # Write tasks as a JSON file with server_tasks format
        server_groups = {}
        for t in task_group:
            skey = ",".join(t.get("servers", []))
            if skey not in server_groups:
                server_groups[skey] = {"server_name": skey, "tasks": [], "servers": t.get("servers", [])}
            server_groups[skey]["tasks"].append(t)
        output = {"server_tasks": list(server_groups.values())}
        json.dump(output, tf, ensure_ascii=False)
        tf.close()
        temp_files.append(tf.name)

        cmd = [
            sys.executable, str(Path(__file__)),
            "--task-file", tf.name,
            "--agent-model", args.agent_model,
            "--output-dir", args.output_dir,
            "--run-name", args.run_name,
            "--max-iters", str(args.max_iters),
        ]
        if args.judge_model:
            cmd.extend(["--judge-model", args.judge_model])
        if args.commands_file:
            cmd.extend(["--commands-file", args.commands_file])
        if args.enable_memory:
            cmd.append("--enable-memory")
        if args.memory_model:
            cmd.extend(["--memory-model", args.memory_model])
        if args.max_model_len is not None:
            cmd.extend(["--max-model-len", str(args.max_model_len)])
        if args.compress_fre is not None:
            cmd.extend(["--compress-fre", str(args.compress_fre)])
        if args.use_concrete_description:
            cmd.append("--use-concrete-description")
        if args.judge_stability:
            cmd.append("--judge-stability")
        if args.tokenizer_model:
            cmd.extend(["--tokenizer-model", args.tokenizer_model])
        if args.tool_cache_path:
            cmd.extend(["--tool-cache-path", args.tool_cache_path])

        print(f"[{time.strftime('%H:%M:%S')}] Process {process_id} starting ({len(task_group)} tasks)")
        try:
            result = subprocess.run(cmd, capture_output=False)
            thread_results[process_id] = result.returncode
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] Process {process_id} failed: {e}")
            thread_results[process_id] = 1

    for i, group in enumerate(groups):
        t = threading.Thread(target=run_worker, args=(i + 1, group))
        threads.append(t)
        t.start()
        time.sleep(1)

    for t in threads:
        t.join()

    # Cleanup
    for tf_path in temp_files:
        try:
            os.unlink(tf_path)
        except Exception:
            pass

    # Collect results from output directory
    results = _collect_results(args.output_dir, args.run_name)

    # Aggregate and report
    if results:
        aggregator = ResultsAggregator()
        agg = aggregator.aggregate_results(results)
        print(aggregator.format_summary(agg))

        storage = BenchmarkResultsStorage(
            base_dir=args.output_dir,
            model_name=args.agent_model,
            timestamp=args.run_name,
        )
        storage.save_aggregated_results(agg)

    # Print parallel summary
    success = sum(1 for c in thread_results.values() if c == 0)
    fail = len(thread_results) - success
    print(f"\nParallel: {success}/{len(thread_results)} processes succeeded, {fail} failed")

    return results


def _collect_results(output_dir: str, run_name: str) -> List[Dict[str, Any]]:
    """Collect all result.json files from benchmark output."""
    results = []
    base = Path(output_dir) / "benchmark" / run_name

    if not base.exists():
        return results

    for result_file in base.rglob("result.json"):
        try:
            with open(result_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            results.append(data)
        except Exception:
            pass

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="MCP-Bench Benchmark Runner with 6-dim LLM Judge")

    # Task input
    parser.add_argument("--task-file", type=str, required=True,
                        help="Path to MCP-Bench task JSON file")
    parser.add_argument("--task-id", type=str, default=None,
                        help="Run a single task by ID")
    parser.add_argument("--begin", type=int, default=0,
                        help="Start index for batch run (default: 0)")
    parser.add_argument("--end", type=int, default=None,
                        help="End index for batch run (exclusive)")

    # Model configuration
    parser.add_argument("--agent-model", type=str, default="qwen3-max",
                        help="Agent model name")
    parser.add_argument("--judge-model", type=str, default="gpt-5-2025-08-07",
                        help="Judge model name (default: gpt-5-2025-08-07)")
    parser.add_argument("--memory-model", type=str, default=None,
                        help="Memory model name")
    parser.add_argument("--enable-memory", action="store_true",
                        help="Enable MemoryManager")
    parser.add_argument("--tokenizer-model", type=str, default=None,
                        help="Tokenizer model name/path")

    # MCP configuration
    parser.add_argument("--commands-file", type=str, default=None,
                        help="Path to MCP commands JSON file")

    # Memory configuration
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--compress-fre", type=int, default=None)

    # Execution
    parser.add_argument("--max-iters", type=int, default=50,
                        help="Max iterations per task")
    parser.add_argument("--parallel", "-p", type=int, default=1,
                        help="Number of parallel processes (default: 1)")

    # Evaluation
    parser.add_argument("--judge-stability", action="store_true",
                        help="Enable judge stability (5 evaluations per task)")
    parser.add_argument("--use-concrete-description", action="store_true",
                        help="Use concrete task_description instead of fuzzy_description")

    # Output
    parser.add_argument("--output-dir", type=str, default="./benchmark_results",
                        help="Output directory")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Run name for output directory")

    # Cache configuration
    parser.add_argument("--tool-cache-path", type=str, default=None,
                        help="Path to JSON file for MCP tool result cache")

    args = parser.parse_args()

    if args.run_name is None:
        args.run_name = f"bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Load tasks
    all_tasks = load_mcpbench_tasks(args.task_file)
    print(f"Loaded {len(all_tasks)} tasks from {args.task_file}")

    # Filter tasks
    if args.task_id:
        tasks = [t for t in all_tasks if t.get("task_id") == args.task_id]
        if not tasks:
            print(f"Task ID '{args.task_id}' not found")
            sys.exit(1)
    else:
        end = args.end or len(all_tasks)
        tasks = all_tasks[args.begin:end]

    print(f"Running {len(tasks)} tasks with model={args.agent_model}, judge={args.judge_model or args.agent_model}")

    if args.parallel > 1 and len(tasks) > 1:
        # Parallel mode
        results = run_parallel_benchmark(tasks, args)
    else:
        # Sequential mode
        results = asyncio.run(run_batch_benchmark(tasks, args))

        # Aggregate and report
        from benchmark.results_aggregator import ResultsAggregator
        from benchmark.results_storage import BenchmarkResultsStorage

        aggregator = ResultsAggregator()
        agg = aggregator.aggregate_results(results)
        print(aggregator.format_summary(agg))

        storage = BenchmarkResultsStorage(
            base_dir=args.output_dir,
            model_name=args.agent_model,
            timestamp=args.run_name,
        )
        storage.save_aggregated_results(agg)
        storage.save_meta_info({
            "agent_model": args.agent_model,
            "judge_model": args.judge_model or args.agent_model,
            "task_file": args.task_file,
            "total_tasks": len(tasks),
            "enable_memory": args.enable_memory,
            "max_iters": args.max_iters,
            "timestamp": datetime.now().isoformat(),
        })

        # Save per-task evaluations
        for r in results:
            tid = r.get("task_id", "unknown")
            storage.save_task_result(str(tid), r)
            if r.get("evaluation"):
                storage.save_task_evaluation(str(tid), r["evaluation"])

        print(f"\nResults saved to: {storage.get_run_directory()}")


if __name__ == "__main__":
    main()
