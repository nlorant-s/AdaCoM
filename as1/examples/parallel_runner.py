#!/usr/bin/env python3
"""
Unified Parallel Runner for BrowseComp-Plus (BCP) and MCP-Bench Evaluation.

Features:
- Parallel task execution with thread-based coordination
- Resume logic (scan existing results, skip completed tasks)
- Error pattern-based retry
- Result collection and analysis report generation
- Uses query_id / task_id from dataset as task identifier

USAGE:
    # BCP mode - run tasks by row range
    python parallel_runner.py --worker-type bcp -t -1 -b 0 -e 100 -p 20 --agent-model qwen3-max

    # BCP mode - run specific tasks by query_id
    python parallel_runner.py --worker-type bcp -t "query_001,query_002,query_003" -p 20

    # MCP mode
    python parallel_runner.py --worker-type mcp -t -1 -b 0 -e 100 -p 20 \
        --agent-model qwen3-max --commands-file <path-to-mcp-commands.json>

    # Resume failed tasks
    python parallel_runner.py --worker-type bcp --resume --run-name batch_20251230_100000
"""

import argparse
import subprocess
import sys
import os
import time
import threading
import json
import tempfile
from typing import List, Optional, Dict, Any
from pathlib import Path
from collections import defaultdict

import numpy as np


def calculate_repeat_metrics(
    all_repeat_results: Dict[str, List[Dict[str, Any]]],
    repeat_times: int,
) -> Dict[str, float]:
    """Calculate mean@k and best@k from repeated experiment runs.

    Args:
        all_repeat_results: {task_id: [result_run_0, result_run_1, ...]}
        repeat_times: total number of repeats (k)

    Returns:
        Dict with metrics:
        - mean@k: average accuracy across k runs
        - best@k: fraction of tasks solved in at least one of k runs
    """
    total_tasks = len(all_repeat_results)
    if total_tasks == 0:
        return {}

    # Group scores by run index: run_scores[i] = list of scores for run i
    run_scores: Dict[int, List[float]] = defaultdict(list)
    for task_id, results in all_repeat_results.items():
        for run_idx, result in enumerate(results):
            run_scores[run_idx].append(float(result.get('score', 0.0)))

    # Per-run accuracy
    run_accuracies = []
    for run_idx in range(repeat_times):
        scores = run_scores.get(run_idx, [])
        if scores:
            run_accuracies.append(sum(1 for s in scores if s > 0) / total_tasks)
        else:
            run_accuracies.append(0.0)

    result = {}
    # mean@k and best@k for k = 1 to repeat_times
    for k in range(1, repeat_times + 1):
        # mean@k: average accuracy of the first k runs
        mean_k = float(np.mean(run_accuracies[:k]))
        result[f"mean@{k}"] = mean_k

        # best@k: for each task, if any of the first k runs scored > 0, count as correct
        best_correct = 0
        for task_id, results in all_repeat_results.items():
            first_k_scores = [float(results[i].get('score', 0.0)) for i in range(min(k, len(results)))]
            if any(s > 0 for s in first_k_scores):
                best_correct += 1
        result[f"best@{k}"] = best_correct / total_tasks

    return result


# Error patterns to trigger retry
RETRY_ERROR_PATTERNS = [
    "Error code: 429",
]


def load_dataset(data_path: Path, file_format: str = None) -> List[Dict[str, Any]]:
    """Load dataset from JSON or JSONL file.

    Returns a list of task dicts with 'task_id' (query_id) as the primary identifier.
    """
    tasks = []

    if file_format == "jsonl" or str(data_path).endswith(".jsonl"):
        with open(data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    entry = json.loads(line.strip())
                    question = entry.get("query") or entry.get("ques") or entry.get("question", "")
                    answer = entry.get("answer") or entry.get("Final answer") or entry.get("ground_truth", "")
                    task_id = entry.get("query_id") or entry.get("task_id") or str(line_num)

                    tasks.append({
                        "question": question,
                        "answer": answer,
                        "task_id": str(task_id),
                        "line_number": line_num,
                        "raw": entry
                    })
                except Exception as e:
                    print(f"WARNING: Failed to parse line {line_num}: {e}")
    else:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, dict) and "server_tasks" in data:
            # MCP-Bench nested format: {server_tasks: [{server_name, tasks: [...], servers: [...]}]}
            flat_idx = 0
            for server_group in data["server_tasks"]:
                servers = server_group.get("servers", [])
                for entry in server_group.get("tasks", []):
                    task_id = entry.get("task_id") or str(flat_idx)
                    tasks.append({
                        "question": entry.get("fuzzy_description") or entry.get("task_description", ""),
                        "answer": entry.get("answer", ""),
                        "task_id": str(task_id),
                        "line_number": flat_idx + 1,
                        "servers": servers,
                        "raw": entry,
                    })
                    flat_idx += 1

        elif isinstance(data, list):
            for idx, entry in enumerate(data):
                question = entry.get("query") or entry.get("ques") or entry.get("question") or entry.get("problem", "")
                answer = entry.get("answer") or entry.get("Final answer") or entry.get("ground_truth", "")
                task_id = entry.get("query_id") or entry.get("task_id") or str(idx)

                tasks.append({
                    "question": question,
                    "answer": answer,
                    "task_id": str(task_id),
                    "line_number": idx + 1,
                    "raw": entry
                })

    return tasks


def find_data_file(args) -> Optional[Path]:
    """Find the data file based on arguments.

    Data file search order:
    1. If --data-dir is specified, look for {split}.jsonl or {split}.json in that directory
    2. Otherwise, look in project_root/data/ for {split}.jsonl or {split}.json
    """
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent

    if args.data_dir:
        data_dir = Path(args.data_dir)
        # If data_dir is a file directly, return it
        if data_dir.is_file():
            return data_dir
    else:
        # Default: project_root/data/
        data_dir = project_root / "data" / args.dataset

    if not data_dir.exists():
        return None

    # Try to find data file
    for ext in [".jsonl", ".json"]:
        candidate = data_dir / f"{args.split}{ext}"
        if candidate.exists():
            return candidate

    return None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Unified Parallel Runner for BCP and MCP evaluation'
    )

    # Worker type
    parser.add_argument('--worker-type', required=True,
                        choices=['bcp', 'mcp'],
                        help='Worker type: bcp or mcp')

    # Task configuration
    parser.add_argument('-t', '--task-id',
                        help='Query IDs to run: comma-separated query_ids (e.g., "qid_001,qid_002") '
                             'or "-1" to use row range with --begin/--end. Required unless --resume is used.')
    parser.add_argument('-p', '--parallel', type=int, default=1,
                        help='Number of parallel processes (default: 1)')
    parser.add_argument('-s', '--split', default='test',
                        choices=['train', 'test'],
                        help='Data split: train or test (default: test)')
    parser.add_argument('-b', '--begin', type=int, default=0,
                        help='Start row index when using -t -1, inclusive (default: 0)')
    parser.add_argument('-e', '--end', type=int, default=10,
                        help='End row index when using -t -1, exclusive (default: 10)')
    parser.add_argument('--data-dir', type=str,
                        help='Path to data directory (auto-detected if not specified)')

    # Resume / Rerun configuration
    parser.add_argument('--resume', action='store_true',
                        help='Resume/Rerun failed tasks in the directory specified by --run-name.')
    parser.add_argument('--max-retries', type=int, default=3,
                        help='Maximum number of retries for failed tasks (default: 3).')

    # Output configuration
    parser.add_argument('--run-name', type=str,
                        help='Custom name for the output directory. Required if --resume is used.')
    parser.add_argument('-o', '--output-dir',
                        default=os.environ.get('AS1_BENCHMARK_RESULTS_DIR', './as_1benchmark_results'),
                        help='Output directory (default: $AS1_BENCHMARK_RESULTS_DIR or ./as1_benchmark_results)')

    # Model configuration (common)
    parser.add_argument('-m', '--agent-model', default='qwen3-max',
                        help='Agent model name (default: qwen3-max)')

    # Judge model (common)
    parser.add_argument('-J', '--judge-model', default='gpt-4o-2024-11-20',
                        help='Judge model for evaluation (default: gpt-4o-2024-11-20). '
                             'Supports: vertex_ai.claude-opus-4-6, vertex_ai.gemini-3.1-pro-preview, etc.')

    # Common agent/memory configuration
    parser.add_argument('-M', '--memory-model', default='gpt-5-2025-08-07',
                        help='Memory model name (default: gpt-5-2025-08-07)')
    parser.add_argument('--enable-memory', action='store_true',
                        help='Enable memory manager')
    parser.add_argument('-i', '--max-iters', type=int, default=50,
                        help='Max iterations (default: 50)')
    parser.add_argument('--con-on-no-tool-use', action='store_true',
                        help='Continue task when no tool_use blocks found instead of stopping (default: stop)')
    parser.add_argument('--no-thinking', action='store_true',
                        help='Disable thinking/reasoning for models that support it (e.g. qwen3-max)')
    parser.add_argument('--stream', action='store_true',
                        help='Enable streaming mode for model calls')
    parser.add_argument('--reasoning-effort', type=str, default=None,
                        choices=['low', 'medium', 'high'],
                        help='Reasoning effort for agent model (e.g. gpt-oss-20b). Options: low, medium, high')
    parser.add_argument('--search-engine', type=str, default='nlp_search',
                        help='Search engine to use (default: nlp_search)')

    # BCP specific
    parser.add_argument('-S', '--searcher', default='bm25',
                        choices=['bm25', 'faiss'],
                        help='Searcher type (default: bm25) (BCP)')
    parser.add_argument('--searcher-model', type=str, default="Qwen/Qwen3-Embedding-8B",
                        help='Model name for searcher (BCP)')
    parser.add_argument('-I', '--index-path', type=str,
                        help='Path to search index (BCP)')
    parser.add_argument('-k', '--top-k', type=int, default=5,
                        help='Top K results (default: 5) (BCP)')
    parser.add_argument('--use-bg-info', action='store_true',
                        help='Use background information (BCP)')
    parser.add_argument('--tokenizer-model', type=str, default=None,
                        help='Model name/path for tokenizer (BCP, defaults to DEFAULT_TOKENIZER_MODEL env var)')

    # MCP specific
    parser.add_argument('--commands-file', type=str, default=None,
                        help='Path to MCP server commands JSON file (MCP)')
    parser.add_argument('--tool-cache-path', type=str, default=None,
                        help='Path to JSON file for MCP tool result cache')

    parser.add_argument('-d', '--dataset', default=None,
                        help='Dataset name (derived from --data-dir if not specified)')

    # Memory configuration (common)
    parser.add_argument('--max-model-len', type=int, default=None,
                        help='Max model length for memory_config (overrides config value)')
    parser.add_argument('--compress-fre', type=int, default=None,
                        help='Fixed compression frequency (compress every N rounds)')
    parser.add_argument('--compress-fre-min', type=int, default=None,
                        help='Minimum compression frequency (rounds between compressions, used with --compress-fre-max)')
    parser.add_argument('--compress-fre-max', type=int, default=None,
                        help='Maximum compression frequency (rounds between compressions, used with --compress-fre-min)')
    parser.add_argument('--calculate-reward', action='store_true',
                        help='Enable reward calculation for memory compression (BCP)')

    # Repeat configuration
    parser.add_argument('--repeat-times', type=int, default=1,
                        help='Number of times to repeat each task for mean@k/best@k metrics (default: 1)')

    # Execution configuration
    parser.add_argument('--timeout', type=int, default=12000,
                        help='Timeout in seconds per task (default: 12000)')

    # ES fallback configuration
    parser.add_argument('--es-fallback', action='store_true',
                        help='Enable Elasticsearch fallback for Wikipedia tools')
    parser.add_argument('--es-url', type=str, default=None,
                        help='Elasticsearch URL')
    parser.add_argument('--es-index', type=str, default=None,
                        help='Elasticsearch index name')
    parser.add_argument('--miss-log-path', type=str, default=None,
                        help='Path for cache miss JSONL log')
    parser.add_argument('--skip-live-mcp', type=str, nargs='*', default=None,
                        help='Server names to skip live MCP calls for')

    return parser.parse_args()


def split_tasks(tasks: List[Dict[str, Any]], num_groups: int) -> List[List[Dict[str, Any]]]:
    """Split tasks into groups for parallel execution."""
    if num_groups <= 1:
        return [tasks]

    group_size = len(tasks) // num_groups
    remainder = len(tasks) % num_groups

    groups = []
    start = 0

    for i in range(num_groups):
        current_group_size = group_size + (1 if i < remainder else 0)
        end = start + current_group_size

        if start < len(tasks):
            groups.append(tasks[start:end])
        start = end

    groups = [g for g in groups if g]
    return groups


def save_analysis_report(all_results: List[Dict[str, Any]], output_file: Path, global_repeat_metrics: Optional[Dict[str, float]] = None):
    """Generate and save analysis report with comprehensive statistics."""
    total_tasks = len(all_results)

    success_count = sum(1 for r in all_results if r.get('success', False))
    failure_count = total_tasks - success_count

    scores = [float(r.get('score', 0.0)) for r in all_results]
    total_score = sum(scores)
    avg_score = total_score / total_tasks if total_tasks > 0 else 0.0
    correct_count = sum(1 for s in scores if s >= 1.0)

    iterations = [r.get('total_iterations', 0) for r in all_results if r.get('total_iterations')]
    avg_iterations = sum(iterations) / len(iterations) if iterations else 0.0
    max_iterations = max(iterations) if iterations else 0
    min_iterations = min(iterations) if iterations else 0

    total_search_calls = sum(r.get('tool_calls', {}).get('search', 0) for r in all_results)
    total_get_doc_calls = sum(r.get('tool_calls', {}).get('get_document', 0) for r in all_results)
    total_visit_calls = sum(r.get('tool_calls', {}).get('visit_webpage', 0) for r in all_results)

    # Duration stats
    durations = [r.get('duration_seconds', 0) for r in all_results if r.get('duration_seconds')]
    avg_duration = sum(durations) / len(durations) if durations else 0.0

    # Visit webpage timing stats
    visit_times = [r.get('avg_visit_time', 0) for r in all_results if r.get('avg_visit_time')]
    avg_visit_time = sum(visit_times) / len(visit_times) if visit_times else 0.0

    # Jina fallback stats
    total_jina_fallbacks = sum(r.get('jina_fallback_count', 0) for r in all_results)

    # 6-dimension benchmark evaluation metrics (MCP-Bench style)
    bench_evals = [r.get('benchmark_evaluation', {}) for r in all_results if r.get('benchmark_evaluation')]
    bench_metrics = {}
    if bench_evals:
        dim_keys = [
            'task_fulfillment', 'grounding', 'tool_appropriateness',
            'parameter_accuracy', 'dependency_awareness', 'parallelism_and_efficiency',
        ]
        for key in dim_keys:
            vals = [e.get(key, 0) for e in bench_evals if e.get(key) is not None]
            bench_metrics[f'avg_{key}'] = sum(vals) / len(vals) if vals else 0.0
        all_combined = []
        for e in bench_evals:
            scores = [e.get(k, 0) for k in dim_keys if e.get(k) is not None]
            if scores:
                all_combined.append(sum(scores) / len(scores))
        bench_metrics['avg_combined'] = sum(all_combined) / len(all_combined) if all_combined else 0.0
        bench_metrics['evaluated_count'] = len(bench_evals)

    # Use task_id (query_id) as the primary identifier
    task_ids = sorted([str(r.get('task_id', '')) for r in all_results])
    failed_task_ids = sorted([str(r.get('task_id', '')) for r in all_results if not r.get('success', False)])

    report = {
        "summary": {
            "total_tasks": total_tasks,
            "success_count": success_count,
            "failure_count": failure_count,
            "execution_success_rate": success_count / total_tasks if total_tasks > 0 else 0.0,
        },
        "scoring": {
            "total_score": total_score,
            "average_score": avg_score,
            "correct_count": correct_count,
            "accuracy": correct_count / total_tasks if total_tasks > 0 else 0.0,
        },
        "iterations": {
            "average": avg_iterations,
            "max": max_iterations,
            "min": min_iterations,
        },
        "duration": {
            "average_seconds": avg_duration,
            "total_seconds": sum(durations),
        },
        "tool_calls": {
            "total_search_calls": total_search_calls,
            "total_get_document_calls": total_get_doc_calls,
            "total_visit_webpage_calls": total_visit_calls,
            "avg_search_per_task": total_search_calls / total_tasks if total_tasks > 0 else 0.0,
            "avg_get_document_per_task": total_get_doc_calls / total_tasks if total_tasks > 0 else 0.0,
            "avg_visit_webpage_per_task": total_visit_calls / total_tasks if total_tasks > 0 else 0.0,
        },
        "visit_webpage": {
            "avg_visit_time_seconds": avg_visit_time,
            "total_jina_fallbacks": total_jina_fallbacks,
        },
        "task_ids": task_ids,
        "failed_task_ids": failed_task_ids,
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    if bench_metrics:
        report["benchmark_evaluation"] = bench_metrics

    if global_repeat_metrics:
        report["repeat_metrics"] = global_repeat_metrics

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Analysis report saved to {output_file}")

        print(f"\n{'='*60}")
        print(f"ANALYSIS REPORT SUMMARY")
        print(f"{'='*60}")
        print(f"Total Tasks: {total_tasks}")
        print(f"Success: {success_count} | Failure: {failure_count}")
        print(f"Execution Success Rate: {report['summary']['execution_success_rate']:.2%}")
        print(f"Correct Answers: {correct_count} | Accuracy: {report['scoring']['accuracy']:.2%}")
        print(f"Average Score: {avg_score:.4f}")
        print(f"Average Iterations: {avg_iterations:.1f}")
        print(f"Average Duration: {avg_duration:.1f}s")
        print(f"Total Tool Calls: search={total_search_calls}, get_document={total_get_doc_calls}, visit_webpage={total_visit_calls}")
        print(f"Avg Visit Time: {avg_visit_time:.2f}s | Jina Fallbacks: {total_jina_fallbacks}")
        if failed_task_ids:
            print(f"Failed Task IDs: {', '.join(failed_task_ids[:20])}{'...' if len(failed_task_ids) > 20 else ''}")
        if bench_metrics:
            print(f"\n6-Dimension Benchmark Evaluation ({bench_metrics.get('evaluated_count', 0)} tasks):")
            print(f"  Task Fulfillment:     {bench_metrics.get('avg_task_fulfillment', 0):.2f}")
            print(f"  Grounding:            {bench_metrics.get('avg_grounding', 0):.2f}")
            print(f"  Tool Appropriateness: {bench_metrics.get('avg_tool_appropriateness', 0):.2f}")
            print(f"  Parameter Accuracy:   {bench_metrics.get('avg_parameter_accuracy', 0):.2f}")
            print(f"  Dependency Awareness: {bench_metrics.get('avg_dependency_awareness', 0):.2f}")
            print(f"  Parallelism:          {bench_metrics.get('avg_parallelism_and_efficiency', 0):.2f}")
            print(f"  Combined Average:     {bench_metrics.get('avg_combined', 0):.2f}")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"Error writing analysis report: {e}")


def scan_existing_results(target_dir: Path) -> Dict[str, Any]:
    """Scan the target directory for existing result.json files.

    Supports two directory structures:
    1. Old: target_dir/tasks/task_id/result.json
    2. New (step-based): target_dir/0/run_0/task_id/result.json

    Uses task_id (query_id) as the primary key for identifying completed tasks.
    """
    existing_results = {}

    # Try old structure first: target_dir/tasks/
    tasks_dir = target_dir / "tasks"
    if tasks_dir.exists():
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Scanning for existing results in {tasks_dir} (old structure)...")
        for task_folder in tasks_dir.iterdir():
            if task_folder.is_dir():
                result_file = task_folder / "result.json"
                if result_file.exists():
                    try:
                        with open(result_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        task_id = str(data.get('task_id', task_folder.name))
                        existing_results[task_id] = data
                    except Exception as e:
                        print(f"Warning: Failed to read {result_file}: {e}")

    # Try new structure: target_dir/0/run_*/task_id/
    step_dir = target_dir / "0"
    if step_dir.exists():
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Scanning for existing results in {step_dir} (step-based structure)...")
        for run_folder in step_dir.iterdir():
            if run_folder.is_dir() and run_folder.name.startswith("run_"):
                for task_folder in run_folder.iterdir():
                    if task_folder.is_dir():
                        result_file = task_folder / "result.json"
                        if result_file.exists():
                            try:
                                with open(result_file, 'r', encoding='utf-8') as f:
                                    data = json.load(f)
                                task_id = str(data.get('task_id', task_folder.name))
                                existing_results[task_id] = data
                            except Exception as e:
                                print(f"Warning: Failed to read {result_file}: {e}")

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Found {len(existing_results)} existing completed tasks.")
    return existing_results


def build_worker_command(args, task_group: List[Dict[str, Any]], run_name: str, dataset_name: str) -> tuple:
    """Build the command to run worker_standalone.py for a group of tasks.

    Args:
        args: Parsed command line arguments
        task_group: List of task dicts with question, answer, task_id, row_index
        run_name: Name for the output directory
        dataset_name: Dataset name for output directory structure

    Returns:
        Tuple of (command list, tasks_file path) - caller should clean up tasks_file after use
    """
    script_dir = Path(__file__).parent
    worker_script = script_dir / 'worker_standalone.py'

    # Write tasks to a temp file
    tasks_file = tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.json',
        prefix='tasks_',
        delete=False,
        encoding='utf-8'
    )
    json.dump(task_group, tasks_file, ensure_ascii=False)
    tasks_file.close()

    cmd = ['python', str(worker_script)]
    cmd.extend(['--worker-type', args.worker_type])
    cmd.extend(['--tasks-file', tasks_file.name])
    cmd.extend(['--agent-model', args.agent_model])
    cmd.extend(['--output-dir', str(Path(args.output_dir) / dataset_name)])
    cmd.extend(['--run-name', run_name])
    cmd.extend(['--max-iters', str(args.max_iters)])
    cmd.extend(['--dataset', dataset_name])

    if args.timeout:
        cmd.extend(['--timeout', str(args.timeout)])
    if args.judge_model:
        cmd.extend(['--judge-model', args.judge_model])

    # Common agent/memory args
    cmd.extend(['--memory-model', args.memory_model])
    if args.enable_memory:
        cmd.append('--enable-memory')
    if args.con_on_no_tool_use:
        cmd.append('--con-on-no-tool-use')
    if args.no_thinking:
        cmd.append('--no-thinking')
    if args.stream:
        cmd.append('--stream')
    if args.reasoning_effort:
        cmd.extend(['--reasoning-effort', args.reasoning_effort])
    cmd.extend(['--search-engine', args.search_engine])

    if args.worker_type == 'bcp':
        # BCP specific
        browsecomp_path = os.environ.get('BROWSECOMP_PATH')
        if args.index_path:
            index_path = Path(browsecomp_path) / args.index_path if browsecomp_path else Path(args.index_path)
        else:
            if browsecomp_path:
                index_path = Path(browsecomp_path) / "indexes" / args.searcher
            else:
                index_path = Path("indexes") / args.searcher

        cmd.extend(['--searcher', args.searcher])
        if args.searcher_model:
            cmd.extend(['--searcher-model', args.searcher_model])
        cmd.extend(['--index-path', str(index_path)])
        cmd.extend(['--top-k', str(args.top_k)])
        if args.use_bg_info:
            cmd.append('--use-bg-info')
        if args.tokenizer_model:
            cmd.extend(['--tokenizer-model', args.tokenizer_model])

    else:  # mcp
        # MCP specific
        if args.commands_file:
            cmd.extend(['--commands-file', args.commands_file])
        if args.tool_cache_path:
            cmd.extend(['--tool-cache-path', args.tool_cache_path])
        if args.es_fallback:
            cmd.append('--es-fallback')
        if args.es_url:
            cmd.extend(['--es-url', args.es_url])
        if args.es_index:
            cmd.extend(['--es-index', args.es_index])
        if args.miss_log_path:
            cmd.extend(['--miss-log-path', args.miss_log_path])
        if args.skip_live_mcp:
            cmd.extend(['--skip-live-mcp'] + args.skip_live_mcp)

    # Common memory configuration
    if args.max_model_len is not None:
        cmd.extend(['--max-model-len', str(args.max_model_len)])
    if args.compress_fre is not None:
        cmd.extend(['--compress-fre', str(args.compress_fre)])
    if args.compress_fre_min is not None:
        cmd.extend(['--compress-fre-min', str(args.compress_fre_min)])
    if args.compress_fre_max is not None:
        cmd.extend(['--compress-fre-max', str(args.compress_fre_max)])
    if args.calculate_reward:
        cmd.append('--calculate-reward')

    return cmd, tasks_file.name


def run_worker_process(cmd: List[str], process_id: int) -> int:
    """Run a worker process and return its exit code."""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting process {process_id} with command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        exit_code = result.returncode

        if exit_code == 0:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Process {process_id} completed successfully")
        else:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Process {process_id} failed with exit code {exit_code}")

        return exit_code

    except Exception as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Process {process_id} failed with exception: {e}")
        return 1


def run_execution_cycle(tasks: List[Dict[str, Any]], args, run_name: str, cycle_id: str, dataset_name: str) -> List[Dict[str, Any]]:
    """Run one execution cycle (batch) of tasks.

    Args:
        tasks: List of task dicts with question, answer, task_id, row_index
        args: Parsed command line arguments
        run_name: Name for the output directory
        cycle_id: Identifier for this execution cycle
        dataset_name: Dataset name for output directory structure
    """
    task_groups = split_tasks(tasks, args.parallel)

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Splitting {len(tasks)} tasks into {len(task_groups)} parallel groups for cycle {cycle_id}")
    for i, group in enumerate(task_groups):
        task_ids_in_group = [t.get('task_id', '') for t in group]
        print(f"  Group {i+1}: {task_ids_in_group}")

    threads = []
    thread_results = {}
    temp_files = []  # Track temp files for cleanup

    def run_worker_thread(process_id: int, task_group: List[Dict[str, Any]]):
        if not task_group:
            thread_results[process_id] = 0
            return

        cmd, tasks_file = build_worker_command(args, task_group, run_name, dataset_name)
        temp_files.append(tasks_file)
        exit_code = run_worker_process(cmd, process_id)
        thread_results[process_id] = exit_code

    for i, task_group in enumerate(task_groups):
        thread = threading.Thread(
            target=run_worker_thread,
            args=(i+1, task_group)
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    # Clean up temp files
    for temp_file in temp_files:
        try:
            os.unlink(temp_file)
        except Exception:
            pass

    failed_processes = [pid for pid, code in thread_results.items() if code != 0]
    if failed_processes:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Warning: Processes {failed_processes} failed in cycle {cycle_id}")

    return []


def main():
    """Main function."""
    args = parse_args()

    if args.worker_type == 'bcp' and not os.environ.get('BROWSECOMP_PATH'):
        print("Warning: BROWSECOMP_PATH not set. Make sure BrowseComp-Plus is accessible.")

    # Derive dataset name for output directory structure
    # Priority: --dataset CLI arg > last component of --data-dir > fallback "default"
    if args.dataset:
        dataset_name = args.dataset
    elif args.data_dir:
        dataset_name = Path(args.data_dir).name
    else:
        dataset_name = "default"

    all_tasks_latest_results = {}
    tasks_to_run = []  # List of task dicts with question, answer, task_id, row_index

    # Determine Run Name and Target Directory
    if args.run_name:
        run_name = args.run_name
    else:
        if args.resume:
            print("Error: --run-name is required when using --resume.")
            sys.exit(1)
        time.sleep(1.1)
        run_name = f"{args.worker_type}_{time.strftime('%Y%m%d_%H%M%S')}"
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Auto-generated Run Name: {run_name}")

    target_dir = Path(args.output_dir) / dataset_name / run_name

    # Load dataset to get task data
    dataset_tasks = []
    task_id_to_task = {}  # Map task_id to full task data
    data_file = find_data_file(args)
    if data_file:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading dataset from: {data_file}")
        dataset_tasks = load_dataset(data_file)
        # Build task_id -> task mapping
        for idx, task in enumerate(dataset_tasks):
            task_entry = {
                'question': task['question'],
                'answer': task['answer'],
                'task_id': task['task_id'],
                'row_index': idx,
            }
            # Preserve MCP-Bench specific fields
            if 'servers' in task:
                task_entry['servers'] = task['servers']
            if 'raw' in task:
                task_entry['raw'] = task['raw']
            task_id_to_task[task['task_id']] = task_entry
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loaded {len(dataset_tasks)} tasks from dataset")

    # --- Resume Logic ---
    if args.resume:
        if not target_dir.exists():
            print(f"Error: Directory {target_dir} does not exist. Cannot resume.")
            sys.exit(1)

        results_file = target_dir / "batch_results.json"
        if results_file.exists():
            try:
                with open(results_file, 'r', encoding='utf-8') as f:
                    prev_results = json.load(f)
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loaded {len(prev_results)} results from {results_file}")

                for res in prev_results:
                    # Use task_id (query_id) as the primary key
                    tid = str(res.get('task_id', ''))
                    if tid:
                        all_tasks_latest_results[tid] = res
            except Exception as e:
                print(f"Error reading {results_file}: {e}")

        scanned_results = scan_existing_results(target_dir)
        all_tasks_latest_results.update(scanned_results)

        task_ids_set = set()

        for tid, res in all_tasks_latest_results.items():
            if not res.get('success', False):
                task_ids_set.add(tid)

        if args.task_id:
            raw_task_ids = args.task_id.split(',')
            if len(raw_task_ids) == 1 and raw_task_ids[0] == '-1':
                # Use dataset to get query_ids for the specified range
                if not dataset_tasks:
                    print("Error: Cannot use -1 mode without a valid dataset file.")
                    sys.exit(1)
                start_idx = args.begin
                end_idx = min(args.end, len(dataset_tasks))
                candidates = [dataset_tasks[i]['task_id'] for i in range(start_idx, end_idx)]
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Resume with range check: rows {args.begin}-{args.end} ({len(candidates)} query_ids)")
            else:
                # Direct query_id list
                candidates = [x.strip() for x in raw_task_ids if x.strip()]
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Resume with query_id list: {len(candidates)} tasks")

            for tid in candidates:
                if tid not in all_tasks_latest_results:
                    task_ids_set.add(tid)
                elif not all_tasks_latest_results[tid].get('success', False):
                    task_ids_set.add(tid)

        # Convert task_ids to full task data
        for tid in sorted(task_ids_set):
            if tid in task_id_to_task:
                tasks_to_run.append(task_id_to_task[tid])
            else:
                print(f"Warning: Task ID '{tid}' not found in dataset, skipping")

        if not tasks_to_run:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] No failed or missing tasks found to resume.")
            sys.exit(0)

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Resuming {len(tasks_to_run)} tasks.")

    # --- New Run Logic ---
    else:
        if target_dir.exists():
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Output directory '{target_dir}' exists. Checking for completed tasks...")
            scanned_results = scan_existing_results(target_dir)
            all_tasks_latest_results.update(scanned_results)

        if not args.task_id:
            print("Error: --task-id is required unless --resume is used.")
            sys.exit(1)

        raw_task_ids = args.task_id.split(',')
        if len(raw_task_ids) == 1 and raw_task_ids[0] == '-1':
            # Use dataset to get query_ids for the specified range
            if not dataset_tasks:
                print("Error: Cannot use -1 mode without a valid dataset file.")
                sys.exit(1)
            start_idx = args.begin
            end_idx = min(args.end, len(dataset_tasks))
            candidates = [dataset_tasks[i]['task_id'] for i in range(start_idx, end_idx)]
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Task ID -1 detected. Rows {args.begin}-{args.end} -> {len(candidates)} query_ids.")
        else:
            # Direct query_id list
            candidates = [x.strip() for x in raw_task_ids if x.strip()]
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Query ID list provided: {len(candidates)} tasks.")

        skipped_count = 0
        for tid in candidates:
            if tid in all_tasks_latest_results and all_tasks_latest_results[tid].get('success', False):
                skipped_count += 1
            else:
                if tid in task_id_to_task:
                    tasks_to_run.append(task_id_to_task[tid])
                else:
                    print(f"Warning: Task ID '{tid}' not found in dataset, skipping")

        if skipped_count > 0:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Skipping {skipped_count} already completed tasks.")

        if not tasks_to_run:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] All requested tasks are already completed. Exiting.")
            sys.exit(0)

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {len(tasks_to_run)} tasks to run.")

    # --- Execution Loop ---
    repeat_times = args.repeat_times
    repeat_run_names = []

    def run_single_repeat(repeat_idx: int) -> str:
        """Run a single repeat (with retry loop). Returns the repeat_run_name."""
        if repeat_times > 1:
            repeat_run_name = f"{run_name}/0/run_{repeat_idx}"
            print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] === Repeat {repeat_idx + 1}/{repeat_times} (run: {repeat_run_name}) ===")
        else:
            repeat_run_name = f"{run_name}/0/run_0"

        current_tasks = list(tasks_to_run)
        iteration_count = 0

        while True:
            if not args.resume and iteration_count == 0:
                cycle_label = "main"
            else:
                ts = time.strftime('%H%M%S')
                cycle_label = f"retry_{iteration_count}_{ts}"

            run_execution_cycle(current_tasks, args, repeat_run_name, cycle_label, dataset_name)

            # Check for tasks to retry by scanning results
            retry_tasks = []
            if RETRY_ERROR_PATTERNS:
                repeat_dir = Path(args.output_dir) / dataset_name / repeat_run_name
                cycle_results = scan_existing_results(repeat_dir)
                for task in current_tasks:
                    tid = task.get('task_id', '')
                    if tid in cycle_results:
                        res = cycle_results[tid]
                        if not res.get('success', False):
                            err_msg = res.get('error', '')
                            if any(pattern in err_msg for pattern in RETRY_ERROR_PATTERNS):
                                retry_tasks.append(task)

            iteration_count += 1

            if not retry_tasks:
                if RETRY_ERROR_PATTERNS:
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Repeat {repeat_idx}: No tasks to retry matching patterns.")
                break

            if iteration_count > args.max_retries:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Repeat {repeat_idx}: Max retries ({args.max_retries}) reached. Stopping.")
                break

            current_tasks = retry_tasks
            retry_ids = [t.get('task_id', '') for t in retry_tasks]
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Repeat {repeat_idx}: Retrying {len(current_tasks)} failed tasks: {retry_ids}")

        return repeat_run_name

    if repeat_times > 1:
        # Run all repeats in parallel
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Launching {repeat_times} repeats in parallel...")
        repeat_threads = []
        repeat_name_results = {}

        def repeat_thread_fn(idx):
            repeat_name_results[idx] = run_single_repeat(idx)

        for repeat_idx in range(repeat_times):
            t = threading.Thread(target=repeat_thread_fn, args=(repeat_idx,))
            repeat_threads.append(t)
            t.start()

        for t in repeat_threads:
            t.join()

        repeat_run_names = [repeat_name_results[i] for i in range(repeat_times)]
    else:
        repeat_run_names = [run_single_repeat(0)]

    # Collect all results and generate analysis report
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Collecting results from all tasks...")

    if repeat_times > 1:
        # Collect results across all repeats, grouped by task_id
        all_repeat_results: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for repeat_run_name in repeat_run_names:
            repeat_dir = Path(args.output_dir) / dataset_name / repeat_run_name
            repeat_results = scan_existing_results(repeat_dir)
            for task_id, result in repeat_results.items():
                all_repeat_results[task_id].append(result)

        # Collect one representative result per task for the analysis report
        all_results_list = []
        for task_id, results in sorted(all_repeat_results.items()):
            all_results_list.append(results[0])

        # Compute mean@k and best@k across all tasks
        global_metrics = calculate_repeat_metrics(all_repeat_results, repeat_times)

        # Print repeat summary
        print(f"\n{'='*60}")
        print(f"REPEAT METRICS SUMMARY (repeat_times={repeat_times})")
        print(f"{'='*60}")
        print(f"Total Tasks: {len(all_repeat_results)}")
        for key, value in sorted(global_metrics.items()):
            print(f"  {key}: {value:.4f}")
        print(f"{'='*60}\n")

        if all_results_list:
            analysis_file = target_dir / "analysis_report.json"
            save_analysis_report(all_results_list, analysis_file, global_repeat_metrics=global_metrics)

            batch_results_file = target_dir / "batch_results.json"
            try:
                with open(batch_results_file, 'w', encoding='utf-8') as f:
                    json.dump(all_results_list, f, indent=2, ensure_ascii=False)
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Batch results saved to {batch_results_file}")
            except Exception as e:
                print(f"Error writing batch results: {e}")
        else:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] No results found to analyze.")
    else:
        final_results = scan_existing_results(target_dir)

        if final_results:
            all_results_list = list(final_results.values())
            analysis_file = target_dir / "analysis_report.json"
            save_analysis_report(all_results_list, analysis_file)

            batch_results_file = target_dir / "batch_results.json"
            try:
                with open(batch_results_file, 'w', encoding='utf-8') as f:
                    json.dump(all_results_list, f, indent=2, ensure_ascii=False)
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Batch results saved to {batch_results_file}")
            except Exception as e:
                print(f"Error writing batch results: {e}")
        else:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] No results found to analyze.")

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Execution finished. Check individual task directories in {target_dir}/tasks for results.")
    sys.exit(0)


if __name__ == "__main__":
    main()
