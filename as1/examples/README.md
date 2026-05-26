# AS1 Examples

This directory contains unified evaluation tools for running BCP (BrowseComp-Plus) and ReAct workers.

## Architecture

```
parallel_runner.py (orchestrator)
    │
    ├── Loads dataset from --data-dir or project_root/data/
    ├── Splits tasks into groups based on -p (parallel processes)
    ├── Writes each task group to a temp JSON file
    │
    └── Spawns worker_standalone.py processes
            │
            └── Reads tasks from --tasks-file
            └── Executes tasks with asyncio concurrency
            └── Saves results to {output_dir}/{run_name}/tasks/{task_id}/
```

## Files

| File | Description |
|------|-------------|
| `parallel_runner.py` | Unified parallel execution orchestrator with resume, retry, and analysis features |
| `worker_standalone.py` | Unified worker executor (called by parallel_runner, not directly) |

## Quick Start

### React Worker

```bash
# Run task range (0-99) with 20 parallel processes
python parallel_runner.py --worker-type react -t -1 -b 0 -e 100 -p 20 \
    --agent-model qwen3-max \
    --data-dir ../data/browsecomp_plus \
    --run-name my_react_batch

# Run specific tasks by query_id
python parallel_runner.py --worker-type react -t "query_001,query_002" -p 2 \
    --agent-model qwen3-max \
    --data-dir ../data/browsecomp_plus

# Resume failed tasks
python parallel_runner.py --worker-type react --resume --run-name my_react_batch \
    --data-dir ../data/browsecomp_plus
```

### BCP Worker

```bash
# Run task range with memory manager enabled
python parallel_runner.py --worker-type bcp -t -1 -b 0 -e 100 -p 20 \
    --agent-model qwen3-max --memory-model gpt-5-2025-08-07 --enable-memory \
    --data-dir ../data/browsecomp_plus \
    --run-name my_bcp_batch

# Resume failed tasks
python parallel_runner.py --worker-type bcp --resume --run-name my_bcp_batch \
    --data-dir ../data/browsecomp_plus
```

## Parameters

### Common Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--worker-type` | Worker type: `bcp` or `react` | Required |
| `-t, --task-id` | Task IDs: comma-separated query_ids or "-1" for range | Required (unless --resume) |
| `-p, --parallel` | Number of parallel worker processes | 1 |
| `-b, --begin` | Start index (when task-id is "-1") | 0 |
| `-e, --end` | End index (when task-id is "-1") | 10 |
| `--data-dir` | Path to dataset directory | `project_root/data/` |
| `-s, --split` | Data split: `train` or `test` | `test` |
| `-o, --output-dir` | Output directory | `./benchmark_results` |
| `--run-name` | Custom run name for output directory | Auto-generated |
| `--resume` | Resume failed tasks from existing run | False |
| `--max-retries` | Maximum retry attempts for failed tasks | 3 |
| `--timeout` | Timeout per task in seconds | 12000 |
| `-m, --agent-model` | Agent model name | `qwen3-max` |
| `-J, --judge-model` | Judge model name | `gpt-4o-2024-11-20` |

### BCP-Specific Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `-M, --memory-model` | Memory manager model | `gpt-5-2025-08-07` |
| `--enable-memory` | Enable memory manager | False |
| `-S, --searcher` | Searcher type: `bm25` or `faiss` | `bm25` |
| `--searcher-model` | Embedding model for searcher | `Qwen/Qwen3-Embedding-8B` |
| `-I, --index-path` | Path to search index | Auto |
| `-k, --top-k` | Top K search results | 5 |
| `-i, --max-iters` | Max agent iterations | 50 |
| `--use-bg-info` | Use background information | False |
| `--tokenizer-model` | Tokenizer model path | `DEFAULT_TOKENIZER_MODEL` env |

### React-Specific Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `-c, --config` | Test memory config name | None |
| `-d, --dataset` | Dataset name (for output dir) | Derived from --data-dir |
| `--search-engine` | Search engine type (see options below) | `nlp_search` |
| `--mode` | Mode: `test_context` or empty | Empty |
| `--context-json-path` | Path to context JSON file | Empty |

### Search Engine Options (React Worker)

The React Worker supports multiple search engines via the `--search-engine` parameter. Each engine has different characteristics in terms of speed, cost, and data source.

#### Available Search Engines

| Engine | Type | Speed | Cost | Offline | Data Source | Use Case |
|--------|------|-------|------|---------|-------------|----------|
| `meilisearch` | Local Index | ⚡ Fast | Free | ✅ Yes | BrowseComp-Plus (100K docs) | Development, evaluation, offline testing |
| `bright_data` | API | 🐢 Slow | Paid | ❌ No | Google (real-time) | Production, real-time web search |
| `nlp_search` | API | 🐢 Slow | Paid | ❌ No | NLP Search API | General web search |
| `qwen_search` | API | 🐢 Slow | Paid | ❌ No | Qwen Search (with sources) | Enhanced search with Qwen models |
| `serpapi` | API | 🐢 Slow | Paid | ❌ No | SerpAPI | Google search via API |

#### Meilisearch: Local Search Engine

**Design**: Meilisearch is an open-source, fast, and typo-tolerant search engine. For this project, it provides local document search over the BrowseComp-Plus corpus (100,195 documents), enabling fast, offline evaluation without external API dependencies.

**Implementation**:
- Location: `as1/asio/agent/react_worker.py:1310-1384`
- Index: Pre-built index with 100,195 BrowseComp-Plus documents
- Fields: `docid`, `title`, `url`, `text` (full document content)
- Features: Text search, snippet extraction, automatic cropping (200 chars)

**Setup**:

1. **Start Meilisearch Server** (one-time setup):
```bash
# Install Meilisearch (if not already installed)
curl -L https://install.meilisearch.com | sh

# Start server
meilisearch --db-path=./meili_data
```

2. **Verify Index** (should already exist):
```bash
# Check server status
curl http://localhost:7700/health

# Check index stats
curl http://localhost:7700/indexes/browsecomp_plus/stats
# Expected: {"numberOfDocuments": 100195, ...}
```

**Usage Examples**:

```bash
# Basic usage with meilisearch
python parallel_runner.py --worker-type react \
    --search-engine meilisearch \
    --agent-model qwen3-max \
    -t -1 -b 0 -e 10 -p 2

# With memory manager enabled
python parallel_runner.py --worker-type react \
    --search-engine meilisearch \
    --agent-model qwen3-max \
    --enable-memory \
    --data-dir ../data/browsecomp_plus \
    -t -1 -b 0 -e 50 -p 5

# Full BrowseComp-Plus evaluation (all 150 test tasks)
python parallel_runner.py --worker-type react \
    --search-engine meilisearch \
    --agent-model qwen3-max \
    --enable-memory \
    --data-dir ../data/browsecomp_plus \
    --run-name browsecomp_eval \
    -t -1 -b 0 -e 150 -p 10 \
    --max-iters 30
```

**Environment Variables** (optional):
```bash
export MEILISEARCH_URL="http://localhost:7700"  # Default
export MEILISEARCH_INDEX="browsecomp_plus"      # Default
export MEILISEARCH_API_KEY=""                   # Default (no auth)
```

**Comparison with Other Engines**:

```bash
# Meilisearch (fast, offline, deterministic)
--search-engine meilisearch

# Bright Data (slow, online, real-time Google)
--search-engine bright_data

# NLP Search (slow, online)
--search-engine nlp_search
```

**When to Use Meilisearch**:
- ✅ **Development & Testing**: Fast iteration without API costs
- ✅ **BrowseComp-Plus Evaluation**: Standard benchmark dataset evaluation
- ✅ **Offline Environments**: No internet connection required
- ✅ **Reproducibility**: Deterministic results (same query → same results)
- ✅ **Cost Reduction**: No API costs for search operations

**When NOT to Use Meilisearch**:
- ❌ Real-time web content (use `bright_data` or `serpapi`)
- ❌ Documents outside BrowseComp-Plus corpus
- ❌ Tasks requiring very recent information

**Technical Details**:
- **Index Size**: ~3.6 GB on disk (100,195 documents)
- **Search Speed**: < 100ms per query (local)
- **Snippet Length**: 200 characters (configurable in code)
- **Return Format**: Markdown list with titles, URLs, and snippets
- **Caching**: Automatic search result caching (via React Worker)

**Troubleshooting**:

```bash
# If Meilisearch server is not running
meilisearch --db-path=./meili_data

# Check if index exists
curl http://localhost:7700/indexes

# Test search manually
curl -X POST 'http://localhost:7700/indexes/browsecomp_plus/search' \
  -H 'Content-Type: application/json' \
  --data-binary '{"q": "climate change", "limit": 5}'
```

**References**:
- Setup Guide: `MEILISEARCH_SETUP.md` (project root)
- Integration Status: `MEILISEARCH_INTEGRATION_SUCCESS.md`
- Code Implementation: `as1/asio/agent/react_worker.py:1310-1384`

### Repeat Experiment Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--repeat-times` | Number of times to repeat each task | 1 |

When `--repeat-times > 1`, results are organized as:
```
{output_dir}/{dataset}/{run_name}/
├── run_0/tasks/...
├── run_1/tasks/...
└── analysis_report.json  # Contains mean@k and best@k metrics
```

## Output Structure

All results are saved in a unified structure:

```
{output_dir}/{dataset}/{run_name}/
├── tasks/
│   ├── {query_id_1}/
│   │   ├── result.json      # Task result with score, answer, etc.
│   │   ├── logging.log      # Detailed execution log
│   │   └── invoke/          # LLM invocation logs
│   ├── {query_id_2}/
│   │   └── ...
│   └── ...
├── batch_results.json       # Aggregated results from all tasks
└── analysis_report.json     # Summary statistics
```

### result.json Schema

```json
{
  "question": "...",
  "ground_truth": "...",
  "answer": "...",
  "task_id": "query_001",
  "row_index": 0,
  "score": 1.0,
  "success": true,
  "reasoning": "...",
  "timestamp": "2025-01-21T14:00:00",
  "duration_seconds": 45.2
}
```

### analysis_report.json Schema

```json
{
  "summary": {
    "total_tasks": 100,
    "success_count": 95,
    "failure_count": 5,
    "execution_success_rate": 0.95
  },
  "scoring": {
    "total_score": 75.0,
    "average_score": 0.75,
    "correct_count": 75,
    "accuracy": 0.75
  },
  "iterations": {
    "average": 12.5,
    "max": 50,
    "min": 3
  },
  "tool_calls": {
    "total_search_calls": 450,
    "total_visit_webpage_calls": 200,
    "avg_search_per_task": 4.5
  },
  "task_ids": ["query_001", "query_002", ...],
  "failed_task_ids": ["query_005", "query_023", ...],
  "repeat_metrics": {
    "mean@1": 0.75,
    "mean@3": 0.78,
    "best@1": 0.75,
    "best@3": 0.85
  },
  "timestamp": "2025-01-21 14:30:00"
}
```

## Features

### Resume Logic

The runner automatically tracks completed tasks and can resume from failures:

1. Scans `{run_name}/tasks/*/result.json` for existing results
2. Skips tasks with `"success": true`
3. Re-runs tasks with `"success": false` or missing results

```bash
# Initial run (interrupted or failed)
python parallel_runner.py --worker-type react -t -1 -b 0 -e 100 -p 20 \
    --data-dir ../data/browsecomp_plus --run-name batch1

# Resume - only runs failed/missing tasks
python parallel_runner.py --worker-type react --resume --run-name batch1 \
    --data-dir ../data/browsecomp_plus
```

### Error Pattern Retry

Tasks failing with specific error patterns (e.g., rate limiting) are automatically retried:

```python
RETRY_ERROR_PATTERNS = [
    "Error code: 429",  # Rate limit
]
```

### Parallel Execution

Parallelism is controlled at two levels:

1. **parallel_runner.py** `-p` parameter: Number of worker_standalone processes
2. **worker_standalone.py** `--max-concurrent`: Asyncio concurrency within each process

Example with 100 tasks and `-p 10`:
- 10 worker_standalone processes are spawned
- Each process handles ~10 tasks sequentially (or concurrently if --max-concurrent > 1)

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` or `DASHSCOPE_API_KEY` | API key for LLM calls |
| `BASE_URL` | Custom API endpoint (optional) |
| `BROWSECOMP_PATH` | Path to BrowseComp-Plus installation (BCP only) |
| `DEFAULT_TOKENIZER_MODEL` | Default tokenizer model path |

## Data Directory

Dataset files should be placed in the directory specified by `--data-dir` (or default `project_root/data/`).

The loader looks for `{split}.jsonl` or `{split}.json` (e.g., `test.jsonl`).

### Dataset Field Mapping

The loader supports flexible field names:

| Field | Accepted Names |
|-------|---------------|
| Question | `question`, `ques`, `query`, `problem` |
| Answer | `answer`, `Final answer`, `ground_truth`, `truth` |
| Task ID | `task_id`, `query_id` |

## Adding a New Worker Type

This section describes how to add a new worker type (e.g., `myworker`) to the evaluation framework.

### Step 1: Create Worker Class

Create a new worker class in `as1/asio/agent/`:

```python
# as1/asio/agent/myworker.py

class MyWorker:
    def __init__(
        self,
        name: str,
        model,
        formatter,
        memory_class: str = "inmemory",
        memory_config: dict = None,
        experiment_logger = None,
        # Add your custom parameters
        my_custom_param: str = "default",
        **kwargs
    ):
        self.name = name
        self.model = model
        self.formatter = formatter
        self.logger = experiment_logger
        self.my_custom_param = my_custom_param
        # Initialize memory, tools, etc.

    async def run_task(self, question: str, ground_truth: str = None) -> dict:
        """Run a single task and return result dict."""
        # Your task execution logic
        return {
            "answer": "...",
            "success": True,
            "tool_calls": {...},
            # Add any metrics you want to track
        }
```

### Step 2: Add Task Runner in worker_standalone.py

Add a new `run_myworker_task()` function:

```python
# as1/examples/worker_standalone.py

async def run_myworker_task(
    question: str,
    ground_truth: Optional[str] = None,
    agent_model: str = "gpt-4o-mini",
    output_dir: str = "./outputs",
    task_id: Any = 0,
    row_index: int = 0,
    run_name: str = "myworker_standalone",
    dataset: str = "default",
    # Add your custom parameters
    my_custom_param: str = "default",
    judge_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a single MyWorker task."""
    from asio.agent.myworker import MyWorker

    logger = ExperimentLogger(
        base_dir=output_dir,
        test_mem_config=run_name,
        dataset="tasks"
    )
    logger.start_run(task_id)

    try:
        model, formatter = create_model_and_formatter(agent_model)

        worker = MyWorker(
            name="assistant",
            model=model,
            formatter=formatter,
            experiment_logger=logger,
            my_custom_param=my_custom_param,
        )

        result = await worker.run_task(question, ground_truth)

        # Run judge if ground truth available
        if ground_truth:
            judge_output = await judge_result(...)
            result.update(judge_output)

        result["task_id"] = task_id
        result["success"] = True
        logger.save_report(result, filename="result.json")
        return result

    except Exception as e:
        # Error handling (see existing implementations)
        ...
```

### Step 3: Update run_batch_tasks()

Add the new worker type branch in `run_batch_tasks()`:

```python
# as1/examples/worker_standalone.py

async def run_batch_tasks(
    worker_type: str,
    # ... existing params ...
    # Add your custom parameters
    my_custom_param: str = "default",
    **kwargs
) -> List[Dict[str, Any]]:

    async def run_with_semaphore(task_idx: int, task: dict) -> Dict[str, Any]:
        # ... existing code ...

        if worker_type == "bcp":
            coro = run_bcp_task(...)
        elif worker_type == "react":
            coro = run_react_task(...)
        elif worker_type == "myworker":  # Add new branch
            coro = run_myworker_task(
                question=question,
                ground_truth=ground_truth,
                agent_model=agent_model,
                output_dir=output_dir,
                task_id=task_id,
                row_index=row_index,
                run_name=run_name,
                dataset=dataset,
                my_custom_param=my_custom_param,
                judge_model=judge_model,
            )
        else:
            raise ValueError(f"Unknown worker type: {worker_type}")
```

### Step 4: Update CLI Arguments

Add new worker type and parameters in both files:

```python
# as1/examples/worker_standalone.py - main()

parser.add_argument("--worker-type", type=str, required=True,
                    choices=["bcp", "react", "myworker"],  # Add here
                    help="Worker type")

# Add custom parameters
parser.add_argument("--my-custom-param", type=str, default="default",
                    help="My custom parameter (MyWorker)")
```

```python
# as1/examples/parallel_runner.py - parse_args()

parser.add_argument('--worker-type', required=True,
                    choices=['bcp', 'react', 'myworker'],  # Add here
                    help='Worker type')

# Add custom parameters under "# MyWorker specific" section
parser.add_argument('--my-custom-param', type=str, default='default',
                    help='My custom parameter (MyWorker)')
```

### Step 5: Update build_worker_command()

Pass new parameters to worker_standalone.py:

```python
# as1/examples/parallel_runner.py

def build_worker_command(args, task_group, run_name, dataset_name):
    # ... existing code ...

    if args.worker_type == 'bcp':
        # BCP specific args
        ...
    elif args.worker_type == 'react':
        # React specific args
        ...
    elif args.worker_type == 'myworker':
        # MyWorker specific args
        if args.my_custom_param:
            cmd.extend(['--my-custom-param', args.my_custom_param])

    return cmd, tasks_file.name
```

### Step 6: (Optional) Add Training Workflow

If you need RL training for the new worker, create a workflow:

```python
# trinity/common/workflows/myworker_training_workflow.py

class MyWorkerTrainingWorkflow:
    """Training workflow for MyWorker."""

    async def run_episode(self, task_data: dict) -> dict:
        # Initialize worker with Trinity model
        # Run task and collect rewards
        # Return training data
        ...
```

And add corresponding YAML config in `examples/myworker/`.

### Checklist

| File | Changes |
|------|---------|
| `as1/asio/agent/myworker.py` | Create worker class |
| `as1/examples/worker_standalone.py` | Add `run_myworker_task()`, update `run_batch_tasks()`, add CLI args |
| `as1/examples/parallel_runner.py` | Add CLI args, update `build_worker_command()` |
| `as1/examples/README.md` | Document new worker and parameters |
| `trinity/common/workflows/` | (Optional) Add training workflow |
| `examples/myworker/` | (Optional) Add training YAML configs |

## Subdirectories

| Directory | Description |
|-----------|-------------|
| `asio/` | ReAct worker additional utilities |
| `bcp/` | BCP worker additional utilities |
| `sciworld/` | ScienceWorld environment examples |
