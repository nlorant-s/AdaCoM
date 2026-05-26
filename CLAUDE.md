# CLAUDE.md

Paper code submission for **Context Manager RL training** (BCP / MCP workers).

## What this repo contains

- `as1/asio/agent/{bcp_worker.py, mcp_worker.py}` — the two task-solving agents.
- `trinity/common/workflows/envs/browse_comp_plus/bcp_simple_react_workflow.py` — BCP RL workflow (registered as `bcp_simple_react_workflow`, imports `BCPWorker`).
- `trinity/common/workflows/mcp_bench_workflow.py` — MCP RL workflow (registered as `mcp_bench_workflow`, imports `MCPWorker`).
- `as1/asio/memory/memorymanager.py` — Context Manager / compression logic, the actual training target.
- Trinity-RFT framework under `trinity/` (algorithm, trainer, explorer, buffer, ...).

## Training

Single-node 8-GPU recipe: see [`TRAINING.md`](TRAINING.md) and
[`examples/browse_comp_plus/bcp_config_singlenode_8gpu_deepseek.yaml`](examples/browse_comp_plus/bcp_config_singlenode_8gpu_deepseek.yaml).

## Local evaluation (no training)

`as1/examples/parallel_runner.py` runs the same workers against an external API
agent, useful for sanity-checking before training.

Copy `export_template.sh` to your own `export_local.sh`, fill in API keys /
local paths (`BROWSECOMP_PATH`, `TOKENIZER_MODEL`, `DASHSCOPE_API_KEY`, …),
then:

```bash
source export_local.sh
cd as1/
python examples/parallel_runner.py --worker-type bcp \
    -t -1 -b 0 -e 5 -p 1 \
    --agent-model qwen3-max \
    --enable-memory --calculate-reward \
    --max-model-len 4000 \
    --tokenizer-model "$TOKENIZER_MODEL"
```

Worker types: `bcp`, `mcp`.

## Architecture notes

**Two-model setup.** The agent (DeepSeek-V3 / GPT-4o / etc.) is called via API
and is *never trained*. Only the Context Manager (local Qwen3-4B served by vLLM)
receives gradient updates.

**Multi-step GRPO.** Each task produces `repeat_times` rollouts; advantage is
`adv_task + alpha_inter_reward * sum(per_step_rewards)` then clipped. See
`trinity/algorithm/advantage_fn/multi_step_grpo_advantage.py`.

**Compression trigger.** The CM rewrites chat history every `compress_fre`
agent turns, and unconditionally when token count exceeds 50% of
`max_model_len`. See `as1/asio/memory/memorymanager.py`.
