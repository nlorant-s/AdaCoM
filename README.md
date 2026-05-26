# Context Manager Training (BCP / MCP)

Paper code release. Trains a **Context Manager** — a small LLM that compresses an
agent's running chat history so long-horizon tool-use trajectories stay within the
context window — via RL on top of the Trinity-RFT framework.

Two task-solving agents drive rollouts:

| Worker | Domain | Worker file | Trinity workflow |
|---|---|---|---|
| **BCP Worker** | BrowseComp-Plus (dense/BM25 retrieval over a corpus) | `as1/asio/agent/bcp_worker.py` | `trinity/common/workflows/envs/browse_comp_plus/bcp_simple_react_workflow.py` |
| **MCP Worker** | MCP-Bench (MCP tool calling) | `as1/asio/agent/mcp_worker.py` | `trinity/common/workflows/mcp_bench_workflow.py` |

The agent itself is an external API model (e.g. DeepSeek-V3, GPT-4o) and is **not
trained**. Only the Context Manager (default: Qwen3-4B-Instruct) is updated by RL.

## Repository layout

```
.
├── as1/                          # Agent implementation + local evaluation
│   ├── asio/
│   │   ├── agent/{bcp_worker.py,mcp_worker.py,...}
│   │   ├── memory/               # MemoryManager (context compression)
│   │   └── utils/                # judge, retry helpers
│   ├── examples/                 # parallel_runner.py, worker_standalone.py
│   └── src/agentscope/           # Vendored agentscope (LLM client abstractions)
├── trinity/                      # Trinity-RFT framework + project workflows
├── examples/
│   ├── browse_comp_plus/         # BCP training config + data prep
│   └── mcp_bench/                # MCP-Bench training config
├── data/browsecomp_plus/         # BCP train/test split (jsonl)
├── data/mcp_bench/               # MCP-Bench train/test split (jsonl)
└── TRAINING.md                   # Single-node 8-GPU training instructions
```

## Install

```bash
bash install.sh
# installs trinity-rft (root) + as1 (agent code)
```

Requires Python 3.10+, CUDA, and a working vLLM build for the Context Manager
serving stack.

## Train

See **[TRAINING.md](TRAINING.md)** for the single-node 8-GPU recipe and the
launch command for the kept config
[`examples/browse_comp_plus/bcp_config_singlenode_8gpu_deepseek.yaml`](examples/browse_comp_plus/bcp_config_singlenode_8gpu_deepseek.yaml).

## License

Apache 2.0 (see [LICENSE](LICENSE)).
