# Single-Node 8-GPU Training (BCP)

End-to-end recipe for training the Context Manager on **one node with 8 GPUs**
using the BCP (BrowseComp-Plus) workflow with an external DeepSeek-V3 agent and
a Qwen3-Embedding-8B dense retriever.

The kept config is
[`examples/browse_comp_plus/bcp_config_singlenode_8gpu_deepseek.yaml`](examples/browse_comp_plus/bcp_config_singlenode_8gpu_deepseek.yaml).

## 1. Hardware / software

- 1 node, 8 GPUs (H800/A100-80GB recommended; rollout uses 4 GPUs, trainer uses 4).
- CUDA + recent vLLM build (installed via `install.sh`).
- ~200GB free disk for the BrowseComp-Plus dense index and checkpoints.

## 2. Prerequisites

### 2.1 BrowseComp-Plus corpus + dense index

Download the BrowseComp-Plus dataset and its `qwen3-embedding-8b` dense index per
the [BrowseComp-Plus instructions](https://github.com/microsoft/BrowseComp-Plus).
Set:

```bash
export BROWSECOMP_PATH=/path/to/browse_comp_plus_root  # must contain indexes/qwen3-embedding-8b/
```

### 2.2 Context Manager base / SFT checkpoint

The config points `model.model_path` at an SFT-warmed Qwen3-4B checkpoint by
default. Either supply one and export `MODEL_PATH`, or change the path to the
raw `Qwen3-4B-Instruct-2507` weights:

```bash
export MODEL_PATH=/path/to/bcp_sft_5_epoch              # or Qwen3-4B base
export TOKENIZER_MODEL=/path/to/Qwen3-4B-Instruct-2507  # used for chat templating
export CHECKPOINT_ROOT_DIR=/path/to/checkpoints
```

### 2.3 Agent API key

The DeepSeek-V3 agent is called via OpenAI-compatible HTTP. Set the credentials
your agentscope `OpenAIChatModel` will pick up (e.g. `DASHSCOPE_API_KEY`,
`BASE_URL`). A convenience template is provided at
[`export_template.sh`](export_template.sh) — copy and fill in.

### 2.4 Training data

`data/browsecomp_plus/{train,test}.jsonl` ships with the repo. Each line:
`{"query_id": ..., "query": ..., "answer": ...}`. If you need a fresh dump from
the upstream HF dataset, run:

```bash
python examples/browse_comp_plus/get_browse_comp_data_for_trinity.py
```

## 3. Launch

```bash
trinity run --config examples/browse_comp_plus/bcp_config_singlenode_8gpu_deepseek.yaml
```

That's it. The harness will:

1. Spin up 4 vLLM rollout engines (one GPU each) hosting the Context Manager.
2. Run the BCP workflow: the DeepSeek-V3 agent calls `search` / `get_document`
   against the local dense index; whenever the prompt risks overflowing
   `max_model_len`, the Context Manager rewrites the chat history.
3. Score each rollout (LLM-judge on the final answer + per-step search-hit /
   penalty rewards), batch into experiences, and train the Context Manager with
   multi-step GRPO on the remaining 4 GPUs.
4. Evaluate on `data/browsecomp_plus/test.jsonl` every 2 training steps; save a
   checkpoint when `eval/browsecomp_eval/score/mean@3/mean` exceeds 0.33.

## 4. Key config knobs

| Field | Value | Meaning |
|---|---|---|
| `cluster.{node_num, gpu_per_node}` | `1, 8` | Single-node 8-GPU layout. |
| `explorer.rollout_model.engine_num` | `4` | vLLM engines for the CM (4 GPUs). The other 4 go to the trainer. |
| `algorithm.repeat_times` | `8` | Rollouts per task (GRPO group size). |
| `buffer.batch_size` / `train_batch_size` | `32` / `768` | Explorer batch (tasks) vs trainer batch (experiences). |
| `algorithm.alpha_inter_reward` | `0.1` | Weight of per-step intermediate reward vs final task reward. |
| `model.max_model_len` | `32768` | CM context window — pressure above ~50% triggers compression. |
| `trainer.ulysses_sequence_parallel_size` | `4` | Sequence parallelism within the 4 trainer GPUs. |

## 5. Outputs

- Checkpoints → `${CHECKPOINT_ROOT_DIR}/<project>/<group>/<name>/`
- W&B metrics → project `bcp_paper`, group `single_node_8gpu` (set
  `WANDB_API_KEY` or change `monitor.monitor_type` if not using W&B).
