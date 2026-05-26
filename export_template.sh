# Copy to e.g. `export.sh` (gitignored) and `source` before launching training.
# Only the agent credentials are strictly required; the rest have sane defaults.

# --- Agent (DeepSeek-V3 via OpenAI-compatible endpoint) ---
export DASHSCOPE_API_KEY=""
export BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"

# --- LLM judge (defaults to OpenAI; override if you route through another provider) ---
export OPENAI_API_KEY=""
# export ANTHROPIC_API_KEY=""
# export NEWAPI_API_KEY=""
# export NEWAPI_BASE_URL=""

# --- BrowseComp-Plus data + dense index ---
# Path to the cloned BrowseComp-Plus repo; must contain data/ and indexes/qwen3-embedding-8b/.
export BROWSECOMP_PATH=""
# Output dir for `get_browse_comp_data_for_trinity.py`; only needed when regenerating data.
# export TRINITY_TASKSET_PATH="$(pwd)/data/browsecomp_plus"

# --- Model + checkpoint paths ---
export MODEL_PATH="./checkpoints/bcp_sft_5_epoch"
export TOKENIZER_MODEL="./models/Qwen3-4B-Instruct-2507"
export CHECKPOINT_ROOT_DIR="./checkpoints"

# --- W&B monitor (bcp_config_*.yaml uses monitor_type: wandb) ---
export WANDB_API_KEY=""
# export WANDB_BASE_URL=""

# --- Optional ---
# export BENCHMARK_RESULTS_DIR="./benchmark_results"
