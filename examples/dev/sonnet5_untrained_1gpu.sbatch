#!/bin/bash
# Dev harness run on UMass Unity: 1x L40S, gpu-preempt, <=2h (spec §13.2).
# Nothing trains here, so preemption costs at most one run -- but the cost
# guard's spend log and the lineage JSONL are written as the run goes, so a
# preempted run still leaves usable data behind.
#
#   sbatch examples/dev/sonnet5_untrained_1gpu.sbatch
#
#SBATCH -J adacom-dev
#SBATCH -p gpu-preempt
#SBATCH -t 02:00:00
#SBATCH --gpus=1
#SBATCH --constraint=l40s
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -o logs/adacom-dev-%j.out
#SBATCH -e logs/adacom-dev-%j.err

set -euo pipefail

module load conda/latest 2>/dev/null || true
conda activate "${ADACOM_ENV:-adacom}"

# Workspace scratch, not $HOME: checkpoints, run dirs and the BM25 index are large.
export WORKSPACE="${WORKSPACE:?set WORKSPACE to an HPC Workspace allocation}"
export ADACOM_RUN_DIR="${ADACOM_RUN_DIR:-$WORKSPACE/adacom_runs/dev_$SLURM_JOB_ID}"
export CHECKPOINT_ROOT_DIR="${CHECKPOINT_ROOT_DIR:-$WORKSPACE/checkpoints}"
export BENCHMARK_RESULTS_DIR="${BENCHMARK_RESULTS_DIR:-$ADACOM_RUN_DIR/benchmark_results}"
export MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the Qwen3-4B-Instruct-2507 dir}"
export BROWSECOMP_PATH="${BROWSECOMP_PATH:?set BROWSECOMP_PATH to the BrowseComp-Plus root}"

# Secrets via the environment only; never commit a filled-in export_template.sh.
: "${ANTHROPIC_API_KEY:?export ANTHROPIC_API_KEY before submitting}"

export HF_HOME="${HF_HOME:-$WORKSPACE/hf_cache}"
export TOKENIZERS_PARALLELISM=false
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_$SLURM_JOB_ID}"

mkdir -p "$ADACOM_RUN_DIR/lineage" "$ADACOM_RUN_DIR/cost" logs

nvidia-smi
echo "run dir: $ADACOM_RUN_DIR"

srun trinity run --config examples/dev/sonnet5_untrained_1gpu.yaml

# What to look at afterwards:
#   $ADACOM_RUN_DIR/lineage/task_*/run_*.jsonl   per-step pre/post context + lineage
#   $ADACOM_RUN_DIR/cost/spend.jsonl             per-rollout tokens and dollars
#   grep MEMORY_LOCK / MEMORY_VALIDITY in the benchmark_results logs
