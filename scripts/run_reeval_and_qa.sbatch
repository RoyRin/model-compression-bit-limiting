#!/bin/bash
#SBATCH --job-name=reeval_qa
#SBATCH --output=/n/home04/rrinberg/catered_out/reeval_qa_%j.out
#SBATCH --error=/n/home04/rrinberg/catered_out/reeval_qa_%j.err
#SBATCH --time=72:00:00
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Run all baselines + QA sweeps (skips existing results)
# Includes: GSM8K, MATH, GPQA, MBPP, MMLU Pro, HLE, AIME
#
# Usage:
#   sbatch scripts/run_reeval_and_qa.sbatch                         # v4.5 models (default)
#   sbatch --export=OLD_MODELS=1 scripts/run_reeval_and_qa.sbatch   # v3.5 models
#   sbatch --export=PARALLEL=10 scripts/run_reeval_and_qa.sbatch    # Parallel mode
#   sbatch --export=GPT_OSS=1 scripts/run_reeval_and_qa.sbatch      # Include GPT-OSS in QA sweep
#   sbatch --export=OLD_MODELS=1,PARALLEL=10,GPT_OSS=1 scripts/run_reeval_and_qa.sbatch

echo "Starting re-evaluation at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"

cd /n/home04/rrinberg/code/MATS/model-compression-bit-limiting

# Activate environment
ml python
module load cuda cudnn
export CONDA_ENVS_PATH=/n/holylabs/vadhan_lab/Lab/rrinberg/envs/conda/envs
export CONDA_PKGS_DIRS=/n/holylabs/vadhan_lab/Lab/rrinberg/envs/conda/pkgs
module load gcc/15.2.0-fasrc01
mamba activate MATS-inference2

export HF_HOME="/n/home04/rrinberg/data_dir/HF_cache"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PYTHONUNBUFFERED=1

# Determine model version
if [ "$OLD_MODELS" = "1" ]; then
    MODEL_FLAG="--use-old-models"
    VERSION="v3.5"
    echo "Using OLD models (v3.5): claude-3-5-haiku, claude-sonnet-4, claude-opus-4"
else
    MODEL_FLAG=""
    VERSION="v4.5"
    echo "Using 4.5 models: claude-haiku-4-5, claude-sonnet-4-5, claude-opus-4-5"
fi

# Set directories
BASELINE_DIR="results/model-baselines/${VERSION}"
QA_OUTPUT_DIR="results/qa-sweep/${VERSION}"
mkdir -p "$BASELINE_DIR" "$QA_OUTPUT_DIR"

echo "Baseline output: $BASELINE_DIR"
echo "QA sweep output: $QA_OUTPUT_DIR"

# Parallel flag
PARALLEL_FLAG=""
if [ -n "$PARALLEL" ]; then
    PARALLEL_FLAG="--parallel $PARALLEL"
    echo "Parallel workers: $PARALLEL"
fi

# GPT-OSS flag for QA sweep
GPT_OSS_FLAG=""
if [ "$GPT_OSS" = "1" ]; then
    GPT_OSS_FLAG="--include-gpt-oss"
    echo "GPT-OSS: ENABLED (4^3=64 combinations per dataset)"
fi

echo ""
echo "============================================================"
echo "PHASE 1: BASELINES (all datasets - skips existing)"
echo "============================================================"

# Run all baselines (skips existing results automatically)
echo ""
echo "--- Running all baselines (GSM8K, MATH, GPQA, MBPP, MMLU Pro, HLE, AIME) ---"
python scripts/run_all_baselines.py \
    --dataset all \
    --output-dir "$BASELINE_DIR" \
    $MODEL_FLAG \
    $PARALLEL_FLAG

echo ""
echo "============================================================"
echo "PHASE 2: QA SWEEPS (all datasets)"
echo "============================================================"

# QA sweep uses --iterative mode (not batch) per CLAUDE.md instructions
QA_FLAGS="--iterative"
if [ -n "$PARALLEL" ]; then
    QA_FLAGS="$QA_FLAGS --parallel $PARALLEL"
fi

# Run QA sweeps for all datasets (uses existing baselines from BASELINE_DIR)
echo ""
echo "--- Running QA sweep for all datasets ---"
python lossy_compression/core/run_qa_sweep.py \
    --all \
    --baseline-dir "$BASELINE_DIR" \
    --output-dir "$QA_OUTPUT_DIR" \
    --hle-very-hard-limit 50 \
    $MODEL_FLAG \
    $GPT_OSS_FLAG \
    $QA_FLAGS

echo ""
echo "============================================================"
echo "COMPLETED at $(date)"
echo "============================================================"
echo "Baseline results: $BASELINE_DIR"
echo "QA sweep results: $QA_OUTPUT_DIR"
