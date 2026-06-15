#!/bin/bash
#SBATCH --job-name=retry_baseline
#SBATCH --output=/n/home04/rrinberg/catered_out/retry_baseline_%j.out
#SBATCH --error=/n/home04/rrinberg/catered_out/retry_baseline_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Retry Baseline Experiment
# Re-runs haiku on problems it got wrong to measure stochastic variance.
# This provides a null hypothesis for QA compression.
#
# Usage:
#   sbatch scripts/run_retry_baseline.sbatch                         # v4.5 models (default)
#   sbatch --export=OLD_MODELS=1 scripts/run_retry_baseline.sbatch   # v3.5 models
#   sbatch --export=PARALLEL=10 scripts/run_retry_baseline.sbatch    # Parallel mode
#   sbatch --export=OLD_MODELS=1,PARALLEL=10 scripts/run_retry_baseline.sbatch

echo "Starting retry baseline at $(date)"
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

# Load Anthropic API key
export ANTHROPIC_API_KEY=$(cat SECRETS/anthropic.key)

# Determine model version
if [ "$OLD_MODELS" = "1" ]; then
    VERSION="v3.5"
    echo "Using OLD models (v3.5): claude-3-5-haiku"
else
    VERSION="v4.5"
    echo "Using 4.5 models: claude-haiku-4-5"
fi

# Set directories
BASELINE_DIR="results/model-baselines/${VERSION}"
OUTPUT_DIR="results/retry_baseline/${VERSION}"
mkdir -p "$OUTPUT_DIR"

echo "Baseline input: $BASELINE_DIR"
echo "Output: $OUTPUT_DIR"

# Parallel flag
PARALLEL_FLAG=""
if [ -n "$PARALLEL" ]; then
    PARALLEL_FLAG="--parallel $PARALLEL"
    echo "Parallel workers: $PARALLEL"
else
    PARALLEL_FLAG="--parallel 10"
    echo "Parallel workers: 10 (default)"
fi

echo ""
echo "============================================================"
echo "Running retry baseline for all datasets"
echo "============================================================"

python scripts/run_retry_baseline.py \
    --all \
    --version "$VERSION" \
    --baseline-dir "$BASELINE_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --temperature 1.0 \
    $PARALLEL_FLAG

echo ""
echo "============================================================"
echo "COMPLETED at $(date)"
echo "============================================================"
echo "Results: $OUTPUT_DIR"
