#!/bin/bash
#SBATCH --job-name=robust_qa_var
#SBATCH --output=/n/home04/rrinberg/catered_out/robust_qa_var_%j.out
#SBATCH --error=/n/home04/rrinberg/catered_out/robust_qa_var_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Robust QA Variance - Multiple trials for SLM=haiku, LLM=opus using robust baselines
#
# Runs K trials of each (haiku, opus, Q-model) combination to compute
# standard deviations, using strict difficulty from robust baselines.
# Only 3 combinations: Q in {haiku, sonnet, opus}
#
# Usage:
#   sbatch --dependency=afterok:56766300 scripts/run_robust_qa_variance.sbatch              # v4.5
#   sbatch --dependency=afterok:56766301 --export=OLD_MODELS=1 scripts/run_robust_qa_variance.sbatch  # v3.5

echo "Starting robust QA variance at $(date)"
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
export ANTHROPIC_API_KEY=$(cat SECRETS/anthropic_high_prio.key)

# Determine model version
if [ "$OLD_MODELS" = "1" ]; then
    VERSION_FLAG="--version v3.5"
    VERSION="v3.5"
    echo "Using OLD models (v3.5)"
else
    VERSION_FLAG="--version v4.5"
    VERSION="v4.5"
    echo "Using 4.5 models"
fi

# Use ROBUST baselines (strict difficulty)
BASELINE_DIR="results/model-baselines-robust/${VERSION}"
OUTPUT_DIR="results/robust-qa-variance/${VERSION}"
mkdir -p "$OUTPUT_DIR"

echo "Baseline input (robust): $BASELINE_DIR"
echo "Output: $OUTPUT_DIR"

# Number of trials
TRIALS_FLAG="--trials 3"
if [ -n "$TRIALS" ]; then
    TRIALS_FLAG="--trials $TRIALS"
    echo "Trials: $TRIALS"
else
    echo "Trials: 3 (default)"
fi

# Parallel flag
PARALLEL_FLAG="--parallel 10"
if [ -n "$PARALLEL" ]; then
    PARALLEL_FLAG="--parallel $PARALLEL"
    echo "Parallel workers: $PARALLEL"
else
    echo "Parallel workers: 10 (default)"
fi

echo ""
echo "============================================================"
echo "Running robust QA variance (SLM=haiku, LLM=opus, 3 Q-models, 3 trials)"
echo "Using strict difficulty from robust baselines"
echo "============================================================"

python scripts/run_qa_variance.py \
    --all \
    --robust \
    $VERSION_FLAG \
    --baseline-dir "$BASELINE_DIR" \
    --output-dir "$OUTPUT_DIR" \
    $TRIALS_FLAG \
    $PARALLEL_FLAG

echo ""
echo "============================================================"
echo "COMPLETED at $(date)"
echo "============================================================"
echo "Results: $OUTPUT_DIR"
