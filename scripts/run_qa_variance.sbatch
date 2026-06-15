#!/bin/bash
#SBATCH --job-name=qa_variance
#SBATCH --output=/n/home04/rrinberg/catered_out/qa_variance_%j.out
#SBATCH --error=/n/home04/rrinberg/catered_out/qa_variance_%j.err
#SBATCH --time=72:00:00
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# QA Variance Experiment - Multiple trials for SLM=haiku configurations
#
# Runs K trials of each (haiku, LLM, Q-model) combination to compute
# standard deviations. Only SLM=haiku (9 combinations per dataset).
#
# Usage:
#   sbatch scripts/run_qa_variance.sbatch                           # v4.5, 3 trials
#   sbatch --export=OLD_MODELS=1 scripts/run_qa_variance.sbatch     # v3.5
#   sbatch --export=TRIALS=5 scripts/run_qa_variance.sbatch         # 5 trials
#   sbatch --export=PARALLEL=10 scripts/run_qa_variance.sbatch      # Parallel mode
#   sbatch --export=DATASET=gsm8k scripts/run_qa_variance.sbatch    # Single dataset
#   sbatch --export=OLD_MODELS=1,TRIALS=5,PARALLEL=10 scripts/run_qa_variance.sbatch

echo "Starting QA variance experiment at $(date)"
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
    VERSION_FLAG="--version v3.5"
    VERSION="v3.5"
    echo "Using OLD models (v3.5): claude-3-5-haiku"
else
    VERSION_FLAG="--version v4.5"
    VERSION="v4.5"
    echo "Using 4.5 models: claude-haiku-4-5"
fi

# Set directories
BASELINE_DIR="results/model-baselines/${VERSION}"
OUTPUT_DIR="results/qa-variance/${VERSION}"
mkdir -p "$OUTPUT_DIR"

echo "Baseline input: $BASELINE_DIR"
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
PARALLEL_FLAG=""
if [ -n "$PARALLEL" ]; then
    PARALLEL_FLAG="--parallel $PARALLEL"
    echo "Parallel workers: $PARALLEL"
else
    PARALLEL_FLAG="--parallel 10"
    echo "Parallel workers: 10 (default)"
fi

# Dataset flag
DATASET_FLAG="--all"
if [ -n "$DATASET" ]; then
    DATASET_FLAG="--dataset $DATASET"
    echo "Dataset: $DATASET"
else
    echo "Datasets: all"
fi

echo ""
echo "============================================================"
echo "Running QA variance experiment (SLM=haiku, 9 combinations)"
echo "============================================================"

python scripts/run_qa_variance.py \
    $DATASET_FLAG \
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
