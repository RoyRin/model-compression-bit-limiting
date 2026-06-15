#!/bin/bash
#SBATCH --job-name=robust_baselines
#SBATCH --output=/n/home04/rrinberg/catered_out/robust_baselines_%j.out
#SBATCH --error=/n/home04/rrinberg/catered_out/robust_baselines_%j.err
#SBATCH --time=72:00:00
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Robust Baseline Experiment - Multiple trials per model
#
# Runs each model K times per problem for robust difficulty classification.
# A problem is "not-easy" only if the model got it wrong K/K times.
#
# Usage:
#   sbatch scripts/run_robust_baselines.sbatch                          # v4.5, 3 trials, all datasets
#   sbatch --export=OLD_MODELS=1 scripts/run_robust_baselines.sbatch    # v3.5
#   sbatch --export=TRIALS=5 scripts/run_robust_baselines.sbatch        # 5 trials
#   sbatch --export=PARALLEL=10 scripts/run_robust_baselines.sbatch     # Parallel mode
#   sbatch --export=DATASET=gsm8k scripts/run_robust_baselines.sbatch   # Single dataset
#   sbatch --export=OLD_MODELS=1,TRIALS=5,PARALLEL=10 scripts/run_robust_baselines.sbatch

echo "Starting robust baselines at $(date)"
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

# Load Anthropic API key (high priority for better rate limits)
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

# Set output directory
OUTPUT_DIR="results/model-baselines-robust/${VERSION}"
mkdir -p "$OUTPUT_DIR"
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
    PARALLEL_FLAG="--parallel 6"
    echo "Parallel workers: 6 (default)"
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
echo "Running robust baselines (3 models × K trials per problem)"
echo "============================================================"

# Models flag (default: haiku only for speed)
MODELS_FLAG="--models haiku"
if [ -n "$MODELS" ]; then
    MODELS_FLAG="--models $MODELS"
    echo "Models: $MODELS"
else
    echo "Models: haiku (default - for speed)"
fi

python scripts/run_robust_baselines.py \
    $DATASET_FLAG \
    $VERSION_FLAG \
    --output-dir "$OUTPUT_DIR" \
    $TRIALS_FLAG \
    $PARALLEL_FLAG \
    $MODELS_FLAG

echo ""
echo "============================================================"
echo "COMPLETED at $(date)"
echo "============================================================"
echo "Results: $OUTPUT_DIR"
