#!/bin/bash
#SBATCH --job-name=easy_sanity
#SBATCH --output=/n/home04/rrinberg/catered_out/easy_sanity_%j.out
#SBATCH --error=/n/home04/rrinberg/catered_out/easy_sanity_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Easy Problems Sanity Check
#
# Verifies that easy problems (ones Haiku got right in baseline) are still
# answered correctly on retry. Measures stochastic variance / failure rate.
#
# Runs: SLM=haiku, LLM=opus, Q={haiku, sonnet, opus}
#
# Usage:
#   sbatch scripts/run_easy_sanity_check.sbatch                    # v4.5
#   sbatch --export=OLD_MODELS=1 scripts/run_easy_sanity_check.sbatch  # v3.5

echo "Starting easy sanity check at $(date)"
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

BASELINE_DIR="results/model-baselines/${VERSION}"
OUTPUT_DIR="results/easy-sanity-check/${VERSION}"
mkdir -p "$OUTPUT_DIR"

echo "Baseline input: $BASELINE_DIR"
echo "Output: $OUTPUT_DIR"

# Parallel flag
PARALLEL_FLAG="--parallel 10"
if [ -n "$PARALLEL" ]; then
    PARALLEL_FLAG="--parallel $PARALLEL"
    echo "Parallel workers: $PARALLEL"
else
    echo "Parallel workers: 10 (default)"
fi

# Limit flag (for testing)
LIMIT_FLAG=""
if [ -n "$LIMIT" ]; then
    LIMIT_FLAG="--limit $LIMIT"
    echo "Limit: $LIMIT problems per dataset"
fi

echo ""
echo "============================================================"
echo "Running easy sanity check (SLM=haiku, LLM=opus, 3 Q-models)"
echo "Checking retry success rate on EASY problems"
echo "============================================================"

python scripts/run_easy_sanity_check.py \
    --all \
    $VERSION_FLAG \
    --baseline-dir "$BASELINE_DIR" \
    --output-dir "$OUTPUT_DIR" \
    $PARALLEL_FLAG \
    $LIMIT_FLAG

echo ""
echo "============================================================"
echo "COMPLETED at $(date)"
echo "============================================================"
echo "Results: $OUTPUT_DIR"
