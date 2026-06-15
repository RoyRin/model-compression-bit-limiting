#!/bin/bash
#SBATCH --job-name=humaneval_all_models
#SBATCH --output=/n/home04/rrinberg/catered_out/humaneval_all_models/humaneval_all_models_%j.out
#SBATCH --error=/n/home04/rrinberg/catered_out/humaneval_all_models/humaneval_all_models_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=seas_compute

# Run HumanEval+ evaluation across all models (haiku, sonnet, opus)
# to classify problem difficulty
#
# Usage:
#   sbatch run_humaneval_all_models.sbatch
#   sbatch --export=NUM_PROBLEMS=50 run_humaneval_all_models.sbatch

# Create output directory
mkdir -p /n/home04/rrinberg/catered_out/humaneval_all_models

cd /n/home04/rrinberg/code/MATS/model-compression-bit-limiting
source ~/source_MATS.sh
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Configuration
NUM_PROBLEMS=${NUM_PROBLEMS:-""}  # Empty = all problems (164)

echo "=============================================="
echo "HumanEval+ Multi-Model Evaluation"
echo "Num problems: ${NUM_PROBLEMS:-all}"
echo "Start time: $(date)"
echo "=============================================="

# Build command
CMD="python lossy_compression/benchmarks/humaneval/run_humaneval_all_models.py"

if [ -n "${NUM_PROBLEMS}" ]; then
    CMD="${CMD} --num-problems ${NUM_PROBLEMS}"
fi

echo "Running: ${CMD}"
${CMD}

echo ""
echo "Finished at: $(date)"
echo "=============================================="
