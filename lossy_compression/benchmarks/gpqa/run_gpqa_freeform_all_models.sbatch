#!/bin/bash
#SBATCH --job-name=gpqa_freeform
#SBATCH --output=/n/home04/rrinberg/catered_out/gpqa_freeform/gpqa_freeform_%j.out
#SBATCH --error=/n/home04/rrinberg/catered_out/gpqa_freeform/gpqa_freeform_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=seas_compute

# Run GPQA-Freeform evaluation across all models (haiku, sonnet, opus)
# Uses LLM-as-judge for free-form answer evaluation
#
# Usage:
#   sbatch run_gpqa_freeform_all_models.sbatch
#   sbatch --export=NUM_PROBLEMS=50,JUDGE_MODEL=opus run_gpqa_freeform_all_models.sbatch

# Create output directory
mkdir -p /n/home04/rrinberg/catered_out/gpqa_freeform

cd /n/home04/rrinberg/code/MATS/model-compression-bit-limiting
source ~/source_MATS.sh
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Configuration
NUM_PROBLEMS=${NUM_PROBLEMS:-""}  # Empty = all problems
JUDGE_MODEL=${JUDGE_MODEL:-"sonnet"}

echo "=============================================="
echo "GPQA-Freeform Multi-Model Evaluation"
echo "Num problems: ${NUM_PROBLEMS:-all}"
echo "Judge model: ${JUDGE_MODEL}"
echo "Start time: $(date)"
echo "=============================================="

# Build command
CMD="python lossy_compression/benchmarks/gpqa/run_gpqa_freeform_all_models.py"
CMD="${CMD} --judge-model ${JUDGE_MODEL}"

if [ -n "${NUM_PROBLEMS}" ]; then
    CMD="${CMD} --num-problems ${NUM_PROBLEMS}"
fi

echo "Running: ${CMD}"
${CMD}

echo ""
echo "Finished at: $(date)"
echo "=============================================="
