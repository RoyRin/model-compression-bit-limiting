#!/bin/bash
#SBATCH --job-name=math_qa_sweep
#SBATCH --output=/n/home04/rrinberg/catered_out/math_qa_sweep/math_qa_sweep_%j.out
#SBATCH --error=/n/home04/rrinberg/catered_out/math_qa_sweep/math_qa_sweep_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=seas_compute

# Run MATH Q&A compression sweep (36 combinations per subject: 3 SLM x 4 LLM x 3 Q)
#
# Usage:
#   # Run all subjects
#   sbatch run_math_qa_sweep.sbatch
#
#   # Run specific subject
#   sbatch --export=SUBJECT=algebra run_math_qa_sweep.sbatch
#
#   # Limit problems for testing
#   sbatch --export=NUM_PROBLEMS=5 run_math_qa_sweep.sbatch

# Create output directory
mkdir -p /n/home04/rrinberg/catered_out/math_qa_sweep

cd /n/home04/rrinberg/code/MATS/model-compression-bit-limiting
source ~/source_MATS.sh
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Configuration (can override with --export)
SUBJECT=${SUBJECT:-""}  # Empty = all subjects
NUM_PROBLEMS=${NUM_PROBLEMS:-""}  # Empty = all problems
MAX_QUESTIONS=${MAX_QUESTIONS:-30}
BATCH_SIZE=${BATCH_SIZE:-10}

echo "=============================================="
echo "MATH Q&A Compression Sweep"
echo "Subject filter: ${SUBJECT:-all}"
echo "Num problems: ${NUM_PROBLEMS:-all}"
echo "Max questions: ${MAX_QUESTIONS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Start time: $(date)"
echo "=============================================="

# Build command
CMD="./lossy_compression/benchmarks/math/run_math_qa_sweep.sh"
CMD="$CMD --max-questions ${MAX_QUESTIONS}"
CMD="$CMD --batch-size ${BATCH_SIZE}"

if [ -n "${SUBJECT}" ]; then
    CMD="$CMD --subject ${SUBJECT}"
fi

if [ -n "${NUM_PROBLEMS}" ]; then
    CMD="$CMD --num-problems ${NUM_PROBLEMS}"
fi

echo "Running: ${CMD}"
${CMD}

echo ""
echo "Finished at: $(date)"
echo "=============================================="
