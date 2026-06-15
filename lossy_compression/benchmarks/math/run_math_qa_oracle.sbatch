#!/bin/bash
#SBATCH --job-name=math_qa_oracle
#SBATCH --output=/n/home04/rrinberg/catered_out/math_qa_oracle/math_qa_oracle_%j.out
#SBATCH --error=/n/home04/rrinberg/catered_out/math_qa_oracle/math_qa_oracle_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=seas_compute

# Run MATH Q&A compression with Oracle mode (dataset solution as reference)
#
# Usage:
#   sbatch run_math_qa_oracle.sbatch
#   sbatch --export=SUBJECT=geometry run_math_qa_oracle.sbatch

# Create output directory
mkdir -p /n/home04/rrinberg/catered_out/math_qa_oracle

cd /n/home04/rrinberg/code/MATS/model-compression-bit-limiting
source ~/source_MATS.sh
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Configuration (can override with --export)
SUBJECT=${SUBJECT:-"algebra"}
DIFFICULTY=${DIFFICULTY:-"not_easy"}
SLM=${SLM:-"haiku"}
LLM=${LLM:-"opus"}
QUESTION_MODEL=${QUESTION_MODEL:-"haiku"}
MAX_QUESTIONS=${MAX_QUESTIONS:-30}
BATCH_SIZE=${BATCH_SIZE:-10}
NUM_PROBLEMS=${NUM_PROBLEMS:-""}  # Empty = all problems

echo "=============================================="
echo "MATH Q&A Oracle Mode"
echo "Subject: ${SUBJECT}"
echo "Difficulty: ${DIFFICULTY}"
echo "Num problems: ${NUM_PROBLEMS:-all}"
echo "SLM: ${SLM}, LLM: ${LLM}, Question: ${QUESTION_MODEL}"
echo "Max questions: ${MAX_QUESTIONS}"
echo "Start time: $(date)"
echo "=============================================="

# Build command
CMD="python lossy_compression/benchmarks/math/evaluate_math_qa_compression.py"
CMD="${CMD} --subject ${SUBJECT}"
CMD="${CMD} --difficulty ${DIFFICULTY}"
CMD="${CMD} --slm ${SLM}"
CMD="${CMD} --llm ${LLM}"
CMD="${CMD} --question-model ${QUESTION_MODEL}"
CMD="${CMD} --max-questions ${MAX_QUESTIONS}"
CMD="${CMD} --batch --batch-size ${BATCH_SIZE}"
CMD="${CMD} --oracle"

if [ -n "${NUM_PROBLEMS}" ]; then
    CMD="${CMD} --num-problems ${NUM_PROBLEMS}"
fi

echo "Running: ${CMD}"
${CMD}

echo ""
echo "Finished at: $(date)"
echo "=============================================="
