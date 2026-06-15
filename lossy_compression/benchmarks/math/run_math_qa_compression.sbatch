#!/bin/bash
#SBATCH --job-name=math_qa
#SBATCH --output=/n/home04/rrinberg/catered_out/math_qa_compression/math_qa_%j.out
#SBATCH --error=/n/home04/rrinberg/catered_out/math_qa_compression/math_qa_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=seas_compute

# Run MATH Q&A compression (20 Questions) experiment
#
# Usage:
#   # Default: algebra, medium difficulty
#   sbatch run_math_qa_compression.sbatch
#
#   # Custom subject and difficulty
#   sbatch --export=SUBJECT=geometry,DIFFICULTY=hard run_math_qa_compression.sbatch
#
#   # Specific number of problems
#   sbatch --export=NUM_PROBLEMS=20,DIFFICULTY=medium run_math_qa_compression.sbatch

cd /n/home04/rrinberg/code/MATS/model-compression-bit-limiting
source ~/source_MATS.sh
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Configuration (can override with --export)
SUBJECT=${SUBJECT:-"algebra"}
DIFFICULTY=${DIFFICULTY:-"medium"}
NUM_PROBLEMS=${NUM_PROBLEMS:-""}  # Empty = all problems in category
SLM=${SLM:-"haiku"}
LLM=${LLM:-"opus"}
QUESTION_MODEL=${QUESTION_MODEL:-"haiku"}
MAX_QUESTIONS=${MAX_QUESTIONS:-30}
BATCH_SIZE=${BATCH_SIZE:-10}

echo "=============================================="
echo "MATH Q&A Compression (20 Questions)"
echo "Subject: ${SUBJECT}"
echo "Difficulty: ${DIFFICULTY}"
echo "Num problems: ${NUM_PROBLEMS:-all}"
echo "SLM: ${SLM}, LLM: ${LLM}, Question: ${QUESTION_MODEL}"
echo "Max questions: ${MAX_QUESTIONS}"
echo "Batch size: ${BATCH_SIZE}"
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

if [ -n "${NUM_PROBLEMS}" ]; then
    CMD="${CMD} --num-problems ${NUM_PROBLEMS}"
fi

echo "Running: ${CMD}"
${CMD}

echo ""
echo "Finished at: $(date)"
echo "=============================================="
