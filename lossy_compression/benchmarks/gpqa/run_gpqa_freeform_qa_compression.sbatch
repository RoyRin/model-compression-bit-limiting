#!/bin/bash
#SBATCH --job-name=gpqa_ff_qa
#SBATCH --output=/n/home04/rrinberg/catered_out/gpqa_freeform_qa/gpqa_freeform_qa_%j.out
#SBATCH --error=/n/home04/rrinberg/catered_out/gpqa_freeform_qa/gpqa_freeform_qa_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=seas_compute

# Run GPQA-Freeform Q&A compression evaluation
#
# Usage:
#   sbatch run_gpqa_freeform_qa_compression.sbatch
#   sbatch --export=DIFFICULTY=hard,SLM=haiku,LLM=opus run_gpqa_freeform_qa_compression.sbatch

mkdir -p /n/home04/rrinberg/catered_out/gpqa_freeform_qa

cd /n/home04/rrinberg/code/MATS/model-compression-bit-limiting
source ~/source_MATS.sh
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Configuration
DIFFICULTY=${DIFFICULTY:-"medium"}
SLM=${SLM:-"haiku"}
LLM=${LLM:-"opus"}
QUESTION_MODEL=${QUESTION_MODEL:-"haiku"}
JUDGE_MODEL=${JUDGE_MODEL:-"sonnet"}
MAX_QUESTIONS=${MAX_QUESTIONS:-30}
BATCH_SIZE=${BATCH_SIZE:-10}
NUM_PROBLEMS=${NUM_PROBLEMS:-""}
BASELINE=${BASELINE:-""}

echo "=============================================="
echo "GPQA-Freeform Q&A Compression"
echo "Difficulty: ${DIFFICULTY}"
echo "SLM: ${SLM}, LLM: ${LLM}, Question: ${QUESTION_MODEL}"
echo "Judge: ${JUDGE_MODEL}"
echo "Max questions: ${MAX_QUESTIONS}"
echo "Start time: $(date)"
echo "=============================================="

CMD="python lossy_compression/benchmarks/gpqa/evaluate_gpqa_freeform_qa_compression.py"
CMD="${CMD} --difficulty ${DIFFICULTY}"
CMD="${CMD} --slm ${SLM}"
CMD="${CMD} --llm ${LLM}"
CMD="${CMD} --question-model ${QUESTION_MODEL}"
CMD="${CMD} --judge-model ${JUDGE_MODEL}"
CMD="${CMD} --max-questions ${MAX_QUESTIONS}"
CMD="${CMD} --batch --batch-size ${BATCH_SIZE}"

if [ -n "${NUM_PROBLEMS}" ]; then
    CMD="${CMD} --num-problems ${NUM_PROBLEMS}"
fi

if [ -n "${BASELINE}" ]; then
    CMD="${CMD} --baseline-results ${BASELINE}"
fi

echo "Running: ${CMD}"
${CMD}

echo ""
echo "Finished at: $(date)"
echo "=============================================="
