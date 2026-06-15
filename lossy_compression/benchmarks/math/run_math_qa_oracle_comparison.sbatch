#!/bin/bash
#SBATCH --job-name=math_qa_oracle
#SBATCH --output=/n/home04/rrinberg/catered_out/math_qa_oracle/math_qa_oracle_%j.out
#SBATCH --error=/n/home04/rrinberg/catered_out/math_qa_oracle/math_qa_oracle_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=seas_compute

# Run MATH Q&A compression comparing Oracle vs Non-Oracle modes
#
# Usage:
#   sbatch run_math_qa_oracle_comparison.sbatch
#   sbatch --export=SUBJECT=geometry run_math_qa_oracle_comparison.sbatch

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

echo "=============================================="
echo "MATH Q&A Oracle vs Non-Oracle Comparison"
echo "Subject: ${SUBJECT}"
echo "Difficulty: ${DIFFICULTY}"
echo "SLM: ${SLM}, LLM: ${LLM}, Question: ${QUESTION_MODEL}"
echo "Max questions: ${MAX_QUESTIONS}"
echo "Start time: $(date)"
echo "=============================================="

# Run with Oracle mode
echo ""
echo "=============================================="
echo "Running WITH Oracle mode"
echo "=============================================="
python lossy_compression/benchmarks/math/evaluate_math_qa_compression.py \
    --subject ${SUBJECT} \
    --difficulty ${DIFFICULTY} \
    --slm ${SLM} \
    --llm ${LLM} \
    --question-model ${QUESTION_MODEL} \
    --max-questions ${MAX_QUESTIONS} \
    --batch --batch-size ${BATCH_SIZE} \
    --oracle

# Run without Oracle mode
echo ""
echo "=============================================="
echo "Running WITHOUT Oracle mode (standard)"
echo "=============================================="
python lossy_compression/benchmarks/math/evaluate_math_qa_compression.py \
    --subject ${SUBJECT} \
    --difficulty ${DIFFICULTY} \
    --slm ${SLM} \
    --llm ${LLM} \
    --question-model ${QUESTION_MODEL} \
    --max-questions ${MAX_QUESTIONS} \
    --batch --batch-size ${BATCH_SIZE}

echo ""
echo "Finished at: $(date)"
echo "=============================================="
