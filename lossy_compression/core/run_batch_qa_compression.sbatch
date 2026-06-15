#!/bin/bash
#SBATCH --job-name=batch_qa
#SBATCH --output=/n/home04/rrinberg/catered_out/batch_qa/batch_qa_%j.out
#SBATCH --error=/n/home04/rrinberg/catered_out/batch_qa/batch_qa_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=sapphire,seas_compute
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

# Batch QA Compression Pipeline
# Uses Anthropic Message Batches API for efficient parallel processing
# 6 batch steps: Proposal → Evaluation → Questions → Answers → Update → Final Eval
#
# Model roles:
#   SLM: proposal and update (default: haiku)
#   QA: question generation (default: haiku)
#   LLM: evaluation and answers (default: opus)
#
# Usage:
#   # Run all MATH subjects, non-easy problems, with haiku/haiku/opus
#   sbatch run_batch_qa_compression.sbatch
#
#   # Test with 10 problems per subject
#   sbatch --export=LIMIT=10 run_batch_qa_compression.sbatch
#
#   # Run specific dataset
#   sbatch --export=DATASET=math_algebra run_batch_qa_compression.sbatch

set -e

cd /n/home04/rrinberg/code/MATS/model-compression-bit-limiting
source ~/source_MATS.sh
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PYTHONUNBUFFERED=1

mkdir -p /n/home04/rrinberg/catered_out/batch_qa

echo "=========================================="
echo "BATCH QA COMPRESSION"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "=========================================="
echo "Dataset: ${DATASET:-math_all}"
echo "Difficulties: ${DIFFICULTIES:-medium,hard,very_hard}"
echo "SLM (proposal/update): ${SLM:-haiku}"
echo "QA (questions): ${QA:-haiku}"
echo "LLM (eval/answers): ${LLM:-opus}"
echo "Limit: ${LIMIT:-all}"
echo "=========================================="

# Build command
CMD="python lossy_compression/core/batch_qa_compression.py"
CMD="$CMD --dataset ${DATASET:-math_all}"
CMD="$CMD --slm ${SLM:-haiku}"
CMD="$CMD --qa ${QA:-haiku}"
CMD="$CMD --llm ${LLM:-opus}"
CMD="$CMD --difficulties ${DIFFICULTIES:-medium,hard,very_hard}"
CMD="$CMD --output-dir results/batch_qa"

if [ -n "$LIMIT" ]; then
    CMD="$CMD --limit $LIMIT"
fi

echo "Running: $CMD"
echo ""

$CMD

echo ""
echo "=========================================="
echo "Finished: $(date)"
echo "=========================================="
