#!/bin/bash
#SBATCH --job-name=batch_qa_all
#SBATCH --output=/n/home04/rrinberg/catered_out/batch_qa_sweep_all/sweep_%j.out
#SBATCH --error=/n/home04/rrinberg/catered_out/batch_qa_sweep_all/sweep_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=sapphire,seas_compute
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

# Batch Q&A Sweep for All Datasets
# Runs 36 model combinations across all datasets using batch API
#
# Datasets: gsm8k, gpqa_freeform, mbpp, humaneval
# Models: 3 SLM × 4 LLM × 3 Q = 36 combinations per dataset
# Total: 36 × 4 = 144 combinations
#
# Usage:
#   # Run all datasets, all combinations
#   sbatch run_batch_qa_sweep_all_datasets.sbatch
#
#   # Run specific dataset
#   sbatch --export=DATASET=gsm8k run_batch_qa_sweep_all_datasets.sbatch
#
#   # Test with limited problems
#   sbatch --export=LIMIT=10 run_batch_qa_sweep_all_datasets.sbatch
#
#   # Run specific combination
#   sbatch --export=DATASET=gsm8k,SLM=haiku,LLM=opus,QA=haiku run_batch_qa_sweep_all_datasets.sbatch

set -e

cd /n/home04/rrinberg/code/MATS/model-compression-bit-limiting
source ~/source_MATS.sh
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PYTHONUNBUFFERED=1
export ANTHROPIC_API_KEY=$(cat SECRETS/anthropic.key)

mkdir -p /n/home04/rrinberg/catered_out/batch_qa_sweep_all

echo "=========================================="
echo "BATCH Q&A SWEEP - ALL DATASETS"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "Dataset: ${DATASET:-all}"
echo "SLM: ${SLM:-all}"
echo "LLM: ${LLM:-all}"
echo "QA: ${QA:-all}"
echo "Limit: ${LIMIT:-all}"
echo "=========================================="

# Build command
CMD="python lossy_compression/core/batch_qa_sweep_all_datasets.py"
CMD="$CMD --output-dir results/batch_qa_sweep_all"

if [ -n "$DATASET" ]; then
    CMD="$CMD --dataset $DATASET"
fi

if [ -n "$SLM" ]; then
    CMD="$CMD --slm $SLM"
fi

if [ -n "$LLM" ]; then
    CMD="$CMD --llm $LLM"
fi

if [ -n "$QA" ]; then
    CMD="$CMD --qa $QA"
fi

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
