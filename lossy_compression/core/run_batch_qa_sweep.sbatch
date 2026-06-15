#!/bin/bash
#SBATCH --job-name=batch_qa_sweep
#SBATCH --output=/n/home04/rrinberg/catered_out/batch_qa_sweep/sweep_%j.out
#SBATCH --error=/n/home04/rrinberg/catered_out/batch_qa_sweep/sweep_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=sapphire,seas_compute
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

# Batch QA Compression Sweep
# Runs all 36 model combinations (3 SLM x 4 LLM x 3 Q) using batch API
# 6 steps per combination: Proposal → Eval → Questions → Answers → Update → Final Eval
#
# Usage:
#   # Run full sweep (all subjects, all combinations)
#   sbatch run_batch_qa_sweep.sbatch
#
#   # Run specific subject
#   sbatch --export=SUBJECT=algebra run_batch_qa_sweep.sbatch
#
#   # Test with limited problems
#   sbatch --export=LIMIT=10 run_batch_qa_sweep.sbatch

set -e

cd /n/home04/rrinberg/code/MATS/model-compression-bit-limiting
source ~/source_MATS.sh
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PYTHONUNBUFFERED=1

mkdir -p /n/home04/rrinberg/catered_out/batch_qa_sweep

echo "=========================================="
echo "BATCH QA SWEEP"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "Subject: ${SUBJECT:-all}"
echo "Limit: ${LIMIT:-all}"
echo "=========================================="

# Build command
CMD="python lossy_compression/core/run_batch_qa_sweep.py"
CMD="$CMD --output-dir results/batch_qa_sweep"

if [ -n "$SUBJECT" ]; then
    CMD="$CMD --subject $SUBJECT"
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
