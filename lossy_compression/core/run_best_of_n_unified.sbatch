#!/bin/bash
#SBATCH --job-name=best_of_n_unified
#SBATCH --output=/n/home04/rrinberg/catered_out/best_of_n_unified/best_of_n_unified_%j.out
#SBATCH --error=/n/home04/rrinberg/catered_out/best_of_n_unified/best_of_n_unified_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=sapphire,seas_gpu,gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Unified Best-of-N Experiment
# Runs all 3 approaches (temperature, single_prompt, just_ask) with multiple trials
#
# Usage:
#   # Standard mode (slow, ~17 hours)
#   sbatch run_best_of_n_unified.sbatch
#
#   # Batch mode (fast, ~1-2 hours)
#   sbatch --export=BATCH=1 run_best_of_n_unified.sbatch
#
#   # Quick test
#   sbatch --export=NUM_PROBLEMS=5,NUM_TRIALS=1,BATCH=1 run_best_of_n_unified.sbatch

set -e

# Setup
cd /n/home04/rrinberg/code/MATS/model-compression-bit-limiting
source ~/source_MATS.sh
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PYTHONUNBUFFERED=1
export ANTHROPIC_API_KEY=$(cat SECRETS/anthropic.key)

# Create log directory
mkdir -p /n/home04/rrinberg/catered_out/best_of_n_unified

echo "=========================================="
echo "Unified Best-of-N Experiment"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Batch mode: ${BATCH:-0}"
echo "Started: $(date)"
echo "=========================================="

# Build command
CMD="python lossy_compression/core/run_best_of_n_unified.py"
CMD="$CMD --num-problems ${NUM_PROBLEMS:-90}"
CMD="$CMD --num-trials ${NUM_TRIALS:-3}"
CMD="$CMD --format pdf"

if [ "${BATCH:-0}" = "1" ]; then
    CMD="$CMD --batch"
fi

echo "Running: $CMD"
echo ""

$CMD

echo ""
echo "=========================================="
echo "Experiment Complete!"
echo "Finished: $(date)"
echo "=========================================="
