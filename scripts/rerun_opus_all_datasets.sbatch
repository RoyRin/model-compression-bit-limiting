#!/bin/bash
#SBATCH --job-name=opus_rerun
#SBATCH --output=/n/home04/rrinberg/catered_out/opus_rerun/opus_rerun_%j.out
#SBATCH --error=/n/home04/rrinberg/catered_out/opus_rerun/opus_rerun_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=sapphire,seas_compute
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Rerun Opus on all datasets with automatic retry on rate limits
#
# Usage:
#   sbatch scripts/rerun_opus_all_datasets.sbatch           # Fresh start
#   sbatch --export=RESUME=1 scripts/rerun_opus_all_datasets.sbatch  # Resume from partial
#   sbatch --export=DATASET=gsm8k scripts/rerun_opus_all_datasets.sbatch  # Single dataset

set -e

cd /n/home04/rrinberg/code/MATS/model-compression-bit-limiting
source ~/source_MATS.sh
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

mkdir -p /n/home04/rrinberg/catered_out/opus_rerun

echo "=========================================="
echo "OPUS RERUN - All Datasets"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "Resume: ${RESUME:-0}"
echo "=========================================="

# Build command with optional resume flag
CMD="python scripts/rerun_opus_all_datasets.py"
CMD="$CMD --dataset ${DATASET:-all}"
CMD="$CMD --output-dir results/opus_rerun"

if [ "${RESUME:-0}" = "1" ]; then
    CMD="$CMD --resume"
fi

echo "Running: $CMD"
$CMD

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ COMPLETED SUCCESSFULLY"
else
    echo "❌ FAILED WITH EXIT CODE $EXIT_CODE"
fi
echo "Finished: $(date)"
echo "=========================================="

exit $EXIT_CODE
