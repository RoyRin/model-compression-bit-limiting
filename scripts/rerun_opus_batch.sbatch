#!/bin/bash
#SBATCH --job-name=opus_batch
#SBATCH --output=/n/home04/rrinberg/catered_out/opus_batch/opus_batch_%j.out
#SBATCH --error=/n/home04/rrinberg/catered_out/opus_batch/opus_batch_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=sapphire,seas_compute
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

# Rerun Opus on all datasets using Anthropic Batch API
# Much faster than sequential calls!

set -e

cd /n/home04/rrinberg/code/MATS/model-compression-bit-limiting
source ~/source_MATS.sh
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

mkdir -p /n/home04/rrinberg/catered_out/opus_batch

echo "=========================================="
echo "OPUS BATCH RERUN"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "=========================================="

# Run full pipeline: submit, poll, download
python scripts/rerun_opus_batch.py --run

echo ""
echo "=========================================="
echo "Finished: $(date)"
echo "=========================================="
