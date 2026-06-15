#!/bin/bash
#SBATCH --job-name=qa_100q
#SBATCH --output=/n/home04/rrinberg/catered_out/qa_100q_%j.out
#SBATCH --error=/n/home04/rrinberg/catered_out/qa_100q_%j.err
#SBATCH --time=72:00:00
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# QA Sweep with 100 questions - BLC, QA, QA+ for haiku SLM
#
# Usage:
#   sbatch scripts/run_qa_100q.sbatch                    # All datasets
#   sbatch --export=DATASET=gsm8k scripts/run_qa_100q.sbatch
#   sbatch --export=DATASET=aime scripts/run_qa_100q.sbatch

echo "Starting QA 100Q Sweep at $(date)"
echo "Job ID: $SLURM_JOB_ID"

cd /n/home04/rrinberg/code/MATS/model-compression-bit-limiting

ml python
module load cuda cudnn
export CONDA_ENVS_PATH=/n/holylabs/vadhan_lab/Lab/rrinberg/envs/conda/envs
export CONDA_PKGS_DIRS=/n/holylabs/vadhan_lab/Lab/rrinberg/envs/conda/pkgs
module load gcc/15.2.0-fasrc01
mamba activate MATS-inference2

export HF_HOME="/n/home04/rrinberg/data_dir/HF_cache"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PYTHONUNBUFFERED=1
export ANTHROPIC_KEY_FILE="anthropic_high_prio.key"

# Build command - use 100 questions, BLC/QA/QA+ only, iterative mode with parallel workers
# HLE limited to 100 very hard questions (otherwise 2128 problems would take ~40h)
CMD="python lossy_compression/core/run_qa_sweep.py --num-questions 100 --blc-qa-only --iterative --parallel 10 --baseline-dir results/model-baselines/v4.5 --output-dir results/qa-sweep-100q --hle-very-hard-limit 100"

if [ -z "$DATASET" ]; then
    CMD="$CMD --all"
else
    CMD="$CMD --dataset $DATASET"

    if [ -n "$SUBJECT" ]; then
        CMD="$CMD --subject $SUBJECT"
    fi

    if [ -n "$FORMAT" ]; then
        CMD="$CMD --format $FORMAT"
    fi
fi

echo "Running: $CMD"
echo ""
$CMD

echo ""
echo "Completed at $(date)"
