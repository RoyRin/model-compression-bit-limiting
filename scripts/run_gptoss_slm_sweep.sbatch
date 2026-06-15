#!/bin/bash
#SBATCH --job-name=gptoss_slm_sweep
#SBATCH --output=/n/home04/rrinberg/catered_out/gptoss_slm_sweep_%j.out
#SBATCH --error=/n/home04/rrinberg/catered_out/gptoss_slm_sweep_%j.err
#SBATCH --time=72:00:00
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# GPT-OSS as SLM: BLC/QA/QA+ sweep (3 configs per dataset)
#
# Configs:
#   BLC: SLM=gpt-oss, LLM=gpt-oss, Q=gpt-oss
#   QA:  SLM=gpt-oss, LLM=opus,    Q=gpt-oss
#   QA+: SLM=gpt-oss, LLM=opus,    Q=opus
#
# Usage:
#   sbatch scripts/run_gptoss_slm_sweep.sbatch
#   sbatch --export=DATASET=gsm8k scripts/run_gptoss_slm_sweep.sbatch
#   sbatch --export=PARALLEL=6 scripts/run_gptoss_slm_sweep.sbatch

echo "Starting GPT-OSS SLM sweep at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"

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
export ANTHROPIC_KEY_FILE=anthropic_high_prio.key

PARALLEL=${PARALLEL:-10}

CMD="python lossy_compression/core/run_qa_sweep.py --gpt-oss-slm --parallel $PARALLEL"

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

if [ -n "$LIMIT" ]; then
    CMD="$CMD --limit $LIMIT"
fi

if [ -n "$HLE_LIMIT" ]; then
    CMD="$CMD --hle-very-hard-limit $HLE_LIMIT"
fi

echo "Running: $CMD"
echo ""
$CMD

echo ""
echo "Completed at $(date)"
