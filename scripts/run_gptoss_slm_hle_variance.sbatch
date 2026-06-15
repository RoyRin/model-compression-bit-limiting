#!/bin/bash
#SBATCH --job-name=gptoss_hle_var
#SBATCH --output=/n/home04/rrinberg/catered_out/gptoss_hle_var_%j.out
#SBATCH --error=/n/home04/rrinberg/catered_out/gptoss_hle_var_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# GPT-OSS as SLM on HLE: 3 trials for variance/std-dev
#
# Runs 3 trials with different random seeds for HLE subsampling.
# Each trial: 100 very-hard + all medium/hard problems
#
# Usage:
#   sbatch scripts/run_gptoss_slm_hle_variance.sbatch

echo "Starting GPT-OSS HLE variance experiment at $(date)"
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
HLE_LIMIT=${HLE_LIMIT:-100}

echo "Running 3 trials with --hle-very-hard-limit $HLE_LIMIT"
echo ""

for TRIAL in 1 2 3; do
    echo "============================================================"
    echo "TRIAL $TRIAL of 3"
    echo "============================================================"

    python lossy_compression/core/run_qa_sweep.py \
        --gpt-oss-slm \
        --parallel $PARALLEL \
        --dataset hle \
        --hle-very-hard-limit $HLE_LIMIT \
        --trial $TRIAL

    echo ""
    echo "Trial $TRIAL completed at $(date)"
    echo ""
done

echo ""
echo "All 3 trials completed at $(date)"
