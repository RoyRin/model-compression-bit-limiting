#!/bin/bash
#SBATCH --job-name=gptoss_baselines
#SBATCH --output=/n/home04/rrinberg/catered_out/gptoss_baselines_%j.out
#SBATCH --error=/n/home04/rrinberg/catered_out/gptoss_baselines_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Add GPT-OSS-120B baselines to existing v4.5 baseline files.
# Saves copies to results/model-baselines/v4.5-gptoss/ (does not overwrite originals).
#
# Usage:
#   sbatch scripts/add_gptoss_baselines.sbatch
#   sbatch --export=PARALLEL=10 scripts/add_gptoss_baselines.sbatch
#   sbatch --export=DATASET=aime scripts/add_gptoss_baselines.sbatch

echo "Starting GPT-OSS baselines at $(date)"
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

PARALLEL=${PARALLEL:-6}

CMD="python scripts/add_gptoss_baselines.py --all --parallel $PARALLEL"

if [ -n "$DATASET" ]; then
    CMD="python scripts/add_gptoss_baselines.py --dataset $DATASET --parallel $PARALLEL"
fi

echo "Running: $CMD"
$CMD

echo "Completed at $(date)"
