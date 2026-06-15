#!/bin/bash
#SBATCH --job-name=enwik9_%a
#SBATCH --output=/n/home04/rrinberg/catered_out/enwik9/enwik9_%a_%j.out
#SBATCH --error=/n/home04/rrinberg/catered_out/enwik9/enwik9_%a_%j.err
#SBATCH --time=2:00:00
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --account=kempner_sham_lab
#SBATCH --array=0-3

echo "Starting enwik9 baselines at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
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

# Model array
MODELS=("mistral-7b" "llama-3.1-8b" "llama-3.1-8b-instruct" "qwen3-30b-a3b")
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "Running model: $MODEL"
echo "Python: $(which python)"

# Create output directory
mkdir -p /n/home04/rrinberg/catered_out/enwik9

# Run single model baseline
python scripts/run_enwik9_baselines.py \
    --models $MODEL \
    --max-tokens 15000 \
    --num-chars 500000 \
    --output-json /n/home04/rrinberg/catered_out/enwik9/enwik9_${MODEL}_15k_$(date +%Y%m%d_%H%M%S).json \
    --verbose

echo "Done with $MODEL at $(date)"
