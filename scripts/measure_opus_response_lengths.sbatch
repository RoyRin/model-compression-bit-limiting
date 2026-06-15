#!/bin/bash
#SBATCH --job-name=opus_resp_len
#SBATCH --output=/n/home04/rrinberg/catered_out/opus_resp_len_%j.out
#SBATCH --error=/n/home04/rrinberg/catered_out/opus_resp_len_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Measure Opus response lengths for non-easy problems across all datasets.
# Samples 30 problems per difficulty (medium/hard/very_hard) per dataset.
#
# Usage:
#   sbatch scripts/measure_opus_response_lengths.sbatch
#   sbatch --export=PARALLEL=10 scripts/measure_opus_response_lengths.sbatch
#   sbatch --export=DATASET=gsm8k scripts/measure_opus_response_lengths.sbatch
#   sbatch --export=RESUME=1 scripts/measure_opus_response_lengths.sbatch

echo "Starting Opus response length measurement at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"

cd /n/home04/rrinberg/code/MATS/model-compression-bit-limiting

# Activate environment
ml python
module load cuda cudnn
export CONDA_ENVS_PATH=/n/holylabs/vadhan_lab/Lab/rrinberg/envs/conda/envs
export CONDA_PKGS_DIRS=/n/holylabs/vadhan_lab/Lab/rrinberg/envs/conda/pkgs
module load gcc/15.2.0-fasrc01
mamba activate MATS-inference2

export HF_HOME="/n/home04/rrinberg/data_dir/HF_cache"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PYTHONUNBUFFERED=1

CMD="python scripts/measure_opus_response_lengths.py"

# Default to 6 parallel workers
PARALLEL=${PARALLEL:-6}
CMD="$CMD --parallel $PARALLEL"

if [ -n "$DATASET" ]; then
    CMD="$CMD --dataset $DATASET"
fi

if [ "$RESUME" = "1" ]; then
    CMD="$CMD --resume"
fi

if [ -n "$MAX_PER_DIFF" ]; then
    CMD="$CMD --max-per-difficulty $MAX_PER_DIFF"
fi

echo "Running: $CMD"
$CMD

echo "Completed at $(date)"
