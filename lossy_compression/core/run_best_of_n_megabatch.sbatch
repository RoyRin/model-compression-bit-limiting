#!/bin/bash
#SBATCH --job-name=best_of_n_mega
#SBATCH --output=/n/home04/rrinberg/catered_out/best_of_n_megabatch/mega_%j.out
#SBATCH --error=/n/home04/rrinberg/catered_out/best_of_n_megabatch/mega_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --account=kempner_sham_lab

# Best-of-N Megabatch Experiment
# Much faster than unified approach - submits all requests at once
#
# Usage:
#   sbatch run_best_of_n_megabatch.sbatch                    # Full run (90 problems, 3 trials)
#   sbatch --export=PROBLEMS=10,TRIALS=1 run_best_of_n_megabatch.sbatch  # Quick test
#   sbatch --export=SKIP_COMPRESS=1 run_best_of_n_megabatch.sbatch       # Skip compression

set -e

cd /n/home04/rrinberg/code/MATS/model-compression-bit-limiting

# Activate environment
ml python
module load cuda cudnn
export CONDA_ENVS_PATH=/n/holylabs/vadhan_lab/Lab/rrinberg/envs/conda/envs
export CONDA_PKGS_DIRS=/n/holylabs/vadhan_lab/Lab/rrinberg/envs/conda/pkgs
module load gcc/15.2.0-fasrc01
mamba activate MATS-inference2

export HF_HOME="/n/home04/rrinberg/data_dir/HF_cache"
echo "Hugging Face cache set to: $HF_HOME"

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_LAUNCH_BLOCKING=1
export PYTHONUNBUFFERED=1

mkdir -p /n/home04/rrinberg/catered_out/best_of_n_megabatch

echo "=========================================="
echo "BEST-OF-N MEGABATCH"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "Problems: ${PROBLEMS:-90}"
echo "Trials: ${TRIALS:-3}"
echo "=========================================="

CMD="python lossy_compression/core/run_best_of_n_megabatch.py"
CMD="$CMD --num-problems ${PROBLEMS:-90}"
CMD="$CMD --num-trials ${TRIALS:-3}"

if [ "${SKIP_COMPRESS:-0}" = "1" ]; then
    CMD="$CMD --skip-compression"
fi

echo "Running: $CMD"
$CMD

echo ""
echo "=========================================="
echo "Finished: $(date)"
echo "=========================================="
