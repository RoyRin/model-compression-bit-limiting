#!/bin/bash
#SBATCH --job-name=baselines_3x
#SBATCH --output=/n/home04/rrinberg/catered_out/baselines_3x_%j.out
#SBATCH --error=/n/home04/rrinberg/catered_out/baselines_3x_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Run the normal baselines script 3 times with temperature>0 to get variance.
# Each trial saves to a separate subdirectory under results/model-baselines-3x/
#
# Usage:
#   sbatch scripts/run_baselines_3x.sbatch                         # v4.5 models
#   sbatch --export=OLD_MODELS=1 scripts/run_baselines_3x.sbatch   # v3.5 models
#   sbatch --export=PARALLEL=10 scripts/run_baselines_3x.sbatch

echo "Starting 3x baseline evaluation at $(date)"
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

# Determine model version
if [ "$OLD_MODELS" = "1" ]; then
    MODEL_FLAG="--use-old-models"
    VERSION="v3.5"
else
    MODEL_FLAG=""
    VERSION="v4.5"
fi

PARALLEL=${PARALLEL:-10}
BASE_DIR="results/model-baselines-3x/${VERSION}"

for TRIAL in 1 2 3; do
    TRIAL_DIR="${BASE_DIR}/trial_${TRIAL}"
    echo ""
    echo "============================================================"
    echo "TRIAL ${TRIAL} of 3 -> ${TRIAL_DIR}"
    echo "Started at $(date)"
    echo "============================================================"

    CMD="python scripts/run_all_baselines.py --output-dir ${TRIAL_DIR} --parallel ${PARALLEL} --temperature 1.0 ${MODEL_FLAG}"
    echo "Running: $CMD"
    $CMD

    echo "Trial ${TRIAL} completed at $(date)"
done

echo ""
echo "All 3 trials completed at $(date)"
