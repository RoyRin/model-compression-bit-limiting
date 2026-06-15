#!/bin/bash
#SBATCH --job-name=qa_variance
#SBATCH --output=/n/home04/rrinberg/catered_out/qa_variance_proper_%j.out
#SBATCH --error=/n/home04/rrinberg/catered_out/qa_variance_proper_%j.err
#SBATCH --time=72:00:00
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# QA Variance using the exact run_qa_sweep.py protocol.
#
# Usage:
#   sbatch scripts/run_qa_variance_proper.sbatch
#   sbatch --export=PARALLEL=10,TRIALS=3 scripts/run_qa_variance_proper.sbatch
#   sbatch --export=DATASET=gsm8k scripts/run_qa_variance_proper.sbatch

echo "Starting QA variance (proper protocol) at $(date)"
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
export ANTHROPIC_KEY_FILE=anthropic_high_prio.key

PARALLEL=${PARALLEL:-18}
TRIALS=${TRIALS:-3}
HLE_LIMIT=${HLE_LIMIT:-100}

CMD="python scripts/run_qa_variance_proper.py --all --parallel $PARALLEL --trials $TRIALS --hle-very-hard-limit $HLE_LIMIT"

if [ -n "$DATASET" ]; then
    CMD="python scripts/run_qa_variance_proper.py --dataset $DATASET --parallel $PARALLEL --trials $TRIALS --hle-very-hard-limit $HLE_LIMIT"
fi

if [ -n "$LIMIT" ]; then
    CMD="$CMD --limit $LIMIT"
fi

echo "Running: $CMD"
$CMD

echo "Completed at $(date)"
