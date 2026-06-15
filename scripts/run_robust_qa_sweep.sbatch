#!/bin/bash
#SBATCH --job-name=robust_qa_sweep
#SBATCH --output=/n/home04/rrinberg/catered_out/robust_qa_sweep_%j.out
#SBATCH --error=/n/home04/rrinberg/catered_out/robust_qa_sweep_%j.err
#SBATCH --time=72:00:00
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Robust QA Sweep - Uses robust baselines with strict difficulty
#
# Runs QA compression on problems that are "not-easy" according to
# strict classification (model got wrong ALL K trials).
#
# Usage:
#   sbatch --dependency=afterok:56765901 scripts/run_robust_qa_sweep.sbatch              # v4.5
#   sbatch --dependency=afterok:56765902 --export=OLD_MODELS=1 scripts/run_robust_qa_sweep.sbatch  # v3.5
#   sbatch --export=PARALLEL=10 scripts/run_robust_qa_sweep.sbatch
#   sbatch --export=OLD_MODELS=1,PARALLEL=10 scripts/run_robust_qa_sweep.sbatch

echo "Starting robust QA sweep at $(date)"
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

# Load Anthropic API key
export ANTHROPIC_API_KEY=$(cat SECRETS/anthropic.key)

# Determine model version
if [ "$OLD_MODELS" = "1" ]; then
    MODEL_FLAG="--use-old-models"
    VERSION="v3.5"
    echo "Using OLD models (v3.5)"
else
    MODEL_FLAG=""
    VERSION="v4.5"
    echo "Using 4.5 models"
fi

# Set directories
BASELINE_DIR="results/model-baselines-robust/${VERSION}"
OUTPUT_DIR="results/robust-qa-sweep/${VERSION}"
mkdir -p "$OUTPUT_DIR"

echo "Baseline input: $BASELINE_DIR"
echo "Output: $OUTPUT_DIR"

# Parallel flag
PARALLEL_FLAG=""
if [ -n "$PARALLEL" ]; then
    PARALLEL_FLAG="--parallel $PARALLEL"
    echo "Parallel workers: $PARALLEL"
else
    PARALLEL_FLAG="--parallel 10"
    echo "Parallel workers: 10 (default)"
fi

echo ""
echo "============================================================"
echo "Running robust QA sweep (strict difficulty from 3-trial baselines)"
echo "============================================================"

python lossy_compression/core/run_qa_sweep.py \
    --all \
    --robust \
    --baseline-dir "$BASELINE_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --iterative \
    --hle-very-hard-limit 50 \
    $MODEL_FLAG \
    $PARALLEL_FLAG

echo ""
echo "============================================================"
echo "COMPLETED at $(date)"
echo "============================================================"
echo "Results: $OUTPUT_DIR"
