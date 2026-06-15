#!/bin/bash
#SBATCH --job-name=freeform_eval
#SBATCH --output=/n/home04/rrinberg/catered_out/freeform_eval_%j.out
#SBATCH --error=/n/home04/rrinberg/catered_out/freeform_eval_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Runs ONLY gpqa_freeform and mmlu_pro (baselines + QA sweeps)
# These use LLM-as-judge evaluation with the stricter prompt
#
# Usage:
#   sbatch scripts/run_freeform_eval.sbatch                         # v4.5 models (default)
#   sbatch --export=OLD_MODELS=1 scripts/run_freeform_eval.sbatch   # v3.5 models
#   sbatch --export=PARALLEL=10 scripts/run_freeform_eval.sbatch    # Parallel mode

echo "Starting freeform evaluation at $(date)"
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
    echo "Using OLD models (v3.5): claude-3-5-haiku, claude-sonnet-4, claude-opus-4"
else
    MODEL_FLAG=""
    VERSION="v4.5"
    echo "Using 4.5 models: claude-haiku-4-5, claude-sonnet-4-5, claude-opus-4-5"
fi

# Set directories
BASELINE_DIR="results/model-baselines/${VERSION}"
QA_OUTPUT_DIR="results/qa-sweep/${VERSION}"
mkdir -p "$BASELINE_DIR" "$QA_OUTPUT_DIR"

echo "Baseline output: $BASELINE_DIR"
echo "QA sweep output: $QA_OUTPUT_DIR"

# Parallel flag
PARALLEL_FLAG=""
if [ -n "$PARALLEL" ]; then
    PARALLEL_FLAG="--parallel $PARALLEL"
    echo "Parallel workers: $PARALLEL"
fi

echo ""
echo "============================================================"
echo "PHASE 1: BASELINES (GPQA Freeform + MMLU Pro only)"
echo "============================================================"

# Run GPQA freeform baseline
echo ""
echo "--- Running GPQA Freeform baseline ---"
python scripts/run_all_baselines.py \
    --dataset gpqa \
    --format freeform \
    --output-dir "$BASELINE_DIR" \
    $MODEL_FLAG \
    $PARALLEL_FLAG

# Run MMLU Pro baseline
echo ""
echo "--- Running MMLU Pro baseline ---"
python scripts/run_all_baselines.py \
    --dataset mmlu_pro \
    --output-dir "$BASELINE_DIR" \
    $MODEL_FLAG \
    $PARALLEL_FLAG

echo ""
echo "============================================================"
echo "PHASE 2: QA SWEEPS (GPQA Freeform + MMLU Pro only)"
echo "============================================================"

# QA sweep uses --iterative mode
QA_FLAGS="--iterative"
if [ -n "$PARALLEL" ]; then
    QA_FLAGS="$QA_FLAGS --parallel $PARALLEL"
fi

# Run QA sweep for GPQA freeform
echo ""
echo "--- Running QA sweep for GPQA Freeform ---"
python lossy_compression/core/run_qa_sweep.py \
    --dataset gpqa \
    --format freeform \
    --baseline-dir "$BASELINE_DIR" \
    --output-dir "$QA_OUTPUT_DIR" \
    $MODEL_FLAG \
    $QA_FLAGS

# Run QA sweep for MMLU Pro
echo ""
echo "--- Running QA sweep for MMLU Pro ---"
python lossy_compression/core/run_qa_sweep.py \
    --dataset mmlu_pro \
    --baseline-dir "$BASELINE_DIR" \
    --output-dir "$QA_OUTPUT_DIR" \
    $MODEL_FLAG \
    $QA_FLAGS

echo ""
echo "============================================================"
echo "COMPLETED at $(date)"
echo "============================================================"
echo "Baseline results: $BASELINE_DIR"
echo "QA sweep results: $QA_OUTPUT_DIR"
