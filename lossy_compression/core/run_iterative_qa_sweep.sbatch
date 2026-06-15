#!/bin/bash
#SBATCH --job-name=iter_qa_sweep
#SBATCH --output=/n/home04/rrinberg/catered_out/iter_qa_sweep_%j.out
#SBATCH --error=/n/home04/rrinberg/catered_out/iter_qa_sweep_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Iterative QA Sweep with Judge-Based Thresholding
#
# Runs BLC (haiku→haiku→haiku) and QA (haiku→opus→haiku) with:
# - LLM uses its own solution as reference (no gold answer)
# - Quality thresholding (judge scores 1-10, threshold=7)
# - 2 rounds × 5 questions = 10 questions max
# - Temperature = 0 for all steps
#
# Usage:
#   sbatch run_iterative_qa_sweep.sbatch                                    # Both judge modes, 10 parallel
#   sbatch --export=JUDGE_MODE=objective run_iterative_qa_sweep.sbatch      # Objective only
#   sbatch --export=JUDGE_MODE=comparison run_iterative_qa_sweep.sbatch     # Comparison only
#   sbatch --export=DATASET=gsm8k run_iterative_qa_sweep.sbatch             # GSM8K only
#   sbatch --export=DATASET=math,SUBJECT=algebra run_iterative_qa_sweep.sbatch
#   sbatch --export=PARALLEL=5 run_iterative_qa_sweep.sbatch                # Fewer parallel workers
#   sbatch --export=LIMIT=5 run_iterative_qa_sweep.sbatch                   # Quick test
#   sbatch --export=QUALITY_THRESHOLD=9 run_iterative_qa_sweep.sbatch       # Higher judge threshold
#   sbatch --export=GOLD_JUDGE=1 run_iterative_qa_sweep.sbatch              # Judge gets gold answer
#   sbatch --export=EASY_ONLY=1 run_iterative_qa_sweep.sbatch              # Easy problems only (regression test)
#   Default: LIMIT=150 non-easy problems per dataset, both judge modes

echo "Starting Iterative QA Sweep at $(date)"
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

# Use high priority API key
export ANTHROPIC_KEY_FILE="anthropic_high_prio.key"

# Defaults
PARALLEL=${PARALLEL:-10}
LIMIT=${LIMIT:-150}
BASELINE_DIR=${BASELINE_DIR:-results/model-baselines/v4.5}

# Build base command
BASE_CMD="python lossy_compression/core/run_iterative_qa_sweep.py"

if [ -z "$DATASET" ]; then
    BASE_CMD="$BASE_CMD --all"
else
    BASE_CMD="$BASE_CMD --dataset $DATASET"
    if [ -n "$SUBJECT" ]; then
        BASE_CMD="$BASE_CMD --subject $SUBJECT"
    fi
    if [ -n "$FORMAT" ]; then
        BASE_CMD="$BASE_CMD --format $FORMAT"
    fi
fi

BASE_CMD="$BASE_CMD --limit $LIMIT"

if [ -n "$BASELINE_DIR" ]; then
    BASE_CMD="$BASE_CMD --baseline-dir $BASELINE_DIR"
fi

if [ -n "$OUTPUT_DIR" ]; then
    BASE_CMD="$BASE_CMD --output-dir $OUTPUT_DIR"
fi

BASE_CMD="$BASE_CMD --parallel $PARALLEL"

if [ "$OLD_MODELS" = "1" ]; then
    BASE_CMD="$BASE_CMD --use-old-models"
fi

if [ -n "$QUALITY_THRESHOLD" ]; then
    BASE_CMD="$BASE_CMD --quality-threshold $QUALITY_THRESHOLD"
fi

if [ "$GOLD_JUDGE" = "1" ]; then
    BASE_CMD="$BASE_CMD --gold-judge"
fi

if [ "$EASY_ONLY" = "1" ]; then
    BASE_CMD="$BASE_CMD --easy-only"
fi

# Determine which judge modes to run
if [ -n "$JUDGE_MODE" ]; then
    # Run only the specified mode
    JUDGE_MODES="$JUDGE_MODE"
else
    # Run both: objective first, then comparison
    JUDGE_MODES="objective comparison"
fi

for MODE in $JUDGE_MODES; do
    echo ""
    echo "============================================================"
    echo "Running judge mode: $MODE at $(date)"
    echo "============================================================"
    CMD="$BASE_CMD --judge-mode $MODE"
    echo "Running: $CMD"
    echo ""
    $CMD
    echo ""
    echo "Finished judge mode: $MODE at $(date)"
done

echo ""
echo "============================================================"
echo "ALL COMPLETED at $(date)"
echo "============================================================"
