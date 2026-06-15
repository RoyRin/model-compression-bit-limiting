#!/bin/bash
#SBATCH --job-name=qa_sweep
#SBATCH --output=/n/home04/rrinberg/catered_out/qa_sweep_%j.out
#SBATCH --error=/n/home04/rrinberg/catered_out/qa_sweep_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# QA Compression Sweep - 3^3 = 27 combinations per dataset
#
# Usage:
#   sbatch run_qa_sweep.sbatch                              # Run all datasets (batch API, 4.5 models)
#   sbatch --export=ITERATIVE=1,PARALLEL=10 run_qa_sweep.sbatch  # Iterative parallel mode
#   sbatch --export=OLD_MODELS=1 run_qa_sweep.sbatch        # Use old models (3.5)
#   sbatch --export=DATASET=gsm8k run_qa_sweep.sbatch       # GSM8K only
#   sbatch --export=DATASET=math run_qa_sweep.sbatch        # All MATH subjects
#   sbatch --export=DATASET=math,SUBJECT=algebra run_qa_sweep.sbatch
#   sbatch --export=DATASET=gpqa run_qa_sweep.sbatch        # Both GPQA formats
#   sbatch --export=DATASET=gpqa,FORMAT=mc run_qa_sweep.sbatch
#   sbatch --export=DATASET=mbpp run_qa_sweep.sbatch
#   sbatch --export=DATASET=mmlu_pro run_qa_sweep.sbatch
#   sbatch --export=ITERATIVE=1,PARALLEL=10,BASELINE_DIR=results/model-baselines run_qa_sweep.sbatch

echo "Starting QA Sweep at $(date)"
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
echo "Hugging Face cache set to: $HF_HOME"

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PYTHONUNBUFFERED=1

# Use high priority API key
export ANTHROPIC_KEY_FILE="anthropic_high_prio.key"

# Build command
CMD="python lossy_compression/core/run_qa_sweep.py"

if [ -z "$DATASET" ]; then
    CMD="$CMD --all"
else
    CMD="$CMD --dataset $DATASET"

    if [ -n "$SUBJECT" ]; then
        CMD="$CMD --subject $SUBJECT"
    fi

    if [ -n "$FORMAT" ]; then
        CMD="$CMD --format $FORMAT"
    fi
fi

if [ -n "$LIMIT" ]; then
    CMD="$CMD --limit $LIMIT"
fi

if [ -n "$BASELINE_DIR" ]; then
    CMD="$CMD --baseline-dir $BASELINE_DIR"
fi

if [ -n "$OUTPUT_DIR" ]; then
    CMD="$CMD --output-dir $OUTPUT_DIR"
fi

if [ "$ITERATIVE" = "1" ]; then
    CMD="$CMD --iterative"
fi

if [ -n "$PARALLEL" ]; then
    CMD="$CMD --parallel $PARALLEL"
fi

if [ "$OLD_MODELS" = "1" ]; then
    CMD="$CMD --use-old-models"
fi

if [ -n "$NUM_QUESTIONS" ]; then
    CMD="$CMD --num-questions $NUM_QUESTIONS"
fi

if [ -n "$TRIAL" ]; then
    CMD="$CMD --trial $TRIAL"
fi

if [ "$BLC_QA_ONLY" = "1" ]; then
    CMD="$CMD --blc-qa-only"
fi

if [ -n "$HLE_VERY_HARD_LIMIT" ]; then
    CMD="$CMD --hle-very-hard-limit $HLE_VERY_HARD_LIMIT"
fi

echo "Running: $CMD"
echo ""
$CMD

echo ""
echo "Completed at $(date)"
