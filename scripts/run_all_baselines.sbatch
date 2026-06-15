#!/bin/bash
#SBATCH --job-name=all_baselines
#SBATCH --output=/n/home04/rrinberg/catered_out/all_baselines_%j.out
#SBATCH --error=/n/home04/rrinberg/catered_out/all_baselines_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Comprehensive baseline evaluation script
# Runs haiku, sonnet, and opus on all datasets to classify problem difficulty
#
# Usage:
#   sbatch scripts/run_all_baselines.sbatch                    # Run all datasets (4.5 models)
#   sbatch --export=OLD_MODELS=1 scripts/run_all_baselines.sbatch  # Use old models (3.5/4)
#   sbatch --export=PARALLEL=10 scripts/run_all_baselines.sbatch   # Parallel iterative
#   sbatch --export=OLD_MODELS=1,PARALLEL=10 scripts/run_all_baselines.sbatch  # Old + parallel
#   sbatch --export=GPT_OSS=1,PARALLEL=10 scripts/run_all_baselines.sbatch  # Include GPT-OSS
#   sbatch --export=OUTPUT_DIR=results/model-baselines scripts/run_all_baselines.sbatch  # Custom output
#   sbatch --export=DATASET=gsm8k scripts/run_all_baselines.sbatch   # GSM8K only
#   sbatch --export=DATASET=math,SUBJECT=algebra scripts/run_all_baselines.sbatch
#   sbatch --export=DATASET=gpqa,FORMAT=mc scripts/run_all_baselines.sbatch
#   sbatch --export=RESUME=1 scripts/run_all_baselines.sbatch  # Resume from partial

echo "Starting baseline evaluation at $(date)"
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

# Build command with optional arguments
CMD="python scripts/run_all_baselines.py"

if [ -n "$DATASET" ]; then
    CMD="$CMD --dataset $DATASET"
fi

if [ -n "$SUBJECT" ]; then
    CMD="$CMD --subject $SUBJECT"
fi

if [ -n "$FORMAT" ]; then
    CMD="$CMD --format $FORMAT"
fi

if [ -n "$NUM_PROBLEMS" ]; then
    CMD="$CMD --num-problems $NUM_PROBLEMS"
fi

if [ "$RESUME" = "1" ]; then
    CMD="$CMD --resume"
fi

if [ "$OLD_MODELS" = "1" ]; then
    CMD="$CMD --use-old-models"
fi

if [ -n "$PARALLEL" ]; then
    CMD="$CMD --parallel $PARALLEL"
fi

if [ "$GPT_OSS" = "1" ]; then
    CMD="$CMD --include-gpt-oss"
fi

if [ -n "$OUTPUT_DIR" ]; then
    CMD="$CMD --output-dir $OUTPUT_DIR"
fi

echo "Running: $CMD"
$CMD

echo "Completed at $(date)"
