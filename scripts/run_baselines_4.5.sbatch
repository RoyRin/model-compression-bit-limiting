#!/bin/bash
#SBATCH --job-name=baselines_4.5
#SBATCH --output=/n/home04/rrinberg/catered_out/baselines_4.5_%j.out
#SBATCH --error=/n/home04/rrinberg/catered_out/baselines_4.5_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Baseline evaluation with Claude 4.5 models (haiku, sonnet, opus) and optionally GPT-OSS-120B
# Results saved to lossy_compression/results/baselines_4.5/
#
# Usage:
#   sbatch scripts/run_baselines_4.5.sbatch                    # Run all datasets (Claude only, iterative)
#   sbatch --export=BATCH=1 scripts/run_baselines_4.5.sbatch   # Use batch API (50% cheaper, more efficient)
#   sbatch --export=PARALLEL=6 scripts/run_baselines_4.5.sbatch  # Parallel iterative (6 workers)
#   sbatch --export=DATASET=gsm8k scripts/run_baselines_4.5.sbatch
#   sbatch --export=DATASET=gpqa,FORMAT=freeform scripts/run_baselines_4.5.sbatch
#   sbatch --export=GPT_OSS=1 scripts/run_baselines_4.5.sbatch  # Include GPT-OSS-120B (runs last)
#   sbatch --export=PARALLEL=6,GPT_OSS=1 scripts/run_baselines_4.5.sbatch  # Parallel + GPT-OSS

echo "Starting 4.5 baseline evaluation at $(date)"
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

# Use the default anthropic.key (low priority)
# No need to set ANTHROPIC_KEY_FILE since anthropic.key is the default

# Output directory for 4.5 model baselines
OUTPUT_DIR="lossy_compression/results/baselines_4.5"
mkdir -p "$OUTPUT_DIR"

echo "Output directory: $OUTPUT_DIR"
echo "Models: haiku=claude-haiku-4-5-20251001, sonnet=claude-sonnet-4-5-20250929, opus=claude-opus-4-5-20251101"
echo ""

# Build command
CMD="python scripts/run_all_baselines.py --output-dir $OUTPUT_DIR"

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

if [ "$GPT_OSS" = "1" ]; then
    CMD="$CMD --include-gpt-oss"
fi

if [ "$BATCH" = "1" ]; then
    CMD="$CMD --batch"
fi

if [ -n "$PARALLEL" ]; then
    CMD="$CMD --parallel $PARALLEL"
fi

echo "Running: $CMD"
$CMD

echo ""
echo "Completed at $(date)"
