#!/bin/sh
#SBATCH --job-name=compress_bon
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH -t 0-08:00:00
#SBATCH -c 8
#SBATCH --output=/n/home04/rrinberg/catered_out/compress_bon/compress_bon_%j.out
#SBATCH --error=/n/home04/rrinberg/catered_out/compress_bon/compress_bon_%j.err
#SBATCH --mail-user=royrinberg@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --account=kempner_sham_lab

# Compress Best-of-N Experiment Results
# Reads text data from JSONL and computes compression ratios using GPU
#
# Usage:
#   # Compress just-ask AIME data
#   sbatch scripts/compress_best_of_n_results.sbatch
#
#   # Custom input file
#   sbatch --export=INPUT=data/simon_best_of_n/just_ask_steps_1_2_mbpp_20260122_085227.jsonl scripts/compress_best_of_n_results.sbatch
#
#   # Quick test (5 problems)
#   sbatch --export=LIMIT=5 scripts/compress_best_of_n_results.sbatch

ml python
module load cuda cudnn
export CONDA_ENVS_PATH=/n/holylabs/vadhan_lab/Lab/rrinberg/envs/conda/envs
export CONDA_PKGS_DIRS=/n/holylabs/vadhan_lab/Lab/rrinberg/envs/conda/pkgs

module load gcc/15.2.0-fasrc01
mamba activate MATS-inference2

export HF_HOME="/n/home04/rrinberg/data_dir/HF_cache"
echo "Hugging Face cache set to: $HF_HOME"

set -x

BASE="/n/home04/rrinberg/code/MATS/model-compression-bit-limiting"

# Create log directory
mkdir -p /n/home04/rrinberg/catered_out/compress_bon

# Default input file
INPUT_FILE="${INPUT:-${BASE}/data/simon_best_of_n/just_ask_steps_1_2_aime_20260122_085224.jsonl}"
OUTPUT_DIR="${BASE}/results/compression_best_of_n"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Compress Best-of-N Experiment Results"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Input: $INPUT_FILE"
echo "Started: $(date)"
echo "=========================================="

cd ${BASE}
export PYTHONPATH="${PYTHONPATH}:${BASE}"

# Build command
CMD="python3 scripts/compress_best_of_n_results.py"
CMD="$CMD $INPUT_FILE"
CMD="$CMD --compression-model meta-llama/Llama-3.1-8B"
CMD="$CMD --device cuda"
CMD="$CMD --n-values 1,3,5,10"
CMD="$CMD --verbose"

if [ -n "${LIMIT}" ]; then
    CMD="$CMD --limit ${LIMIT}"
fi

if [ -n "${OUTPUT}" ]; then
    CMD="$CMD --output ${OUTPUT}"
else
    # Auto-generate output filename
    BASENAME=$(basename "$INPUT_FILE" .jsonl)
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    CMD="$CMD --output ${OUTPUT_DIR}/compression_${BASENAME}_${TIMESTAMP}.json"
fi

echo "Running: $CMD"
echo ""

$CMD

echo ""
echo "=========================================="
echo "Compression Complete!"
echo "Finished: $(date)"
echo "=========================================="
