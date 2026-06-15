#!/bin/bash
#SBATCH --job-name=mbpp_all_models
#SBATCH --output=/n/home04/rrinberg/catered_out/mbpp_all_models/mbpp_all_models_%j.out
#SBATCH --error=/n/home04/rrinberg/catered_out/mbpp_all_models/mbpp_all_models_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=seas_compute

# Run MBPP evaluation across haiku, sonnet, opus to classify problem difficulty
# Results are saved to lossy_compression/results/mbpp_all_models_*.json

cd /n/home04/rrinberg/code/MATS/model-compression-bit-limiting
source ~/source_MATS.sh
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Configuration - can override with: sbatch --export=NUM_PROBLEMS=200,SPLIT=test
NUM_PROBLEMS=${NUM_PROBLEMS:-9999}  # Default to all problems (500 in test split)
SPLIT=${SPLIT:-"test"}

echo "=============================================="
echo "MBPP All Models Evaluation"
echo "Split: ${SPLIT}"
echo "Num problems: ${NUM_PROBLEMS}"
echo "Start time: $(date)"
echo "=============================================="

python lossy_compression/benchmarks/mbpp/run_mbpp_all_models.py \
    --num-problems "${NUM_PROBLEMS}" \
    --split "${SPLIT}"

echo ""
echo "Finished at: $(date)"
echo "=============================================="
