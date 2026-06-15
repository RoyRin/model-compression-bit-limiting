#!/bin/bash
#SBATCH --job-name=math_all_models
#SBATCH --output=/n/home04/rrinberg/catered_out/math_all_models/math_all_models_%j.out
#SBATCH --error=/n/home04/rrinberg/catered_out/math_all_models/math_all_models_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=seas_compute

# Run MATH evaluation across haiku, sonnet, opus to classify problem difficulty
# Results are saved to lossy_compression/results/math_all_models_*.json

cd /n/home04/rrinberg/code/MATS/model-compression-bit-limiting
source ~/source_MATS.sh
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Configuration - can override with: sbatch --export=SUBJECT=geometry,NUM_PROBLEMS=500
SUBJECT=${SUBJECT:-"all"}
NUM_PROBLEMS=${NUM_PROBLEMS:-9999}  # Default to all problems

echo "=============================================="
echo "MATH All Models Evaluation"
echo "Subject: ${SUBJECT}"
echo "Num problems: ${NUM_PROBLEMS}"
echo "Start time: $(date)"
echo "=============================================="

python lossy_compression/benchmarks/math/run_math_all_models.py \
    --subject "${SUBJECT}" \
    --num-problems "${NUM_PROBLEMS}"

echo ""
echo "Finished at: $(date)"
echo "=============================================="
