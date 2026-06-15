#!/bin/sh
#SBATCH --job-name=merge_rag
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH -t 0-00:30
#SBATCH -c 2
#SBATCH --output=/n/home04/rrinberg/catered_out/merge_rag_%j.out
#SBATCH --error=/n/home04/rrinberg/catered_out/merge_rag_%j.err
#SBATCH --mail-user=royrinberg@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --account=kempner_sham_lab

# Merge results from parallel RAG LoRA evaluation
# Usage: sbatch --dependency=afterok:ARRAY_JOB_ID scripts/lora/merge_rag_lora_full.sbatch

ml python
mamba activate MATS-inference2

BASE="/n/home04/rrinberg/code/MATS/model-compression-bit-limiting"

python3 ${BASE}/scripts/lora/merge_rag_lora_full_results.py \
    --base-dir "${BASE}/results/rag_lora_evaluation_enwik9-50-full" \
    --num-jobs 6

echo "Merge complete!"
