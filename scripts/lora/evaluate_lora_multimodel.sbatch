#!/bin/sh
#SBATCH --job-name=eval_lora_mm
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH -t 0-40:00
#SBATCH -c 8
#SBATCH --output=/n/home04/rrinberg/catered_out/eval_lora_%j.out
#SBATCH --error=/n/home04/rrinberg/catered_out/eval_lora_%j.err
#SBATCH --account=kempner_sham_lab

# Usage:
#   sbatch --export=MODEL=mistral scripts/lora/evaluate_lora_multimodel.sbatch
#   sbatch --export=MODEL=llama-base scripts/lora/evaluate_lora_multimodel.sbatch
#   sbatch --export=MODEL=llama-instruct scripts/lora/evaluate_lora_multimodel.sbatch

ml python
module load cuda cudnn
export CONDA_ENVS_PATH=/n/holylabs/vadhan_lab/Lab/rrinberg/envs/conda/envs
export CONDA_PKGS_DIRS=/n/holylabs/vadhan_lab/Lab/rrinberg/envs/conda/pkgs

module load gcc/15.2.0-fasrc01
mamba activate MATS-inference2

export HF_HOME="/n/home04/rrinberg/data_dir/HF_cache"
echo "Hugging Face cache set to: $HF_HOME"

set -x

# Default to mistral if not specified
MODEL=${MODEL:-mistral}

BASE="/n/home04/rrinberg/code/MATS/model-compression-bit-limiting"
DATA_ROOT="/n/netscratch/sham_lab/Lab/rrinberg/compression"
CLUSTERS_ROOT="${DATA_ROOT}/enwik9-clustered/clusters"

# Set model-specific paths and HF model ID
if [ "$MODEL" = "mistral" ]; then
    HF_MODEL="mistralai/Mistral-7B-Instruct-v0.2"
    LORAS_ROOT="${DATA_ROOT}/enwik9-loras-mistral"
elif [ "$MODEL" = "llama-base" ]; then
    HF_MODEL="meta-llama/Llama-3.1-8B"
    LORAS_ROOT="${DATA_ROOT}/enwik9-loras-llama-base"
elif [ "$MODEL" = "llama-instruct" ]; then
    HF_MODEL="meta-llama/Llama-3.1-8B-Instruct"
    LORAS_ROOT="${DATA_ROOT}/enwik9-loras-llama-instruct"
else
    echo "Unknown model: $MODEL. Use 'mistral', 'llama-base', or 'llama-instruct'"
    exit 1
fi

# Use existing mistral loras if model-specific ones don't exist yet
if [ ! -d "$LORAS_ROOT" ]; then
    echo "LoRAs not found at $LORAS_ROOT, checking if training is complete..."
    exit 1
fi

echo "========================================"
echo "Model: $MODEL ($HF_MODEL)"
echo "Clusters: $CLUSTERS_ROOT"
echo "LoRAs: $LORAS_ROOT"
echo "Bit precision: 120"
echo "========================================"

python3 ${BASE}/scripts/lora/evaluate_lora_compression.py \
    --clusters-root "$CLUSTERS_ROOT" \
    --loras-root "$LORAS_ROOT" \
    --base-model "$HF_MODEL" \
    --num-clusters 10 \
    --max-samples 50 \
    --min-tokens 200 \
    --split test \
    --compression \
    --bit-precision 120 \
    --output-dir "${BASE}/results/lora_evaluation_enwik9_${MODEL}_120bit"

echo "Done evaluating LoRA compression for $MODEL"
