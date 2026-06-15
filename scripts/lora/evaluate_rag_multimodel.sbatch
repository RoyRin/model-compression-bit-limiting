#!/bin/sh
#SBATCH --job-name=eval_rag_mm
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH -t 0-24:00
#SBATCH -c 8
#SBATCH --output=/n/home04/rrinberg/catered_out/eval_rag_lora_%j.out
#SBATCH --error=/n/home04/rrinberg/catered_out/eval_rag_lora_%j.err
#SBATCH --account=kempner_sham_lab

# Usage:
#   sbatch --export=MODEL=mistral scripts/lora/evaluate_rag_multimodel.sbatch
#   sbatch --export=MODEL=llama-base scripts/lora/evaluate_rag_multimodel.sbatch
#   sbatch --export=MODEL=llama-instruct scripts/lora/evaluate_rag_multimodel.sbatch
#   sbatch --export=MODEL=mistral,MODE=accuracy scripts/lora/evaluate_rag_multimodel.sbatch

ml python
module load cuda cudnn
export CONDA_ENVS_PATH=/n/holylabs/vadhan_lab/Lab/rrinberg/envs/conda/envs
export CONDA_PKGS_DIRS=/n/holylabs/vadhan_lab/Lab/rrinberg/envs/conda/pkgs

module load gcc/15.2.0-fasrc01
mamba activate MATS-inference2

# Install sentence-transformers if not present
pip install -q sentence-transformers

export HF_HOME="/n/home04/rrinberg/data_dir/HF_cache"
echo "Hugging Face cache set to: $HF_HOME"

set -x

# Default to mistral
MODEL=${MODEL:-mistral}
MODE=${MODE:-both}
MAX_SAMPLES=${MAX_SAMPLES:-50}

BASE="/n/home04/rrinberg/code/MATS/model-compression-bit-limiting"
DATA_ROOT="/n/netscratch/sham_lab/Lab/rrinberg/compression"

CLUSTERS_ROOT="${DATA_ROOT}/enwik9-clustered/clusters"
INDEX_PATH="${DATA_ROOT}/enwik9-clustered/lora_rag_index"

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

# Check if LoRAs exist
if [ ! -d "$LORAS_ROOT" ]; then
    echo "LoRAs not found at $LORAS_ROOT"
    exit 1
fi

echo "========================================"
echo "Model: $MODEL ($HF_MODEL)"
echo "Mode: $MODE"
echo "Clusters: $CLUSTERS_ROOT"
echo "LoRAs: $LORAS_ROOT"
echo "Index: $INDEX_PATH"
echo "Bit precision: 120"
echo "========================================"

# Build command based on mode
MODE_FLAG=""
if [ "$MODE" = "accuracy" ]; then
    MODE_FLAG="--accuracy-only"
elif [ "$MODE" = "compression" ]; then
    MODE_FLAG="--compression-only"
fi

python3 ${BASE}/scripts/lora/evaluate_rag_lora.py \
    --clusters-root "$CLUSTERS_ROOT" \
    --loras-root "$LORAS_ROOT" \
    --index-path "$INDEX_PATH" \
    --base-model "$HF_MODEL" \
    --num-clusters 10 \
    --max-samples $MAX_SAMPLES \
    --min-tokens 200 \
    --k 10 \
    --bit-precision 120 \
    --output-dir "${BASE}/results/rag_lora_enwik9_${MODEL}_120bit" \
    --with-correct-lora \
    $MODE_FLAG \
    --cluster-ids 0,1,2,3,4,5,6,7,8,9

echo "Done evaluating RAG LoRA for $MODEL"
