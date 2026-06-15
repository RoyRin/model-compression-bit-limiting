#!/bin/sh
#SBATCH --job-name=rag_eval_%a
#SBATCH --array=0-9
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH -t 0-4:00
#SBATCH -c 8
#SBATCH --output=/n/home04/rrinberg/catered_out/rag_eval_%a_%j.out
#SBATCH --error=/n/home04/rrinberg/catered_out/rag_eval_%a_%j.err
#SBATCH --account=kempner_sham_lab

# Parallelized RAG LoRA evaluation - one cluster per GPU
# Usage:
#   sbatch --export=MODEL=mistral scripts/lora/evaluate_rag_parallel.sbatch
#   sbatch --export=MODEL=llama-base scripts/lora/evaluate_rag_parallel.sbatch
#   sbatch --export=MODEL=llama-instruct scripts/lora/evaluate_rag_parallel.sbatch

ml python
module load cuda cudnn
export CONDA_ENVS_PATH=/n/holylabs/vadhan_lab/Lab/rrinberg/envs/conda/envs
export CONDA_PKGS_DIRS=/n/holylabs/vadhan_lab/Lab/rrinberg/envs/conda/pkgs

module load gcc/15.2.0-fasrc01
mamba activate MATS-inference2

pip install -q sentence-transformers

export HF_HOME="/n/home04/rrinberg/data_dir/HF_cache"

set -x

MODEL=${MODEL:-mistral}
MAX_SAMPLES=${MAX_SAMPLES:-50}
CLUSTER_ID=$SLURM_ARRAY_TASK_ID

BASE="/n/home04/rrinberg/code/MATS/model-compression-bit-limiting"
DATA_ROOT="/n/netscratch/sham_lab/Lab/rrinberg/compression"

CLUSTERS_ROOT="${DATA_ROOT}/enwik9-clustered/clusters"
INDEX_PATH="${DATA_ROOT}/enwik9-clustered/lora_rag_index"

# Set model-specific paths
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
    echo "Unknown model: $MODEL"
    exit 1
fi

OUTPUT_DIR="${BASE}/results/rag_lora_enwik9_${MODEL}_120bit_parallel"
mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "Model: $MODEL ($HF_MODEL)"
echo "Cluster: $CLUSTER_ID"
echo "Max samples: $MAX_SAMPLES"
echo "Output: $OUTPUT_DIR"
echo "========================================"

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
    --output-dir "$OUTPUT_DIR" \
    --with-correct-lora \
    --cluster-ids $CLUSTER_ID

echo "Done with cluster $CLUSTER_ID for $MODEL"
