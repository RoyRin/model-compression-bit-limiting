#!/bin/sh
#SBATCH --job-name=build_lora_idx
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH -t 0-32:00
#SBATCH -c 8
#SBATCH --output=/n/home04/rrinberg/catered_out/build_lora_idx_%j.out
#SBATCH --error=/n/home04/rrinberg/catered_out/build_lora_idx_%j.err
#SBATCH --mail-user=royrinberg@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --account=kempner_sham_lab

# Usage:
#   sbatch scripts/lora/build_lora_index.sbatch
#   sbatch --export=DATASET=wildchat scripts/lora/build_lora_index.sbatch
#   sbatch --export=DATASET=enwik8 scripts/lora/build_lora_index.sbatch
#   sbatch --export=DATASET=enwik9 scripts/lora/build_lora_index.sbatch

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

# Default to lmsys if not specified
DATASET=${DATASET:-lmsys}

BASE="/n/home04/rrinberg/code/MATS/model-compression-bit-limiting"
DATA_ROOT="/n/netscratch/sham_lab/Lab/rrinberg/compression"

# Set paths based on dataset
if [ "$DATASET" = "lmsys" ]; then
    CLUSTERS_ROOT="${DATA_ROOT}/lmsys-clustered/clusters"
    LORAS_ROOT="${DATA_ROOT}/lmsys-loras"
    INDEX_PATH="${DATA_ROOT}/lmsys-clustered/lora_rag_index"
    CLUSTER_IDS_ARG=""
elif [ "$DATASET" = "wildchat" ]; then
    CLUSTERS_ROOT="${DATA_ROOT}/wildchat-clustered/clusters"
    LORAS_ROOT="${DATA_ROOT}/wildchat-loras"
    INDEX_PATH="${DATA_ROOT}/wildchat-clustered/lora_rag_index"
    CLUSTER_IDS_ARG=""
elif [ "$DATASET" = "enwik8" ]; then
    CLUSTERS_ROOT="${DATA_ROOT}/enwik8-clustered/clusters"
    LORAS_ROOT="${DATA_ROOT}/enwik8-loras"
    INDEX_PATH="${DATA_ROOT}/enwik8-clustered/lora_rag_index"
    CLUSTER_IDS_ARG="--cluster-ids 0,1,2,3,4,5,6,7,8,9"
elif [ "$DATASET" = "enwik9" ]; then
    CLUSTERS_ROOT="${DATA_ROOT}/enwik9-clustered/clusters"
    LORAS_ROOT="${DATA_ROOT}/enwik9-loras"
    INDEX_PATH="${DATA_ROOT}/enwik9-clustered/lora_rag_index"
    CLUSTER_IDS_ARG="--cluster-ids 0,1,2,3,4,5,6,7,8,9"
elif [ "$DATASET" = "enwik9-50" ]; then
    CLUSTERS_ROOT="${DATA_ROOT}/enwik9-clustered-50/clusters"
    LORAS_ROOT="${DATA_ROOT}/enwik9-loras-50"
    INDEX_PATH="${DATA_ROOT}/enwik9-clustered-50/lora_rag_index"
    CLUSTER_IDS_ARG="--cluster-ids 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49"
else
    echo "Unknown dataset: $DATASET. Use 'lmsys', 'wildchat', 'enwik8', 'enwik9', or 'enwik9-50'"
    exit 1
fi

echo "========================================"
echo "Dataset: $DATASET"
echo "Clusters: $CLUSTERS_ROOT"
echo "LoRAs: $LORAS_ROOT"
echo "Index output: $INDEX_PATH"
echo "========================================"

python3 ${BASE}/scripts/lora/lora_router.py \
    --build-index \
    --clusters-root "$CLUSTERS_ROOT" \
    --loras-root "$LORAS_ROOT" \
    --samples-per-cluster 1000 \
    --min-tokens 150 \
    --index-path "$INDEX_PATH" \
    $CLUSTER_IDS_ARG

echo "Done building LoRA RAG index for $DATASET"
