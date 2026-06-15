#!/bin/sh
#SBATCH --job-name=lora_train
#SBATCH --array=0-9%6
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH -t 0-8:00
#SBATCH -c 8
#SBATCH --output=/n/home04/rrinberg/catered_out/lora_train_%a_%j.out
#SBATCH --error=/n/home04/rrinberg/catered_out/lora_train_%a_%j.err
#SBATCH --account=kempner_sham_lab

# Usage:
#   sbatch --export=MODEL=mistral scripts/lora/train_lora_multimodel.sbatch
#   sbatch --export=MODEL=llama-base scripts/lora/train_lora_multimodel.sbatch
#   sbatch --export=MODEL=llama-instruct scripts/lora/train_lora_multimodel.sbatch

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
# Default to 10 clusters, use N_CLUSTERS=50 for 50-cluster
N_CLUSTERS=${N_CLUSTERS:-10}

BASE="/n/home04/rrinberg/code/MATS/model-compression-bit-limiting"
DATA_ROOT="/n/netscratch/sham_lab/Lab/rrinberg/compression"

# Set cluster path based on N_CLUSTERS
if [ "$N_CLUSTERS" = "10" ]; then
    CLUSTERS_ROOT="${DATA_ROOT}/enwik9-clustered/clusters"
    SUFFIX=""
else
    CLUSTERS_ROOT="${DATA_ROOT}/enwik9-clustered-${N_CLUSTERS}/clusters"
    SUFFIX="-${N_CLUSTERS}"
fi

# Set model-specific paths and HF model ID
if [ "$MODEL" = "mistral" ]; then
    HF_MODEL="mistralai/Mistral-7B-Instruct-v0.2"
    OUTPUT_ROOT="${DATA_ROOT}/enwik9-loras${SUFFIX}-mistral"
elif [ "$MODEL" = "llama-base" ]; then
    HF_MODEL="meta-llama/Llama-3.1-8B"
    OUTPUT_ROOT="${DATA_ROOT}/enwik9-loras${SUFFIX}-llama-base"
elif [ "$MODEL" = "llama-instruct" ]; then
    HF_MODEL="meta-llama/Llama-3.1-8B-Instruct"
    OUTPUT_ROOT="${DATA_ROOT}/enwik9-loras${SUFFIX}-llama-instruct"
else
    echo "Unknown model: $MODEL. Use 'mistral', 'llama-base', or 'llama-instruct'"
    exit 1
fi

# Format cluster ID with leading zeros (000, 001, ..., 009)
CLUSTER_ID=$(printf "%03d" $SLURM_ARRAY_TASK_ID)

CLUSTER_DIR="${CLUSTERS_ROOT}/cluster_${CLUSTER_ID}"
OUTPUT_DIR="${OUTPUT_ROOT}/cluster_${CLUSTER_ID}"

echo "========================================"
echo "Model: $MODEL ($HF_MODEL)"
echo "N_CLUSTERS: $N_CLUSTERS"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Cluster: $CLUSTER_ID"
echo "Cluster dir: $CLUSTER_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "Host: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "========================================"

# Skip if already trained
if [ -f "${OUTPUT_DIR}/adapter_config.json" ]; then
    echo "Cluster ${CLUSTER_ID} already trained for $MODEL, skipping."
    exit 0
fi

# Check if cluster exists
if [ ! -d "$CLUSTER_DIR" ]; then
    echo "Cluster directory not found: $CLUSTER_DIR"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

python3 ${BASE}/scripts/lora/train_lora_cluster.py \
    --cluster-dir "$CLUSTER_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --base-model "$HF_MODEL" \
    --lora-rank 16 \
    --lora-alpha 32 \
    --num-epochs 1 \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --max-length 2048 \
    --logging-steps 100 \
    --save-steps 1000

echo "Done with cluster ${CLUSTER_ID} for model $MODEL"
