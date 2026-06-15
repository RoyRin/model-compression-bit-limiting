#!/bin/sh
#SBATCH --job-name=lora_train
#SBATCH --array=0-49%6
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH -t 0-8:00
#SBATCH -c 8
#SBATCH --output=/n/home04/rrinberg/catered_out/lora_train_%a.out
#SBATCH --error=/n/home04/rrinberg/catered_out/lora_train_%a.err
#SBATCH --mail-user=royrinberg@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --account=kempner_sham_lab

# Usage:
#   sbatch --export=DATASET=lmsys scripts/lora/train_lora_array.sbatch
#   sbatch --export=DATASET=wildchat scripts/lora/train_lora_array.sbatch
#   sbatch --array=0-9%6 --export=DATASET=enwik8 scripts/lora/train_lora_array.sbatch
#   sbatch --array=0-9%6 --export=DATASET=enwik9 scripts/lora/train_lora_array.sbatch

ml python
module load cuda cudnn
export CONDA_ENVS_PATH=/n/holylabs/vadhan_lab/Lab/rrinberg/envs/conda/envs
export CONDA_PKGS_DIRS=/n/holylabs/vadhan_lab/Lab/rrinberg/envs/conda/pkgs

module load gcc/15.2.0-fasrc01
mamba activate MATS-inference2

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
    OUTPUT_ROOT="${DATA_ROOT}/lmsys-loras"
elif [ "$DATASET" = "wildchat" ]; then
    CLUSTERS_ROOT="${DATA_ROOT}/wildchat-clustered/clusters"
    OUTPUT_ROOT="${DATA_ROOT}/wildchat-loras"
elif [ "$DATASET" = "enwik8" ]; then
    CLUSTERS_ROOT="${DATA_ROOT}/enwik8-clustered/clusters"
    OUTPUT_ROOT="${DATA_ROOT}/enwik8-loras"
elif [ "$DATASET" = "enwik9" ]; then
    CLUSTERS_ROOT="${DATA_ROOT}/enwik9-clustered/clusters"
    OUTPUT_ROOT="${DATA_ROOT}/enwik9-loras"
elif [ "$DATASET" = "enwik9-50" ]; then
    CLUSTERS_ROOT="${DATA_ROOT}/enwik9-clustered-50/clusters"
    OUTPUT_ROOT="${DATA_ROOT}/enwik9-loras-50"
else
    echo "Unknown dataset: $DATASET. Use 'lmsys', 'wildchat', 'enwik8', 'enwik9', or 'enwik9-50'"
    exit 1
fi

# Format cluster ID with leading zeros (000, 001, ..., 049)
CLUSTER_ID=$(printf "%03d" $SLURM_ARRAY_TASK_ID)

CLUSTER_DIR="${CLUSTERS_ROOT}/cluster_${CLUSTER_ID}"
OUTPUT_DIR="${OUTPUT_ROOT}/cluster_${CLUSTER_ID}"

echo "========================================"
echo "Dataset: $DATASET"
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
    echo "Cluster ${CLUSTER_ID} already trained, skipping."
    exit 0
fi

# Check if cluster exists
if [ ! -d "$CLUSTER_DIR" ]; then
    echo "Cluster directory not found: $CLUSTER_DIR"
    exit 1
fi

python3 ${BASE}/scripts/lora/train_lora_cluster.py \
    --cluster-dir "$CLUSTER_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --base-model "mistralai/Mistral-7B-Instruct-v0.2" \
    --lora-rank 16 \
    --lora-alpha 32 \
    --num-epochs 1 \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --max-length 2048 \
    --logging-steps 100 \
    --save-steps 1000

echo "Done with cluster ${CLUSTER_ID}"
