#!/bin/sh
#SBATCH --job-name=lora_qwen3
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH -t 0-8:00
#SBATCH -c 8
#SBATCH --output=/n/home04/rrinberg/catered_out/compression_loras/lora_qwen3_%A_%a.out
#SBATCH --error=/n/home04/rrinberg/catered_out/compression_loras/lora_qwen3_%A_%a.err
#SBATCH --mail-user=royrinberg@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --account=kempner_sham_lab

# Usage:
#   sbatch --array=0-49%6 --export=DATASET=lmsys scripts/lora/train_lora_qwen3.sbatch
#   sbatch --array=0-49%6 --export=DATASET=wildchat scripts/lora/train_lora_qwen3.sbatch
#   sbatch --array=0-9%6 --export=DATASET=enwik9 scripts/lora/train_lora_qwen3.sbatch
#
# After training, upload:
#   python scripts/upload_loras_to_hf.py --only loras/lmsys-qwen3-4b loras/wildchat-qwen3-4b loras/enwik9-qwen3-4b

ml python
module load cuda cudnn
export CONDA_ENVS_PATH=/n/holylabs/vadhan_lab/Lab/rrinberg/envs/conda/envs
export CONDA_PKGS_DIRS=/n/holylabs/vadhan_lab/Lab/rrinberg/envs/conda/pkgs

module load gcc/15.2.0-fasrc01
mamba activate MATS-inference2

export HF_HOME="/n/home04/rrinberg/data_dir/HF_cache"
echo "Hugging Face cache set to: $HF_HOME"

set -x

# Avoid Keras 3 / TF import crash in transformers
export TF_CPP_MIN_LOG_LEVEL=3
export TRANSFORMERS_NO_TF=1

DATASET=${DATASET:-lmsys}
HF_MODEL="Qwen/Qwen3-4B"

BASE="/n/home04/rrinberg/code/MATS/model-compression-bit-limiting"
DATA_ROOT="/n/netscratch/sham_lab/Lab/rrinberg/compression"

if [ "$DATASET" = "lmsys" ]; then
    CLUSTERS_ROOT="${DATA_ROOT}/lmsys-clustered/clusters"
    OUTPUT_ROOT="${DATA_ROOT}/lmsys-loras-qwen3-4b"
elif [ "$DATASET" = "wildchat" ]; then
    CLUSTERS_ROOT="${DATA_ROOT}/wildchat-clustered/clusters"
    OUTPUT_ROOT="${DATA_ROOT}/wildchat-loras-qwen3-4b"
elif [ "$DATASET" = "enwik9" ]; then
    CLUSTERS_ROOT="${DATA_ROOT}/enwik9-clustered/clusters"
    OUTPUT_ROOT="${DATA_ROOT}/enwik9-loras-qwen3-4b"
elif [ "$DATASET" = "enwik9-50" ]; then
    CLUSTERS_ROOT="${DATA_ROOT}/enwik9-clustered-50/clusters"
    OUTPUT_ROOT="${DATA_ROOT}/enwik9-loras-50-qwen3-4b"
else
    echo "Unknown dataset: $DATASET. Use 'lmsys', 'wildchat', 'enwik9', or 'enwik9-50'"
    exit 1
fi

CLUSTER_ID=$(printf "%03d" $SLURM_ARRAY_TASK_ID)
CLUSTER_DIR="${CLUSTERS_ROOT}/cluster_${CLUSTER_ID}"
OUTPUT_DIR="${OUTPUT_ROOT}/cluster_${CLUSTER_ID}"

echo "========================================"
echo "Dataset: $DATASET"
echo "Model: $HF_MODEL"
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

if [ ! -d "$CLUSTER_DIR" ]; then
    echo "Cluster directory not found: $CLUSTER_DIR"
    exit 1
fi

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

echo "Done with cluster ${CLUSTER_ID}"
