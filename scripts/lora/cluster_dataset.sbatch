#!/bin/sh
#SBATCH --job-name=cluster_data
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH -t 0-6:00
#SBATCH -c 8
#SBATCH --output=/n/home04/rrinberg/catered_out/cluster_%x_%j.out
#SBATCH --error=/n/home04/rrinberg/catered_out/cluster_%x_%j.err
#SBATCH --mail-user=royrinberg@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --account=kempner_sham_lab

# Usage:
#   sbatch --export=DATASET=lmsys scripts/lora/cluster_dataset.sbatch
#   sbatch --export=DATASET=wildchat scripts/lora/cluster_dataset.sbatch

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
    INPUT_PATH="${DATA_ROOT}/lmsys-chat-1m"
    OUTPUT_PATH="${DATA_ROOT}/lmsys-clustered"
elif [ "$DATASET" = "wildchat" ]; then
    INPUT_PATH="${DATA_ROOT}/wildchat-1m"
    OUTPUT_PATH="${DATA_ROOT}/wildchat-clustered"
else
    echo "Unknown dataset: $DATASET. Use 'lmsys' or 'wildchat'"
    exit 1
fi

echo "========================================"
echo "Dataset: $DATASET"
echo "Input: $INPUT_PATH"
echo "Output: $OUTPUT_PATH"
echo "========================================"

python3 ${BASE}/scripts/cluster_chat_datasets.py \
    --dataset "$INPUT_PATH" \
    --output-dir "$OUTPUT_PATH" \
    --n-clusters 50 \
    --embedding-model qwen \
    --batch-size 64

echo "Done clustering $DATASET"
