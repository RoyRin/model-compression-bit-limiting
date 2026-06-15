#!/bin/sh
#SBATCH --job-name=enwik_cluster
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH -t 0-8:00
#SBATCH -c 8
#SBATCH --output=/n/home04/rrinberg/catered_out/enwik_cluster_%j.out
#SBATCH --error=/n/home04/rrinberg/catered_out/enwik_cluster_%j.err
#SBATCH --mail-user=royrinberg@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --account=kempner_sham_lab

# Cluster enwik8 or enwik9 data for the LoRA compression pipeline.
#
# Usage:
#   sbatch --export=DATASET=enwik8 scripts/lora/run_enwik_lora_pipeline.sbatch
#   sbatch --export=DATASET=enwik9 scripts/lora/run_enwik_lora_pipeline.sbatch
#   sbatch --export=DATASET=enwik9,N_CLUSTERS=50 scripts/lora/run_enwik_lora_pipeline.sbatch
#
# After clustering, run the remaining pipeline steps:
#   sbatch --array=0-9%6 --export=DATASET=enwik8 scripts/lora/train_lora_array.sbatch
#   sbatch --export=DATASET=enwik8 scripts/lora/build_lora_index.sbatch
#   sbatch --export=DATASET=enwik8 scripts/lora/evaluate_lora_compression.sbatch

ml python
module load cuda cudnn
export CONDA_ENVS_PATH=/n/holylabs/vadhan_lab/Lab/rrinberg/envs/conda/envs
export CONDA_PKGS_DIRS=/n/holylabs/vadhan_lab/Lab/rrinberg/envs/conda/pkgs

module load gcc/15.2.0-fasrc01
mamba activate MATS-inference2

export HF_HOME="/n/home04/rrinberg/data_dir/HF_cache"
echo "Hugging Face cache set to: $HF_HOME"

set -x

# Default to enwik8 if not specified
DATASET=${DATASET:-enwik8}
N_CLUSTERS=${N_CLUSTERS:-10}

BASE="/n/home04/rrinberg/code/MATS/model-compression-bit-limiting"
DATA_ROOT="/n/netscratch/sham_lab/Lab/rrinberg/compression"
INPUT_FILE="${BASE}/data/enwiki9/enwik9"

# Suffix for non-default cluster counts (e.g., enwik9-clustered-50)
if [ "$N_CLUSTERS" = "10" ]; then
    SUFFIX=""
else
    SUFFIX="-${N_CLUSTERS}"
fi

if [ "$DATASET" = "enwik8" ]; then
    OUTPUT_DIR="${DATA_ROOT}/enwik8-clustered${SUFFIX}"
    MAX_BYTES_ARG="--max-bytes 100000000"
elif [ "$DATASET" = "enwik9" ]; then
    OUTPUT_DIR="${DATA_ROOT}/enwik9-clustered${SUFFIX}"
    MAX_BYTES_ARG=""
else
    echo "Unknown dataset: $DATASET. Use 'enwik8' or 'enwik9'"
    exit 1
fi

# Dataset name for downstream sbatches (e.g., enwik9-50)
DS_NAME="${DATASET}${SUFFIX}"

echo "========================================"
echo "Dataset: $DS_NAME"
echo "N_CLUSTERS: $N_CLUSTERS"
echo "Input: $INPUT_FILE"
echo "Output: $OUTPUT_DIR"
echo "========================================"

python3 ${BASE}/scripts/lora/cluster_text_datasets.py \
    --input "$INPUT_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --n-clusters $N_CLUSTERS \
    --embedding-model qwen \
    --batch-size 64 \
    $MAX_BYTES_ARG

echo "Done clustering $DS_NAME"
echo ""
echo "Next steps:"
echo "  sbatch --array=0-9%6 --export=DATASET=$DS_NAME scripts/lora/train_lora_array.sbatch"
echo "  sbatch --export=DATASET=$DS_NAME scripts/lora/build_lora_index.sbatch"
echo "  sbatch --export=DATASET=$DS_NAME scripts/lora/evaluate_lora_compression.sbatch"
