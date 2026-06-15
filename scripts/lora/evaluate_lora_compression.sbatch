#!/bin/sh
#SBATCH --job-name=eval_lora_pairwise
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH -t 0-40:00
#SBATCH -c 8
#SBATCH --output=/n/home04/rrinberg/catered_out/eval_lora_%j.out
#SBATCH --error=/n/home04/rrinberg/catered_out/eval_lora_%j.err
#SBATCH --mail-user=royrinberg@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --account=kempner_sham_lab

# Usage:
#   sbatch scripts/lora/evaluate_lora_compression.sbatch                    # lmsys (default)
#   sbatch --export=DATASET=wildchat scripts/lora/evaluate_lora_compression.sbatch
#   sbatch --export=DATASET=enwik8 scripts/lora/evaluate_lora_compression.sbatch
#   sbatch --export=DATASET=enwik9 scripts/lora/evaluate_lora_compression.sbatch
#
# This runs pairwise evaluation by default (5 clusters, each tested against ALL other LoRAs).
# To disable pairwise and use single wrong-LoRA comparison, add --no-pairwise to the python command.

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
    LORAS_ROOT="${DATA_ROOT}/lmsys-loras"
    NUM_CLUSTERS=5
elif [ "$DATASET" = "wildchat" ]; then
    CLUSTERS_ROOT="${DATA_ROOT}/wildchat-clustered/clusters"
    LORAS_ROOT="${DATA_ROOT}/wildchat-loras"
    NUM_CLUSTERS=5
elif [ "$DATASET" = "enwik8" ]; then
    CLUSTERS_ROOT="${DATA_ROOT}/enwik8-clustered/clusters"
    LORAS_ROOT="${DATA_ROOT}/enwik8-loras"
    NUM_CLUSTERS=10
elif [ "$DATASET" = "enwik9" ]; then
    CLUSTERS_ROOT="${DATA_ROOT}/enwik9-clustered/clusters"
    LORAS_ROOT="${DATA_ROOT}/enwik9-loras"
    NUM_CLUSTERS=10
elif [ "$DATASET" = "enwik9-50" ]; then
    CLUSTERS_ROOT="${DATA_ROOT}/enwik9-clustered-50/clusters"
    LORAS_ROOT="${DATA_ROOT}/enwik9-loras-50"
    NUM_CLUSTERS=10
else
    echo "Unknown dataset: $DATASET. Use 'lmsys', 'wildchat', 'enwik8', 'enwik9', or 'enwik9-50'"
    exit 1
fi

echo "========================================"
echo "Dataset: $DATASET"
echo "Clusters: $CLUSTERS_ROOT"
echo "LoRAs: $LORAS_ROOT"
echo "========================================"

python3 ${BASE}/scripts/lora/evaluate_lora_compression.py \
    --clusters-root "$CLUSTERS_ROOT" \
    --loras-root "$LORAS_ROOT" \
    --base-model "mistralai/Mistral-7B-Instruct-v0.2" \
    --num-clusters $NUM_CLUSTERS \
    --max-samples 50 \
    --min-tokens 200 \
    --split test \
    --compression \
    --output-dir "${BASE}/results/lora_evaluation_pairwise_${DATASET}"

echo "Done evaluating LoRA compression for $DATASET"
