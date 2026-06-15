#!/bin/sh
#SBATCH --job-name=cascade_lora
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH -t 0-06:00:00
#SBATCH -c 4
#SBATCH --output=/n/home04/rrinberg/catered_out/cascade_lora_%j.out
#SBATCH --error=/n/home04/rrinberg/catered_out/cascade_lora_%j.err
#SBATCH --mail-user=royrinberg@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --account=kempner_sham_lab

# Usage:
#   sbatch scripts/lora/evaluate_cascade_lora_selection.sbatch
#   sbatch --export=TOP_N=5,NO_ORACLE=1 scripts/lora/evaluate_cascade_lora_selection.sbatch

ml python
module load cuda cudnn
export CONDA_ENVS_PATH=/n/holylabs/vadhan_lab/Lab/rrinberg/envs/conda/envs
export CONDA_PKGS_DIRS=/n/holylabs/vadhan_lab/Lab/rrinberg/envs/conda/pkgs

mamba activate MATS-inference2

export HF_HOME="/n/home04/rrinberg/data_dir/HF_cache"
echo "Hugging Face cache set to: $HF_HOME"

set -x

# Default parameters
DATASET=${DATASET:-lmsys}
NUM_CLUSTERS=${NUM_CLUSTERS:-10}
MAX_SAMPLES=${MAX_SAMPLES:-30}
TOP_N=${TOP_N:-10}
K_RAG=${K_RAG:-10}

BASE="/n/home04/rrinberg/code/MATS/model-compression-bit-limiting"

echo "========================================"
echo "Cascade LoRA Selection: RAG → Perplexity"
echo "Dataset: $DATASET"
echo "Num clusters: $NUM_CLUSTERS"
echo "Top-N from RAG: $TOP_N"
echo "Max samples: $MAX_SAMPLES"
echo "========================================"

cd ${BASE}

python3 scripts/lora/evaluate_cascade_lora_selection.py \
    --dataset $DATASET \
    --num-clusters $NUM_CLUSTERS \
    --max-samples $MAX_SAMPLES \
    --top-n $TOP_N \
    --k-rag $K_RAG \
    --output-dir "${BASE}/results/cascade_lora_selection_${DATASET}"

echo "Done with cascade LoRA selection evaluation"
