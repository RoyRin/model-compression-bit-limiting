#!/bin/sh
#SBATCH --job-name=rag_routing_strategies
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 0-02:00:00
#SBATCH -c 4
#SBATCH --output=/n/home04/rrinberg/catered_out/rag_routing_strategies_%j.out
#SBATCH --error=/n/home04/rrinberg/catered_out/rag_routing_strategies_%j.err
#SBATCH --mail-user=royrinberg@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --account=kempner_sham_lab

# Usage:
#   sbatch scripts/lora/evaluate_rag_routing_strategies.sbatch
#   sbatch --export=DATASET=wildchat scripts/lora/evaluate_rag_routing_strategies.sbatch

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
MAX_SAMPLES=${MAX_SAMPLES:-50}
K=${K:-10}

BASE="/n/home04/rrinberg/code/MATS/model-compression-bit-limiting"

echo "========================================"
echo "RAG Routing Strategy Comparison"
echo "Dataset: $DATASET"
echo "Num clusters: $NUM_CLUSTERS"
echo "Max samples: $MAX_SAMPLES"
echo "k: $K"
echo "========================================"

cd ${BASE}

python3 scripts/lora/evaluate_rag_routing_strategies.py \
    --dataset $DATASET \
    --num-clusters $NUM_CLUSTERS \
    --max-samples $MAX_SAMPLES \
    --k $K \
    --output-dir "${BASE}/results/rag_routing_strategies_${DATASET}"

echo "Done with RAG routing strategy comparison"
