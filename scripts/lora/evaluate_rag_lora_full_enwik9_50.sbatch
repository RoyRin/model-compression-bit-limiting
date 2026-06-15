#!/bin/sh
#SBATCH --job-name=rag_lora_full
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH -t 2-00:00
#SBATCH -c 8
#SBATCH --output=/n/home04/rrinberg/catered_out/rag_lora_full_%A_%a.out
#SBATCH --error=/n/home04/rrinberg/catered_out/rag_lora_full_%A_%a.err
#SBATCH --mail-user=royrinberg@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --account=kempner_sham_lab
#SBATCH --array=0-5

# Full enwik9-50 evaluation: all 50 clusters, all samples, split across 6 GPUs
# Usage: sbatch scripts/lora/evaluate_rag_lora_full_enwik9_50.sbatch

ml python
module load cuda cudnn
export CONDA_ENVS_PATH=/n/holylabs/vadhan_lab/Lab/rrinberg/envs/conda/envs
export CONDA_PKGS_DIRS=/n/holylabs/vadhan_lab/Lab/rrinberg/envs/conda/pkgs

module load gcc/15.2.0-fasrc01
mamba activate MATS-inference2

pip install -q sentence-transformers

export HF_HOME="/n/home04/rrinberg/data_dir/HF_cache"
echo "Hugging Face cache set to: $HF_HOME"

set -x

BASE="/n/home04/rrinberg/code/MATS/model-compression-bit-limiting"
DATA_ROOT="/n/netscratch/sham_lab/Lab/rrinberg/compression"

CLUSTERS_ROOT="${DATA_ROOT}/enwik9-clustered-50/clusters"
LORAS_ROOT="${DATA_ROOT}/enwik9-loras-50"
INDEX_PATH="${DATA_ROOT}/enwik9-clustered-50/lora_rag_index"

# Split 50 clusters across 6 jobs (array index 0-5)
# Job 0: 0-8   (9 clusters)
# Job 1: 9-16  (8 clusters)
# Job 2: 17-24 (8 clusters)
# Job 3: 25-32 (8 clusters)
# Job 4: 33-40 (8 clusters)
# Job 5: 41-49 (9 clusters)

case $SLURM_ARRAY_TASK_ID in
    0) CLUSTER_IDS="0,1,2,3,4,5,6,7,8" ;;
    1) CLUSTER_IDS="9,10,11,12,13,14,15,16" ;;
    2) CLUSTER_IDS="17,18,19,20,21,22,23,24" ;;
    3) CLUSTER_IDS="25,26,27,28,29,30,31,32" ;;
    4) CLUSTER_IDS="33,34,35,36,37,38,39,40" ;;
    5) CLUSTER_IDS="41,42,43,44,45,46,47,48,49" ;;
esac

OUTPUT_DIR="${BASE}/results/rag_lora_evaluation_enwik9-50-full/job_${SLURM_ARRAY_TASK_ID}"

echo "========================================"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Cluster IDs: $CLUSTER_IDS"
echo "Output Dir: $OUTPUT_DIR"
echo "Clusters Root: $CLUSTERS_ROOT"
echo "LoRAs Root: $LORAS_ROOT"
echo "Index Path: $INDEX_PATH"
echo "========================================"

python3 ${BASE}/scripts/lora/evaluate_rag_lora.py \
    --clusters-root "$CLUSTERS_ROOT" \
    --loras-root "$LORAS_ROOT" \
    --index-path "$INDEX_PATH" \
    --base-model "mistralai/Mistral-7B-Instruct-v0.2" \
    --num-clusters 50 \
    --max-samples 0 \
    --min-tokens 200 \
    --k 10 \
    --bit-precision 64 \
    --output-dir "$OUTPUT_DIR" \
    --with-correct-lora \
    --use-prefill \
    --cluster-ids "$CLUSTER_IDS"

echo "Done with array task $SLURM_ARRAY_TASK_ID (clusters: $CLUSTER_IDS)"
