#!/bin/sh
#SBATCH --job-name=cluster_wildchat
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH -t 0-6:00
#SBATCH -c 8
#SBATCH --output=/n/home04/rrinberg/catered_out/cluster_wildchat.out
#SBATCH --error=/n/home04/rrinberg/catered_out/cluster_wildchat.err
#SBATCH --mail-user=royrinberg@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --account=kempner_sham_lab

ml python
module load cuda cudnn
export CONDA_ENVS_PATH=/n/holylabs/vadhan_lab/Lab/rrinberg/envs/conda/envs
export CONDA_PKGS_DIRS=/n/holylabs/vadhan_lab/Lab/rrinberg/envs/conda/pkgs

module load gcc/15.2.0-fasrc01
mamba activate MATS-inference2

export HF_HOME="/n/home04/rrinberg/data_dir/HF_cache"
echo "Hugging Face cache set to: $HF_HOME"

set -x

BASE="/n/home04/rrinberg/code/MATS/model-compression-bit-limiting"

python3 ${BASE}/scripts/cluster_chat_datasets.py \
    --dataset /n/netscratch/sham_lab/Lab/rrinberg/compression/wildchat-1m \
    --output-dir /n/netscratch/sham_lab/Lab/rrinberg/compression/wildchat-clustered \
    --n-clusters 50 \
    --embedding-model qwen \
    --batch-size 64

echo "Done clustering wildchat"
