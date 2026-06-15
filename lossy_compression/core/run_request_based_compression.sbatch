#!/bin/sh
#SBATCH --job-name=request_compress
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH -t 0-12:00:00
#SBATCH -c 8
#SBATCH --output=/n/home04/rrinberg/catered_out/request_based_compression/request_compress_%j.out
#SBATCH --error=/n/home04/rrinberg/catered_out/request_based_compression/request_compress_%j.err
#SBATCH --mail-user=royrinberg@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --account=kempner_sham_lab

# Usage:
#   sbatch lossy_compression/core/run_request_based_compression.sbatch

ml python
module load cuda cudnn
export CONDA_ENVS_PATH=/n/holylabs/vadhan_lab/Lab/rrinberg/envs/conda/envs
export CONDA_PKGS_DIRS=/n/holylabs/vadhan_lab/Lab/rrinberg/envs/conda/pkgs

module load gcc/15.2.0-fasrc01
mamba activate MATS-inference2

export HF_HOME="/n/home04/rrinberg/data_dir/HF_cache"
echo "Hugging Face cache set to: $HF_HOME"

set -x

# Default parameters (can be overridden via --export)
NUM_PROBLEMS=${NUM_PROBLEMS:-100}

BASE="/n/home04/rrinberg/code/MATS/model-compression-bit-limiting"

echo "========================================"
echo "Request-Based Compression Experiment"
echo "Num problems: $NUM_PROBLEMS"
echo "========================================"

cd ${BASE}/lossy_compression/core

python3 request_based_compression_experiment.py \
    --num-problems $NUM_PROBLEMS \
    --output-dir "${BASE}/results/request_based_compression" \
    --verbose

echo "Done with Request-Based Compression experiment"
