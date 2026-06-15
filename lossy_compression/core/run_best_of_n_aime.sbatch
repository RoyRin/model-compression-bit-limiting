#!/bin/sh
#SBATCH --job-name=best_of_n_aime
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH -t 0-42:00:00
#SBATCH -c 8
#SBATCH --output=/n/home04/rrinberg/catered_out/best_of_n_aime/best_of_n_aime_%j.out
#SBATCH --error=/n/home04/rrinberg/catered_out/best_of_n_aime/best_of_n_aime_%j.err
#SBATCH --mail-user=royrinberg@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --account=kempner_sham_lab

# Usage:
#   sbatch lossy_compression/core/run_best_of_n_aime.sbatch
#   sbatch --export=NUM_PROBLEMS=20 lossy_compression/core/run_best_of_n_aime.sbatch

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
N_VALUES=${N_VALUES:-"1 3 5 10"}

BASE="/n/home04/rrinberg/code/MATS/model-compression-bit-limiting"

echo "========================================"
echo "Best-of-N AIME Experiment"
echo "Num problems: $NUM_PROBLEMS"
echo "N values: $N_VALUES"
echo "========================================"

cd ${BASE}/lossy_compression/core

python3 best_of_n_approaches_experiment.py \
    --num-problems $NUM_PROBLEMS \
    --n-values $N_VALUES \
    --approaches temperature single_prompt \
    --output-dir "${BASE}/results/best_of_n_aime" \
    --verbose

echo "Done with Best-of-N AIME experiment"
