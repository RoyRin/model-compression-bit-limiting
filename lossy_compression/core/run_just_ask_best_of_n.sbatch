#!/bin/sh
#SBATCH --job-name=just_ask_bon
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH -t 0-12:00:00
#SBATCH -c 8
#SBATCH --output=/n/home04/rrinberg/catered_out/just_ask_best_of_n/just_ask_bon_%j.out
#SBATCH --error=/n/home04/rrinberg/catered_out/just_ask_best_of_n/just_ask_bon_%j.err
#SBATCH --mail-user=royrinberg@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --account=kempner_sham_lab

# Usage:
#   sbatch lossy_compression/core/run_just_ask_best_of_n.sbatch

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
NUM_PROBLEMS=${NUM_PROBLEMS:-90}
N_VALUES=${N_VALUES:-"1 3 5 10"}

BASE="/n/home04/rrinberg/code/MATS/model-compression-bit-limiting"

echo "========================================"
echo "Just Ask + Best-of-N Experiment"
echo "Num problems: $NUM_PROBLEMS"
echo "N values: $N_VALUES"
echo "========================================"

cd ${BASE}/lossy_compression/core

python3 just_ask_best_of_n_experiment.py \
    --num-problems $NUM_PROBLEMS \
    --n-values $N_VALUES \
    --output-dir "${BASE}/results/just_ask_best_of_n" \
    --verbose

echo "Done with Just Ask + Best-of-N experiment"
