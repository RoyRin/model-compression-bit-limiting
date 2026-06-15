#!/bin/bash
#SBATCH --job-name=lora_cartesian
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --mem=320G
#SBATCH -t 0-12:00
#SBATCH -c 8
#SBATCH --output=/n/home04/rrinberg/logs/lora_cartesian_%j.out
#SBATCH --mail-user=royrinberg@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --account=kempner_sham_lab
#SBATCH --exclude=holygpu8a17202

# Load modules
ml python
module load cuda cudnn
export CONDA_ENVS_PATH=/n/holylabs/vadhan_lab/Lab/rrinberg/envs/conda/envs
export CONDA_PKGS_DIRS=/n/holylabs/vadhan_lab/Lab/rrinberg/envs/conda/pkgs
module load gcc/15.2.0-fasrc01

# Activate environment
mamba activate MATS-compression

# Set HuggingFace cache
export HF_HOME="/n/home04/rrinberg/data_dir/HF_cache"
echo "Hugging Face cache set to: $HF_HOME"

# Configuration
BASE_MODEL="mistralai/Mistral-7B-Instruct-v0.2"
OUTPUT_DIR="/n/netscratch/sham_lab/Lab/rrinberg/compression/lora_cartesian"
MAX_TOKENS=200
LORA_RANK=16
LORA_BITS=4

echo "=========================================="
echo "LoRA Cartesian Product Text Generation"
echo "=========================================="
echo "Base model: ${BASE_MODEL}"
echo "Max tokens: ${MAX_TOKENS}"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="
echo ""

# Change to project directory
cd /n/home04/rrinberg/code/MATS/model-compression-bit-limiting

# Run generation
python scripts/generate_lora_cartesian.py \
    --base-model "${BASE_MODEL}" \
    --output-dir "${OUTPUT_DIR}" \
    --max-tokens ${MAX_TOKENS} \
    --lora-rank ${LORA_RANK} \
    --lora-bits ${LORA_BITS} \
    --split valid

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Successfully generated cartesian product"
    echo "Results saved to: ${OUTPUT_DIR}"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "✗ Generation failed"
    echo "Exit code: ${EXIT_CODE}"
    echo "=========================================="
    exit $EXIT_CODE
fi
