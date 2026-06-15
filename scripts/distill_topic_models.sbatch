#!/bin/bash
#SBATCH --job-name=distill_topics
#SBATCH --array=0-7%4
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --mem=320G
#SBATCH -t 0-10:00
#SBATCH -c 8
#SBATCH --output=/n/home04/rrinberg/logs/distill/distill_%A_%a.out
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
TEACHER_MODEL="meta-llama/Llama-3.1-8B"
DATA_BASE="/n/netscratch/sham_lab/Lab/rrinberg/compression/llama-3.1-8b"
SAVE_BASE="/n/netscratch/sham_lab/Lab/rrinberg/compression/distilled"
NUM_EPOCHS=3
BATCH_SIZE=4
TEACHER_BATCH_SIZE=8
TEMPERATURE=2.0
LEARNING_RATE=5e-5

# Create logs directory
mkdir -p /n/home04/rrinberg/logs/distill

# Map array index to student model and topic
# Indices 0-3: 3B model (music, math, coding, financial)
# Indices 4-7: 1B model (music, math, coding, financial)

TOPICS=(music math coding financial music math coding financial)
STUDENTS=(
    "meta-llama/Llama-3.2-3B"
    "meta-llama/Llama-3.2-3B"
    "meta-llama/Llama-3.2-3B"
    "meta-llama/Llama-3.2-3B"
    "meta-llama/Llama-3.2-1B"
    "meta-llama/Llama-3.2-1B"
    "meta-llama/Llama-3.2-1B"
    "meta-llama/Llama-3.2-1B"
)

# Get student and topic for this array task
STUDENT_MODEL=${STUDENTS[$SLURM_ARRAY_TASK_ID]}
TOPIC=${TOPICS[$SLURM_ARRAY_TASK_ID]}

# Determine student name for directory
if [[ $STUDENT_MODEL == *"3B"* ]]; then
    STUDENT_NAME="Llama-3.2-3B"
else
    STUDENT_NAME="Llama-3.2-1B"
fi

# Set paths
DATA_FILE="${DATA_BASE}/${TOPIC}/${TOPIC}_train.txt"
SAVE_DIR="${SAVE_BASE}/8b-to-${STUDENT_NAME}/${TOPIC}"

echo "=========================================="
echo "Distillation Task ${SLURM_ARRAY_TASK_ID}"
echo "=========================================="
echo "Teacher: ${TEACHER_MODEL}"
echo "Student: ${STUDENT_MODEL}"
echo "Topic: ${TOPIC}"
echo "Data: ${DATA_FILE}"
echo "Save: ${SAVE_DIR}"
echo "=========================================="
echo ""

# Check if distillation already completed
if [ -d "${SAVE_DIR}/final" ] && [ -f "${SAVE_DIR}/final/config.json" ]; then
    echo "✓ Distillation already completed for ${STUDENT_NAME} on ${TOPIC}"
    echo "  Final model exists at: ${SAVE_DIR}/final"
    echo "  Skipping..."
    exit 0
fi

# Check if data file exists
if [ ! -f "${DATA_FILE}" ]; then
    echo "ERROR: Data file not found: ${DATA_FILE}"
    exit 1
fi

# Count training samples
NUM_SAMPLES=$(wc -l < "${DATA_FILE}")
echo "Training samples: ${NUM_SAMPLES}"
echo ""

# Run distillation
cd /n/home04/rrinberg/code/MATS/model-compression-bit-limiting

python compression/distill_with_saved_logits.py \
    --teacher-model "${TEACHER_MODEL}" \
    --student-model "${STUDENT_MODEL}" \
    --data-file "${DATA_FILE}" \
    --save-dir "${SAVE_DIR}" \
    --num-epochs ${NUM_EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --teacher-batch-size ${TEACHER_BATCH_SIZE} \
    --temperature ${TEMPERATURE} \
    --learning-rate ${LEARNING_RATE}

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Successfully distilled ${STUDENT_NAME} on ${TOPIC}"
    echo "Saved to: ${SAVE_DIR}"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "✗ Failed to distill ${STUDENT_NAME} on ${TOPIC}"
    echo "Exit code: ${EXIT_CODE}"
    echo "=========================================="
    exit $EXIT_CODE
fi
