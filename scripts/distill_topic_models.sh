#!/bin/bash
# Distill Llama-3.1-8B to smaller models (3B, 1B) on topic-specific data

set -e  # Exit on error

# Configuration
TEACHER_MODEL="meta-llama/Llama-3.1-8B"
STUDENT_MODELS=("meta-llama/Llama-3.2-3B" "meta-llama/Llama-3.2-1B")
TOPICS=("music" "math" "coding" "financial")
DATA_DIR="/n/netscratch/sham_lab/Lab/rrinberg/compression/llama-3.1-8b"
SAVE_BASE_DIR="/n/netscratch/sham_lab/Lab/rrinberg/compression/distilled"
NUM_EPOCHS=3
BATCH_SIZE=4
TEMPERATURE=2.0
LEARNING_RATE=5e-5

# Log setup
LOG_DIR="${SAVE_BASE_DIR}/logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/distillation_${TIMESTAMP}.log"

echo "========================================" | tee -a "${LOG_FILE}"
echo "Topic-Specific Model Distillation" | tee -a "${LOG_FILE}"
echo "Started: $(date)" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"
echo "Teacher: ${TEACHER_MODEL}" | tee -a "${LOG_FILE}"
echo "Students: ${STUDENT_MODELS[@]}" | tee -a "${LOG_FILE}"
echo "Topics: ${TOPICS[@]}" | tee -a "${LOG_FILE}"
echo "Epochs: ${NUM_EPOCHS}, Batch size: ${BATCH_SIZE}" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"

# Function to run distillation for a single pair
run_distillation() {
    local teacher=$1
    local student=$2
    local topic=$3
    local data_file=$4
    local save_dir=$5

    # Extract short names for logging
    local student_name=$(basename "${student}")

    echo "" | tee -a "${LOG_FILE}"
    echo "----------------------------------------" | tee -a "${LOG_FILE}"
    echo "Distilling: 8B → ${student_name} (${topic})" | tee -a "${LOG_FILE}"
    echo "Data: ${data_file}" | tee -a "${LOG_FILE}"
    echo "Save: ${save_dir}" | tee -a "${LOG_FILE}"
    echo "----------------------------------------" | tee -a "${LOG_FILE}"

    # Check if distillation already completed
    if [ -d "${save_dir}/final" ] && [ -f "${save_dir}/final/config.json" ]; then
        echo "✓ Distillation already completed for ${student_name} on ${topic}" | tee -a "${LOG_FILE}"
        echo "  Final model exists at: ${save_dir}/final" | tee -a "${LOG_FILE}"
        echo "  Skipping..." | tee -a "${LOG_FILE}"
        return 0
    fi

    # Check if data file exists
    if [ ! -f "${data_file}" ]; then
        echo "ERROR: Data file not found: ${data_file}" | tee -a "${LOG_FILE}"
        return 1
    fi

    # Count lines in data file
    local num_lines=$(wc -l < "${data_file}")
    echo "Training samples: ${num_lines}" | tee -a "${LOG_FILE}"

    # Run distillation (two-stage with saved logits)
    python compression/distill_with_saved_logits.py \
        --teacher-model "${teacher}" \
        --student-model "${student}" \
        --data-file "${data_file}" \
        --num-epochs ${NUM_EPOCHS} \
        --batch-size ${BATCH_SIZE} \
        --teacher-batch-size 8 \
        --temperature ${TEMPERATURE} \
        --learning-rate ${LEARNING_RATE} \
        --save-dir "${save_dir}" 2>&1 | tee -a "${LOG_FILE}"

    if [ $? -eq 0 ]; then
        echo "✓ Successfully distilled ${student_name} on ${topic}" | tee -a "${LOG_FILE}"
    else
        echo "✗ Failed to distill ${student_name} on ${topic}" | tee -a "${LOG_FILE}"
        return 1
    fi

    # Clear GPU cache
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    sleep 5
}

# Run distillations for all combinations
total_distillations=$((${#STUDENT_MODELS[@]} * ${#TOPICS[@]}))
current=0

for student in "${STUDENT_MODELS[@]}"; do
    student_name=$(basename "${student}")

    for topic in "${TOPICS[@]}"; do
        current=$((current + 1))

        echo "" | tee -a "${LOG_FILE}"
        echo "========================================"  | tee -a "${LOG_FILE}"
        echo "Progress: ${current}/${total_distillations}" | tee -a "${LOG_FILE}"
        echo "========================================"  | tee -a "${LOG_FILE}"

        # Paths
        data_file="${DATA_DIR}/${topic}/${topic}_train.txt"
        save_dir="${SAVE_BASE_DIR}/8b-to-${student_name}/${topic}"

        # Run distillation
        run_distillation \
            "${TEACHER_MODEL}" \
            "${student}" \
            "${topic}" \
            "${data_file}" \
            "${save_dir}"
    done
done

echo "" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"
echo "All distillations complete!" | tee -a "${LOG_FILE}"
echo "Completed: $(date)" | tee -a "${LOG_FILE}"
echo "Log saved to: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"

# Print summary
echo "" | tee -a "${LOG_FILE}"
echo "Distilled models saved to:" | tee -a "${LOG_FILE}"
for student in "${STUDENT_MODELS[@]}"; do
    student_name=$(basename "${student}")
    for topic in "${TOPICS[@]}"; do
        model_dir="${SAVE_BASE_DIR}/8b-to-${student_name}/${topic}"
        if [ -d "${model_dir}" ]; then
            echo "  ✓ ${model_dir}" | tee -a "${LOG_FILE}"
        else
            echo "  ✗ ${model_dir} (not found)" | tee -a "${LOG_FILE}"
        fi
    done
done
