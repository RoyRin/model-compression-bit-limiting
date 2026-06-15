#!/bin/bash
# Generate topic-specific data for multiple Llama models
# Usage: ./scripts/generate_all_models.sh

set -e  # Exit on error

# Base output directory
BASE_OUTPUT_DIR="/n/netscratch/sham_lab/Lab/rrinberg/compression"

# Configuration
SEQUENCES_PER_TOPIC=2000  # 2000 sequences per topic
MAX_LENGTH=512
TEMPERATURE=0.9
TOP_P=0.95

# Log file
LOG_DIR="${BASE_OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/generation_${TIMESTAMP}.log"

echo "========================================" | tee -a "${LOG_FILE}"
echo "Topic Data Generation for Multiple Models" | tee -a "${LOG_FILE}"
echo "Started: $(date)" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"

# Function to run generation for a single model
run_generation() {
    local model_name=$1
    local model_size=$2
    local quantization=$3
    local output_subdir="${BASE_OUTPUT_DIR}/${model_size}"

    echo "" | tee -a "${LOG_FILE}"
    echo "----------------------------------------" | tee -a "${LOG_FILE}"
    echo "Processing: ${model_name} (${model_size})" | tee -a "${LOG_FILE}"
    echo "Output: ${output_subdir}" | tee -a "${LOG_FILE}"
    if [ -n "${quantization}" ]; then
        echo "Quantization: ${quantization}" | tee -a "${LOG_FILE}"
    fi
    echo "----------------------------------------" | tee -a "${LOG_FILE}"

    # Build command
    cmd="python scripts/generate_topic_data.py ${model_name} \
        --output-dir ${output_subdir} \
        --sequences-per-topic ${SEQUENCES_PER_TOPIC} \
        --max-length ${MAX_LENGTH} \
        --temperature ${TEMPERATURE} \
        --top-p ${TOP_P}"

    # Add quantization if specified
    if [ -n "${quantization}" ]; then
        cmd="${cmd} --quantization ${quantization}"
    fi

    echo "Command: ${cmd}" | tee -a "${LOG_FILE}"
    echo "" | tee -a "${LOG_FILE}"

    # Run with output to both console and log
    if eval "${cmd}" 2>&1 | tee -a "${LOG_FILE}"; then
        echo "✓ Successfully completed ${model_size}" | tee -a "${LOG_FILE}"
    else
        echo "✗ Failed to process ${model_size}" | tee -a "${LOG_FILE}"
        return 1
    fi

    # Clear GPU cache between models
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    sleep 5
}

# Generate data for each model (ordered largest to smallest)
echo "" | tee -a "${LOG_FILE}"
echo "Models to process (largest first):" | tee -a "${LOG_FILE}"
echo "  1. Llama-3.1-70B (FP8 quantized)" | tee -a "${LOG_FILE}"
echo "  2. Llama-3.1-8B" | tee -a "${LOG_FILE}"
echo "  3. Llama-3.2-3B" | tee -a "${LOG_FILE}"
echo "  4. Llama-3.2-1B" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# 1. Llama-3.1-70B (use FP8 quantization for H100 80GB)
# FP8 brings 70B model to ~35-40GB, perfect for single H100 80GB
run_generation "meta-llama/Llama-3.1-70B" "llama-3.1-70b" "fp8"

# 2. Llama-3.1-8B (no quantization - vLLM is efficient enough)
run_generation "meta-llama/Llama-3.1-8B" "llama-3.1-8b" ""

# 3. Llama-3.2-3B (no quantization needed)
run_generation "meta-llama/Llama-3.2-3B" "llama-3.2-3b" ""

# 4. Llama-3.2-1B (no quantization needed)
run_generation "meta-llama/Llama-3.2-1B" "llama-3.2-1b" ""

echo "" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"
echo "All models processed successfully!" | tee -a "${LOG_FILE}"
echo "Completed: $(date)" | tee -a "${LOG_FILE}"
echo "Log saved to: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"

# Print summary of generated data
echo "" | tee -a "${LOG_FILE}"
echo "Generated datasets:" | tee -a "${LOG_FILE}"
for size in llama-3.2-1b llama-3.2-3b llama-3.1-8b llama-3.1-70b; do
    output_dir="${BASE_OUTPUT_DIR}/${size}"
    if [ -d "${output_dir}" ]; then
        train_file="${output_dir}/all_topics_train.txt"
        test_file="${output_dir}/all_topics_test.yaml"
        if [ -f "${train_file}" ]; then
            train_lines=$(wc -l < "${train_file}")
            echo "  ${size}: ${train_lines} train sequences" | tee -a "${LOG_FILE}"
        fi
    fi
done
