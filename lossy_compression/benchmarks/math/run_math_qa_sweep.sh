#!/bin/bash
#
# Run full cartesian sweep of MATH Q&A compression experiments
# Tests all combinations of SLM, LLM, and Question models
# Total: 3x4x3 = 36 combinations per subject (LLM includes opus-oracle)
#
# Usage:
#   # Run all subjects (algebra, geometry, number_theory)
#   ./run_math_qa_sweep.sh
#
#   # Run specific subject
#   ./run_math_qa_sweep.sh --subject algebra
#
#   # Restart to retry failed problems (uses same directory by default)
#   ./run_math_qa_sweep.sh
#
#   # Use timestamped directory (for fresh runs)
#   ./run_math_qa_sweep.sh --datestring
#
#   # Copy results from old directory and retry failed problems
#   ./run_math_qa_sweep.sh --resume-from results/math_qa_sweep_20260115_120000
#
#   # Limit number of problems per combination (for testing)
#   ./run_math_qa_sweep.sh --num-problems 10

set -e

# Change to project root
cd "$(dirname "$0")/../../.."
source .venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Model arrays
MODELS=("haiku" "sonnet" "opus")
LLM_OPTIONS=("haiku" "sonnet" "opus" "opus-oracle")  # 4 LLM options including oracle
SUBJECTS=("algebra" "geometry" "number_theory")

# Configuration defaults
DIFFICULTY="not_easy"
MAX_QUESTIONS=30
BATCH_SIZE=10
NUM_PROBLEMS=""
RESULTS_DIR=""
RESUME_FROM=""
SUBJECT_FILTER=""
PARALLEL=true
MAX_WORKERS=8
USE_DATESTRING=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --subject)
            SUBJECT_FILTER="$2"
            shift 2
            ;;
        --num-problems)
            NUM_PROBLEMS="$2"
            shift 2
            ;;
        --max-questions|-q)
            MAX_QUESTIONS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --resume-from)
            RESUME_FROM="$2"
            shift 2
            ;;
        --max-workers)
            MAX_WORKERS="$2"
            shift 2
            ;;
        --no-parallel)
            PARALLEL=false
            shift
            ;;
        --datestring)
            USE_DATESTRING=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --subject SUBJECT      Run only this subject (algebra, geometry, number_theory)"
            echo "  --num-problems N       Limit to N problems per combination (default: all)"
            echo "  --max-questions N      Max Q&A iterations (default: 30)"
            echo "  --batch-size N         Questions per batch (default: 10)"
            echo "  --results-dir DIR      Use existing results directory (skip completed)"
            echo "  --resume-from DIR      Copy results from old directory and retry failed problems"
            echo "  --max-workers N        Parallel workers (default: 8)"
            echo "  --no-parallel          Disable parallel execution"
            echo "  --datestring           Add timestamp to results directory name"
            echo ""
            echo "This script runs all 36 combinations of:"
            echo "  SLM: haiku, sonnet, opus (3)"
            echo "  LLM: haiku, sonnet, opus, opus-oracle (4)"
            echo "  Question model: haiku, sonnet, opus (3)"
            echo ""
            echo "On MATH problems with difficulty: not_easy (medium + hard + very_hard)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Filter subjects if specified
if [ -n "$SUBJECT_FILTER" ]; then
    SUBJECTS=("$SUBJECT_FILTER")
fi

# Create or use results directory
if [ -z "$RESULTS_DIR" ]; then
    if [ "$USE_DATESTRING" = true ]; then
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        RESULTS_DIR="results/math_qa_sweep_${TIMESTAMP}"
    else
        RESULTS_DIR="results/math_qa_sweep"
    fi
fi

if [ ! -d "$RESULTS_DIR" ]; then
    mkdir -p "$RESULTS_DIR"
    echo "Created results directory: $RESULTS_DIR"
else
    echo "Using existing results directory: $RESULTS_DIR"
fi

# If --resume-from is specified, copy existing results to new directory
if [ -n "$RESUME_FROM" ]; then
    if [ ! -d "$RESUME_FROM" ]; then
        echo "ERROR: Resume source directory does not exist: $RESUME_FROM"
        exit 1
    fi
    echo ""
    echo "📂 Copying results from: $RESUME_FROM"
    echo "   to: $RESULTS_DIR"
    rsync -av --ignore-existing "$RESUME_FROM"/*.json "$RESULTS_DIR/" 2>/dev/null || true
    JSON_COUNT=$(ls -1 "$RESULTS_DIR"/*.json 2>/dev/null | wc -l)
    echo "   Copied $JSON_COUNT JSON files"
    echo ""
fi

# Save configuration
CONFIG_FILE="$RESULTS_DIR/sweep_config.json"
cat > "$CONFIG_FILE" << EOF
{
    "timestamp": "$(date -Iseconds)",
    "difficulty": "$DIFFICULTY",
    "max_questions": $MAX_QUESTIONS,
    "batch_size": $BATCH_SIZE,
    "num_problems": ${NUM_PROBLEMS:-null},
    "subjects": $(printf '%s\n' "${SUBJECTS[@]}" | jq -R . | jq -s .),
    "slm_models": ["haiku", "sonnet", "opus"],
    "llm_models": ["haiku", "sonnet", "opus", "opus-oracle"],
    "question_models": ["haiku", "sonnet", "opus"],
    "total_combinations": $((${#SUBJECTS[@]} * 36))
}
EOF

echo "=========================================="
echo "MATH Q&A Compression Sweep"
echo "=========================================="
echo "Subjects: ${SUBJECTS[*]}"
echo "Difficulty: $DIFFICULTY"
echo "Max questions: $MAX_QUESTIONS"
echo "Batch size: $BATCH_SIZE"
echo "Num problems: ${NUM_PROBLEMS:-all}"
echo "Parallel: $PARALLEL (workers: $MAX_WORKERS)"
echo "Results dir: $RESULTS_DIR"
echo "Total combinations: $((${#SUBJECTS[@]} * 36)) (3 SLM x 4 LLM x 3 Q)"
echo "=========================================="
echo ""

# Counters
TOTAL=0
COMPLETED=0
SKIPPED=0
FAILED=0

# Calculate total (3 SLM x 4 LLM x 3 Q = 36 per subject)
for subject in "${SUBJECTS[@]}"; do
    TOTAL=$((TOTAL + 36))
done

COUNTER=0

# Loop through all combinations
for subject in "${SUBJECTS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Subject: $subject"
    echo "=========================================="

    for slm in "${MODELS[@]}"; do
        for llm_option in "${LLM_OPTIONS[@]}"; do
            for q_model in "${MODELS[@]}"; do
                COUNTER=$((COUNTER + 1))

                # Handle oracle mode
                if [ "$llm_option" = "opus-oracle" ]; then
                    llm="opus"
                    ORACLE_FLAG="--oracle"
                    LLM_LABEL="opus-oracle"
                else
                    llm="$llm_option"
                    ORACLE_FLAG=""
                    LLM_LABEL="$llm_option"
                fi

                # Build result filename
                RESULT_NAME="math_qa_${subject}_SLM-${slm}_LLM-${LLM_LABEL}_Q-${q_model}"
                RESULT_FILE="$RESULTS_DIR/${RESULT_NAME}.json"

                # Check if file exists - if so, resume to retry any failed problems
                RESUME_FLAG=""
                if [ -f "$RESULT_FILE" ]; then
                    echo ""
                    echo "[$COUNTER/$TOTAL] RESUME: SLM=$slm LLM=$LLM_LABEL Q=$q_model subject=$subject"
                    echo "-------------------------------------------"
                    RESUME_FLAG="--resume $RESULT_FILE"
                else
                    echo ""
                    echo "[$COUNTER/$TOTAL] Running: SLM=$slm LLM=$LLM_LABEL Q=$q_model subject=$subject"
                    echo "-------------------------------------------"
                fi

                # Build command
                CMD="python lossy_compression/benchmarks/math/evaluate_math_qa_compression.py"
                CMD="$CMD --subject $subject"
                CMD="$CMD --difficulty $DIFFICULTY"
                CMD="$CMD --slm $slm"
                CMD="$CMD --llm $llm"
                CMD="$CMD --question-model $q_model"
                CMD="$CMD --max-questions $MAX_QUESTIONS"
                CMD="$CMD --batch --batch-size $BATCH_SIZE"
                CMD="$CMD --output $RESULT_FILE"

                if [ "$PARALLEL" = true ]; then
                    CMD="$CMD --parallel --max-workers $MAX_WORKERS"
                fi

                if [ -n "$ORACLE_FLAG" ]; then
                    CMD="$CMD $ORACLE_FLAG"
                fi

                if [ -n "$NUM_PROBLEMS" ]; then
                    CMD="$CMD --num-problems $NUM_PROBLEMS"
                fi

                if [ -n "$RESUME_FLAG" ]; then
                    CMD="$CMD $RESUME_FLAG"
                fi

                echo "Command: $CMD"
                echo ""

                # Run the experiment
                if $CMD; then
                    echo "✓ Completed: $RESULT_NAME"
                    COMPLETED=$((COMPLETED + 1))
                else
                    echo "✗ Failed: $RESULT_NAME"
                    FAILED=$((FAILED + 1))
                fi

                # Small delay between runs
                sleep 1
            done
        done
    done
done

echo ""
echo "=========================================="
echo "MATH Q&A Sweep Complete!"
echo "=========================================="
echo "Summary:"
echo "  Total combinations: $TOTAL"
echo "  Completed: $COMPLETED"
echo "  Skipped (already done): $SKIPPED"
echo "  Failed: $FAILED"
echo ""
echo "Results saved to: $RESULTS_DIR"
echo "=========================================="
