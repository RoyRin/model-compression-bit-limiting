#!/bin/bash

# Default values
DEFAULT_MODEL="sonnet"
DEFAULT_NUM_QUESTIONS=25

# Initialize variables
MODEL_NAME=""
TASK_IDS=()
NUM_QUESTIONS=""
EXTRA_ARGS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -q|--num-questions)
      NUM_QUESTIONS="$2"
      shift 2
      ;;
    --parallel)
      EXTRA_ARGS+=("--parallel")
      shift
      ;;
    --max-workers)
      EXTRA_ARGS+=("--max-workers" "$2")
      shift 2
      ;;
    --verbose)
      EXTRA_ARGS+=("--verbose")
      shift
      ;;
    --status-interval)
      EXTRA_ARGS+=("--status-interval" "$2")
      shift 2
      ;;
    --rate-limit-delay)
      EXTRA_ARGS+=("--rate-limit-delay" "$2")
      shift 2
      ;;
    --batch)
      EXTRA_ARGS+=("--batch")
      shift
      ;;
    --batch-size)
      EXTRA_ARGS+=("--batch-size" "$2")
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [model] [task_ids...] [options]"
      echo "  model: Model name or alias (default: sonnet)"
      echo "  task_ids: Space-separated task IDs (default: ALL - runs 0-163)"
      echo ""
      echo "Options:"
      echo "  -q, --num-questions N      Number of Q&A iterations for QA method (default: 25)"
      echo "  --parallel                  Enable parallel execution"
      echo "  --max-workers N            Number of parallel workers (default: 4)"
      echo "  --batch                    Enable batch Q&A generation (generate all questions at once)"
      echo "  --batch-size N             Number of questions to generate at once in batch mode (default: 10)"
      echo "  --verbose                  Enable verbose output"
      echo "  --status-interval N        Seconds between status updates (default: 10)"
      echo "  --rate-limit-delay N       Base delay for rate limit backoff (default: 1.0)"
      echo ""
      echo "Examples:"
      echo "  $0 haiku                           # Run all problems with haiku model"
      echo "  $0 qa -q 5                         # Run all problems with QA method, 5 iterations"
      echo "  $0 sonnet 1 2 3                    # Run specific problems (1,2,3) with sonnet"
      echo "  $0 qa 10 -q 2                      # Run problem 10 with QA method, 2 iterations"
      echo "  $0 qa --parallel --max-workers 10  # Run QA in parallel with 10 workers"
      echo "  $0 haiku --parallel --verbose      # Run haiku in parallel with verbose output"
      echo "  $0 qa --batch --batch-size 15      # Run QA with batch mode, 15 questions at once"
      exit 0
      ;;
    -*)
      echo "Unknown option: $1"
      echo "Use -h or --help for usage information"
      exit 1
      ;;
    *)
      # First non-flag argument is the model name
      if [ -z "$MODEL_NAME" ]; then
        MODEL_NAME="$1"
      else
        # Subsequent arguments are task IDs (only if they're numbers)
        if [[ "$1" =~ ^[0-9]+$ ]]; then
          TASK_IDS+=("$1")
        else
          echo "Invalid task ID: $1 (must be a number)"
          exit 1
        fi
      fi
      shift
      ;;
  esac
done

# Use defaults if not set
MODEL_NAME="${MODEL_NAME:-$DEFAULT_MODEL}"

# Build task arguments
if [ ${#TASK_IDS[@]} -gt 0 ]; then
  TASK_ARGS=("--task-ids" "${TASK_IDS[@]}")
  echo "Task IDs: ${TASK_IDS[@]}"
else
  # Run ALL HumanEval problems (0-163) by default
  DEFAULT_TASK_IDS=($(seq 0 163))
  TASK_ARGS=("--task-ids" "${DEFAULT_TASK_IDS[@]}")
  echo "Running ALL HumanEval problems (0-163)"
fi

# Add num-questions if specified
if [ -n "$NUM_QUESTIONS" ]; then
  EXTRA_ARGS+=("-q" "$NUM_QUESTIONS")
  echo "Number of Q&A iterations: $NUM_QUESTIONS"
fi

echo "=========================================="
echo "Running HumanEval with model: $MODEL_NAME"
echo "=========================================="

# unlimited (recommended unless you really need a cap)
export EVALPLUS_MAX_MEMORY_BYTES=-1

# Run the Python script with all collected arguments
python benchmarks/humaneval/run_human_eval.py --model "$MODEL_NAME" "${TASK_ARGS[@]}" "${EXTRA_ARGS[@]}"


