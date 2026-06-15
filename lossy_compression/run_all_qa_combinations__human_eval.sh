#!/bin/bash

# Run all combinations of LLM, SLM, and Question models for QA evaluation
# Total: 3x3x3 = 27 combinations

# Model arrays
LLMS=("haiku" "sonnet" "opus")
SLMS=("haiku" "sonnet" "opus")
Q_MODELS=("haiku" "sonnet" "opus")

# Configuration
MAX_WORKERS=50
NUM_QUESTIONS=25  # Default number of Q&A iterations
RESULTS_DIR=""  # Will be set later
PROBLEM_SET="all"  # Default to all problems
HAIKU_RESULTS=""
SONNET_RESULTS=""
OPUS_RESULTS=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --max-workers)
      MAX_WORKERS="$2"
      shift 2
      ;;
    --num-questions|-q)
      NUM_QUESTIONS="$2"
      shift 2
      ;;
    --results-dir)
      RESULTS_DIR="$2"
      shift 2
      ;;
    --problem-set)
      PROBLEM_SET="$2"
      shift 2
      ;;
    --haiku)
      HAIKU_RESULTS="$2"
      shift 2
      ;;
    --sonnet)
      SONNET_RESULTS="$2"
      shift 2
      ;;
    --opus)
      OPUS_RESULTS="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --max-workers N        Number of parallel workers (default: 50)"
      echo "  -q, --num-questions N  Number of Q&A iterations (default: 25)"
      echo "  --results-dir DIR      Directory to save/resume results (default: auto-generated)"
      echo "  --problem-set SET      Problem set to evaluate (default: all)"
      echo "                         Options: all, easy, medium, hard, medium+hard"
      echo "  --haiku DIR            Path to haiku results directory (required for problem-set filtering)"
      echo "  --sonnet DIR           Path to sonnet results directory (required for problem-set filtering)"
      echo "  --opus DIR             Path to opus results directory (required for problem-set filtering)"
      echo ""
      echo "Problem sets (requires all three result directories):"
      echo "  all:         All HumanEval problems (0-163)"
      echo "  easy:        Problems where all models pass"
      echo "  medium:      Problems where haiku fails but sonnet/opus pass"
      echo "  hard:        Problems where only opus passes"
      echo "  medium+hard: Both medium and hard problems"
      echo ""
      echo "Examples:"
      echo "  # Run all problems"
      echo "  $0 --max-workers 40 -q 10"
      echo ""
      echo "  # Run only medium problems (same structure as evaluate_QA_performance_on_hard_problems.py)"
      echo "  $0 --problem-set medium \\"
      echo "     --haiku results/claude-3-haiku-20240307/20250914_175801/ \\"
      echo "     --sonnet results/claude-3-7-sonnet-20250219/20250914_181015/ \\"
      echo "     --opus results/claude-opus-4-1-20250805/20250914_181338/"
      echo ""
      echo "This script runs all 27 combinations of:"
      echo "  LLM: haiku, sonnet, opus"
      echo "  SLM: haiku, sonnet, opus"
      echo "  Question model: haiku, sonnet, opus"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use -h or --help for usage information"
      exit 1
      ;;
  esac
done

# Determine which task IDs to use based on problem set
if [ "$PROBLEM_SET" != "all" ]; then
  # Check if we have all required result directories for problem set filtering
  if [ -z "$HAIKU_RESULTS" ] || [ -z "$SONNET_RESULTS" ] || [ -z "$OPUS_RESULTS" ]; then
    echo "Error: Problem set '$PROBLEM_SET' requires all three result directories:"
    echo "  --haiku DIR"
    echo "  --sonnet DIR"
    echo "  --opus DIR"
    exit 1
  fi
  
  # Check directories exist
  if [ ! -d "$HAIKU_RESULTS" ]; then
    echo "Error: Haiku results directory not found: $HAIKU_RESULTS"
    exit 1
  fi
  if [ ! -d "$SONNET_RESULTS" ]; then
    echo "Error: Sonnet results directory not found: $SONNET_RESULTS"
    exit 1
  fi
  if [ ! -d "$OPUS_RESULTS" ]; then
    echo "Error: Opus results directory not found: $OPUS_RESULTS"
    exit 1
  fi
  
  # Use Python to categorize problems based on which models pass
  TASK_IDS=$(python3 -c "
import json
import os
from pathlib import Path

# Load results from directories
def load_results_from_dir(result_dir):
    results = {}
    result_path = Path(result_dir)
    
    # Check if there's a 'problems' subdirectory
    problems_dir = result_path / 'problems'
    if problems_dir.exists():
        search_path = problems_dir
    else:
        search_path = result_path
    
    # Load individual task files
    for task_file in search_path.glob('*.json'):
        if task_file.name.startswith('HumanEval_'):
            with open(task_file, 'r') as f:
                data = json.load(f)
                task_id = task_file.stem.replace('HumanEval_', '')
                # Check if the task passed
                results[task_id] = data.get('passed', False)
    
    return results

# Load results from each directory
haiku_results = load_results_from_dir('$HAIKU_RESULTS')
sonnet_results = load_results_from_dir('$SONNET_RESULTS')
opus_results = load_results_from_dir('$OPUS_RESULTS')

# Categorize problems
easy = []
medium = []
hard = []

# Get all task IDs
all_tasks = set()
all_tasks.update(haiku_results.keys())
all_tasks.update(sonnet_results.keys())
all_tasks.update(opus_results.keys())

for task_id in sorted(all_tasks, key=lambda x: int(x)):
    # Get pass status for each model
    haiku_pass = haiku_results.get(task_id, False)
    sonnet_pass = sonnet_results.get(task_id, False)
    opus_pass = opus_results.get(task_id, False)
    
    # Categorize based on who passes
    if haiku_pass and sonnet_pass and opus_pass:
        easy.append(task_id)
    elif not haiku_pass and sonnet_pass and opus_pass:
        medium.append(task_id)
    elif not haiku_pass and not sonnet_pass and opus_pass:
        hard.append(task_id)

# Output based on requested problem set
problem_set = '$PROBLEM_SET'
if problem_set == 'easy':
    print(' '.join(easy))
elif problem_set == 'medium':
    print(' '.join(medium))
elif problem_set == 'hard':
    print(' '.join(hard))
elif problem_set == 'medium+hard':
    print(' '.join(medium + hard))
")
  
  if [ -z "$TASK_IDS" ]; then
    echo "Warning: No tasks found for problem set '$PROBLEM_SET'"
    echo "This might mean no problems match the criteria."
    exit 0
  fi
  
  # Count tasks
  NUM_TASKS=$(echo $TASK_IDS | wc -w)
  PROBLEM_DESC="$PROBLEM_SET problems ($NUM_TASKS tasks)"
else
  # All HumanEval problems
  TASK_IDS=$(seq 0 163)
  PROBLEM_DESC="all HumanEval problems (0-163)"
fi

# Create or use results directory
if [ -z "$RESULTS_DIR" ]; then
  # Create new timestamped directory
  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  RESULTS_DIR="qa_sweep_${TIMESTAMP}"
  mkdir -p "$RESULTS_DIR"
  echo "Created new results directory: $RESULTS_DIR"
else
  # Use existing directory
  if [ ! -d "$RESULTS_DIR" ]; then
    mkdir -p "$RESULTS_DIR"
    echo "Created results directory: $RESULTS_DIR"
  else
    echo "Using existing results directory: $RESULTS_DIR"
  fi
fi

# Save configuration to the results directory
CONFIG_FILE="$RESULTS_DIR/sweep_config.txt"
echo "QA Hyperparameter Sweep Configuration" > "$CONFIG_FILE"
echo "=====================================" >> "$CONFIG_FILE"
echo "Date: $(date)" >> "$CONFIG_FILE"
echo "Max workers: $MAX_WORKERS" >> "$CONFIG_FILE"
echo "Q&A iterations: $NUM_QUESTIONS" >> "$CONFIG_FILE"
echo "Problem set: $PROBLEM_DESC" >> "$CONFIG_FILE"
echo "Total combinations: 27" >> "$CONFIG_FILE"
echo "" >> "$CONFIG_FILE"
echo "Models:" >> "$CONFIG_FILE"
echo "  LLMs: ${LLMS[*]}" >> "$CONFIG_FILE"
echo "  SLMs: ${SLMS[*]}" >> "$CONFIG_FILE"
echo "  Question models: ${Q_MODELS[*]}" >> "$CONFIG_FILE"

echo "=========================================="
echo "Running ALL QA Model Combinations"
echo "Results directory: $RESULTS_DIR"
echo "Max workers: $MAX_WORKERS"
echo "Q&A iterations: $NUM_QUESTIONS"
echo "Problem set: $PROBLEM_DESC"
echo "Total combinations: 27"
echo "=========================================="

# unlimited (recommended unless you really need a cap)
export EVALPLUS_MAX_MEMORY_BYTES=-1

# Counter for progress
COUNTER=0
TOTAL=27
SKIPPED=0
COMPLETED=0
FAILED=0

# Loop through all combinations
for llm in "${LLMS[@]}"; do
  for slm in "${SLMS[@]}"; do
    for q_model in "${Q_MODELS[@]}"; do
      COUNTER=$((COUNTER + 1))
      
      # Build the expected result directory name
      # Format: results/QA_q{num_questions}/LLM-{llm}_SLM-{slm}_Q-{q_model}
      RESULT_DIR_NAME="LLM-${llm}_SLM-${slm}_Q-${q_model}"
      RESULT_DIR_PATH="$RESULTS_DIR/results/QA_q${NUM_QUESTIONS}/${RESULT_DIR_NAME}"
      
      # Check if this experiment already completed
      if [ -d "$RESULT_DIR_PATH" ] && [ -f "$RESULT_DIR_PATH/summary.json" ]; then
        echo ""
        echo "[$COUNTER/$TOTAL] =========================================="
        echo "SKIPPING (already completed):"
        echo "  LLM: $llm"
        echo "  SLM: $slm"
        echo "  Question Model: $q_model"
        echo "  Found: $RESULT_DIR_PATH"
        echo "=========================================="
        SKIPPED=$((SKIPPED + 1))
        continue
      fi
      
      echo ""
      echo "[$COUNTER/$TOTAL] =========================================="
      echo "Configuration:"
      echo "  LLM: $llm"
      echo "  SLM: $slm"
      echo "  Question Model: $q_model"
      echo "=========================================="
      
      # Change to results directory before running
      cd "$RESULTS_DIR" || exit 1
      
      # Run the evaluation using qa model with specific component models
      python benchmarks/humaneval/run_human_eval.py \
        --model qa \
        --llm-model "$llm" \
        --slm-model "$slm" \
        --question-model "$q_model" \
        --task-ids $TASK_IDS \
        -q "$NUM_QUESTIONS" \
        --parallel \
        --max-workers "$MAX_WORKERS" \
        --verbose \
        --output-name "$RESULT_DIR_NAME"
      
      # Check exit status
      if [ $? -eq 0 ]; then
        echo "✓ Successfully completed: LLM=$llm, SLM=$slm, Q=$q_model"
        COMPLETED=$((COMPLETED + 1))
      else
        echo "✗ Failed: LLM=$llm, SLM=$slm, Q=$q_model"
        echo "  Continuing with next configuration..."
        FAILED=$((FAILED + 1))
      fi
      
      # Go back to original directory
      cd - > /dev/null
      
      # Small delay between runs to avoid overwhelming the API
      sleep 2
    done
  done
done

echo ""
echo "=========================================="
echo "Batch Run Complete!"
echo "=========================================="
echo "Summary:"
echo "  Total configurations: $TOTAL"
echo "  Completed: $COMPLETED"
echo "  Skipped (already done): $SKIPPED"
echo "  Failed: $FAILED"
echo "=========================================="

# Show where results are saved
echo ""
echo "Results saved in: $RESULTS_DIR/"
echo ""
echo "To analyze results for this sweep:"
echo "  python benchmarks/humaneval/analyze_human_eval_results.py $RESULTS_DIR/results_humaneval_*.json"
echo ""
echo "To visualize performance comparison:"
echo "  python benchmarks/aime/evaluate_QA_performance_on_hard_problems.py --analyze-dir $RESULTS_DIR"