#!/bin/bash
# Run LLM-SLM compression experiment (binary Q&A only)

# Activate virtual environment
source ../.venv/bin/activate

# Configuration
BASE_NAME="qa_compression_$(date +%Y%m%d_%H%M%S)"
MAX_ITERATIONS=20
QUALITY_THRESHOLD=8
LLM_MODEL="claude-opus-4-1-20250805"
SLM_MODEL="claude-3-haiku-20240307"
QUESTION_MODEL="claude-3-haiku-20240307"

echo "=========================================="
echo "LLM-SLM Compression Experiment"
echo "=========================================="
echo "Base name: $BASE_NAME"
echo "Questions: 20"
echo "Max iterations per question: $MAX_ITERATIONS"
echo "Quality threshold: $QUALITY_THRESHOLD/10"
echo "LLM: $LLM_MODEL"
echo "SLM: $SLM_MODEL"
echo "Question Model: $QUESTION_MODEL"
echo "=========================================="
echo ""

# Run experiment (binary Q&A only)
echo "📊 Running Q&A Compression Experiment"
echo "=========================================="
python core/run_question_answering_compression.py \
    --name "$BASE_NAME" \
    --iterations $MAX_ITERATIONS \
    --threshold $QUALITY_THRESHOLD \
    --llm "$LLM_MODEL" \
    --slm "$SLM_MODEL" \
    --question-model "$QUESTION_MODEL" \
    --questions experiment_questions.json --verbose 
 
if [ $? -ne 0 ]; then
    echo "❌ Experiment failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ EXPERIMENT COMPLETE!"
echo "=========================================="
echo "📁 Results saved in: experiments/$BASE_NAME/"
echo ""