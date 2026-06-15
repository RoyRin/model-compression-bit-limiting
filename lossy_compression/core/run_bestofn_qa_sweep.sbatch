#!/bin/bash
#SBATCH --job-name=bestofn_qa
#SBATCH --output=/n/home04/rrinberg/catered_out/bestofn_qa_%j.out
#SBATCH --error=/n/home04/rrinberg/catered_out/bestofn_qa_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Best-of-N + QA Sweep
#
# Protocol:
# 1. SLM generates N=4 candidates (temp=0.7)
# 2. LLM selects correct/best using gold answer
# 3. If correct → done (0 QA bits); if not → 10 QA questions
#
# Usage:
#   sbatch run_bestofn_qa_sweep.sbatch                          # All datasets, easy-only
#   sbatch --export=EASY_ONLY=0 run_bestofn_qa_sweep.sbatch     # All difficulties
#   sbatch --export=DATASET=gsm8k run_bestofn_qa_sweep.sbatch   # Single dataset
#   sbatch --export=N_CANDIDATES=2 run_bestofn_qa_sweep.sbatch  # Fewer candidates
#   sbatch --export=NO_GOLD=1 run_bestofn_qa_sweep.sbatch        # LLM uses own solution

echo "Starting Best-of-N + QA Sweep at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"

cd /n/home04/rrinberg/code/MATS/model-compression-bit-limiting

# Activate environment
ml python
module load cuda cudnn
export CONDA_ENVS_PATH=/n/holylabs/vadhan_lab/Lab/rrinberg/envs/conda/envs
export CONDA_PKGS_DIRS=/n/holylabs/vadhan_lab/Lab/rrinberg/envs/conda/pkgs
module load gcc/15.2.0-fasrc01
mamba activate MATS-inference2

export HF_HOME="/n/home04/rrinberg/data_dir/HF_cache"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PYTHONUNBUFFERED=1

# Use high priority API key
export ANTHROPIC_KEY_FILE="anthropic_high_prio.key"

# Defaults
PARALLEL=${PARALLEL:-6}
LIMIT=${LIMIT:-150}
BASELINE_DIR=${BASELINE_DIR:-results/model-baselines/v4.5}
N_CANDIDATES=${N_CANDIDATES:-4}
EASY_ONLY=${EASY_ONLY:-1}
NO_GOLD=${NO_GOLD:-0}

# Build command
CMD="python lossy_compression/core/run_bestofn_qa_sweep.py"

if [ -z "$DATASET" ]; then
    CMD="$CMD --all"
else
    CMD="$CMD --dataset $DATASET"
    if [ -n "$SUBJECT" ]; then
        CMD="$CMD --subject $SUBJECT"
    fi
fi

CMD="$CMD --limit $LIMIT"
CMD="$CMD --baseline-dir $BASELINE_DIR"
CMD="$CMD --parallel $PARALLEL"
CMD="$CMD --n-candidates $N_CANDIDATES"

if [ "$EASY_ONLY" = "1" ]; then
    CMD="$CMD --easy-only"
fi

if [ "$NO_GOLD" = "1" ]; then
    CMD="$CMD --no-gold"
fi

echo "Running: $CMD"
echo ""
$CMD

echo ""
echo "============================================================"
echo "COMPLETED at $(date)"
echo "============================================================"
