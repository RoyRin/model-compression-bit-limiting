# Project Notes

## ⚠️ Important Notes

**NEVER use em-dashes (`---`) in the paper.** Use commas, colons, semicolons, or parentheses instead.

**ALWAYS use iterative mode for API calls** (not batch mode) until batch API issues are debugged:
```bash
# For QA sweep - use --iterative --parallel 10
python lossy_compression/core/run_qa_sweep.py --all --iterative --parallel 10

# For baselines - use --parallel (iterative is default when not using --batch)
python scripts/run_all_baselines.py --parallel 6
```

---

## Script Organization

### Baseline Scripts

**Use the unified script** for all baseline evaluations:

| Script | Purpose |
|--------|---------|
| `scripts/run_all_baselines.py` | **PRIMARY** - Unified baseline script for all datasets |
| `scripts/run_all_baselines.sbatch` | Sbatch for old models (claude-3.5) |
| `scripts/run_baselines_4.5.sbatch` | Sbatch for 4.5 models |

The per-dataset scripts in `lossy_compression/benchmarks/{dataset}/run_{dataset}_all_models.py` are **legacy** - they do the same thing but with fewer features. Use the unified script instead.

### QA Compression Scripts

There are **two types** of QA scripts that serve different purposes:

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `lossy_compression/core/run_qa_sweep.py` | Runs ALL 27 model combinations (3×3×3) | Production sweeps |
| `lossy_compression/benchmarks/{dataset}/evaluate_{dataset}_qa_compression.py` | Runs ONE specific combination | Debugging, targeted experiments |

**Unified sweep** (`run_qa_sweep.py`):
- Automatically runs all 27 combinations of SLM × LLM × Question-model
- Uses baseline files for initial correctness
- Supports `--iterative` and batch modes

**Per-dataset scripts** (`evaluate_{dataset}_qa_compression.py`):
- Run a single model combination with `--slm`, `--llm`, `--question-model`
- Fine-grained control: `--max-questions`, `--difficulty`
- Useful for testing specific configurations

### Legacy/Redundant Scripts (can be archived)

These scripts in `lossy_compression/core/` are older versions superseded by `run_qa_sweep.py`:
- `run_batch_qa_sweep.py` - older batch-only version
- `batch_qa_sweep_all_datasets.py` - another batch variant
- `batch_qa_compression.py` - yet another variant

---

## LoRA Compression Experiments

### Data Locations

**lmsys dataset:**
- Clusters: `/n/netscratch/sham_lab/Lab/rrinberg/compression/lmsys-clustered/clusters/`
- LoRAs: `/n/netscratch/sham_lab/Lab/rrinberg/compression/lmsys-loras/`
- RAG Index: `/n/netscratch/sham_lab/Lab/rrinberg/compression/lmsys-clustered/lora_rag_index/`

**wildchat dataset:**
- Clusters: `/n/netscratch/sham_lab/Lab/rrinberg/compression/wildchat-clustered/clusters/`
- LoRAs: `/n/netscratch/sham_lab/Lab/rrinberg/compression/wildchat-loras/`

### Experiment Scripts

| Experiment | Script | Sbatch | Description |
|------------|--------|--------|-------------|
| LoRA Pairwise Evaluation | `scripts/lora/evaluate_lora_compression.py` | `scripts/lora/evaluate_lora_compression.sbatch` | Compares correct LoRA vs wrong LoRAs vs baseline |
| RAG LoRA Evaluation | `scripts/lora/evaluate_rag_lora.py` | `scripts/lora/evaluate_rag_lora.sbatch` | Tests RAG routing accuracy + compression with RAG-selected LoRA |
| RAG Routing Strategies | `scripts/lora/evaluate_rag_routing_strategies.py` | `scripts/lora/evaluate_rag_routing_strategies.sbatch` | Compares prompt-RAG vs response-RAG vs full-RAG routing |
| Cascade LoRA Selection | `scripts/lora/evaluate_cascade_lora_selection.py` | `scripts/lora/evaluate_cascade_lora_selection.sbatch` | RAG → top-N → perplexity refinement |
| Build RAG Index | `scripts/lora/build_lora_index.py` | `scripts/lora/build_lora_index.sbatch` | Builds FAISS index for LoRA routing |

### Results Locations

| Experiment | Results Directory |
|------------|-------------------|
| LoRA Pairwise (lmsys) | `results/lora_evaluation_pairwise_lmsys/` |
| LoRA Evaluation (lmsys) | `results/lora_evaluation_lmsys/` |
| RAG LoRA Evaluation | `results/rag_lora_evaluation_lmsys/` |
| RAG Routing Strategies | `results/rag_routing_strategies_lmsys/` |
| Cascade LoRA Selection | `results/cascade_lora_selection_lmsys/` |

### Running Experiments

```bash
# LoRA pairwise evaluation (correct vs wrong LoRAs)
sbatch scripts/lora/evaluate_lora_compression.sbatch

# RAG LoRA evaluation (both accuracy + compression)
sbatch scripts/lora/evaluate_rag_lora.sbatch

# RAG accuracy only (faster, no compression)
sbatch --export=MODE=accuracy scripts/lora/evaluate_rag_lora.sbatch

# RAG compression only
sbatch --export=MODE=compression scripts/lora/evaluate_rag_lora.sbatch

# RAG routing strategies comparison (prompt vs response vs full)
sbatch scripts/lora/evaluate_rag_routing_strategies.sbatch

# Cascade LoRA selection (RAG → top-N → perplexity refinement)
sbatch scripts/lora/evaluate_cascade_lora_selection.sbatch
```

### Key Results (LoRA Pairwise - lmsys, 2026-01-12)

| Cluster | Baseline | Correct LoRA | Wrong LoRA (avg) | Improvement |
|---------|----------|--------------|------------------|-------------|
| 0 | 3.18 | 1.57 | 1.92 | -50.6% |
| 1 | 2.68 | 1.61 | 1.82 | -40.0% |
| 2 | 4.22 | 1.43 | 2.90 | -66.2% |
| 3 | 2.41 | 1.41 | 1.63 | -41.6% |
| 4 | 1.59 | 0.75 | 1.16 | -52.9% |
| **Avg** | **2.81** | **1.35** | **1.89** | **-52.0%** |

### Enwik9 LoRA Experiments

**⚠️ IMPORTANT: Multiple configurations exist. Be careful not to mix them up!**

#### Data Sources

| Directory | Clusters | Description |
|-----------|----------|-------------|
| `enwik9-clustered` | 10 | Original 10-cluster split |
| `enwik9-clustered-50` | 50 | Finer 50-cluster split |

All at: `/n/netscratch/sham_lab/Lab/rrinberg/compression/`

#### LoRA Models

| Directory | Base Model | Clusters |
|-----------|------------|----------|
| `enwik9-loras` | Mistral-7B-Instruct-v0.2 | 10 |
| `enwik9-loras-50` | Mistral-7B-Instruct-v0.2 | 50 |
| `enwik9-loras-llama-base` | Llama-3.1-8B | 10 |
| `enwik9-loras-llama-instruct` | Llama-3.1-8B-Instruct | 10 |

#### Experiment Status (2026-02-03)

| Experiment | Base Model | Clusters | Bits | Status |
|------------|------------|----------|------|--------|
| `rag_lora_evaluation_enwik9-50` | Mistral-Instruct | 50 | 64 | ✓ COMPLETE (50/50) |
| `rag_lora_evaluation_enwik9-50-full` | Mistral-Instruct | 50 | 64 | ✓ COMPLETE (50/50) |
| `rag_lora_enwik9_mistral_120bit_parallel` | Mistral-Instruct | 10 | 120 | ⚠️ INCOMPLETE (9/10, missing cluster 2) |
| `rag_lora_enwik9_llama-base_120bit_parallel` | Llama-3.1-8B | 10 | 120 | ⚠️ INCOMPLETE (8/10, missing clusters 2,3) |
| `rag_lora_enwik9_llama-instruct_120bit_parallel` | Llama-3.1-8B-Inst | 10 | 120 | ⚠️ INCOMPLETE (9/10, missing cluster 2) |
| `rag_lora_evaluation_enwik9` | Mistral-Instruct | 10 | 64 | ⚠️ INCOMPLETE (5/10) |
| `rag_lora_evaluation_enwik9_base` | Mistral-7B-v0.1 ⚠️ | 10 | 64 | ⚠️ WRONG MODEL + INCOMPLETE |

**⚠️ `enwik9_base` uses Mistral-7B-v0.1 (base) but LoRAs were trained on Mistral-7B-Instruct-v0.2!**

#### Verified Experiments (for paper)

The 50-cluster experiments are verified correct:
- ✓ Base model matches LoRA training model (Mistral-7B-Instruct-v0.2)
- ✓ All 50 clusters complete
- ✓ Results in appendix: `writing/arxiv/sections/appendix-enwik-lora.tex`

```bash
# Print compression table
python scripts/lora/print_compression_table.py results/rag_lora_evaluation_enwik9-50/rag_lora_results_20260202_153357.json --sort rag
```

#### Compression Ratio Formula

```
compression_ratio = total_bits / (num_tokens × log2(vocab_size))
                  = total_bits / (num_tokens × 14.97)  # for Mistral, vocab=32000

where total_bits = num_blocks × (bit_precision + 7)
                 = num_blocks × 71  # for 64-bit precision
```

---

## Best-of-N Compression Experiments (AIME)

### Quick Reference

```bash
# Plot all 3 approaches with std dev (RECOMMENDED)
python lossy_compression/core/plot_best_of_n_results.py \
    results/best_of_n_aime/best_of_n_comparison_20260115_065015.json \
    --just-ask results/just_ask_best_of_n/just_ask_best_of_n_20260115_174758.json \
    --output-dir results/best_of_n_aime/plots --format pdf

# Run new unified experiment
sbatch lossy_compression/core/run_best_of_n_unified.sbatch
```

### Results Locations

| Location | Description | Has Compression? |
|----------|-------------|------------------|
| `results/best_of_n_aime/best_of_n_comparison_20260115_065015.json` | **Best results** (90 problems, temp+single_prompt) | **Yes** |
| `results/just_ask_best_of_n/just_ask_best_of_n_20260115_174758.json` | **Just-ask results** (90 problems) | **Yes** |
| `results/best_of_n_aime/plots/` | Generated plots (PDF + PNG) | - |
| `results/best_of_n_unified/` | Unified experiment results | No (cluster GPU issues) |

### Code Location

| File | Description |
|------|-------------|
| `lossy_compression/core/plot_best_of_n_results.py` | **Plotting script** |
| `lossy_compression/core/run_best_of_n_unified.py` | Unified experiment (all 3 approaches) |
| `lossy_compression/core/run_best_of_n_unified.sbatch` | Sbatch for unified experiment |
| `lossy_compression/core/best_of_n_approaches_experiment.py` | Legacy: temp + single_prompt only |

### Approaches

**Approach 1: Temperature Sampling**
- Generate N independent solutions using high temperature (0.8)
- Each solution is a separate API call to Claude

**Approach 2: Single-Prompt N-Responses**
- Ask Claude to generate N different solutions in a single prompt
- One API call that explicitly requests N distinct approaches

**Approach 3: Just Ask**
- Generate verbose solution first, then N succinct rewrites
- Tests whether asking for brevity improves compressibility

### Key Results (2026-01-15, 90 AIME problems)

| Approach | N=1 | N=5 | N=10 | Notes |
|----------|-----|-----|------|-------|
| Temperature | 6.4% comp, 40% acc | 5.9% comp, 41% acc | 5.6% comp, 39% acc | Compression improves with N |
| Single Prompt | 9.7% comp, 40% acc | 7.0% comp, 41% acc | 7.7% comp, 34% acc | Higher variance |

### Running

```bash
# Plot all 3 approaches (temperature, single_prompt, just_ask) with std dev
python lossy_compression/core/plot_best_of_n_results.py \
    results/best_of_n_aime/best_of_n_comparison_20260115_065015.json \
    --just-ask results/just_ask_best_of_n/just_ask_best_of_n_20260115_174758.json \
    --output-dir results/best_of_n_aime/plots --format pdf

# Run unified experiment (90 problems, 3 trials, N=1,3,5,10)
sbatch lossy_compression/core/run_best_of_n_unified.sbatch

# Run with batch API (faster, 50% cheaper)
sbatch --export=BATCH=1 lossy_compression/core/run_best_of_n_unified.sbatch

# Quick test run
python lossy_compression/core/run_best_of_n_unified.py --num-problems 5 --num-trials 1
```

---

## Request-Based Compression ("Just Ask") Experiments

### What It Does

Tests whether explicitly asking a model to write a more succinct solution improves compression. The hypothesis is that shorter, denser solutions might compress better.

**Process:**
1. Generate an initial verbose solution to an AIME problem
2. Strip the final answer from the solution
3. Ask Claude to rewrite the solution as succinctly as possible while preserving enough info to infer the answer
4. Compress both versions with arithmetic coding and compare

**Key Finding (2026-01-12):**
Surprisingly, succinct solutions are often LESS compressible than verbose ones:
- Verbose original: 2115 chars, 5.3% compression
- Succinct rewrite: 589 chars, 9.5% compression

This suggests that verbose solutions use predictable boilerplate ("Step 1:", "Therefore,") that LLMs easily predict, while dense solutions use unusual phrasing that's harder to predict.

### Code Location

| File | Description |
|------|-------------|
| `lossy_compression/core/request_based_compression_experiment.py` | Main experiment script |
| `lossy_compression/core/run_request_based_compression.sbatch` | Sbatch script for cluster |

### Results Location

- Directory: `results/request_based_compression/`
- JSON includes: initial prompt, initial response, compress prompt, compressed response, lengths, compression ratios

### Running

```bash
# Run on cluster (100 problems)
sbatch lossy_compression/core/run_request_based_compression.sbatch

# Run locally with verbose output (see prompts/responses)
cd lossy_compression/core
python request_based_compression_experiment.py --num-problems 1 --verbose
```

---

## Benchmark Evaluation Scripts

### Comprehensive Baseline Script (RECOMMENDED)

**Use this single script to run ALL baseline evaluations consistently:**

```bash
# Run ALL datasets (GSM8K, MATH, GPQA, MBPP) - iterative mode
sbatch scripts/run_all_baselines.sbatch

# Use batch API (50% cheaper, more efficient for large runs)
sbatch --export=BATCH=1 scripts/run_all_baselines.sbatch

# Run specific dataset
sbatch --export=DATASET=gsm8k scripts/run_all_baselines.sbatch
sbatch --export=DATASET=math,SUBJECT=algebra scripts/run_all_baselines.sbatch
sbatch --export=DATASET=gpqa,FORMAT=mc scripts/run_all_baselines.sbatch

# Resume from partial results (if job dies)
sbatch --export=RESUME=1 scripts/run_all_baselines.sbatch

# Quick test run
python scripts/run_all_baselines.py --dataset gsm8k --num-problems 10

# Batch mode via command line
python scripts/run_all_baselines.py --dataset gsm8k --batch
```

**Script:** `scripts/run_all_baselines.py`
**Sbatch:** `scripts/run_all_baselines.sbatch`

Features:
- Consistent evaluation logic across all datasets
- **Batch API mode** (`--batch`): Uses Anthropic Message Batches API for Claude models (50% cheaper)
- Exponential backoff on rate limits (30s → 480s)
- Resume support from partial results
- Saves progress every 10 problems
- Correct GPQA answer shuffling (seed = 42 + problem_id)
- Non-Claude models (e.g., gpt-oss) run iteratively even in batch mode

### Current Baseline Results (Paper Table)

Results in paper appendix Table 2 (`writing/.../sections/appendix.tex`):

| Dataset | n | Haiku | Sonnet | Opus | Source File |
|---------|---|-------|--------|------|-------------|
| MATH (Algebra) | 1,187 | 83.2% | 89.0% | 90.5% | `math_all_models_algebra_20260115_001427.json` |
| MATH (Geometry) | 479 | 68.7% | 71.4% | 69.7% | `math_all_models_geometry_20260114_213358.json` |
| MATH (Num. Theory) | 540 | 93.1% | 93.7% | 93.1% | `math_all_models_number_theory_20260114_213908.json` |
| GSM8K | 1,319 | 63.0% | 96.6% | 95.8% | `gsm8k_all_models_20260115_215021.json` |
| GPQA (MC) | 198 | 58.6% | 73.7% | 71.7% | `gpqa_all_models_20260115_185611.json` |
| GPQA (Freeform) | 126 | 36.5% | 55.6% | 52.4% | `gpqa_freeform_all_models_20260115_184911.json` |
| MBPP | 500 | 49.8% | 55.6% | 58.2% | `mbpp_all_models_test_20260115_154846.json` |

All result files are in: `lossy_compression/results/`

### Legacy Per-Dataset Scripts

These scripts run haiku, sonnet, and opus on each dataset to classify problem difficulty:
- **easy**: all models pass
- **medium**: haiku fails, sonnet/opus pass
- **hard**: haiku/sonnet fail, opus passes
- **very_hard**: all models fail

| Dataset | Baseline Script | Sbatch | Problems |
|---------|-----------------|--------|----------|
| MATH | `lossy_compression/benchmarks/math/run_math_all_models.py` | `run_math_all_models.sbatch` | 2206 (algebra, geometry, number_theory) |
| GSM8K | `lossy_compression/benchmarks/gsm8k/run_gsm8k_all_models.py` | `run_gsm8k_all_models.sbatch` | 1319 |
| GPQA | `lossy_compression/benchmarks/gpqa/run_gpqa_all_models.py` | `run_gpqa_all_models.sbatch` | 198 (multiple choice) |
| GPQA-Freeform | `lossy_compression/benchmarks/gpqa/run_gpqa_freeform_all_models.py` | `run_gpqa_freeform_all_models.sbatch` | 126 (free-form, LLM-as-judge) |
| HumanEval | `lossy_compression/benchmarks/humaneval/run_humaneval_all_models.py` | `run_humaneval_all_models.sbatch` | 164 |
| MBPP | `lossy_compression/benchmarks/mbpp/run_mbpp_all_models.py` | `run_mbpp_all_models.sbatch` | 257 (sanitized) |

### Q&A Compression Scripts (20 Questions)

These scripts run the iterative Q&A compression approach on problems:

| Dataset | Q&A Script | Sbatch | Notes |
|---------|------------|--------|-------|
| MATH | `lossy_compression/benchmarks/math/evaluate_math_qa_compression.py` | `run_math_qa_compression.sbatch` | Has `--oracle` mode |
| GSM8K | `lossy_compression/benchmarks/gsm8k/evaluate_gsm8k_qa_compression.py` | `run_gsm8k_qa_compression.sbatch` | ✓ Ready |
| GPQA | `lossy_compression/benchmarks/gpqa/evaluate_gpqa_qa_compression.py` | `run_gpqa_qa_compression.sbatch` | ✓ Ready (multiple choice) |
| HumanEval | `lossy_compression/benchmarks/humaneval/run_human_eval.py` | - | Q&A built into main script |
| MBPP | `lossy_compression/benchmarks/mbpp/evaluate_mbpp_qa_compression.py` | `run_mbpp_qa_compression.sbatch` | ✓ Ready |
| GPQA-Freeform | `lossy_compression/benchmarks/gpqa/evaluate_gpqa_freeform_qa_compression.py` | `run_gpqa_freeform_qa_compression.sbatch` | ✓ Ready, uses LLM-as-judge |

### Results Location

All results saved to: `lossy_compression/results/`

Naming patterns:
- `{dataset}_all_models_{timestamp}.json` - Baseline difficulty classification
- `{dataset}_qa_{slm}_{llm}_{difficulty}_{timestamp}.json` - Q&A compression results

---

## Experimental Plan (Paper)

### Phase 1: Baseline Difficulty Classification (IN PROGRESS)
1. Run all-models baseline on each dataset
2. Classify problems as easy/medium/hard/very_hard
3. Add difficulty distribution tables to paper appendix

### Phase 2: Q&A Compression Experiments (NEXT)
1. Filter to medium/hard problems (where haiku fails but stronger models pass)
2. Run Q&A compression (20 questions) with haiku as SLM
3. Measure: Does Q&A help haiku recover accuracy to sonnet/opus level?
4. Key metric: Accuracy improvement on medium/hard problems

### Phase 3: Analysis
1. Compare Q&A compression vs baseline across datasets
2. Analyze which problem types benefit most from Q&A
3. Cost analysis: API calls vs accuracy gain

---

## Dataset References

**IMPORTANT:** Use these exact HuggingFace dataset identifiers:

| Dataset | HuggingFace ID | Split |
|---------|----------------|-------|
| GSM8K | `openai/gsm8k` (config: `main`) | `test` |
| MATH | `EleutherAI/hendrycks_math` (config: `algebra`, `geometry`, etc.) | `test` |
| GPQA | `Idavidrein/gpqa` (config: `gpqa_diamond`) | `train` |
| MBPP | `google-research-datasets/mbpp` (config: `sanitized`) | `test` |
| HumanEval | `evalplus/humanevalplus` | `test` |
| AIME | `AI-MO/aimo-validation-aime` | `train` |
| HLE | `cais/hle` | `test` (2158 text-only of 2500 total) |

---

## Logs

Slurm logs are stored in: `/n/home04/rrinberg/catered_out/`

Common log patterns:
- `eval_lora_*.out` - LoRA evaluation logs
- `eval_rag_lora_*.out` - RAG LoRA evaluation logs
- `best_of_n_aime_*.out` - Best-of-N AIME logs
- `request_compress_*.out` - Request-based compression logs
- `gsm8k_all_models_*.out` - GSM8K baseline logs
- `gpqa_all_models_*.out` - GPQA baseline logs
- `gpqa_freeform_*.out` - GPQA freeform logs
- `humaneval_all_models_*.out` - HumanEval baseline logs
- `mbpp_all_models_*.out` - MBPP baseline logs
- `math_qa_*.out` - MATH Q&A compression logs

---

## BigCodeBench Integration (IN PROGRESS)

**Status**: Initial testing complete, ready to build baseline and QA scripts.

**Submodule**: `external/bigcodebench/` (git@github.com:bigcode-project/bigcodebench.git)

**Dataset**:
- HuggingFace: `bigcode/bigcodebench`
- Full: 1140 problems
- Hard subset: 148 problems
- Uses `unittest.TestCase` for validation

**Problem format**:
```python
{
  "task_id": "BigCodeBench/25",
  "entry_point": "task_func",           # Function name to implement
  "instruct_prompt": "...",             # Use this as problem description
  "test": "class TestCases(...)...",    # unittest-based validation
  "canonical_solution": "..."           # Reference solution
}
```

**Loading**:
```python
from bigcodebench.data import get_bigcodebench
problems = get_bigcodebench(subset='full')  # or 'hard'
```

**Next steps**:
1. Create `lossy_compression/benchmarks/bigcodebench/run_bigcodebench_all_models.py` - baseline difficulty classification
2. Create `lossy_compression/benchmarks/bigcodebench/evaluate_bigcodebench_qa_compression.py` - QA compression

**Notes**:
- Tested loading and basic evaluation - works
- Similar integration pattern to MBPP
- Use `instruct_prompt` for QA compression (not `complete_prompt`)

---

## Claude 4.5 Models Experiment

**Model IDs** (in `lossy_compression/__init__.py` and `lossy_compression/core/run_qa_sweep.py`):
```python
MODEL_IDS = {
    'haiku': 'claude-haiku-4-5-20251001',
    'sonnet': 'claude-sonnet-4-5-20250929',
    'opus': 'claude-opus-4-5-20251101',
    'gpt-oss': 'openai/gpt-oss-120b',  # Via OpenRouter
}
```

**Data Locations**:
- Baselines: `lossy_compression/results/baselines_4.5/`
- QA Sweep: `results/qa_sweep_4.5/`

**Scripts**:
- Baseline sbatch: `scripts/run_baselines_4.5.sbatch`
- QA Sweep sbatch: `lossy_compression/core/run_qa_sweep.sbatch` with `--export=BASELINE_DIR=...,OUTPUT_DIR=...`

**GPT-OSS Notes**:
- Reasoning model via OpenRouter (`openai/gpt-oss-120b`)
- Needs higher max_tokens due to chain-of-thought (uses ~150-200 tokens for reasoning before output)
- Answer task: 500 tokens (vs 10 for Claude models)
- Other tasks: +250 tokens over base

**Running**:
```bash
# Baselines with 4.5 models + gpt-oss
sbatch --export=GPT_OSS=1 scripts/run_baselines_4.5.sbatch

# QA sweep using 4.5 baselines
sbatch --dependency=afterok:BASELINE_JOB_ID \
    --export=BASELINE_DIR=lossy_compression/results/baselines_4.5,OUTPUT_DIR=results/qa_sweep_4.5 \
    lossy_compression/core/run_qa_sweep.sbatch
```

---

## Iterative QA Sweep with Judge Ablations

**Script**: `lossy_compression/core/run_iterative_qa_sweep.py`
**Sbatch**: `lossy_compression/core/run_iterative_qa_sweep.sbatch`
**Results**: `results/iterative-qa-sweep/v4.5/`

**Key differences from `run_qa_sweep.py`**:
- LLM uses its own solution as reference (not gold answer) for answering questions
- Quality-thresholding judge can early-stop the protocol
- Questions generated in 2 batches of 5 (not 1 batch of 10)

### Completed Runs

| Job ID | Description | Results |
|--------|-------------|---------|
| 57094775 | Iterative QA sweep, v4.5, objective+comparison judge, threshold=7 | 32 files in `results/iterative-qa-sweep/v4.5/` |

### Ablation Runs (2026-01-26)

These ablations disentangle two confounds in the iterative protocol:
1. **Higher threshold (≥9)**: Does the judge accept answers too easily at threshold=7?
2. **Gold-answer judge**: Does the judge evaluate poorly because the LLM's own solution is wrong?

**Important**: In both ablations, the LLM still answers questions using its OWN solution (not gold). Only the judge's evaluation reference changes for `--gold-judge`.

| Job ID | Ablation | Command | Output Pattern |
|--------|----------|---------|----------------|
| 57134466 | Threshold=9 | `--judge-mode objective --quality-threshold 9 --parallel 6` | `{dataset}_{BLC\|QA}_objective_t9_v4.5_*.json` |
| 57134609 | Gold judge | `--judge-mode objective --gold-judge --parallel 6` | `{dataset}_{BLC\|QA}_objective_goldjudge_v4.5_*.json` |

**CLI args added**:
- `--quality-threshold N` (default: 7) — judge early-stop threshold
- `--gold-judge` (flag) — give judge the gold answer for evaluation

**Gold judge caveat**: For MATH-type datasets, gold_answer enables exact-match checking (score=10 if correct, else standalone eval). For CODE/DEFAULT datasets, it falls back to standalone evaluation (same as objective mode without gold). So the gold-judge ablation is most meaningful for MATH/GSM8K/AIME.

**Paper location**: `appendix-negative-results.tex`, Section \ref{app:judge-ablation}

---

## Current Jobs (2026-01-27)

| Job ID | Name | Description | Status |
|--------|------|-------------|--------|
| 57134466 | iter_qa_sweep | Ablation: threshold=9, objective judge, v4.5 | Running |
| 57134609 | iter_qa_sweep | Ablation: gold-judge, objective judge, v4.5 | Running |
| 57248689 | qa_variance | Proper QA variance (3 trials, BLC/QA/QA+, 8 datasets, parallel=10) | Cancelled (gsm8k + math_algebra done) |
| 57286593 | qa_variance | Resumed proper QA variance (parallel=18, high-prio key, 6 remaining datasets) | Running |
| 57307078 | gptoss_baselines | GPT-OSS baselines for 6 missing datasets (aime, gpqa_mc, gpqa_freeform, mbpp, hle, mmlu_pro) | Running |

---

## Plotting Conventions

- **Format**: Always save plots as PDF (not PNG) for paper quality
- **Filenames**: Always include timestamp in format `{plot_name}_{YYYYMMDD_HHMMSS}.pdf`
- **Example**: `best_of_n_accuracy_comparison_20260116_141953.pdf`
