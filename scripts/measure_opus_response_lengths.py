#!/usr/bin/env python3
"""
Measure Opus response lengths for problems where Haiku got wrong.

For each dataset, loads the baseline difficulty classification, samples up to 30
problems per difficulty (medium, hard, very_hard), runs Opus to get full responses,
and records the number of tokens using tiktoken.

Usage:
    # Full run (all datasets, 30 per difficulty)
    python scripts/measure_opus_response_lengths.py

    # Quick test
    python scripts/measure_opus_response_lengths.py --max-per-difficulty 3

    # Specific dataset
    python scripts/measure_opus_response_lengths.py --dataset gsm8k

    # Resume from partial results
    python scripts/measure_opus_response_lengths.py --resume

    # Parallel API calls
    python scripts/measure_opus_response_lengths.py --parallel 6
"""

import json
import time
import argparse
import random
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))

import tiktoken
from datasets import load_dataset
from lossy_compression import MODEL_ALIAS_MAP, model_completion
from lossy_compression.benchmarks.hle import load_hle_dataset, get_hle_problem, build_hle_prompt
from lossy_compression.benchmarks.aime import load_aime_dataset, get_aime_problem, build_aime_prompt
from utils.llm_api import get_anthropic_key

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OPUS_MODEL = MODEL_ALIAS_MAP["opus"]  # claude-opus-4-5-20251101
ENCODER = tiktoken.get_encoding("cl100k_base")
BASELINE_DIR = Path("results/model-baselines/v4.5")
RATE_LIMIT_BACKOFFS = [30, 60, 120, 240, 480, 480, 480, 480, 480, 480]
RATE_LIMIT_ERRORS = [
    'rate_limit', 'rate limit', 'too many requests', '429', 'overloaded',
    'capacity', 'throttl'
]

# ---------------------------------------------------------------------------
# Dataset definitions: how to build prompts from baseline problem records
# ---------------------------------------------------------------------------


def _gsm8k_prompt(problem: Dict) -> Tuple[str, str, int]:
    """Returns (prompt, system, max_tokens)."""
    system = "You are a math tutor. Solve problems step by step. End with #### followed by the numerical answer."
    question = problem["question"]
    prompt = f"Solve this math problem:\n\n{question}\n\nShow your work, then give the final answer after ####"
    return prompt, system, 2000


def _math_prompt(problem: Dict) -> Tuple[str, str, int]:
    system = "You are a math expert. Solve problems step by step. Put your final answer in \\boxed{}."
    question = problem["question"]
    prompt = f"Solve this problem:\n\n{question}\n\nPut your final answer in \\boxed{{}}"
    return prompt, system, 2000


def _gpqa_mc_prompt(problem: Dict) -> Tuple[str, str, int]:
    """For GPQA MC, the baseline stores the full question text.
    We need to reconstruct the MC prompt, but we don't have the original
    choices stored. Instead, we reload from HuggingFace.
    """
    system = "You are an expert scientist with deep knowledge in physics, chemistry, and biology."
    # The question field in the baseline is the raw question without choices.
    # We'll reconstruct it when we load the dataset.
    # This function is a placeholder; actual prompt is built in _sample_gpqa.
    raise NotImplementedError("GPQA uses dataset-level loading")


def _mbpp_prompt(problem: Dict) -> Tuple[str, str, int]:
    system = "You are an expert Python programmer. Write clean, efficient code."
    prompt_text = problem.get("prompt", problem.get("question", ""))
    prompt = f"""Write a Python function that solves this problem:

{prompt_text}

Provide only the Python code, no explanation needed."""
    return prompt, system, 1000


def _aime_prompt(problem: Dict) -> Tuple[str, str, int]:
    system = "You are an expert competition mathematician. Solve problems carefully and show your work."
    prompt = build_aime_prompt(problem["question"] if "question" in
                               problem else problem["problem"])
    return prompt, system, 2048


def _hle_prompt(problem: Dict) -> Tuple[str, str, int]:
    system = "You are a highly knowledgeable assistant. Answer questions carefully and precisely."
    # We need answer_type to build the prompt properly.
    # The baseline stores 'question' but not answer_type; we'll handle in the sampler.
    question = problem.get("question", problem.get("problem", ""))
    # Default to exactMatch format
    prompt = build_hle_prompt(question, problem.get("answer_type",
                                                    "exactMatch"))
    return prompt, system, 2048


# ---------------------------------------------------------------------------
# API call with retry
# ---------------------------------------------------------------------------


def call_opus(prompt: str,
              system: str,
              max_tokens: int,
              max_retries: int = 10) -> Dict:
    """Call Opus with exponential backoff on rate limits."""
    for attempt in range(max_retries):
        try:
            start = time.time()
            response = model_completion(
                model=OPUS_MODEL,
                system=system,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.0,
            )
            elapsed = time.time() - start
            n_tokens = len(ENCODER.encode(response))
            return {
                "response": response,
                "n_tokens": n_tokens,
                "n_chars": len(response),
                "time": elapsed,
                "error": None,
            }
        except Exception as e:
            error_str = str(e).lower()
            if any(term in error_str for term in RATE_LIMIT_ERRORS):
                wait = RATE_LIMIT_BACKOFFS[min(attempt,
                                               len(RATE_LIMIT_BACKOFFS) - 1)]
                print(
                    f"\n  Rate limited, waiting {wait}s (attempt {attempt+1}/{max_retries})..."
                )
                time.sleep(wait)
            else:
                return {
                    "response": None,
                    "n_tokens": 0,
                    "n_chars": 0,
                    "time": 0,
                    "error": str(e),
                }
    return {
        "response": None,
        "n_tokens": 0,
        "n_chars": 0,
        "time": 0,
        "error": "Max retries exceeded"
    }


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------


def sample_problems(baseline_path: Path,
                    difficulties: List[str],
                    max_per_difficulty: int,
                    seed: int = 42) -> Dict[str, List[Dict]]:
    """Load baseline and sample problems by difficulty.

    Returns dict: difficulty -> list of problem dicts from baseline.
    """
    with open(baseline_path) as f:
        data = json.load(f)

    by_diff = {d: [] for d in difficulties}
    for r in data["results"]:
        d = r["difficulty"]
        if d in by_diff:
            by_diff[d].append(r)

    rng = random.Random(seed)
    sampled = {}
    for d in difficulties:
        pool = by_diff[d]
        n = min(max_per_difficulty, len(pool))
        sampled[d] = rng.sample(pool, n) if n > 0 else []
    return sampled


# ---------------------------------------------------------------------------
# Per-dataset runners
# ---------------------------------------------------------------------------


def run_dataset_simple(name: str, baseline_file: str, prompt_fn,
                       difficulties: List[str], max_per_difficulty: int,
                       parallel: int) -> Dict:
    """Generic runner for datasets where we can build the prompt from the baseline record."""
    baseline_path = BASELINE_DIR / baseline_file
    if not baseline_path.exists():
        print(f"  SKIP: {baseline_path} not found")
        return {"dataset": name, "error": "baseline not found", "results": []}

    sampled = sample_problems(baseline_path, difficulties, max_per_difficulty)
    total = sum(len(v) for v in sampled.values())
    print(f"  Sampled {total} problems: " + ", ".join(f"{d}={len(sampled[d])}"
                                                      for d in difficulties))

    all_results = []
    done = 0

    def process_one(problem: Dict, difficulty: str):
        prompt, system, max_tokens = prompt_fn(problem)
        result = call_opus(prompt, system, max_tokens)
        return {
            "problem_idx": problem["problem_idx"],
            "difficulty": difficulty,
            "n_tokens": result["n_tokens"],
            "n_chars": result["n_chars"],
            "time": result["time"],
            "error": result["error"],
        }

    tasks = []
    for d in difficulties:
        for problem in sampled[d]:
            tasks.append((problem, d))

    if parallel > 1:
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = {
                pool.submit(process_one, p, d): (p, d)
                for p, d in tasks
            }
            for future in as_completed(futures):
                r = future.result()
                all_results.append(r)
                done += 1
                print(
                    f"\r  [{done}/{total}] {r['difficulty']} idx={r['problem_idx']} -> {r['n_tokens']} tokens",
                    end="",
                    flush=True)
    else:
        for problem, d in tasks:
            r = process_one(problem, d)
            all_results.append(r)
            done += 1
            print(
                f"\r  [{done}/{total}] {r['difficulty']} idx={r['problem_idx']} -> {r['n_tokens']} tokens",
                end="",
                flush=True)

    print()
    return {"dataset": name, "results": all_results}


def run_gpqa_mc(difficulties: List[str], max_per_difficulty: int,
                parallel: int) -> Dict:
    """GPQA MC needs special handling: we must reload the dataset to get answer choices."""
    baseline_path = BASELINE_DIR / "gpqa_mc_v4.5.json"
    if not baseline_path.exists():
        print(f"  SKIP: {baseline_path} not found")
        return {
            "dataset": "gpqa_mc",
            "error": "baseline not found",
            "results": []
        }

    # Load baseline for difficulty + problem_idx
    with open(baseline_path) as f:
        baseline_data = json.load(f)

    by_diff = {d: [] for d in difficulties}
    for r in baseline_data["results"]:
        d = r["difficulty"]
        if d in by_diff:
            by_diff[d].append(r)

    rng = random.Random(42)
    sampled = {}
    for d in difficulties:
        pool = by_diff[d]
        n = min(max_per_difficulty, len(pool))
        sampled[d] = rng.sample(pool, n) if n > 0 else []

    total = sum(len(v) for v in sampled.values())
    print(f"  Sampled {total} problems: " + ", ".join(f"{d}={len(sampled[d])}"
                                                      for d in difficulties))

    # Load the HuggingFace dataset to get full problem details
    ds = load_dataset('Idavidrein/gpqa', 'gpqa_diamond')
    hf_problems = list(ds['train'])

    system = "You are an expert scientist with deep knowledge in physics, chemistry, and biology."
    max_tokens = 1250

    all_results = []
    done = 0

    def process_one(problem: Dict, difficulty: str):
        idx = problem["problem_idx"]
        hf = hf_problems[idx]
        # Reconstruct the MC prompt with deterministic shuffling
        answers = [
            hf['Incorrect Answer 1'],
            hf['Incorrect Answer 2'],
            hf['Incorrect Answer 3'],
            hf['Correct Answer'],
        ]
        shuffle_rng = random.Random(42 + idx)
        indices = [0, 1, 2, 3]
        shuffle_rng.shuffle(indices)
        letters = ['A', 'B', 'C', 'D']
        choices = [(letters[i], answers[idx_])
                   for i, idx_ in enumerate(indices)]

        prompt = f"""{hf['Question']}

Choices:
A) {choices[0][1]}
B) {choices[1][1]}
C) {choices[2][1]}
D) {choices[3][1]}

Analyze this question carefully and select the best answer. State your answer as A, B, C, or D."""

        result = call_opus(prompt, system, max_tokens)
        return {
            "problem_idx": idx,
            "difficulty": difficulty,
            "n_tokens": result["n_tokens"],
            "n_chars": result["n_chars"],
            "time": result["time"],
            "error": result["error"],
        }

    tasks = []
    for d in difficulties:
        for problem in sampled[d]:
            tasks.append((problem, d))

    if parallel > 1:
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = {
                pool.submit(process_one, p, d): (p, d)
                for p, d in tasks
            }
            for future in as_completed(futures):
                r = future.result()
                all_results.append(r)
                done += 1
                print(
                    f"\r  [{done}/{total}] {r['difficulty']} idx={r['problem_idx']} -> {r['n_tokens']} tokens",
                    end="",
                    flush=True)
    else:
        for problem, d in tasks:
            r = process_one(problem, d)
            all_results.append(r)
            done += 1
            print(
                f"\r  [{done}/{total}] {r['difficulty']} idx={r['problem_idx']} -> {r['n_tokens']} tokens",
                end="",
                flush=True)

    print()
    return {"dataset": "gpqa_mc", "results": all_results}


def run_aime(difficulties: List[str], max_per_difficulty: int,
             parallel: int) -> Dict:
    """AIME needs dataset reload to build prompts from problem text."""
    baseline_path = BASELINE_DIR / "aime_v4.5.json"
    if not baseline_path.exists():
        print(f"  SKIP: {baseline_path} not found")
        return {
            "dataset": "aime",
            "error": "baseline not found",
            "results": []
        }

    with open(baseline_path) as f:
        baseline_data = json.load(f)

    by_diff = {d: [] for d in difficulties}
    for r in baseline_data["results"]:
        d = r["difficulty"]
        if d in by_diff:
            by_diff[d].append(r)

    rng = random.Random(42)
    sampled = {}
    for d in difficulties:
        pool = by_diff[d]
        n = min(max_per_difficulty, len(pool))
        sampled[d] = rng.sample(pool, n) if n > 0 else []

    total = sum(len(v) for v in sampled.values())
    print(f"  Sampled {total} problems: " + ", ".join(f"{d}={len(sampled[d])}"
                                                      for d in difficulties))

    # AIME baseline stores 'problem' field with full text
    system = "You are an expert competition mathematician. Solve problems carefully and show your work."
    max_tokens = 2048

    all_results = []
    done = 0

    def process_one(problem_rec: Dict, difficulty: str):
        # AIME baseline uses 'problem' key (not 'question')
        problem_text = problem_rec.get("problem",
                                       problem_rec.get("question", ""))
        prompt = build_aime_prompt(problem_text)
        result = call_opus(prompt, system, max_tokens)
        return {
            "problem_idx": problem_rec["problem_idx"],
            "difficulty": difficulty,
            "n_tokens": result["n_tokens"],
            "n_chars": result["n_chars"],
            "time": result["time"],
            "error": result["error"],
        }

    tasks = []
    for d in difficulties:
        for problem in sampled[d]:
            tasks.append((problem, d))

    if parallel > 1:
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = {
                pool.submit(process_one, p, d): (p, d)
                for p, d in tasks
            }
            for future in as_completed(futures):
                r = future.result()
                all_results.append(r)
                done += 1
                print(
                    f"\r  [{done}/{total}] {r['difficulty']} idx={r['problem_idx']} -> {r['n_tokens']} tokens",
                    end="",
                    flush=True)
    else:
        for problem, d in tasks:
            r = process_one(problem, d)
            all_results.append(r)
            done += 1
            print(
                f"\r  [{done}/{total}] {r['difficulty']} idx={r['problem_idx']} -> {r['n_tokens']} tokens",
                end="",
                flush=True)

    print()
    return {"dataset": "aime", "results": all_results}


def run_hle(difficulties: List[str], max_per_difficulty: int,
            parallel: int) -> Dict:
    """HLE needs dataset reload for answer_type to build prompts."""
    baseline_path = BASELINE_DIR / "hle_v4.5.json"
    if not baseline_path.exists():
        print(f"  SKIP: {baseline_path} not found")
        return {"dataset": "hle", "error": "baseline not found", "results": []}

    with open(baseline_path) as f:
        baseline_data = json.load(f)

    # HLE baseline uses 'ds_idx' to map back to HuggingFace
    by_diff = {d: [] for d in difficulties}
    for r in baseline_data["results"]:
        d = r["difficulty"]
        if d in by_diff:
            by_diff[d].append(r)

    rng = random.Random(42)
    sampled = {}
    for d in difficulties:
        pool = by_diff[d]
        n = min(max_per_difficulty, len(pool))
        sampled[d] = rng.sample(pool, n) if n > 0 else []

    total = sum(len(v) for v in sampled.values())
    print(f"  Sampled {total} problems: " + ", ".join(f"{d}={len(sampled[d])}"
                                                      for d in difficulties))

    # Load HF dataset for answer_type
    ds, _ = load_hle_dataset(text_only=True)
    system = "You are a highly knowledgeable assistant. Answer questions carefully and precisely."
    max_tokens = 2048

    all_results = []
    done = 0

    def process_one(problem_rec: Dict, difficulty: str):
        ds_idx = problem_rec.get("ds_idx", problem_rec["problem_idx"])
        hle_problem = get_hle_problem(ds, ds_idx)
        prompt = build_hle_prompt(hle_problem["question"],
                                  hle_problem["answer_type"])
        result = call_opus(prompt, system, max_tokens)
        return {
            "problem_idx": problem_rec["problem_idx"],
            "ds_idx": ds_idx,
            "difficulty": difficulty,
            "n_tokens": result["n_tokens"],
            "n_chars": result["n_chars"],
            "time": result["time"],
            "error": result["error"],
        }

    tasks = []
    for d in difficulties:
        for problem in sampled[d]:
            tasks.append((problem, d))

    if parallel > 1:
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = {
                pool.submit(process_one, p, d): (p, d)
                for p, d in tasks
            }
            for future in as_completed(futures):
                r = future.result()
                all_results.append(r)
                done += 1
                print(
                    f"\r  [{done}/{total}] {r['difficulty']} idx={r['problem_idx']} -> {r['n_tokens']} tokens",
                    end="",
                    flush=True)
    else:
        for problem, d in tasks:
            r = process_one(problem, d)
            all_results.append(r)
            done += 1
            print(
                f"\r  [{done}/{total}] {r['difficulty']} idx={r['problem_idx']} -> {r['n_tokens']} tokens",
                end="",
                flush=True)

    print()
    return {"dataset": "hle", "results": all_results}


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def print_summary_table(all_datasets: List[Dict]):
    """Print a summary table of token counts by dataset and difficulty."""
    import statistics

    BITS_PER_TOKEN = 16.61  # log2(100277)

    print("\n" + "=" * 90)
    print(
        f"{'Dataset':<20} {'Difficulty':<12} {'N':>4} {'Mean Tok':>10} {'Median Tok':>11} {'Std Tok':>9} {'Mean Bits':>10}"
    )
    print("-" * 90)

    for ds_data in all_datasets:
        name = ds_data["dataset"]
        if ds_data.get("error"):
            print(f"{name:<20} ERROR: {ds_data['error']}")
            continue

        results = [r for r in ds_data["results"] if r["error"] is None]
        if not results:
            print(f"{name:<20} No valid results")
            continue

        # Group by difficulty
        by_diff = {}
        for r in results:
            d = r["difficulty"]
            by_diff.setdefault(d, []).append(r["n_tokens"])

        for d in ["medium", "hard", "very_hard"]:
            if d not in by_diff:
                continue
            tokens = by_diff[d]
            mean_t = statistics.mean(tokens)
            median_t = statistics.median(tokens)
            std_t = statistics.stdev(tokens) if len(tokens) > 1 else 0
            mean_bits = mean_t * BITS_PER_TOKEN
            print(
                f"{name:<20} {d:<12} {len(tokens):>4} {mean_t:>10.1f} {median_t:>11.1f} {std_t:>9.1f} {mean_bits:>10.0f}"
            )

        # All combined
        all_tokens = [r["n_tokens"] for r in results]
        mean_t = statistics.mean(all_tokens)
        median_t = statistics.median(all_tokens)
        std_t = statistics.stdev(all_tokens) if len(all_tokens) > 1 else 0
        mean_bits = mean_t * BITS_PER_TOKEN
        print(
            f"{name:<20} {'ALL':<12} {len(all_tokens):>4} {mean_t:>10.1f} {median_t:>11.1f} {std_t:>9.1f} {mean_bits:>10.0f}"
        )
        print()

    print("=" * 90)
    print(f"Bits per token = log2(100,277) = {BITS_PER_TOKEN:.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DATASET_CONFIGS = {
    "gsm8k": ("gsm8k_v4.5.json", _gsm8k_prompt),
    "math_algebra": ("math_algebra_v4.5.json", _math_prompt),
    "math_geometry": ("math_geometry_v4.5.json", _math_prompt),
    "math_number_theory": ("math_number_theory_v4.5.json", _math_prompt),
    "mbpp": ("mbpp_v4.5.json", _mbpp_prompt),
    # These need special handling:
    "gpqa_mc": None,
    "aime": None,
    "hle": None,
}


def main():
    parser = argparse.ArgumentParser(
        description="Measure Opus response lengths by difficulty")
    parser.add_argument(
        "--max-per-difficulty",
        type=int,
        default=30,
        help="Max problems to sample per difficulty level (default: 30)")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help=
        "Run only this dataset (e.g., gsm8k, math_algebra, gpqa_mc, aime, hle)"
    )
    parser.add_argument("--parallel",
                        type=int,
                        default=1,
                        help="Number of parallel API calls (default: 1)")
    parser.add_argument("--resume",
                        action="store_true",
                        help="Resume from partial output file")
    parser.add_argument("--output-dir",
                        type=str,
                        default="results/opus_response_lengths",
                        help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"opus_response_lengths_{timestamp}.json"

    difficulties = ["medium", "hard", "very_hard"]

    # Determine which datasets to run
    if args.dataset:
        datasets_to_run = [args.dataset]
    else:
        datasets_to_run = list(DATASET_CONFIGS.keys())

    # Check for resume
    completed_datasets = set()
    all_datasets = []
    if args.resume:
        # Find most recent output file
        existing = sorted(output_dir.glob("opus_response_lengths_*.json"))
        if existing:
            latest = existing[-1]
            print(f"Resuming from {latest}")
            with open(latest) as f:
                prev = json.load(f)
            all_datasets = prev.get("datasets", [])
            completed_datasets = {
                d["dataset"]
                for d in all_datasets if not d.get("error")
            }
            output_path = latest  # Overwrite the same file
            print(f"  Already completed: {completed_datasets}")

    print(f"Model: {OPUS_MODEL}")
    print(f"Max per difficulty: {args.max_per_difficulty}")
    print(f"Parallel workers: {args.parallel}")
    print(f"Output: {output_path}")
    print()

    for ds_name in datasets_to_run:
        if ds_name in completed_datasets:
            print(f"[{ds_name}] Already completed, skipping")
            continue

        print(f"[{ds_name}] Running...")

        if ds_name == "gpqa_mc":
            result = run_gpqa_mc(difficulties, args.max_per_difficulty,
                                 args.parallel)
        elif ds_name == "aime":
            result = run_aime(difficulties, args.max_per_difficulty,
                              args.parallel)
        elif ds_name == "hle":
            result = run_hle(difficulties, args.max_per_difficulty,
                             args.parallel)
        elif ds_name in DATASET_CONFIGS and DATASET_CONFIGS[
                ds_name] is not None:
            baseline_file, prompt_fn = DATASET_CONFIGS[ds_name]
            result = run_dataset_simple(ds_name, baseline_file, prompt_fn,
                                        difficulties, args.max_per_difficulty,
                                        args.parallel)
        else:
            print(f"  Unknown dataset: {ds_name}")
            continue

        all_datasets.append(result)

        # Save after each dataset
        out = {
            "metadata": {
                "model": OPUS_MODEL,
                "max_per_difficulty": args.max_per_difficulty,
                "difficulties": difficulties,
                "timestamp": timestamp,
                "encoding": "cl100k_base",
            },
            "datasets": all_datasets,
        }
        with open(output_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  Saved to {output_path}")

    # Print summary
    print_summary_table(all_datasets)


if __name__ == "__main__":
    main()
