#!/usr/bin/env python3
"""
Best-of-N Experiment using Mega-Batch approach for speed.

Instead of submitting per-problem batches, this submits ALL requests at once
and processes results in bulk. Much faster than the unified approach.

Phases:
1. Submit ALL temperature sampling requests (all problems × all N × all trials)
2. Submit ALL single-prompt requests
3. Submit ALL just-ask initial solutions
4. Submit ALL just-ask rewrites (after initial solutions complete)
5. Process all results locally (compression)

Usage:
    python run_best_of_n_megabatch.py --num-problems 90 --num-trials 3
    python run_best_of_n_megabatch.py --num-problems 10 --num-trials 1  # Quick test
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))

import argparse
import json
import time
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np
from datasets import load_dataset
import anthropic

from lossy_compression_tools import load_compression_model, compress_text
from utils.api_cost_tracker import log_batch_spending
from utils.llm_api import get_anthropic_key

# Defaults
DEFAULT_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_COMPRESSION_MODEL = "meta-llama/Llama-3.1-8B"
DEFAULT_N_VALUES = [1, 3, 5, 10]
DEFAULT_NUM_PROBLEMS = 90
DEFAULT_NUM_TRIALS = 3

MATH_SYSTEM = """You are a skilled mathematician solving AIME problems.
Provide a clear, step-by-step solution and end with the numerical answer.
Format your final answer using \\boxed{answer} notation.
AIME answers are always integers from 0 to 999."""

COMPRESS_SYSTEM = """You are an expert at condensing mathematical solutions to their essential core while preserving the reasoning needed to derive the answer."""


def extract_answer(text: str) -> Optional[str]:
    """Extract numerical answer from response."""
    if not text:
        return None
    # Look for boxed
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    if match:
        return match.group(1).strip()
    # Look for "answer is X"
    match = re.search(r'answer\s+is:?\s*(\d+)', text.lower())
    if match:
        return match.group(1)
    # Last number
    nums = re.findall(r'\b(\d+)\b', text)
    return nums[-1] if nums else None


def strip_answer(text: str) -> str:
    """Replace boxed answer with ???"""
    return re.sub(r'\\boxed\{[^}]+\}', '\\boxed{???}', text)


def submit_mega_batch(client: anthropic.Anthropic, requests: List[Dict],
                      description: str) -> Dict[str, str]:
    """Submit a mega-batch and wait for all results."""
    if not requests:
        return {}

    print(f"\n📦 Submitting {len(requests)} requests for {description}...",
          flush=True)

    batch = client.messages.batches.create(requests=requests)
    batch_id = batch.id
    print(f"   Batch ID: {batch_id}", flush=True)

    # Poll for completion
    start_time = time.time()
    last_print = 0
    while True:
        status = client.messages.batches.retrieve(batch_id)
        elapsed = time.time() - start_time

        if elapsed - last_print > 30:  # Print every 30 seconds
            counts = status.request_counts
            print(
                f"   [{elapsed/60:.1f}m] Processing: {counts.processing}, Succeeded: {counts.succeeded}, Errored: {counts.errored}",
                flush=True)
            last_print = elapsed

        if status.processing_status == 'ended':
            counts = status.request_counts
            print(
                f"   ✅ Complete in {elapsed/60:.1f}m: {counts.succeeded} succeeded, {counts.errored} errors",
                flush=True)
            break

        time.sleep(10)

    # Retrieve results
    all_results = list(client.messages.batches.results(batch_id))

    # Log spending
    model = requests[0]['params'].get('model',
                                      'unknown') if requests else 'unknown'
    log_batch_spending(model, all_results, description)

    # Extract responses
    results = {}
    for result in all_results:
        if result.result.type == 'succeeded':
            results[result.custom_id] = result.result.message.content[0].text
        else:
            results[result.custom_id] = None

    return results


def run_megabatch_experiment(
    num_problems: int = DEFAULT_NUM_PROBLEMS,
    n_values: List[int] = None,
    num_trials: int = DEFAULT_NUM_TRIALS,
    model: str = DEFAULT_MODEL,
    compression_model_name: str = DEFAULT_COMPRESSION_MODEL,
    output_dir: str = None,
    skip_compression: bool = False,
):
    """Run Best-of-N experiment using mega-batches."""

    if n_values is None:
        n_values = DEFAULT_N_VALUES

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if output_dir is None:
        output_dir = f"results/best_of_n_megabatch/run_{timestamp}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    client = anthropic.Anthropic(api_key=get_anthropic_key())

    # Load compression model (unless skipping)
    compression_model, tokenizer = None, None
    if not skip_compression:
        print(f"Loading compression model: {compression_model_name}",
              flush=True)
        compression_model, tokenizer = load_compression_model(
            compression_model_name)

    # Load problems
    print("Loading AIME dataset...", flush=True)
    ds = load_dataset("AI-MO/aimo-validation-aime")
    problems = list(ds['train'])[:num_problems]
    print(f"Loaded {len(problems)} problems")

    # =========================================================================
    # PHASE 1: Submit ALL temperature sampling requests
    # =========================================================================
    print("\n" + "=" * 60)
    print("PHASE 1: Temperature Sampling")
    print("=" * 60)

    temp_requests = []
    for trial in range(num_trials):
        for n in n_values:
            for prob_idx, problem in enumerate(problems):
                for sample_idx in range(n):
                    custom_id = f"temp_t{trial}_n{n}_p{prob_idx}_s{sample_idx}"
                    temp_requests.append({
                        "custom_id": custom_id,
                        "params": {
                            "model":
                            model,
                            "max_tokens":
                            2000,
                            "temperature":
                            0.8,
                            "system":
                            MATH_SYSTEM,
                            "messages": [{
                                "role":
                                "user",
                                "content":
                                f"""Solve this AIME problem:

{problem['problem']}

Provide a complete solution with clear reasoning. End with \\boxed{{answer}}."""
                            }],
                        }
                    })

    temp_results = submit_mega_batch(client, temp_requests,
                                     "temperature sampling")

    # =========================================================================
    # PHASE 2: Submit ALL single-prompt requests
    # =========================================================================
    print("\n" + "=" * 60)
    print("PHASE 2: Single Prompt")
    print("=" * 60)

    single_requests = []
    for trial in range(num_trials):
        for n in n_values:
            for prob_idx, problem in enumerate(problems):
                custom_id = f"single_t{trial}_n{n}_p{prob_idx}"
                single_requests.append({
                    "custom_id": custom_id,
                    "params": {
                        "model":
                        model,
                        "max_tokens":
                        4000,
                        "temperature":
                        0.3,
                        "system":
                        MATH_SYSTEM,
                        "messages": [{
                            "role":
                            "user",
                            "content":
                            f"""Solve this AIME problem in {n} DIFFERENT ways.

Problem: {problem['problem']}

Provide {n} complete solutions, each labeled "Solution 1:", "Solution 2:", etc.
Each solution should end with \\boxed{{answer}}. Use genuinely different approaches."""
                        }],
                    }
                })

    single_results = submit_mega_batch(client, single_requests,
                                       "single prompt")

    # =========================================================================
    # PHASE 3: Submit ALL just-ask initial solutions
    # =========================================================================
    print("\n" + "=" * 60)
    print("PHASE 3: Just-Ask Initial Solutions")
    print("=" * 60)

    # Only need one initial solution per problem per trial (not per N)
    initial_requests = []
    for trial in range(num_trials):
        for prob_idx, problem in enumerate(problems):
            custom_id = f"initial_t{trial}_p{prob_idx}"
            initial_requests.append({
                "custom_id": custom_id,
                "params": {
                    "model":
                    model,
                    "max_tokens":
                    2000,
                    "temperature":
                    0.3,
                    "system":
                    MATH_SYSTEM,
                    "messages": [{
                        "role":
                        "user",
                        "content":
                        f"""Solve this AIME problem:

{problem['problem']}

Provide a complete solution with clear reasoning. End with \\boxed{{answer}}."""
                    }],
                }
            })

    initial_results = submit_mega_batch(client, initial_requests,
                                        "just-ask initial")

    # =========================================================================
    # PHASE 4: Submit ALL just-ask rewrites
    # =========================================================================
    print("\n" + "=" * 60)
    print("PHASE 4: Just-Ask Rewrites")
    print("=" * 60)

    rewrite_requests = []
    for trial in range(num_trials):
        for n in n_values:
            for prob_idx, problem in enumerate(problems):
                initial_id = f"initial_t{trial}_p{prob_idx}"
                initial_response = initial_results.get(initial_id, "")

                if not initial_response:
                    continue

                stripped = strip_answer(initial_response)

                for rewrite_idx in range(n):
                    custom_id = f"rewrite_t{trial}_n{n}_p{prob_idx}_r{rewrite_idx}"
                    rewrite_requests.append({
                        "custom_id": custom_id,
                        "params": {
                            "model":
                            model,
                            "max_tokens":
                            1000,
                            "temperature":
                            0.7,
                            "system":
                            COMPRESS_SYSTEM,
                            "messages": [{
                                "role":
                                "user",
                                "content":
                                f"""Here is a solution to an AIME problem with the answer hidden:

{stripped}

Rewrite this solution as SUCCINCTLY as possible while preserving enough reasoning to derive the final answer. Be brief but complete."""
                            }],
                        }
                    })

    rewrite_results = submit_mega_batch(client, rewrite_requests,
                                        "just-ask rewrites")

    # =========================================================================
    # PHASE 5: Process all results (compression + accuracy)
    # =========================================================================
    print("\n" + "=" * 60)
    print("PHASE 5: Processing Results")
    print("=" * 60)

    def compress_if_enabled(text):
        if skip_compression or not text:
            return {'compression_pct': 0, 'bits_per_token': 0}
        try:
            # compress_text returns (compressed_data, compression_ratio, metrics)
            _, compression_ratio, metrics = compress_text(
                text, compression_model, tokenizer)
            return {
                'compression_pct': compression_ratio * 100,
                'bits_per_token': metrics.get('bits_per_token', 0),
            }
        except Exception as e:
            print(f"    Compression error: {e}")
            return {'compression_pct': 0, 'bits_per_token': 0}

    all_data = {
        'parameters': {
            'num_problems': num_problems,
            'n_values': n_values,
            'num_trials': num_trials,
            'model': model,
            'timestamp': timestamp,
            'skip_compression': skip_compression,
        },
        'trials': []
    }

    for trial in range(num_trials):
        print(f"\nProcessing trial {trial + 1}/{num_trials}...")

        trial_data = {
            'trial': trial + 1,
            'temperature': {
                'by_n': {
                    n: []
                    for n in n_values
                }
            },
            'single_prompt': {
                'by_n': {
                    n: []
                    for n in n_values
                }
            },
            'just_ask': {
                'by_n': {
                    n: []
                    for n in n_values
                }
            },
        }

        for prob_idx, problem in enumerate(problems):
            correct_answer = str(problem['answer'])

            for n in n_values:
                # Process temperature samples
                samples = []
                for s in range(n):
                    resp = temp_results.get(
                        f"temp_t{trial}_n{n}_p{prob_idx}_s{s}", "")
                    if resp:
                        comp = compress_if_enabled(resp)
                        samples.append({
                            'answer':
                            extract_answer(resp),
                            'compression_pct':
                            comp['compression_pct'],
                        })

                # Select best by compression
                valid = [s for s in samples if s['compression_pct'] > 0]
                if valid:
                    best = min(valid, key=lambda x: x['compression_pct'])
                else:
                    best = samples[0] if samples else {
                        'answer': None,
                        'compression_pct': 0
                    }

                trial_data['temperature']['by_n'][n].append({
                    'problem_idx':
                    prob_idx,
                    'compression_pct':
                    best['compression_pct'],
                    'is_correct':
                    str(best.get('answer')) == correct_answer,
                })

                # Process single prompt
                single_resp = single_results.get(
                    f"single_t{trial}_n{n}_p{prob_idx}", "")
                if single_resp:
                    # Split into solutions
                    parts = re.split(r'Solution\s+\d+:',
                                     single_resp,
                                     flags=re.IGNORECASE)
                    solutions = [p.strip() for p in parts[1:] if p.strip()]
                    if not solutions:
                        solutions = [single_resp]

                    sol_data = []
                    for sol in solutions:
                        comp = compress_if_enabled(sol)
                        sol_data.append({
                            'answer':
                            extract_answer(sol),
                            'compression_pct':
                            comp['compression_pct'],
                        })

                    valid = [s for s in sol_data if s['compression_pct'] > 0]
                    best = min(valid, key=lambda x: x['compression_pct']
                               ) if valid else (sol_data[0] if sol_data else {
                                   'answer': None,
                                   'compression_pct': 0
                               })
                else:
                    best = {'answer': None, 'compression_pct': 0}

                trial_data['single_prompt']['by_n'][n].append({
                    'problem_idx':
                    prob_idx,
                    'compression_pct':
                    best['compression_pct'],
                    'is_correct':
                    str(best.get('answer')) == correct_answer,
                })

                # Process just-ask
                initial_resp = initial_results.get(
                    f"initial_t{trial}_p{prob_idx}", "")
                initial_answer = extract_answer(
                    initial_resp) if initial_resp else None
                initial_comp = compress_if_enabled(initial_resp)

                rewrites = []
                for r in range(n):
                    rw = rewrite_results.get(
                        f"rewrite_t{trial}_n{n}_p{prob_idx}_r{r}", "")
                    if rw:
                        comp = compress_if_enabled(rw)
                        rewrites.append(
                            {'compression_pct': comp['compression_pct']})

                valid = [r for r in rewrites if r['compression_pct'] > 0]
                best_rewrite = min(
                    valid, key=lambda x: x['compression_pct']) if valid else (
                        rewrites[0] if rewrites else {
                            'compression_pct': 0
                        })

                trial_data['just_ask']['by_n'][n].append({
                    'problem_idx':
                    prob_idx,
                    'compression_pct':
                    best_rewrite['compression_pct'],
                    'verbose_compression_pct':
                    initial_comp['compression_pct'],
                    'is_correct':
                    str(initial_answer) == correct_answer,
                })

        all_data['trials'].append(trial_data)

        # Save trial
        trial_path = Path(output_dir) / f"trial_{trial + 1}.json"
        with open(trial_path, 'w') as f:
            json.dump(trial_data, f, indent=2)

    # Compute summary
    all_data['summary'] = compute_summary(all_data)

    # Save final
    final_path = Path(output_dir) / "all_results.json"
    with open(final_path, 'w') as f:
        json.dump(all_data, f, indent=2)
    print(f"\n✅ Saved results to: {final_path}")

    # Print summary
    print_summary(all_data)

    return all_data


def compute_summary(data: Dict) -> Dict:
    """Compute summary statistics."""
    n_values = data['parameters']['n_values']
    summary = {}

    for approach in ['temperature', 'single_prompt', 'just_ask']:
        summary[approach] = {}
        for n in n_values:
            comp_means = []
            acc_means = []

            for trial in data['trials']:
                probs = trial[approach]['by_n'][n]
                comps = [
                    p['compression_pct'] for p in probs
                    if p.get('compression_pct', 0) > 0
                ]
                accs = [1 if p['is_correct'] else 0 for p in probs]

                if comps:
                    comp_means.append(np.mean(comps))
                if accs:
                    acc_means.append(np.mean(accs) * 100)

            summary[approach][n] = {
                'compression_pct_mean':
                float(np.mean(comp_means)) if comp_means else 0,
                'compression_pct_std':
                float(np.std(comp_means)) if len(comp_means) > 1 else 0,
                'accuracy_mean':
                float(np.mean(acc_means)) if acc_means else 0,
                'accuracy_std':
                float(np.std(acc_means)) if len(acc_means) > 1 else 0,
            }

            if approach == 'just_ask':
                verbose_means = []
                for trial in data['trials']:
                    probs = trial[approach]['by_n'][n]
                    verbose = [
                        p.get('verbose_compression_pct', 0) for p in probs
                        if p.get('verbose_compression_pct', 0) > 0
                    ]
                    if verbose:
                        verbose_means.append(np.mean(verbose))
                summary[approach][n]['verbose_compression_pct_mean'] = float(
                    np.mean(verbose_means)) if verbose_means else 0

    return summary


def print_summary(data: Dict):
    """Print summary table."""
    summary = data['summary']
    n_values = data['parameters']['n_values']

    print("\n" + "=" * 80)
    print("COMPRESSION % (lower = better)")
    print("=" * 80)
    print(
        f"{'N':>3} | {'Temperature':>18} | {'Single Prompt':>18} | {'Just Ask':>18}"
    )
    print("-" * 80)

    for n in n_values:
        t = summary['temperature'].get(n, {})
        s = summary['single_prompt'].get(n, {})
        j = summary['just_ask'].get(n, {})

        print(
            f"{n:>3} | {t.get('compression_pct_mean',0):>6.2f} ± {t.get('compression_pct_std',0):>5.2f} | "
            f"{s.get('compression_pct_mean',0):>6.2f} ± {s.get('compression_pct_std',0):>5.2f} | "
            f"{j.get('compression_pct_mean',0):>6.2f} ± {j.get('compression_pct_std',0):>5.2f}"
        )

    print("\n" + "=" * 80)
    print("ACCURACY %")
    print("=" * 80)
    print(
        f"{'N':>3} | {'Temperature':>18} | {'Single Prompt':>18} | {'Just Ask':>18}"
    )
    print("-" * 80)

    for n in n_values:
        t = summary['temperature'].get(n, {})
        s = summary['single_prompt'].get(n, {})
        j = summary['just_ask'].get(n, {})

        print(
            f"{n:>3} | {t.get('accuracy_mean',0):>6.1f} ± {t.get('accuracy_std',0):>5.1f} | "
            f"{s.get('accuracy_mean',0):>6.1f} ± {s.get('accuracy_std',0):>5.1f} | "
            f"{j.get('accuracy_mean',0):>6.1f} ± {j.get('accuracy_std',0):>5.1f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description='Best-of-N Megabatch Experiment')
    parser.add_argument('--num-problems',
                        type=int,
                        default=DEFAULT_NUM_PROBLEMS)
    parser.add_argument('--num-trials', type=int, default=DEFAULT_NUM_TRIALS)
    parser.add_argument('--n-values',
                        type=int,
                        nargs='+',
                        default=DEFAULT_N_VALUES)
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL)
    parser.add_argument('--compression-model',
                        type=str,
                        default=DEFAULT_COMPRESSION_MODEL)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--skip-compression',
                        action='store_true',
                        help='Skip compression step (for faster testing)')

    args = parser.parse_args()

    run_megabatch_experiment(
        num_problems=args.num_problems,
        n_values=args.n_values,
        num_trials=args.num_trials,
        model=args.model,
        compression_model_name=args.compression_model,
        output_dir=args.output_dir,
        skip_compression=args.skip_compression,
    )


if __name__ == "__main__":
    main()
