#!/usr/bin/env python3
"""
Comprehensive baseline evaluation script for ALL datasets.

This script runs haiku, sonnet, and opus on each dataset to classify problem difficulty:
- easy: all models pass
- medium: haiku fails, sonnet/opus pass
- hard: haiku/sonnet fail, opus passes
- very_hard: all models fail

Uses consistent evaluation logic across all datasets with proper retry/resume support.

Usage:
    python scripts/run_all_baselines.py
    python scripts/run_all_baselines.py --dataset gsm8k
    python scripts/run_all_baselines.py --dataset gpqa --format mc
    python scripts/run_all_baselines.py --resume

Datasets:
    - gsm8k: Grade school math (1319 problems)
    - math: Competition math (algebra, geometry, number_theory)
    - gpqa: Graduate science QA (mc=multiple choice, freeform=open ended)
    - mbpp: Python programming (257 problems, sanitized split)
    - humaneval: Python programming (164 problems)
    - mmlu_pro: MMLU Pro freeform (493 problems, LLM-as-judge evaluation)
    - hle: Humanity's Last Exam (2158 text-only problems, extremely difficult)
    - aime: AIME competition math (90 problems, integer answers 0-999)
"""

import json
import time
import argparse
import re
import os
import sys
import random
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))

import anthropic
from datasets import load_dataset
from lossy_compression import MODEL_ALIAS_MAP, MODEL_ALIAS_MAP_old, model_completion
from lossy_compression.judge import judge_freeform_answer
from lossy_compression.benchmarks.hle import (load_hle_dataset,
                                              get_hle_problem,
                                              build_hle_prompt,
                                              extract_hle_answer,
                                              check_hle_answer)
from lossy_compression.benchmarks.aime import (load_aime_dataset,
                                               get_aime_problem,
                                               build_aime_prompt,
                                               extract_aime_answer,
                                               check_aime_answer)

# Active model map (can be switched via --use-old-models flag)
ACTIVE_MODEL_MAP = MODEL_ALIAS_MAP  # Default to 4.5 models
MODEL_VERSION = "v4.5"  # Version suffix for output files
from utils.llm_api import get_anthropic_key

# Models to evaluate (gpt-oss added via --include-gpt-oss flag)
MODELS_CLAUDE = ['haiku', 'sonnet', 'opus']
MODELS_GPT_OSS = ['gpt-oss']
MODELS = MODELS_CLAUDE  # Default, updated in main() if --include-gpt-oss

# Rate limit handling
RATE_LIMIT_ERRORS = [
    'rate_limit', 'rate limit', 'too many requests', '429', 'overloaded',
    'capacity', 'throttl'
]
RATE_LIMIT_BACKOFFS = [30, 60, 120, 240, 480, 480, 480, 480, 480, 480]

# Save progress every N problems
SAVE_EVERY = 10

# Model name abbreviations for progress display
MODEL_ABBREV = {
    'haiku': 'H',
    'sonnet': 'S',
    'opus': 'O',
    'gpt-oss': 'G',
}

# Batch API settings
BATCH_POLL_INTERVAL = 30  # seconds between polls
BATCH_MAX_RETRIES = 10
BATCH_INITIAL_BACKOFF = 30
BATCH_MAX_BACKOFF = 480

# Global flag for batch mode
USE_BATCH = False

# Global flag for parallel workers in iterative mode
PARALLEL_WORKERS = 1  # Default to sequential, can be set via --parallel

# Global temperature (0.0 = deterministic, >0 = stochastic for variance runs)
TEMPERATURE = 0.0

# =============================================================================
# Batch API helpers
# =============================================================================


def batch_retry_with_backoff(func, *args, **kwargs):
    """Retry a function with exponential backoff for batch API."""
    backoff = BATCH_INITIAL_BACKOFF
    last_error = None

    for attempt in range(BATCH_MAX_RETRIES):
        try:
            return func(*args, **kwargs)
        except anthropic.RateLimitError as e:
            last_error = e
            wait_time = min(backoff * (2**attempt), BATCH_MAX_BACKOFF)
            print(
                f"\n  Rate limited, waiting {wait_time}s (attempt {attempt + 1}/{BATCH_MAX_RETRIES})..."
            )
            time.sleep(wait_time)
        except anthropic.APIConnectionError as e:
            last_error = e
            wait_time = min(backoff * (2**attempt), BATCH_MAX_BACKOFF)
            print(
                f"\n  Connection error, waiting {wait_time}s (attempt {attempt + 1}/{BATCH_MAX_RETRIES})..."
            )
            time.sleep(wait_time)
        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                last_error = e
                wait_time = min(backoff * (2**attempt), BATCH_MAX_BACKOFF)
                print(
                    f"\n  Server error {e.status_code}, waiting {wait_time}s (attempt {attempt + 1}/{BATCH_MAX_RETRIES})..."
                )
                time.sleep(wait_time)
            else:
                raise

    raise last_error


def submit_batch(client: anthropic.Anthropic, requests: List[Dict]) -> str:
    """Submit batch to Anthropic API and return batch ID."""

    def _submit():
        batch = client.messages.batches.create(requests=requests)
        return batch.id

    return batch_retry_with_backoff(_submit)


def poll_batch(client: anthropic.Anthropic, batch_id: str) -> Dict:
    """Poll batch status until complete."""
    consecutive_errors = 0
    max_consecutive_errors = 10

    while True:
        try:
            batch = client.messages.batches.retrieve(batch_id)
            consecutive_errors = 0

            status = batch.processing_status
            counts = batch.request_counts

            total_reqs = counts.processing + counts.succeeded + counts.errored
            print(
                f"  [{counts.succeeded + counts.errored}/{total_reqs}] "
                f"processing: {counts.processing}",
                end='\r')

            if status == 'ended':
                print()
                return {
                    'status': status,
                    'succeeded': counts.succeeded,
                    'errored': counts.errored,
                }

        except (anthropic.RateLimitError, anthropic.APIConnectionError,
                anthropic.APIStatusError) as e:
            consecutive_errors += 1
            if consecutive_errors >= max_consecutive_errors:
                raise RuntimeError(
                    f"Too many consecutive errors polling batch: {e}")
            wait_time = min(BATCH_INITIAL_BACKOFF * (2**consecutive_errors),
                            BATCH_MAX_BACKOFF)
            print(
                f"\n  Poll error, waiting {wait_time}s ({consecutive_errors}/{max_consecutive_errors})..."
            )
            time.sleep(wait_time)
            continue

        time.sleep(BATCH_POLL_INTERVAL)


def download_batch_results(client: anthropic.Anthropic,
                           batch_id: str) -> Dict[str, str]:
    """Download and parse batch results."""

    def _download():
        results = {}
        for result in client.messages.batches.results(batch_id):
            if result.result.type == 'succeeded':
                content = result.result.message.content[
                    0].text if result.result.message.content else ""
                results[result.custom_id] = content
            else:
                # Store error info
                results[result.custom_id] = None
        return results

    return batch_retry_with_backoff(_download)


def run_batch_step(client: anthropic.Anthropic, requests: List[Dict],
                   step_name: str) -> Dict[str, str]:
    """Run a batch step: submit, poll, download."""
    if not requests:
        return {}

    print(f"\n  {step_name}: {len(requests)} requests...")

    batch_id = submit_batch(client, requests)
    print(f"  Batch ID: {batch_id}")

    poll_batch(client, batch_id)
    results = download_batch_results(client, batch_id)
    print(f"  Got {len(results)} results")

    return results


def create_batch_request(custom_id: str, model: str, system: str, prompt: str,
                         max_tokens: int) -> Dict:
    """Create a single batch request in Anthropic format."""
    model_full = ACTIVE_MODEL_MAP.get(model.lower(), model)
    return {
        "custom_id": custom_id,
        "params": {
            "model": model_full,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{
                "role": "user",
                "content": prompt
            }],
        }
    }


# =============================================================================
# Parallel execution helpers
# =============================================================================


def run_parallel_evaluations(eval_func,
                             problems: List[Dict],
                             total: int,
                             save_func,
                             save_interval: int = 10) -> List[Dict]:
    """Run evaluations in parallel using ThreadPoolExecutor.

    Args:
        eval_func: Function that takes a problem dict and returns a result dict
        problems: List of problem dicts to evaluate
        total: Total number of problems (for progress display)
        save_func: Function to call for saving partial results
        save_interval: Save every N completed problems

    Returns:
        List of result dicts
    """
    results = []

    if PARALLEL_WORKERS <= 1:
        # Sequential mode
        for i, problem in enumerate(problems):
            result = eval_func(problem)
            results.append(result)

            # Progress
            models_correct = {
                m: result.get('models', {}).get(m, {}).get('correct', False)
                for m in MODELS
            }
            print(format_progress(results, models_correct, len(results),
                                  total),
                  end='',
                  flush=True)

            if len(results) % save_interval == 0:
                save_func(results)
    else:
        # Parallel mode
        completed = 0
        with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
            # Submit all tasks
            future_to_problem = {
                executor.submit(eval_func, p): p
                for p in problems
            }

            for future in as_completed(future_to_problem):
                result = future.result()
                results.append(result)
                completed += 1

                # Progress (simplified for parallel - just show count)
                print(f"\r[{completed}/{total}] completed", end='', flush=True)

                if completed % save_interval == 0:
                    save_func(results)

        # Sort results by problem_idx to maintain order
        results.sort(key=lambda x: x.get('problem_idx', 0))

    return results


def format_progress(results: List[Dict], models_correct: Dict[str, bool],
                    done: int, total: int) -> str:
    """Format progress string dynamically based on active models."""
    status = ' '.join([('✓' if models_correct.get(m, False) else '✗')
                       for m in MODELS])
    acc_parts = []
    for m in MODELS:
        correct_count = sum(
            1 for r in results
            if r.get('models', {}).get(m, {}).get('correct', False))
        acc = correct_count / len(results) * 100 if results else 0
        abbrev = MODEL_ABBREV.get(m, m[0].upper())
        acc_parts.append(f"{abbrev}:{acc:.1f}%")
    return f"\r[{done}/{total}] {status} | {' '.join(acc_parts)}"


class RateLimitError(Exception):
    pass


def check_rate_limit(error_msg: str) -> bool:
    """Check if error indicates rate limiting."""
    return any(term in str(error_msg).lower() for term in RATE_LIMIT_ERRORS)


def call_model_with_retry(model: str,
                          prompt: str,
                          system: str,
                          max_tokens: int = 2000,
                          max_retries: int = 10) -> Dict:
    """Call model with exponential backoff on rate limits."""
    model_full = ACTIVE_MODEL_MAP.get(model.lower(), model)

    for attempt in range(max_retries):
        try:
            start = time.time()
            response = model_completion(
                model=model_full,
                system=system,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=TEMPERATURE,
            )
            return {
                'response': response,
                'success': True,
                'time': time.time() - start,
                'error': None,
            }
        except Exception as e:
            error_str = str(e)
            if check_rate_limit(error_str):
                wait = RATE_LIMIT_BACKOFFS[min(attempt,
                                               len(RATE_LIMIT_BACKOFFS) - 1)]
                print(
                    f"\n  Rate limited, waiting {wait}s (attempt {attempt + 1}/{max_retries})..."
                )
                time.sleep(wait)
            else:
                return {
                    'response': None,
                    'success': False,
                    'time': 0,
                    'error': error_str,
                }

    return {
        'response': None,
        'success': False,
        'time': 0,
        'error': 'Max retries exceeded'
    }


def find_partial_results(output_dir: Path,
                         dataset_name: str) -> Tuple[Optional[Path], Set[int]]:
    """Find most recent partial results for resuming."""
    # Match both old format (dataset_all_models_*.json) and new format (dataset_v*.json)
    pattern = f"{dataset_name}_*.json"
    files = list(output_dir.glob(pattern))

    if not files:
        return None, set()

    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

    for filepath in files:
        try:
            with open(filepath) as f:
                data = json.load(f)

            if 'results' in data and data['results']:
                # Find completed problem indices (all 3 models evaluated)
                completed = set()
                for r in data['results']:
                    if all(m in r.get('models', {}) for m in MODELS):
                        completed.add(r['problem_idx'])
                print(
                    f"  Found partial: {filepath.name} ({len(completed)} complete)"
                )
                return filepath, completed
        except (json.JSONDecodeError, KeyError):
            continue

    return None, set()


def save_results(output_path: Path,
                 dataset_name: str,
                 results: List[Dict],
                 metadata: Dict,
                 partial: bool = True):
    """Save results to file."""
    # Calculate model accuracies
    model_accuracy = {}
    for model in MODELS:
        correct = sum(
            1 for r in results
            if r.get('models', {}).get(model, {}).get('correct', False))
        model_accuracy[model] = correct / len(results) if results else 0

    # Calculate difficulty distribution
    difficulty_counts = {'easy': 0, 'medium': 0, 'hard': 0, 'very_hard': 0}
    for r in results:
        difficulty_counts[r.get('difficulty', 'very_hard')] += 1

    difficulty_dist = {
        k: v / len(results) if results else 0
        for k, v in difficulty_counts.items()
    }

    # Build model IDs dict showing actual model versions used
    model_ids = {alias: ACTIVE_MODEL_MAP.get(alias, alias) for alias in MODELS}

    data = {
        'metadata': {
            **metadata,
            'model_version': MODEL_VERSION,
            'model_ids': model_ids,
            'partial': partial,
            'last_updated': datetime.now().isoformat(),
        },
        'model_accuracy': model_accuracy,
        'difficulty_distribution': difficulty_dist,
        'difficulty_counts': difficulty_counts,
        'results': results,
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def classify_difficulty(models_correct: Dict[str, bool]) -> str:
    """Classify problem difficulty based on which models got it correct."""
    haiku = models_correct.get('haiku', False)
    sonnet = models_correct.get('sonnet', False)
    opus = models_correct.get('opus', False)

    if haiku and sonnet and opus:
        return 'easy'
    elif not haiku and sonnet and opus:
        return 'medium'
    elif not haiku and not sonnet and opus:
        return 'hard'
    else:
        return 'very_hard'


# ============================================================================
# GSM8K
# ============================================================================


def extract_gsm8k_answer(text: str) -> Optional[str]:
    """Extract numerical answer from GSM8K response."""
    if not text:
        return None
    # Look for #### format
    match = re.search(r'####\s*(\-?\d[\d,]*)', text)
    if match:
        return match.group(1).replace(',', '')
    # Look for boxed
    boxed = re.findall(r'\\boxed\{([^{}]*)\}', text)
    if boxed:
        nums = re.findall(r'\-?\d+', boxed[-1])
        return nums[-1] if nums else None
    # Last number
    nums = re.findall(r'\-?\d[\d,]*', text)
    return nums[-1].replace(',', '') if nums else None


def run_gsm8k(output_dir: Path,
              resume: bool = False,
              num_problems: Optional[int] = None) -> Dict:
    """Run GSM8K evaluation."""
    print(
        f"\n{'='*60}\nRunning GSM8K {'(BATCH)' if USE_BATCH else '(iterative)'}\n{'='*60}"
    )

    ds = load_dataset('openai/gsm8k', 'main')
    problems = list(ds['test'])
    if num_problems:
        problems = problems[:num_problems]

    total = len(problems)
    print(f"Total problems: {total}")

    output_path = output_dir / f"gsm8k_{MODEL_VERSION}.json"

    # Skip if already exists
    if output_path.exists() and not resume:
        print(f"SKIP: {output_path.name} already exists")
        with open(output_path) as f:
            data = json.load(f)
        return data

    # Check for partial results
    completed = set()
    results = []
    if resume:
        partial_path, completed = find_partial_results(output_dir, 'gsm8k')
        if partial_path and completed:
            with open(partial_path) as f:
                data = json.load(f)
            results = data.get('results', [])

    system = "You are a math tutor. Solve problems step by step. End with #### followed by the numerical answer."

    if USE_BATCH:
        # Batch mode: collect all requests, submit in batches per model
        client = anthropic.Anthropic(api_key=get_anthropic_key())

        # Get Claude models only (batch API doesn't support OpenRouter)
        claude_models = [m for m in MODELS if m in MODELS_CLAUDE]
        non_claude_models = [m for m in MODELS if m not in MODELS_CLAUDE]

        # Build problem info for non-completed problems
        pending_problems = []
        for idx, problem in enumerate(problems):
            if idx in completed:
                continue
            question = problem['question']
            gold = extract_gsm8k_answer(problem['answer'])
            prompt = f"Solve this math problem:\n\n{question}\n\nShow your work, then give the final answer after ####"
            pending_problems.append({
                'idx': idx,
                'question': question,
                'gold': gold,
                'prompt': prompt,
            })

        print(f"  Pending problems: {len(pending_problems)}")

        # Collect batch responses per model
        model_responses = {}

        # Submit batch for each Claude model
        for model in claude_models:
            requests = []
            for p in pending_problems:
                custom_id = f"gsm8k_{p['idx']}_{model}"
                requests.append(
                    create_batch_request(custom_id,
                                         model,
                                         system,
                                         p['prompt'],
                                         max_tokens=2000))

            if requests:
                responses = run_batch_step(client, requests,
                                           f"{model.upper()} batch")
                model_responses[model] = responses

        # Run non-Claude models iteratively (e.g., gpt-oss via OpenRouter)
        for model in non_claude_models:
            print(f"\n  Running {model} iteratively (non-Anthropic model)...")
            responses = {}
            for i, p in enumerate(pending_problems):
                result = call_model_with_retry(model, p['prompt'], system)
                responses[f"gsm8k_{p['idx']}_{model}"] = result['response']
                print(f"  [{i+1}/{len(pending_problems)}]", end='\r')
            print()
            model_responses[model] = responses

        # Process all results
        for p in pending_problems:
            models_data = {}
            models_correct = {}

            for model in MODELS:
                custom_id = f"gsm8k_{p['idx']}_{model}"
                response = model_responses.get(model, {}).get(custom_id)
                extracted = extract_gsm8k_answer(
                    response) if response else None
                correct = str(extracted) == str(
                    p['gold']) if extracted and p['gold'] else False

                models_data[model] = {
                    'answer': extracted,
                    'correct': correct,
                    'time': 0,  # Batch doesn't track individual times
                    'error': None if response else 'No response',
                }
                models_correct[model] = correct

            difficulty = classify_difficulty(models_correct)

            results.append({
                'problem_idx': p['idx'],
                'question': p['question'],
                'gold_answer': p['gold'],
                'models': models_data,
                'difficulty': difficulty,
            })

    else:
        # Iterative mode (with optional parallelism)
        def process_problem(idx_problem):
            idx, problem = idx_problem
            question = problem['question']
            gold = extract_gsm8k_answer(problem['answer'])
            prompt = f"Solve this math problem:\n\n{question}\n\nShow your work, then give the final answer after ####"

            models_data = {}
            models_correct = {}

            for model in MODELS:
                result = call_model_with_retry(model, prompt, system)
                extracted = extract_gsm8k_answer(
                    result['response']) if result['response'] else None
                correct = str(extracted) == str(
                    gold) if extracted and gold else False

                models_data[model] = {
                    'answer': extracted,
                    'correct': correct,
                    'time': result['time'],
                    'error': result['error'],
                }
                models_correct[model] = correct

            difficulty = classify_difficulty(models_correct)

            return {
                'problem_idx': idx,
                'question': question,
                'gold_answer': gold,
                'models': models_data,
                'difficulty': difficulty,
                '_models_correct': models_correct,  # For progress display
            }

        # Filter out completed problems
        pending = [(idx, p) for idx, p in enumerate(problems)
                   if idx not in completed]

        if PARALLEL_WORKERS > 1:
            # Parallel execution
            import threading
            lock = threading.Lock()
            completed_count = [0]  # Use list for mutable in closure

            with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
                futures = {
                    executor.submit(process_problem, item): item
                    for item in pending
                }
                for future in as_completed(futures):
                    result = future.result()
                    models_correct = result.pop('_models_correct')
                    with lock:
                        results.append(result)
                        completed_count[0] += 1
                        print(format_progress(
                            results, models_correct,
                            completed_count[0] + len(completed), total),
                              end='',
                              flush=True)
                        if len(results) % SAVE_EVERY == 0:
                            save_results(output_path, 'gsm8k', results, {
                                'dataset': 'gsm8k',
                                'total': total
                            })
        else:
            # Sequential execution
            for item in pending:
                result = process_problem(item)
                models_correct = result.pop('_models_correct')
                results.append(result)
                print(format_progress(results, models_correct,
                                      len(results) + len(completed), total),
                      end='',
                      flush=True)
                if len(results) % SAVE_EVERY == 0:
                    save_results(output_path, 'gsm8k', results, {
                        'dataset': 'gsm8k',
                        'total': total
                    })

    print()
    save_results(output_path,
                 'gsm8k',
                 results, {
                     'dataset': 'gsm8k',
                     'total': total
                 },
                 partial=False)
    print(f"Saved: {output_path}")

    return {'path': str(output_path), 'n_problems': len(results)}


# ============================================================================
# MATH
# ============================================================================


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract answer from \\boxed{...} format."""
    if not text:
        return None
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)
    return matches[-1].strip() if matches else None


def normalize_math_answer(answer: str) -> str:
    """Normalize MATH answer for comparison."""
    if not answer:
        return ''
    # Remove spaces, convert to lowercase
    ans = answer.strip().lower()
    # Remove $ signs
    ans = ans.replace('$', '')
    # Normalize fractions
    ans = re.sub(r'\\frac\{(\d+)\}\{(\d+)\}', r'\1/\2', ans)
    return ans


def run_math(subject: str,
             output_dir: Path,
             resume: bool = False,
             num_problems: Optional[int] = None) -> Dict:
    """Run MATH evaluation for a specific subject."""
    print(
        f"\n{'='*60}\nRunning MATH ({subject}) {'(BATCH)' if USE_BATCH else '(iterative)'}\n{'='*60}"
    )

    ds = load_dataset('EleutherAI/hendrycks_math', subject)
    problems = list(ds['test'])
    if num_problems:
        problems = problems[:num_problems]

    total = len(problems)
    print(f"Total problems: {total}")

    dataset_name = f"math_{subject}"
    output_path = output_dir / f"{dataset_name}_{MODEL_VERSION}.json"

    # Skip if already exists
    if output_path.exists() and not resume:
        print(f"SKIP: {output_path.name} already exists")
        with open(output_path) as f:
            data = json.load(f)
        return data

    # Check for partial results
    completed = set()
    results = []
    if resume:
        partial_path, completed = find_partial_results(output_dir,
                                                       dataset_name)
        if partial_path and completed:
            with open(partial_path) as f:
                data = json.load(f)
            results = data.get('results', [])

    system = "You are a math expert. Solve problems step by step. Put your final answer in \\boxed{}."

    if USE_BATCH:
        # Batch mode
        client = anthropic.Anthropic(api_key=get_anthropic_key())
        claude_models = [m for m in MODELS if m in MODELS_CLAUDE]
        non_claude_models = [m for m in MODELS if m not in MODELS_CLAUDE]

        # Build pending problems
        pending_problems = []
        for idx, problem in enumerate(problems):
            if idx in completed:
                continue
            question = problem['problem']
            gold = extract_boxed_answer(problem['solution'])
            prompt = f"Solve this problem:\n\n{question}\n\nPut your final answer in \\boxed{{}}"
            pending_problems.append({
                'idx': idx,
                'question': question,
                'gold': gold,
                'prompt': prompt,
            })

        print(f"  Pending problems: {len(pending_problems)}")
        model_responses = {}

        # Submit batch for each Claude model
        for model in claude_models:
            requests = []
            for p in pending_problems:
                custom_id = f"math_{subject}_{p['idx']}_{model}"
                requests.append(
                    create_batch_request(custom_id,
                                         model,
                                         system,
                                         p['prompt'],
                                         max_tokens=2000))
            if requests:
                responses = run_batch_step(client, requests,
                                           f"{model.upper()} batch")
                model_responses[model] = responses

        # Run non-Claude models iteratively
        for model in non_claude_models:
            print(f"\n  Running {model} iteratively (non-Anthropic model)...")
            responses = {}
            for i, p in enumerate(pending_problems):
                result = call_model_with_retry(model, p['prompt'], system)
                responses[f"math_{subject}_{p['idx']}_{model}"] = result[
                    'response']
                print(f"  [{i+1}/{len(pending_problems)}]", end='\r')
            print()
            model_responses[model] = responses

        # Process all results
        for p in pending_problems:
            models_data = {}
            models_correct = {}

            for model in MODELS:
                custom_id = f"math_{subject}_{p['idx']}_{model}"
                response = model_responses.get(model, {}).get(custom_id)
                extracted = extract_boxed_answer(
                    response) if response else None
                correct = normalize_math_answer(
                    extracted) == normalize_math_answer(p['gold'])

                models_data[model] = {
                    'answer': extracted,
                    'correct': correct,
                    'time': 0,
                    'error': None if response else 'No response',
                }
                models_correct[model] = correct

            difficulty = classify_difficulty(models_correct)
            results.append({
                'problem_idx': p['idx'],
                'question': p['question'],
                'gold_answer': p['gold'],
                'models': models_data,
                'difficulty': difficulty,
            })

    else:
        # Iterative mode (with optional parallelism)
        def process_problem(idx_problem):
            idx, problem = idx_problem
            question = problem['problem']
            gold = extract_boxed_answer(problem['solution'])
            prompt = f"Solve this problem:\n\n{question}\n\nPut your final answer in \\boxed{{}}"

            models_data = {}
            models_correct = {}

            for model in MODELS:
                result = call_model_with_retry(model, prompt, system)
                extracted = extract_boxed_answer(
                    result['response']) if result['response'] else None
                correct = normalize_math_answer(
                    extracted) == normalize_math_answer(gold)

                models_data[model] = {
                    'answer': extracted,
                    'correct': correct,
                    'time': result['time'],
                    'error': result['error'],
                }
                models_correct[model] = correct

            difficulty = classify_difficulty(models_correct)

            return {
                'problem_idx': idx,
                'question': question,
                'gold_answer': gold,
                'models': models_data,
                'difficulty': difficulty,
                '_models_correct': models_correct,
            }

        pending = [(idx, p) for idx, p in enumerate(problems)
                   if idx not in completed]

        if PARALLEL_WORKERS > 1:
            import threading
            lock = threading.Lock()
            completed_count = [0]

            with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
                futures = {
                    executor.submit(process_problem, item): item
                    for item in pending
                }
                for future in as_completed(futures):
                    result = future.result()
                    models_correct = result.pop('_models_correct')
                    with lock:
                        results.append(result)
                        completed_count[0] += 1
                        print(format_progress(
                            results, models_correct,
                            completed_count[0] + len(completed), total),
                              end='',
                              flush=True)
                        if len(results) % SAVE_EVERY == 0:
                            save_results(
                                output_path, dataset_name, results, {
                                    'dataset': dataset_name,
                                    'subject': subject,
                                    'total': total
                                })
        else:
            for item in pending:
                result = process_problem(item)
                models_correct = result.pop('_models_correct')
                results.append(result)
                print(format_progress(results, models_correct,
                                      len(results) + len(completed), total),
                      end='',
                      flush=True)
                if len(results) % SAVE_EVERY == 0:
                    save_results(
                        output_path, dataset_name, results, {
                            'dataset': dataset_name,
                            'subject': subject,
                            'total': total
                        })

    print()
    save_results(output_path,
                 dataset_name,
                 results, {
                     'dataset': dataset_name,
                     'subject': subject,
                     'total': total
                 },
                 partial=False)
    print(f"Saved: {output_path}")

    return {'path': str(output_path), 'n_problems': len(results)}


# ============================================================================
# GPQA
# ============================================================================


def create_gpqa_mc_prompt(problem: Dict, problem_id: int) -> Tuple[str, str]:
    """Create GPQA multiple choice prompt with deterministic shuffling.

    Returns: (prompt, correct_letter)
    """
    # Use same shuffle logic as original baseline: seed = 42 + problem_id
    # Answers in order: [Incorrect1, Incorrect2, Incorrect3, Correct]
    answers = [
        problem['Incorrect Answer 1'],
        problem['Incorrect Answer 2'],
        problem['Incorrect Answer 3'],
        problem['Correct Answer'],
    ]

    rng = random.Random(42 + problem_id)
    indices = [0, 1, 2, 3]
    rng.shuffle(indices)

    letters = ['A', 'B', 'C', 'D']
    choices = []
    correct_letter = None

    for i, idx in enumerate(indices):
        letter = letters[i]
        answer = answers[idx]
        choices.append((letter, answer))
        if idx == 3:  # Correct answer was at index 3
            correct_letter = letter

    prompt = f"""{problem['Question']}

Choices:
A) {choices[0][1]}
B) {choices[1][1]}
C) {choices[2][1]}
D) {choices[3][1]}

Analyze this question carefully and select the best answer. State your answer as A, B, C, or D."""

    return prompt, correct_letter


def run_gpqa(format_type: str,
             output_dir: Path,
             resume: bool = False,
             num_problems: Optional[int] = None) -> Dict:
    """Run GPQA evaluation (mc or freeform)."""
    print(
        f"\n{'='*60}\nRunning GPQA ({format_type}) {'(BATCH)' if USE_BATCH else '(iterative)'}\n{'='*60}"
    )

    ds = load_dataset('Idavidrein/gpqa', 'gpqa_diamond')
    problems = list(ds['train'])
    if num_problems:
        problems = problems[:num_problems]

    total = len(problems)
    print(f"Total problems: {total}")

    dataset_name = f"gpqa_{format_type}"
    output_path = output_dir / f"{dataset_name}_{MODEL_VERSION}.json"

    # Skip if already exists
    if output_path.exists() and not resume:
        print(f"SKIP: {output_path.name} already exists")
        with open(output_path) as f:
            data = json.load(f)
        return data

    # Check for partial results
    completed = set()
    results = []
    if resume:
        partial_path, completed = find_partial_results(output_dir,
                                                       dataset_name)
        if partial_path and completed:
            with open(partial_path) as f:
                data = json.load(f)
            results = data.get('results', [])

    system = "You are an expert scientist with deep knowledge in physics, chemistry, and biology."
    # MC needs enough tokens for reasoning + answer (100 was too short)
    # Freeform needs even more for detailed explanations
    max_tokens = 1250 if format_type == 'mc' else 2000

    if USE_BATCH:
        # Batch mode
        client = anthropic.Anthropic(api_key=get_anthropic_key())
        claude_models = [m for m in MODELS if m in MODELS_CLAUDE]
        non_claude_models = [m for m in MODELS if m not in MODELS_CLAUDE]

        # Build pending problems
        pending_problems = []
        for idx, problem in enumerate(problems):
            if idx in completed:
                continue
            correct_answer = problem['Correct Answer']
            if format_type == 'mc':
                prompt, correct_letter = create_gpqa_mc_prompt(problem, idx)
                gold = correct_letter
            else:
                prompt = f"Question: {problem['Question']}\n\nProvide a clear, detailed answer."
                gold = correct_answer
            pending_problems.append({
                'idx': idx,
                'question': problem['Question'],
                'correct_answer': correct_answer,
                'gold': gold,
                'prompt': prompt,
            })

        print(f"  Pending problems: {len(pending_problems)}")
        model_responses = {}

        # Submit batch for each Claude model
        for model in claude_models:
            requests = []
            for p in pending_problems:
                custom_id = f"gpqa_{format_type}_{p['idx']}_{model}"
                requests.append(
                    create_batch_request(custom_id,
                                         model,
                                         system,
                                         p['prompt'],
                                         max_tokens=max_tokens))
            if requests:
                responses = run_batch_step(client, requests,
                                           f"{model.upper()} batch")
                model_responses[model] = responses

        # Run non-Claude models iteratively
        for model in non_claude_models:
            print(f"\n  Running {model} iteratively (non-Anthropic model)...")
            responses = {}
            for i, p in enumerate(pending_problems):
                result = call_model_with_retry(model,
                                               p['prompt'],
                                               system,
                                               max_tokens=max_tokens)
                responses[f"gpqa_{format_type}_{p['idx']}_{model}"] = result[
                    'response']
                print(f"  [{i+1}/{len(pending_problems)}]", end='\r')
            print()
            model_responses[model] = responses

        # Process all results
        for p in pending_problems:
            models_data = {}
            models_correct = {}

            for model in MODELS:
                custom_id = f"gpqa_{format_type}_{p['idx']}_{model}"
                response = model_responses.get(model, {}).get(custom_id) or ''

                if format_type == 'mc':
                    match = re.search(r'\b([A-D])\b', response.upper())
                    extracted = match.group(1) if match else None
                    correct = extracted == p['gold']
                else:
                    # Use LLM-as-judge for freeform evaluation
                    extracted = response[:500] if response else None
                    correct = judge_freeform_answer(p['question'],
                                                    p['correct_answer'],
                                                    response,
                                                    model_map=ACTIVE_MODEL_MAP)

                models_data[model] = {
                    'answer': extracted,
                    'correct': correct,
                    'time': 0,
                    'error': None if response else 'No response',
                }
                models_correct[model] = correct

            difficulty = classify_difficulty(models_correct)
            results.append({
                'problem_idx': p['idx'],
                'question': p['question'],
                'gold_answer': p['gold'],
                'models': models_data,
                'difficulty': difficulty,
            })

    else:
        # Iterative mode (with optional parallelism)
        def process_problem(idx_problem):
            idx, problem = idx_problem
            correct_answer = problem['Correct Answer']

            if format_type == 'mc':
                prompt, correct_letter = create_gpqa_mc_prompt(problem, idx)
                gold = correct_letter
            else:
                prompt = f"Question: {problem['Question']}\n\nProvide a clear, detailed answer."
                gold = correct_answer

            models_data = {}
            models_correct = {}

            for model in MODELS:
                result = call_model_with_retry(model,
                                               prompt,
                                               system,
                                               max_tokens=max_tokens)

                if format_type == 'mc':
                    response = result['response'] or ''
                    match = re.search(r'\b([A-D])\b', response.upper())
                    extracted = match.group(1) if match else None
                    correct = extracted == gold
                else:
                    # Use LLM-as-judge for freeform evaluation
                    response = result['response'] or ''
                    extracted = response[:500] if response else None
                    correct = judge_freeform_answer(problem['Question'],
                                                    correct_answer,
                                                    response,
                                                    model_map=ACTIVE_MODEL_MAP)

                models_data[model] = {
                    'answer': extracted,
                    'correct': correct,
                    'time': result['time'],
                    'error': result['error'],
                }
                models_correct[model] = correct

            difficulty = classify_difficulty(models_correct)

            return {
                'problem_idx': idx,
                'question': problem['Question'],
                'gold_answer': gold,
                'models': models_data,
                'difficulty': difficulty,
                '_models_correct': models_correct,
            }

        pending = [(idx, p) for idx, p in enumerate(problems)
                   if idx not in completed]

        if PARALLEL_WORKERS > 1:
            import threading
            lock = threading.Lock()
            completed_count = [0]

            with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
                futures = {
                    executor.submit(process_problem, item): item
                    for item in pending
                }
                for future in as_completed(futures):
                    result = future.result()
                    models_correct = result.pop('_models_correct')
                    with lock:
                        results.append(result)
                        completed_count[0] += 1
                        print(format_progress(
                            results, models_correct,
                            completed_count[0] + len(completed), total),
                              end='',
                              flush=True)
                        if len(results) % SAVE_EVERY == 0:
                            save_results(
                                output_path, dataset_name, results, {
                                    'dataset': dataset_name,
                                    'format': format_type,
                                    'total': total
                                })
        else:
            for item in pending:
                result = process_problem(item)
                models_correct = result.pop('_models_correct')
                results.append(result)
                print(format_progress(results, models_correct,
                                      len(results) + len(completed), total),
                      end='',
                      flush=True)
                if len(results) % SAVE_EVERY == 0:
                    save_results(
                        output_path, dataset_name, results, {
                            'dataset': dataset_name,
                            'format': format_type,
                            'total': total
                        })

    print()
    save_results(output_path,
                 dataset_name,
                 results, {
                     'dataset': dataset_name,
                     'format': format_type,
                     'total': total
                 },
                 partial=False)
    print(f"Saved: {output_path}")

    return {'path': str(output_path), 'n_problems': len(results)}


# ============================================================================
# MBPP
# ============================================================================


def extract_python_code(response: str) -> Optional[str]:
    """Extract Python code from response."""
    if not response:
        return None
    # Look for code blocks
    code_match = re.findall(r'```(?:python)?\s*(.*?)```', response, re.DOTALL)
    if code_match:
        return code_match[0].strip()
    # Look for function definitions
    func_match = re.findall(r'(def\s+\w+.*?)(?=\ndef\s|\Z)', response,
                            re.DOTALL)
    if func_match:
        return func_match[0].strip()
    return response.strip()


def extract_function_name_from_tests(tests: List[str]) -> Optional[str]:
    """Extract the expected function name from MBPP test assertions.

    Tests are like: 'assert square_perimeter(10)==40'
    We extract 'square_perimeter'.
    """
    if not tests:
        return None
    match = re.search(r'assert\s+(\w+)\s*\(', tests[0])
    if match:
        return match.group(1)
    return None


def run_code_tests(code: str,
                   tests: List[str],
                   timeout: int = 5) -> Tuple[int, int]:
    """Run tests on code. Returns (passed, total)."""
    if not code:
        return 0, len(tests)

    test_code = code + '\n\n' + '\n'.join(tests)

    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py',
                                         delete=False) as f:
            f.write(test_code)
            f.flush()
            temp_path = f.name

        result = subprocess.run(
            ['python3', temp_path],
            capture_output=True,
            timeout=timeout,
        )

        os.unlink(temp_path)

        if result.returncode == 0:
            return len(tests), len(tests)
        return 0, len(tests)
    except subprocess.TimeoutExpired:
        return 0, len(tests)
    except Exception:
        return 0, len(tests)
    finally:
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)


def run_mbpp(output_dir: Path,
             resume: bool = False,
             num_problems: Optional[int] = None) -> Dict:
    """Run MBPP evaluation (sanitized split)."""
    print(
        f"\n{'='*60}\nRunning MBPP (sanitized) {'(BATCH)' if USE_BATCH else '(iterative)'}\n{'='*60}"
    )

    ds = load_dataset('google-research-datasets/mbpp', 'sanitized')
    problems = list(ds['test'])
    if num_problems:
        problems = problems[:num_problems]

    total = len(problems)
    print(f"Total problems: {total}")

    output_path = output_dir / f"mbpp_{MODEL_VERSION}.json"

    # Skip if already exists
    if output_path.exists() and not resume:
        print(f"SKIP: {output_path.name} already exists")
        with open(output_path) as f:
            data = json.load(f)
        return data

    # Check for partial results
    completed = set()
    results = []
    if resume:
        partial_path, completed = find_partial_results(output_dir, 'mbpp')
        if partial_path and completed:
            with open(partial_path) as f:
                data = json.load(f)
            results = data.get('results', [])

    system = "You are an expert Python programmer. Write clean, efficient code."

    if USE_BATCH:
        # Batch mode
        client = anthropic.Anthropic(api_key=get_anthropic_key())
        claude_models = [m for m in MODELS if m in MODELS_CLAUDE]
        non_claude_models = [m for m in MODELS if m not in MODELS_CLAUDE]

        # Build pending problems
        pending_problems = []
        for idx, problem in enumerate(problems):
            if idx in completed:
                continue
            prompt_text = problem['prompt']
            tests = problem['test_list']
            func_name = extract_function_name_from_tests(tests)
            func_instruction = f"\n\nThe function must be named: {func_name}" if func_name else ""
            prompt = f"""Write a Python function that solves this problem:

{prompt_text}{func_instruction}

Provide only the Python code, no explanation needed."""
            pending_problems.append({
                'idx': idx,
                'task_id': problem.get('task_id'),
                'prompt_text': prompt_text,
                'prompt': prompt,
                'tests': tests,
            })

        print(f"  Pending problems: {len(pending_problems)}")
        model_responses = {}

        # Submit batch for each Claude model
        for model in claude_models:
            requests = []
            for p in pending_problems:
                custom_id = f"mbpp_{p['idx']}_{model}"
                requests.append(
                    create_batch_request(custom_id,
                                         model,
                                         system,
                                         p['prompt'],
                                         max_tokens=1000))
            if requests:
                responses = run_batch_step(client, requests,
                                           f"{model.upper()} batch")
                model_responses[model] = responses

        # Run non-Claude models iteratively
        for model in non_claude_models:
            print(f"\n  Running {model} iteratively (non-Anthropic model)...")
            responses = {}
            for i, p in enumerate(pending_problems):
                result = call_model_with_retry(model,
                                               p['prompt'],
                                               system,
                                               max_tokens=1000)
                responses[f"mbpp_{p['idx']}_{model}"] = result['response']
                print(f"  [{i+1}/{len(pending_problems)}]", end='\r')
            print()
            model_responses[model] = responses

        # Process all results (run tests locally)
        print("  Running code tests...")
        for i, p in enumerate(pending_problems):
            models_data = {}
            models_correct = {}

            for model in MODELS:
                custom_id = f"mbpp_{p['idx']}_{model}"
                response = model_responses.get(model, {}).get(custom_id)
                code = extract_python_code(response) if response else None
                passed, total_tests = run_code_tests(code, p['tests'])
                correct = passed == total_tests and total_tests > 0

                models_data[model] = {
                    'code': code[:500] if code else None,
                    'tests_passed': passed,
                    'tests_total': total_tests,
                    'correct': correct,
                    'time': 0,
                    'error': None if response else 'No response',
                }
                models_correct[model] = correct

            difficulty = classify_difficulty(models_correct)
            results.append({
                'problem_idx': p['idx'],
                'task_id': p['task_id'],
                'prompt': p['prompt_text'],
                'models': models_data,
                'difficulty': difficulty,
            })
            print(f"  Testing [{i+1}/{len(pending_problems)}]", end='\r')
        print()

    else:
        # Iterative mode (with optional parallelism)
        def process_problem(idx_problem):
            idx, problem = idx_problem
            prompt_text = problem['prompt']
            tests = problem['test_list']
            func_name = extract_function_name_from_tests(tests)
            func_instruction = f"\n\nThe function must be named: {func_name}" if func_name else ""

            prompt = f"""Write a Python function that solves this problem:

{prompt_text}{func_instruction}

Provide only the Python code, no explanation needed."""

            models_data = {}
            models_correct = {}

            for model in MODELS:
                result = call_model_with_retry(model,
                                               prompt,
                                               system,
                                               max_tokens=1000)
                code = extract_python_code(
                    result['response']) if result['response'] else None
                passed, total_tests = run_code_tests(code, tests)
                correct = passed == total_tests and total_tests > 0

                models_data[model] = {
                    'code': code[:500] if code else None,
                    'tests_passed': passed,
                    'tests_total': total_tests,
                    'correct': correct,
                    'time': result['time'],
                    'error': result['error'],
                }
                models_correct[model] = correct

            difficulty = classify_difficulty(models_correct)

            return {
                'problem_idx': idx,
                'task_id': problem.get('task_id'),
                'prompt': prompt_text,
                'models': models_data,
                'difficulty': difficulty,
                '_models_correct': models_correct,
            }

        pending = [(idx, p) for idx, p in enumerate(problems)
                   if idx not in completed]

        if PARALLEL_WORKERS > 1:
            import threading
            lock = threading.Lock()
            completed_count = [0]

            with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
                futures = {
                    executor.submit(process_problem, item): item
                    for item in pending
                }
                for future in as_completed(futures):
                    result = future.result()
                    models_correct = result.pop('_models_correct')
                    with lock:
                        results.append(result)
                        completed_count[0] += 1
                        print(format_progress(
                            results, models_correct,
                            completed_count[0] + len(completed), total),
                              end='',
                              flush=True)
                        if len(results) % SAVE_EVERY == 0:
                            save_results(
                                output_path, 'mbpp', results, {
                                    'dataset': 'mbpp',
                                    'split': 'sanitized',
                                    'total': total
                                })
        else:
            for item in pending:
                result = process_problem(item)
                models_correct = result.pop('_models_correct')
                results.append(result)
                print(format_progress(results, models_correct,
                                      len(results) + len(completed), total),
                      end='',
                      flush=True)
                if len(results) % SAVE_EVERY == 0:
                    save_results(output_path, 'mbpp', results, {
                        'dataset': 'mbpp',
                        'split': 'sanitized',
                        'total': total
                    })

    print()
    save_results(output_path,
                 'mbpp',
                 results, {
                     'dataset': 'mbpp',
                     'split': 'sanitized',
                     'total': total
                 },
                 partial=False)
    print(f"Saved: {output_path}")

    return {'path': str(output_path), 'n_problems': len(results)}


# ============================================================================
# MMLU Pro (freeform)
# ============================================================================


def run_mmlu_pro(output_dir: Path,
                 resume: bool = False,
                 num_problems: Optional[int] = None) -> Dict:
    """Run MMLU Pro evaluation (freeform answers, LLM-as-judge)."""
    print(
        f"\n{'='*60}\nRunning MMLU Pro {'(BATCH)' if USE_BATCH else '(iterative)'}\n{'='*60}"
    )

    ds = load_dataset('nikhilchandak/freeform-datasets', split='mmlu_pro')
    problems = list(ds)
    if num_problems:
        problems = problems[:num_problems]

    total = len(problems)
    print(f"Total problems: {total}")

    output_path = output_dir / f"mmlu_pro_{MODEL_VERSION}.json"

    # Skip if already exists
    if output_path.exists() and not resume:
        print(f"SKIP: {output_path.name} already exists")
        with open(output_path) as f:
            data = json.load(f)
        return data

    # Check for partial results
    completed = set()
    results = []
    if resume:
        partial_path, completed = find_partial_results(output_dir, 'mmlu_pro')
        if partial_path and completed:
            with open(partial_path) as f:
                data = json.load(f)
            results = data.get('results', [])

    system = "You are an expert across many academic disciplines. Provide clear, accurate answers."
    max_tokens = 2000  # Freeform answers need room for explanation

    if USE_BATCH:
        # Batch mode
        client = anthropic.Anthropic(api_key=get_anthropic_key())
        claude_models = [m for m in MODELS if m in MODELS_CLAUDE]
        non_claude_models = [m for m in MODELS if m not in MODELS_CLAUDE]

        # Build pending problems
        pending_problems = []
        for idx, problem in enumerate(problems):
            if idx in completed:
                continue
            prompt = f"Question: {problem['question']}\n\nProvide a clear, accurate answer."
            pending_problems.append({
                'idx': idx,
                'question_id': problem.get('question_id'),
                'question': problem['question'],
                'correct_answer': problem['answer'],
                'category': problem.get('category', ''),
                'prompt': prompt,
            })

        print(f"  Pending problems: {len(pending_problems)}")
        model_responses = {}

        # Submit batch for each Claude model
        for model in claude_models:
            requests = []
            for p in pending_problems:
                custom_id = f"mmlu_pro_{p['idx']}_{model}"
                requests.append(
                    create_batch_request(custom_id,
                                         model,
                                         system,
                                         p['prompt'],
                                         max_tokens=max_tokens))
            if requests:
                responses = run_batch_step(client, requests,
                                           f"{model.upper()} batch")
                model_responses[model] = responses

        # Run non-Claude models iteratively
        for model in non_claude_models:
            print(f"\n  Running {model} iteratively (non-Anthropic model)...")
            responses = {}
            for i, p in enumerate(pending_problems):
                result = call_model_with_retry(model,
                                               p['prompt'],
                                               system,
                                               max_tokens=max_tokens)
                responses[f"mmlu_pro_{p['idx']}_{model}"] = result['response']
                print(f"  [{i+1}/{len(pending_problems)}]", end='\r')
            print()
            model_responses[model] = responses

        # Process all results with LLM-as-judge
        for p in pending_problems:
            models_data = {}
            models_correct = {}

            for model in MODELS:
                custom_id = f"mmlu_pro_{p['idx']}_{model}"
                response = model_responses.get(model, {}).get(custom_id) or ''

                # Use LLM-as-judge for freeform evaluation
                extracted = response[:500] if response else None
                correct = judge_freeform_answer(p['question'],
                                                p['correct_answer'],
                                                response,
                                                model_map=ACTIVE_MODEL_MAP)

                models_data[model] = {
                    'answer': extracted,
                    'correct': correct,
                    'time': 0,
                    'error': None if response else 'No response',
                }
                models_correct[model] = correct

            difficulty = classify_difficulty(models_correct)
            results.append({
                'problem_idx': p['idx'],
                'question_id': p['question_id'],
                'question': p['question'],
                'category': p['category'],
                'gold_answer': p['correct_answer'],
                'models': models_data,
                'difficulty': difficulty,
            })

    else:
        # Iterative mode (with optional parallelism)
        def process_problem(idx_problem):
            idx, problem = idx_problem
            question = problem['question']
            correct_answer = problem['answer']
            category = problem.get('category', '')

            prompt = f"Question: {question}\n\nProvide a clear, accurate answer."

            models_data = {}
            models_correct = {}

            for model in MODELS:
                result = call_model_with_retry(model,
                                               prompt,
                                               system,
                                               max_tokens=max_tokens)

                # Use LLM-as-judge for freeform evaluation
                response = result['response'] or ''
                extracted = response[:500] if response else None
                correct = judge_freeform_answer(question,
                                                correct_answer,
                                                response,
                                                model_map=ACTIVE_MODEL_MAP)

                models_data[model] = {
                    'answer': extracted,
                    'correct': correct,
                    'time': result['time'],
                    'error': result['error'],
                }
                models_correct[model] = correct

            difficulty = classify_difficulty(models_correct)

            return {
                'problem_idx': idx,
                'question_id': problem.get('question_id'),
                'question': question,
                'category': category,
                'gold_answer': correct_answer,
                'models': models_data,
                'difficulty': difficulty,
                '_models_correct': models_correct,
            }

        pending = [(idx, p) for idx, p in enumerate(problems)
                   if idx not in completed]

        if PARALLEL_WORKERS > 1:
            import threading
            lock = threading.Lock()
            completed_count = [0]

            with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
                futures = {
                    executor.submit(process_problem, item): item
                    for item in pending
                }
                for future in as_completed(futures):
                    result = future.result()
                    models_correct = result.pop('_models_correct')
                    with lock:
                        results.append(result)
                        completed_count[0] += 1
                        print(format_progress(
                            results, models_correct,
                            completed_count[0] + len(completed), total),
                              end='',
                              flush=True)
                        if len(results) % SAVE_EVERY == 0:
                            save_results(output_path, 'mmlu_pro', results, {
                                'dataset': 'mmlu_pro',
                                'total': total
                            })
        else:
            for item in pending:
                result = process_problem(item)
                models_correct = result.pop('_models_correct')
                results.append(result)
                print(format_progress(results, models_correct,
                                      len(results) + len(completed), total),
                      end='',
                      flush=True)
                if len(results) % SAVE_EVERY == 0:
                    save_results(output_path, 'mmlu_pro', results, {
                        'dataset': 'mmlu_pro',
                        'total': total
                    })

    print()
    save_results(output_path,
                 'mmlu_pro',
                 results, {
                     'dataset': 'mmlu_pro',
                     'total': total
                 },
                 partial=False)
    print(f"Saved: {output_path}")

    return {'path': str(output_path), 'n_problems': len(results)}


# ============================================================================
# HLE (Humanity's Last Exam)
# ============================================================================


def run_hle(output_dir: Path,
            resume: bool = False,
            num_problems: Optional[int] = None) -> Dict:
    """Run HLE (Humanity's Last Exam) evaluation.

    HLE is an extremely difficult benchmark with 2500 problems across
    math, physics, CS/AI, biology, chemistry, humanities, and more.

    We filter to text-only problems (2158 total) since we don't support
    image inputs. Answer types are exactMatch (1909) and multipleChoice (591).
    """
    print(
        f"\n{'='*60}\nRunning HLE {'(BATCH)' if USE_BATCH else '(iterative)'}\n{'='*60}"
    )

    ds, indices = load_hle_dataset(text_only=True)
    if num_problems:
        indices = indices[:num_problems]

    total = len(indices)
    print(f"Total text-only problems: {total}")

    output_path = output_dir / f"hle_{MODEL_VERSION}.json"

    # Skip if already exists
    if output_path.exists() and not resume:
        print(f"SKIP: {output_path.name} already exists")
        with open(output_path) as f:
            data = json.load(f)
        return data

    # Check for partial results
    completed = set()
    results = []
    if resume:
        partial_path, completed = find_partial_results(output_dir, 'hle')
        if partial_path and completed:
            with open(partial_path) as f:
                data = json.load(f)
            results = data.get('results', [])

    system = "You are a highly knowledgeable assistant. Answer questions carefully and precisely."

    if USE_BATCH:
        # Batch mode: collect all requests, submit in batches per model
        client = anthropic.Anthropic(api_key=get_anthropic_key())

        # Get Claude models only (batch API doesn't support OpenRouter)
        claude_models = [m for m in MODELS if m in MODELS_CLAUDE]
        non_claude_models = [m for m in MODELS if m not in MODELS_CLAUDE]

        # Build problem info for non-completed problems
        pending_problems = []
        for list_idx, ds_idx in enumerate(indices):
            if list_idx in completed:
                continue
            problem = get_hle_problem(ds, ds_idx)
            prompt = build_hle_prompt(problem['question'],
                                      problem['answer_type'])
            pending_problems.append({
                'list_idx': list_idx,
                'ds_idx': ds_idx,
                'problem': problem,
                'prompt': prompt,
            })

        print(f"  Pending problems: {len(pending_problems)}")

        # Collect batch responses per model
        model_responses = {}

        # Submit batch for each Claude model
        for model in claude_models:
            requests = []
            for p in pending_problems:
                custom_id = f"hle_{p['list_idx']}_{model}"
                requests.append(
                    create_batch_request(custom_id,
                                         model,
                                         system,
                                         p['prompt'],
                                         max_tokens=2048))

            if requests:
                responses = run_batch_step(client, requests,
                                           f"{model.upper()} batch")
                model_responses[model] = responses

        # Run non-Claude models iteratively (e.g., gpt-oss via OpenRouter)
        for model in non_claude_models:
            print(f"\n  Running {model} iteratively (non-Anthropic model)...")
            responses = {}
            for i, p in enumerate(pending_problems):
                custom_id = f"hle_{p['list_idx']}_{model}"
                result = call_model_with_retry(model,
                                               p['prompt'],
                                               system,
                                               max_tokens=2048)
                responses[custom_id] = result['response'] if result[
                    'response'] else ''
                if (i + 1) % 10 == 0:
                    print(f"    [{i+1}/{len(pending_problems)}]", end='\r')
            model_responses[model] = responses
            print()

        # Process results
        for p in pending_problems:
            problem = p['problem']
            models_data = {}
            models_correct = {}

            for model in MODELS:
                custom_id = f"hle_{p['list_idx']}_{model}"
                response_text = model_responses.get(model,
                                                    {}).get(custom_id, '')
                extracted = extract_hle_answer(response_text,
                                               problem['answer_type'])
                correct = check_hle_answer(extracted, problem['answer'],
                                           problem['answer_type'])

                models_data[model] = {
                    'response': response_text[:500]
                    if response_text else None,  # Truncate for storage
                    'extracted': extracted,
                    'correct': correct,
                }
                models_correct[model] = correct

            difficulty = classify_difficulty(models_correct)

            results.append({
                'problem_idx': p['list_idx'],
                'ds_idx': p['ds_idx'],
                'question': problem['question'][:200],  # Truncate for storage
                'answer': problem['answer'],
                'answer_type': problem['answer_type'],
                'category': problem['category'],
                'subject': problem['subject'],
                'models': models_data,
                'difficulty': difficulty,
            })

    else:
        # Iterative mode (with optional parallelism)
        def process_problem(list_idx_ds_idx):
            list_idx, ds_idx = list_idx_ds_idx
            problem = get_hle_problem(ds, ds_idx)
            prompt = build_hle_prompt(problem['question'],
                                      problem['answer_type'])

            models_data = {}
            models_correct = {}

            for model in MODELS:
                result = call_model_with_retry(model,
                                               prompt,
                                               system,
                                               max_tokens=2048)
                response_text = result['response'] if result['response'] else ''
                extracted = extract_hle_answer(response_text,
                                               problem['answer_type'])
                correct = check_hle_answer(extracted, problem['answer'],
                                           problem['answer_type'])

                models_data[model] = {
                    'response': response_text[:500] if response_text else None,
                    'extracted': extracted,
                    'correct': correct,
                    'time': result['time'],
                    'error': result['error'],
                }
                models_correct[model] = correct

            difficulty = classify_difficulty(models_correct)

            return {
                'problem_idx': list_idx,
                'ds_idx': ds_idx,
                'question': problem['question'][:200],
                'answer': problem['answer'],
                'answer_type': problem['answer_type'],
                'category': problem['category'],
                'subject': problem['subject'],
                'models': models_data,
                'difficulty': difficulty,
                '_models_correct': models_correct,  # For progress display
            }

        # Filter out completed problems
        pending = [(list_idx, ds_idx)
                   for list_idx, ds_idx in enumerate(indices)
                   if list_idx not in completed]

        if PARALLEL_WORKERS > 1:
            # Parallel execution
            import threading
            lock = threading.Lock()
            completed_count = [0]

            with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
                futures = {
                    executor.submit(process_problem, item): item
                    for item in pending
                }
                for future in as_completed(futures):
                    result = future.result()
                    models_correct = result.pop('_models_correct')
                    with lock:
                        results.append(result)
                        completed_count[0] += 1
                        print(format_progress(
                            results, models_correct,
                            completed_count[0] + len(completed), total),
                              end='',
                              flush=True)
                        if len(results) % SAVE_EVERY == 0:
                            save_results(output_path, 'hle', results, {
                                'dataset': 'hle',
                                'total': total
                            })
        else:
            # Sequential execution
            for item in pending:
                result = process_problem(item)
                models_correct = result.pop('_models_correct')
                results.append(result)
                print(format_progress(results, models_correct,
                                      len(results) + len(completed), total),
                      end='',
                      flush=True)
                if len(results) % SAVE_EVERY == 0:
                    save_results(output_path, 'hle', results, {
                        'dataset': 'hle',
                        'total': total
                    })

    print()
    save_results(output_path,
                 'hle',
                 results, {
                     'dataset': 'hle',
                     'total': total
                 },
                 partial=False)
    print(f"Saved: {output_path}")

    return {'path': str(output_path), 'n_problems': len(results)}


# ============================================================================
# AIME (American Invitational Mathematics Examination)
# ============================================================================


def run_aime(output_dir: Path,
             resume: bool = False,
             num_problems: Optional[int] = None) -> Dict:
    """Run AIME (American Invitational Mathematics Examination) evaluation.

    AIME is a competition-level math exam with 90 problems.
    All answers are integers from 0 to 999.
    """
    print(
        f"\n{'='*60}\nRunning AIME {'(BATCH)' if USE_BATCH else '(iterative)'}\n{'='*60}"
    )

    ds, indices = load_aime_dataset()
    if num_problems:
        indices = indices[:num_problems]

    total = len(indices)
    print(f"Total problems: {total}")

    output_path = output_dir / f"aime_{MODEL_VERSION}.json"

    # Skip if already exists
    if output_path.exists() and not resume:
        print(f"SKIP: {output_path.name} already exists")
        with open(output_path) as f:
            data = json.load(f)
        return data

    # Check for partial results
    completed = set()
    results = []
    if resume:
        partial_path, completed = find_partial_results(output_dir, 'aime')
        if partial_path and completed:
            with open(partial_path) as f:
                data = json.load(f)
            results = data.get('results', [])

    system = "You are an expert competition mathematician. Solve problems carefully and show your work."

    if USE_BATCH:
        # Batch mode
        client = anthropic.Anthropic(api_key=get_anthropic_key())

        claude_models = [m for m in MODELS if m in MODELS_CLAUDE]
        non_claude_models = [m for m in MODELS if m not in MODELS_CLAUDE]

        pending_problems = []
        for list_idx, ds_idx in enumerate(indices):
            if list_idx in completed:
                continue
            problem = get_aime_problem(ds, ds_idx)
            prompt = build_aime_prompt(problem['problem'])
            pending_problems.append({
                'list_idx': list_idx,
                'ds_idx': ds_idx,
                'problem': problem,
                'prompt': prompt,
            })

        print(f"  Pending problems: {len(pending_problems)}")

        model_responses = {}

        for model in claude_models:
            requests = []
            for p in pending_problems:
                custom_id = f"aime_{p['list_idx']}_{model}"
                requests.append(
                    create_batch_request(custom_id,
                                         model,
                                         system,
                                         p['prompt'],
                                         max_tokens=2048))

            if requests:
                responses = run_batch_step(client, requests,
                                           f"{model.upper()} batch")
                model_responses[model] = responses

        for model in non_claude_models:
            print(f"\n  Running {model} iteratively (non-Anthropic model)...")
            responses = {}
            for i, p in enumerate(pending_problems):
                custom_id = f"aime_{p['list_idx']}_{model}"
                result = call_model_with_retry(model,
                                               p['prompt'],
                                               system,
                                               max_tokens=2048)
                responses[custom_id] = result['response'] if result[
                    'response'] else ''
                if (i + 1) % 10 == 0:
                    print(f"    [{i+1}/{len(pending_problems)}]", end='\r')
            model_responses[model] = responses
            print()

        for p in pending_problems:
            problem = p['problem']
            models_data = {}
            models_correct = {}

            for model in MODELS:
                custom_id = f"aime_{p['list_idx']}_{model}"
                response_text = model_responses.get(model,
                                                    {}).get(custom_id, '')
                extracted = extract_aime_answer(response_text)
                correct = check_aime_answer(extracted, problem['answer'])

                models_data[model] = {
                    'response': response_text[:500] if response_text else None,
                    'extracted': extracted,
                    'correct': correct,
                }
                models_correct[model] = correct

            difficulty = classify_difficulty(models_correct)

            results.append({
                'problem_idx': p['list_idx'],
                'ds_idx': p['ds_idx'],
                'problem': problem['problem'][:200],
                'gold_answer': problem['answer'],
                'models': models_data,
                'difficulty': difficulty,
            })

    else:
        # Iterative mode
        def process_problem(list_idx_ds_idx):
            list_idx, ds_idx = list_idx_ds_idx
            problem = get_aime_problem(ds, ds_idx)
            prompt = build_aime_prompt(problem['problem'])

            models_data = {}
            models_correct = {}

            for model in MODELS:
                result = call_model_with_retry(model,
                                               prompt,
                                               system,
                                               max_tokens=2048)
                response_text = result['response'] if result['response'] else ''
                extracted = extract_aime_answer(response_text)
                correct = check_aime_answer(extracted, problem['answer'])

                models_data[model] = {
                    'response': response_text[:500] if response_text else None,
                    'extracted': extracted,
                    'correct': correct,
                    'time': result['time'],
                    'error': result['error'],
                }
                models_correct[model] = correct

            difficulty = classify_difficulty(models_correct)

            return {
                'problem_idx': list_idx,
                'ds_idx': ds_idx,
                'problem': problem['problem'][:200],
                'gold_answer': problem['answer'],
                'models': models_data,
                'difficulty': difficulty,
                '_models_correct': models_correct,
            }

        pending = [(list_idx, ds_idx)
                   for list_idx, ds_idx in enumerate(indices)
                   if list_idx not in completed]

        if PARALLEL_WORKERS > 1:
            import threading
            lock = threading.Lock()
            completed_count = [0]

            with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
                futures = {
                    executor.submit(process_problem, item): item
                    for item in pending
                }
                for future in as_completed(futures):
                    result = future.result()
                    models_correct = result.pop('_models_correct')
                    with lock:
                        results.append(result)
                        completed_count[0] += 1
                        print(format_progress(
                            results, models_correct,
                            completed_count[0] + len(completed), total),
                              end='',
                              flush=True)
                        if len(results) % SAVE_EVERY == 0:
                            save_results(output_path, 'aime', results, {
                                'dataset': 'aime',
                                'total': total
                            })
        else:
            for item in pending:
                result = process_problem(item)
                models_correct = result.pop('_models_correct')
                results.append(result)
                print(format_progress(results, models_correct,
                                      len(results) + len(completed), total),
                      end='',
                      flush=True)
                if len(results) % SAVE_EVERY == 0:
                    save_results(output_path, 'aime', results, {
                        'dataset': 'aime',
                        'total': total
                    })

    print()
    save_results(output_path,
                 'aime',
                 results, {
                     'dataset': 'aime',
                     'total': total
                 },
                 partial=False)
    print(f"Saved: {output_path}")

    return {'path': str(output_path), 'n_problems': len(results)}


# ============================================================================
# Main
# ============================================================================


def main():
    global MODELS, USE_BATCH, PARALLEL_WORKERS, ACTIVE_MODEL_MAP  # Need to modify global variables

    parser = argparse.ArgumentParser(
        description='Run baseline evaluation on all datasets')
    parser.add_argument('--dataset',
                        type=str,
                        default='all',
                        choices=[
                            'all', 'gsm8k', 'math', 'gpqa', 'mbpp', 'mmlu_pro',
                            'hle', 'aime'
                        ],
                        help='Dataset to run (default: all)')
    parser.add_argument(
        '--subject',
        type=str,
        default='all',
        choices=['all', 'algebra', 'geometry', 'number_theory'],
        help='MATH subject (default: all)')
    parser.add_argument('--format',
                        type=str,
                        default='all',
                        choices=['all', 'mc', 'freeform'],
                        help='GPQA format (default: all)')
    parser.add_argument('--num-problems',
                        type=int,
                        default=None,
                        help='Limit number of problems per dataset')
    parser.add_argument('--resume',
                        action='store_true',
                        help='Resume from partial results')
    parser.add_argument('--output-dir',
                        type=str,
                        default='lossy_compression/results',
                        help='Output directory')
    parser.add_argument('--include-gpt-oss',
                        action='store_true',
                        help='Include GPT-OSS-120B via OpenRouter (run last)')
    parser.add_argument(
        '--batch',
        action='store_true',
        help=
        'Use Anthropic Message Batches API for Claude models (50%% cheaper, more efficient)'
    )
    parser.add_argument(
        '--parallel',
        type=int,
        default=1,
        help=
        'Number of parallel workers for iterative mode (default: 1, use 6 for faster runs)'
    )
    parser.add_argument(
        '--use-old-models',
        action='store_true',
        help=
        'Use old model versions (3.5 haiku, sonnet 4, opus 4) instead of 4.5 models'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.0,
        help=
        'Sampling temperature (default: 0.0 = deterministic, use 1.0 for variance runs)'
    )

    args = parser.parse_args()

    # Update global flags
    USE_BATCH = args.batch
    PARALLEL_WORKERS = args.parallel
    global TEMPERATURE
    TEMPERATURE = args.temperature

    # Set model map based on flag
    global ACTIVE_MODEL_MAP, MODEL_VERSION
    if args.use_old_models:
        ACTIVE_MODEL_MAP = MODEL_ALIAS_MAP_old
        MODEL_VERSION = "v3.5"
        print(
            "Using OLD models: claude-3-5-haiku, claude-sonnet-4, claude-opus-4"
        )
    else:
        ACTIVE_MODEL_MAP = MODEL_ALIAS_MAP
        MODEL_VERSION = "v4.5"
        print(
            "Using 4.5 models: claude-haiku-4-5, claude-sonnet-4-5, claude-opus-4-5"
        )

    # Update MODELS list based on flags
    if args.include_gpt_oss:
        MODELS = MODELS_CLAUDE + MODELS_GPT_OSS
        print(f"Models: {MODELS} (GPT-OSS included, will run last)")
    else:
        MODELS = MODELS_CLAUDE
        print(f"Models: {MODELS}")

    if TEMPERATURE > 0:
        print(f"Temperature: {TEMPERATURE}")

    if USE_BATCH:
        print("Mode: BATCH (using Anthropic Message Batches API)")
    elif PARALLEL_WORKERS > 1:
        print(f"Mode: ITERATIVE PARALLEL ({PARALLEL_WORKERS} workers)")
    else:
        print("Mode: ITERATIVE (individual API calls)")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(f"Resume: {args.resume}")

    all_results = {}

    # GSM8K
    if args.dataset in ['all', 'gsm8k']:
        all_results['gsm8k'] = run_gsm8k(output_dir, args.resume,
                                         args.num_problems)

    # MATH
    if args.dataset in ['all', 'math']:
        subjects = ['algebra', 'geometry', 'number_theory'
                    ] if args.subject == 'all' else [args.subject]
        for subject in subjects:
            all_results[f'math_{subject}'] = run_math(subject, output_dir,
                                                      args.resume,
                                                      args.num_problems)

    # GPQA
    if args.dataset in ['all', 'gpqa']:
        formats = ['mc', 'freeform'] if args.format == 'all' else [args.format]
        for fmt in formats:
            all_results[f'gpqa_{fmt}'] = run_gpqa(fmt, output_dir, args.resume,
                                                  args.num_problems)

    # MBPP
    if args.dataset in ['all', 'mbpp']:
        all_results['mbpp'] = run_mbpp(output_dir, args.resume,
                                       args.num_problems)

    # MMLU Pro
    if args.dataset in ['all', 'mmlu_pro']:
        all_results['mmlu_pro'] = run_mmlu_pro(output_dir, args.resume,
                                               args.num_problems)

    # HLE (Humanity's Last Exam)
    if args.dataset in ['all', 'hle']:
        all_results['hle'] = run_hle(output_dir, args.resume,
                                     args.num_problems)

    # AIME (American Invitational Mathematics Examination)
    if args.dataset in ['all', 'aime']:
        all_results['aime'] = run_aime(output_dir, args.resume,
                                       args.num_problems)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, info in all_results.items():
        print(f"{name}: {info['n_problems']} problems -> {info['path']}")


if __name__ == '__main__':
    main()
