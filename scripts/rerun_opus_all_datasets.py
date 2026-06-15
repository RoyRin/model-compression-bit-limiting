#!/usr/bin/env python3
"""
Rerun Opus evaluation on ALL datasets to verify results.

This script:
1. Runs ONLY Opus (not haiku/sonnet) on each dataset
2. Implements exponential backoff on rate limits (up to 10 min wait)
3. Saves results incrementally and supports resuming from partial results
4. Saves results that can be merged with existing baselines to update difficulty

Usage:
    python scripts/rerun_opus_all_datasets.py
    python scripts/rerun_opus_all_datasets.py --dataset gsm8k
    python scripts/rerun_opus_all_datasets.py --dataset math --subject algebra

    # Resume from partial results
    python scripts/rerun_opus_all_datasets.py --resume
    python scripts/rerun_opus_all_datasets.py --resume --resume-dir results/opus_rerun
"""

import json
import time
import argparse
import re
import os
import sys
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
from lossy_compression import MODEL_ALIAS_MAP, model_completion

# Rate limit detection
RATE_LIMIT_ERRORS = [
    'rate_limit', 'rate limit', 'too many requests', '429', 'overloaded',
    'capacity', 'throttl'
]

# How often to save progress (every N problems)
SAVE_EVERY = 10

# Datasets to run
DATASETS = {
    'math_algebra': {
        'type': 'math',
        'subject': 'algebra'
    },
    'math_geometry': {
        'type': 'math',
        'subject': 'geometry'
    },
    'math_number_theory': {
        'type': 'math',
        'subject': 'number_theory'
    },
    'gsm8k': {
        'type': 'gsm8k'
    },
    'gpqa_mc': {
        'type': 'gpqa',
        'format': 'mc'
    },
    'gpqa_freeform': {
        'type': 'gpqa',
        'format': 'freeform'
    },
    'mbpp': {
        'type': 'mbpp'
    },
}


class RateLimitError(Exception):
    """Raised when rate limiting is detected."""
    pass


class APIError(Exception):
    """Raised when API returns an error."""
    pass


def find_latest_partial_results(output_dir: Path, dataset_name: str) -> tuple:
    """Find the most recent partial results file for a dataset.

    Returns: (filepath, completed_indices) or (None, set())
    """
    pattern = f"opus_{dataset_name}_*.json"
    files = list(output_dir.glob(pattern))

    if not files:
        return None, set()

    # Sort by modification time, most recent first
    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

    for filepath in files:
        try:
            with open(filepath) as f:
                data = json.load(f)

            if 'results' in data and data['results']:
                completed = {
                    r['problem_idx']
                    for r in data['results'] if r.get('opus_success', False)
                }
                print(
                    f"  Found partial results: {filepath.name} ({len(completed)} completed)"
                )
                return filepath, completed
        except (json.JSONDecodeError, KeyError):
            continue

    return None, set()


def load_partial_results(filepath: Path) -> dict:
    """Load partial results from a file."""
    with open(filepath) as f:
        return json.load(f)


def save_partial_results(output_path: Path, dataset_name: str, results: list,
                         correct: int, total: int):
    """Save partial results to enable resuming."""
    data = {
        'dataset': dataset_name,
        'total': total,
        'completed': len(results),
        'opus_correct': correct,
        'opus_accuracy': correct / len(results) if results else 0,
        'results': results,
        'partial': len(results) < total,
        'last_updated': datetime.now().isoformat(),
    }
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def check_for_rate_limit(error_msg: str) -> bool:
    """Check if error message indicates rate limiting."""
    error_lower = str(error_msg).lower()
    return any(term in error_lower for term in RATE_LIMIT_ERRORS)


def extract_boxed_answer(text):
    """Extract answer from \\boxed{...} format."""
    if not text:
        return None
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    return None


def extract_numerical_answer(text):
    """Extract numerical answer for GSM8K."""
    if not text:
        return None
    # Look for #### format first
    match = re.search(r'####\s*(\d+)', text)
    if match:
        return match.group(1)
    # Look for boxed
    boxed = extract_boxed_answer(text)
    if boxed:
        nums = re.findall(r'\d+', boxed)
        if nums:
            return nums[-1]
    # Last number in text
    nums = re.findall(r'\b(\d+)\b', text)
    if nums:
        return nums[-1]
    return None


def normalize_answer(answer):
    """Normalize answer for comparison."""
    if answer is None:
        return None
    ans = str(answer).strip().lower()
    ans = ans.replace('\\$', '').replace('$', '')
    ans = ans.replace('\\text{', '').replace('}', '')
    ans = ans.replace('\\', '').replace(' ', '')
    return ans


def call_opus_with_retry(prompt: str,
                         system: str,
                         max_retries: int = 10,
                         max_tokens: int = 2000) -> dict:
    """Call Opus with retry logic and exponential backoff for rate limits.

    Rate limit backoff: 30s, 60s, 120s, 240s, 480s (max 8 min between retries)
    Other errors: 2s, 4s, 8s backoff
    """
    model_full = MODEL_ALIAS_MAP['opus']

    # Backoff times for rate limits (in seconds)
    rate_limit_backoffs = [30, 60, 120, 240, 480, 480, 480, 480, 480, 480]

    for attempt in range(max_retries):
        try:
            start_time = time.time()
            response = model_completion(prompt,
                                        model=model_full,
                                        system=system,
                                        temperature=0.0,
                                        max_tokens=max_tokens)
            return {
                'response': response,
                'solve_time': time.time() - start_time,
                'success': True,
                'attempts': attempt + 1,
            }
        except Exception as e:
            error_msg = str(e)

            # Check for rate limiting - use exponential backoff
            if check_for_rate_limit(error_msg):
                if attempt < max_retries - 1:
                    wait_time = rate_limit_backoffs[min(
                        attempt,
                        len(rate_limit_backoffs) - 1)]
                    print(
                        f"\n  ⚠️  Rate limit (attempt {attempt + 1}/{max_retries}). Waiting {wait_time}s..."
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    # Max retries exceeded for rate limit
                    print(
                        f"\n  ❌ Rate limit: max retries ({max_retries}) exceeded"
                    )
                    return {
                        'response': None,
                        'solve_time': 0,
                        'success': False,
                        'error': f'Rate limit max retries: {error_msg}',
                        'attempts': attempt + 1,
                    }

            # Other errors - retry with shorter backoff
            if attempt < max_retries - 1:
                wait_time = min(2**attempt,
                                30)  # Cap at 30s for non-rate-limit errors
                print(
                    f"  API error (attempt {attempt + 1}): {error_msg[:100]}... Retrying in {wait_time}s"
                )
                time.sleep(wait_time)
            else:
                return {
                    'response': None,
                    'solve_time': 0,
                    'success': False,
                    'error': error_msg,
                    'attempts': attempt + 1,
                }

    return {
        'response': None,
        'success': False,
        'error': 'Max retries exceeded'
    }


def run_opus_on_math(subject: str,
                     num_problems: int = None,
                     output_dir: Path = None,
                     resume: bool = False,
                     timestamp: str = None) -> dict:
    """Run Opus on MATH dataset with resume support."""
    dataset_name = f"math_{subject}"
    print(f"\n{'='*60}")
    print(f"Running Opus on MATH ({subject})")
    print(f"{'='*60}")

    ds = load_dataset('EleutherAI/hendrycks_math', subject)
    problems = list(ds['test'])
    if num_problems:
        problems = problems[:num_problems]

    total_problems = len(problems)
    print(f"Total problems: {total_problems}")

    # Check for partial results if resuming
    completed_indices = set()
    results = []
    correct = 0

    if resume and output_dir:
        partial_path, completed_indices = find_latest_partial_results(
            output_dir, dataset_name)
        if partial_path and completed_indices:
            partial_data = load_partial_results(partial_path)
            results = partial_data.get('results', [])
            correct = sum(1 for r in results if r.get('opus_correct', False))
            print(
                f"  Resuming from {len(completed_indices)} completed problems")

    system = """You are a mathematical problem solver. Show clear step-by-step reasoning.
Always put your final answer in \\boxed{} format at the end."""

    # Output path for incremental saves
    out_path = output_dir / f"opus_{dataset_name}_{timestamp}.json" if output_dir else None

    for idx, problem in enumerate(problems):
        # Skip already completed problems
        if idx in completed_indices:
            continue

        problem_text = problem['problem']
        gold_answer = extract_boxed_answer(problem['solution'])

        prompt = f"""Solve this math problem step by step. Put your final answer in \\boxed{{}}.

Problem: {problem_text}

Solution:"""

        result = call_opus_with_retry(prompt, system)

        extracted = extract_boxed_answer(result.get('response', ''))
        is_correct = normalize_answer(extracted) == normalize_answer(
            gold_answer)
        if is_correct:
            correct += 1

        results.append({
            'problem_idx': idx,
            'gold_answer': gold_answer,
            'opus_answer': extracted,
            'opus_correct': is_correct,
            'opus_time': result.get('solve_time', 0),
            'opus_success': result.get('success', False),
            'opus_error': result.get('error'),
        })

        status = "✓" if is_correct else "✗"
        completed = len(completed_indices) + len(
            [r for r in results if r['problem_idx'] not in completed_indices])
        acc = 100 * correct / completed if completed > 0 else 0
        print(f"\r[{completed}/{total_problems}] {status} Acc: {acc:.1f}%",
              end="",
              flush=True)

        # Save progress periodically
        if out_path and len(results) % SAVE_EVERY == 0:
            save_partial_results(out_path, dataset_name, results, correct,
                                 total_problems)

    print()

    # Final save
    final_results = {
        'dataset': dataset_name,
        'total': total_problems,
        'opus_correct': correct,
        'opus_accuracy': correct / len(results) if results else 0,
        'results': results,
    }

    if out_path:
        with open(out_path, 'w') as f:
            json.dump(final_results, f, indent=2)

    return final_results


def run_opus_on_gsm8k(num_problems: int = None,
                      output_dir: Path = None,
                      resume: bool = False,
                      timestamp: str = None) -> dict:
    """Run Opus on GSM8K dataset with resume support."""
    dataset_name = 'gsm8k'
    print(f"\n{'='*60}")
    print(f"Running Opus on GSM8K")
    print(f"{'='*60}")

    ds = load_dataset('openai/gsm8k', 'main')
    problems = list(ds['test'])
    if num_problems:
        problems = problems[:num_problems]

    total_problems = len(problems)
    print(f"Total problems: {total_problems}")

    # Check for partial results if resuming
    completed_indices = set()
    results = []
    correct = 0

    if resume and output_dir:
        partial_path, completed_indices = find_latest_partial_results(
            output_dir, dataset_name)
        if partial_path and completed_indices:
            partial_data = load_partial_results(partial_path)
            results = partial_data.get('results', [])
            correct = sum(1 for r in results if r.get('opus_correct', False))
            print(
                f"  Resuming from {len(completed_indices)} completed problems")

    system = """You are a math tutor helping with grade school math problems.
Show your work step by step and give the final numerical answer."""

    out_path = output_dir / f"opus_{dataset_name}_{timestamp}.json" if output_dir else None

    for idx, problem in enumerate(problems):
        if idx in completed_indices:
            continue

        question = problem['question']
        gold = extract_numerical_answer(problem['answer'])

        prompt = f"""Solve this math problem step by step:

{question}

Show your reasoning and give the final answer as a number."""

        result = call_opus_with_retry(prompt, system)

        extracted = extract_numerical_answer(result.get('response', ''))
        is_correct = str(extracted) == str(gold)
        if is_correct:
            correct += 1

        results.append({
            'problem_idx': idx,
            'gold_answer': gold,
            'opus_answer': extracted,
            'opus_correct': is_correct,
            'opus_time': result.get('solve_time', 0),
            'opus_success': result.get('success', False),
            'opus_error': result.get('error'),
        })

        status = "✓" if is_correct else "✗"
        completed = len(completed_indices) + len(
            [r for r in results if r['problem_idx'] not in completed_indices])
        acc = 100 * correct / completed if completed > 0 else 0
        print(f"\r[{completed}/{total_problems}] {status} Acc: {acc:.1f}%",
              end="",
              flush=True)

        if out_path and len(results) % SAVE_EVERY == 0:
            save_partial_results(out_path, dataset_name, results, correct,
                                 total_problems)

    print()

    final_results = {
        'dataset': dataset_name,
        'total': total_problems,
        'opus_correct': correct,
        'opus_accuracy': correct / len(results) if results else 0,
        'results': results,
    }

    if out_path:
        with open(out_path, 'w') as f:
            json.dump(final_results, f, indent=2)

    return final_results


def run_opus_on_gpqa(format_type: str = 'mc',
                     num_problems: int = None,
                     output_dir: Path = None,
                     resume: bool = False,
                     timestamp: str = None) -> dict:
    """Run Opus on GPQA dataset with resume support."""
    dataset_name = f'gpqa_{format_type}'
    print(f"\n{'='*60}")
    print(f"Running Opus on GPQA ({format_type})")
    print(f"{'='*60}")

    ds = load_dataset('Idavidrein/gpqa', 'gpqa_diamond')
    problems = list(ds['train'])
    if num_problems:
        problems = problems[:num_problems]

    total_problems = len(problems)
    print(f"Total problems: {total_problems}")

    # Check for partial results if resuming
    completed_indices = set()
    results = []
    correct = 0

    if resume and output_dir:
        partial_path, completed_indices = find_latest_partial_results(
            output_dir, dataset_name)
        if partial_path and completed_indices:
            partial_data = load_partial_results(partial_path)
            results = partial_data.get('results', [])
            correct = sum(1 for r in results if r.get('opus_correct', False))
            print(
                f"  Resuming from {len(completed_indices)} completed problems")

    if format_type == 'mc':
        system = """You are an expert scientist answering graduate-level questions.
Analyze the question carefully and select the best answer from the options."""
    else:
        system = """You are an expert scientist answering graduate-level questions.
Provide a clear, well-reasoned answer."""

    out_path = output_dir / f"opus_{dataset_name}_{timestamp}.json" if output_dir else None

    for idx, problem in enumerate(problems):
        if idx in completed_indices:
            continue

        question = problem['Question']
        correct_answer = problem['Correct Answer']

        if format_type == 'mc':
            choices = [
                problem['Correct Answer'],
                problem['Incorrect Answer 1'],
                problem['Incorrect Answer 2'],
                problem['Incorrect Answer 3'],
            ]
            import random
            random.seed(idx)
            random.shuffle(choices)
            correct_idx = choices.index(correct_answer)
            letters = ['A', 'B', 'C', 'D']

            options_text = "\n".join(
                [f"{letters[i]}. {c}" for i, c in enumerate(choices)])
            prompt = f"""Question: {question}

{options_text}

Which answer is correct? Reply with just the letter (A, B, C, or D)."""

            result = call_opus_with_retry(prompt, system, max_tokens=100)
            response = result.get('response') or ''

            # Extract letter answer
            match = re.search(r'\b([A-D])\b',
                              response.upper()) if response else None
            extracted = match.group(1) if match else None
            is_correct = extracted == letters[correct_idx]
        else:
            prompt = f"""Question: {question}

Provide a clear, detailed answer."""

            result = call_opus_with_retry(prompt, system, max_tokens=1000)
            response = result.get('response') or ''

            # For freeform, use simple string matching (imperfect but fast)
            is_correct = response and correct_answer.lower() in response.lower(
            )
            extracted = response[:200] if response else None

        if is_correct:
            correct += 1

        results.append({
            'problem_idx':
            idx,
            'gold_answer':
            correct_answer
            if format_type == 'freeform' else letters[correct_idx],
            'opus_answer':
            extracted,
            'opus_correct':
            is_correct,
            'opus_time':
            result.get('solve_time', 0),
            'opus_success':
            result.get('success', False),
            'opus_error':
            result.get('error'),
        })

        status = "✓" if is_correct else "✗"
        completed = len(completed_indices) + len(
            [r for r in results if r['problem_idx'] not in completed_indices])
        acc = 100 * correct / completed if completed > 0 else 0
        print(f"\r[{completed}/{total_problems}] {status} Acc: {acc:.1f}%",
              end="",
              flush=True)

        if out_path and len(results) % SAVE_EVERY == 0:
            save_partial_results(out_path, dataset_name, results, correct,
                                 total_problems)

    print()

    final_results = {
        'dataset': dataset_name,
        'total': total_problems,
        'opus_correct': correct,
        'opus_accuracy': correct / len(results) if results else 0,
        'results': results,
    }

    if out_path:
        with open(out_path, 'w') as f:
            json.dump(final_results, f, indent=2)

    return final_results


def run_opus_on_mbpp(num_problems: int = None,
                     output_dir: Path = None,
                     resume: bool = False,
                     timestamp: str = None) -> dict:
    """Run Opus on MBPP dataset with resume support."""
    dataset_name = 'mbpp'
    print(f"\n{'='*60}")
    print(f"Running Opus on MBPP")
    print(f"{'='*60}")

    ds = load_dataset('mbpp', 'sanitized')
    problems = list(ds['test'])
    if num_problems:
        problems = problems[:num_problems]

    total_problems = len(problems)
    print(f"Total problems: {total_problems}")

    # Check for partial results if resuming
    completed_indices = set()
    results = []
    correct = 0

    if resume and output_dir:
        partial_path, completed_indices = find_latest_partial_results(
            output_dir, dataset_name)
        if partial_path and completed_indices:
            partial_data = load_partial_results(partial_path)
            results = partial_data.get('results', [])
            correct = sum(1 for r in results if r.get('opus_correct', False))
            print(
                f"  Resuming from {len(completed_indices)} completed problems")

    system = """You are an expert Python programmer. Write clean, correct code.
Provide ONLY the function implementation, no explanations."""

    out_path = output_dir / f"opus_{dataset_name}_{timestamp}.json" if output_dir else None

    for idx, problem in enumerate(problems):
        if idx in completed_indices:
            continue

        prompt_text = problem['prompt']
        test_cases = problem['test_list']

        prompt = f"""{prompt_text}

Write the Python function. Provide only the code, no explanations."""

        result = call_opus_with_retry(prompt, system, max_tokens=1000)
        response = result.get('response', '')

        # Extract code
        code_match = re.search(r'```python\n(.*?)```', response, re.DOTALL)
        if code_match:
            code = code_match.group(1)
        else:
            code = response

        # Test the code
        is_correct = False
        try:
            exec_globals = {}
            exec(code, exec_globals)

            all_pass = True
            for test in test_cases:
                try:
                    exec(test, exec_globals)
                except:
                    all_pass = False
                    break
            is_correct = all_pass
        except:
            is_correct = False

        if is_correct:
            correct += 1

        results.append({
            'problem_idx': idx,
            'opus_correct': is_correct,
            'opus_time': result.get('solve_time', 0),
            'opus_success': result.get('success', False),
            'opus_error': result.get('error'),
        })

        status = "✓" if is_correct else "✗"
        completed = len(completed_indices) + len(
            [r for r in results if r['problem_idx'] not in completed_indices])
        acc = 100 * correct / completed if completed > 0 else 0
        print(f"\r[{completed}/{total_problems}] {status} Acc: {acc:.1f}%",
              end="",
              flush=True)

        if out_path and len(results) % SAVE_EVERY == 0:
            save_partial_results(out_path, dataset_name, results, correct,
                                 total_problems)

    print()

    final_results = {
        'dataset': dataset_name,
        'total': total_problems,
        'opus_correct': correct,
        'opus_accuracy': correct / len(results) if results else 0,
        'results': results,
    }

    if out_path:
        with open(out_path, 'w') as f:
            json.dump(final_results, f, indent=2)

    return final_results


def update_baseline_with_opus(baseline_path: str, opus_results: dict) -> dict:
    """Merge Opus results into existing baseline and recalculate difficulty."""
    with open(baseline_path) as f:
        baseline = json.load(f)

    # Create lookup for opus results
    opus_lookup = {r['problem_idx']: r for r in opus_results['results']}

    # Update each result
    updated_results = []
    for result in baseline['results']:
        idx = result['problem_idx']
        if idx in opus_lookup:
            opus_data = opus_lookup[idx]
            result['models']['opus'] = {
                'answer': opus_data.get('opus_answer'),
                'correct': opus_data.get('opus_correct', False),
                'solve_time': opus_data.get('opus_time', 0),
            }

        # Recalculate difficulty
        haiku_ok = result.get('models', {}).get('haiku',
                                                {}).get('correct', False)
        sonnet_ok = result.get('models', {}).get('sonnet',
                                                 {}).get('correct', False)
        opus_ok = result.get('models', {}).get('opus',
                                               {}).get('correct', False)

        if haiku_ok and sonnet_ok and opus_ok:
            result['difficulty'] = 'easy'
        elif not haiku_ok and (sonnet_ok or opus_ok):
            result['difficulty'] = 'medium'
        elif not haiku_ok and not sonnet_ok and opus_ok:
            result['difficulty'] = 'hard'
        else:
            result['difficulty'] = 'very_hard'

        updated_results.append(result)

    baseline['results'] = updated_results

    # Recalculate summary stats
    from collections import Counter
    difficulties = Counter(r['difficulty'] for r in updated_results)
    baseline['difficulty_counts'] = dict(difficulties)

    total = len(updated_results)
    baseline['model_accuracy'] = {
        'haiku':
        sum(1 for r in updated_results
            if r['models'].get('haiku', {}).get('correct', False)) / total,
        'sonnet':
        sum(1 for r in updated_results
            if r['models'].get('sonnet', {}).get('correct', False)) / total,
        'opus':
        sum(1 for r in updated_results
            if r['models'].get('opus', {}).get('correct', False)) / total,
    }

    return baseline


def main():
    parser = argparse.ArgumentParser(description='Rerun Opus on all datasets')
    parser.add_argument(
        '--dataset',
        type=str,
        default='all',
        choices=['all', 'math', 'gsm8k', 'gpqa_mc', 'gpqa_freeform', 'mbpp'],
        help='Dataset to run (default: all)')
    parser.add_argument(
        '--subject',
        type=str,
        default='all',
        choices=['all', 'algebra', 'geometry', 'number_theory'],
        help='MATH subject (only used with --dataset math)')
    parser.add_argument('--num-problems',
                        type=int,
                        default=None,
                        help='Limit number of problems (for testing)')
    parser.add_argument('--output-dir',
                        type=str,
                        default='results/opus_rerun',
                        help='Output directory')
    parser.add_argument('--resume',
                        action='store_true',
                        help='Resume from partial results in output directory')

    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("OPUS RERUN - All Datasets")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Timestamp: {timestamp}")
    print(f"Resume mode: {args.resume}")
    print("=" * 60)

    all_results = {}

    try:
        # MATH datasets
        if args.dataset in ['all', 'math']:
            subjects = ['algebra', 'geometry', 'number_theory'
                        ] if args.subject == 'all' else [args.subject]
            for subject in subjects:
                result = run_opus_on_math(subject,
                                          args.num_problems,
                                          output_dir=output_dir,
                                          resume=args.resume,
                                          timestamp=timestamp)
                all_results[f'math_{subject}'] = result
                print(
                    f"Saved: {output_dir}/opus_math_{subject}_{timestamp}.json"
                )

        # GSM8K
        if args.dataset in ['all', 'gsm8k']:
            result = run_opus_on_gsm8k(args.num_problems,
                                       output_dir=output_dir,
                                       resume=args.resume,
                                       timestamp=timestamp)
            all_results['gsm8k'] = result
            print(f"Saved: {output_dir}/opus_gsm8k_{timestamp}.json")

        # GPQA MC
        if args.dataset in ['all', 'gpqa_mc']:
            result = run_opus_on_gpqa('mc',
                                      args.num_problems,
                                      output_dir=output_dir,
                                      resume=args.resume,
                                      timestamp=timestamp)
            all_results['gpqa_mc'] = result
            print(f"Saved: {output_dir}/opus_gpqa_mc_{timestamp}.json")

        # GPQA Freeform
        if args.dataset in ['all', 'gpqa_freeform']:
            result = run_opus_on_gpqa('freeform',
                                      args.num_problems,
                                      output_dir=output_dir,
                                      resume=args.resume,
                                      timestamp=timestamp)
            all_results['gpqa_freeform'] = result
            print(f"Saved: {output_dir}/opus_gpqa_freeform_{timestamp}.json")

        # MBPP
        if args.dataset in ['all', 'mbpp']:
            result = run_opus_on_mbpp(args.num_problems,
                                      output_dir=output_dir,
                                      resume=args.resume,
                                      timestamp=timestamp)
            all_results['mbpp'] = result
            print(f"Saved: {output_dir}/opus_mbpp_{timestamp}.json")

        # Save combined results
        combined_path = output_dir / f"opus_all_datasets_{timestamp}.json"
        with open(combined_path, 'w') as f:
            json.dump(
                {
                    'timestamp': timestamp,
                    'datasets': all_results,
                    'summary': {
                        name: {
                            'total': r['total'],
                            'opus_correct': r['opus_correct'],
                            'opus_accuracy': r['opus_accuracy'],
                        }
                        for name, r in all_results.items()
                    }
                },
                f,
                indent=2)
        print(f"\nCombined results: {combined_path}")

        # Print summary
        print("\n" + "=" * 60)
        print("SUMMARY - Opus Accuracy")
        print("=" * 60)
        for name, result in all_results.items():
            acc = 100 * result['opus_accuracy']
            print(
                f"{name:<25}: {acc:>6.1f}% ({result['opus_correct']}/{result['total']})"
            )
        print("=" * 60)

    except Exception as e:
        print("\n" + "=" * 60)
        print("🚨 UNEXPECTED ERROR 🚨")
        print("=" * 60)
        print(f"Error: {e}")
        print("Progress has been saved. Run with --resume to continue.")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
