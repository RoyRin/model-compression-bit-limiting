#!/usr/bin/env python3
"""
Unified Best-of-N Experiment: Compare all 3 approaches with multiple trials.

Runs all approaches for generating multiple solutions and selecting the best:
1. Temperature Sampling: Generate N independent solutions with high temperature
2. Single Prompt: Ask for N different solutions in one prompt
3. Just Ask: Generate verbose solution, then N succinct rewrites

Each approach is run multiple times (default: 3) to compute std deviations.
All results are saved together in one folder for easy plotting.

Usage:
    # Run full experiment (3 trials, N=1,3,5,10, 90 problems)
    python run_best_of_n_unified.py

    # Quick test run
    python run_best_of_n_unified.py --num-problems 5 --num-trials 1

    # Plot from existing results
    python run_best_of_n_unified.py --plot-only --results-dir results/best_of_n_unified/best_of_n_unified_20260116_120000
"""

import sys
import os

# Add paths for imports
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
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import numpy as np
from datasets import load_dataset

from lossy_compression_tools import load_compression_model, compress_text
from utils.llm_api import anthropic_completion
from utils.api_cost_tracker import log_batch_spending
import anthropic

# Default parameters
DEFAULT_GENERATION_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_COMPRESSION_MODEL = "meta-llama/Llama-3.1-8B"
DEFAULT_N_VALUES = [1, 3, 5, 10]
DEFAULT_NUM_PROBLEMS = 90
DEFAULT_NUM_TRIALS = 3

MATH_SYSTEM_PROMPT = """You are a skilled mathematician solving AIME (American Invitational Mathematics Examination) problems.
Provide a clear, step-by-step solution and end with the numerical answer.
Format your final answer clearly using \\boxed{answer} notation.
AIME answers are always integers from 0 to 999."""

COMPRESS_SYSTEM_PROMPT = """You are an expert at condensing mathematical solutions to their essential core while preserving the reasoning needed to derive the answer."""

# Model ID mapping for batch API
BATCH_MODEL_IDS = {
    'claude-haiku-4-5-20251001': 'claude-haiku-4-5-20251001',
    'claude-3-5-haiku-20241022': 'claude-3-5-haiku-20241022',
    'claude-3-5-sonnet-20241022': 'claude-3-5-sonnet-20241022',
    'claude-3-opus-20240229': 'claude-3-opus-20240229',
}

# =============================================================================
# BATCH API HELPERS
# =============================================================================


def submit_batch_and_wait(client: anthropic.Anthropic,
                          requests: List[Dict],
                          description: str = "batch") -> Dict[str, str]:
    """Submit batch requests and wait for completion.

    Args:
        client: Anthropic client
        requests: List of batch request dicts with custom_id and params
        description: Description for logging

    Returns:
        Dict mapping custom_id -> response text
    """
    if not requests:
        return {}

    print(f"    Submitting {len(requests)} {description} requests...")

    batch = client.messages.batches.create(requests=requests)
    batch_id = batch.id

    # Get model from first request for spending tracking
    model = requests[0]['params'].get('model', 'unknown')

    # Poll for completion
    while True:
        status = client.messages.batches.retrieve(batch_id)

        if status.processing_status == 'ended':
            counts = status.request_counts
            print(
                f"    Batch complete: {counts.succeeded} succeeded, {counts.errored} errors"
            )
            break

        time.sleep(5)

    # Retrieve results and track spending
    all_results = list(client.messages.batches.results(batch_id))

    # Log spending for batch API calls
    log_batch_spending(model, all_results, description)

    # Extract response text from successful results
    results = {}
    for result in all_results:
        if result.result.type == 'succeeded':
            content = result.result.message.content[0].text
            results[result.custom_id] = content

    return results


def run_temperature_sampling_batch(
    problem_text: str,
    correct_answer: str,
    n: int,
    model: str,
    compression_model,
    tokenizer,
    client: anthropic.Anthropic,
    problem_idx: int = 0,
    temperature: float = 0.8,
    max_tokens: int = 2000,
) -> Dict[str, Any]:
    """Generate N solutions using temperature sampling with batch API."""

    prompt = f"""Solve this AIME problem:

{problem_text}

Please provide a complete solution with clear mathematical reasoning, and state the final numerical answer using \\boxed{{answer}} notation."""

    model_id = BATCH_MODEL_IDS.get(model, model)

    # Create batch requests for all N solutions
    requests = []
    for i in range(n):
        requests.append({
            "custom_id": f"temp_{problem_idx}_{i}",
            "params": {
                "model": model_id,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": MATH_SYSTEM_PROMPT,
                "messages": [{
                    "role": "user",
                    "content": prompt
                }],
            }
        })

    start_time = time.time()

    # Submit batch and wait
    results = submit_batch_and_wait(client, requests, f"temperature N={n}")

    # Process results
    solutions = []
    for i in range(n):
        custom_id = f"temp_{problem_idx}_{i}"
        if custom_id in results:
            response = results[custom_id]
            answer = extract_numerical_answer(response)
            compression = compress_solution(response, compression_model,
                                            tokenizer)

            solutions.append({
                'response': response,
                'answer': answer,
                'is_correct': str(answer) == str(correct_answer),
                'length': len(response),
                **compression,
            })
        else:
            solutions.append({
                'response': '',
                'answer': None,
                'is_correct': False,
                'length': 0,
                'error': 'batch_failed',
            })

    total_time = time.time() - start_time

    # Select best by compression
    valid_solutions = [
        s for s in solutions if s.get('compression_pct', 100) > 0
    ]
    if valid_solutions:
        best_idx = min(range(len(valid_solutions)),
                       key=lambda i: valid_solutions[i]['compression_pct'])
        best_solution = valid_solutions[best_idx]
    else:
        best_solution = solutions[0] if solutions else {}

    return {
        'n': n,
        'total_time': total_time,
        'solutions': solutions,
        'best_by_compression': best_solution,
        'best_compression_pct': best_solution.get('compression_pct', 0),
        'best_is_correct': best_solution.get('is_correct', False),
    }


def run_just_ask_batch(
    problem_text: str,
    correct_answer: str,
    n: int,
    model: str,
    compression_model,
    tokenizer,
    client: anthropic.Anthropic,
    problem_idx: int = 0,
    max_tokens_initial: int = 2000,
    max_tokens_compress: int = 1000,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """Generate verbose solution, then N succinct rewrites using batch API for rewrites."""

    # Step 1: Generate verbose solution (single call, not batched)
    initial_prompt = f"""Solve this AIME problem:

{problem_text}

Please provide a complete solution with clear mathematical reasoning, and state the final numerical answer using \\boxed{{answer}} notation."""

    start_time = time.time()

    try:
        initial_response = anthropic_completion(
            prompt=initial_prompt,
            model=model,
            system=MATH_SYSTEM_PROMPT,
            max_tokens=max_tokens_initial,
            temperature=0.3,
        )
        initial_answer = extract_numerical_answer(initial_response)
        initial_compression = compress_solution(initial_response,
                                                compression_model, tokenizer)

    except Exception as e:
        return {
            'n': n,
            'error': str(e),
            'best_compression_pct': 0,
            'best_is_correct': False,
        }

    # Step 2: Strip answer and batch N succinct rewrites
    stripped_solution = strip_final_answer(initial_response)

    compress_prompt = f"""Here is a solution to an AIME problem, but with the final answer hidden:

{stripped_solution}

Rewrite this solution as SUCCINCTLY as possible while preserving enough reasoning that someone could derive the final numerical answer. Focus on the key mathematical insights. Be brief but complete."""

    model_id = BATCH_MODEL_IDS.get(model, model)

    # Create batch requests for all N rewrites
    requests = []
    for i in range(n):
        requests.append({
            "custom_id": f"rewrite_{problem_idx}_{i}",
            "params": {
                "model": model_id,
                "max_tokens": max_tokens_compress,
                "temperature": temperature,
                "system": COMPRESS_SYSTEM_PROMPT,
                "messages": [{
                    "role": "user",
                    "content": compress_prompt
                }],
            }
        })

    # Submit batch and wait
    results = submit_batch_and_wait(client, requests, f"just_ask N={n}")

    # Process results
    rewrites = []
    for i in range(n):
        custom_id = f"rewrite_{problem_idx}_{i}"
        if custom_id in results:
            rewrite = results[custom_id]
            compression = compress_solution(rewrite, compression_model,
                                            tokenizer)

            rewrites.append({
                'response':
                rewrite,
                'length':
                len(rewrite),
                'is_correct':
                str(initial_answer) == str(correct_answer),
                **compression,
            })
        else:
            rewrites.append({
                'response': '',
                'length': 0,
                'error': 'batch_failed',
            })

    total_time = time.time() - start_time

    # Select best by compression
    valid_rewrites = [r for r in rewrites if r.get('compression_pct', 100) > 0]
    if valid_rewrites:
        best_idx = min(range(len(valid_rewrites)),
                       key=lambda i: valid_rewrites[i]['compression_pct'])
        best_rewrite = valid_rewrites[best_idx]
    else:
        best_rewrite = rewrites[0] if rewrites else {}

    return {
        'n': n,
        'total_time': total_time,
        'initial_response': initial_response,
        'initial_answer': initial_answer,
        'initial_compression_pct':
        initial_compression.get('compression_pct', 0),
        'rewrites': rewrites,
        'best_by_compression': best_rewrite,
        'best_compression_pct': best_rewrite.get('compression_pct', 0),
        'best_is_correct': str(initial_answer) == str(correct_answer),
    }


def extract_numerical_answer(response_text: str) -> Optional[str]:
    """Extract numerical answer from model response."""
    if not response_text:
        return None

    # Look for boxed answers first
    boxed_pattern = r'\\boxed\{([^}]+)\}'
    boxed_match = re.search(boxed_pattern, response_text)
    if boxed_match:
        return boxed_match.group(1).strip()

    # Look for explicit answer patterns
    answer_patterns = [
        r'(?:the\s+)?answer\s+is:?\s*(\d+)',
        r'(?:final\s+)?answer:?\s*(\d+)',
        r'therefore:?\s*(\d+)',
        r'=\s*(\d+)\s*(?:$|\n)',
    ]

    for pattern in answer_patterns:
        match = re.search(pattern, response_text.lower())
        if match:
            return match.group(1)

    # Look for the last standalone number
    numbers = re.findall(r'\b(\d+)\b', response_text)
    if numbers:
        return numbers[-1]

    return None


def strip_final_answer(response_text: str) -> str:
    """Strip the final answer from a solution, keeping the reasoning."""
    text = re.sub(r'\\boxed\{[^}]+\}', '\\boxed{???}', response_text)
    return text.strip()


def compress_solution(text: str, model, tokenizer) -> Dict[str, Any]:
    """Compress a solution and return metrics."""
    try:
        result = compress_text(text, model, tokenizer, return_metrics=True)
        return {
            'original_bytes': result['original_bytes'],
            'compressed_bytes': result['compressed_bytes'],
            'compression_pct': result['compression_ratio'] * 100,
            'n_tokens': result.get('n_tokens', 0),
            'bits_per_token': result.get('bits_per_token', 0),
        }
    except Exception as e:
        return {
            'original_bytes': len(text.encode()),
            'compressed_bytes': 0,
            'compression_pct': 0,
            'n_tokens': 0,
            'bits_per_token': 0,
            'error': str(e),
        }


# =============================================================================
# APPROACH 1: Temperature Sampling
# =============================================================================


def run_temperature_sampling(
    problem_text: str,
    correct_answer: str,
    n: int,
    model: str,
    compression_model,
    tokenizer,
    temperature: float = 0.8,
    max_tokens: int = 2000,
) -> Dict[str, Any]:
    """Generate N solutions using temperature sampling."""

    prompt = f"""Solve this AIME problem:

{problem_text}

Please provide a complete solution with clear mathematical reasoning, and state the final numerical answer using \\boxed{{answer}} notation."""

    solutions = []
    start_time = time.time()

    for i in range(n):
        try:
            response = anthropic_completion(
                prompt=prompt,
                model=model,
                system=MATH_SYSTEM_PROMPT,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            answer = extract_numerical_answer(response)
            compression = compress_solution(response, compression_model,
                                            tokenizer)

            solutions.append({
                'response': response,
                'answer': answer,
                'is_correct': str(answer) == str(correct_answer),
                'length': len(response),
                **compression,
            })
        except Exception as e:
            solutions.append({
                'response': '',
                'answer': None,
                'is_correct': False,
                'length': 0,
                'error': str(e),
            })

    total_time = time.time() - start_time

    # Select best by compression
    valid_solutions = [
        s for s in solutions if s.get('compression_pct', 100) > 0
    ]
    if valid_solutions:
        best_idx = min(range(len(valid_solutions)),
                       key=lambda i: valid_solutions[i]['compression_pct'])
        best_solution = valid_solutions[best_idx]
    else:
        best_solution = solutions[0] if solutions else {}

    return {
        'n': n,
        'total_time': total_time,
        'solutions': solutions,
        'best_by_compression': best_solution,
        'best_compression_pct': best_solution.get('compression_pct', 0),
        'best_is_correct': best_solution.get('is_correct', False),
    }


# =============================================================================
# APPROACH 2: Single Prompt
# =============================================================================


def run_single_prompt(
    problem_text: str,
    correct_answer: str,
    n: int,
    model: str,
    compression_model,
    tokenizer,
    max_tokens: int = 4000,
) -> Dict[str, Any]:
    """Ask for N solutions in a single prompt."""

    prompt = f"""Solve this AIME problem in {n} DIFFERENT ways. For each approach, use a different method or perspective.

Problem: {problem_text}

Provide {n} complete solutions, each labeled as "Solution 1:", "Solution 2:", etc.
For each solution, end with the answer in \\boxed{{answer}} notation.
Try genuinely different approaches - don't just rephrase the same method."""

    start_time = time.time()

    try:
        response = anthropic_completion(
            prompt=prompt,
            model=model,
            system=MATH_SYSTEM_PROMPT,
            max_tokens=max_tokens,
            temperature=0.3,
        )

        # Split into individual solutions
        solution_pattern = r'(?:Solution|Approach|Method)\s*(\d+)[:\s]'
        parts = re.split(solution_pattern, response, flags=re.IGNORECASE)

        solutions = []
        i = 1
        while i < len(parts):
            if i + 1 < len(parts):
                sol_text = parts[i + 1].strip()
                answer = extract_numerical_answer(sol_text)
                compression = compress_solution(sol_text, compression_model,
                                                tokenizer)

                solutions.append({
                    'response':
                    sol_text,
                    'answer':
                    answer,
                    'is_correct':
                    str(answer) == str(correct_answer),
                    'length':
                    len(sol_text),
                    **compression,
                })
            i += 2

        # If splitting failed, treat whole response as one solution
        if not solutions:
            answer = extract_numerical_answer(response)
            compression = compress_solution(response, compression_model,
                                            tokenizer)
            solutions.append({
                'response': response,
                'answer': answer,
                'is_correct': str(answer) == str(correct_answer),
                'length': len(response),
                **compression,
            })

    except Exception as e:
        solutions = [{
            'response': '',
            'answer': None,
            'is_correct': False,
            'error': str(e),
        }]

    total_time = time.time() - start_time

    # Select best by compression
    valid_solutions = [
        s for s in solutions if s.get('compression_pct', 100) > 0
    ]
    if valid_solutions:
        best_idx = min(range(len(valid_solutions)),
                       key=lambda i: valid_solutions[i]['compression_pct'])
        best_solution = valid_solutions[best_idx]
    else:
        best_solution = solutions[0] if solutions else {}

    return {
        'n': n,
        'n_actual': len(solutions),
        'total_time': total_time,
        'solutions': solutions,
        'best_by_compression': best_solution,
        'best_compression_pct': best_solution.get('compression_pct', 0),
        'best_is_correct': best_solution.get('is_correct', False),
    }


# =============================================================================
# APPROACH 3: Just Ask (Generate verbose, then N succinct rewrites)
# =============================================================================


def run_just_ask(
    problem_text: str,
    correct_answer: str,
    n: int,
    model: str,
    compression_model,
    tokenizer,
    max_tokens_initial: int = 2000,
    max_tokens_compress: int = 1000,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """Generate verbose solution, then N succinct rewrites."""

    # Step 1: Generate verbose solution
    initial_prompt = f"""Solve this AIME problem:

{problem_text}

Please provide a complete solution with clear mathematical reasoning, and state the final numerical answer using \\boxed{{answer}} notation."""

    start_time = time.time()

    try:
        initial_response = anthropic_completion(
            prompt=initial_prompt,
            model=model,
            system=MATH_SYSTEM_PROMPT,
            max_tokens=max_tokens_initial,
            temperature=0.3,
        )
        initial_answer = extract_numerical_answer(initial_response)
        initial_compression = compress_solution(initial_response,
                                                compression_model, tokenizer)

    except Exception as e:
        return {
            'n': n,
            'error': str(e),
            'best_compression_pct': 0,
            'best_is_correct': False,
        }

    # Step 2: Strip answer and generate N succinct rewrites
    stripped_solution = strip_final_answer(initial_response)

    compress_prompt = f"""Here is a solution to an AIME problem, but with the final answer hidden:

{stripped_solution}

Rewrite this solution as SUCCINCTLY as possible while preserving enough reasoning that someone could derive the final numerical answer. Focus on the key mathematical insights. Be brief but complete."""

    rewrites = []
    for i in range(n):
        try:
            rewrite = anthropic_completion(
                prompt=compress_prompt,
                model=model,
                system=COMPRESS_SYSTEM_PROMPT,
                max_tokens=max_tokens_compress,
                temperature=temperature,
            )

            compression = compress_solution(rewrite, compression_model,
                                            tokenizer)

            rewrites.append({
                'response':
                rewrite,
                'length':
                len(rewrite),
                'is_correct':
                str(initial_answer) == str(
                    correct_answer),  # Use initial answer
                **compression,
            })
        except Exception as e:
            rewrites.append({
                'response': '',
                'length': 0,
                'error': str(e),
            })

    total_time = time.time() - start_time

    # Select best by compression
    valid_rewrites = [r for r in rewrites if r.get('compression_pct', 100) > 0]
    if valid_rewrites:
        best_idx = min(range(len(valid_rewrites)),
                       key=lambda i: valid_rewrites[i]['compression_pct'])
        best_rewrite = valid_rewrites[best_idx]
    else:
        best_rewrite = rewrites[0] if rewrites else {}

    return {
        'n': n,
        'total_time': total_time,
        'initial_response': initial_response,
        'initial_answer': initial_answer,
        'initial_compression_pct':
        initial_compression.get('compression_pct', 0),
        'rewrites': rewrites,
        'best_by_compression': best_rewrite,
        'best_compression_pct': best_rewrite.get('compression_pct', 0),
        'best_is_correct': str(initial_answer) == str(correct_answer),
    }


# =============================================================================
# Main Experiment Runner
# =============================================================================


def run_unified_experiment(
    num_problems: int = DEFAULT_NUM_PROBLEMS,
    n_values: List[int] = None,
    num_trials: int = DEFAULT_NUM_TRIALS,
    generation_model: str = DEFAULT_GENERATION_MODEL,
    compression_model_name: str = DEFAULT_COMPRESSION_MODEL,
    output_dir: str = None,
    verbose: bool = False,
    use_batch: bool = False,
) -> Dict[str, Any]:
    """Run the unified experiment with all 3 approaches.

    Args:
        use_batch: If True, use batch API for temperature sampling and just_ask rewrites.
                   This is ~10x faster but requires more setup.
    """

    if n_values is None:
        n_values = DEFAULT_N_VALUES

    # Setup output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if output_dir is None:
        output_dir = f"results/best_of_n_unified/best_of_n_unified_{timestamp}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Initialize batch client if needed
    batch_client = None
    if use_batch:
        print("Batch mode enabled - using Anthropic Message Batches API")
        batch_client = anthropic.Anthropic()

    # Load compression model
    print(f"Loading compression model: {compression_model_name}")
    compression_model, tokenizer = load_compression_model(
        compression_model_name)

    # Load AIME dataset (using same source as working scripts)
    print("Loading AIME dataset...")
    ds = load_dataset("AI-MO/aimo-validation-aime")
    problems = list(ds['train'])[:num_problems]
    print(f"Loaded {len(problems)} problems")

    # Initialize results structure
    all_results = {
        'parameters': {
            'num_problems': num_problems,
            'n_values': n_values,
            'num_trials': num_trials,
            'generation_model': generation_model,
            'compression_model': compression_model_name,
            'timestamp': timestamp,
        },
        'trials': [],
    }

    # Run trials
    for trial in range(num_trials):
        print(f"\n{'='*60}")
        print(f"TRIAL {trial + 1}/{num_trials}")
        print(f"{'='*60}")

        trial_results = {
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
            problem_text = problem['problem']
            correct_answer = str(problem['answer'])

            print(f"\n[{prob_idx + 1}/{num_problems}] Problem {prob_idx}")

            for n in n_values:
                print(f"  N={n}:", end=" ", flush=True)

                # Temperature sampling
                print("temp", end="", flush=True)
                if use_batch and batch_client:
                    temp_result = run_temperature_sampling_batch(
                        problem_text,
                        correct_answer,
                        n,
                        generation_model,
                        compression_model,
                        tokenizer,
                        client=batch_client,
                        problem_idx=prob_idx)
                else:
                    temp_result = run_temperature_sampling(
                        problem_text, correct_answer, n, generation_model,
                        compression_model, tokenizer)
                trial_results['temperature']['by_n'][n].append({
                    'problem_idx':
                    prob_idx,
                    'correct_answer':
                    correct_answer,
                    'compression_pct':
                    temp_result['best_compression_pct'],
                    'is_correct':
                    temp_result['best_is_correct'],
                    'time':
                    temp_result['total_time'],
                })

                # Single prompt (no batch version - it's already a single request)
                print(", single", end="", flush=True)
                single_result = run_single_prompt(problem_text, correct_answer,
                                                  n, generation_model,
                                                  compression_model, tokenizer)
                trial_results['single_prompt']['by_n'][n].append({
                    'problem_idx':
                    prob_idx,
                    'correct_answer':
                    correct_answer,
                    'compression_pct':
                    single_result['best_compression_pct'],
                    'is_correct':
                    single_result['best_is_correct'],
                    'time':
                    single_result['total_time'],
                })

                # Just ask
                print(", just_ask", end="", flush=True)
                if use_batch and batch_client:
                    just_ask_result = run_just_ask_batch(problem_text,
                                                         correct_answer,
                                                         n,
                                                         generation_model,
                                                         compression_model,
                                                         tokenizer,
                                                         client=batch_client,
                                                         problem_idx=prob_idx)
                else:
                    just_ask_result = run_just_ask(problem_text,
                                                   correct_answer, n,
                                                   generation_model,
                                                   compression_model,
                                                   tokenizer)
                trial_results['just_ask']['by_n'][n].append({
                    'problem_idx':
                    prob_idx,
                    'correct_answer':
                    correct_answer,
                    'compression_pct':
                    just_ask_result['best_compression_pct'],
                    'verbose_compression_pct':
                    just_ask_result.get('initial_compression_pct', 0),
                    'is_correct':
                    just_ask_result['best_is_correct'],
                    'time':
                    just_ask_result['total_time'],
                })

                print(" done", flush=True)

        all_results['trials'].append(trial_results)

        # Save intermediate results after each trial
        intermediate_path = Path(output_dir) / f"trial_{trial + 1}.json"
        with open(intermediate_path, 'w') as f:
            json.dump(trial_results, f, indent=2)
        print(f"Saved trial {trial + 1} to: {intermediate_path}")

    # Compute aggregated summary across trials
    all_results['summary'] = compute_summary(all_results)

    # Save final results
    final_path = Path(output_dir) / "all_results.json"
    with open(final_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFinal results saved to: {final_path}")

    return all_results


def compute_summary(all_results: Dict) -> Dict[str, Any]:
    """Compute summary statistics across all trials."""

    n_values = all_results['parameters']['n_values']
    num_trials = len(all_results['trials'])

    summary = {}

    for approach in ['temperature', 'single_prompt', 'just_ask']:
        summary[approach] = {}

        for n in n_values:
            # Collect compression_pct and accuracy across trials
            trial_compression_means = []
            trial_accuracy_means = []

            for trial in all_results['trials']:
                problems = trial[approach]['by_n'][n]
                compression_pcts = [
                    p['compression_pct'] for p in problems
                    if p.get('compression_pct', 0) > 0
                ]
                accuracies = [1 if p['is_correct'] else 0 for p in problems]

                if compression_pcts:
                    trial_compression_means.append(np.mean(compression_pcts))
                if accuracies:
                    trial_accuracy_means.append(np.mean(accuracies) * 100)

            summary[approach][n] = {
                'compression_pct_mean':
                float(np.mean(trial_compression_means))
                if trial_compression_means else 0,
                'compression_pct_std':
                float(np.std(trial_compression_means))
                if len(trial_compression_means) > 1 else 0,
                'accuracy_mean':
                float(np.mean(trial_accuracy_means))
                if trial_accuracy_means else 0,
                'accuracy_std':
                float(np.std(trial_accuracy_means))
                if len(trial_accuracy_means) > 1 else 0,
                'num_trials':
                num_trials,
            }

            # For just_ask, also track verbose baseline
            if approach == 'just_ask':
                trial_verbose_means = []
                for trial in all_results['trials']:
                    problems = trial[approach]['by_n'][n]
                    verbose_pcts = [
                        p.get('verbose_compression_pct', 0) for p in problems
                        if p.get('verbose_compression_pct', 0) > 0
                    ]
                    if verbose_pcts:
                        trial_verbose_means.append(np.mean(verbose_pcts))

                summary[approach][n]['verbose_compression_pct_mean'] = float(
                    np.mean(trial_verbose_means)) if trial_verbose_means else 0
                summary[approach][n]['verbose_compression_pct_std'] = float(
                    np.std(trial_verbose_means)) if len(
                        trial_verbose_means) > 1 else 0

    return summary


# =============================================================================
# Plotting
# =============================================================================


def plot_compression_comparison(results_dir: str, output_format: str = 'pdf'):
    """Generate compression comparison plot from saved results."""
    import matplotlib.pyplot as plt

    results_path = Path(results_dir) / "all_results.json"
    with open(results_path) as f:
        data = json.load(f)

    summary = data['summary']
    n_values = data['parameters']['n_values']
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        'temperature': '#9b59b6',  # purple
        'single_prompt': '#f39c12',  # orange
        'just_ask': '#2ecc71',  # green
    }

    labels = {
        'temperature': 'Temperature Sampling (T=0.8)',
        'single_prompt': 'Single Prompt ("Give N solutions")',
        'just_ask': 'Just Ask ("Write succinctly")',
    }

    for approach in ['temperature', 'single_prompt', 'just_ask']:
        means = []
        stds = []

        for n in n_values:
            stats = summary[approach].get(n, summary[approach].get(str(n), {}))
            means.append(stats.get('compression_pct_mean', 0))
            stds.append(stats.get('compression_pct_std', 0))

        means = np.array(means)
        stds = np.array(stds)

        ax.plot(n_values,
                means,
                'o-',
                color=colors[approach],
                label=labels[approach],
                linewidth=2,
                markersize=8)
        ax.fill_between(n_values,
                        means - stds,
                        means + stds,
                        color=colors[approach],
                        alpha=0.2)

    # Also plot verbose baseline for just_ask
    verbose_means = []
    verbose_stds = []
    for n in n_values:
        stats = summary['just_ask'].get(n, summary['just_ask'].get(str(n), {}))
        verbose_means.append(stats.get('verbose_compression_pct_mean', 0))
        verbose_stds.append(stats.get('verbose_compression_pct_std', 0))

    verbose_means = np.array(verbose_means)
    verbose_stds = np.array(verbose_stds)

    ax.plot(n_values,
            verbose_means,
            's--',
            color=colors['just_ask'],
            label='Just Ask (before rewrite)',
            linewidth=1.5,
            markersize=6,
            alpha=0.6)
    ax.fill_between(n_values,
                    verbose_means - verbose_stds,
                    verbose_means + verbose_stds,
                    color=colors['just_ask'],
                    alpha=0.1)

    ax.set_xlabel('N (Number of Solutions)', fontsize=12)
    ax.set_ylabel('Compression % (lower = more compressible)', fontsize=12)
    ax.set_title(
        'Compression Ratio vs N: Comparing Generation Strategies\n(with std dev across trials)',
        fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(n_values)

    plt.tight_layout()

    output_path = Path(
        results_dir
    ) / f"compression_vs_n_comparison_{timestamp}.{output_format}"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {output_path}")

    return fig


def print_summary_table(results_dir: str):
    """Print summary table from saved results."""

    results_path = Path(results_dir) / "all_results.json"
    with open(results_path) as f:
        data = json.load(f)

    summary = data['summary']
    n_values = data['parameters']['n_values']

    print("\n" + "=" * 80)
    print("COMPRESSION % vs N (lower = more compressible)")
    print("=" * 80)
    print(
        f"{'N':>3} | {'Temperature':>18} | {'Single Prompt':>18} | {'Just Ask':>18} | {'JA Verbose':>18}"
    )
    print("-" * 80)

    for n in n_values:
        temp = summary['temperature'].get(
            n, summary['temperature'].get(str(n), {}))
        single = summary['single_prompt'].get(
            n, summary['single_prompt'].get(str(n), {}))
        ja = summary['just_ask'].get(n, summary['just_ask'].get(str(n), {}))

        temp_str = f"{temp.get('compression_pct_mean', 0):.2f} +/- {temp.get('compression_pct_std', 0):.2f}"
        single_str = f"{single.get('compression_pct_mean', 0):.2f} +/- {single.get('compression_pct_std', 0):.2f}"
        ja_str = f"{ja.get('compression_pct_mean', 0):.2f} +/- {ja.get('compression_pct_std', 0):.2f}"
        verbose_str = f"{ja.get('verbose_compression_pct_mean', 0):.2f} +/- {ja.get('verbose_compression_pct_std', 0):.2f}"

        print(
            f"{n:>3} | {temp_str:>18} | {single_str:>18} | {ja_str:>18} | {verbose_str:>18}"
        )

    print("\n" + "=" * 80)
    print("ACCURACY % (selecting by best compression)")
    print("=" * 80)
    print(
        f"{'N':>3} | {'Temperature':>18} | {'Single Prompt':>18} | {'Just Ask':>18}"
    )
    print("-" * 80)

    for n in n_values:
        temp = summary['temperature'].get(
            n, summary['temperature'].get(str(n), {}))
        single = summary['single_prompt'].get(
            n, summary['single_prompt'].get(str(n), {}))
        ja = summary['just_ask'].get(n, summary['just_ask'].get(str(n), {}))

        temp_str = f"{temp.get('accuracy_mean', 0):.1f} +/- {temp.get('accuracy_std', 0):.1f}"
        single_str = f"{single.get('accuracy_mean', 0):.1f} +/- {single.get('accuracy_std', 0):.1f}"
        ja_str = f"{ja.get('accuracy_mean', 0):.1f} +/- {ja.get('accuracy_std', 0):.1f}"

        print(f"{n:>3} | {temp_str:>18} | {single_str:>18} | {ja_str:>18}")


def main():
    parser = argparse.ArgumentParser(
        description='Unified Best-of-N Experiment')

    parser.add_argument(
        '--num-problems',
        type=int,
        default=DEFAULT_NUM_PROBLEMS,
        help=f'Number of problems to test (default: {DEFAULT_NUM_PROBLEMS})')
    parser.add_argument(
        '--num-trials',
        type=int,
        default=DEFAULT_NUM_TRIALS,
        help=f'Number of trials for std dev (default: {DEFAULT_NUM_TRIALS})')
    parser.add_argument(
        '--n-values',
        type=int,
        nargs='+',
        default=DEFAULT_N_VALUES,
        help=f'Values of N to test (default: {DEFAULT_N_VALUES})')
    parser.add_argument(
        '--generation-model',
        type=str,
        default=DEFAULT_GENERATION_MODEL,
        help=f'Model for generation (default: {DEFAULT_GENERATION_MODEL})')
    parser.add_argument(
        '--compression-model',
        type=str,
        default=DEFAULT_COMPRESSION_MODEL,
        help=f'Model for compression (default: {DEFAULT_COMPRESSION_MODEL})')
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: auto-generated with timestamp)')
    parser.add_argument('--plot-only',
                        action='store_true',
                        help='Only generate plot from existing results')
    parser.add_argument('--results-dir',
                        type=str,
                        default=None,
                        help='Results directory for --plot-only mode')
    parser.add_argument('--format',
                        type=str,
                        choices=['pdf', 'png'],
                        default='pdf',
                        help='Output format for plots (default: pdf)')
    parser.add_argument('--verbose',
                        action='store_true',
                        help='Verbose output')
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Use batch API for temperature sampling and just_ask (10x faster)'
    )

    args = parser.parse_args()

    if args.plot_only:
        if not args.results_dir:
            # Find most recent results
            results_base = Path("results/best_of_n_unified")
            if results_base.exists():
                dirs = sorted(
                    [d for d in results_base.iterdir() if d.is_dir()],
                    reverse=True)
                if dirs:
                    args.results_dir = str(dirs[0])
                else:
                    print("No results found. Run experiment first.")
                    return
            else:
                print("No results directory found. Run experiment first.")
                return

        print(f"Generating plots from: {args.results_dir}")
        print_summary_table(args.results_dir)
        plot_compression_comparison(args.results_dir, args.format)
    else:
        # Run full experiment
        results = run_unified_experiment(
            num_problems=args.num_problems,
            n_values=args.n_values,
            num_trials=args.num_trials,
            generation_model=args.generation_model,
            compression_model_name=args.compression_model,
            output_dir=args.output_dir,
            verbose=args.verbose,
            use_batch=args.batch,
        )

        # Print summary and generate plot
        output_dir = args.output_dir or f"results/best_of_n_unified/best_of_n_unified_{results['parameters']['timestamp']}"
        print_summary_table(output_dir)
        plot_compression_comparison(output_dir, args.format)


if __name__ == "__main__":
    main()
