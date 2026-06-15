#!/usr/bin/env python3
"""
Best-of-N Approaches Comparison Experiment on AIME Problems.

Compares two different approaches for generating multiple solutions:
1. Temperature Sampling: Generate N responses with high temperature (independent calls)
2. Single-Prompt N-Responses: Ask model to give N different solutions in one prompt

Tests across multiple values of N and measures:
- Accuracy (does compression-based selection pick correct answer?)
- Agreement (how often do the N responses agree on the answer?)
- Timing (how long does each approach take?)
- Compression percentages
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
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import numpy as np
from datasets import load_dataset

from lossy_compression_tools import (
    load_compression_model,
    compress_text,
)
from utils.llm_api import anthropic_completion

# Default parameters
DEFAULT_GENERATION_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_COMPRESSION_MODEL = "meta-llama/Llama-3.1-8B"
DEFAULT_N_VALUES = [1, 3, 5, 10]
DEFAULT_NUM_PROBLEMS = 100

MATH_SYSTEM_PROMPT = """You are a skilled mathematician solving AIME (American Invitational Mathematics Examination) problems.
Provide a clear, step-by-step solution and end with the numerical answer.
Format your final answer clearly using \\boxed{answer} notation.
AIME answers are always integers from 0 to 999."""


def extract_numerical_answer(response_text: str) -> Optional[str]:
    """Extract numerical answer from model response."""
    # Look for boxed answers first (common in math solutions)
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

    # Look for the last standalone number in the text
    numbers = re.findall(r'\b(\d+)\b', response_text)
    if numbers:
        return numbers[-1]

    return None


def calculate_agreement_metrics(answers: List[str]) -> Dict[str, Any]:
    """Calculate agreement metrics for a list of answers."""
    if not answers:
        return {
            'agreement_rate': 0,
            'majority_answer': None,
            'majority_count': 0,
            'unique_answers': 0
        }

    # Filter out None answers
    valid_answers = [a for a in answers if a is not None]
    if not valid_answers:
        return {
            'agreement_rate': 0,
            'majority_answer': None,
            'majority_count': 0,
            'unique_answers': 0
        }

    counter = Counter(valid_answers)
    majority_answer, majority_count = counter.most_common(1)[0]

    return {
        'agreement_rate': majority_count / len(valid_answers),
        'majority_answer': majority_answer,
        'majority_count': majority_count,
        'unique_answers': len(counter),
        'answer_distribution': dict(counter),
        'total_valid': len(valid_answers),
    }


# =============================================================================
# APPROACH 1: Temperature Sampling
# =============================================================================


def generate_temperature_samples(
    problem_text: str,
    n_generations: int,
    model: str = DEFAULT_GENERATION_MODEL,
    max_tokens: int = 2000,
    temperature: float = 0.8,
    seed: int = 42,
    verbose: bool = False,
) -> Tuple[List[Dict[str, Any]], float]:
    """Generate N solutions using temperature sampling (independent calls).

    Returns:
        Tuple of (solutions list, total generation time)
    """

    prompt = f"""Solve this AIME problem:

{problem_text}

Please provide a complete solution with clear mathematical reasoning, and state the final numerical answer using \\boxed{{answer}} notation."""

    solutions = []
    total_time = 0

    for i in range(n_generations):
        print(f"  [Temp] Generating solution {i+1}/{n_generations}...")
        sys.stdout.flush()

        try:
            start_time = time.time()
            response = anthropic_completion(
                prompt=prompt,
                model=model,
                system=MATH_SYSTEM_PROMPT,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            gen_time = time.time() - start_time
            total_time += gen_time

            extracted = extract_numerical_answer(response)

            solutions.append({
                'index': i,
                'response': response,
                'extracted_answer': extracted,
                'generation_time': gen_time,
                'length': len(response),
            })

            print(
                f"    Answer: {extracted}, Length: {len(response)} chars, Time: {gen_time:.2f}s"
            )
            sys.stdout.flush()

        except Exception as e:
            print(f"    Error generating solution {i+1}: {e}")
            solutions.append({
                'index': i,
                'response': None,
                'extracted_answer': None,
                'error': str(e),
            })

    return solutions, total_time


# =============================================================================
# APPROACH 2: Single-Prompt N-Responses
# =============================================================================


def generate_single_prompt_n_responses(
    problem_text: str,
    n_responses: int,
    model: str = DEFAULT_GENERATION_MODEL,
    max_tokens: int = 8000,
    temperature: float = 0.3,
    verbose: bool = False,
) -> Tuple[List[Dict[str, Any]], float]:
    """Ask the model to generate N different solutions in a single prompt.

    Returns:
        Tuple of (solutions list, total generation time)
    """

    prompt = f"""Solve this AIME problem in {n_responses} DIFFERENT ways. For each solution, use a completely different approach or method.

Problem:
{problem_text}

Provide {n_responses} distinct solutions, each with different reasoning. Format each solution as:

=== SOLUTION 1 ===
[Your first approach]
Final Answer: \\boxed{{answer}}

=== SOLUTION 2 ===
[Your second approach, using a DIFFERENT method]
Final Answer: \\boxed{{answer}}

... and so on for all {n_responses} solutions.

Make sure each solution uses a genuinely different mathematical approach (e.g., algebraic vs geometric, direct calculation vs recursion, etc.)."""

    if verbose:
        print(
            f"  [SinglePrompt] Requesting {n_responses} solutions in one prompt..."
        )

    try:
        start_time = time.time()
        response = anthropic_completion(
            prompt=prompt,
            model=model,
            system=MATH_SYSTEM_PROMPT,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        total_time = time.time() - start_time

        if verbose:
            print(
                f"    Total response length: {len(response)} chars, Time: {total_time:.2f}s"
            )

        # Parse the N solutions from the response
        solutions = parse_n_solutions(response, n_responses, total_time,
                                      verbose)

        return solutions, total_time

    except Exception as e:
        print(f"    Error generating solutions: {e}")
        return [{
            'index': 0,
            'response': None,
            'extracted_answer': None,
            'error': str(e),
        }], 0


def parse_n_solutions(response: str,
                      expected_n: int,
                      total_time: float,
                      verbose: bool = False) -> List[Dict[str, Any]]:
    """Parse N solutions from a single response."""
    solutions = []

    # Split by solution markers
    parts = re.split(r'===\s*SOLUTION\s*(\d+)\s*===',
                     response,
                     flags=re.IGNORECASE)

    # parts[0] is text before first marker, then alternating: number, content, number, content...
    solution_texts = []
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            solution_texts.append(parts[i + 1].strip())

    # If we couldn't parse with markers, try splitting by "Solution X:" pattern
    if len(solution_texts) < 2:
        parts = re.split(r'\n\s*(?:Solution\s*)?(\d+)[.):]\s*',
                         response,
                         flags=re.IGNORECASE)
        solution_texts = []
        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                solution_texts.append(parts[i + 1].strip())

    # If still can't parse, treat whole response as one solution
    if len(solution_texts) < 1:
        solution_texts = [response]

    per_solution_time = total_time / max(len(solution_texts), 1)

    for i, text in enumerate(solution_texts):
        extracted = extract_numerical_answer(text)
        solutions.append({
            'index': i,
            'response': text,
            'extracted_answer': extracted,
            'generation_time': per_solution_time,
            'length': len(text),
        })

        if verbose:
            print(
                f"    Solution {i+1}: Answer={extracted}, Length={len(text)} chars"
            )

    # Pad with empty solutions if we didn't get enough
    while len(solutions) < expected_n:
        solutions.append({
            'index': len(solutions),
            'response': None,
            'extracted_answer': None,
            'error': 'Could not parse from response',
        })

    return solutions[:expected_n]


# =============================================================================
# Main Experiment Runner
# =============================================================================


def run_comparison_experiment(
    problems: List[Dict],
    n_values: List[int] = DEFAULT_N_VALUES,
    generation_model: str = DEFAULT_GENERATION_MODEL,
    compression_model_path: str = DEFAULT_COMPRESSION_MODEL,
    max_tokens: int = 2000,
    temperature: float = 0.8,
    seed: int = 42,
    output_dir: str = "results/best_of_n_approaches",
    verbose: bool = False,
    approaches: List[str] = None,
) -> Dict[str, Any]:
    """Run comparison experiment across different best-of-N approaches and N values."""

    if approaches is None:
        approaches = ['temperature', 'single_prompt']

    print(f"\n{'='*70}")
    print("Best-of-N Approaches Comparison Experiment")
    print(f"{'='*70}")
    print(f"Problems: {len(problems)}")
    print(f"N values: {n_values}")
    print(f"Approaches: {approaches}")
    print(f"Generation model: {generation_model}")
    print(f"Compression model: {compression_model_path}")
    print(f"{'='*70}\n")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load compression model once
    print("Loading compression model...")
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compression_model, compression_tokenizer = load_compression_model(
        model_path=compression_model_path,
        device=device,
    )
    print(f"Compression model loaded on {device}")
    print(f"\n{'='*70}")
    print("Starting experiment...")
    print(f"{'='*70}")
    import sys
    sys.stdout.flush()

    # Initialize results structure
    all_results = []

    # Summary structure: approach -> n_value -> metrics
    summary = {}
    for approach in approaches:
        summary[approach] = {}
        for n in n_values:
            summary[approach][n] = {
                'correct_by_compression': 0,
                'correct_by_random': 0,
                'correct_by_majority': 0,
                'total': 0,
                'compression_pcts': [],
                'agreement_rates': [],
                'generation_times': [],
            }

    max_n = max(n_values)

    for prob_idx, problem in enumerate(problems):
        problem_text = problem['problem']
        correct_answer = str(problem['answer'])
        problem_id = problem.get('id', f'problem_{prob_idx}')

        print(f"\n{'='*70}")
        print(f"Problem {prob_idx + 1}/{len(problems)} (ID: {problem_id})")
        print(f"{'='*70}")
        print(f"Problem: {problem_text[:200]}...")
        print(f"Correct answer: {correct_answer}")
        sys.stdout.flush()

        problem_results = {
            'problem_id': problem_id,
            'problem_text': problem_text,
            'correct_answer': correct_answer,
            'approaches': {},
        }

        # =====================================================================
        # Approach 1: Temperature Sampling
        # =====================================================================
        if 'temperature' in approaches:
            print(
                f"\n--- Approach 1: Temperature Sampling (generating {max_n} samples) ---"
            )
            sys.stdout.flush()

            # Generate max_n samples, then subsample for smaller N
            all_solutions, total_gen_time = generate_temperature_samples(
                problem_text=problem_text,
                n_generations=max_n,
                model=generation_model,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed + prob_idx * 1000,
                verbose=verbose,
            )

            # Compress all solutions once
            for sol in all_solutions:
                if sol.get('response') is not None:
                    _, pct, metrics = compress_text(
                        text=sol['response'],
                        compression_model=compression_model,
                        compression_tokenizer=compression_tokenizer,
                        bit_precision=64,
                        device=device,
                        verbose=False,
                    )
                    sol['compression_pct'] = pct
                    sol['original_bytes'] = metrics.get('original_bytes', 0)
                    sol['compressed_bytes'] = metrics.get(
                        'compressed_bytes', 0)
                    sol['compressed_bits'] = metrics.get('compressed_bits', 0)
                    sol['n_tokens'] = metrics.get('n_tokens', 0)
                    sol['bits_per_token'] = metrics.get('bits_per_token', 0)
                    sol['is_correct'] = sol[
                        'extracted_answer'] == correct_answer
                else:
                    sol['compression_pct'] = None
                    sol['original_bytes'] = None
                    sol['compressed_bytes'] = None
                    sol['compressed_bits'] = None
                    sol['n_tokens'] = None
                    sol['bits_per_token'] = None
                    sol['is_correct'] = False

            problem_results['approaches']['temperature'] = {'by_n': {}}

            # Analyze for each N value
            for n in n_values:
                subset = all_solutions[:n]
                valid_solutions = [
                    s for s in subset if s.get('response') is not None
                ]

                if not valid_solutions:
                    continue

                # Calculate cumulative time for first n samples
                cumulative_time = sum(
                    s.get('generation_time', 0) for s in subset)

                # Get all answers
                all_answers = [s['extracted_answer'] for s in valid_solutions]
                agreement = calculate_agreement_metrics(all_answers)

                # Selection strategies
                # 1. Best compression
                best_by_compression = min(
                    valid_solutions,
                    key=lambda x: x.get('compression_pct', float('inf')))

                # 2. Random (first one)
                random_solution = valid_solutions[0]

                # 3. Majority vote
                majority_answer = agreement['majority_answer']
                majority_correct = majority_answer == correct_answer

                n_results = {
                    'n':
                    n,
                    'n_valid':
                    len(valid_solutions),
                    'generation_time':
                    cumulative_time,
                    'all_answers':
                    all_answers,
                    'all_compression_pcts':
                    [s.get('compression_pct') for s in valid_solutions],
                    'all_original_bytes':
                    [s.get('original_bytes') for s in valid_solutions],
                    'all_compressed_bytes':
                    [s.get('compressed_bytes') for s in valid_solutions],
                    'all_n_tokens':
                    [s.get('n_tokens') for s in valid_solutions],
                    'all_bits_per_token':
                    [s.get('bits_per_token') for s in valid_solutions],
                    'all_text_lengths':
                    [s.get('length') for s in valid_solutions],
                    'agreement':
                    agreement,
                    'selection': {
                        'by_compression': {
                            'answer':
                            best_by_compression['extracted_answer'],
                            'is_correct':
                            best_by_compression['is_correct'],
                            'compression_pct':
                            best_by_compression['compression_pct'],
                            'original_bytes':
                            best_by_compression.get('original_bytes'),
                            'compressed_bytes':
                            best_by_compression.get('compressed_bytes'),
                            'n_tokens':
                            best_by_compression.get('n_tokens'),
                            'bits_per_token':
                            best_by_compression.get('bits_per_token'),
                            'text_length':
                            best_by_compression.get('length'),
                        },
                        'by_random': {
                            'answer':
                            random_solution['extracted_answer'],
                            'is_correct':
                            random_solution['is_correct'],
                            'compression_pct':
                            random_solution['compression_pct'],
                            'original_bytes':
                            random_solution.get('original_bytes'),
                            'compressed_bytes':
                            random_solution.get('compressed_bytes'),
                            'n_tokens':
                            random_solution.get('n_tokens'),
                            'bits_per_token':
                            random_solution.get('bits_per_token'),
                            'text_length':
                            random_solution.get('length'),
                        },
                        'by_majority': {
                            'answer': majority_answer,
                            'is_correct': majority_correct,
                        },
                    },
                }

                problem_results['approaches']['temperature']['by_n'][
                    n] = n_results

                # Update summary
                summary['temperature'][n]['total'] += 1
                if best_by_compression['is_correct']:
                    summary['temperature'][n]['correct_by_compression'] += 1
                if random_solution['is_correct']:
                    summary['temperature'][n]['correct_by_random'] += 1
                if majority_correct:
                    summary['temperature'][n]['correct_by_majority'] += 1
                summary['temperature'][n]['compression_pcts'].append(
                    best_by_compression['compression_pct'])
                summary['temperature'][n]['agreement_rates'].append(
                    agreement['agreement_rate'])
                summary['temperature'][n]['generation_times'].append(
                    cumulative_time)

                # Log compression metrics
                all_comp_pcts = [
                    s.get('compression_pct') for s in valid_solutions
                    if s.get('compression_pct') is not None
                ]
                all_bpt = [
                    s.get('bits_per_token') for s in valid_solutions
                    if s.get('bits_per_token') is not None
                ]
                avg_comp_pct = sum(all_comp_pcts) / len(
                    all_comp_pcts) if all_comp_pcts else 0
                min_comp_pct = min(all_comp_pcts) if all_comp_pcts else 0
                avg_bpt = sum(all_bpt) / len(all_bpt) if all_bpt else 0
                min_bpt = min(all_bpt) if all_bpt else 0

                print(
                    f"  N={n}: Compression={best_by_compression['extracted_answer']} ({'OK' if best_by_compression['is_correct'] else 'X'}), "
                    f"Majority={majority_answer} ({'OK' if majority_correct else 'X'}), "
                    f"Agreement={agreement['agreement_rate']:.0%}, Time={cumulative_time:.1f}s"
                )
                print(
                    f"    Compression%: min={min_comp_pct:.2f}%, avg={avg_comp_pct:.2f}% | "
                    f"Bits/token: min={min_bpt:.3f}, avg={avg_bpt:.3f}")
                sys.stdout.flush()

        # =====================================================================
        # Approach 2: Single-Prompt N-Responses
        # =====================================================================
        if 'single_prompt' in approaches:
            problem_results['approaches']['single_prompt'] = {'by_n': {}}

            for n in n_values:
                print(f"\n--- Approach 2: Single-Prompt (N={n}) ---")
                sys.stdout.flush()

                solutions, gen_time = generate_single_prompt_n_responses(
                    problem_text=problem_text,
                    n_responses=n,
                    model=generation_model,
                    max_tokens=max_tokens * n,
                    temperature=0.3,
                    verbose=verbose,
                )

                valid_solutions = [
                    s for s in solutions if s.get('response') is not None
                ]

                if not valid_solutions:
                    continue

                # Compress all solutions
                for sol in valid_solutions:
                    _, pct, metrics = compress_text(
                        text=sol['response'],
                        compression_model=compression_model,
                        compression_tokenizer=compression_tokenizer,
                        bit_precision=64,
                        device=device,
                        verbose=False,
                    )
                    sol['compression_pct'] = pct
                    sol['original_bytes'] = metrics.get('original_bytes', 0)
                    sol['compressed_bytes'] = metrics.get(
                        'compressed_bytes', 0)
                    sol['compressed_bits'] = metrics.get('compressed_bits', 0)
                    sol['n_tokens'] = metrics.get('n_tokens', 0)
                    sol['bits_per_token'] = metrics.get('bits_per_token', 0)
                    sol['is_correct'] = sol[
                        'extracted_answer'] == correct_answer

                # Get all answers
                all_answers = [s['extracted_answer'] for s in valid_solutions]
                agreement = calculate_agreement_metrics(all_answers)

                # Selection strategies
                best_by_compression = min(
                    valid_solutions,
                    key=lambda x: x.get('compression_pct', float('inf')))
                random_solution = valid_solutions[0]
                majority_answer = agreement['majority_answer']
                majority_correct = majority_answer == correct_answer

                n_results = {
                    'n':
                    n,
                    'n_valid':
                    len(valid_solutions),
                    'generation_time':
                    gen_time,
                    'all_answers':
                    all_answers,
                    'all_compression_pcts':
                    [s.get('compression_pct') for s in valid_solutions],
                    'all_original_bytes':
                    [s.get('original_bytes') for s in valid_solutions],
                    'all_compressed_bytes':
                    [s.get('compressed_bytes') for s in valid_solutions],
                    'all_n_tokens':
                    [s.get('n_tokens') for s in valid_solutions],
                    'all_bits_per_token':
                    [s.get('bits_per_token') for s in valid_solutions],
                    'all_text_lengths':
                    [s.get('length') for s in valid_solutions],
                    'agreement':
                    agreement,
                    'selection': {
                        'by_compression': {
                            'answer':
                            best_by_compression['extracted_answer'],
                            'is_correct':
                            best_by_compression['is_correct'],
                            'compression_pct':
                            best_by_compression['compression_pct'],
                            'original_bytes':
                            best_by_compression.get('original_bytes'),
                            'compressed_bytes':
                            best_by_compression.get('compressed_bytes'),
                            'n_tokens':
                            best_by_compression.get('n_tokens'),
                            'bits_per_token':
                            best_by_compression.get('bits_per_token'),
                            'text_length':
                            best_by_compression.get('length'),
                        },
                        'by_random': {
                            'answer':
                            random_solution['extracted_answer'],
                            'is_correct':
                            random_solution['is_correct'],
                            'compression_pct':
                            random_solution['compression_pct'],
                            'original_bytes':
                            random_solution.get('original_bytes'),
                            'compressed_bytes':
                            random_solution.get('compressed_bytes'),
                            'n_tokens':
                            random_solution.get('n_tokens'),
                            'bits_per_token':
                            random_solution.get('bits_per_token'),
                            'text_length':
                            random_solution.get('length'),
                        },
                        'by_majority': {
                            'answer': majority_answer,
                            'is_correct': majority_correct,
                        },
                    },
                }

                problem_results['approaches']['single_prompt']['by_n'][
                    n] = n_results

                # Update summary
                summary['single_prompt'][n]['total'] += 1
                if best_by_compression['is_correct']:
                    summary['single_prompt'][n]['correct_by_compression'] += 1
                if random_solution['is_correct']:
                    summary['single_prompt'][n]['correct_by_random'] += 1
                if majority_correct:
                    summary['single_prompt'][n]['correct_by_majority'] += 1
                summary['single_prompt'][n]['compression_pcts'].append(
                    best_by_compression['compression_pct'])
                summary['single_prompt'][n]['agreement_rates'].append(
                    agreement['agreement_rate'])
                summary['single_prompt'][n]['generation_times'].append(
                    gen_time)

                # Log compression metrics
                all_comp_pcts = [
                    s.get('compression_pct') for s in valid_solutions
                    if s.get('compression_pct') is not None
                ]
                all_bpt = [
                    s.get('bits_per_token') for s in valid_solutions
                    if s.get('bits_per_token') is not None
                ]
                avg_comp_pct = sum(all_comp_pcts) / len(
                    all_comp_pcts) if all_comp_pcts else 0
                min_comp_pct = min(all_comp_pcts) if all_comp_pcts else 0
                avg_bpt = sum(all_bpt) / len(all_bpt) if all_bpt else 0
                min_bpt = min(all_bpt) if all_bpt else 0

                print(
                    f"  N={n}: Compression={best_by_compression['extracted_answer']} ({'OK' if best_by_compression['is_correct'] else 'X'}), "
                    f"Majority={majority_answer} ({'OK' if majority_correct else 'X'}), "
                    f"Agreement={agreement['agreement_rate']:.0%}, Time={gen_time:.1f}s"
                )
                print(
                    f"    Compression%: min={min_comp_pct:.2f}%, avg={avg_comp_pct:.2f}% | "
                    f"Bits/token: min={min_bpt:.3f}, avg={avg_bpt:.3f}")
                sys.stdout.flush()

        all_results.append(problem_results)
        print(f"\n>>> Completed problem {prob_idx + 1}/{len(problems)} <<<")
        sys.stdout.flush()

    # =========================================================================
    # Calculate Final Statistics
    # =========================================================================
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}")

    final_summary = {}
    for approach in approaches:
        final_summary[approach] = {}
        print(f"\n{approach.upper()}:")
        print(
            f"{'N':>5} {'Compress':>10} {'Random':>10} {'Majority':>10} {'Agreement':>10} {'Time(s)':>10} {'AvgComp%':>10}"
        )
        print("-" * 80)

        for n in n_values:
            stats = summary[approach][n]
            if stats['total'] > 0:
                acc_compress = stats['correct_by_compression'] / stats[
                    'total'] * 100
                acc_random = stats['correct_by_random'] / stats['total'] * 100
                acc_majority = stats['correct_by_majority'] / stats[
                    'total'] * 100
                avg_agreement = np.mean(
                    stats['agreement_rates']
                ) * 100 if stats['agreement_rates'] else 0
                avg_time = np.mean(stats['generation_times']
                                   ) if stats['generation_times'] else 0
                avg_comp_pct = np.mean(stats['compression_pcts']
                                       ) if stats['compression_pcts'] else 0
                std_compress = np.std([
                    1 if c else 0 for c in [stats['correct_by_compression']]
                ]) if stats['total'] > 1 else 0

                # Calculate std for compression accuracy
                compression_correct_list = [
                    r['approaches'][approach]['by_n'][n]['selection']
                    ['by_compression']['is_correct'] for r in all_results
                    if approach in r.get('approaches', {})
                    and n in r['approaches'][approach].get('by_n', {})
                ]
                acc_compress_std = np.std(
                    compression_correct_list) * 100 if len(
                        compression_correct_list) > 1 else 0

                final_summary[approach][n] = {
                    'accuracy_by_compression':
                    acc_compress,
                    'accuracy_by_random':
                    acc_random,
                    'accuracy_by_majority':
                    acc_majority,
                    'accuracy_by_compression_std':
                    acc_compress_std,
                    'avg_agreement':
                    avg_agreement,
                    'avg_agreement_std':
                    np.std(stats['agreement_rates']) *
                    100 if len(stats['agreement_rates']) > 1 else 0,
                    'avg_compression_pct':
                    avg_comp_pct,
                    'avg_compression_pct_std':
                    np.std(stats['compression_pcts'])
                    if len(stats['compression_pcts']) > 1 else 0,
                    'avg_time':
                    avg_time,
                    'avg_time_std':
                    np.std(stats['generation_times'])
                    if len(stats['generation_times']) > 1 else 0,
                    'total_problems':
                    stats['total'],
                }

                print(
                    f"{n:>5} {acc_compress:>9.1f}% {acc_random:>9.1f}% {acc_majority:>9.1f}% {avg_agreement:>9.1f}% {avg_time:>9.1f} {avg_comp_pct:>9.2f}%"
                )

    # Save results
    experiment_data = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'parameters': {
            'num_problems': len(problems),
            'n_values': n_values,
            'approaches': approaches,
            'generation_model': generation_model,
            'compression_model': compression_model_path,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'seed': seed,
        },
        'summary': final_summary,
        'raw_summary': {
            approach: {
                n: {
                    k: v if not isinstance(v, list) else v
                    for k, v in stats.items()
                }
                for n, stats in approach_stats.items()
            }
            for approach, approach_stats in summary.items()
        },
        'results': all_results,
    }

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    json_file = output_path / f"best_of_n_comparison_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(experiment_data, f, indent=2, default=str)

    print(f"\nResults saved to: {json_file}")

    return experiment_data


def main():
    parser = argparse.ArgumentParser(
        description="Compare different best-of-N approaches on AIME problems")

    parser.add_argument(
        "--num-problems",
        type=int,
        default=DEFAULT_NUM_PROBLEMS,
        help=
        f"Number of AIME problems to test (default: {DEFAULT_NUM_PROBLEMS})")
    parser.add_argument("--problem-indices",
                        type=int,
                        nargs='+',
                        default=None,
                        help="Specific problem indices to test")
    parser.add_argument("--n-values",
                        type=int,
                        nargs='+',
                        default=DEFAULT_N_VALUES,
                        help=f"N values to test (default: {DEFAULT_N_VALUES})")
    parser.add_argument(
        "--generation-model",
        type=str,
        default=DEFAULT_GENERATION_MODEL,
        help=
        f"Model for generating solutions (default: {DEFAULT_GENERATION_MODEL})"
    )
    parser.add_argument(
        "--compression-model",
        type=str,
        default=DEFAULT_COMPRESSION_MODEL,
        help=f"Model for compression (default: {DEFAULT_COMPRESSION_MODEL})")
    parser.add_argument("--max-tokens",
                        type=int,
                        default=2000,
                        help="Max tokens per solution (default: 2000)")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Generation temperature for temp sampling (default: 0.8)")
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="Random seed (default: 42)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/best_of_n_approaches",
        help="Output directory (default: results/best_of_n_approaches)")
    parser.add_argument("--verbose",
                        action="store_true",
                        help="Verbose output")
    parser.add_argument("--approaches",
                        type=str,
                        nargs='+',
                        choices=['temperature', 'single_prompt'],
                        default=['temperature', 'single_prompt'],
                        help="Which approaches to test")

    args = parser.parse_args()

    # Load AIME dataset
    print("Loading AIME dataset...")
    ds = load_dataset("AI-MO/aimo-validation-aime")
    dataset = ds['train']
    print(f"Loaded {len(dataset)} problems")

    # Select problems
    if args.problem_indices:
        problems = [
            dataset[i] for i in args.problem_indices if i < len(dataset)
        ]
    else:
        problems = [
            dataset[i] for i in range(min(args.num_problems, len(dataset)))
        ]

    print(f"Selected {len(problems)} problems for experiment")

    # Run experiment
    results = run_comparison_experiment(
        problems=problems,
        n_values=args.n_values,
        generation_model=args.generation_model,
        compression_model_path=args.compression_model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        seed=args.seed,
        output_dir=args.output_dir,
        verbose=args.verbose,
        approaches=args.approaches,
    )

    print("\nExperiment complete!")
    return results


if __name__ == "__main__":
    main()
