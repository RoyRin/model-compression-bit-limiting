#!/usr/bin/env python3
"""
Best-of-N Compression Experiment on AIME Problems.

Generate N solutions to AIME math problems, select the one that compresses best,
and evaluate whether compression-guided selection improves accuracy.
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
from typing import List, Dict, Any, Optional
import numpy as np
from datasets import load_dataset

from lossy_compression_tools import (
    load_compression_model,
    compress_text,
    select_best_by_compression,
)
from utils.llm_api import anthropic_completion

# Default parameters
DEFAULT_GENERATION_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_COMPRESSION_MODEL = "meta-llama/Llama-3.1-8B"
DEFAULT_N_VALUES = [1, 2, 5, 10]
DEFAULT_NUM_PROBLEMS = 10

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


def generate_solutions(
    problem_text: str,
    n_generations: int,
    model: str = DEFAULT_GENERATION_MODEL,
    max_tokens: int = 2000,
    temperature: float = 0.8,
    seed: int = 42,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """Generate N solutions to an AIME problem."""

    prompt = f"""Solve this AIME problem:

{problem_text}

Please provide a complete solution with clear mathematical reasoning, and state the final numerical answer using \\boxed{{answer}} notation."""

    solutions = []

    for i in range(n_generations):
        current_seed = seed + i

        if verbose:
            print(f"  Generating solution {i+1}/{n_generations}...")

        try:
            start_time = time.time()
            response = anthropic_completion(
                prompt=prompt,
                model=model,
                system=MATH_SYSTEM_PROMPT,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=current_seed,
            )
            gen_time = time.time() - start_time

            extracted = extract_numerical_answer(response)

            solutions.append({
                'index': i,
                'response': response,
                'extracted_answer': extracted,
                'generation_time': gen_time,
                'length': len(response),
            })

            if verbose:
                print(
                    f"    Answer: {extracted}, Length: {len(response)} chars")

        except Exception as e:
            print(f"    Error generating solution {i+1}: {e}")
            solutions.append({
                'index': i,
                'response': None,
                'extracted_answer': None,
                'error': str(e),
            })

    return solutions


def run_best_of_n_experiment(
    problems: List[Dict],
    n_values: List[int] = DEFAULT_N_VALUES,
    generation_model: str = DEFAULT_GENERATION_MODEL,
    compression_model_path: str = DEFAULT_COMPRESSION_MODEL,
    max_tokens: int = 2000,
    temperature: float = 0.8,
    seed: int = 42,
    output_dir: str = "results/aime_best_of_n",
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run best-of-N experiment on AIME problems."""

    print(f"\n{'='*70}")
    print("Best-of-N AIME Compression Experiment")
    print(f"{'='*70}")
    print(f"Problems: {len(problems)}")
    print(f"N values: {n_values}")
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

    all_results = []
    summary_by_n = {
        n: {
            'correct_random': 0,
            'correct_best_compression': 0,
            'total': 0
        }
        for n in n_values
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

        # Generate max_n solutions (we'll subsample for smaller N values)
        print(f"\nGenerating {max_n} solutions...")
        solutions = generate_solutions(
            problem_text=problem_text,
            n_generations=max_n,
            model=generation_model,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed + prob_idx * 1000,
            verbose=verbose,
        )

        # Filter out failed generations
        valid_solutions = [
            s for s in solutions if s.get('response') is not None
        ]
        print(f"Generated {len(valid_solutions)}/{max_n} valid solutions")

        if not valid_solutions:
            print("  No valid solutions generated, skipping problem")
            continue

        # Compress all solutions
        print("\nCompressing solutions...")
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
            sol['bits_per_token'] = metrics['bits_per_token']
            sol['is_correct'] = sol['extracted_answer'] == correct_answer

        # Analyze results for each N value
        problem_results = {
            'problem_id': problem_id,
            'problem_text': problem_text,
            'correct_answer': correct_answer,
            'all_solutions': valid_solutions,
            'results_by_n': {},
        }

        for n in n_values:
            if n > len(valid_solutions):
                print(
                    f"  N={n}: Not enough solutions (only {len(valid_solutions)})"
                )
                continue

            # Take first N solutions
            subset = valid_solutions[:n]

            # Random selection (first one)
            random_solution = subset[0]
            random_correct = random_solution['is_correct']

            # Best compression selection (lowest percentage)
            best_compression_solution = min(subset,
                                            key=lambda x: x['compression_pct'])
            best_compression_correct = best_compression_solution['is_correct']

            # Also track: what if we picked the one with the correct answer?
            # (oracle selection for comparison)
            any_correct = any(s['is_correct'] for s in subset)

            problem_results['results_by_n'][n] = {
                'random_answer':
                random_solution['extracted_answer'],
                'random_correct':
                random_correct,
                'random_compression_pct':
                random_solution['compression_pct'],
                'best_compression_answer':
                best_compression_solution['extracted_answer'],
                'best_compression_correct':
                best_compression_correct,
                'best_compression_pct':
                best_compression_solution['compression_pct'],
                'any_correct_in_n':
                any_correct,
                'num_correct_in_n':
                sum(1 for s in subset if s['is_correct']),
            }

            # Update summary
            summary_by_n[n]['total'] += 1
            if random_correct:
                summary_by_n[n]['correct_random'] += 1
            if best_compression_correct:
                summary_by_n[n]['correct_best_compression'] += 1

            print(f"  N={n}: Random={'✓' if random_correct else '✗'}, "
                  f"BestComp={'✓' if best_compression_correct else '✗'} "
                  f"({best_compression_solution['compression_pct']:.1f}%)")

        all_results.append(problem_results)

    # Calculate final statistics
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}")
    print(
        f"{'N':>5} {'Random Acc':>12} {'BestComp Acc':>14} {'Improvement':>12}"
    )
    print("-" * 50)

    for n in n_values:
        stats = summary_by_n[n]
        if stats['total'] > 0:
            random_acc = stats['correct_random'] / stats['total'] * 100
            best_acc = stats['correct_best_compression'] / stats['total'] * 100
            improvement = best_acc - random_acc
            print(
                f"{n:>5} {random_acc:>11.1f}% {best_acc:>13.1f}% {improvement:>+11.1f}%"
            )

    # Save results
    experiment_data = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'parameters': {
            'num_problems': len(problems),
            'n_values': n_values,
            'generation_model': generation_model,
            'compression_model': compression_model_path,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'seed': seed,
        },
        'summary_by_n': summary_by_n,
        'results': all_results,
    }

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    json_file = output_path / f"aime_best_of_n_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(experiment_data, f, indent=2, default=str)

    print(f"\nResults saved to: {json_file}")

    return experiment_data


def main():
    parser = argparse.ArgumentParser(
        description="Best-of-N compression experiment on AIME problems")

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
    parser.add_argument("--temperature",
                        type=float,
                        default=0.8,
                        help="Generation temperature (default: 0.8)")
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="Random seed (default: 42)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/aime_best_of_n",
        help="Output directory (default: results/aime_best_of_n)")
    parser.add_argument("--verbose",
                        action="store_true",
                        help="Verbose output")

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
    results = run_best_of_n_experiment(
        problems=problems,
        n_values=args.n_values,
        generation_model=args.generation_model,
        compression_model_path=args.compression_model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        seed=args.seed,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )

    print("\nExperiment complete!")
    return results


if __name__ == "__main__":
    main()
