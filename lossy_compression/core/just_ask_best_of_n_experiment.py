#!/usr/bin/env python3
"""
Just Ask + Best-of-N Experiment on AIME Problems.

Combines two lossy compression approaches:
1. Just Ask: Request a succinct rewrite of a verbose solution
2. Best-of-N: Generate N succinct rewrites and select the best-compressed one

Process:
1. Generate a full verbose solution to an AIME problem
2. Strip the final numerical answer from the solution
3. Generate N succinct rewrites (via temperature sampling)
4. Compress each rewrite with arithmetic coding
5. Select the rewrite with the lowest compression ratio
6. Compare against: single Just Ask, verbose baseline, and Best-of-N without Just Ask
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
)
from utils.llm_api import anthropic_completion

# Default parameters
DEFAULT_GENERATION_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_COMPRESSION_MODEL = "meta-llama/Llama-3.1-8B"
DEFAULT_NUM_PROBLEMS = 100
DEFAULT_N_VALUES = [1, 3, 5, 10]

MATH_SYSTEM_PROMPT = """You are a skilled mathematician solving AIME (American Invitational Mathematics Examination) problems.
Provide a clear, step-by-step solution and end with the numerical answer.
Format your final answer clearly using \\boxed{answer} notation.
AIME answers are always integers from 0 to 999."""

COMPRESS_SYSTEM_PROMPT = """You are an expert at condensing mathematical solutions to their essential core while preserving the reasoning needed to derive the answer."""


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
    # Remove boxed answers - replace with placeholder
    text = re.sub(r'\\boxed\{[^}]+\}', '\\boxed{???}', response_text)

    # Remove explicit answer statements at the end
    answer_patterns = [
        r'(?:the\s+)?(?:final\s+)?answer\s+is:?\s*\d+\.?\s*$',
        r'therefore,?\s+(?:the\s+)?answer\s+is:?\s*\d+\.?\s*$',
        r'=\s*\d+\s*$',
    ]

    for pattern in answer_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)

    return text.strip()


def generate_initial_solution(
    problem_text: str,
    model: str = DEFAULT_GENERATION_MODEL,
    max_tokens: int = 2000,
    temperature: float = 0.3,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Generate the initial full solution."""

    prompt = f"""Solve this AIME problem:

{problem_text}

Please provide a complete solution with clear mathematical reasoning, and state the final numerical answer using \\boxed{{answer}} notation."""

    if verbose:
        print("  Generating initial solution...")

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

        answer = extract_numerical_answer(response)

        if verbose:
            print(
                f"    Answer: {answer}, Length: {len(response)} chars, Time: {gen_time:.2f}s"
            )
            sys.stdout.flush()

        return {
            'prompt': prompt,
            'response': response,
            'answer': answer,
            'length': len(response),
            'generation_time': gen_time,
        }

    except Exception as e:
        print(f"    Error: {e}")
        return {
            'prompt': prompt,
            'response': None,
            'answer': None,
            'error': str(e),
        }


def generate_succinct_rewrite(
    problem_text: str,
    masked_solution: str,
    model: str = DEFAULT_GENERATION_MODEL,
    max_tokens: int = 1000,
    temperature: float = 0.7,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Generate a single succinct rewrite of the solution."""

    compress_prompt = f"""Here is a solution to an AIME math problem, but with the final numerical answer removed:

PROBLEM:
{problem_text}

SOLUTION (answer hidden):
{masked_solution}

Your task: Rewrite this solution to be as SUCCINCT as possible while still containing enough information that a small language model (like GPT-3.5 or Llama-7B) could correctly infer the final numerical answer.

Guidelines:
- Keep only the essential reasoning steps and key intermediate results
- Remove all unnecessary explanation and verbose language
- Preserve the logical chain leading to the answer
- Use mathematical notation efficiently
- The goal is MINIMUM text that still allows answer inference

End with: "Therefore, the answer is \\boxed{{X}}" where X is the correct numerical answer."""

    try:
        start_time = time.time()
        response = anthropic_completion(
            prompt=compress_prompt,
            model=model,
            system=COMPRESS_SYSTEM_PROMPT,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        gen_time = time.time() - start_time

        answer = extract_numerical_answer(response)

        return {
            'response': response,
            'answer': answer,
            'length': len(response),
            'generation_time': gen_time,
        }

    except Exception as e:
        if verbose:
            print(f"    Error generating rewrite: {e}")
        return {
            'response': None,
            'answer': None,
            'error': str(e),
        }


def run_just_ask_best_of_n_experiment(
    problems: List[Dict],
    n_values: List[int] = DEFAULT_N_VALUES,
    generation_model: str = DEFAULT_GENERATION_MODEL,
    compression_model_path: str = DEFAULT_COMPRESSION_MODEL,
    max_tokens_initial: int = 2000,
    max_tokens_compress: int = 1000,
    temperature: float = 0.7,
    output_dir: str = "results/just_ask_best_of_n",
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run the Just Ask + Best-of-N experiment."""

    print(f"\n{'='*70}")
    print("Just Ask + Best-of-N Experiment")
    print(f"{'='*70}")
    print(f"Problems: {len(problems)}")
    print(f"N values: {n_values}")
    print(f"Generation model: {generation_model}")
    print(f"Compression model: {compression_model_path}")
    print(f"Temperature: {temperature}")
    print(f"{'='*70}\n")
    sys.stdout.flush()

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load compression model
    print("Loading compression model...")
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compression_model, compression_tokenizer = load_compression_model(
        model_path=compression_model_path,
        device=device,
    )
    print(f"Compression model loaded on {device}\n")

    max_n = max(n_values)
    all_results = []

    # Summary statistics per N
    summary = {
        n: {
            'compression_pcts':
            [],  # Best compression % at this N (relative to original)
            'answer_correct': [],  # Whether best-compressed answer is correct
            'verbose_compression_pcts': [],  # Verbose baseline compression %
        }
        for n in n_values
    }

    for prob_idx, problem in enumerate(problems):
        problem_text = problem['problem']
        correct_answer = str(problem['answer'])
        problem_id = problem.get('id', prob_idx)

        print(f"\n{'='*70}")
        print(f"Problem {prob_idx + 1}/{len(problems)} (ID: {problem_id})")
        print(f"{'='*70}")
        print(f"Problem: {problem_text[:150]}...")
        print(f"Correct answer: {correct_answer}")
        sys.stdout.flush()

        # Step 1: Generate verbose solution
        print("\n--- Step 1: Generate Verbose Solution ---")
        initial = generate_initial_solution(
            problem_text=problem_text,
            model=generation_model,
            max_tokens=max_tokens_initial,
            verbose=verbose,
        )

        if initial.get('response') is None:
            print("  Failed to generate initial solution, skipping...")
            continue

        # Compress verbose solution for baseline
        _, verbose_compression_pct, verbose_metrics = compress_text(
            text=initial['response'],
            compression_model=compression_model,
            compression_tokenizer=compression_tokenizer,
            bit_precision=64,
            device=device,
            verbose=False,
        )

        verbose_original_bytes = verbose_metrics.get('original_bytes',
                                                     len(initial['response']))

        print(
            f"  Verbose: {initial['length']} chars, {verbose_compression_pct:.2f}% compression"
        )
        print(
            f"  Verbose answer: {initial['answer']} ({'OK' if initial['answer'] == correct_answer else 'X'})"
        )

        # Step 2: Mask the answer
        masked_solution = strip_final_answer(initial['response'])

        # Step 3: Generate N succinct rewrites
        print(f"\n--- Step 2: Generate {max_n} Succinct Rewrites ---")

        rewrites = []
        for i in range(max_n):
            print(f"  [Rewrite {i+1}/{max_n}]", end=" ")
            sys.stdout.flush()

            rewrite = generate_succinct_rewrite(
                problem_text=problem_text,
                masked_solution=masked_solution,
                model=generation_model,
                max_tokens=max_tokens_compress,
                temperature=temperature,
                verbose=verbose,
            )

            if rewrite.get('response') is None:
                print("FAILED")
                continue

            # Compress the rewrite
            _, rewrite_compression_pct, rewrite_metrics = compress_text(
                text=rewrite['response'],
                compression_model=compression_model,
                compression_tokenizer=compression_tokenizer,
                bit_precision=64,
                device=device,
                verbose=False,
            )

            rewrite['compression_pct'] = rewrite_compression_pct
            rewrite['compressed_bytes'] = rewrite_metrics.get(
                'compressed_bytes', 0)
            rewrite['original_bytes'] = rewrite_metrics.get(
                'original_bytes', 0)

            # Calculate compression relative to verbose original
            rewrite['compression_vs_verbose'] = (rewrite['compressed_bytes'] /
                                                 verbose_original_bytes) * 100

            is_correct = rewrite['answer'] == correct_answer
            rewrite['is_correct'] = is_correct

            print(
                f"Answer={rewrite['answer']} ({'OK' if is_correct else 'X'}), "
                f"Length={rewrite['length']}, "
                f"Compression={rewrite['compression_vs_verbose']:.2f}%")

            rewrites.append(rewrite)

        if not rewrites:
            print("  No valid rewrites generated, skipping...")
            continue

        # Step 4: Analyze results for each N
        print(f"\n--- Step 3: Results by N ---")

        problem_results = {
            'problem_id': problem_id,
            'problem_text': problem_text,
            'correct_answer': correct_answer,
            'verbose': {
                'response': initial['response'],
                'answer': initial['answer'],
                'length': initial['length'],
                'compression_pct': verbose_compression_pct,
                'original_bytes': verbose_original_bytes,
                'compressed_bytes': verbose_metrics.get('compressed_bytes', 0),
                'is_correct': initial['answer'] == correct_answer,
            },
            'rewrites': rewrites,
            'by_n': {},
        }

        for n in n_values:
            if n > len(rewrites):
                continue

            # Take first N rewrites and find best compressed
            n_rewrites = rewrites[:n]
            best_rewrite = min(n_rewrites,
                               key=lambda x: x['compression_vs_verbose'])

            # Also compute average compression
            avg_compression = np.mean(
                [r['compression_vs_verbose'] for r in n_rewrites])

            # Check if majority answer is correct
            answers = [
                r['answer'] for r in n_rewrites if r['answer'] is not None
            ]
            if answers:
                from collections import Counter
                majority_answer = Counter(answers).most_common(1)[0][0]
                majority_correct = majority_answer == correct_answer
            else:
                majority_answer = None
                majority_correct = False

            problem_results['by_n'][n] = {
                'best_compression_vs_verbose':
                best_rewrite['compression_vs_verbose'],
                'best_answer':
                best_rewrite['answer'],
                'best_is_correct':
                best_rewrite['is_correct'],
                'avg_compression_vs_verbose':
                avg_compression,
                'majority_answer':
                majority_answer,
                'majority_correct':
                majority_correct,
            }

            # Update summary
            summary[n]['compression_pcts'].append(
                best_rewrite['compression_vs_verbose'])
            summary[n]['answer_correct'].append(best_rewrite['is_correct'])
            summary[n]['verbose_compression_pcts'].append(
                verbose_compression_pct)

            status = "OK" if best_rewrite['is_correct'] else "X"
            print(
                f"  N={n}: Best={best_rewrite['compression_vs_verbose']:.2f}%, "
                f"Avg={avg_compression:.2f}%, "
                f"Answer={best_rewrite['answer']} ({status})")

        all_results.append(problem_results)
        print(f"\n>>> Completed problem {prob_idx + 1}/{len(problems)} <<<")
        sys.stdout.flush()

    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}")
    print(
        f"\n{'N':>5} {'BestComp%':>12} {'Verbose%':>12} {'Accuracy':>12} {'Improvement':>12}"
    )
    print("-" * 60)

    for n in n_values:
        if summary[n]['compression_pcts']:
            avg_best = np.mean(summary[n]['compression_pcts'])
            avg_verbose = np.mean(summary[n]['verbose_compression_pcts'])
            accuracy = np.mean(summary[n]['answer_correct']) * 100
            improvement = ((avg_verbose - avg_best) / avg_verbose) * 100
            print(
                f"{n:>5} {avg_best:>11.2f}% {avg_verbose:>11.2f}% {accuracy:>11.1f}% {improvement:>11.1f}%"
            )

    # Save results
    experiment_data = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'parameters': {
            'num_problems': len(problems),
            'n_values': n_values,
            'generation_model': generation_model,
            'compression_model': compression_model_path,
            'max_tokens_initial': max_tokens_initial,
            'max_tokens_compress': max_tokens_compress,
            'temperature': temperature,
        },
        'summary': {
            n: {
                'avg_best_compression_pct':
                float(np.mean(summary[n]['compression_pcts']))
                if summary[n]['compression_pcts'] else None,
                'avg_verbose_compression_pct':
                float(np.mean(summary[n]['verbose_compression_pcts']))
                if summary[n]['verbose_compression_pcts'] else None,
                'accuracy':
                float(np.mean(summary[n]['answer_correct'])) *
                100 if summary[n]['answer_correct'] else None,
                'n_problems':
                len(summary[n]['compression_pcts']),
            }
            for n in n_values
        },
        'results': all_results,
    }

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    json_file = output_path / f"just_ask_best_of_n_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(experiment_data, f, indent=2, default=str)

    print(f"\nResults saved to: {json_file}")

    # Save separate file with just the texts for easy analysis
    texts_for_analysis = []
    for result in all_results:
        problem_texts = {
            'problem_id':
            result['problem_id'],
            'problem_text':
            result['problem_text'],
            'correct_answer':
            result['correct_answer'],
            'verbose_solution':
            result['verbose']['response'],
            'verbose_answer':
            result['verbose']['answer'],
            'verbose_is_correct':
            result['verbose']['is_correct'],
            'succinct_rewrites': [{
                'rewrite_idx':
                i,
                'text':
                r['response'],
                'answer':
                r['answer'],
                'is_correct':
                r['is_correct'],
                'length':
                r['length'],
                'compression_vs_verbose':
                r['compression_vs_verbose'],
            } for i, r in enumerate(result['rewrites'])],
            'best_rewrite_idx':
            min(range(len(result['rewrites'])),
                key=lambda i: result['rewrites'][i]['compression_vs_verbose'])
            if result['rewrites'] else None,
        }
        texts_for_analysis.append(problem_texts)

    texts_file = output_path / f"texts_for_analysis_{timestamp}.json"
    with open(texts_file, 'w') as f:
        json.dump(texts_for_analysis, f, indent=2, default=str)

    print(f"Texts for analysis saved to: {texts_file}")

    return experiment_data


def main():
    parser = argparse.ArgumentParser(
        description="Just Ask + Best-of-N experiment on AIME problems")

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
    parser.add_argument("--max-tokens-initial",
                        type=int,
                        default=2000,
                        help="Max tokens for initial solution (default: 2000)")
    parser.add_argument(
        "--max-tokens-compress",
        type=int,
        default=1000,
        help="Max tokens for compressed solution (default: 1000)")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for generating diverse rewrites (default: 0.7)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/just_ask_best_of_n",
        help="Output directory (default: results/just_ask_best_of_n)")
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
    results = run_just_ask_best_of_n_experiment(
        problems=problems,
        n_values=args.n_values,
        generation_model=args.generation_model,
        compression_model_path=args.compression_model,
        max_tokens_initial=args.max_tokens_initial,
        max_tokens_compress=args.max_tokens_compress,
        temperature=args.temperature,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )

    print("\nExperiment complete!")
    return results


if __name__ == "__main__":
    main()
