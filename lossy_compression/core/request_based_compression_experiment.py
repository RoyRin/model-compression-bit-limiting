#!/usr/bin/env python3
"""
Request-Based Compression Experiment on AIME Problems.

This experiment tests whether we can compress a solution by explicitly
requesting a more succinct version that preserves the essential reasoning
needed for a smaller model to infer the correct answer.

Process:
1. Generate a full solution to an AIME problem
2. Strip the final numerical answer from the solution
3. Ask the model to rewrite the solution as succinctly as possible
   while preserving enough information for a small LM to infer the answer
4. Measure compression (both text length and arithmetic coding)
5. Evaluate whether the compressed solution still leads to correct answers
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

MATH_SYSTEM_PROMPT = """You are a skilled mathematician solving AIME (American Invitational Mathematics Examination) problems.
Provide a clear, step-by-step solution and end with the numerical answer.
Format your final answer clearly using \\boxed{answer} notation.
AIME answers are always integers from 0 to 999."""


def extract_numerical_answer(response_text: str) -> Optional[str]:
    """Extract numerical answer from model response."""
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
            print(f"\n  === INITIAL SOLUTION ({len(response)} chars) ===")
            print(response)
            print(f"  === END INITIAL SOLUTION ===\n")
            print(f"    Answer: {answer}, Time: {gen_time:.2f}s")
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


def compress_solution(
    problem_text: str,
    original_solution: str,
    model: str = DEFAULT_GENERATION_MODEL,
    max_tokens: int = 1000,
    temperature: float = 0.2,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Compress a solution to its most succinct form via explicit request."""

    # Strip the answer from the original solution
    stripped_solution = strip_final_answer(original_solution)

    compress_prompt = f"""Here is a solution to an AIME math problem, but with the final numerical answer removed:

PROBLEM:
{problem_text}

SOLUTION (answer hidden):
{stripped_solution}

Your task: Rewrite this solution to be as SUCCINCT as possible while still containing enough information that a small language model (like GPT-3.5 or Llama-7B) could correctly infer the final numerical answer.

Guidelines:
- Keep only the essential reasoning steps and key intermediate results
- Remove all unnecessary explanation and verbose language
- Preserve the logical chain leading to the answer
- Use mathematical notation efficiently
- The goal is MINIMUM text that still allows answer inference

End with: "Therefore, the answer is \\boxed{{X}}" where X is the correct numerical answer."""

    if verbose:
        print("  Compressing solution...")
        print(
            f"\n  === ORIGINAL SOLUTION ({len(original_solution)} chars) ===")
        print(original_solution)
        print(f"  === END ORIGINAL SOLUTION ===\n")
        print(f"\n  === COMPRESS PROMPT ===")
        print(compress_prompt)
        print(f"  === END COMPRESS PROMPT ===\n")
        sys.stdout.flush()

    try:
        start_time = time.time()
        response = anthropic_completion(
            prompt=compress_prompt,
            model=model,
            system=
            "You are an expert at condensing mathematical solutions to their essential core while preserving the reasoning needed to derive the answer.",
            max_tokens=max_tokens,
            temperature=temperature,
        )
        gen_time = time.time() - start_time

        answer = extract_numerical_answer(response)

        if verbose:
            print(f"\n  === COMPRESSED SOLUTION ({len(response)} chars) ===")
            print(response)
            print(f"  === END COMPRESSED SOLUTION ===\n")
            print(f"    Answer: {answer}, Time: {gen_time:.2f}s")
            sys.stdout.flush()

        return {
            'prompt': compress_prompt,
            'response': response,
            'answer': answer,
            'length': len(response),
            'generation_time': gen_time,
        }

    except Exception as e:
        print(f"    Error: {e}")
        return {
            'prompt': compress_prompt,
            'response': None,
            'answer': None,
            'error': str(e),
        }


def run_request_based_compression_experiment(
    problems: List[Dict],
    generation_model: str = DEFAULT_GENERATION_MODEL,
    compression_model_path: str = DEFAULT_COMPRESSION_MODEL,
    max_tokens_initial: int = 2000,
    max_tokens_compress: int = 1000,
    output_dir: str = "results/request_based_compression",
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run the request-based compression experiment on AIME problems."""

    print(f"\n{'='*70}")
    print("Request-Based Compression Experiment")
    print(f"{'='*70}")
    print(f"Problems: {len(problems)}")
    print(f"Generation model: {generation_model}")
    print(f"Compression model: {compression_model_path}")
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
    print(f"Compression model loaded on {device}")

    all_results = []
    summary = {
        'initial_correct': 0,
        'compressed_correct': 0,
        'total': 0,
        'text_compression_ratios': [],
        'initial_compression_pcts': [],
        'compressed_compression_pcts': [],
    }

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

        # Step 1: Generate initial solution
        print("\n--- Step 1: Initial Solution ---")
        sys.stdout.flush()
        initial = generate_initial_solution(
            problem_text=problem_text,
            model=generation_model,
            max_tokens=max_tokens_initial,
            verbose=verbose,
        )

        if initial.get('response') is None:
            print("  Failed to generate initial solution, skipping...")
            continue

        # Step 2: Compress the solution via request
        print("\n--- Step 2: Compressed Solution ---")
        sys.stdout.flush()
        compressed = compress_solution(
            problem_text=problem_text,
            original_solution=initial['response'],
            model=generation_model,
            max_tokens=max_tokens_compress,
            verbose=verbose,
        )

        if compressed.get('response') is None:
            print("  Failed to compress solution, skipping...")
            continue

        # Step 3: Compress both solutions
        print("\n--- Step 3: Compression Analysis ---")

        _, initial_pct, initial_metrics = compress_text(
            text=initial['response'],
            compression_model=compression_model,
            compression_tokenizer=compression_tokenizer,
            bit_precision=64,
            device=device,
            verbose=False,
        )

        _, compressed_pct, compressed_metrics = compress_text(
            text=compressed['response'],
            compression_model=compression_model,
            compression_tokenizer=compression_tokenizer,
            bit_precision=64,
            device=device,
            verbose=False,
        )

        # Calculate metrics
        initial_correct = initial['answer'] == correct_answer
        compressed_correct = compressed['answer'] == correct_answer
        text_compression_ratio = compressed['length'] / initial[
            'length'] if initial['length'] > 0 else 1.0

        # Print results
        print(
            f"  Initial:   {initial['length']:5d} chars, {initial_pct:5.1f}% compressed, "
            f"answer={initial['answer']} ({'CORRECT' if initial_correct else 'WRONG'})"
        )
        print(
            f"  Compressed: {compressed['length']:5d} chars, {compressed_pct:5.1f}% compressed, "
            f"answer={compressed['answer']} ({'CORRECT' if compressed_correct else 'WRONG'})"
        )
        print(
            f"  Text reduction: {text_compression_ratio*100:.1f}% of original")
        print(
            f"  Compression improvement: {initial_pct:.1f}% -> {compressed_pct:.1f}%"
        )
        sys.stdout.flush()

        # Store results
        problem_results = {
            'problem_id': problem_id,
            'problem_text': problem_text,
            'correct_answer': correct_answer,
            'initial': {
                'prompt': initial.get('prompt', ''),
                'response': initial['response'],
                'answer': initial['answer'],
                'length': initial['length'],
                'compression_pct': initial_pct,
                'original_bytes': initial_metrics.get('original_bytes', 0),
                'compressed_bytes': initial_metrics.get('compressed_bytes', 0),
                'n_tokens': initial_metrics.get('n_tokens', 0),
                'bits_per_token': initial_metrics.get('bits_per_token', 0),
                'is_correct': initial_correct,
                'generation_time': initial.get('generation_time', 0),
            },
            'compressed': {
                'prompt': compressed.get('prompt', ''),
                'response': compressed['response'],
                'answer': compressed['answer'],
                'length': compressed['length'],
                'compression_pct': compressed_pct,
                'original_bytes': compressed_metrics.get('original_bytes', 0),
                'compressed_bytes':
                compressed_metrics.get('compressed_bytes', 0),
                'n_tokens': compressed_metrics.get('n_tokens', 0),
                'bits_per_token': compressed_metrics.get('bits_per_token', 0),
                'is_correct': compressed_correct,
                'generation_time': compressed.get('generation_time', 0),
            },
            'text_compression_ratio': text_compression_ratio,
        }

        all_results.append(problem_results)

        # Update summary
        summary['total'] += 1
        if initial_correct:
            summary['initial_correct'] += 1
        if compressed_correct:
            summary['compressed_correct'] += 1
        summary['text_compression_ratios'].append(text_compression_ratio)
        summary['initial_compression_pcts'].append(initial_pct)
        summary['compressed_compression_pcts'].append(compressed_pct)

        print(f"\n>>> Completed problem {prob_idx + 1}/{len(problems)} <<<")
        sys.stdout.flush()

    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}")

    if summary['total'] > 0:
        initial_acc = summary['initial_correct'] / summary['total'] * 100
        compressed_acc = summary['compressed_correct'] / summary['total'] * 100
        avg_text_ratio = np.mean(summary['text_compression_ratios']) * 100
        avg_initial_pct = np.mean(summary['initial_compression_pcts'])
        avg_compressed_pct = np.mean(summary['compressed_compression_pcts'])

        print(f"Problems tested: {summary['total']}")
        print(f"\nAccuracy:")
        print(
            f"  Initial solutions:   {initial_acc:.1f}% ({summary['initial_correct']}/{summary['total']})"
        )
        print(
            f"  Compressed solutions: {compressed_acc:.1f}% ({summary['compressed_correct']}/{summary['total']})"
        )

        print(f"\nText Length:")
        print(f"  Average reduction: {avg_text_ratio:.1f}% of original")

        print(f"\nArithmetic Coding Compression:")
        print(f"  Initial avg:   {avg_initial_pct:.1f}%")
        print(f"  Compressed avg: {avg_compressed_pct:.1f}%")
        print(
            f"  Improvement:   {avg_initial_pct - avg_compressed_pct:.1f} percentage points"
        )

    # Save results
    experiment_data = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'parameters': {
            'num_problems': len(problems),
            'generation_model': generation_model,
            'compression_model': compression_model_path,
            'max_tokens_initial': max_tokens_initial,
            'max_tokens_compress': max_tokens_compress,
        },
        'summary': {
            'total':
            summary['total'],
            'initial_correct':
            summary['initial_correct'],
            'compressed_correct':
            summary['compressed_correct'],
            'avg_text_compression_ratio':
            float(np.mean(summary['text_compression_ratios']))
            if summary['text_compression_ratios'] else None,
            'avg_initial_compression_pct':
            float(np.mean(summary['initial_compression_pcts']))
            if summary['initial_compression_pcts'] else None,
            'avg_compressed_compression_pct':
            float(np.mean(summary['compressed_compression_pcts']))
            if summary['compressed_compression_pcts'] else None,
        },
        'results': all_results,
    }

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    json_file = output_path / f"request_based_compression_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(experiment_data, f, indent=2, default=str)

    print(f"\nResults saved to: {json_file}")

    return experiment_data


def main():
    parser = argparse.ArgumentParser(
        description="Request-based compression experiment on AIME problems")

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
        "--output-dir",
        type=str,
        default="results/request_based_compression",
        help="Output directory (default: results/request_based_compression)")
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
    results = run_request_based_compression_experiment(
        problems=problems,
        generation_model=args.generation_model,
        compression_model_path=args.compression_model,
        max_tokens_initial=args.max_tokens_initial,
        max_tokens_compress=args.max_tokens_compress,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )

    print("\nExperiment complete!")
    return results


if __name__ == "__main__":
    main()
