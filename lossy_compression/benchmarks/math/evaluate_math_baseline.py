#!/usr/bin/env python3
"""
Evaluate MATH dataset baseline (no Q&A compression).

Usage:
    python evaluate_math_baseline.py --model haiku --num-problems 10
    python evaluate_math_baseline.py --model opus --subject algebra --num-problems 50
"""

import json
import time
import argparse
import re
import os
import sys
from pathlib import Path
from datetime import datetime

# Add parent paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from datasets import load_dataset

from lossy_compression import MODEL_ALIAS_MAP, model_completion

# MATH subjects available in EleutherAI/hendrycks_math
MATH_SUBJECTS = [
    'algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra',
    'number_theory', 'prealgebra', 'precalculus'
]


def extract_boxed_answer(text):
    """Extract answer from \\boxed{...} format."""
    # Find all boxed expressions
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()

    # Fallback: look for "answer is X" pattern
    answer_pattern = r'(?:answer|result)(?:\s+is)?[:\s]+([^\n.,]+)'
    match = re.search(answer_pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    return None


def normalize_answer(answer):
    """Normalize answer for comparison."""
    if answer is None:
        return None
    # Remove whitespace, convert to lowercase
    ans = str(answer).strip().lower()
    # Remove common LaTeX formatting
    ans = ans.replace('\\$', '').replace('$', '')
    ans = ans.replace('\\text{', '').replace('}', '')
    ans = ans.replace('\\', '')
    ans = ans.replace(' ', '')
    return ans


def solve_math_problem(problem, model_name, verbose=False):
    """Solve a MATH problem using the specified model."""

    prompt = f"""Solve this math problem step by step. Put your final answer in \\boxed{{}}.

Problem: {problem}

Solution:"""

    system = """You are a mathematical problem solver. Show clear step-by-step reasoning.
Always put your final answer in \\boxed{} format at the end."""

    start_time = time.time()

    try:
        response = model_completion(prompt,
                                    model=model_name,
                                    system=system,
                                    temperature=0.0,
                                    max_tokens=2000)

        solve_time = time.time() - start_time
        extracted = extract_boxed_answer(response)

        return {
            'response': response,
            'extracted_answer': extracted,
            'solve_time': solve_time,
            'success': True
        }
    except Exception as e:
        return {
            'response': str(e),
            'extracted_answer': None,
            'solve_time': time.time() - start_time,
            'success': False,
            'error': str(e)
        }


def evaluate_baseline(model_name,
                      subject='algebra',
                      num_problems=None,
                      verbose=False):
    """Evaluate baseline model on MATH dataset."""

    # Resolve model alias
    model_full = MODEL_ALIAS_MAP.get(model_name.lower(), model_name)

    print(f"\n{'='*60}")
    print(f"Evaluating MATH Baseline")
    print(f"Model: {model_name} ({model_full})")
    print(f"Subject: {subject}")
    print(f"{'='*60}")

    # Load dataset
    if subject == 'all':
        # Load all subjects
        all_problems = []
        for subj in MATH_SUBJECTS:
            ds = load_dataset('EleutherAI/hendrycks_math', subj)
            for item in ds['test']:
                item['subject'] = subj
                all_problems.append(item)
        print(f"Loaded {len(all_problems)} problems across all subjects")
    else:
        ds = load_dataset('EleutherAI/hendrycks_math', subject)
        all_problems = [dict(item, subject=subject) for item in ds['test']]
        print(f"Loaded {len(all_problems)} {subject} problems")

    # Select problems
    if num_problems:
        problems = all_problems[:num_problems]
    else:
        problems = all_problems

    print(f"Evaluating {len(problems)} problems")

    results = []
    correct_count = 0
    start_time = time.time()

    for i, problem in enumerate(problems):
        solution = solve_math_problem(problem['problem'], model_full, verbose)

        # Extract gold answer
        gold_answer = extract_boxed_answer(problem['solution'])

        # Compare answers
        pred_norm = normalize_answer(solution['extracted_answer'])
        gold_norm = normalize_answer(gold_answer)
        is_correct = pred_norm == gold_norm if pred_norm and gold_norm else False

        if is_correct:
            correct_count += 1

        result = {
            'problem_id':
            i,
            'subject':
            problem.get('subject', subject),
            'level':
            problem['level'],
            'problem':
            problem['problem'][:200] +
            '...' if len(problem['problem']) > 200 else problem['problem'],
            'gold_answer':
            gold_answer,
            'model_answer':
            solution['extracted_answer'],
            'is_correct':
            is_correct,
            'solve_time':
            solution['solve_time']
        }
        results.append(result)

        # Progress update
        elapsed = time.time() - start_time
        acc = correct_count / len(results) * 100
        print(
            f"\rProblem {i+1}/{len(problems)} | Acc: {correct_count}/{len(results)} ({acc:.1f}%) | "
            f"Time: {solution['solve_time']:.1f}s | Elapsed: {elapsed:.1f}s",
            end='',
            flush=True)

        if verbose:
            print(f"\n  Level: {problem['level']}")
            print(
                f"  Gold: {gold_answer} | Pred: {solution['extracted_answer']} | {'✓' if is_correct else '✗'}"
            )

    print()  # Newline after progress

    # Summary
    accuracy = correct_count / len(results) if results else 0
    total_time = time.time() - start_time

    summary = {
        'model': model_name,
        'model_full': model_full,
        'subject': subject,
        'total_problems': len(results),
        'correct_count': correct_count,
        'accuracy': accuracy,
        'total_time': total_time,
        'avg_time': total_time / len(results) if results else 0,
        'results': results
    }

    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Subject: {subject}")
    print(f"Accuracy: {accuracy:.1%} ({correct_count}/{len(results)})")
    print(f"Total time: {total_time:.1f}s")
    print(f"Avg time per problem: {total_time/len(results):.1f}s")

    return summary


def main():
    parser = argparse.ArgumentParser(description='Evaluate MATH baseline')
    parser.add_argument('--model',
                        type=str,
                        default='haiku',
                        help='Model to evaluate (default: haiku)')
    parser.add_argument('--subject',
                        type=str,
                        default='algebra',
                        choices=MATH_SUBJECTS + ['all'],
                        help='Math subject (default: algebra)')
    parser.add_argument('--num-problems',
                        type=int,
                        default=None,
                        help='Number of problems to evaluate')
    parser.add_argument('--verbose',
                        action='store_true',
                        help='Show detailed output')
    parser.add_argument('--output', type=str, help='Output file for results')

    args = parser.parse_args()

    results = evaluate_baseline(model_name=args.model,
                                subject=args.subject,
                                num_problems=args.num_problems,
                                verbose=args.verbose)

    # Save results
    if args.output:
        output_path = args.output
    else:
        os.makedirs('lossy_compression/results', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f'lossy_compression/results/math_baseline_{args.model}_{args.subject}_{timestamp}.json'

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
