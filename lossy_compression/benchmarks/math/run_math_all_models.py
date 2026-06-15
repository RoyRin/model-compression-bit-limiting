#!/usr/bin/env python3
"""
Run MATH evaluation across haiku, sonnet, and opus to classify problem difficulty.

Difficulty classification:
- easy: all models pass
- medium: haiku fails, sonnet/opus pass
- hard: haiku/sonnet fail, opus passes
- very_hard: all models fail

Usage:
    python run_math_all_models.py --subject algebra --num-problems 50
    python run_math_all_models.py --subject all --num-problems 100
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

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from datasets import load_dataset
from lossy_compression import MODEL_ALIAS_MAP, model_completion

MATH_SUBJECTS = [
    'algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra',
    'number_theory', 'prealgebra', 'precalculus'
]

MODELS = ['haiku', 'sonnet', 'opus']


def extract_boxed_answer(text):
    """Extract answer from \\boxed{...} format."""
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    answer_pattern = r'(?:answer|result)(?:\s+is)?[:\s]+([^\n.,]+)'
    match = re.search(answer_pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
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


def solve_problem(problem_text, model_name):
    """Solve a MATH problem using the specified model."""
    model_full = MODEL_ALIAS_MAP.get(model_name.lower(), model_name)

    prompt = f"""Solve this math problem step by step. Put your final answer in \\boxed{{}}.

Problem: {problem_text}

Solution:"""

    system = """You are a mathematical problem solver. Show clear step-by-step reasoning.
Always put your final answer in \\boxed{} format at the end."""

    start_time = time.time()
    try:
        response = model_completion(prompt,
                                    model=model_full,
                                    system=system,
                                    temperature=0.0,
                                    max_tokens=2000)
        return {
            'response': response,
            'extracted_answer': extract_boxed_answer(response),
            'solve_time': time.time() - start_time,
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


def evaluate_problem(problem_idx, problem, subject):
    """Evaluate a single problem across all models."""
    gold_answer = extract_boxed_answer(problem['solution'])
    gold_norm = normalize_answer(gold_answer)

    result = {
        'problem_idx': problem_idx,
        'subject': subject,
        'level': problem['level'],
        'problem': problem['problem'],
        'gold_answer': gold_answer,
        'models': {}
    }

    for model in MODELS:
        solution = solve_problem(problem['problem'], model)
        pred_norm = normalize_answer(solution['extracted_answer'])
        is_correct = pred_norm == gold_norm if pred_norm and gold_norm else False

        result['models'][model] = {
            'answer': solution['extracted_answer'],
            'correct': is_correct,
            'solve_time': solution['solve_time']
        }

    # Classify difficulty
    haiku_pass = result['models']['haiku']['correct']
    sonnet_pass = result['models']['sonnet']['correct']
    opus_pass = result['models']['opus']['correct']

    if haiku_pass and sonnet_pass and opus_pass:
        result['difficulty'] = 'easy'
    elif not haiku_pass and sonnet_pass and opus_pass:
        result['difficulty'] = 'medium'
    elif not haiku_pass and not sonnet_pass and opus_pass:
        result['difficulty'] = 'hard'
    else:
        result['difficulty'] = 'very_hard'

    return result


def main():
    parser = argparse.ArgumentParser(description='Run MATH on all models')
    parser.add_argument('--subject',
                        type=str,
                        default='algebra',
                        choices=MATH_SUBJECTS + ['all'])
    parser.add_argument('--num-problems', type=int, default=50)
    parser.add_argument('--output', type=str, help='Output file')
    parser.add_argument('--parallel',
                        action='store_true',
                        help='Run models in parallel (uses more API quota)')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"MATH Multi-Model Evaluation")
    print(f"Subject: {args.subject}")
    print(f"Models: {', '.join(MODELS)}")
    print(f"{'='*60}\n")

    # Load dataset
    if args.subject == 'all':
        all_problems = []
        for subj in MATH_SUBJECTS:
            ds = load_dataset('EleutherAI/hendrycks_math', subj)
            for item in ds['test']:
                all_problems.append((dict(item), subj))
        print(f"Loaded {len(all_problems)} problems across all subjects")
    else:
        ds = load_dataset('EleutherAI/hendrycks_math', args.subject)
        all_problems = [(dict(item), args.subject) for item in ds['test']]
        print(f"Loaded {len(all_problems)} {args.subject} problems")

    problems = all_problems[:args.num_problems]
    print(f"Evaluating {len(problems)} problems\n")

    results = []
    counts = {'easy': 0, 'medium': 0, 'hard': 0, 'very_hard': 0}
    model_correct = {m: 0 for m in MODELS}

    start_time = time.time()

    for i, (problem, subject) in enumerate(problems):
        result = evaluate_problem(i, problem, subject)
        results.append(result)

        counts[result['difficulty']] += 1
        for m in MODELS:
            if result['models'][m]['correct']:
                model_correct[m] += 1

        # Progress
        elapsed = time.time() - start_time
        print(
            f"\rProblem {i+1}/{len(problems)} | "
            f"H:{model_correct['haiku']}/{i+1} "
            f"S:{model_correct['sonnet']}/{i+1} "
            f"O:{model_correct['opus']}/{i+1} | "
            f"Elapsed: {elapsed:.0f}s",
            end='',
            flush=True)

    print("\n")

    # Summary
    total = len(results)
    summary = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'subject': args.subject,
            'num_problems': total,
            'models': MODELS
        },
        'model_accuracy': {
            m: model_correct[m] / total
            for m in MODELS
        },
        'difficulty_distribution': {
            d: counts[d] / total
            for d in counts
        },
        'difficulty_counts': counts,
        'results': results
    }

    print(f"{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"\nModel Accuracy:")
    for m in MODELS:
        print(
            f"  {m}: {model_correct[m]}/{total} ({model_correct[m]/total:.1%})"
        )

    print(f"\nDifficulty Distribution:")
    for d, c in counts.items():
        print(f"  {d}: {c}/{total} ({c/total:.1%})")

    # Save
    if args.output:
        output_path = args.output
    else:
        os.makedirs('lossy_compression/results', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f'lossy_compression/results/math_all_models_{args.subject}_{timestamp}.json'

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
