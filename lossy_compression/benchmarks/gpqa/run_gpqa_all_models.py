#!/usr/bin/env python3
"""
Run GPQA-diamond evaluation across haiku, sonnet, and opus to classify problem difficulty.

Difficulty classification:
- easy: all models pass
- medium: haiku fails, sonnet/opus pass
- hard: haiku/sonnet fail, opus passes
- very_hard: all models fail

Usage:
    python run_gpqa_all_models.py --num-problems 50
    python run_gpqa_all_models.py  # All 198 problems
"""

import json
import time
import argparse
import re
import os
import sys
import random
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from datasets import load_dataset
from lossy_compression import MODEL_ALIAS_MAP, model_completion

MODELS = ['haiku', 'sonnet', 'opus']


def extract_answer_letter(response):
    """Extract answer letter (A, B, C, or D) from model response."""
    patterns = [
        r'answer is[\s:]*([A-D])\b',
        r'answer:[\s]*([A-D])\b',
        r'correct answer[\s:]+([A-D])\b',
        r'choose[\s:]+([A-D])\b',
        r'select[\s:]+([A-D])\b',
        r'^([A-D])\b',
        r'\b([A-D])\)',
        r'\*\*([A-D])\*\*',
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()

    letter_match = re.search(r'\b([A-D])\b', response)
    if letter_match:
        return letter_match.group(1).upper()

    return None


def solve_problem(problem, model_name, problem_id):
    """Solve a GPQA problem using the specified model."""
    model_full = MODEL_ALIAS_MAP.get(model_name.lower(), model_name)

    # Create answer choices and shuffle deterministically
    answers = [
        problem['Incorrect Answer 1'], problem['Incorrect Answer 2'],
        problem['Incorrect Answer 3'], problem['Correct Answer']
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
        if idx == 3:
            correct_letter = letter

    prompt = f"""{problem['Question']}

Choices:
A) {choices[0][1]}
B) {choices[1][1]}
C) {choices[2][1]}
D) {choices[3][1]}

Please analyze this question carefully and select the best answer. Provide your reasoning, then clearly state your answer as A, B, C, or D."""

    system = """You are an expert scientist with deep knowledge across physics, chemistry, biology, and other scientific domains.
Analyze questions carefully, show your reasoning, and provide clear answers."""

    start_time = time.time()
    try:
        response = model_completion(prompt,
                                    model=model_full,
                                    system=system,
                                    temperature=0.0,
                                    max_tokens=2000)
        return {
            'response': response,
            'extracted_answer': extract_answer_letter(response),
            'correct_answer': correct_letter,
            'solve_time': time.time() - start_time,
            'success': True
        }
    except Exception as e:
        return {
            'response': str(e),
            'extracted_answer': None,
            'correct_answer': correct_letter,
            'solve_time': time.time() - start_time,
            'success': False,
            'error': str(e)
        }


def evaluate_problem(problem_idx, problem):
    """Evaluate a single problem across all models."""
    result = {
        'problem_idx':
        problem_idx,
        'question':
        problem['Question'][:200] +
        '...' if len(problem['Question']) > 200 else problem['Question'],
        'models': {}
    }

    correct_letter = None  # Will be set by first model solve

    for model in MODELS:
        solution = solve_problem(problem, model, problem_idx)
        if correct_letter is None:
            correct_letter = solution['correct_answer']

        is_correct = solution['extracted_answer'] == solution['correct_answer'] \
            if solution['extracted_answer'] and solution['correct_answer'] else False

        result['models'][model] = {
            'answer': solution['extracted_answer'],
            'correct': is_correct,
            'solve_time': solution['solve_time']
        }

    result['gold_answer'] = correct_letter

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
    parser = argparse.ArgumentParser(description='Run GPQA on all models')
    parser.add_argument('--num-problems',
                        type=int,
                        default=None,
                        help='Number of problems (default: all)')
    parser.add_argument('--output', type=str, help='Output file')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"GPQA-Diamond Multi-Model Evaluation")
    print(f"Models: {', '.join(MODELS)}")
    print(f"{'='*60}\n")

    # Load dataset
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond")
    dataset = ds['train']  # GPQA uses 'train' for main eval
    print(f"Loaded {len(dataset)} problems")

    num_to_eval = args.num_problems if args.num_problems else len(dataset)
    problems = [dataset[i] for i in range(min(num_to_eval, len(dataset)))]
    print(f"Evaluating {len(problems)} problems\n")

    results = []
    counts = {'easy': 0, 'medium': 0, 'hard': 0, 'very_hard': 0}
    model_correct = {m: 0 for m in MODELS}

    start_time = time.time()

    for i, problem in enumerate(problems):
        result = evaluate_problem(i, problem)
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
            'dataset': 'GPQA-Diamond',
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
        output_path = f'lossy_compression/results/gpqa_all_models_{timestamp}.json'

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
