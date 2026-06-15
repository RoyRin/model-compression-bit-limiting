#!/usr/bin/env python3
"""
Run GSM8K evaluation across haiku, sonnet, and opus to classify problem difficulty.

Difficulty classification:
- easy: all models pass
- medium: haiku fails, sonnet/opus pass
- hard: haiku/sonnet fail, opus passes
- very_hard: all models fail

Usage:
    python run_gsm8k_all_models.py --num-problems 50
    python run_gsm8k_all_models.py --num-problems 100
"""

import json
import time
import argparse
import re
import os
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from datasets import load_dataset
from lossy_compression import MODEL_ALIAS_MAP, model_completion

MODELS = ['haiku', 'sonnet', 'opus']


def extract_gsm8k_answer(response):
    """Extract numerical answer from model response."""
    clean_response = response.replace('$', '').replace('¥', '').replace(
        '€', '').replace('£', '')

    patterns = [
        r'answer is[\s:]*\$?([\d,]+)',
        r'answer:[\s]*\$?([\d,]+)',
        r'=\s*\$?([\d,]+)\s*(?:$|\n)',
        r'total[s]?[\s:]+\$?([\d,]+)',
        r'therefore[\s,]+.*?\$?([\d,]+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, clean_response.lower())
        if match:
            return match.group(1).replace(',', '')

    numbers = re.findall(r'\b([\d,]+)\b', clean_response)
    if numbers:
        return numbers[-1].replace(',', '')

    return None


def solve_problem(problem_text, model_name):
    """Solve a GSM8K problem using the specified model."""
    model_full = MODEL_ALIAS_MAP.get(model_name.lower(), model_name)

    prompt = f"""Solve this math word problem step by step.

Problem: {problem_text}

Show your work and provide your final answer. End with "The answer is [number]"."""

    system = """You are a helpful math tutor solving grade school math problems.
Show your step-by-step reasoning and end with the final numerical answer.
Format your final answer clearly as: The answer is [number]."""

    start_time = time.time()
    try:
        response = model_completion(prompt,
                                    model=model_full,
                                    system=system,
                                    temperature=0.0,
                                    max_tokens=1000)
        return {
            'response': response,
            'extracted_answer': extract_gsm8k_answer(response),
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


def evaluate_problem(problem_idx, problem):
    """Evaluate a single problem across all models."""
    # Extract correct answer from the dataset
    answer_match = re.search(r'####\s*(\d+)', problem['answer'])
    correct_answer = answer_match.group(1) if answer_match else None

    result = {
        'problem_idx': problem_idx,
        'question': problem['question'],
        'gold_answer': correct_answer,
        'models': {}
    }

    for model in MODELS:
        solution = solve_problem(problem['question'], model)
        pred_answer = solution['extracted_answer']
        is_correct = pred_answer == correct_answer if pred_answer and correct_answer else False

        result['models'][model] = {
            'answer': pred_answer,
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
    parser = argparse.ArgumentParser(description='Run GSM8K on all models')
    parser.add_argument('--num-problems',
                        type=int,
                        default=None,
                        help='Number of problems (default: all)')
    parser.add_argument('--output', type=str, help='Output file')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"GSM8K Multi-Model Evaluation")
    print(f"Models: {', '.join(MODELS)}")
    print(f"{'='*60}\n")

    # Load dataset
    ds = load_dataset("openai/gsm8k", "main")
    dataset = ds['test']
    print(f"Loaded {len(dataset)} problems from test split")

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
            'dataset': 'GSM8K',
            'split': 'test',
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
        output_path = f'lossy_compression/results/gsm8k_all_models_{timestamp}.json'

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
