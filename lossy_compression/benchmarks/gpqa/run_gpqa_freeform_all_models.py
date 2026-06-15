#!/usr/bin/env python3
"""
Run GPQA-Freeform evaluation across haiku, sonnet, and opus to classify problem difficulty.

Uses free-form answers (no multiple choice) and LLM-as-judge for evaluation.

Difficulty classification:
- easy: all models pass
- medium: haiku fails, sonnet/opus pass
- hard: haiku/sonnet fail, opus passes
- very_hard: all models fail

Usage:
    python run_gpqa_freeform_all_models.py --num-problems 50
    python run_gpqa_freeform_all_models.py  # All problems
    python run_gpqa_freeform_all_models.py --judge-model sonnet
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


def solve_problem(question, model_name):
    """Solve a GPQA problem using the specified model (free-form answer)."""
    model_full = MODEL_ALIAS_MAP.get(model_name.lower(), model_name)

    prompt = f"""Answer this science question. Provide your reasoning, then give a clear, concise final answer.

Question: {question}

Think through this step-by-step, then provide your final answer."""

    system = """You are an expert scientist with deep knowledge across physics, chemistry, biology, and other scientific domains.
Analyze questions carefully, show your reasoning, and provide clear, accurate answers.
End with a clear statement of your final answer."""

    start_time = time.time()
    try:
        response = model_completion(prompt,
                                    model=model_full,
                                    system=system,
                                    temperature=0.0,
                                    max_tokens=2000)
        return {
            'response': response,
            'solve_time': time.time() - start_time,
            'success': True
        }
    except Exception as e:
        return {
            'response': str(e),
            'solve_time': time.time() - start_time,
            'success': False,
            'error': str(e)
        }


def judge_answer(question, gold_answer, model_answer, judge_model_name):
    """Use LLM-as-judge to evaluate if the model's answer is correct."""
    judge_model_full = MODEL_ALIAS_MAP.get(judge_model_name.lower(),
                                           judge_model_name)

    prompt = f"""You are evaluating whether a student's answer to a science question is correct.

QUESTION:
{question}

CORRECT ANSWER:
{gold_answer}

STUDENT'S ANSWER:
{model_answer}

Evaluate whether the student's answer is essentially correct. The student's answer does not need to match the correct answer word-for-word, but it must convey the same meaning and be factually accurate.

Consider:
1. Does the student's answer capture the key concept/fact in the correct answer?
2. Is the student's answer factually accurate?
3. Would the student receive credit for this answer in an exam setting?

Respond with ONLY one of the following:
- CORRECT: if the answer is essentially correct
- INCORRECT: if the answer is wrong or missing key information
- PARTIAL: if the answer is partially correct but missing important details

Then briefly explain your reasoning in one sentence."""

    system = """You are a fair and accurate grader for science questions.
Evaluate answers based on correctness of the core concept, not exact wording.
Be strict but fair - give credit for correct answers even if phrased differently."""

    try:
        response = model_completion(prompt,
                                    model=judge_model_full,
                                    system=system,
                                    temperature=0.0,
                                    max_tokens=200)

        # Parse the judgment
        response_upper = response.upper()
        if response_upper.startswith('CORRECT'):
            return {
                'judgment': 'correct',
                'reasoning': response,
                'success': True
            }
        elif response_upper.startswith('INCORRECT'):
            return {
                'judgment': 'incorrect',
                'reasoning': response,
                'success': True
            }
        elif response_upper.startswith('PARTIAL'):
            return {
                'judgment': 'partial',
                'reasoning': response,
                'success': True
            }
        else:
            # Try to find judgment in response
            if 'CORRECT' in response_upper and 'INCORRECT' not in response_upper:
                return {
                    'judgment': 'correct',
                    'reasoning': response,
                    'success': True
                }
            elif 'INCORRECT' in response_upper:
                return {
                    'judgment': 'incorrect',
                    'reasoning': response,
                    'success': True
                }
            elif 'PARTIAL' in response_upper:
                return {
                    'judgment': 'partial',
                    'reasoning': response,
                    'success': True
                }
            else:
                return {
                    'judgment': 'unknown',
                    'reasoning': response,
                    'success': False
                }
    except Exception as e:
        return {'judgment': 'error', 'reasoning': str(e), 'success': False}


def evaluate_problem(problem_idx, problem, judge_model):
    """Evaluate a single problem across all models."""
    question = problem['question']
    gold_answer = problem['answer']
    category = problem.get('category', 'unknown')

    result = {
        'problem_idx': problem_idx,
        'question':
        question[:300] + '...' if len(question) > 300 else question,
        'gold_answer': gold_answer,
        'category': category,
        'models': {}
    }

    for model in MODELS:
        # Get model's answer
        solution = solve_problem(question, model)

        if solution['success']:
            # Judge the answer
            judgment = judge_answer(question, gold_answer,
                                    solution['response'], judge_model)
            is_correct = judgment['judgment'] == 'correct'
            is_partial = judgment['judgment'] == 'partial'
        else:
            judgment = {
                'judgment': 'error',
                'reasoning': solution.get('error', 'Unknown error')
            }
            is_correct = False
            is_partial = False

        result['models'][model] = {
            'response':
            solution['response'][:500] +
            '...' if len(solution['response']) > 500 else solution['response'],
            'judgment':
            judgment['judgment'],
            'correct':
            is_correct,
            'partial':
            is_partial,
            'solve_time':
            solution['solve_time']
        }

    # Classify difficulty (treating partial as incorrect for difficulty classification)
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
    parser = argparse.ArgumentParser(
        description='Run GPQA-Freeform on all models with LLM-as-judge')
    parser.add_argument('--num-problems',
                        type=int,
                        default=None,
                        help='Number of problems (default: all)')
    parser.add_argument('--judge-model',
                        type=str,
                        default='sonnet',
                        choices=['haiku', 'sonnet', 'opus'],
                        help='Model to use as judge (default: sonnet)')
    parser.add_argument('--output', type=str, help='Output file')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"GPQA-Freeform Multi-Model Evaluation")
    print(f"Models: {', '.join(MODELS)}")
    print(f"Judge Model: {args.judge_model}")
    print(f"{'='*60}\n")

    # Load dataset
    dataset = load_dataset("nikhilchandak/freeform-datasets",
                           split="gpqa_diamond")
    print(f"Loaded {len(dataset)} problems from GPQA-Diamond (freeform)")

    num_to_eval = args.num_problems if args.num_problems else len(dataset)
    problems = [dataset[i] for i in range(min(num_to_eval, len(dataset)))]
    print(f"Evaluating {len(problems)} problems\n")

    results = []
    counts = {'easy': 0, 'medium': 0, 'hard': 0, 'very_hard': 0}
    model_correct = {m: 0 for m in MODELS}
    model_partial = {m: 0 for m in MODELS}

    # Track by category
    category_stats = {}

    start_time = time.time()

    for i, problem in enumerate(problems):
        result = evaluate_problem(i, problem, args.judge_model)
        results.append(result)

        counts[result['difficulty']] += 1

        # Track category
        cat = result['category']
        if cat not in category_stats:
            category_stats[cat] = {
                'total': 0,
                'correct': {
                    m: 0
                    for m in MODELS
                }
            }
        category_stats[cat]['total'] += 1

        for m in MODELS:
            if result['models'][m]['correct']:
                model_correct[m] += 1
                category_stats[cat]['correct'][m] += 1
            if result['models'][m]['partial']:
                model_partial[m] += 1

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
            'dataset': 'GPQA-Diamond-Freeform',
            'source': 'nikhilchandak/freeform-datasets',
            'num_problems': total,
            'models': MODELS,
            'judge_model': args.judge_model
        },
        'model_accuracy': {
            m: model_correct[m] / total
            for m in MODELS
        },
        'model_partial': {
            m: model_partial[m] / total
            for m in MODELS
        },
        'difficulty_distribution': {
            d: counts[d] / total
            for d in counts
        },
        'difficulty_counts': counts,
        'category_stats': category_stats,
        'results': results
    }

    print(f"{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"\nModel Accuracy (strict):")
    for m in MODELS:
        print(
            f"  {m}: {model_correct[m]}/{total} ({model_correct[m]/total:.1%})"
        )

    print(f"\nModel Partial Credit:")
    for m in MODELS:
        print(
            f"  {m}: {model_partial[m]}/{total} ({model_partial[m]/total:.1%})"
        )

    print(f"\nDifficulty Distribution:")
    for d, c in counts.items():
        print(f"  {d}: {c}/{total} ({c/total:.1%})")

    if category_stats:
        print(f"\nAccuracy by Category:")
        for cat, stats in sorted(category_stats.items()):
            if stats['total'] > 0:
                print(f"  {cat} (n={stats['total']}):")
                for m in MODELS:
                    acc = stats['correct'][m] / stats['total']
                    print(f"    {m}: {acc:.1%}")

    # Save
    if args.output:
        output_path = args.output
    else:
        os.makedirs('lossy_compression/results', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f'lossy_compression/results/gpqa_freeform_all_models_{timestamp}.json'

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
