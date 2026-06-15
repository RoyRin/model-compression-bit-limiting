#!/usr/bin/env python3
"""
Run HumanEval evaluation across haiku, sonnet, and opus to classify problem difficulty.

Difficulty classification:
- easy: all models pass
- medium: haiku fails, sonnet/opus pass
- hard: haiku/sonnet fail, opus passes
- very_hard: all models fail

Usage:
    python run_humaneval_all_models.py --num-problems 50
    python run_humaneval_all_models.py  # All 164 problems
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

from evalplus.data import get_human_eval_plus
from evalplus.evaluate import check_correctness, get_groundtruth
from lossy_compression import MODEL_ALIAS_MAP, model_completion

MODELS = ['haiku', 'sonnet', 'opus']


def extract_code(response):
    """Extract Python code from response."""
    if not response or len(response.strip()) == 0:
        return None

    code_pattern = r'```(?:python)?\s*\n(.*?)\n```'
    matches = re.findall(code_pattern, response, re.DOTALL)

    if matches:
        return matches[0].strip()

    return response.strip()


def solve_problem(problem, model_name):
    """Solve a HumanEval problem using the specified model."""
    model_full = MODEL_ALIAS_MAP.get(model_name.lower(), model_name)

    prompt = f"""Complete this Python function. Return only the complete function implementation without any explanation.

{problem['prompt']}"""

    system = """You are a Python code completion assistant.
Complete the given Python function by providing the full implementation including the function signature.
Return only valid Python code without any markdown formatting, explanations, or additional text."""

    start_time = time.time()
    try:
        response = model_completion(prompt,
                                    model=model_full,
                                    system=system,
                                    temperature=0.0,
                                    max_tokens=2048)
        code = extract_code(response)
        return {
            'response': response,
            'code': code,
            'solve_time': time.time() - start_time,
            'success': True
        }
    except Exception as e:
        return {
            'response': str(e),
            'code': None,
            'solve_time': time.time() - start_time,
            'success': False,
            'error': str(e)
        }


def check_solution(task_id, solution_code, problem, expected_output):
    """Check if a solution passes the test cases."""
    if not solution_code:
        return False

    try:
        result = check_correctness(dataset="humaneval",
                                   completion_id=0,
                                   expected_output=expected_output,
                                   problem=problem,
                                   solution=solution_code,
                                   base_only=False,
                                   fast_check=True)
        return result['base'][0] == 'passed' or result.get(
            'plus', [None])[0] == 'passed'
    except Exception as e:
        return False


def evaluate_problem(task_id, problem, expected_output):
    """Evaluate a single problem across all models."""
    result = {
        'task_id': task_id,
        'entry_point': problem['entry_point'],
        'models': {}
    }

    for model in MODELS:
        solution = solve_problem(problem, model)

        if solution['code']:
            is_correct = check_solution(task_id, solution['code'], problem,
                                        expected_output)
        else:
            is_correct = False

        result['models'][model] = {
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
    parser = argparse.ArgumentParser(description='Run HumanEval on all models')
    parser.add_argument('--num-problems',
                        type=int,
                        default=None,
                        help='Number of problems (default: all 164)')
    parser.add_argument('--output', type=str, help='Output file')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"HumanEval Multi-Model Evaluation")
    print(f"Models: {', '.join(MODELS)}")
    print(f"{'='*60}\n")

    # Load dataset
    problems = get_human_eval_plus()
    task_ids = list(problems.keys())
    print(f"Loaded {len(task_ids)} problems")

    # Compute expected outputs (required for check_correctness)
    print("Computing expected outputs...")
    expected_outputs = get_groundtruth(problems, "humaneval", None)
    print(f"Expected outputs computed for {len(expected_outputs)} problems")

    num_to_eval = args.num_problems if args.num_problems else len(task_ids)
    task_ids = task_ids[:num_to_eval]
    print(f"Evaluating {len(task_ids)} problems\n")

    results = []
    counts = {'easy': 0, 'medium': 0, 'hard': 0, 'very_hard': 0}
    model_correct = {m: 0 for m in MODELS}

    start_time = time.time()

    for i, task_id in enumerate(task_ids):
        problem = problems[task_id]
        expected_output = expected_outputs[task_id]
        result = evaluate_problem(task_id, problem, expected_output)
        results.append(result)

        counts[result['difficulty']] += 1
        for m in MODELS:
            if result['models'][m]['correct']:
                model_correct[m] += 1

        # Progress
        elapsed = time.time() - start_time
        print(
            f"\rProblem {i+1}/{len(task_ids)} | "
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
            'dataset': 'HumanEval+',
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
        output_path = f'lossy_compression/results/humaneval_all_models_{timestamp}.json'

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
