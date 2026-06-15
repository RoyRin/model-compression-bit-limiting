#!/usr/bin/env python3
"""
Run MBPP evaluation across haiku, sonnet, and opus to classify problem difficulty.

Difficulty classification:
- easy: all models pass
- medium: haiku fails, sonnet/opus pass
- hard: haiku/sonnet fail, opus passes
- very_hard: all models fail

Usage:
    python run_mbpp_all_models.py --num-problems 50
    python run_mbpp_all_models.py --num-problems 100 --split test
"""

import json
import time
import argparse
import re
import os
import sys
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from datasets import load_dataset
from lossy_compression import MODEL_ALIAS_MAP, model_completion

MODELS = ['haiku', 'sonnet', 'opus']


def extract_function_name(test_list):
    """Extract the expected function name from test cases."""
    if not test_list:
        return None
    match = re.search(r'assert\s+(\w+)\s*\(', test_list[0])
    return match.group(1) if match else None


def extract_code(response):
    """Extract Python code from response."""
    code_block_pattern = r'```(?:python)?\s*(.*?)```'
    matches = re.findall(code_block_pattern, response, re.DOTALL)
    if matches:
        return matches[0].strip()
    func_pattern = r'(def\s+\w+.*?)(?=\ndef\s|\Z)'
    matches = re.findall(func_pattern, response, re.DOTALL)
    if matches:
        return matches[0].strip()
    return response.strip()


def run_tests(code, test_list, test_setup_code='', timeout=10):
    """Run test cases against the generated code."""
    full_code = f"{test_setup_code}\n\n{code}\n\n"
    passed = 0

    for test in test_list:
        test_code = full_code + f"\n{test}"
        try:
            with tempfile.NamedTemporaryFile(mode='w',
                                             suffix='.py',
                                             delete=False) as f:
                f.write(test_code)
                temp_path = f.name
            result = subprocess.run(['python', temp_path],
                                    capture_output=True,
                                    text=True,
                                    timeout=timeout)
            os.unlink(temp_path)
            if result.returncode == 0:
                passed += 1
        except:
            pass

    return {
        'passed': passed,
        'total': len(test_list),
        'all_passed': passed == len(test_list)
    }


def solve_problem(problem_text, test_list, model_name):
    """Solve an MBPP problem using the specified model."""
    model_full = MODEL_ALIAS_MAP.get(model_name.lower(), model_name)

    func_name = extract_function_name(test_list)
    func_hint = f"\nName your function: {func_name}" if func_name else ""

    prompt = f"""Write a Python function to solve this problem:

{problem_text}{func_hint}

Provide only the Python code without any explanation. Include the complete function definition."""

    system = """You are a Python code generation assistant.
Write clean, correct Python code that solves the given problem.
Return only the code without any markdown formatting or explanations."""

    start_time = time.time()
    try:
        response = model_completion(prompt,
                                    model=model_full,
                                    system=system,
                                    temperature=0.0,
                                    max_tokens=1000)
        return {
            'response': response,
            'code': extract_code(response),
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


def evaluate_problem(problem_idx, problem):
    """Evaluate a single problem across all models."""
    result = {
        'problem_idx': problem_idx,
        'task_id': problem['task_id'],
        'text': problem['prompt'],  # sanitized uses 'prompt' instead of 'text'
        'models': {}
    }

    for model in MODELS:
        solution = solve_problem(problem['prompt'], problem['test_list'],
                                 model)

        if solution['code']:
            test_result = run_tests(solution['code'], problem['test_list'],
                                    problem.get('test_setup_code', ''))
            is_correct = test_result['all_passed']
            tests_passed = test_result['passed']
        else:
            is_correct = False
            tests_passed = 0

        result['models'][model] = {
            'correct': is_correct,
            'tests_passed': tests_passed,
            'tests_total': len(problem['test_list']),
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
    parser = argparse.ArgumentParser(description='Run MBPP on all models')
    parser.add_argument('--num-problems', type=int, default=50)
    parser.add_argument('--split',
                        type=str,
                        default='test',
                        choices=['train', 'test', 'validation'])
    parser.add_argument('--output', type=str, help='Output file')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"MBPP Multi-Model Evaluation")
    print(f"Split: {args.split}")
    print(f"Models: {', '.join(MODELS)}")
    print(f"{'='*60}\n")

    # Load dataset (sanitized has cleaner problem descriptions, 257 vs 500 problems)
    ds = load_dataset('google-research-datasets/mbpp', 'sanitized')
    dataset = ds[args.split]
    print(f"Loaded {len(dataset)} problems from {args.split} split")

    problems = [
        dataset[i] for i in range(min(args.num_problems, len(dataset)))
    ]
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
            'split': args.split,
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
        output_path = f'lossy_compression/results/mbpp_all_models_{args.split}_{timestamp}.json'

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
