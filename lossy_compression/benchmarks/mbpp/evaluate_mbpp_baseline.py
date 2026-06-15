#!/usr/bin/env python3
"""
Evaluate MBPP dataset baseline (no Q&A compression).

Usage:
    python evaluate_mbpp_baseline.py --model haiku --num-problems 10
    python evaluate_mbpp_baseline.py --model opus --num-problems 50
"""

import json
import time
import argparse
import re
import os
import sys
from pathlib import Path
from datetime import datetime
import tempfile
import subprocess

# Add parent paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from datasets import load_dataset

from lossy_compression import MODEL_ALIAS_MAP, model_completion


def extract_code(response):
    """Extract Python code from response."""
    # Try to find code in markdown blocks
    code_block_pattern = r'```(?:python)?\s*(.*?)```'
    matches = re.findall(code_block_pattern, response, re.DOTALL)
    if matches:
        return matches[0].strip()

    # If no code blocks, try to find function definition
    func_pattern = r'(def\s+\w+.*?)(?=\ndef\s|\Z)'
    matches = re.findall(func_pattern, response, re.DOTALL)
    if matches:
        return matches[0].strip()

    # Fallback: return cleaned response
    return response.strip()


def run_tests(code, test_list, test_setup_code='', timeout=10):
    """Run test cases against the generated code."""
    # Combine setup code, generated code, and tests
    full_code = f"{test_setup_code}\n\n{code}\n\n"

    # Run each test
    passed = 0
    failed = 0
    errors = []

    for test in test_list:
        test_code = full_code + f"\n{test}"

        try:
            # Write to temp file and execute
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
            else:
                failed += 1
                errors.append(
                    f"Test failed: {test}\nError: {result.stderr[:200]}")

        except subprocess.TimeoutExpired:
            failed += 1
            errors.append(f"Test timed out: {test}")
        except Exception as e:
            failed += 1
            errors.append(f"Test error: {test}\n{str(e)[:200]}")

    return {
        'passed': passed,
        'failed': failed,
        'total': len(test_list),
        'all_passed': passed == len(test_list),
        'errors': errors[:3]  # Keep first 3 errors
    }


def extract_function_name(test_list):
    """Extract the expected function name from test cases."""
    if not test_list:
        return None
    # First test usually has the function call
    test = test_list[0]
    # Look for function call pattern
    match = re.search(r'assert\s+(\w+)\s*\(', test)
    if match:
        return match.group(1)
    return None


def solve_mbpp_problem(problem_text, test_list, model_name, verbose=False):
    """Solve an MBPP problem using the specified model."""

    # Extract expected function name from tests
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
                                    model=model_name,
                                    system=system,
                                    temperature=0.0,
                                    max_tokens=1000)

        solve_time = time.time() - start_time
        code = extract_code(response)

        return {
            'response': response,
            'code': code,
            'solve_time': solve_time,
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


def evaluate_baseline(model_name,
                      num_problems=None,
                      split='test',
                      verbose=False):
    """Evaluate baseline model on MBPP dataset."""

    # Resolve model alias
    model_full = MODEL_ALIAS_MAP.get(model_name.lower(), model_name)

    print(f"\n{'='*60}")
    print(f"Evaluating MBPP Baseline")
    print(f"Model: {model_name} ({model_full})")
    print(f"Split: {split}")
    print(f"{'='*60}")

    # Load dataset
    ds = load_dataset('google-research-datasets/mbpp', 'full')
    dataset = ds[split]
    print(f"Loaded {len(dataset)} problems from {split} split")

    # Select problems
    if num_problems:
        indices = list(range(min(num_problems, len(dataset))))
    else:
        indices = list(range(len(dataset)))

    print(f"Evaluating {len(indices)} problems")

    results = []
    correct_count = 0
    start_time = time.time()

    for i, idx in enumerate(indices):
        problem = dataset[idx]

        # Generate solution
        solution = solve_mbpp_problem(problem['text'], problem['test_list'],
                                      model_full, verbose)

        # Run tests
        if solution['code']:
            test_result = run_tests(solution['code'], problem['test_list'],
                                    problem.get('test_setup_code', ''))
            is_correct = test_result['all_passed']
        else:
            test_result = {
                'passed': 0,
                'failed': len(problem['test_list']),
                'total': len(problem['test_list']),
                'all_passed': False,
                'errors': ['No code generated']
            }
            is_correct = False

        if is_correct:
            correct_count += 1

        result = {
            'problem_id':
            problem['task_id'],
            'text':
            problem['text'][:150] +
            '...' if len(problem['text']) > 150 else problem['text'],
            'is_correct':
            is_correct,
            'tests_passed':
            test_result['passed'],
            'tests_total':
            test_result['total'],
            'solve_time':
            solution['solve_time'],
            'errors':
            test_result.get('errors', [])
        }
        results.append(result)

        # Progress update
        elapsed = time.time() - start_time
        acc = correct_count / len(results) * 100
        print(
            f"\rProblem {i+1}/{len(indices)} | Acc: {correct_count}/{len(results)} ({acc:.1f}%) | "
            f"Tests: {test_result['passed']}/{test_result['total']} | "
            f"Time: {solution['solve_time']:.1f}s | Elapsed: {elapsed:.1f}s",
            end='',
            flush=True)

        if verbose:
            status = '✓' if is_correct else '✗'
            print(
                f"\n  {status} Task {problem['task_id']}: {test_result['passed']}/{test_result['total']} tests"
            )
            if test_result.get('errors'):
                print(f"    Error: {test_result['errors'][0][:100]}...")

    print()  # Newline after progress

    # Summary
    accuracy = correct_count / len(results) if results else 0
    total_time = time.time() - start_time

    summary = {
        'model': model_name,
        'model_full': model_full,
        'split': split,
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
    print(f"Split: {split}")
    print(f"Accuracy: {accuracy:.1%} ({correct_count}/{len(results)})")
    print(f"Total time: {total_time:.1f}s")
    print(f"Avg time per problem: {total_time/len(results):.1f}s")

    return summary


def main():
    parser = argparse.ArgumentParser(description='Evaluate MBPP baseline')
    parser.add_argument('--model',
                        type=str,
                        default='haiku',
                        help='Model to evaluate (default: haiku)')
    parser.add_argument('--num-problems',
                        type=int,
                        default=None,
                        help='Number of problems to evaluate')
    parser.add_argument('--split',
                        type=str,
                        default='test',
                        choices=['train', 'test', 'validation', 'prompt'],
                        help='Dataset split (default: test)')
    parser.add_argument('--verbose',
                        action='store_true',
                        help='Show detailed output')
    parser.add_argument('--output', type=str, help='Output file for results')

    args = parser.parse_args()

    results = evaluate_baseline(model_name=args.model,
                                num_problems=args.num_problems,
                                split=args.split,
                                verbose=args.verbose)

    # Save results
    if args.output:
        output_path = args.output
    else:
        os.makedirs('lossy_compression/results', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f'lossy_compression/results/mbpp_baseline_{args.model}_{args.split}_{timestamp}.json'

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
