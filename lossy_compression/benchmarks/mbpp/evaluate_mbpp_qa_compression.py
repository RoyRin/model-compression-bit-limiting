#!/usr/bin/env python3
"""
Evaluate MBPP problems using SLM question-answering compression.

Usage:
    # Test on medium problems (haiku fails, sonnet/opus pass)
    python evaluate_mbpp_qa_compression.py --difficulty medium --num-problems 10

    # Test with specific models
    python evaluate_mbpp_qa_compression.py --difficulty medium --slm haiku --llm opus

    # Test on specific indices
    python evaluate_mbpp_qa_compression.py --indices 5 10 15 --verbose
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
from lossy_compression.core.qa_compression import iterative_SLM_loop, EVAL_MODE_CODE
from lossy_compression import MODEL_ALIAS_MAP


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


def get_problem_difficulty_indices(baseline_results_path):
    """Load problem indices categorized by difficulty from baseline results."""
    with open(baseline_results_path, 'r') as f:
        data = json.load(f)

    categories = {'easy': [], 'medium': [], 'hard': [], 'very_hard': []}

    for result in data.get('results', []):
        difficulty = result.get('difficulty', 'unknown')
        if difficulty in categories:
            categories[difficulty].append(result['problem_idx'])

    return categories


def solve_problem_with_qa(problem_text,
                          test_list,
                          slm_model,
                          llm_model,
                          question_model,
                          max_questions=30,
                          batch_mode=True,
                          batch_size=10,
                          verbose=False):
    """Solve an MBPP problem using Q&A compression."""

    func_name = extract_function_name(test_list)
    func_hint = f"\nName your function: {func_name}" if func_name else ""

    prompt = f"""Write a Python function to solve this problem:

{problem_text}{func_hint}

Provide only the Python code without any explanation. Include the complete function definition."""

    start_time = time.time()

    try:
        result = iterative_SLM_loop(prompt=prompt,
                                    large_model_name=llm_model,
                                    small_model_name=slm_model,
                                    question_model_name=question_model,
                                    max_iterations=max_questions,
                                    verbose=verbose,
                                    batch_mode=batch_mode,
                                    batch_size=batch_size,
                                    evaluation_mode=EVAL_MODE_CODE)

        final_answer, qa_tuple, metrics = result
        guiding_questions, guiding_answers = qa_tuple

        solve_time = time.time() - start_time
        code = extract_code(final_answer)
        qa_pairs = list(zip(guiding_questions, guiding_answers))

        return {
            'response': final_answer,
            'code': code,
            'solve_time': solve_time,
            'num_questions': len(guiding_questions),
            'qa_pairs': qa_pairs if verbose else None,
            'metrics': metrics,
            'success': True
        }

    except Exception as e:
        import traceback
        if verbose:
            print(f"Error solving problem: {e}")
            traceback.print_exc()
        return {
            'response': str(e),
            'code': None,
            'solve_time': time.time() - start_time,
            'num_questions': 0,
            'qa_pairs': None,
            'metrics': None,
            'success': False,
            'error': str(e)
        }


def evaluate_qa_compression(slm_model,
                            llm_model,
                            question_model,
                            problem_indices,
                            num_problems=None,
                            max_questions=30,
                            batch_mode=True,
                            batch_size=10,
                            verbose=False):
    """Evaluate Q&A compression on specific MBPP problems."""

    slm_full = MODEL_ALIAS_MAP.get(slm_model.lower(), slm_model)
    llm_full = MODEL_ALIAS_MAP.get(llm_model.lower(), llm_model)
    question_full = MODEL_ALIAS_MAP.get(question_model.lower(), question_model)

    print(f"\n{'='*60}")
    print(f"Evaluating MBPP with Q&A Compression")
    print(f"SLM: {slm_model} ({slm_full})")
    print(f"LLM: {llm_model} ({llm_full})")
    print(f"Question Model: {question_model} ({question_full})")
    print(f"Max questions: {max_questions}")
    if batch_mode:
        print(f"Batch mode: Enabled (size={batch_size})")
    print(f"{'='*60}")

    # Load MBPP dataset (sanitized)
    ds = load_dataset('google-research-datasets/mbpp', 'sanitized')
    dataset = ds['test']

    if num_problems is None:
        selected_indices = problem_indices
    else:
        selected_indices = problem_indices[:num_problems]

    results = []
    correct_count = 0
    overall_start_time = time.time()

    for i, idx in enumerate(selected_indices):
        problem = dataset[idx]

        print(
            f"\rEvaluating problem {i+1}/{len(selected_indices)} (index {idx})",
            end="",
            flush=True)

        solution = solve_problem_with_qa(problem['prompt'],
                                         problem['test_list'],
                                         slm_full,
                                         llm_full,
                                         question_full,
                                         max_questions=max_questions,
                                         batch_mode=batch_mode,
                                         batch_size=batch_size,
                                         verbose=verbose)

        # Test the code
        if solution['code']:
            test_result = run_tests(solution['code'], problem['test_list'],
                                    problem.get('test_setup_code', ''))
            is_correct = test_result['all_passed']
            tests_passed = test_result['passed']
        else:
            is_correct = False
            tests_passed = 0

        if is_correct:
            correct_count += 1

        result = {
            'problem_idx': idx,
            'task_id': problem['task_id'],
            'is_correct': is_correct,
            'tests_passed': tests_passed,
            'tests_total': len(problem['test_list']),
            'solve_time': solution['solve_time'],
            'num_questions': solution['num_questions'],
            'metrics': solution['metrics'],
            'full_response': solution['response'] if verbose else None
        }

        results.append(result)

        # Update status
        elapsed_time = time.time() - overall_start_time
        acc_pct = 100 * correct_count / len(results)
        avg_time = elapsed_time / len(results)
        eta = avg_time * (len(selected_indices) - len(results))
        status = f"\rProblem {i+1}/{len(selected_indices)} | Acc: {correct_count}/{len(results)} ({acc_pct:.1f}%) | Q: {solution['num_questions']} | ETA: {eta:.0f}s"
        print(status, end="", flush=True)

    print()

    total_elapsed_time = time.time() - overall_start_time
    accuracy = correct_count / len(results) if results else 0
    avg_time = sum(r['solve_time']
                   for r in results) / len(results) if results else 0
    avg_questions = sum(r['num_questions']
                        for r in results) / len(results) if results else 0

    return {
        'slm': slm_model,
        'llm': llm_model,
        'question_model': question_model,
        'slm_full': slm_full,
        'llm_full': llm_full,
        'question_model_full': question_full,
        'total_problems': len(results),
        'correct_count': correct_count,
        'accuracy': accuracy,
        'avg_solve_time': avg_time,
        'avg_questions': avg_questions,
        'total_elapsed_time': total_elapsed_time,
        'results': results
    }


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate MBPP with Q&A compression')
    parser.add_argument('--slm',
                        type=str,
                        default='haiku',
                        help='Small language model (default: haiku)')
    parser.add_argument('--llm',
                        type=str,
                        default='opus',
                        help='Large language model (default: opus)')
    parser.add_argument('--question-model',
                        type=str,
                        default='haiku',
                        help='Question generation model (default: haiku)')
    parser.add_argument('--difficulty',
                        type=str,
                        default='medium',
                        choices=[
                            'easy', 'medium', 'hard', 'very_hard',
                            'medium+hard', 'not_easy'
                        ],
                        help='Problem difficulty to test (default: medium)')
    parser.add_argument('--num-problems',
                        type=int,
                        default=None,
                        help='Number of problems to evaluate (default: all)')
    parser.add_argument('--max-questions',
                        type=int,
                        default=30,
                        help='Maximum questions to ask (default: 30)')
    parser.add_argument('--batch',
                        action='store_true',
                        help='Enable batch mode for Q&A generation')
    parser.add_argument('--batch-size',
                        type=int,
                        default=10,
                        help='Batch size for questions (default: 10)')
    parser.add_argument(
        '--baseline-results',
        type=str,
        default=
        'lossy_compression/results/mbpp_all_models_test_20260115_163950.json',
        help='Path to baseline results for categorization')
    parser.add_argument('--verbose',
                        action='store_true',
                        help='Show detailed output')
    parser.add_argument('--output', type=str, help='Output file for results')
    parser.add_argument('--indices',
                        type=int,
                        nargs='+',
                        default=None,
                        help='Specific problem indices to test')

    args = parser.parse_args()

    if args.indices:
        problem_indices = args.indices
        difficulty_label = 'custom'
        print(f"Using {len(problem_indices)} custom problem indices")
    else:
        print(f"Loading problem categorization from: {args.baseline_results}")
        if not Path(args.baseline_results).exists():
            print(
                f"Error: Baseline results not found at {args.baseline_results}"
            )
            print("Please run run_mbpp_all_models.py first.")
            return

        categories = get_problem_difficulty_indices(args.baseline_results)

        if args.difficulty == 'medium+hard':
            problem_indices = categories['medium'] + categories['hard']
            difficulty_label = 'medium+hard'
        elif args.difficulty == 'not_easy':
            problem_indices = categories['medium'] + categories[
                'hard'] + categories['very_hard']
            difficulty_label = 'not_easy'
        else:
            problem_indices = categories[args.difficulty]
            difficulty_label = args.difficulty

        if not problem_indices:
            print(f"No {args.difficulty} problems found.")
            return

        print(f"Found {len(problem_indices)} {difficulty_label} problems")

    if args.num_problems is None:
        actual_num_problems = len(problem_indices)
    else:
        actual_num_problems = min(args.num_problems, len(problem_indices))

    print(f"Will evaluate {actual_num_problems} problems")

    results = evaluate_qa_compression(slm_model=args.slm,
                                      llm_model=args.llm,
                                      question_model=args.question_model,
                                      problem_indices=problem_indices,
                                      num_problems=actual_num_problems,
                                      max_questions=args.max_questions,
                                      batch_mode=args.batch,
                                      batch_size=args.batch_size,
                                      verbose=args.verbose)

    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Difficulty: {difficulty_label}")
    print(
        f"Accuracy: {results['accuracy']:.1%} ({results['correct_count']}/{results['total_problems']})"
    )
    print(f"Avg solve time: {results['avg_solve_time']:.2f}s")
    print(f"Avg questions used: {results['avg_questions']:.1f}")

    # Save results
    if args.output:
        output_path = args.output
    else:
        os.makedirs('lossy_compression/results', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f'lossy_compression/results/mbpp_qa_{args.slm}_{args.llm}_{difficulty_label}_{timestamp}.json'

    save_data = results.copy()
    save_data['config'] = {
        'difficulty': args.difficulty,
        'max_questions': args.max_questions,
        'batch_mode': args.batch,
        'batch_size': args.batch_size if args.batch else None,
        'baseline_results': args.baseline_results,
        'num_problems_requested': args.num_problems,
        'evaluation_mode': 'EVAL_MODE_CODE'
    }
    save_data['metadata'] = {
        'timestamp': datetime.now().isoformat(),
        'output_file': output_path,
        'dataset': {
            'name': 'MBPP',
            'config': 'sanitized',
            'split': 'test',
            'difficulty': difficulty_label,
            'problems_evaluated': results['total_problems']
        }
    }

    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
