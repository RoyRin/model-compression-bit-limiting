#!/usr/bin/env python3
"""
Evaluate MATH problems using SLM question-answering compression (20 Questions).

Usage:
    # Test on easy algebra problems
    python evaluate_math_qa_compression.py --subject algebra --difficulty easy --num-problems 10

    # Test on medium problems with specific models
    python evaluate_math_qa_compression.py --subject algebra --difficulty medium --slm haiku --llm opus

    # Test on all difficulties
    python evaluate_math_qa_compression.py --subject algebra --difficulty all --num-problems 5

    # Use specific indices directly
    python evaluate_math_qa_compression.py --subject algebra --indices 0 1 2 3 4
"""

import json
import time
import argparse
import re
import os
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, desc=None, **kwargs):
        total = len(iterable) if hasattr(iterable, '__len__') else None
        for i, item in enumerate(iterable):
            if desc and total:
                print(f"\r{desc}: {i+1}/{total}", end="", flush=True)
            yield item
        if desc:
            print()


from lossy_compression.core.qa_compression import iterative_SLM_loop, EVAL_MODE_MATH
from lossy_compression import MODEL_ALIAS_MAP


def extract_math_answer(response):
    """Extract numerical/mathematical answer from model response."""
    if not response:
        return None

    # Look for boxed answers first (MATH format) - handle nested braces
    # Find \boxed{ and then match balanced braces
    boxed_start = response.rfind('\\boxed{')
    if boxed_start != -1:
        # Find matching closing brace
        depth = 0
        start_idx = boxed_start + 7  # len('\\boxed{')
        for i, c in enumerate(response[start_idx:], start_idx):
            if c == '{':
                depth += 1
            elif c == '}':
                if depth == 0:
                    return response[start_idx:i].strip()
                depth -= 1

    # Simple regex fallback for non-nested boxed
    boxed_pattern = r'\\boxed\{([^{}]+)\}'
    boxed_match = re.search(boxed_pattern, response)
    if boxed_match:
        return boxed_match.group(1).strip()

    # Look for common answer patterns
    patterns = [
        r'answer is[\s:]*\$?([^\n,]+)',
        r'answer:[\s]*\$?([^\n,]+)',
        r'therefore[\s,]+.*?(\d+)',
        r'=\s*(\d+)\s*(?:$|\n)',
    ]

    for pattern in patterns:
        match = re.search(pattern, response.lower())
        if match:
            return match.group(1).strip()

    # Fallback: find the last standalone number
    numbers = re.findall(r'\b(\d+)\b', response)
    if numbers:
        return numbers[-1]

    return None


def normalize_answer(answer):
    """Normalize answer for comparison."""
    if answer is None:
        return None

    answer = str(answer).strip()

    # Remove common prefixes like "x = " or "h^{-1}(x) = "
    answer = re.sub(r'^[a-zA-Z_\^\{\}\-\d\(\)]+\s*=\s*', '', answer)

    # Remove LaTeX formatting
    answer = answer.replace('$', '')
    answer = re.sub(r'\s+', '', answer)  # Remove whitespace

    # Normalize fractions: \frac{a}{b} -> a/b
    answer = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'(\1)/(\2)', answer)

    # Remove remaining backslashes
    answer = answer.replace('\\', '')

    # Try to extract just the number if it's a simple numeric answer
    num_match = re.match(r'^-?\d+\.?\d*$', answer)
    if num_match:
        return num_match.group(0)

    return answer.lower()


def get_problem_difficulty_indices(baseline_path, subject):
    """Get problem indices categorized by difficulty from baseline results."""
    with open(baseline_path) as f:
        data = json.load(f)

    indices = {'easy': [], 'medium': [], 'hard': [], 'very_hard': []}

    for result in data.get('results', []):
        difficulty = result.get('difficulty', 'unknown')
        if difficulty in indices:
            indices[difficulty].append(result['problem_idx'])

    return indices


def load_existing_results(resume_path):
    """Load existing results and identify which problems failed.

    A problem is considered failed if:
    - metrics is None
    - model_answer is None
    - has an 'error' key

    Returns:
        tuple: (existing_results dict by problem_idx, set of failed problem indices, metadata)
    """
    with open(resume_path) as f:
        data = json.load(f)

    existing_results = {}
    failed_indices = set()

    for result in data.get('results', []):
        idx = result['problem_idx']
        existing_results[idx] = result

        # Check if this problem failed
        if (result.get('metrics') is None or result.get('model_answer') is None
                or result.get('error') is not None):
            failed_indices.add(idx)

    # Extract metadata for merging later
    metadata = {k: v for k, v in data.items() if k != 'results'}

    return existing_results, failed_indices, metadata


def merge_results(existing_results, new_results, failed_indices):
    """Merge new results into existing results, replacing failed ones.

    Args:
        existing_results: dict mapping problem_idx -> result
        new_results: list of new result dicts
        failed_indices: set of indices that were re-run

    Returns:
        list of merged results sorted by problem_idx
    """
    merged = dict(existing_results)  # Copy existing

    # Replace failed results with new ones
    for result in new_results:
        idx = result['problem_idx']
        if idx in failed_indices:
            merged[idx] = result

    # Sort by problem index and return as list
    return [merged[idx] for idx in sorted(merged.keys())]


def solve_problem_with_qa(problem_text,
                          slm_model,
                          llm_model,
                          question_model,
                          max_questions=30,
                          batch_mode=True,
                          batch_size=10,
                          oracle_solution=None,
                          verbose=False):
    """Solve a MATH problem using Q&A compression.

    Args:
        oracle_solution: If provided, use this as the reference solution instead of
                        generating one with the LLM. This creates an "oracle" mode
                        where the LLM has perfect knowledge of the correct answer.
    """

    prompt = f"""Solve this math problem step by step and provide a clear answer.

Problem: {problem_text}

Show your work and provide your final answer using \\boxed{{answer}} notation."""

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
                                    evaluation_mode=EVAL_MODE_MATH,
                                    oracle_solution=oracle_solution)

        final_answer, qa_tuple, metrics = result
        guiding_questions, guiding_answers = qa_tuple

        solve_time = time.time() - start_time
        extracted_answer = extract_math_answer(final_answer)
        qa_pairs = list(zip(guiding_questions, guiding_answers))

        return {
            'response': final_answer,
            'extracted_answer': extracted_answer,
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
            'extracted_answer': None,
            'solve_time': time.time() - start_time,
            'num_questions': 0,
            'qa_pairs': None,
            'metrics': None,
            'success': False,
            'error': str(e)
        }


def _process_single_problem(args):
    """Worker function for parallel execution."""
    (idx, problem_text, correct_answer, gold_answer, slm_full, llm_full,
     question_full, max_questions, batch_mode, batch_size, oracle_mode,
     verbose) = args

    solution = solve_problem_with_qa(
        problem_text,
        slm_full,
        llm_full,
        question_full,
        max_questions=max_questions,
        batch_mode=batch_mode,
        batch_size=batch_size,
        oracle_solution=correct_answer if oracle_mode else None,
        verbose=verbose)

    # Check if correct
    is_correct = False
    if solution['extracted_answer'] and gold_answer:
        norm_model = normalize_answer(solution['extracted_answer'])
        norm_gold = normalize_answer(gold_answer)
        is_correct = norm_model == norm_gold

    return {
        'problem_idx': idx,
        'problem': problem_text,
        'gold_answer': gold_answer,
        'model_answer': solution['extracted_answer'],
        'is_correct': is_correct,
        'solve_time': solution['solve_time'],
        'num_questions': solution['num_questions'],
        'metrics': solution['metrics'],
        'full_response': solution['response'] if verbose else None
    }


def evaluate_qa_compression(slm_model,
                            llm_model,
                            question_model,
                            subject,
                            problem_indices,
                            num_problems=None,
                            max_questions=30,
                            batch_mode=True,
                            batch_size=10,
                            oracle_mode=False,
                            parallel=False,
                            max_workers=4,
                            resume_path=None,
                            verbose=False):
    """Evaluate Q&A compression on specific MATH problems.

    Args:
        oracle_mode: If True, use the dataset's solution as the reference instead
                    of generating one with the LLM. This tests the upper bound of
                    Q&A compression when the oracle has perfect knowledge.
        resume_path: If provided, load existing results and only re-run failed problems.
    """

    # Handle resume mode
    existing_results = {}
    failed_indices = set()
    resume_metadata = {}

    if resume_path and Path(resume_path).exists():
        print(f"\n📂 Loading existing results from: {resume_path}")
        existing_results, failed_indices, resume_metadata = load_existing_results(
            resume_path)
        print(f"   Found {len(existing_results)} existing results")
        print(f"   {len(failed_indices)} failed problems to re-run")

        if not failed_indices:
            print(
                "   ✅ All problems completed successfully, nothing to re-run!")
            # Return existing results as-is
            return resume_metadata

    # Resolve model aliases
    slm_full = MODEL_ALIAS_MAP.get(slm_model.lower(), slm_model)
    llm_full = MODEL_ALIAS_MAP.get(llm_model.lower(), llm_model)
    question_full = MODEL_ALIAS_MAP.get(question_model.lower(), question_model)

    print(f"\n{'='*60}")
    print(f"Evaluating MATH ({subject}) with Q&A Compression")
    if resume_path:
        print(
            f"🔄 RESUME MODE: Re-running {len(failed_indices)} failed problems")
    if oracle_mode:
        print(f"🔮 ORACLE MODE: Using dataset solutions as reference")
    print(f"SLM: {slm_model} ({slm_full})")
    print(f"LLM: {llm_model} ({llm_full})")
    print(f"Question Model: {question_model} ({question_full})")
    print(f"Max questions: {max_questions}")
    if batch_mode:
        print(f"Batch mode: Enabled (size={batch_size})")
    else:
        print(f"Batch mode: Disabled (sequential Q&A)")
    if parallel:
        print(f"🚀 Parallel mode: Enabled ({max_workers} workers)")
    print(f"{'='*60}")

    # Load MATH dataset
    ds = load_dataset('EleutherAI/hendrycks_math', subject)
    dataset = ds['test']

    # Select problems to evaluate
    if num_problems is None:
        selected_indices = problem_indices
    else:
        selected_indices = problem_indices[:num_problems]

    # In resume mode, only run failed problems
    if resume_path and failed_indices:
        original_count = len(selected_indices)
        selected_indices = [
            idx for idx in selected_indices if idx in failed_indices
        ]
        print(
            f"   Filtered to {len(selected_indices)} failed problems (from {original_count} total)"
        )

        if not selected_indices:
            print("   ✅ No failed problems to re-run in selected range!")
            return resume_metadata

    results = []
    correct_count = 0
    overall_start_time = time.time()

    if parallel:
        # Parallel execution
        print(
            f"\n🚀 Running {len(selected_indices)} problems in parallel with {max_workers} workers..."
        )

        # Prepare task arguments
        task_args = []
        for idx in selected_indices:
            problem = dataset[idx]
            problem_text = problem['problem']
            correct_answer = problem['solution']
            gold_answer = extract_math_answer(correct_answer)
            task_args.append((idx, problem_text, correct_answer, gold_answer,
                              slm_full, llm_full, question_full, max_questions,
                              batch_mode, batch_size, oracle_mode, verbose))

        # Execute in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(_process_single_problem, args): args[0]
                for args in task_args
            }

            completed = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                completed += 1
                try:
                    result = future.result()
                    results.append(result)
                    if result['is_correct']:
                        correct_count += 1

                    # Progress update
                    elapsed = time.time() - overall_start_time
                    acc_pct = 100 * correct_count / len(results)
                    status = "✓" if result['is_correct'] else "✗"
                    print(
                        f"[{completed}/{len(selected_indices)}] idx={idx}: {status} | Acc: {acc_pct:.1f}% | Elapsed: {elapsed:.0f}s"
                    )

                except Exception as e:
                    print(f"❌ Problem {idx} failed: {e}")
                    results.append({
                        'problem_idx': idx,
                        'problem': None,
                        'gold_answer': None,
                        'model_answer': None,
                        'is_correct': False,
                        'solve_time': 0,
                        'num_questions': 0,
                        'metrics': None,
                        'error': str(e)
                    })

        # Sort by problem index
        results.sort(key=lambda x: x['problem_idx'])

        total_elapsed = time.time() - overall_start_time
        print(f"\n🏁 Parallel execution complete in {total_elapsed:.1f}s")
        print(
            f"   Problems/minute: {len(selected_indices) / (total_elapsed/60):.1f}"
        )

    else:
        # Sequential execution (original code)
        for i, idx in enumerate(selected_indices):
            problem = dataset[idx]
            problem_text = problem['problem']
            correct_answer = problem['solution']  # Full solution text

            # Extract just the answer from the solution
            gold_answer = extract_math_answer(correct_answer)

            # Solve with Q&A compression
            solution = solve_problem_with_qa(
                problem_text,
                slm_full,
                llm_full,
                question_full,
                max_questions=max_questions,
                batch_mode=batch_mode,
                batch_size=batch_size,
                oracle_solution=correct_answer if oracle_mode else None,
                verbose=verbose)

            # Check if correct
            is_correct = False
            if solution['extracted_answer'] and gold_answer:
                norm_model = normalize_answer(solution['extracted_answer'])
                norm_gold = normalize_answer(gold_answer)
                is_correct = norm_model == norm_gold
                if is_correct:
                    correct_count += 1

            result = {
                'problem_idx': idx,
                'problem': problem_text,
                'gold_answer': gold_answer,
                'model_answer': solution['extracted_answer'],
                'is_correct': is_correct,
                'solve_time': solution['solve_time'],
                'num_questions': solution['num_questions'],
                'metrics': solution['metrics'],
                'full_response': solution['response'] if verbose else None
            }

            results.append(result)

            # Update status
            if len(results) > 0:
                acc_pct = 100 * correct_count / len(results)
                elapsed = time.time() - overall_start_time
                avg_time = elapsed / len(results)
                eta = avg_time * (len(selected_indices) - len(results))
                status = f"\rProblem {i+1}/{len(selected_indices)} (idx {idx}) | Acc: {correct_count}/{len(results)} ({acc_pct:.1f}%) | Q: {solution['num_questions']} | Time: {solution['solve_time']:.1f}s | ETA: {eta:.1f}s"
                print(status, end="", flush=True)

            if verbose:
                print(
                    f"\n\nProblem #{i+1} (index {idx}): {'✓' if is_correct else '✗'}"
                )
                print(f"  Gold answer: {gold_answer}")
                print(f"  Model answer: {solution['extracted_answer']}")
                print(f"  Questions used: {solution['num_questions']}")

    print()  # Clear status line

    # Merge with existing results if in resume mode
    if resume_path and existing_results:
        print(
            f"\n🔀 Merging {len(results)} new results with {len(existing_results)} existing results..."
        )
        results = merge_results(existing_results, results, failed_indices)
        print(f"   Total merged results: {len(results)}")

    # Recalculate stats from merged results
    total_elapsed = time.time() - overall_start_time
    correct_count = sum(1 for r in results if r.get('is_correct', False))
    accuracy = correct_count / len(results) if results else 0
    avg_time = sum(r.get('solve_time', 0)
                   for r in results) / len(results) if results else 0
    avg_questions = sum(r.get('num_questions', 0)
                        for r in results) / len(results) if results else 0

    return {
        'slm': slm_model,
        'llm': llm_model,
        'question_model': question_model,
        'slm_full': slm_full,
        'llm_full': llm_full,
        'question_model_full': question_full,
        'subject': subject,
        'oracle_mode': oracle_mode,
        'total_problems': len(results),
        'correct_count': correct_count,
        'accuracy': accuracy,
        'avg_solve_time': avg_time,
        'avg_questions': avg_questions,
        'total_elapsed_time': total_elapsed,
        'results': results
    }


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate MATH with Q&A compression')

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
    parser.add_argument('--subject',
                        type=str,
                        default='algebra',
                        choices=[
                            'algebra', 'counting_and_probability', 'geometry',
                            'intermediate_algebra', 'number_theory',
                            'prealgebra', 'precalculus'
                        ],
                        help='MATH subject (default: algebra)')
    parser.add_argument('--difficulty',
                        type=str,
                        default='medium',
                        choices=[
                            'easy', 'medium', 'hard', 'very_hard', 'all',
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
    parser.add_argument('--parallel',
                        action='store_true',
                        help='Enable parallel execution of problems')
    parser.add_argument('--max-workers',
                        type=int,
                        default=4,
                        help='Maximum parallel workers (default: 4)')
    parser.add_argument('--baseline-results',
                        type=str,
                        default=None,
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
    parser.add_argument(
        '--oracle',
        action='store_true',
        help=
        'Oracle mode: use dataset solutions as reference (tests upper bound)')
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help=
        'Resume from existing results file, only re-running failed problems')

    args = parser.parse_args()

    # Determine problem indices
    if args.indices:
        problem_indices = args.indices
        difficulty_label = 'custom'
        print(
            f"Using {len(problem_indices)} custom problem indices: {problem_indices}"
        )
    else:
        # Find baseline results file
        if args.baseline_results:
            baseline_path = args.baseline_results
        else:
            # Look for baseline results in default location
            results_dir = Path('lossy_compression/results')
            pattern = f"math_all_models_{args.subject}_*.json"
            candidates = sorted(results_dir.glob(pattern), reverse=True)

            if not candidates:
                print(
                    f"Error: No baseline results found for subject '{args.subject}'"
                )
                print(
                    f"Please run: python lossy_compression/benchmarks/math/run_math_all_models.py --subject {args.subject}"
                )
                return

            baseline_path = candidates[0]  # Most recent

        print(f"Loading problem categorization from: {baseline_path}")

        if not Path(baseline_path).exists():
            print(f"Error: Baseline results not found at {baseline_path}")
            return

        categories = get_problem_difficulty_indices(baseline_path,
                                                    args.subject)

        # Select problems based on difficulty
        if args.difficulty == 'all':
            n_per = (args.num_problems or 15) // 3
            problem_indices = (categories['easy'][:n_per] +
                               categories['medium'][:n_per] +
                               categories['hard'][:n_per])
            difficulty_label = 'mixed'
        elif args.difficulty == 'medium+hard':
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
            print(f"No {args.difficulty} problems found in baseline results.")
            return

        print(f"Found {len(problem_indices)} {difficulty_label} problems")

    # Determine number of problems
    if args.num_problems is None:
        actual_num = len(problem_indices)
        print(f"Will evaluate ALL {actual_num} problems")
    else:
        actual_num = min(args.num_problems, len(problem_indices))
        print(f"Will evaluate {actual_num} problems")

    print(f"Will ask up to {args.max_questions} questions per problem")

    # Run evaluation
    results = evaluate_qa_compression(slm_model=args.slm,
                                      llm_model=args.llm,
                                      question_model=args.question_model,
                                      subject=args.subject,
                                      problem_indices=problem_indices,
                                      num_problems=actual_num,
                                      max_questions=args.max_questions,
                                      batch_mode=args.batch,
                                      batch_size=args.batch_size,
                                      oracle_mode=args.oracle,
                                      parallel=args.parallel,
                                      max_workers=args.max_workers,
                                      resume_path=args.resume,
                                      verbose=args.verbose)

    # Print summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Subject: {args.subject}")
    print(f"Difficulty: {difficulty_label}")
    print(
        f"Accuracy: {results['accuracy']:.1%} ({results['correct_count']}/{results['total_problems']})"
    )
    print(f"Avg solve time: {results['avg_solve_time']:.2f}s per problem")
    print(f"Total elapsed time: {results['total_elapsed_time']:.1f}s")
    print(f"Avg questions used: {results['avg_questions']:.1f}")

    # Save results
    if args.output:
        output_path = args.output
    else:
        os.makedirs('lossy_compression/results', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        oracle_suffix = '_oracle' if args.oracle else ''
        output_path = f'lossy_compression/results/math_qa_{args.subject}_{args.slm}_{args.llm}_{difficulty_label}{oracle_suffix}_{timestamp}.json'

    save_data = results.copy()
    if not args.verbose:
        save_data['results'] = [{
            k: v
            for k, v in r.items() if k != 'full_response'
        } for r in results['results']]

    save_data['config'] = {
        'subject': args.subject,
        'difficulty': args.difficulty,
        'max_questions': args.max_questions,
        'batch_mode': args.batch,
        'batch_size': args.batch_size if args.batch else None,
        'baseline_results': str(baseline_path) if not args.indices else None,
        'num_problems_requested': args.num_problems,
        'evaluation_mode': 'EVAL_MODE_MATH',
        'oracle_mode': args.oracle
    }

    save_data['metadata'] = {
        'timestamp': datetime.now().isoformat(),
        'output_file': output_path,
        'dataset': {
            'name': 'MATH',
            'subject': args.subject,
            'split': 'test',
            'difficulty': difficulty_label,
            'problems_evaluated': results['total_problems']
        },
        'performance': {
            'accuracy': results['accuracy'],
            'correct_count': results['correct_count'],
            'total_problems': results['total_problems'],
            'avg_solve_time': results['avg_solve_time'],
            'total_elapsed_time': results['total_elapsed_time'],
            'avg_questions_used': results['avg_questions']
        }
    }

    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
