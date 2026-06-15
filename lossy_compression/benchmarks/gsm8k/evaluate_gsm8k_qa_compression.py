#!/usr/bin/env python3
"""
Evaluate GSM8K problems using SLM question-answering compression.

Usage:
    # Test on easy problems
    python evaluate_gsm8k_qa_compression.py --difficulty easy --num-problems 10
    
    # Test on medium problems with specific models
    python evaluate_gsm8k_qa_compression.py --difficulty medium --slm haiku --llm opus
    
    # Test on all difficulties
    python evaluate_gsm8k_qa_compression.py --difficulty all --num-problems 5
"""

import json
import time
import argparse
from pathlib import Path
from datetime import datetime
import os
# from tqdm import tqdm  # Optional, will use simple progress if not available
try:
    from tqdm import tqdm
except ImportError:
    # Simple fallback if tqdm is not installed
    def tqdm(iterable, desc=None, **kwargs):
        total = len(iterable) if hasattr(iterable, '__len__') else None
        for i, item in enumerate(iterable):
            if desc and total:
                print(f"\r{desc}: {i+1}/{total}", end="", flush=True)
            yield item
        if desc:
            print()  # New line after completion


from datasets import load_dataset
import re

from lossy_compression.core.qa_compression import iterative_SLM_loop, EVAL_MODE_MATH
from lossy_compression import MODEL_ALIAS_MAP


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


def extract_gsm8k_answer(response):
    """Extract numerical answer from model response."""
    # First remove any currency symbols for cleaner matching
    clean_response = response.replace('$', '').replace('¥', '').replace(
        '€', '').replace('£', '')

    # Look for common answer patterns (now handles numbers with commas)
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
            # Remove commas from the number before returning
            return match.group(1).replace(',', '')

    # Fallback: find the last standalone number in the text
    # This now also handles numbers with commas
    numbers = re.findall(r'\b([\d,]+)\b', clean_response)
    if numbers:
        # Return the last number, removing commas
        return numbers[-1].replace(',', '')

    return None


def solve_problem_with_qa(problem_text,
                          slm_model,
                          llm_model,
                          question_model,
                          max_questions=30,
                          batch_mode=True,
                          batch_size=10,
                          verbose=False):
    """Solve a GSM8K problem using Q&A compression."""

    # Create prompt for the math problem
    prompt = f"""Solve this math problem step by step and provide a numerical answer.

Problem: {problem_text}

Show your work and end with: The answer is [number]."""

    start_time = time.time()

    try:
        # Use the iterative SLM loop for Q&A compression
        result = iterative_SLM_loop(prompt=prompt,
                                    large_model_name=llm_model,
                                    small_model_name=slm_model,
                                    question_model_name=question_model,
                                    max_iterations=max_questions,
                                    verbose=verbose,
                                    batch_mode=batch_mode,
                                    batch_size=batch_size,
                                    evaluation_mode=EVAL_MODE_MATH)

        # Unpack the result
        final_answer, qa_tuple, metrics = result
        guiding_questions, guiding_answers = qa_tuple

        solve_time = time.time() - start_time

        # Extract numerical answer
        extracted_answer = extract_gsm8k_answer(final_answer)

        # Combine Q&A pairs
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


def evaluate_qa_compression(slm_model,
                            llm_model,
                            question_model,
                            problem_indices,
                            num_problems=None,
                            max_questions=30,
                            batch_mode=True,
                            batch_size=10,
                            verbose=False):
    """Evaluate Q&A compression on specific GSM8K problems."""

    # Resolve model aliases
    slm_full = MODEL_ALIAS_MAP.get(slm_model.lower(), slm_model)
    llm_full = MODEL_ALIAS_MAP.get(llm_model.lower(), llm_model)
    question_full = MODEL_ALIAS_MAP.get(question_model.lower(), question_model)

    print(f"\n{'='*60}")
    print(f"Evaluating GSM8K with Q&A Compression")
    print(f"SLM: {slm_model} ({slm_full})")
    print(f"LLM: {llm_model} ({llm_full})")
    print(f"Question Model: {question_model} ({question_full})")
    print(f"Max questions: {max_questions}")
    if batch_mode:
        print(f"Batch mode: Enabled (size={batch_size})")
    else:
        print(f"Batch mode: Disabled (sequential Q&A)")
    print(f"{'='*60}")

    # Load GSM8K dataset
    ds = load_dataset("openai/gsm8k", "main")
    dataset = ds['test']

    # Select problems to evaluate
    if num_problems is None:
        selected_indices = problem_indices
    else:
        selected_indices = problem_indices[:num_problems]

    results = []
    correct_count = 0

    # Track overall timing
    import time
    overall_start_time = time.time()

    for i, idx in enumerate(selected_indices):
        problem = dataset[idx]
        question = problem['question']

        print(
            f"\rEvaluating problem {i+1}/{len(selected_indices)} (index {idx})",
            end="",
            flush=True)

        # Extract correct answer
        answer_match = re.search(r'####\s*(\d+)', problem['answer'])
        correct_answer = answer_match.group(1) if answer_match else None

        # Solve with Q&A compression
        solution = solve_problem_with_qa(question,
                                         slm_full,
                                         llm_full,
                                         question_full,
                                         max_questions=max_questions,
                                         batch_mode=batch_mode,
                                         batch_size=batch_size,
                                         verbose=verbose)

        # Check if correct
        is_correct = False
        if solution['extracted_answer'] and correct_answer:
            is_correct = solution['extracted_answer'] == correct_answer
            if is_correct:
                correct_count += 1

        result = {
            'problem_id': idx,
            'question': question,
            'correct_answer': correct_answer,
            'model_answer': solution['extracted_answer'],
            'is_correct': is_correct,
            'solve_time': solution['solve_time'],
            'num_questions': solution['num_questions'],
            'metrics': solution['metrics'],
            'full_response': solution['response'] if verbose else None
        }

        results.append(result)

        # Update status line with timing
        if len(results) > 0:
            acc_pct = 100 * correct_count / len(results)
            elapsed_time = time.time() - overall_start_time
            avg_time_per_problem = elapsed_time / len(results)
            eta = avg_time_per_problem * (len(selected_indices) - len(results))
            status = f"\rProblem {i+1}/{len(selected_indices)} (idx {idx}) | Acc: {correct_count}/{len(results)} ({acc_pct:.1f}%) | Q: {solution['num_questions']} | Time: {solution['solve_time']:.1f}s | Elapsed: {elapsed_time:.1f}s | ETA: {eta:.1f}s"
            print(status, end="", flush=True)

        if verbose:
            print(
                f"\n\nProblem #{i+1} (index {idx}): {'✓' if is_correct else '✗'}"
            )
            print(f"  Correct answer: {correct_answer}")
            print(f"  Model answer: {solution['extracted_answer']}")
            print(f"  Questions used: {solution['num_questions']}")
            print(f"\n  Full model response:")
            print(f"  {'-'*50}")
            print(f"  {solution['response']}")
            print(f"  {'-'*50}")

    print()  # Clear the status line

    # Calculate total elapsed time
    total_elapsed_time = time.time() - overall_start_time

    # Calculate statistics
    accuracy = correct_count / len(results) if results else 0
    avg_time = sum(r['solve_time']
                   for r in results) / len(results) if results else 0
    avg_questions = sum(r['num_questions']
                        for r in results) / len(results) if results else 0
    total_solve_time = sum(r['solve_time'] for r in results) if results else 0

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
        'total_solve_time': total_solve_time,
        'total_elapsed_time': total_elapsed_time,
        'results': results
    }


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate GSM8K with Q&A compression')
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
                        default='easy',
                        choices=[
                            'easy', 'medium', 'hard', 'very_hard', 'all',
                            'medium+hard', 'not_easy'
                        ],
                        help='Problem difficulty to test (default: easy)')
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
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help='Batch size for questions when using --batch (default: 10)')
    parser.add_argument(
        '--baseline-results',
        type=str,
        default=
        'lossy_compression/results/gsm8k_all_models_20260115_215021.json',
        help='Path to baseline results for categorization')
    parser.add_argument('--verbose',
                        action='store_true',
                        help='Show detailed output')
    parser.add_argument('--output', type=str, help='Output file for results')
    parser.add_argument(
        '--indices',
        type=int,
        nargs='+',
        default=None,
        help=
        'Specific problem indices to test (bypasses baseline categorization)')

    args = parser.parse_args()

    # If indices provided directly, skip baseline categorization
    if args.indices:
        problem_indices = args.indices
        difficulty_label = 'custom'
        print(
            f"Using {len(problem_indices)} custom problem indices: {problem_indices}"
        )
    else:
        # Get problem indices based on difficulty
        print(f"Loading problem categorization from: {args.baseline_results}")
        if not Path(args.baseline_results).exists():
            print(
                f"Error: Baseline results not found at {args.baseline_results}"
            )
            print(
                "Please run evaluate_gsm8k_baseline.py first to generate baseline results."
            )
            return

        categories = get_problem_difficulty_indices(args.baseline_results)

        # Select problems based on difficulty
        if args.difficulty == 'all':
            # Take some from each category
            problems_per_category = args.num_problems // 3
            problem_indices = (categories['easy'][:problems_per_category] +
                               categories['medium'][:problems_per_category] +
                               categories['hard'][:problems_per_category])
            difficulty_label = 'mixed'
        elif args.difficulty == 'medium+hard':
            # Combine medium and hard problems
            problem_indices = categories['medium'] + categories['hard']
            difficulty_label = 'medium+hard'
        elif args.difficulty == 'not_easy':
            # All non-easy problems (medium + hard + very_hard)
            problem_indices = categories['medium'] + categories[
                'hard'] + categories['very_hard']
            difficulty_label = 'not_easy'
        else:
            problem_indices = categories[args.difficulty]
            difficulty_label = args.difficulty

        if not problem_indices:
            print(f"No {args.difficulty} problems found in baseline results.")
            return

        print(
            f"Found {len(problem_indices)} {difficulty_label} problems in dataset"
        )

    # Determine number of problems to evaluate
    if args.num_problems is None:
        actual_num_problems = len(problem_indices)
        print(f"Will evaluate ALL {actual_num_problems} problems")
    else:
        actual_num_problems = min(args.num_problems, len(problem_indices))
        print(
            f"Will evaluate {actual_num_problems} problems (use --num-problems to change)"
        )

    print(f"Will ask up to {args.max_questions} questions per problem")

    # Run evaluation
    results = evaluate_qa_compression(slm_model=args.slm,
                                      llm_model=args.llm,
                                      question_model=args.question_model,
                                      problem_indices=problem_indices,
                                      num_problems=actual_num_problems,
                                      max_questions=args.max_questions,
                                      batch_mode=args.batch,
                                      batch_size=args.batch_size,
                                      verbose=args.verbose)

    # Print summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Difficulty: {difficulty_label}")
    print(
        f"Accuracy: {results['accuracy']:.1%} ({results['correct_count']}/{results['total_problems']})"
    )
    print(f"Avg solve time: {results['avg_solve_time']:.2f}s per problem")
    print(f"Total solve time: {results['total_solve_time']:.1f}s")
    print(f"Total elapsed time: {results['total_elapsed_time']:.1f}s")
    print(f"Avg questions used: {results['avg_questions']:.1f}")

    # Save results
    if args.output:
        output_path = args.output
    else:
        # Default output path
        os.makedirs('lossy_compression/results', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f'lossy_compression/results/gsm8k_qa_{args.slm}_{args.llm}_{args.question_model}_{difficulty_label}_{timestamp}.json'

    # Prepare data for saving
    save_data = results.copy()
    # Remove full responses to save space unless verbose
    if not args.verbose:
        save_data['results'] = [{
            k: v
            for k, v in r.items() if k != 'full_response'
        } for r in results['results']]

    save_data['config'] = {
        'difficulty': args.difficulty,
        'max_questions': args.max_questions,
        'batch_mode': args.batch,
        'batch_size': args.batch_size if args.batch else None,
        'baseline_results': args.baseline_results,
        'num_problems_requested': args.num_problems,
        'evaluation_mode': 'EVAL_MODE_MATH'
    }

    save_data['metadata'] = {
        'timestamp': datetime.now().isoformat(),
        'output_file': output_path,
        'models': {
            'slm': args.slm,
            'slm_full': results.get('slm_full', ''),
            'llm': args.llm,
            'llm_full': results.get('llm_full', ''),
            'question_model': args.question_model,
            'question_model_full': results.get('question_model_full', '')
        },
        'dataset': {
            'name': 'GSM8K',
            'split': 'test',
            'difficulty': difficulty_label,
            'problem_indices_count': len(problem_indices),
            'problems_evaluated': results['total_problems']
        },
        'performance': {
            'accuracy': results['accuracy'],
            'correct_count': results['correct_count'],
            'total_problems': results['total_problems'],
            'avg_solve_time': results['avg_solve_time'],
            'total_solve_time': results['total_solve_time'],
            'total_elapsed_time': results['total_elapsed_time'],
            'avg_questions_used': results['avg_questions']
        }
    }

    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
