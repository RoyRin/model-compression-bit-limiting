#!/usr/bin/env python3
"""
Evaluate GPQA-Freeform problems using SLM question-answering compression.

Uses LLM-as-judge for evaluation since answers are free-form.

Usage:
    # Test on medium problems (haiku fails, sonnet/opus pass)
    python evaluate_gpqa_freeform_qa_compression.py --difficulty medium --num-problems 10

    # Test with specific models
    python evaluate_gpqa_freeform_qa_compression.py --difficulty medium --slm haiku --llm opus

    # Test on specific indices
    python evaluate_gpqa_freeform_qa_compression.py --indices 5 10 15 --verbose
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
from lossy_compression.core.qa_compression import iterative_SLM_loop, EVAL_MODE_MATH
from lossy_compression import MODEL_ALIAS_MAP, model_completion


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


def solve_problem_with_qa(question,
                          gold_answer,
                          slm_model,
                          llm_model,
                          question_model,
                          judge_model,
                          max_questions=30,
                          batch_mode=True,
                          batch_size=10,
                          verbose=False):
    """Solve a GPQA-freeform problem using Q&A compression."""

    prompt = f"""Answer this science question. Provide your reasoning, then give a clear, concise final answer.

Question: {question}

Think through this step-by-step, then provide your final answer."""

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
                                    evaluation_mode=EVAL_MODE_MATH)

        final_answer, qa_tuple, metrics = result
        guiding_questions, guiding_answers = qa_tuple

        solve_time = time.time() - start_time
        qa_pairs = list(zip(guiding_questions, guiding_answers))

        # Judge the answer
        judgment = judge_answer(question, gold_answer, final_answer,
                                judge_model)

        return {
            'response': final_answer,
            'solve_time': solve_time,
            'num_questions': len(guiding_questions),
            'qa_pairs': qa_pairs if verbose else None,
            'metrics': metrics,
            'judgment': judgment,
            'success': True
        }

    except Exception as e:
        import traceback
        if verbose:
            print(f"Error solving problem: {e}")
            traceback.print_exc()
        return {
            'response': str(e),
            'solve_time': time.time() - start_time,
            'num_questions': 0,
            'qa_pairs': None,
            'metrics': None,
            'judgment': {
                'judgment': 'error',
                'reasoning': str(e),
                'success': False
            },
            'success': False,
            'error': str(e)
        }


def evaluate_qa_compression(slm_model,
                            llm_model,
                            question_model,
                            judge_model,
                            problem_indices,
                            num_problems=None,
                            max_questions=30,
                            batch_mode=True,
                            batch_size=10,
                            verbose=False):
    """Evaluate Q&A compression on specific GPQA-freeform problems."""

    slm_full = MODEL_ALIAS_MAP.get(slm_model.lower(), slm_model)
    llm_full = MODEL_ALIAS_MAP.get(llm_model.lower(), llm_model)
    question_full = MODEL_ALIAS_MAP.get(question_model.lower(), question_model)
    judge_full = MODEL_ALIAS_MAP.get(judge_model.lower(), judge_model)

    print(f"\n{'='*60}")
    print(f"Evaluating GPQA-Freeform with Q&A Compression")
    print(f"SLM: {slm_model} ({slm_full})")
    print(f"LLM: {llm_model} ({llm_full})")
    print(f"Question Model: {question_model} ({question_full})")
    print(f"Judge Model: {judge_model} ({judge_full})")
    print(f"Max questions: {max_questions}")
    if batch_mode:
        print(f"Batch mode: Enabled (size={batch_size})")
    print(f"{'='*60}")

    # Load GPQA-freeform dataset
    dataset = load_dataset("nikhilchandak/freeform-datasets",
                           split="gpqa_diamond")

    if num_problems is None:
        selected_indices = problem_indices
    else:
        selected_indices = problem_indices[:num_problems]

    results = []
    correct_count = 0
    partial_count = 0
    overall_start_time = time.time()

    # Track by category
    category_stats = {}

    for i, idx in enumerate(selected_indices):
        problem = dataset[idx]
        question = problem['question']
        gold_answer = problem['answer']
        category = problem.get('category', 'unknown')

        print(
            f"\rEvaluating problem {i+1}/{len(selected_indices)} (index {idx})",
            end="",
            flush=True)

        solution = solve_problem_with_qa(question,
                                         gold_answer,
                                         slm_full,
                                         llm_full,
                                         question_full,
                                         judge_full,
                                         max_questions=max_questions,
                                         batch_mode=batch_mode,
                                         batch_size=batch_size,
                                         verbose=verbose)

        is_correct = solution['judgment']['judgment'] == 'correct'
        is_partial = solution['judgment']['judgment'] == 'partial'

        if is_correct:
            correct_count += 1
        if is_partial:
            partial_count += 1

        # Track category
        if category not in category_stats:
            category_stats[category] = {'total': 0, 'correct': 0, 'partial': 0}
        category_stats[category]['total'] += 1
        if is_correct:
            category_stats[category]['correct'] += 1
        if is_partial:
            category_stats[category]['partial'] += 1

        result = {
            'problem_idx':
            idx,
            'question':
            question[:200] + '...' if len(question) > 200 else question,
            'gold_answer':
            gold_answer,
            'category':
            category,
            'is_correct':
            is_correct,
            'is_partial':
            is_partial,
            'judgment':
            solution['judgment']['judgment'],
            'solve_time':
            solution['solve_time'],
            'num_questions':
            solution['num_questions'],
            'metrics':
            solution['metrics'],
            'response':
            solution['response'][:500] +
            '...' if len(solution['response']) > 500 else solution['response']
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
    partial_rate = partial_count / len(results) if results else 0
    avg_time = sum(r['solve_time']
                   for r in results) / len(results) if results else 0
    avg_questions = sum(r['num_questions']
                        for r in results) / len(results) if results else 0

    return {
        'slm': slm_model,
        'llm': llm_model,
        'question_model': question_model,
        'judge_model': judge_model,
        'slm_full': slm_full,
        'llm_full': llm_full,
        'question_model_full': question_full,
        'judge_model_full': judge_full,
        'total_problems': len(results),
        'correct_count': correct_count,
        'partial_count': partial_count,
        'accuracy': accuracy,
        'partial_rate': partial_rate,
        'avg_solve_time': avg_time,
        'avg_questions': avg_questions,
        'total_elapsed_time': total_elapsed_time,
        'category_stats': category_stats,
        'results': results
    }


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate GPQA-Freeform with Q&A compression')
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
    parser.add_argument('--judge-model',
                        type=str,
                        default='sonnet',
                        help='Judge model for evaluation (default: sonnet)')
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
        'lossy_compression/results/gpqa_freeform_all_models_latest.json',
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
            print("Please run run_gpqa_freeform_all_models.py first.")
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
                                      judge_model=args.judge_model,
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
        f"Accuracy (strict): {results['accuracy']:.1%} ({results['correct_count']}/{results['total_problems']})"
    )
    print(
        f"Partial credit: {results['partial_rate']:.1%} ({results['partial_count']}/{results['total_problems']})"
    )
    print(f"Avg solve time: {results['avg_solve_time']:.2f}s")
    print(f"Avg questions used: {results['avg_questions']:.1f}")

    if results['category_stats']:
        print(f"\nAccuracy by Category:")
        for cat, stats in sorted(results['category_stats'].items()):
            if stats['total'] > 0:
                acc = stats['correct'] / stats['total']
                print(f"  {cat} (n={stats['total']}): {acc:.1%}")

    # Save results
    if args.output:
        output_path = args.output
    else:
        os.makedirs('lossy_compression/results', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f'lossy_compression/results/gpqa_freeform_qa_{args.slm}_{args.llm}_{difficulty_label}_{timestamp}.json'

    save_data = results.copy()
    save_data['config'] = {
        'difficulty': args.difficulty,
        'max_questions': args.max_questions,
        'batch_mode': args.batch,
        'batch_size': args.batch_size if args.batch else None,
        'baseline_results': args.baseline_results,
        'num_problems_requested': args.num_problems,
        'evaluation_mode': 'EVAL_MODE_MATH',
        'judge_model': args.judge_model
    }
    save_data['metadata'] = {
        'timestamp': datetime.now().isoformat(),
        'output_file': output_path,
        'dataset': {
            'name': 'GPQA-Diamond-Freeform',
            'source': 'nikhilchandak/freeform-datasets',
            'difficulty': difficulty_label,
            'problems_evaluated': results['total_problems']
        }
    }

    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
