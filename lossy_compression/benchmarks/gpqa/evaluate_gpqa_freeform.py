#!/usr/bin/env python3
"""
Evaluate GPQA freeform (open-ended) problems using SLM question-answering compression.

Unlike the MCQ version, this uses open-ended questions where the model must generate
the answer rather than select from choices.

Usage:
    python evaluate_gpqa_freeform.py --num-problems 10
    python evaluate_gpqa_freeform.py --slm haiku --llm opus --category "Quantum Mechanics"
"""

import json
import time
import argparse
import sys
import os
import re
from pathlib import Path
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, desc=None, **kwargs):
        for i, item in enumerate(iterable):
            if desc:
                print(f"\r{desc}: {i+1}", end="", flush=True)
            yield item
        if desc:
            print()


from datasets import load_dataset
from lossy_compression.core.qa_compression import iterative_SLM_loop, EVAL_MODE_SCIENCE

try:
    from lossy_compression import MODEL_ALIAS_MAP
except ImportError:
    MODEL_ALIAS_MAP = {
        "haiku": "claude-3-haiku-20240307",
        "sonnet": "claude-3-7-sonnet-20250219",
        "opus": "claude-opus-4-1-20250805",
    }


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    if answer is None:
        return ""
    # Remove extra whitespace
    answer = " ".join(answer.strip().split())
    # Remove common prefixes
    answer = re.sub(r'^(the answer is|answer:|the result is)\s*',
                    '',
                    answer,
                    flags=re.IGNORECASE)
    return answer.strip()


def extract_numeric(text: str) -> float | None:
    """Try to extract a numeric value from text."""
    if text is None:
        return None
    text = normalize_answer(text)
    # Try to find a number (including negative and decimal)
    match = re.search(r'-?\d+\.?\d*', text)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None
    return None


def answers_match(model_answer: str,
                  correct_answer: str,
                  tolerance: float = 0.01) -> tuple[bool, str]:
    """
    Check if model answer matches correct answer.

    Returns (is_match, match_type) where match_type is one of:
    - "exact": Exact string match
    - "numeric": Numeric match within tolerance
    - "contains": Correct answer contained in model answer
    - "none": No match
    """
    if model_answer is None or correct_answer is None:
        return False, "none"

    model_norm = normalize_answer(model_answer).lower()
    correct_norm = normalize_answer(correct_answer).lower()

    # Exact match
    if model_norm == correct_norm:
        return True, "exact"

    # Try numeric comparison
    model_num = extract_numeric(model_answer)
    correct_num = extract_numeric(correct_answer)

    if model_num is not None and correct_num is not None:
        if correct_num == 0:
            if abs(model_num) < tolerance:
                return True, "numeric"
        else:
            if abs(model_num - correct_num) / abs(correct_num) < tolerance:
                return True, "numeric"

    # Check if correct answer is contained in model answer
    if correct_norm in model_norm:
        return True, "contains"

    return False, "none"


def extract_final_answer(response: str) -> str | None:
    """Extract the final answer from model response."""
    if response is None:
        return None

    # Look for explicit answer patterns
    patterns = [
        r'(?:final answer|the answer|answer is|result is)[:\s]*(.+?)(?:\n|$)',
        r'\*\*(.+?)\*\*',  # Bold text often indicates answer
        r'(?:therefore|thus|so)[,\s]+(?:the answer is\s*)?(.+?)(?:\.|$)',
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            # Clean up the answer
            answer = re.sub(r'[.\s]+$', '', answer)
            if len(answer) > 0 and len(
                    answer) < 200:  # Reasonable answer length
                return answer

    # Fallback: return last line if it's short
    lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
    if lines:
        last_line = lines[-1]
        if len(last_line) < 100:
            return last_line

    return None


def create_freeform_prompt(problem: dict) -> str:
    """Create a prompt for freeform question."""
    prompt = f"""{problem['question']}

Please solve this problem step by step, showing your reasoning. At the end, clearly state your final answer."""
    return prompt


def solve_problem_with_qa(problem: dict,
                          problem_id: int,
                          slm_model: str,
                          llm_model: str,
                          question_model: str,
                          max_questions: int = 30,
                          batch_mode: bool = False,
                          batch_size: int = 10,
                          verbose: bool = False) -> dict:
    """Solve a freeform GPQA problem using Q&A compression."""

    prompt = create_freeform_prompt(problem)
    correct_answer = problem['answer']

    start_time = time.time()

    try:
        result = iterative_SLM_loop(prompt=prompt,
                                    system_prompt=None,
                                    large_model_name=llm_model,
                                    small_model_name=slm_model,
                                    question_model_name=question_model,
                                    max_iterations=max_questions,
                                    verbose=verbose,
                                    batch_mode=batch_mode,
                                    batch_size=batch_size,
                                    evaluation_mode=EVAL_MODE_SCIENCE)

        final_answer, qa_tuple, metrics = result
        guiding_questions, guiding_answers = qa_tuple

        solve_time = time.time() - start_time

        # Extract the model's answer
        extracted_answer = extract_final_answer(final_answer)

        # Check if correct
        is_correct, match_type = answers_match(extracted_answer,
                                               correct_answer)

        return {
            'response':
            final_answer,
            'extracted_answer':
            extracted_answer,
            'correct_answer':
            correct_answer,
            'is_correct':
            is_correct,
            'match_type':
            match_type,
            'solve_time':
            solve_time,
            'num_questions':
            len(guiding_questions),
            'qa_pairs':
            list(zip(guiding_questions, guiding_answers)) if verbose else None,
            'metrics':
            metrics,
            'success':
            True
        }

    except Exception as e:
        import traceback
        if verbose:
            print(f"Error solving problem: {e}")
            traceback.print_exc()
        return {
            'response': str(e),
            'extracted_answer': None,
            'correct_answer': correct_answer,
            'is_correct': False,
            'match_type': 'error',
            'solve_time': time.time() - start_time,
            'num_questions': 0,
            'qa_pairs': None,
            'metrics': None,
            'success': False,
            'error': str(e)
        }


def evaluate_freeform(slm_model: str,
                      llm_model: str,
                      question_model: str,
                      dataset_split: str = "gpqa_diamond",
                      category: str | None = None,
                      num_problems: int | None = None,
                      max_questions: int = 30,
                      batch_mode: bool = False,
                      batch_size: int = 10,
                      verbose: bool = False) -> dict:
    """Evaluate Q&A compression on freeform GPQA problems."""

    # Resolve model aliases
    slm_full = MODEL_ALIAS_MAP.get(slm_model.lower(), slm_model)
    llm_full = MODEL_ALIAS_MAP.get(llm_model.lower(), llm_model)
    question_full = MODEL_ALIAS_MAP.get(question_model.lower(), question_model)

    print(f"\n{'='*60}")
    print(f"Evaluating Freeform GPQA with Q&A Compression")
    print(f"Dataset: nikhilchandak/freeform-datasets ({dataset_split})")
    print(f"SLM: {slm_model} ({slm_full})")
    print(f"LLM: {llm_model} ({llm_full})")
    print(f"Question Model: {question_model} ({question_full})")
    print(f"Max questions: {max_questions}")
    if category:
        print(f"Category filter: {category}")
    print(f"{'='*60}")

    # Load dataset
    print("\nLoading freeform dataset...")
    ds = load_dataset("nikhilchandak/freeform-datasets")
    dataset = ds[dataset_split]

    # Filter by category if specified
    if category:
        dataset = dataset.filter(
            lambda x: category.lower() in x['category'].lower())
        print(f"Filtered to {len(dataset)} problems in category '{category}'")

    # Get categories for statistics
    categories = {}
    for example in dataset:
        cat = example['category']
        categories[cat] = categories.get(cat, 0) + 1

    print(f"\nCategories in dataset:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")

    # Select problems
    if num_problems is not None:
        num_problems = min(num_problems, len(dataset))
    else:
        num_problems = len(dataset)

    print(f"\nEvaluating {num_problems} problems...")

    results = []
    correct_count = 0
    match_types = {
        'exact': 0,
        'numeric': 0,
        'contains': 0,
        'none': 0,
        'error': 0
    }

    overall_start_time = time.time()

    for i in range(num_problems):
        problem = dataset[i]

        solution = solve_problem_with_qa(problem,
                                         i,
                                         slm_full,
                                         llm_full,
                                         question_full,
                                         max_questions=max_questions,
                                         batch_mode=batch_mode,
                                         batch_size=batch_size,
                                         verbose=verbose)

        if solution['is_correct']:
            correct_count += 1

        match_types[solution['match_type']] += 1

        result = {
            'problem_id':
            problem['question_id'],
            'category':
            problem['category'],
            'question':
            problem['question'][:200] +
            '...' if len(problem['question']) > 200 else problem['question'],
            'correct_answer':
            solution['correct_answer'],
            'model_answer':
            solution['extracted_answer'],
            'is_correct':
            solution['is_correct'],
            'match_type':
            solution['match_type'],
            'solve_time':
            solution['solve_time'],
            'num_questions':
            solution['num_questions'],
            'metrics':
            solution['metrics'],
            'full_response':
            solution['response'] if verbose else None
        }

        results.append(result)

        # Progress update
        elapsed = time.time() - overall_start_time
        acc_pct = 100 * correct_count / len(results)
        eta = elapsed / len(results) * (num_problems - len(results))
        status = f"\rProblem {i+1}/{num_problems} | Acc: {correct_count}/{len(results)} ({acc_pct:.1f}%) | Q: {solution['num_questions']} | ETA: {eta:.0f}s"
        print(status, end="", flush=True)

        if verbose:
            print(
                f"\n\nProblem #{i+1}: {'✓' if solution['is_correct'] else '✗'} ({solution['match_type']})"
            )
            print(f"  Category: {problem['category']}")
            print(f"  Correct: {solution['correct_answer']}")
            print(f"  Model: {solution['extracted_answer']}")

    print()

    total_elapsed = time.time() - overall_start_time
    accuracy = correct_count / len(results) if results else 0
    avg_time = sum(r['solve_time']
                   for r in results) / len(results) if results else 0
    avg_questions = sum(r['num_questions']
                        for r in results) / len(results) if results else 0

    return {
        'slm': slm_model,
        'llm': llm_model,
        'question_model': question_model,
        'dataset_split': dataset_split,
        'category_filter': category,
        'total_problems': len(results),
        'correct_count': correct_count,
        'accuracy': accuracy,
        'match_types': match_types,
        'avg_solve_time': avg_time,
        'avg_questions': avg_questions,
        'total_elapsed_time': total_elapsed,
        'results': results
    }


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate freeform GPQA with Q&A compression')
    parser.add_argument('--slm',
                        type=str,
                        default='haiku',
                        help='Small language model')
    parser.add_argument('--llm',
                        type=str,
                        default='opus',
                        help='Large language model')
    parser.add_argument('--question-model',
                        type=str,
                        default='haiku',
                        help='Question generation model')
    parser.add_argument('--dataset',
                        type=str,
                        default='gpqa_diamond',
                        choices=['gpqa_diamond', 'mmlu_pro'],
                        help='Dataset split to use')
    parser.add_argument('--category',
                        type=str,
                        default=None,
                        help='Filter by category')
    parser.add_argument('--num-problems',
                        type=int,
                        default=None,
                        help='Number of problems')
    parser.add_argument('--max-questions',
                        type=int,
                        default=30,
                        help='Max questions per problem')
    parser.add_argument('--batch',
                        action='store_true',
                        help='Enable batch mode')
    parser.add_argument('--batch-size',
                        type=int,
                        default=10,
                        help='Batch size')
    parser.add_argument('--verbose',
                        action='store_true',
                        help='Show detailed output')
    parser.add_argument('--output', type=str, help='Output file path')

    args = parser.parse_args()

    results = evaluate_freeform(slm_model=args.slm,
                                llm_model=args.llm,
                                question_model=args.question_model,
                                dataset_split=args.dataset,
                                category=args.category,
                                num_problems=args.num_problems,
                                max_questions=args.max_questions,
                                batch_mode=args.batch,
                                batch_size=args.batch_size,
                                verbose=args.verbose)

    if results is None:
        return

    # Print summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset}")
    if args.category:
        print(f"Category: {args.category}")
    print(
        f"Accuracy: {results['accuracy']:.1%} ({results['correct_count']}/{results['total_problems']})"
    )
    print(f"Match types: {results['match_types']}")
    print(f"Avg solve time: {results['avg_solve_time']:.2f}s")
    print(f"Avg questions: {results['avg_questions']:.1f}")
    print(f"Total time: {results['total_elapsed_time']:.1f}s")

    # Save results
    if args.output:
        output_path = args.output
    else:
        os.makedirs('lossy_compression/results', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f'lossy_compression/results/gpqa_freeform_{args.slm}_{args.llm}_{args.dataset}_{timestamp}.json'

    save_data = {
        'config': vars(args),
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'dataset': 'nikhilchandak/freeform-datasets',
            'split': args.dataset
        },
        **results
    }

    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
