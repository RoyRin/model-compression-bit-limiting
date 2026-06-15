#!/usr/bin/env python3
"""
Evaluate baseline accuracies for Claude models on GSM8K dataset.

Usage:
    python evaluate_gsm8k_baseline.py --model haiku
    python evaluate_gsm8k_baseline.py --model sonnet
    python evaluate_gsm8k_baseline.py --model opus
    python evaluate_gsm8k_baseline.py --model all
"""

import json
import time
import argparse
from datasets import load_dataset
from lossy_compression import model_completion, MODEL_ALIAS_MAP
import re
from datetime import datetime
import os
from tqdm import tqdm


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


def solve_problem(problem_text, model_name, verbose=False):
    """Solve a GSM8K problem with the specified model."""

    system_prompt = """You are a helpful math tutor solving grade school math problems. 
Show your step-by-step reasoning and end with the final numerical answer.
Format your final answer clearly as: The answer is [number]."""

    if verbose:
        print(f"Solving with {model_name}...")

    start_time = time.time()
    try:
        response = model_completion(problem_text,
                                    model=model_name,
                                    system=system_prompt,
                                    temperature=0.1,
                                    max_tokens=1000)
        solve_time = time.time() - start_time

        extracted_answer = extract_gsm8k_answer(response)

        return {
            'response': response,
            'extracted_answer': extracted_answer,
            'solve_time': solve_time,
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


def evaluate_model_on_gsm8k(model_name,
                            num_problems=100,
                            verbose=False,
                            problem_indices=None):
    """Evaluate a single model on GSM8K problems."""

    # Resolve model alias
    if model_name.lower() in MODEL_ALIAS_MAP:
        full_model_name = MODEL_ALIAS_MAP[model_name.lower()]
    else:
        full_model_name = model_name

    print(f"\n{'='*60}")
    print(f"Evaluating {model_name.upper()} on GSM8K")
    print(f"Model: {full_model_name}")
    print(f"{'='*60}")

    # Load dataset
    ds = load_dataset("openai/gsm8k", "main")
    dataset = ds['test']  # Use test set for evaluation

    # Determine which problems to evaluate
    if problem_indices is not None:
        # Use specified problem indices
        indices_to_eval = problem_indices[:num_problems] if num_problems else problem_indices
        total_problems = len(indices_to_eval)
        print(f"Using {total_problems} specified problem indices")
    else:
        # Use first num_problems from dataset
        indices_to_eval = list(range(min(num_problems, len(dataset))))
        total_problems = len(indices_to_eval)

    results = []
    correct_count = 0

    # Progress bar
    pbar = tqdm(indices_to_eval, desc=f"Evaluating {model_name}")

    for i, idx in enumerate(pbar):
        problem = dataset[idx]
        question = problem['question']

        # Extract correct answer from the dataset
        answer_match = re.search(r'####\s*(\d+)', problem['answer'])
        correct_answer = answer_match.group(1) if answer_match else None

        # Solve with model
        solution = solve_problem(question, full_model_name, verbose=False)

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
            'full_response': solution['response'] if verbose else None
        }

        results.append(result)

        # Update progress bar with accuracy
        pbar.set_postfix({
            'accuracy':
            f"{correct_count}/{i+1} ({100*correct_count/(i+1):.1f}%)"
        })

        if verbose:
            print(f"\nProblem {idx}: {'✓' if is_correct else '✗'}")
            print(f"  Correct: {correct_answer}")
            print(f"  Model: {solution['extracted_answer']}")

    # Calculate final statistics
    accuracy = correct_count / total_problems
    avg_time = sum(r['solve_time'] for r in results) / len(results)

    return {
        'model': model_name,
        'full_model_name': full_model_name,
        'total_problems': total_problems,
        'correct_count': correct_count,
        'accuracy': accuracy,
        'avg_solve_time': avg_time,
        'results': results
    }


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Claude models on GSM8K')
    parser.add_argument('--model',
                        type=str,
                        default='all',
                        choices=['haiku', 'sonnet', 'opus', 'all'],
                        help='Model to evaluate (default: all)')
    parser.add_argument('--num-problems',
                        type=int,
                        default=100,
                        help='Number of problems to evaluate (default: 100)')
    parser.add_argument('--verbose',
                        action='store_true',
                        help='Show detailed output')
    parser.add_argument('--output',
                        type=str,
                        help='Output file for results (JSON format)')
    parser.add_argument('--no-save',
                        action='store_true',
                        help='Do not save results to file')
    parser.add_argument(
        '--problem-indices',
        type=str,
        help=
        'Path to JSON file with problem indices to evaluate (e.g., from analyze_gsm8k_results.py)'
    )
    parser.add_argument(
        '--difficulty',
        type=str,
        choices=['easy', 'medium', 'hard', 'medium+hard'],
        help='If using problem-indices, which difficulty to select')

    args = parser.parse_args()

    # Load problem indices if specified
    problem_indices_to_use = None
    if args.problem_indices and args.difficulty:
        from analyze_gsm8k_results import get_problem_difficulty_indices
        categories = get_problem_difficulty_indices(args.problem_indices)

        if args.difficulty == 'medium+hard':
            problem_indices_to_use = categories['medium'] + categories['hard']
            print(
                f"Selected {len(categories['medium'])} medium + {len(categories['hard'])} hard problems"
            )
        else:
            problem_indices_to_use = categories[args.difficulty]
            print(
                f"Selected {len(problem_indices_to_use)} {args.difficulty} problems"
            )

    # Determine which models to evaluate
    if args.model == 'all':
        models = ['haiku', 'sonnet', 'opus']
    else:
        models = [args.model]

    # Evaluate each model
    all_results = {}

    for model_name in models:
        results = evaluate_model_on_gsm8k(
            model_name,
            num_problems=args.num_problems
            if not problem_indices_to_use else None,
            verbose=args.verbose,
            problem_indices=problem_indices_to_use)
        all_results[model_name] = results

        # Print summary
        print(f"\n{model_name.upper()} Results:")
        print(
            f"  Accuracy: {results['accuracy']:.1%} ({results['correct_count']}/{results['total_problems']})"
        )
        print(f"  Avg solve time: {results['avg_solve_time']:.2f}s")

    # Print comparison if multiple models
    if len(models) > 1:
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(
            f"{'Model':<10} {'Accuracy':>10} {'Correct':>10} {'Avg Time':>10}")
        print("-" * 40)

        for model_name in models:
            r = all_results[model_name]
            print(
                f"{model_name.upper():<10} {r['accuracy']:>9.1%} {r['correct_count']:>4}/{r['total_problems']:<4} {r['avg_solve_time']:>9.2f}s"
            )

    # Save results
    if not args.no_save:
        if args.output:
            output_path = args.output
        else:
            # Default output path
            os.makedirs('lossy_compression/results', exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            models_str = '_'.join(models) if len(models) > 1 else models[0]
            output_path = f'lossy_compression/results/gsm8k_{models_str}_{timestamp}.json'

        # Prepare data for saving (remove full responses to save space)
        save_data = {}
        for model_name, results in all_results.items():
            save_results = results.copy()
            # Remove full responses from individual results to save space
            save_results['results'] = [{
                k: v
                for k, v in r.items() if k != 'full_response'
            } for r in results['results']]
            save_data[model_name] = save_results

        with open(output_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
