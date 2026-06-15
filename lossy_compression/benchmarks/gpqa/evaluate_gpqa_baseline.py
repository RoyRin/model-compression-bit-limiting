#!/usr/bin/env python3
"""
Evaluate baseline accuracies for Claude models on GPQA-diamond dataset.

Usage:
    python evaluate_gpqa_baseline.py --model haiku
    python evaluate_gpqa_baseline.py --model sonnet
    python evaluate_gpqa_baseline.py --model opus
    python evaluate_gpqa_baseline.py --model all
"""

import json
import time
import argparse
import sys
import os
import re
import random
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import model_completion, but provide fallback if imports fail
try:
    from lossy_compression import model_completion, MODEL_ALIAS_MAP
except ImportError:
    # Fallback definitions if main import fails
    from utils.llm_api import anthropic_completion as model_completion
    MODEL_ALIAS_MAP = {
        "haiku": "claude-3-haiku-20240307",
        "sonnet": "claude-3-7-sonnet-20250219",
        "opus": "claude-opus-4-1-20250805",
        "gpt-5": "gpt-5-2025-08-07",
    }
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


# Will need huggingface login for this dataset
from datasets import load_dataset


def extract_answer_letter(response):
    """Extract answer letter (A, B, C, or D) from model response."""

    # Look for explicit answer patterns
    patterns = [
        r'answer is[\s:]*([A-D])\b',
        r'answer:[\s]*([A-D])\b',
        r'correct answer[\s:]+([A-D])\b',
        r'choose[\s:]+([A-D])\b',
        r'select[\s:]+([A-D])\b',
        r'^([A-D])\b',  # Answer at start of response
        r'\b([A-D])\)',  # Format like "A)" 
        r'\*\*([A-D])\*\*',  # Bold formatting
    ]

    # Try each pattern
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()

    # Fallback: find any standalone A, B, C, or D
    # But be careful not to match random letters in words
    letter_match = re.search(r'\b([A-D])\b', response)
    if letter_match:
        return letter_match.group(1).upper()

    return None


def solve_problem(problem, model_name, verbose=False, problem_id=0):
    """Solve a GPQA problem with the specified model."""

    # Create answer choices list
    answers = [
        problem['Incorrect Answer 1'], problem['Incorrect Answer 2'],
        problem['Incorrect Answer 3'], problem['Correct Answer']
    ]

    # Create deterministic seed based on problem_id
    # This ensures same shuffling for same problem across runs
    rng = random.Random(42 + problem_id)

    # Create indices and shuffle them
    indices = [0, 1, 2, 3]
    rng.shuffle(indices)

    # Map to letters and track correct answer position
    letters = ['A', 'B', 'C', 'D']
    choices = []
    correct_letter = None

    for i, idx in enumerate(indices):
        letter = letters[i]
        answer = answers[idx]
        choices.append((letter, answer))
        if idx == 3:  # The correct answer was at index 3 in original list
            correct_letter = letter

    # Construct the prompt with shuffled choices
    prompt = f"""{problem['Question']}

Choices:
A) {choices[0][1]}
B) {choices[1][1]} 
C) {choices[2][1]}
D) {choices[3][1]}

Please analyze this question carefully and select the best answer. Provide your reasoning, then clearly state your answer as A, B, C, or D."""

    system_prompt = """You are an expert scientist with deep knowledge across physics, chemistry, biology, and other scientific domains. 
Analyze questions carefully, show your reasoning, and provide clear answers."""

    if verbose:
        print(f"Solving with {model_name}...")
        print(f"Question preview: {problem['Question'][:100]}...")
        print(f"Correct answer is at position: {correct_letter}")

    start_time = time.time()
    try:
        response = model_completion(prompt,
                                    model=model_name,
                                    system=system_prompt,
                                    temperature=0.1,
                                    max_tokens=2000)
        solve_time = time.time() - start_time

        extracted_answer = extract_answer_letter(response)

        return {
            'response': response,
            'extracted_answer': extracted_answer,
            'correct_answer': correct_letter,
            'solve_time': solve_time,
            'success': True
        }
    except Exception as e:
        return {
            'response': str(e),
            'extracted_answer': None,
            'correct_answer': correct_letter,
            'solve_time': time.time() - start_time,
            'success': False,
            'error': str(e)
        }


def save_checkpoint(output_path,
                    model_name,
                    results,
                    checkpoint=True,
                    verbose=False):
    """Save intermediate results as checkpoint."""
    checkpoint_path = f"{output_path}.temp" if checkpoint else output_path

    save_data = {
        'model': model_name,
        'checkpoint': checkpoint,
        'last_problem_id': results[-1]['problem_id'] if results else -1,
        'total_problems_evaluated': len(results),
        'results': results
    }

    with open(checkpoint_path, 'w') as f:
        json.dump(save_data, f, indent=2)

    if verbose:
        print(f"\n  Checkpoint saved: {checkpoint_path}")

    return checkpoint_path


def load_checkpoint(checkpoint_path):
    """Load checkpoint file if it exists."""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            data = json.load(f)
        print(
            f"Resuming from checkpoint: {data['total_problems_evaluated']} problems already completed"
        )
        return data
    return None


def evaluate_model_on_gpqa(model_name,
                           num_problems=None,
                           verbose=False,
                           output_path=None,
                           checkpoint_interval=50,
                           problem_indices=None):
    """Evaluate a single model on GPQA-diamond problems."""

    # Resolve model alias
    if model_name.lower() in MODEL_ALIAS_MAP:
        full_model_name = MODEL_ALIAS_MAP[model_name.lower()]
    else:
        full_model_name = model_name

    print(f"\n{'='*60}")
    print(f"Evaluating {model_name.upper()} on GPQA-diamond")
    print(f"Model: {full_model_name}")
    print(f"{'='*60}")

    # Load dataset - requires HuggingFace login
    try:
        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond")
    except Exception as e:
        print(f"\nError loading dataset: {e}")
        print("Please login to HuggingFace using: huggingface-cli login")
        return None

    # GPQA-diamond has train and test splits
    # Use test set for evaluation
    dataset = ds['train']  # Note: GPQA uses 'train' for the main eval set

    print(f"Dataset size: {len(dataset)} problems")

    # Determine which problems to evaluate
    if problem_indices is not None:
        # Use specified problem indices
        indices_to_eval = problem_indices[:num_problems] if num_problems else problem_indices
        total_problems = len(indices_to_eval)
        print(f"Using {total_problems} specified problem indices")
    else:
        # Use first num_problems or all
        if num_problems is None:
            indices_to_eval = list(range(len(dataset)))
            total_problems = len(dataset)
        else:
            indices_to_eval = list(range(min(num_problems, len(dataset))))
            total_problems = len(indices_to_eval)

    print(f"Evaluating on {total_problems} problems")

    # Track distribution of correct answers if verbose
    if verbose:
        correct_positions = {'A': 0, 'B': 0, 'C': 0, 'D': 0}

    # Set up output path for checkpointing
    if output_path is None:
        os.makedirs('lossy_compression/results', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f'lossy_compression/results/gpqa_{model_name}_{timestamp}.json'

    checkpoint_path = f"{output_path}.temp"

    # Try to load checkpoint
    checkpoint_data = load_checkpoint(checkpoint_path)
    if checkpoint_data and checkpoint_data['model'] == model_name:
        results = checkpoint_data['results']
        start_idx = checkpoint_data['last_problem_id'] + 1
        correct_count = sum(1 for r in results if r['is_correct'])
        print(f"Starting from problem {start_idx}")
    else:
        results = []
        correct_count = 0
        start_idx = 0

    # Progress bar - iterate through the selected indices
    for i in range(start_idx, total_problems):
        idx = indices_to_eval[i]
        problem = dataset[idx]

        print(f"\rEvaluating problem {i+1}/{total_problems} (index {idx})",
              end="",
              flush=True)

        # Solve with model
        solution = solve_problem(problem,
                                 full_model_name,
                                 verbose=verbose,
                                 problem_id=idx)

        # Track correct answer position if verbose
        if verbose and solution['correct_answer']:
            correct_positions[solution['correct_answer']] += 1

        # Check if correct
        is_correct = False
        if solution['extracted_answer'] and solution['correct_answer']:
            is_correct = solution['extracted_answer'] == solution[
                'correct_answer']
            if is_correct:
                correct_count += 1

        result = {
            'problem_id':
            idx,
            'question':
            problem['Question'][:200] +
            '...' if len(problem['Question']) > 200 else problem['Question'],
            'correct_answer':
            solution['correct_answer'],
            'model_answer':
            solution['extracted_answer'],
            'is_correct':
            is_correct,
            'solve_time':
            solution['solve_time'],
            'full_response':
            solution['response'] if verbose else None
        }

        results.append(result)

        # Update progress
        total_evaluated = len(results)
        acc_pct = 100 * correct_count / total_evaluated
        print(
            f"\rEvaluating problem {i+1}/{total_problems} (index {idx}) | Accuracy: {correct_count}/{total_evaluated} ({acc_pct:.1f}%)",
            end="",
            flush=True)

        # Save checkpoint every N problems
        if (i + 1) % checkpoint_interval == 0 or (i + 1) == total_problems:
            save_checkpoint(output_path,
                            model_name,
                            results,
                            checkpoint=True,
                            verbose=verbose)
            if not verbose and (i + 1) % checkpoint_interval == 0:
                print(f" [Checkpoint saved]", end="")

        if verbose:
            print(f"\nProblem {idx}: {'✓' if is_correct else '✗'}")
            print(f"  Correct: {solution['correct_answer']}")
            print(f"  Model: {solution['extracted_answer']}")

    print()  # New line after progress

    # Print distribution of correct answers if verbose
    if verbose:
        print(f"\nCorrect answer position distribution:")
        for letter in ['A', 'B', 'C', 'D']:
            pct = 100 * correct_positions[letter] / sum(
                correct_positions.values()) if sum(
                    correct_positions.values()) > 0 else 0
            print(f"  {letter}: {correct_positions[letter]} ({pct:.1f}%)")

    # Calculate final statistics
    total_evaluated = len(results)
    accuracy = correct_count / total_evaluated if total_evaluated > 0 else 0
    avg_time = sum(r['solve_time']
                   for r in results) / len(results) if results else 0

    # Clean up checkpoint file after successful completion
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        if verbose:
            print(f"Checkpoint file removed: {checkpoint_path}")

    return {
        'model': model_name,
        'full_model_name': full_model_name,
        'total_problems': total_evaluated,
        'correct_count': correct_count,
        'accuracy': accuracy,
        'avg_solve_time': avg_time,
        'results': results,
        'output_path': output_path
    }


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Claude models on GPQA-diamond')
    parser.add_argument('--model',
                        type=str,
                        default='all',
                        choices=['haiku', 'sonnet', 'opus', 'all'],
                        help='Model to evaluate (default: all)')
    parser.add_argument('--num-problems',
                        type=int,
                        default=None,
                        help='Number of problems to evaluate (default: all)')
    parser.add_argument('--verbose',
                        action='store_true',
                        help='Show detailed output')
    parser.add_argument('--output',
                        type=str,
                        help='Output file for results (JSON format)')
    parser.add_argument('--no-save',
                        action='store_true',
                        help='Do not save results to file')
    parser.add_argument('--checkpoint-interval',
                        type=int,
                        default=50,
                        help='Save checkpoint every N problems (default: 50)')
    parser.add_argument('--resume',
                        action='store_true',
                        help='Resume from checkpoint if available')
    parser.add_argument(
        '--problem-indices',
        type=str,
        help=
        'Path to JSON file with problem indices to evaluate (e.g., from analyze_gpqa_results.py)'
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
        from analyze_gpqa_results import get_problem_difficulty_indices
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
        # Determine output path for this model
        if args.output and len(models) == 1:
            model_output_path = args.output
        else:
            model_output_path = None  # Will be auto-generated

        results = evaluate_model_on_gpqa(
            model_name,
            num_problems=args.num_problems
            if not problem_indices_to_use else None,
            verbose=args.verbose,
            output_path=model_output_path,
            checkpoint_interval=args.checkpoint_interval,
            problem_indices=problem_indices_to_use)

        if results is None:
            print(f"Skipping {model_name} due to error")
            continue

        all_results[model_name] = results

        # Print summary
        print(f"\n{model_name.upper()} Results:")
        print(
            f"  Accuracy: {results['accuracy']:.1%} ({results['correct_count']}/{results['total_problems']})"
        )
        print(f"  Avg solve time: {results['avg_solve_time']:.2f}s")

    # Print comparison if multiple models
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(
            f"{'Model':<10} {'Accuracy':>10} {'Correct':>10} {'Avg Time':>10}")
        print("-" * 40)

        for model_name in models:
            if model_name in all_results:
                r = all_results[model_name]
                print(
                    f"{model_name.upper():<10} {r['accuracy']:>9.1%} {r['correct_count']:>4}/{r['total_problems']:<4} {r['avg_solve_time']:>9.2f}s"
                )

    # Save results
    if not args.no_save and all_results:
        if args.output:
            output_path = args.output
        else:
            # Default output path
            os.makedirs('lossy_compression/results', exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            models_str = '_'.join(all_results.keys())
            output_path = f'lossy_compression/results/gpqa_{models_str}_{timestamp}.json'

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
