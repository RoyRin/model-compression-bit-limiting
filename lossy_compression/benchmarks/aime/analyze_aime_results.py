#!/usr/bin/env python3
"""
Analyze AIME results from different models (haiku, sonnet, opus, QA method).

Usage:
    python analyze_aime_results.py results/aime_results.json
    python analyze_aime_results.py results/aime_results.json --plot
    python analyze_aime_results.py results/aime_results.json --verbose
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import re
from collections import defaultdict


def extract_numerical_answer(text: str) -> Optional[int]:
    """Extract numerical answer from model response text.
    
    Returns None if no valid AIME answer (0-999) found.
    """
    if not text:
        return None

    # Look for boxed answers first
    boxed_pattern = r'\\boxed\{([^}]+)\}'
    boxed_match = re.search(boxed_pattern, text)
    if boxed_match:
        try:
            num = int(boxed_match.group(1).strip())
            if 0 <= num <= 999:
                return num
        except:
            pass

    # Look for explicit answer patterns
    answer_patterns = [
        r'(?:the\s+)?answer\s+is:?\s*(\d+)',
        r'(?:final\s+)?answer:?\s*(\d+)',
        r'therefore:?\s*(\d+)',
        r'=\s*(\d+)\s*(?:$|\n)',
    ]

    for pattern in answer_patterns:
        match = re.search(pattern, text.lower())
        if match:
            try:
                num = int(match.group(1))
                if 0 <= num <= 999:
                    return num
            except:
                pass

    # Look for the last standalone number
    numbers = re.findall(r'\b(\d{1,3})\b', text)
    if numbers:
        try:
            num = int(numbers[-1])
            if 0 <= num <= 999:
                return num
        except:
            pass

    return None


def analyze_model_results(results: List[Dict], model_name: str) -> Dict:
    """Analyze results for a specific model.
    
    Returns statistics about the model's performance.
    """
    stats = {
        'model': model_name,
        'total_problems': 0,
        'correct': 0,
        'incorrect': 0,
        'no_answer': 0,
        'accuracy': 0.0,
        'correct_problems': [],
        'incorrect_problems': [],
        'no_answer_problems': [],
        'extracted_answers': {},
        'solution_lengths': []
    }

    for result in results:
        problem_id = result.get('problem_id', 'unknown')
        correct_answer = result.get('correct_answer')

        # Check if this model has results
        if 'model_solutions' not in result:
            continue

        model_solution = result['model_solutions'].get(model_name)
        if not model_solution:
            continue

        stats['total_problems'] += 1

        # Get solution text
        solution_text = ""
        if isinstance(model_solution, dict):
            solution_text = model_solution.get(
                'solution', '') or model_solution.get('response', '')
        elif isinstance(model_solution, str):
            solution_text = model_solution

        # Track solution length
        if solution_text:
            stats['solution_lengths'].append(len(solution_text))

        # Extract answer
        extracted = extract_numerical_answer(solution_text)
        stats['extracted_answers'][problem_id] = extracted

        # Check correctness
        if extracted is None:
            stats['no_answer'] += 1
            stats['no_answer_problems'].append(problem_id)
        elif correct_answer is not None:
            try:
                correct_num = int(correct_answer)
                if extracted == correct_num:
                    stats['correct'] += 1
                    stats['correct_problems'].append(problem_id)
                else:
                    stats['incorrect'] += 1
                    stats['incorrect_problems'].append({
                        'id': problem_id,
                        'predicted': extracted,
                        'correct': correct_num
                    })
            except:
                stats['incorrect'] += 1
                stats['incorrect_problems'].append({
                    'id': problem_id,
                    'predicted': extracted,
                    'correct': correct_answer
                })

    # Calculate accuracy
    if stats['total_problems'] > 0:
        stats['accuracy'] = stats['correct'] / stats['total_problems']
        stats['avg_solution_length'] = np.mean(
            stats['solution_lengths']) if stats['solution_lengths'] else 0

    return stats


def analyze_qa_results(results: List[Dict]) -> Dict:
    """Analyze QA compression method results.
    
    Returns detailed statistics about the QA method.
    """
    stats = {
        'model': 'qa',
        'total_problems': 0,
        'correct': 0,
        'incorrect': 0,
        'no_answer': 0,
        'accuracy': 0.0,
        'correct_problems': [],
        'incorrect_problems': [],
        'no_answer_problems': [],
        'extracted_answers': {},
        'solution_lengths': [],
        'num_questions_used': [],
        'iterations_to_correct': [],
        'quality_improvements': []
    }

    for result in results:
        problem_id = result.get('problem_id', 'unknown')
        correct_answer = result.get('correct_answer')

        # Check if QA method has results
        if 'model_solutions' not in result:
            continue

        qa_result = result['model_solutions'].get('qa')
        if not qa_result:
            continue

        stats['total_problems'] += 1

        # Get final solution
        final_solution = qa_result.get('final_solution', '')
        if final_solution:
            stats['solution_lengths'].append(len(final_solution))

        # Track QA metrics
        if 'qa_pairs' in qa_result:
            stats['num_questions_used'].append(len(qa_result['qa_pairs']))

        if 'metrics' in qa_result:
            metrics = qa_result['metrics']
            if 'quality_scores' in metrics:
                scores = metrics['quality_scores']
                if len(scores) >= 2:
                    improvement = scores[-1] - scores[0]
                    stats['quality_improvements'].append(improvement)

            # Check if any iteration got it right
            if 'iteration_correct' in metrics:
                for i, correct in enumerate(metrics['iteration_correct']):
                    if correct:
                        stats['iterations_to_correct'].append(i)
                        break

        # Extract answer from final solution
        extracted = extract_numerical_answer(final_solution)
        stats['extracted_answers'][problem_id] = extracted

        # Check correctness
        if extracted is None:
            stats['no_answer'] += 1
            stats['no_answer_problems'].append(problem_id)
        elif correct_answer is not None:
            try:
                correct_num = int(correct_answer)
                if extracted == correct_num:
                    stats['correct'] += 1
                    stats['correct_problems'].append(problem_id)
                else:
                    stats['incorrect'] += 1
                    stats['incorrect_problems'].append({
                        'id': problem_id,
                        'predicted': extracted,
                        'correct': correct_num
                    })
            except:
                stats['incorrect'] += 1
                stats['incorrect_problems'].append({
                    'id': problem_id,
                    'predicted': extracted,
                    'correct': correct_answer
                })

    # Calculate statistics
    if stats['total_problems'] > 0:
        stats['accuracy'] = stats['correct'] / stats['total_problems']
        stats['avg_solution_length'] = np.mean(
            stats['solution_lengths']) if stats['solution_lengths'] else 0
        stats['avg_questions_used'] = np.mean(
            stats['num_questions_used']) if stats['num_questions_used'] else 0
        stats['avg_quality_improvement'] = np.mean(
            stats['quality_improvements']
        ) if stats['quality_improvements'] else 0
        if stats['iterations_to_correct']:
            stats['avg_iterations_to_correct'] = np.mean(
                stats['iterations_to_correct'])

    return stats


def print_model_summary(stats: Dict, verbose: bool = False):
    """Print summary statistics for a model."""
    print(f"\n{'='*60}")
    print(f"MODEL: {stats['model'].upper()}")
    print(f"{'='*60}")

    print(f"Total problems: {stats['total_problems']}")
    print(
        f"Correct: {stats['correct']} ({stats['correct']}/{stats['total_problems']})"
    )
    print(f"Incorrect: {stats['incorrect']}")
    print(f"No answer extracted: {stats['no_answer']}")
    print(f"Accuracy: {stats['accuracy']:.1%}")

    if 'avg_solution_length' in stats:
        print(f"Avg solution length: {stats['avg_solution_length']:.0f} chars")

    # QA-specific stats
    if stats['model'] == 'qa' and 'avg_questions_used' in stats:
        print(f"\nQA Method Statistics:")
        print(f"  Avg questions used: {stats['avg_questions_used']:.1f}")
        if 'avg_quality_improvement' in stats:
            print(
                f"  Avg quality improvement: {stats['avg_quality_improvement']:.2f}"
            )
        if 'avg_iterations_to_correct' in stats:
            print(
                f"  Avg iterations to correct: {stats['avg_iterations_to_correct']:.1f}"
            )

    if verbose:
        if stats['correct_problems']:
            print(
                f"\nCorrect problems: {sorted(stats['correct_problems'])[:10]}"
            )
            if len(stats['correct_problems']) > 10:
                print(f"  ... and {len(stats['correct_problems']) - 10} more")

        if stats['incorrect_problems'] and len(
                stats['incorrect_problems']) > 0:
            print(f"\nIncorrect problems (first 5):")
            for prob in stats['incorrect_problems'][:5]:
                if isinstance(prob, dict):
                    print(
                        f"  {prob['id']}: predicted {prob['predicted']}, correct {prob['correct']}"
                    )
                else:
                    print(f"  {prob}")

        if stats['no_answer_problems']:
            print(f"\nNo answer extracted: {stats['no_answer_problems'][:5]}")
            if len(stats['no_answer_problems']) > 5:
                print(f"  ... and {len(stats['no_answer_problems']) - 5} more")


def compare_models(all_stats: Dict[str, Dict]):
    """Print comparison between different models."""
    print(f"\n{'='*70}")
    print("MODEL COMPARISON")
    print(f"{'='*70}")

    # Create comparison table
    models = sorted(all_stats.keys())

    # Header
    print(
        f"{'Model':<10} {'Problems':<10} {'Correct':<10} {'Accuracy':<10} {'No Answer':<10}"
    )
    print("-" * 50)

    # Data rows
    for model in models:
        stats = all_stats[model]
        print(
            f"{model:<10} {stats['total_problems']:<10} {stats['correct']:<10} "
            f"{stats['accuracy']:<10.1%} {stats['no_answer']:<10}")

    # Find problems solved by some but not all
    if len(models) > 1:
        print(f"\n{'='*70}")
        print("PROBLEM DIFFICULTY ANALYSIS")
        print(f"{'='*70}")

        # Collect all problem IDs
        all_problems = set()
        for stats in all_stats.values():
            all_problems.update(stats['correct_problems'])
            all_problems.update([
                p['id'] if isinstance(p, dict) else p
                for p in stats['incorrect_problems']
            ])

        # Categorize problems by who solved them
        problem_solvers = defaultdict(list)
        for prob_id in all_problems:
            solvers = []
            for model, stats in all_stats.items():
                if prob_id in stats['correct_problems']:
                    solvers.append(model)
            problem_solvers[tuple(sorted(solvers))].append(prob_id)

        # Print categories
        for solvers, problems in sorted(problem_solvers.items()):
            if solvers:
                solver_str = ', '.join(solvers)
                print(
                    f"\nSolved by {solver_str} only: {len(problems)} problems")
                if len(problems) <= 10:
                    print(f"  Problems: {sorted(problems)}")
                else:
                    print(f"  First 10: {sorted(problems)[:10]}")


def create_plots(all_stats: Dict[str, Dict],
                 output_path: Optional[str] = None):
    """Create visualization plots for the results."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed. Skipping plots.")
        return

    models = sorted(all_stats.keys())

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Accuracy comparison bar chart
    ax = axes[0, 0]
    accuracies = [all_stats[m]['accuracy'] * 100 for m in models]
    colors = ['#ff7f0e' if m == 'qa' else '#1f77b4' for m in models]
    bars = ax.bar(models, accuracies, color=colors)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Model Accuracy Comparison')
    ax.set_ylim(0, 100)
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f'{acc:.1f}%',
                ha='center',
                va='bottom')

    # 2. Problems breakdown stacked bar
    ax = axes[0, 1]
    correct = [all_stats[m]['correct'] for m in models]
    incorrect = [all_stats[m]['incorrect'] for m in models]
    no_answer = [all_stats[m]['no_answer'] for m in models]

    width = 0.6
    x = np.arange(len(models))
    ax.bar(x, correct, width, label='Correct', color='green', alpha=0.7)
    ax.bar(x,
           incorrect,
           width,
           bottom=correct,
           label='Incorrect',
           color='red',
           alpha=0.7)
    ax.bar(x,
           no_answer,
           width,
           bottom=np.array(correct) + np.array(incorrect),
           label='No Answer',
           color='gray',
           alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel('Number of Problems')
    ax.set_title('Problem Results Breakdown')
    ax.legend()

    # 3. Solution length comparison (if available)
    ax = axes[1, 0]
    avg_lengths = []
    for m in models:
        if 'avg_solution_length' in all_stats[m]:
            avg_lengths.append(all_stats[m]['avg_solution_length'])
        else:
            avg_lengths.append(0)

    if any(avg_lengths):
        bars = ax.bar(models, avg_lengths, color=colors)
        ax.set_ylabel('Average Solution Length (chars)')
        ax.set_title('Solution Length Comparison')
        for bar, length in zip(bars, avg_lengths):
            if length > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 50,
                        f'{length:.0f}',
                        ha='center',
                        va='bottom')
    else:
        ax.text(0.5, 0.5, 'No solution length data', ha='center', va='center')

    # 4. QA method specific plot (if QA results exist)
    ax = axes[1, 1]
    if 'qa' in all_stats and 'avg_questions_used' in all_stats['qa']:
        qa_stats = all_stats['qa']
        metrics = ['Avg Questions', 'Avg Quality Improvement', 'Accuracy']
        values = [
            qa_stats.get('avg_questions_used', 0) / 25 *
            100,  # Normalize to percentage
            qa_stats.get('avg_quality_improvement', 0) *
            10,  # Scale for visibility
            qa_stats['accuracy'] * 100
        ]

        bars = ax.bar(metrics, values, color=['blue', 'green', 'orange'])
        ax.set_ylabel('Value')
        ax.set_title('QA Method Metrics')
        ax.set_ylim(0, max(values) * 1.2)

        for bar, val, metric in zip(bars, values, metrics):
            label = f'{val:.1f}'
            if metric == 'Avg Questions':
                label = f'{qa_stats.get("avg_questions_used", 0):.1f}'
            elif metric == 'Avg Quality Improvement':
                label = f'{qa_stats.get("avg_quality_improvement", 0):.2f}'
            else:
                label = f'{val:.1f}%'
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    label,
                    ha='center',
                    va='bottom')
    else:
        ax.text(0.5, 0.5, 'No QA method data', ha='center', va='center')

    plt.suptitle('AIME Results Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze AIME results from different models")
    parser.add_argument("results_file", help="Path to AIME results JSON file")
    parser.add_argument("--verbose",
                        "-v",
                        action="store_true",
                        help="Show detailed problem-level results")
    parser.add_argument("--plot",
                        action="store_true",
                        help="Generate visualization plots")
    parser.add_argument("--output", "-o", help="Output path for plots")
    parser.add_argument("--models",
                        nargs="+",
                        help="Specific models to analyze (default: all found)")

    args = parser.parse_args()

    # Load results
    results_path = Path(args.results_file)
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        return 1

    with open(results_path, 'r') as f:
        results = json.load(f)

    print(f"Loaded {len(results)} problems from {results_path}")

    # Determine which models to analyze
    available_models = set()
    for result in results:
        if 'model_solutions' in result:
            available_models.update(result['model_solutions'].keys())

    if args.models:
        models_to_analyze = [m for m in args.models if m in available_models]
        if not models_to_analyze:
            print(
                f"Error: None of the specified models found. Available: {available_models}"
            )
            return 1
    else:
        models_to_analyze = list(available_models)

    print(f"Analyzing models: {', '.join(models_to_analyze)}")

    # Analyze each model
    all_stats = {}
    for model in models_to_analyze:
        if model == 'qa':
            stats = analyze_qa_results(results)
        else:
            stats = analyze_model_results(results, model)

        if stats['total_problems'] > 0:
            all_stats[model] = stats
            print_model_summary(stats, verbose=args.verbose)

    # Compare models
    if len(all_stats) > 1:
        compare_models(all_stats)

    # Generate plots if requested
    if args.plot:
        output_path = args.output
        if not output_path:
            # Default output path
            output_path = results_path.parent / f"aime_analysis_{results_path.stem}.png"
        create_plots(all_stats, output_path)

    return 0


if __name__ == "__main__":
    exit(main())
