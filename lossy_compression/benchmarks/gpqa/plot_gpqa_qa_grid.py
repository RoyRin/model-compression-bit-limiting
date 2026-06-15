#!/usr/bin/env python3
"""
Plot GPQA QA sweep results in a 9x3 grid showing accuracy for different model combinations.
Creates two heatmap tables for medium and hard problems.

Usage (run from lossy_compression directory):
    python plot_gpqa_qa_grid.py --show
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, List, Tuple
import pandas as pd
from datetime import datetime


def load_gpqa_baseline(baseline_path: str) -> Dict:
    """Load GPQA baseline results to categorize problems by difficulty."""
    with open(baseline_path, 'r') as f:
        data = json.load(f)

    results = {}
    for problem in data['results']:
        problem_id = problem['problem_id']
        correct = problem['correct']
        results[problem_id] = correct

    return results


def categorize_gpqa_problems(
        baseline_path: str = None) -> Dict[str, List[int]]:
    """
    Categorize GPQA problems based on which baseline models solved them.

    Categories:
    - easy: All models get it (haiku, sonnet, and opus)
    - medium: Haiku doesn't get it, but sonnet and opus do
    - hard: Only opus gets it (or no model gets it)
    """

    # Default path if not provided
    if not baseline_path:
        # Try to find a baseline file automatically
        from pathlib import Path
        results_dir = Path("results")
        if results_dir.exists():
            baseline_files = list(
                results_dir.glob("gpqa_haiku_sonnet_opus_*.json"))
            if baseline_files:
                baseline_path = str(baseline_files[0])
                print(f"Using baseline file: {baseline_path}")
            else:
                raise FileNotFoundError(
                    "No baseline file found. Please specify with --baseline")
        else:
            raise FileNotFoundError("Results directory not found")

    # Load the combined baseline file
    with open(baseline_path, 'r') as f:
        data = json.load(f)

    haiku_results = {}
    sonnet_results = {}
    opus_results = {}

    # Check if it's the old format with separate model sections
    if 'haiku' in data and 'sonnet' in data and 'opus' in data:
        # New format with separate sections for each model
        for problem in data['haiku']['results']:
            problem_id = problem['problem_id']
            haiku_results[problem_id] = problem['is_correct']

        for problem in data['sonnet']['results']:
            problem_id = problem['problem_id']
            sonnet_results[problem_id] = problem['is_correct']

        for problem in data['opus']['results']:
            problem_id = problem['problem_id']
            opus_results[problem_id] = problem['is_correct']
    else:
        # Old format with combined results
        for problem in data.get('results', []):
            problem_id = problem['problem_id']
            haiku_results[problem_id] = problem.get('haiku_correct', False)
            sonnet_results[problem_id] = problem.get('sonnet_correct', False)
            opus_results[problem_id] = problem.get('opus_correct', False)

    categories = {
        'easy': [],
        'medium': [],
        'hard': [],
        'very_hard': [],
        'other': []
    }

    # Get all unique problem IDs
    all_problems = set(haiku_results.keys())

    for problem_id in sorted(all_problems):
        haiku_correct = haiku_results.get(problem_id, False)
        sonnet_correct = sonnet_results.get(problem_id, False)
        opus_correct = opus_results.get(problem_id, False)

        # Exact categorization from analyze_gpqa_results.py
        # This matches what get_problem_difficulty_indices uses
        if haiku_correct and sonnet_correct and opus_correct:
            categories['easy'].append(problem_id)
        elif not haiku_correct and sonnet_correct and opus_correct:
            categories['medium'].append(problem_id)
        elif not haiku_correct and not sonnet_correct and opus_correct:
            categories['hard'].append(problem_id)
        elif not haiku_correct and not sonnet_correct and not opus_correct:
            categories['very_hard'].append(problem_id)
        else:
            # Unexpected pattern (e.g., haiku passes but opus doesn't)
            categories['other'].append(problem_id)

    return categories


def load_qa_results(results_dir: str = "results",
                    include_timestamped: bool = False) -> Dict:
    """Load all GPQA QA results from individual JSON files.

    Args:
        results_dir: Directory containing result files
        include_timestamped: If True, include files with timestamp suffixes (default: False)
    """
    results = {}
    results_path = Path(results_dir)

    # Pattern: gpqa_qa_{slm}_{llm}_{question}_{difficulty}.json
    for result_file in results_path.glob("gpqa_qa_*_*_*_*.json"):
        filename_stem = result_file.stem

        # Check if filename ends with a timestamp (YYYYMMDD_HHMMSS pattern)
        # Timestamp would be last part after splitting by underscore
        parts_all = filename_stem.split('_')
        if len(parts_all) >= 2:
            last_part = parts_all[-1]
            second_last = parts_all[-2] if len(parts_all) > 1 else ""

            # Check if it looks like a timestamp (8 digits _ 6 digits)
            is_timestamped = (len(last_part) == 6 and last_part.isdigit()
                              and len(second_last) == 8
                              and second_last.isdigit())

            if is_timestamped and not include_timestamped:
                continue  # Skip timestamped files

        # Parse filename
        parts = filename_stem.replace('gpqa_qa_', '').split('_')
        if len(parts) >= 4:
            slm = parts[0]
            llm = parts[1]
            question = parts[2]
            # difficulty is the rest (might be "medium+hard")
            # Remove timestamp if present
            if is_timestamped and len(parts) >= 6:
                # Remove last two parts (date and time)
                difficulty = '_'.join(parts[3:-2])
            else:
                difficulty = '_'.join(parts[3:])

            # Only process medium+hard files
            if 'medium' not in difficulty and 'hard' not in difficulty:
                continue

            config_key = (slm, llm, question)

            # Load the results
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)

                # Convert accuracy from decimal to percentage if needed
                accuracy = data.get('accuracy', 0)
                if accuracy <= 1.0:  # Assume it's a decimal
                    accuracy = accuracy * 100

                results[config_key] = {
                    'accuracy': accuracy,
                    'correct_count': data.get('correct_count', 0),
                    'total_problems': data.get('total_problems', 0),
                    'results': data.get('results', []),
                    'avg_questions': data.get('avg_questions', 0)
                }

                print(
                    f"Loaded: SLM={slm}, LLM={llm}, Q={question} - Accuracy: {results[config_key]['accuracy']:.1f}%"
                )
            except Exception as e:
                print(f"Error loading {result_file}: {e}")

    return results


def calculate_accuracy_matrix(results: Dict,
                              problem_subset: List[int] = None,
                              debug: bool = False) -> pd.DataFrame:
    """Calculate accuracy matrix for the 9x3 grid."""
    # Define the order of models
    models = ['haiku', 'sonnet', 'opus']

    if debug and problem_subset:
        print(
            f"\nDEBUG: Calculating matrix for subset with {len(problem_subset)} problems"
        )
        print(f"  First 5 problem IDs in subset: {problem_subset[:5]}")

    # Create all SLM-LLM pairs (SLM first, then LLM)
    slm_llm_pairs = []
    for slm in models:
        for llm in models:
            slm_llm_pairs.append((slm, llm))

    # Initialize the matrix
    matrix = np.zeros((9, 3))

    for i, (slm, llm) in enumerate(slm_llm_pairs):
        for j, q_model in enumerate(models):
            config_key = (slm, llm, q_model)

            if config_key in results:
                config_results = results[config_key]

                if problem_subset is None:
                    # Use overall accuracy
                    accuracy = config_results['accuracy']
                else:
                    # Calculate accuracy for subset
                    problem_results = config_results['results']
                    relevant_results = [
                        r for r in problem_results
                        if r.get('problem_id') in problem_subset
                    ]

                    if debug and i == 0 and j == 0:  # Debug first cell
                        print(f"\n  DEBUG: Config {config_key}")
                        if problem_results:
                            print(
                                f"    Sample problem IDs from results: {[r.get('problem_id') for r in problem_results[:5]]}"
                            )
                        print(
                            f"    Found {len(relevant_results)} matching problems"
                        )

                    if relevant_results:
                        # Check for both 'correct' and 'is_correct' field names
                        correct = sum(
                            1 for r in relevant_results
                            if r.get('is_correct', r.get('correct', False)))
                        total = len(relevant_results)
                        accuracy = (correct / total * 100) if total > 0 else 0
                    else:
                        accuracy = 0

                matrix[i, j] = accuracy

    # Create DataFrame with proper labels
    row_labels = [f"SLM:{slm}/LLM:{llm}" for slm, llm in slm_llm_pairs]
    col_labels = [f"Q:{model}" for model in models]

    df = pd.DataFrame(matrix, index=row_labels, columns=col_labels)
    return df


def plot_accuracy_heatmaps(results: Dict,
                           medium_problems: List[int],
                           hard_problems: List[int],
                           output_path: str = None):
    """Create two heatmap plots for medium and hard problems."""

    print("\nCalculating accuracy matrices...")
    medium_matrix = calculate_accuracy_matrix(results,
                                              medium_problems,
                                              debug=True)
    hard_matrix = calculate_accuracy_matrix(results, hard_problems, debug=True)

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))

    # Color map - use RdYlGn for red (bad) to green (good)
    cmap = 'RdYlGn'

    # Plot Medium Problems
    sns.heatmap(medium_matrix,
                annot=True,
                fmt='.1f',
                cmap=cmap,
                vmin=0,
                vmax=100,
                cbar=False,
                ax=axes[0],
                square=False)
    axes[0].set_title(
        f'Medium Problems (n={len(medium_problems)})\n(Haiku fails, Sonnet & Opus pass)',
        fontsize=14,
        fontweight='bold')
    axes[0].set_xlabel('Question-Generating Model (Q)',
                       fontsize=12,
                       fontweight='bold')
    axes[0].set_ylabel('Answer Model (SLM) / Reference Model (LLM)',
                       fontsize=12,
                       fontweight='bold')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0, ha='center')
    axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation=0, va='center')

    # Plot Hard Problems (with colorbar)
    sns.heatmap(hard_matrix,
                annot=True,
                fmt='.1f',
                cmap=cmap,
                vmin=0,
                vmax=100,
                cbar=True,
                cbar_kws={'label': 'Accuracy (%)'},
                ax=axes[1],
                square=False)
    axes[1].set_title(
        f'Hard Problems (n={len(hard_problems)})\n(Only Opus passes)',
        fontsize=14,
        fontweight='bold')
    axes[1].set_xlabel('Question-Generating Model (Q)',
                       fontsize=12,
                       fontweight='bold')
    axes[1].set_ylabel('')  # No y-axis label for second plot
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0, ha='center')
    axes[1].set_yticklabels(axes[1].get_yticklabels(), rotation=0, va='center')

    # Adjust layout
    plt.suptitle('GPQA Q&A Method Accuracy: Model Configuration Grid',
                 fontsize=16,
                 fontweight='bold')
    plt.tight_layout()

    # Save if path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")

        # Also save as PDF
        pdf_path = output_path.rsplit('.', 1)[0] + '.pdf'
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
        print(f"PDF saved to: {pdf_path}")

    return fig, axes, (medium_matrix, hard_matrix)


def print_summary_statistics(results: Dict, medium_problems: List[int],
                             hard_problems: List[int]):
    """Print summary statistics about the results."""
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    # Count total problems evaluated
    if results:
        first_result = next(iter(results.values()))
        total_problems = first_result.get('total_problems', 0)
    else:
        total_problems = 0

    print(f"\nProblem Distribution:")
    print(f"  Medium: {len(medium_problems):3d}")
    print(f"  Hard:   {len(hard_problems):3d}")
    print(f"  Total evaluated: {total_problems:3d}")

    # Best configurations
    print("\n" + "-" * 40)
    print("Best Configurations by Difficulty:")
    print("-" * 40)

    for difficulty, problem_set in [("Medium", medium_problems),
                                    ("Hard", hard_problems)]:
        best_config = None
        best_accuracy = 0

        for config_key, config_results in results.items():
            if problem_set:
                problem_results = config_results['results']
                relevant_results = [
                    r for r in problem_results
                    if r.get('problem_id') in problem_set
                ]

                if relevant_results:
                    correct = sum(1 for r in relevant_results
                                  if r.get('correct', False))
                    total = len(relevant_results)
                    accuracy = (correct / total * 100) if total > 0 else 0
                else:
                    accuracy = 0
            else:
                accuracy = config_results['accuracy']

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_config = config_key

        if best_config:
            slm, llm, q_model = best_config
            print(f"\n{difficulty}:")
            print(f"  Best: SLM={slm}, LLM={llm}, Q={q_model}")
            print(f"  Accuracy: {best_accuracy:.1f}%")

    # Average questions asked
    print("\n" + "-" * 40)
    print("Average Questions Asked:")
    print("-" * 40)

    total_questions = []
    for config_results in results.values():
        avg_q = config_results.get('avg_questions', 0)
        if avg_q > 0:
            total_questions.append(avg_q)

    if total_questions:
        print(
            f"  Overall average: {np.mean(total_questions):.1f} questions per problem"
        )
        print(
            f"  Range: {min(total_questions):.1f} - {max(total_questions):.1f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Plot GPQA QA sweep results in a grid format")
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing GPQA QA result files (default: results)")
    parser.add_argument(
        "--baseline",
        help="Path to baseline results file for problem categorization")
    parser.add_argument("--output", "-o", help="Output path for the plot")
    parser.add_argument("--show",
                        action="store_true",
                        help="Show plot in window")
    parser.add_argument("--no-plot",
                        action="store_true",
                        help="Skip plotting, only show statistics")
    parser.add_argument(
        "--include-timestamped",
        action="store_true",
        help="Include files with timestamp suffixes (default: ignore them)")

    args = parser.parse_args()

    # Load QA results
    print(f"Loading QA results from: {args.results_dir}")
    if not args.include_timestamped:
        print(
            "(Ignoring files with timestamp suffixes - use --include-timestamped to include them)"
        )
    results = load_qa_results(args.results_dir,
                              include_timestamped=args.include_timestamped)

    if not results:
        print("Error: No QA results found in directory")
        print("Looking for files matching pattern: gpqa_qa_*_*_*_*.json")
        return

    print(f"\nFound {len(results)} model configurations")

    # Categorize problems
    print("\nCategorizing problems by baseline difficulty...")
    categories = categorize_gpqa_problems(args.baseline)

    medium_problems = categories['medium']
    hard_problems = categories['hard']  # Only problems where opus alone passes

    print(f"\nProblem categorization:")
    print(f"  Easy:      {len(categories['easy'])} (all models pass)")
    print(
        f"  Medium:    {len(categories['medium'])} (haiku fails, sonnet & opus pass)"
    )
    print(f"  Hard:      {len(categories['hard'])} (only opus passes)")
    print(f"  Very Hard: {len(categories['very_hard'])} (no model passes)")
    if 'other' in categories:
        print(f"  Other:     {len(categories['other'])} (unexpected patterns)")

    print(f"\nUsing for plots (matching evaluate_gpqa_qa_compression.py):")
    print(f"  Medium problems: {len(medium_problems)}")
    print(f"  Hard problems:   {len(hard_problems)}")
    print(
        f"  Total (medium+hard): {len(medium_problems) + len(hard_problems)}")

    # Print summary statistics
    print_summary_statistics(results, medium_problems, hard_problems)

    # Create plots
    if not args.no_plot:
        # Determine output path
        if args.output:
            output_path = args.output
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"results/gpqa_qa_grid_{timestamp}.png"

        print("\nGenerating heatmap plots...")
        fig, axes, matrices = plot_accuracy_heatmaps(results, medium_problems,
                                                     hard_problems,
                                                     output_path)

        if args.show:
            print("Displaying plot...")
            plt.show()

        # Save matrices as CSV
        csv_dir = Path(
            output_path
        ).parent / f"gpqa_qa_grid_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        csv_dir.mkdir(exist_ok=True)

        medium_matrix, hard_matrix = matrices
        medium_matrix.to_csv(csv_dir / "medium_accuracy.csv")
        hard_matrix.to_csv(csv_dir / "hard_accuracy.csv")
        print(f"\nCSV data saved to: {csv_dir}")


if __name__ == "__main__":
    main()
