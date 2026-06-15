#!/usr/bin/env python3
"""
Plot QA sweep results in a 9x3 grid showing accuracy for different model combinations.
Creates three heatmap tables for medium, hard, and overall problems.

python plot_qa_sweep_grid.py qa_sweep_medium_hard__human_eval/results/QA_q10 \
    --haiku results/claude-3-haiku-20240307/20250914_175801 \
    --sonnet results/claude-3-7-sonnet-20250219/20250914_181015 \
    --opus results/claude-opus-4-1-20250805/20250914_181338 \
    --show
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


def load_baseline_results_from_path(baseline_path: Path) -> Dict[str, bool]:
    """Load baseline results from a specific path."""
    if not baseline_path.exists():
        print(f"Warning: Baseline path not found: {baseline_path}")
        return {}

    results = {}
    problems_dir = baseline_path / "problems"

    if problems_dir.exists():
        # Load from individual problem files
        for problem_file in problems_dir.glob("*.json"):
            # Extract task ID from filename (e.g., "HumanEval_43.json" -> "HumanEval/43")
            task_id = problem_file.stem.replace("HumanEval_", "HumanEval/")
            with open(problem_file, 'r') as f:
                data = json.load(f)
                results[task_id] = data.get('passed', False)
    else:
        # Try loading from detailed_results.json if problems dir doesn't exist
        detailed_path = baseline_path / "detailed_results.json"
        if detailed_path.exists():
            with open(detailed_path, 'r') as f:
                data = json.load(f)
                for result in data.get('results', []):
                    task_id = result['task_id']
                    results[task_id] = result['passed']

    return results


def load_baseline_results(results_dir: Path,
                          model_dir_name: str) -> Dict[str, bool]:
    """Load baseline results for a single model from its directory."""
    model_path = results_dir / model_dir_name

    if not model_path.exists():
        print(f"Warning: Model results directory not found: {model_path}")
        return {}

    results = {}

    # Find the latest timestamp directory
    timestamp_dirs = sorted([d for d in model_path.iterdir() if d.is_dir()],
                            key=lambda x: x.name,
                            reverse=True)

    if not timestamp_dirs:
        print(f"Warning: No timestamp directories found in {model_path}")
        return {}

    # Use the latest timestamp directory
    latest_dir = timestamp_dirs[0]
    print(f"  Using baseline from: {model_dir_name}/{latest_dir.name}")

    problems_dir = latest_dir / "problems"

    if problems_dir.exists():
        # Load from individual problem files
        for problem_file in problems_dir.glob("*.json"):
            # Extract task ID from filename (e.g., "HumanEval_43.json" -> "HumanEval/43")
            task_id = problem_file.stem.replace("HumanEval_", "HumanEval/")
            with open(problem_file, 'r') as f:
                data = json.load(f)
                results[task_id] = data.get('passed', False)
    else:
        # Try loading from detailed_results.json if problems dir doesn't exist
        detailed_path = latest_dir / "detailed_results.json"
        if detailed_path.exists():
            with open(detailed_path, 'r') as f:
                data = json.load(f)
                for result in data.get('results', []):
                    task_id = result['task_id']
                    results[task_id] = result['passed']

    return results


def categorize_problems(haiku_results: Dict[str, bool],
                        sonnet_results: Dict[str, bool],
                        opus_results: Dict[str, bool]) -> Dict[str, List[str]]:
    """
    Categorize problems based on which baseline models solved them.
    
    Categories:
    - easy: All models get it (haiku, sonnet, and opus)
    - medium: Haiku doesn't get it, but sonnet and opus do
    - hard: Only opus gets it
    - very_hard: No model gets it
    - other: Any other combination (e.g., haiku gets it but opus doesn't - unexpected!)
    """
    categories = {
        'easy': [],  # All models pass
        'medium': [],  # Sonnet and Opus pass, Haiku fails
        'hard': [],  # Only Opus passes
        'very_hard': [],  # No model passes
        'other': []  # Unexpected patterns
    }

    # Get all unique task IDs
    all_tasks = set(haiku_results.keys()) | set(sonnet_results.keys()) | set(
        opus_results.keys())

    for task_id in all_tasks:
        haiku_passed = haiku_results.get(task_id, False)
        sonnet_passed = sonnet_results.get(task_id, False)
        opus_passed = opus_results.get(task_id, False)

        # Categorize based on pass pattern
        if haiku_passed and sonnet_passed and opus_passed:
            categories['easy'].append(task_id)
        elif not haiku_passed and sonnet_passed and opus_passed:
            categories['medium'].append(task_id)
        elif not haiku_passed and not sonnet_passed and opus_passed:
            categories['hard'].append(task_id)
        elif not haiku_passed and not sonnet_passed and not opus_passed:
            categories['very_hard'].append(task_id)
        else:
            # Unexpected pattern (e.g., haiku passes but opus doesn't)
            categories['other'].append(task_id)

    return categories


def load_qa_sweep_results(qa_sweep_dir: Path) -> Dict:
    """Load all QA sweep results from the directory structure."""
    results = {}

    # Parse directory names to extract model configurations
    for result_dir in qa_sweep_dir.glob("LLM-*_SLM-*_Q-*"):
        # Parse the directory name
        dir_name = result_dir.name
        parts = dir_name.split('_')

        # Extract model names
        llm = parts[0].replace('LLM-', '')
        slm = parts[1].replace('SLM-', '')
        q_model = parts[2].replace('Q-', '')

        # Create a key for this configuration
        config_key = (llm, slm, q_model)

        # Load problems from this directory
        problems_dir = result_dir / "problems"
        if problems_dir.exists():
            config_results = {'passed': [], 'failed': [], 'all_tasks': []}

            for problem_file in problems_dir.glob("*.json"):
                # Remove HumanEval_ prefix to get just the number
                task_id = problem_file.stem.replace("HumanEval_", "")
                config_results['all_tasks'].append(task_id)

                with open(problem_file, 'r') as f:
                    data = json.load(f)
                    if data.get('passed', False):
                        config_results['passed'].append(task_id)
                    else:
                        config_results['failed'].append(task_id)

            results[config_key] = config_results

    return results


def calculate_accuracy_matrix(results: Dict,
                              problem_subset: List[str] = None,
                              debug=False) -> pd.DataFrame:
    """Calculate accuracy matrix for the 9x3 grid."""
    # Define the order of models
    models = ['haiku', 'sonnet', 'opus']

    # Create all SLM-LLM pairs (SLM first, then LLM)
    # First 3 rows: SLM=haiku, vary LLM
    # Next 3 rows: SLM=sonnet, vary LLM
    # Last 3 rows: SLM=opus, vary LLM
    slm_llm_pairs = []
    for slm in models:
        for llm in models:
            slm_llm_pairs.append((slm, llm))

    # Initialize the matrix
    matrix = np.zeros((9, 3))

    if debug and problem_subset:
        print(
            f"\nDEBUG: Calculating accuracy for subset with {len(problem_subset)} problems"
        )
        print(f"  First 5 problems in subset: {problem_subset[:5]}")

    for i, (slm, llm) in enumerate(slm_llm_pairs):
        for j, q_model in enumerate(models):
            config_key = (
                llm, slm, q_model
            )  # Note: config_key still uses (llm, slm, q) order from directory names

            if config_key in results:
                config_results = results[config_key]

                if problem_subset is None:
                    # Use all problems
                    total = len(config_results['all_tasks'])
                    passed = len(config_results['passed'])
                else:
                    # Use only the specified subset
                    relevant_tasks = [
                        t for t in config_results['all_tasks']
                        if t in problem_subset
                    ]
                    total = len(relevant_tasks)
                    passed = len([
                        t for t in config_results['passed']
                        if t in problem_subset
                    ])

                    if debug and i == 0 and j == 0:  # Debug first cell
                        print(f"\nDEBUG: Config {config_key}:")
                        print(
                            f"  All tasks sample: {config_results['all_tasks'][:5]}"
                        )
                        print(f"  Relevant tasks found: {total}")
                        print(f"  Passed in subset: {passed}")

                accuracy = (passed / total * 100) if total > 0 else 0
                matrix[i, j] = accuracy

    # Create DataFrame with proper labels
    row_labels = [f"SLM:{slm}/LLM:{llm}" for slm, llm in slm_llm_pairs]
    col_labels = [f"Q:{model}"
                  for model in models]  # Q = Question-generating model

    df = pd.DataFrame(matrix, index=row_labels, columns=col_labels)
    return df


def plot_accuracy_heatmaps(results: Dict,
                           easy_problems: List[str],
                           medium_problems: List[str],
                           hard_problems: List[str],
                           output_path: str = None):
    """Create three heatmap plots for easy, medium, and hard problems."""

    # Calculate accuracy matrices (with debug for first one)
    print("\nCalculating accuracy matrices...")
    medium_matrix = calculate_accuracy_matrix(results,
                                              medium_problems,
                                              debug=True)
    hard_matrix = calculate_accuracy_matrix(results, hard_problems)
    overall_matrix = calculate_accuracy_matrix(results, None)  # All problems

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 10))

    # Color map - use RdYlGn for red (bad) to green (good)
    cmap = 'RdYlGn'

    # Plot Medium Problems (leftmost - with y-axis label)
    sns.heatmap(
        medium_matrix,
        annot=True,
        fmt='.1f',
        cmap=cmap,
        vmin=0,
        vmax=100,
        cbar=False,  # No colorbar for this one
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
    # Ensure x-axis labels are visible
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0, ha='center')
    axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation=0, va='center')

    # Plot Hard Problems (middle - no y-axis label)
    sns.heatmap(
        hard_matrix,
        annot=True,
        fmt='.1f',
        cmap=cmap,
        vmin=0,
        vmax=100,
        cbar=False,  # No colorbar for this one
        ax=axes[1],
        square=False)
    axes[1].set_title(
        f'Hard Problems (n={len(hard_problems)})\n(Only Opus passes)',
        fontsize=14,
        fontweight='bold')
    axes[1].set_xlabel('Question-Generating Model (Q)',
                       fontsize=12,
                       fontweight='bold')
    axes[1].set_ylabel('')  # No y-axis label for middle plot
    # Ensure x-axis labels are visible
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0, ha='center')
    axes[1].set_yticklabels(axes[1].get_yticklabels(), rotation=0, va='center')

    # Plot Overall (rightmost - with single colorbar)
    all_problems = len(set().union(*[r['all_tasks']
                                     for r in results.values()]))
    sns.heatmap(overall_matrix,
                annot=True,
                fmt='.1f',
                cmap=cmap,
                vmin=0,
                vmax=100,
                cbar=True,
                cbar_kws={'label': 'Accuracy (%)'},
                ax=axes[2],
                square=False)
    axes[2].set_title(f'Overall (n={all_problems})\n(All problems tested)',
                      fontsize=14,
                      fontweight='bold')
    axes[2].set_xlabel('Question-Generating Model (Q)',
                       fontsize=12,
                       fontweight='bold')
    axes[2].set_ylabel('')  # No y-axis label for rightmost plot
    # Ensure x-axis labels are visible
    axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=0, ha='center')
    axes[2].set_yticklabels(axes[2].get_yticklabels(), rotation=0, va='center')

    # Adjust layout
    plt.suptitle('QA Method Accuracy: Model Configuration Grid',
                 fontsize=16,
                 fontweight='bold')
    plt.tight_layout()

    # Save if path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")

    return fig, axes, (medium_matrix, hard_matrix, overall_matrix)


def print_summary_statistics(results: Dict, easy_problems: List[str],
                             medium_problems: List[str],
                             hard_problems: List[str]):
    """Print summary statistics about the results."""
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    # Problem distribution
    total_problems = len(
        set().union(*[r['all_tasks'] for r in results.values()]))
    print(f"\nProblem Distribution:")
    print(
        f"  Easy:   {len(easy_problems):3d} ({len(easy_problems)/total_problems*100:.1f}%)"
    )
    print(
        f"  Medium: {len(medium_problems):3d} ({len(medium_problems)/total_problems*100:.1f}%)"
    )
    print(
        f"  Hard:   {len(hard_problems):3d} ({len(hard_problems)/total_problems*100:.1f}%)"
    )
    print(f"  Total:  {total_problems:3d}")

    # Best configurations for each difficulty
    print("\n" + "-" * 40)
    print("Best Configurations by Difficulty:")
    print("-" * 40)

    for difficulty, problem_set in [("Medium", medium_problems),
                                    ("Hard", hard_problems),
                                    ("Overall", None)]:
        best_config = None
        best_accuracy = 0

        for config_key, config_results in results.items():
            if problem_set is None:
                total = len(config_results['all_tasks'])
                passed = len(config_results['passed'])
            else:
                relevant_tasks = [
                    t for t in config_results['all_tasks'] if t in problem_set
                ]
                total = len(relevant_tasks)
                passed = len(
                    [t for t in config_results['passed'] if t in problem_set])

            accuracy = (passed / total * 100) if total > 0 else 0

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_config = config_key

        if best_config:
            llm, slm, q_model = best_config
            print(f"\n{difficulty}:")
            print(f"  Best: LLM={llm}, SLM={slm}, Q={q_model}")
            print(f"  Accuracy: {best_accuracy:.1f}%")

    # Model performance by role
    print("\n" + "-" * 40)
    print("Average Performance by Model Role:")
    print("-" * 40)

    for role in ['LLM', 'SLM', 'Q']:
        model_accuracies = {'haiku': [], 'sonnet': [], 'opus': []}

        for config_key, config_results in results.items():
            llm, slm, q_model = config_key
            total = len(config_results['all_tasks'])
            passed = len(config_results['passed'])
            accuracy = (passed / total * 100) if total > 0 else 0

            if role == 'LLM':
                model_accuracies[llm].append(accuracy)
            elif role == 'SLM':
                model_accuracies[slm].append(accuracy)
            else:  # Q model
                model_accuracies[q_model].append(accuracy)

        print(f"\n{role} Role:")
        for model in ['haiku', 'sonnet', 'opus']:
            if model_accuracies[model]:
                avg_acc = np.mean(model_accuracies[model])
                print(f"  {model:6s}: {avg_acc:5.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Plot QA sweep results in a grid format")
    parser.add_argument(
        "qa_dir",
        help=
        "Path to QA sweep results directory (e.g., qa_sweep_medium_hard__human_eval/results/QA_q10)"
    )
    parser.add_argument(
        "--haiku",
        required=True,
        help=
        "Path to haiku baseline results (e.g., results/claude-3-haiku-20240307/20250916_224234)"
    )
    parser.add_argument(
        "--sonnet",
        required=True,
        help=
        "Path to sonnet baseline results (e.g., results/claude-3-7-sonnet-20250219/20250916_224234)"
    )
    parser.add_argument(
        "--opus",
        required=True,
        help=
        "Path to opus baseline results (e.g., results/claude-opus-4-1-20250805/20250916_224234)"
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output path for the plot (default: saves to qa_dir parent)")
    parser.add_argument("--show",
                        action="store_true",
                        help="Show plot in window")
    parser.add_argument("--no-plot",
                        action="store_true",
                        help="Skip plotting, only show statistics")

    args = parser.parse_args()

    # Load results
    qa_dir = Path(args.qa_dir)
    if not qa_dir.exists():
        print(f"Error: Directory {qa_dir} does not exist")
        return

    print(f"Loading results from: {qa_dir}")
    results = load_qa_sweep_results(qa_dir)

    if not results:
        print("Error: No results found in directory")
        return

    print(f"Found {len(results)} model configurations")

    # Load problem categories from baseline results
    print("\nLoading baseline results:")
    print(f"  Haiku:  {args.haiku}")
    print(f"  Sonnet: {args.sonnet}")
    print(f"  Opus:   {args.opus}")

    haiku_results = load_baseline_results_from_path(Path(args.haiku))
    sonnet_results = load_baseline_results_from_path(Path(args.sonnet))
    opus_results = load_baseline_results_from_path(Path(args.opus))

    if not haiku_results:
        print(
            f"Error: Could not load haiku baseline results from {args.haiku}")
        return
    if not sonnet_results:
        print(
            f"Error: Could not load sonnet baseline results from {args.sonnet}"
        )
        return
    if not opus_results:
        print(f"Error: Could not load opus baseline results from {args.opus}")
        return

    print(f"\nLoaded baseline results:")
    print(f"  Haiku:  {len(haiku_results)} problems")
    print(f"  Sonnet: {len(sonnet_results)} problems")
    print(f"  Opus:   {len(opus_results)} problems")

    # Categorize problems
    categories = categorize_problems(haiku_results, sonnet_results,
                                     opus_results)

    # Extract medium and hard problems (removing HumanEval/ prefix for matching)
    easy_problems = [
        task_id.replace("HumanEval/", "") for task_id in categories['easy']
    ]
    medium_problems = [
        task_id.replace("HumanEval/", "") for task_id in categories['medium']
    ]
    hard_problems = [
        task_id.replace("HumanEval/", "") for task_id in categories['hard']
    ]
    very_hard_problems = [
        task_id.replace("HumanEval/", "")
        for task_id in categories['very_hard']
    ]

    print(f"\nProblem categorization:")
    print(f"  Easy:      {len(easy_problems)} (all models pass)")
    print(
        f"  Medium:    {len(medium_problems)} (haiku fails, sonnet & opus pass)"
    )
    print(f"  Hard:      {len(hard_problems)} (only opus passes)")
    print(f"  Very Hard: {len(very_hard_problems)} (no model passes)")

    # Print summary statistics
    print_summary_statistics(results, easy_problems, medium_problems,
                             hard_problems)

    # Create plots
    if not args.no_plot:
        # Determine output path
        if args.output:
            output_path = args.output
        else:
            # Save to parent directory with descriptive name and timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = qa_dir.parent.parent / f"qa_grid_heatmap_{timestamp}.png"

        print("\nGenerating heatmap plots...")
        fig, axes, matrices = plot_accuracy_heatmaps(results, easy_problems,
                                                     medium_problems,
                                                     hard_problems,
                                                     output_path)

        if args.show:
            print("Displaying plot...")
            plt.show()

        # Also save the matrices as CSV for further analysis
        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S") if not args.output else ""
        csv_dir_name = f"qa_grid_data_{timestamp}" if timestamp else "qa_grid_data"
        csv_dir = Path(output_path).parent / csv_dir_name
        csv_dir.mkdir(exist_ok=True)

        medium_matrix, hard_matrix, overall_matrix = matrices
        medium_matrix.to_csv(csv_dir / "medium_accuracy.csv")
        hard_matrix.to_csv(csv_dir / "hard_accuracy.csv")
        overall_matrix.to_csv(csv_dir / "overall_accuracy.csv")
        print(f"\nCSV data saved to: {csv_dir}")


if __name__ == "__main__":
    main()
