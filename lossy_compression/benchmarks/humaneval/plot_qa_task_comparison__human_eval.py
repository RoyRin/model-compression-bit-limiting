#!/usr/bin/env python3
"""
Plot QA task comparison results, showing success rates for medium vs hard problems
across different question-generating models (haiku, sonnet, opus).
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple


def load_task_comparison(filepath: str) -> Dict:
    """Load the task comparison JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def load_problem_categories(
        analysis_filepath: str) -> Tuple[List[str], List[str]]:
    """Load medium and hard problem lists from the main analysis file."""
    with open(analysis_filepath, 'r') as f:
        data = json.load(f)

    medium_problems = data['problem_categories']['medium']
    hard_problems = data['problem_categories']['hard']

    return medium_problems, hard_problems


def calculate_success_rates(task_comparison: Dict, medium_problems: List[str],
                            hard_problems: List[str]) -> Dict:
    """Calculate success rates for each model on medium and hard problems."""

    results = {
        'medium': {
            'haiku': [],
            'sonnet': [],
            'opus': []
        },
        'hard': {
            'haiku': [],
            'sonnet': [],
            'opus': []
        }
    }

    # Process each task
    for task_id, task_data in task_comparison.items():
        # Determine if this is a medium or hard problem
        if task_id in medium_problems:
            difficulty = 'medium'
        elif task_id in hard_problems:
            difficulty = 'hard'
        else:
            continue  # Skip if not in our target set

        # Check which models passed this task
        for model_config, model_result in task_data['details'].items():
            # Extract the question model from the config name
            # Format: QA_LLM-opus_SLM-haiku_Q-{model}
            if '_Q-' in model_config:
                q_model = model_config.split('_Q-')[-1]
                # Handle full model names
                if 'haiku' in q_model:
                    q_model = 'haiku'
                elif 'sonnet' in q_model:
                    q_model = 'sonnet'
                elif 'opus' in q_model:
                    q_model = 'opus'

                if q_model in results[difficulty]:
                    results[difficulty][q_model].append(
                        1 if model_result['passed'] else 0)

    # Calculate success rates
    success_rates = {'medium': {}, 'hard': {}}

    for difficulty in ['medium', 'hard']:
        for model in ['haiku', 'sonnet', 'opus']:
            if results[difficulty][model]:
                success_rates[difficulty][model] = (
                    sum(results[difficulty][model]) /
                    len(results[difficulty][model]) * 100)
            else:
                success_rates[difficulty][model] = 0

    # Also return counts for the labels
    counts = {
        'medium': {
            model: len(results['medium'][model])
            for model in ['haiku', 'sonnet', 'opus']
        },
        'hard': {
            model: len(results['hard'][model])
            for model in ['haiku', 'sonnet', 'opus']
        }
    }

    return success_rates, counts


def create_bar_plot(success_rates: Dict,
                    counts: Dict,
                    output_path: str = None):
    """Create a grouped bar plot showing success rates."""

    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Data for plotting
    models = ['Haiku', 'Sonnet', 'Opus']
    medium_rates = [
        success_rates['medium']['haiku'], success_rates['medium']['sonnet'],
        success_rates['medium']['opus']
    ]
    hard_rates = [
        success_rates['hard']['haiku'], success_rates['hard']['sonnet'],
        success_rates['hard']['opus']
    ]

    # Set up bar positions
    x = np.arange(len(models))  # Label locations
    width = 0.35  # Width of bars

    # Create bars
    bars1 = ax.bar(x - width / 2,
                   medium_rates,
                   width,
                   label='Medium Problems',
                   color='skyblue',
                   edgecolor='black')
    bars2 = ax.bar(x + width / 2,
                   hard_rates,
                   width,
                   label='Hard Problems',
                   color='salmon',
                   edgecolor='black')

    # Add value labels on bars
    def add_value_labels(bars, rates, counts_dict, difficulty):
        for bar, rate, model in zip(bars, rates, ['haiku', 'sonnet', 'opus']):
            height = bar.get_height()
            n = counts_dict[difficulty][model]
            ax.text(bar.get_x() + bar.get_width() / 2.,
                    height,
                    f'{rate:.1f}%\n(n={n})',
                    ha='center',
                    va='bottom',
                    fontsize=10)

    add_value_labels(bars1, medium_rates, counts, 'medium')
    add_value_labels(bars2, hard_rates, counts, 'hard')

    # Customize the plot
    ax.set_xlabel('Question-Generating Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('QA Performance: Medium vs Hard Problems by Question Model',
                 fontsize=14,
                 fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(loc='upper left', fontsize=11)
    ax.set_ylim(0, 105)  # Give some space for labels

    # Add grid for better readability
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Add a subtle background
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('white')

    plt.tight_layout()

    # Always save if output_path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")

    # Note: plt.show() is now handled in main() if --show flag is used

    return fig, ax


def print_summary_statistics(success_rates: Dict, counts: Dict):
    """Print a summary of the results."""
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    for difficulty in ['medium', 'hard']:
        print(f"\n{difficulty.upper()} PROBLEMS:")
        print("-" * 40)
        for model in ['haiku', 'sonnet', 'opus']:
            rate = success_rates[difficulty][model]
            n = counts[difficulty][model]
            print(
                f"  {model.capitalize():8s}: {rate:5.1f}% success rate (n={n})"
            )

    print("\n" + "=" * 60)
    print("KEY INSIGHTS:")
    print("=" * 60)

    # Calculate improvements
    for difficulty in ['medium', 'hard']:
        haiku_rate = success_rates[difficulty]['haiku']
        opus_rate = success_rates[difficulty]['opus']
        if haiku_rate > 0:
            improvement = ((opus_rate - haiku_rate) / haiku_rate) * 100
            print(
                f"• {difficulty.capitalize()} problems: Opus is {improvement:.1f}% better than Haiku"
            )
        else:
            print(
                f"• {difficulty.capitalize()} problems: Opus: {opus_rate:.1f}%, Haiku: {haiku_rate:.1f}%"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Plot QA task comparison results")
    parser.add_argument("task_comparison",
                        help="Path to qa_task_comparison JSON file")
    parser.add_argument(
        "analysis",
        help="Path to qa_analysis JSON file (for problem categories)")
    parser.add_argument(
        "--output",
        "-o",
        help=
        "Output path for the plot (default: saves to same directory as task_comparison file)",
        default=None)
    parser.add_argument("--show",
                        action="store_true",
                        help="Show plot in window after saving")
    parser.add_argument("--no-plot",
                        action="store_true",
                        help="Skip plotting, only show statistics")

    args = parser.parse_args()

    # Load data
    print("Loading task comparison data...")
    task_comparison = load_task_comparison(args.task_comparison)

    print("Loading problem categories...")
    medium_problems, hard_problems = load_problem_categories(args.analysis)

    print(
        f"Found {len(medium_problems)} medium problems and {len(hard_problems)} hard problems"
    )

    # Calculate success rates
    success_rates, counts = calculate_success_rates(task_comparison,
                                                    medium_problems,
                                                    hard_problems)

    # Print summary
    print_summary_statistics(success_rates, counts)

    # Create plot
    if not args.no_plot:
        # Determine output path
        if args.output:
            output_path = args.output
        else:
            # Default: save to same directory as task_comparison file
            task_comparison_path = Path(args.task_comparison)
            timestamp = task_comparison_path.stem.split('_')[
                -1]  # Extract timestamp
            output_path = task_comparison_path.parent / f"qa_performance_plot_{timestamp}.png"

        print("\nGenerating plot...")
        create_bar_plot(success_rates, counts, output_path)

        # If --show is specified, also display the plot
        if args.show:
            print("Displaying plot in window...")
            plt.show()
            print("Plot window closed.")


if __name__ == "__main__":
    main()
