#!/usr/bin/env python3
"""
Plot Best-of-N Experiment Results.

Creates line plots with standard deviation bands showing:
- Accuracy vs N for different selection methods
- Agreement rate vs N
- Generation time vs N
"""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional


def load_results(json_path: str) -> Dict[str, Any]:
    """Load experiment results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def calculate_per_problem_accuracy(results: Dict[str, Any], approach: str,
                                   n: int, method: str) -> List[float]:
    """Calculate per-problem accuracy for error bars.

    Returns list of 0s and 1s (incorrect/correct) for each problem.
    """
    accuracies = []
    for problem in results['results']:
        if approach not in problem.get('approaches', {}):
            continue
        by_n = problem['approaches'][approach].get('by_n', {})
        # Handle both string and int keys
        n_key = n if n in by_n else str(n)
        if n_key not in by_n:
            continue

        is_correct = by_n[n_key]['selection'][method]['is_correct']
        accuracies.append(1.0 if is_correct else 0.0)

    return accuracies


def calculate_stderr(values: List[float]) -> float:
    """Calculate standard error of the mean."""
    if len(values) <= 1:
        return 0
    return np.std(values) / np.sqrt(len(values))


def plot_accuracy_comparison(results: Dict[str, Any],
                             output_path: Optional[str] = None):
    """Plot accuracy vs N for different selection methods."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    n_values = results['parameters']['n_values']
    approaches = results['parameters']['approaches']

    colors = {
        'by_compression': '#2ecc71',  # green
        'by_random': '#3498db',  # blue
        'by_majority': '#e74c3c',  # red
    }

    labels = {
        'by_compression': 'Best Compression',
        'by_random': 'Random (First)',
        'by_majority': 'Majority Vote',
    }

    for ax_idx, approach in enumerate(approaches):
        ax = axes[ax_idx]

        for method in ['by_compression', 'by_random', 'by_majority']:
            means = []
            stderrs = []

            for n in n_values:
                per_problem = calculate_per_problem_accuracy(
                    results, approach, n, method)
                if per_problem:
                    means.append(np.mean(per_problem) * 100)
                    stderrs.append(calculate_stderr(per_problem) * 100)
                else:
                    means.append(0)
                    stderrs.append(0)

            means = np.array(means)
            stderrs = np.array(stderrs)

            ax.plot(n_values,
                    means,
                    'o-',
                    color=colors[method],
                    label=labels[method],
                    linewidth=2,
                    markersize=8)
            ax.fill_between(n_values,
                            means - stderrs,
                            means + stderrs,
                            color=colors[method],
                            alpha=0.2)

        ax.set_xlabel('N (Number of Solutions)', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title(f'{approach.replace("_", " ").title()} Sampling',
                     fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(n_values)
        ax.set_ylim(0, 100)

    plt.suptitle('Accuracy vs N: Selection Method Comparison',
                 fontsize=16,
                 y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved accuracy plot to: {output_path}")

    return fig


def plot_agreement_and_time(results: Dict[str, Any],
                            output_path: Optional[str] = None):
    """Plot agreement rate and generation time vs N."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    n_values = results['parameters']['n_values']
    approaches = results['parameters']['approaches']

    colors = {
        'temperature': '#9b59b6',  # purple
        'single_prompt': '#f39c12',  # orange
    }

    # Agreement plot
    ax = axes[0]
    for approach in approaches:
        means = []
        stds = []

        for n in n_values:
            n_key = n if n in results['summary'].get(approach, {}) else str(n)
            if n_key in results['summary'].get(approach, {}):
                stats = results['summary'][approach][n_key]
                means.append(stats.get('avg_agreement', 0))
                stds.append(stats.get('avg_agreement_std', 0))
            else:
                means.append(0)
                stds.append(0)

        means = np.array(means)
        stds = np.array(stds)

        ax.plot(n_values,
                means,
                'o-',
                color=colors[approach],
                label=approach.replace('_', ' ').title(),
                linewidth=2,
                markersize=8)
        ax.fill_between(n_values,
                        means - stds,
                        means + stds,
                        color=colors[approach],
                        alpha=0.2)

    ax.set_xlabel('N (Number of Solutions)', fontsize=12)
    ax.set_ylabel('Agreement Rate (%)', fontsize=12)
    ax.set_title('Answer Agreement vs N', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(n_values)
    ax.set_ylim(0, 100)

    # Time plot
    ax = axes[1]
    for approach in approaches:
        means = []
        stds = []

        for n in n_values:
            n_key = n if n in results['summary'].get(approach, {}) else str(n)
            if n_key in results['summary'].get(approach, {}):
                stats = results['summary'][approach][n_key]
                means.append(stats.get('avg_time', 0))
                stds.append(stats.get('avg_time_std', 0))
            else:
                means.append(0)
                stds.append(0)

        means = np.array(means)
        stds = np.array(stds)

        ax.plot(n_values,
                means,
                'o-',
                color=colors[approach],
                label=approach.replace('_', ' ').title(),
                linewidth=2,
                markersize=8)
        ax.fill_between(n_values,
                        means - stds,
                        means + stds,
                        color=colors[approach],
                        alpha=0.2)

    ax.set_xlabel('N (Number of Solutions)', fontsize=12)
    ax.set_ylabel('Generation Time (seconds)', fontsize=12)
    ax.set_title('Generation Time vs N', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(n_values)

    plt.suptitle('Agreement and Timing Analysis', fontsize=16, y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved agreement/time plot to: {output_path}")

    return fig


def plot_compression_selection_detail(results: Dict[str, Any],
                                      output_path: Optional[str] = None):
    """Plot detailed compression selection analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    n_values = results['parameters']['n_values']
    approaches = results['parameters']['approaches']

    colors = {
        'temperature': '#9b59b6',
        'single_prompt': '#f39c12',
    }

    # Compression percentage of selected solution
    ax = axes[0]
    for approach in approaches:
        means = []
        stds = []

        for n in n_values:
            n_key = n if n in results['summary'].get(approach, {}) else str(n)
            if n_key in results['summary'].get(approach, {}):
                stats = results['summary'][approach][n_key]
                means.append(stats.get('avg_compression_pct', 0))
                stds.append(stats.get('avg_compression_pct_std', 0))
            else:
                means.append(0)
                stds.append(0)

        means = np.array(means)
        stds = np.array(stds)

        ax.plot(n_values,
                means,
                'o-',
                color=colors[approach],
                label=approach.replace('_', ' ').title(),
                linewidth=2,
                markersize=8)
        ax.fill_between(n_values,
                        means - stds,
                        means + stds,
                        color=colors[approach],
                        alpha=0.2)

    ax.set_xlabel('N (Number of Solutions)', fontsize=12)
    ax.set_ylabel('Compression % (lower = better)', fontsize=12)
    ax.set_title('Best Compression % vs N', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(n_values)

    # Accuracy improvement over random
    ax = axes[1]
    for approach in approaches:
        improvements = []

        for n in n_values:
            n_key = n if n in results['summary'].get(approach, {}) else str(n)
            if n_key in results['summary'].get(approach, {}):
                stats = results['summary'][approach][n_key]
                compress_acc = stats.get('accuracy_by_compression', 0)
                random_acc = stats.get('accuracy_by_random', 0)
                improvements.append(compress_acc - random_acc)
            else:
                improvements.append(0)

        ax.bar([
            x + (0.2 if approach == 'single_prompt' else -0.2)
            for x in n_values
        ],
               improvements,
               width=0.35,
               color=colors[approach],
               label=approach.replace('_', ' ').title(),
               alpha=0.8)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('N (Number of Solutions)', fontsize=12)
    ax.set_ylabel('Accuracy Improvement (%)', fontsize=12)
    ax.set_title('Compression Selection vs Random', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(n_values)

    plt.suptitle('Compression Selection Analysis', fontsize=16, y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved compression analysis plot to: {output_path}")

    return fig


def plot_compression_vs_n_comparison(best_of_n_results: Dict[str, Any],
                                     just_ask_results: Dict[str, Any],
                                     output_path: Optional[str] = None):
    """Plot compression % vs N for all three approaches: temperature, single_prompt, just_ask.

    Args:
        best_of_n_results: Results from best_of_n_comparison experiment
        just_ask_results: Results from just_ask_best_of_n experiment
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    n_values = best_of_n_results['parameters']['n_values']

    colors = {
        'temperature': '#9b59b6',  # purple
        'single_prompt': '#f39c12',  # orange
        'just_ask': '#2ecc71',  # green
    }

    labels = {
        'temperature': 'Temperature Sampling (T=0.8)',
        'single_prompt': 'Single Prompt ("Give N solutions")',
        'just_ask': 'Just Ask ("Write succinctly")',
    }

    # Plot temperature and single_prompt from best_of_n_results
    for approach in ['temperature', 'single_prompt']:
        means = []
        stds = []

        for n in n_values:
            n_key = n if n in best_of_n_results['summary'].get(approach,
                                                               {}) else str(n)
            if n_key in best_of_n_results['summary'].get(approach, {}):
                stats = best_of_n_results['summary'][approach][n_key]
                means.append(stats.get('avg_compression_pct', 0))
                stds.append(stats.get('avg_compression_pct_std', 0))
            else:
                means.append(0)
                stds.append(0)

        means = np.array(means)
        stds = np.array(stds)

        ax.plot(n_values,
                means,
                'o-',
                color=colors[approach],
                label=labels[approach],
                linewidth=2,
                markersize=8)
        ax.fill_between(n_values,
                        means - stds,
                        means + stds,
                        color=colors[approach],
                        alpha=0.2)

    # Plot just_ask from just_ask_results (compute std dev from rewrites data)
    # IMPORTANT: Use compression_vs_verbose (compressed_bytes / verbose_original_bytes)
    # NOT compression_pct (which is compressed_bytes / rewrite_own_bytes)
    just_ask_means = []
    just_ask_stds = []
    for n in n_values:
        # Get best compression vs verbose from first N rewrites for each problem
        best_compressions = []
        for problem in just_ask_results.get('results', []):
            rewrites = problem.get('rewrites', [])[:n]  # Get first N rewrites
            if rewrites:
                # Find best (lowest) compression_vs_verbose among first N
                # This measures: rewrite_compressed_bytes / verbose_original_bytes
                comp_values = [
                    r.get('compression_vs_verbose', 100) for r in rewrites
                    if r.get('compression_vs_verbose', 0) > 0
                ]
                if comp_values:
                    best_compressions.append(min(comp_values))

        if best_compressions:
            just_ask_means.append(np.mean(best_compressions))
            just_ask_stds.append(np.std(best_compressions))
        else:
            just_ask_means.append(0)
            just_ask_stds.append(0)

    just_ask_means = np.array(just_ask_means)
    just_ask_stds = np.array(just_ask_stds)
    ax.plot(n_values,
            just_ask_means,
            'o-',
            color=colors['just_ask'],
            label=labels['just_ask'],
            linewidth=2,
            markersize=8)
    ax.fill_between(n_values,
                    just_ask_means - just_ask_stds,
                    just_ask_means + just_ask_stds,
                    color=colors['just_ask'],
                    alpha=0.2)

    # Also plot the verbose baseline for just_ask (before compression)
    verbose_means = []
    for n in n_values:
        n_key = str(n)
        if n_key in just_ask_results['summary']:
            stats = just_ask_results['summary'][n_key]
            verbose_means.append(stats.get('avg_verbose_compression_pct', 0))
        else:
            verbose_means.append(0)

    verbose_means = np.array(verbose_means)
    ax.plot(n_values,
            verbose_means,
            's--',
            color=colors['just_ask'],
            label='Just Ask (before compression)',
            linewidth=1.5,
            markersize=6,
            alpha=0.6)

    ax.set_xlabel('N (Number of Solutions)', fontsize=12)
    ax.set_ylabel('Compression % (lower = more compressible)', fontsize=12)
    ax.set_title('Compression Ratio vs N: Comparing Generation Strategies',
                 fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(n_values)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved compression comparison plot to: {output_path}")

    return fig


def plot_all(results: Dict[str, Any],
             output_dir: str,
             fmt: str = "pdf",
             timestamp: str = None):
    """Generate all plots and save to output directory."""
    from datetime import datetime

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate timestamp if not provided
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Plot 1: Accuracy comparison
    plot_accuracy_comparison(
        results,
        output_path / f"best_of_n_accuracy_comparison_{timestamp}.{fmt}")

    # Plot 2: Agreement and time
    plot_agreement_and_time(
        results,
        output_path / f"best_of_n_agreement_and_time_{timestamp}.{fmt}")

    # Plot 3: Compression selection detail
    plot_compression_selection_detail(
        results,
        output_path / f"best_of_n_compression_analysis_{timestamp}.{fmt}")

    print(f"\nAll plots saved to: {output_dir}")

    return timestamp


def print_summary_table(results: Dict[str, Any]):
    """Print a summary table of results."""
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    n_values = results['parameters']['n_values']
    approaches = results['parameters']['approaches']

    for approach in approaches:
        print(f"\n{approach.upper().replace('_', ' ')}:")
        print("-" * 70)
        print(
            f"{'N':>5} {'Compress%':>12} {'Random%':>12} {'Majority%':>12} {'Agreement%':>12}"
        )
        print("-" * 70)

        for n in n_values:
            n_key = n if n in results['summary'].get(approach, {}) else str(n)
            if n_key in results['summary'].get(approach, {}):
                stats = results['summary'][approach][n_key]
                print(f"{n:>5} "
                      f"{stats.get('accuracy_by_compression', 0):>11.1f}% "
                      f"{stats.get('accuracy_by_random', 0):>11.1f}% "
                      f"{stats.get('accuracy_by_majority', 0):>11.1f}% "
                      f"{stats.get('avg_agreement', 0):>11.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Plot Best-of-N experiment results")

    parser.add_argument("json_file",
                        type=str,
                        help="Path to experiment results JSON file")
    parser.add_argument(
        "--just-ask",
        type=str,
        default=None,
        help=
        "Path to just-ask best-of-N results JSON file (for comparison plot)")
    parser.add_argument("--output-dir",
                        type=str,
                        default="writing/695fe28d3a9ed52bd3824bba/assets/plts",
                        help="Output directory for plots")
    parser.add_argument("--format",
                        "-f",
                        type=str,
                        choices=["png", "pdf"],
                        default="pdf",
                        help="Output format for plots (default: pdf)")
    parser.add_argument("--show",
                        action="store_true",
                        help="Show plots interactively")
    parser.add_argument("--summary",
                        action="store_true",
                        help="Print summary table only")

    args = parser.parse_args()

    # Load results
    results = load_results(args.json_file)

    # Print summary
    print_summary_table(results)

    if args.summary:
        return

    # Handle relative paths
    base_dir = Path(__file__).parent.parent.parent
    output_dir = base_dir / args.output_dir if not Path(
        args.output_dir).is_absolute() else Path(args.output_dir)

    # Generate standard plots (returns timestamp used)
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_all(results, str(output_dir), args.format, timestamp)

    # Generate comparison plot if just-ask results provided
    if args.just_ask:
        just_ask_results = load_results(args.just_ask)
        comparison_path = Path(
            output_dir
        ) / f"compression_vs_n_all_approaches_{timestamp}.{args.format}"
        plot_compression_vs_n_comparison(results, just_ask_results,
                                         str(comparison_path))

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
