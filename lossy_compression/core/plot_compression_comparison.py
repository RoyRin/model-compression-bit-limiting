#!/usr/bin/env python3
"""
Plot compression comparison: Best-of-N vs Just Ask.

Creates a single figure with:
- Best-of-N temperature sampling (line)
- Best-of-N single prompt (line)
- Just Ask succinct compression (horizontal line)
- Initial verbose compression (dashed horizontal line)

All metrics are relative to original text size.
"""

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pathlib import Path


def load_best_of_n_data(json_path: str) -> dict:
    """Load Best-of-N experiment results."""
    with open(json_path) as f:
        return json.load(f)


def load_just_ask_data(json_path: str) -> dict:
    """Load Just Ask (request-based compression) results."""
    with open(json_path) as f:
        return json.load(f)


def extract_best_of_n_compression(data: dict) -> dict:
    """Extract compression % per N for each approach."""
    n_values = data['parameters']['n_values']
    approaches = data['parameters']['approaches']

    results_by_approach = {}
    for approach in approaches:
        results_by_approach[approach] = {n: [] for n in n_values}

    for problem in data['results']:
        for approach in approaches:
            if approach not in problem.get('approaches', {}):
                continue
            by_n = problem['approaches'][approach].get('by_n', {})
            for n in n_values:
                n_key = str(n)
                if n_key in by_n:
                    comp_pct = by_n[n_key]['selection']['by_compression'].get(
                        'compression_pct')
                    if comp_pct is not None:
                        results_by_approach[approach][n].append(comp_pct)

    return results_by_approach, n_values, approaches


def extract_just_ask_compression(data: dict) -> tuple:
    """
    Extract Just Ask compression metrics.

    Returns:
        initial_compression: avg(verbose_compressed / verbose_original)
        just_ask_compression: avg(succinct_compressed / verbose_original)
    """
    just_ask_ratios = []
    initial_ratios = []

    for r in data['results']:
        original_verbose_bytes = r['initial']['original_bytes']
        initial_compressed_bytes = r['initial']['compressed_bytes']
        succinct_compressed_bytes = r['compressed']['compressed_bytes']

        # Initial: compressed verbose / original verbose
        initial_ratio = (initial_compressed_bytes /
                         original_verbose_bytes) * 100
        initial_ratios.append(initial_ratio)

        # Just Ask: compressed succinct / original verbose
        just_ask_ratio = (succinct_compressed_bytes /
                          original_verbose_bytes) * 100
        just_ask_ratios.append(just_ask_ratio)

    return np.mean(initial_ratios), np.mean(just_ask_ratios)


def plot_compression_comparison(
    best_of_n_data: dict,
    just_ask_data: dict,
    output_path: str,
    title: str = "Compression Comparison: Best-of-N vs Just Ask",
):
    """Create the comparison plot."""

    # Extract data
    results_by_approach, n_values, approaches = extract_best_of_n_compression(
        best_of_n_data)
    initial_compression, just_ask_compression = extract_just_ask_compression(
        just_ask_data)

    print(f"Initial (verbose): {initial_compression:.2f}%")
    print(
        f"Just Ask (succinct compressed / original): {just_ask_compression:.2f}%"
    )

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = {
        'temperature': '#9b59b6',  # purple
        'single_prompt': '#f39c12',  # orange
        'initial': '#3498db',  # blue
        'just_ask': '#2ecc71',  # green
    }

    # Plot Best-of-N lines
    for approach in approaches:
        means = []
        stds = []
        for n in n_values:
            vals = results_by_approach[approach][n]
            if vals:
                means.append(np.mean(vals))
                stds.append(np.std(vals) /
                            np.sqrt(len(vals)))  # standard error
            else:
                means.append(0)
                stds.append(0)

        means = np.array(means)
        stds = np.array(stds)

        label = f'Best-of-N ({approach.replace("_", " ").title()})'
        marker = 'o' if approach == 'temperature' else 's'
        ax.plot(n_values,
                means,
                f'{marker}-',
                color=colors[approach],
                label=label,
                linewidth=2.5,
                markersize=10)
        ax.fill_between(n_values,
                        means - stds,
                        means + stds,
                        color=colors[approach],
                        alpha=0.2)

    # Plot horizontal lines for Just Ask
    ax.axhline(y=initial_compression,
               color=colors['initial'],
               linestyle='--',
               linewidth=2.5,
               label=f'Initial (verbose): {initial_compression:.1f}%')
    ax.axhline(y=just_ask_compression,
               color=colors['just_ask'],
               linestyle='-',
               linewidth=2.5,
               label=f'Just Ask (succinct): {just_ask_compression:.1f}%')

    ax.set_xlabel('N (Number of Solutions Generated)', fontsize=14)
    ax.set_ylabel('Compression % (lower = better)', fontsize=14)
    ax.set_title(f'{title}\n(All relative to original text size)', fontsize=16)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(n_values)
    ax.set_xlim(0.5, max(n_values) + 0.5)
    ax.set_ylim(0, 12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'\nSaved to: {output_path}')

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Plot compression comparison: Best-of-N vs Just Ask")

    parser.add_argument("--best-of-n",
                        type=str,
                        required=True,
                        help="Path to Best-of-N results JSON")
    parser.add_argument(
        "--just-ask",
        type=str,
        required=True,
        help="Path to Just Ask (request-based compression) results JSON")
    parser.add_argument("--output-dir",
                        type=str,
                        default="results/best_of_n_aime/plots",
                        help="Output directory for plot")
    parser.add_argument(
        "--title",
        type=str,
        default="Compression Comparison: Best-of-N vs Just Ask",
        help="Plot title")
    parser.add_argument("--format",
                        type=str,
                        choices=["png", "pdf"],
                        default="pdf",
                        help="Output format")
    parser.add_argument("--show",
                        action="store_true",
                        help="Show plot interactively")

    args = parser.parse_args()

    # Load data
    best_of_n_data = load_best_of_n_data(args.best_of_n)
    just_ask_data = load_just_ask_data(args.just_ask)

    # Create output path with datestring
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    datestr = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = output_dir / f"compression_comparison_{datestr}.{args.format}"

    # Create plot
    plot_compression_comparison(
        best_of_n_data,
        just_ask_data,
        str(output_path),
        title=args.title,
    )

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
