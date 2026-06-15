#!/usr/bin/env python3
"""
Plot results from request-based compression experiment.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_results(results_path: str):
    """Load results from JSON file."""
    with open(results_path) as f:
        return json.load(f)


def plot_compression_ratio_comparison(data: dict, output_path: str = None):
    """
    Scatter plot showing compression ratio vs original length.
    Two series: original compressed, and succinct compressed.
    Both ratios are relative to the original text length.
    """
    results = data['results']

    # Extract data
    orig_lengths = [r['initial']['length'] for r in results]
    orig_compressed_bytes = [r['initial']['compressed_bytes'] for r in results]
    succ_compressed_bytes = [
        r['compressed']['compressed_bytes'] for r in results
    ]

    # Compression ratios (both relative to original length)
    orig_ratio = [
        cb / ol * 100 for cb, ol in zip(orig_compressed_bytes, orig_lengths)
    ]
    succ_ratio = [
        cb / ol * 100 for cb, ol in zip(succ_compressed_bytes, orig_lengths)
    ]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot both series
    ax.scatter(orig_lengths,
               orig_ratio,
               alpha=0.7,
               s=60,
               c='steelblue',
               edgecolors='black',
               linewidth=0.5,
               label='Original (compressed)')
    ax.scatter(orig_lengths,
               succ_ratio,
               alpha=0.7,
               s=60,
               c='coral',
               edgecolors='black',
               linewidth=0.5,
               label='Succinct (compressed)')

    # Add trend lines
    z1 = np.polyfit(orig_lengths, orig_ratio, 1)
    z2 = np.polyfit(orig_lengths, succ_ratio, 1)
    x_line = np.linspace(min(orig_lengths), max(orig_lengths), 100)
    ax.plot(x_line,
            np.poly1d(z1)(x_line),
            'steelblue',
            linestyle='--',
            linewidth=2,
            alpha=0.7)
    ax.plot(x_line,
            np.poly1d(z2)(x_line),
            'coral',
            linestyle='--',
            linewidth=2,
            alpha=0.7)

    # Labels and title
    ax.set_xlabel('Original Response Length (chars)', fontsize=12)
    ax.set_ylabel('Compression Ratio (% of original length)', fontsize=12)
    ax.set_title('Compression Ratio: Original vs "Just Ask" Succinct',
                 fontsize=14)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add summary stats annotation
    avg_orig = np.mean(orig_ratio)
    avg_succ = np.mean(succ_ratio)
    improvement = (avg_orig - avg_succ) / avg_orig * 100

    ax.annotate(
        f'Avg Original: {avg_orig:.2f}%\n'
        f'Avg Succinct: {avg_succ:.2f}%\n'
        f'Improvement: {improvement:.1f}%',
        xy=(0.02, 0.98),
        xycoords='axes fraction',
        ha='left',
        va='top',
        fontsize=11,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")

    plt.show()
    return fig


def plot_compression_before_vs_after(data: dict, output_path: str = None):
    """
    Scatter plot: x-axis = compression before, y-axis = compression after.
    Both ratios relative to original length.
    Points below diagonal = improvement from "just ask".
    """
    results = data['results']

    # Extract data
    orig_lengths = [r['initial']['length'] for r in results]
    orig_compressed_bytes = [r['initial']['compressed_bytes'] for r in results]
    succ_compressed_bytes = [
        r['compressed']['compressed_bytes'] for r in results
    ]

    # Compression ratios (both relative to original length)
    comp_before = [
        cb / ol * 100 for cb, ol in zip(orig_compressed_bytes, orig_lengths)
    ]
    comp_after = [
        cb / ol * 100 for cb, ol in zip(succ_compressed_bytes, orig_lengths)
    ]

    # Create figure
    fig, ax = plt.subplots(figsize=(9, 9))

    # Plot scatter
    ax.scatter(comp_before,
               comp_after,
               alpha=0.7,
               s=70,
               c='steelblue',
               edgecolors='black',
               linewidth=0.5)

    # Add diagonal line (y=x) for reference
    min_val = min(min(comp_before), min(comp_after))
    max_val = max(max(comp_before), max(comp_after))
    ax.plot([min_val, max_val], [min_val, max_val],
            'k--',
            alpha=0.5,
            linewidth=2,
            label='No change (y=x)')

    # Shade improvement region (below diagonal)
    ax.fill_between([min_val, max_val], [min_val, min_val], [min_val, max_val],
                    alpha=0.15,
                    color='green',
                    label='Improvement region')

    # Add trend line
    z = np.polyfit(comp_before, comp_after, 1)
    x_line = np.linspace(min(comp_before), max(comp_before), 100)
    ax.plot(x_line,
            np.poly1d(z)(x_line),
            'r-',
            linewidth=2,
            alpha=0.7,
            label=f'Fit: y = {z[0]:.2f}x + {z[1]:.2f}')

    # Labels and title
    ax.set_xlabel('Compression Before (% of original length)', fontsize=12)
    ax.set_ylabel('Compression After "Just Ask" (% of original length)',
                  fontsize=12)
    ax.set_title('Compression Improvement from "Just Ask"', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Summary stats
    improved = sum(1 for b, a in zip(comp_before, comp_after) if a < b)
    avg_before = np.mean(comp_before)
    avg_after = np.mean(comp_after)
    improvement = (avg_before - avg_after) / avg_before * 100

    ax.annotate(
        f'Improved: {improved}/{len(results)} ({improved/len(results)*100:.0f}%)\n'
        f'Avg before: {avg_before:.2f}%\n'
        f'Avg after: {avg_after:.2f}%\n'
        f'Avg improvement: {improvement:.1f}%',
        xy=(0.98, 0.98),
        xycoords='axes fraction',
        ha='right',
        va='top',
        fontsize=11,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")

    plt.show()
    return fig


def plot_before_after_comparison(data: dict, output_path: str = None):
    """
    Create side-by-side scatter plots showing:
    1. Length Before vs Length After
    2. Compression Before vs Compression After
    """
    results = data['results']

    # Extract data
    lens_before = [r['initial']['length'] for r in results]
    lens_after = [r['compressed']['length'] for r in results]
    comps_before = [r['initial']['compression_pct'] for r in results]
    comps_after = [r['compressed']['compression_pct'] for r in results]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Length Before vs After
    ax1.scatter(lens_before,
                lens_after,
                alpha=0.6,
                s=50,
                c='steelblue',
                edgecolors='black',
                linewidth=0.5)

    # Add diagonal line (y=x) for reference
    max_len = max(max(lens_before), max(lens_after))
    min_len = min(min(lens_before), min(lens_after))
    ax1.plot([min_len, max_len], [min_len, max_len],
             'k--',
             alpha=0.3,
             label='No change (y=x)')

    # Add trend line
    z = np.polyfit(lens_before, lens_after, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(lens_before), max(lens_before), 100)
    ax1.plot(x_line,
             p(x_line),
             'r-',
             alpha=0.7,
             linewidth=2,
             label=f'Fit: y = {z[0]:.2f}x + {z[1]:.0f}')

    ax1.set_xlabel('Length Before (chars)', fontsize=12)
    ax1.set_ylabel('Length After (chars)', fontsize=12)
    ax1.set_title('Text Length: Before vs After "Just Ask"', fontsize=14)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Add annotation
    avg_reduction = np.mean(lens_after) / np.mean(lens_before) * 100
    ax1.annotate(f'Avg reduction to {avg_reduction:.0f}% of original',
                 xy=(0.95, 0.05),
                 xycoords='axes fraction',
                 ha='right',
                 fontsize=11,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: Compression Before vs After
    ax2.scatter(comps_before,
                comps_after,
                alpha=0.6,
                s=50,
                c='coral',
                edgecolors='black',
                linewidth=0.5)

    # Add diagonal line (y=x) for reference
    max_comp = max(max(comps_before), max(comps_after))
    min_comp = min(min(comps_before), min(comps_after))
    ax2.plot([min_comp, max_comp], [min_comp, max_comp],
             'k--',
             alpha=0.3,
             label='No change (y=x)')

    # Add trend line
    z2 = np.polyfit(comps_before, comps_after, 1)
    p2 = np.poly1d(z2)
    x_line2 = np.linspace(min(comps_before), max(comps_before), 100)
    ax2.plot(x_line2,
             p2(x_line2),
             'r-',
             alpha=0.7,
             linewidth=2,
             label=f'Fit: y = {z2[0]:.2f}x + {z2[1]:.1f}')

    ax2.set_xlabel('Compression % Before', fontsize=12)
    ax2.set_ylabel('Compression % After', fontsize=12)
    ax2.set_title('Arithmetic Coding: Before vs After "Just Ask"', fontsize=14)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    # Add annotation - count how many improved vs worsened
    improved = sum(1 for i in range(len(results))
                   if comps_after[i] < comps_before[i])
    worsened = len(results) - improved
    ax2.annotate(
        f'Compression worsened: {worsened}/{len(results)} ({worsened/len(results)*100:.0f}%)\n'
        f'(Points above diagonal = worse)',
        xy=(0.95, 0.05),
        xycoords='axes fraction',
        ha='right',
        fontsize=11,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Shade the "worse" region (above diagonal)
    ax2.fill_between([min_comp, max_comp], [min_comp, max_comp],
                     [max_comp, max_comp],
                     alpha=0.1,
                     color='red',
                     label='Worse compression')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")

    plt.show()
    return fig


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(
        description="Plot request-based compression results")
    parser.add_argument(
        "--results",
        type=str,
        default=
        "results/request_based_compression/request_based_compression_20260112_192617.json",
        help="Path to results JSON")
    parser.add_argument("--output-dir",
                        type=str,
                        default="writing/695fe28d3a9ed52bd3824bba/assets/plts",
                        help="Output directory for plots")
    parser.add_argument(
        "--plot-type",
        type=str,
        choices=["ratio", "before_after", "scatter"],
        default="scatter",
        help=
        "Type of plot: 'ratio' for compression ratio vs length, 'before_after' for side-by-side, 'scatter' for before vs after"
    )
    parser.add_argument("--format",
                        type=str,
                        choices=["png", "pdf"],
                        default="pdf",
                        help="Output format for plots")

    args = parser.parse_args()

    # Handle relative paths
    base_dir = Path(__file__).parent.parent.parent
    results_path = base_dir / args.results if not Path(
        args.results).is_absolute() else Path(args.results)
    output_dir = base_dir / args.output_dir if not Path(
        args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Informative output filenames
    plot_names = {
        "ratio": "just_ask_compression_vs_length",
        "scatter": "just_ask_compression_scatter",
        "before_after": "just_ask_before_after_comparison",
    }
    output_filename = f"{plot_names[args.plot_type]}.{args.format}"
    output_path = output_dir / output_filename

    print(f"Loading results from: {results_path}")
    data = load_results(results_path)

    print(f"Plotting {len(data['results'])} problems...")
    if args.plot_type == "ratio":
        plot_compression_ratio_comparison(data, str(output_path))
    elif args.plot_type == "scatter":
        plot_compression_before_vs_after(data, str(output_path))
    else:
        plot_before_after_comparison(data, str(output_path))
