#!/usr/bin/env python3
"""
Plot best compression ratio as a function of N.

This script reads experiment results and creates visualizations showing
how the best compression ratio improves with more generation samples.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


def load_experiment_data(json_file: str) -> Dict:
    """Load experiment data from JSON file."""
    with open(json_file, 'r') as f:
        return json.load(f)


def load_csv_summary(csv_file: str) -> pd.DataFrame:
    """Load summary statistics from CSV file."""
    return pd.read_csv(csv_file)


def plot_mean_ratio_vs_n(data: Dict,
                         output_file: Optional[str] = None,
                         show_individual: bool = True):
    """
    Plot mean compression ratio vs N with error bars.
    
    Args:
        data: Experiment data dictionary
        output_file: Path to save plot (if None, displays)
        show_individual: Whether to show individual prompt results
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Get sampling method from parameters if available
    sampling_method = data.get('parameters', {}).get('sampling_method',
                                                     'unknown')
    sampling_label = 'Iterative Diverse Sampling' if sampling_method == 'iterative_diverse' else 'Standard Sampling'

    # Extract summary statistics
    summary = data['summary_stats']
    n_values = sorted([int(n) for n in summary.keys()])

    means = [summary[str(n)]['mean'] for n in n_values]
    stds = [summary[str(n)]['std'] for n in n_values]

    # Plot mean with error bars
    ax.errorbar(n_values,
                means,
                yerr=stds,
                marker='o',
                markersize=8,
                linewidth=2,
                capsize=5,
                label='Mean ± Std',
                color='blue',
                alpha=0.8)

    # Optionally show individual prompt trajectories
    if show_individual and 'results' in data:
        for prompt_result in data['results']:
            prompt_n_values = []
            prompt_ratios = []

            for n in n_values:
                if str(n) in prompt_result['results_by_n']:
                    ratio = prompt_result['results_by_n'][str(n)].get(
                        'best_ratio')
                    if ratio is not None:
                        prompt_n_values.append(n)
                        prompt_ratios.append(ratio)

            if prompt_n_values:
                prompt_label = prompt_result['prompt'][:30] + '...' if len(
                    prompt_result['prompt']) > 30 else prompt_result['prompt']
                ax.plot(prompt_n_values,
                        prompt_ratios,
                        marker='.',
                        alpha=0.3,
                        linewidth=1,
                        markersize=4)

    ax.set_xlabel('N (Number of Generations)', fontsize=14)
    ax.set_ylabel('Best Compression Ratio', fontsize=14)
    ax.set_title(
        f'Best-of-N Compression: How Best Score Improves with N\n({sampling_label})',
        fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=12)

    # Set x-axis to show all tested values
    ax.set_xticks(n_values)
    ax.set_xticklabels(n_values)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"📊 Plot saved to: {output_file}")
    else:
        plt.show()


def plot_simple_ratio_vs_n(json_files: List[str],
                           output_file: Optional[str] = None):
    """
    Simple plot: Y-axis = Compression Ratio, X-axis = N
    Shows both diverse and standard sampling on the same plot.
    
    Args:
        json_files: List of paths to experiment JSON files
        output_file: Path to save plot (if None, displays)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define colors and markers for each sampling method
    styles = {
        'iterative_diverse': {
            'color': 'blue',
            'marker': 'o',
            'label': 'Diverse Sampling'
        },
        'standard': {
            'color': 'red',
            'marker': 's',
            'label': 'Standard Sampling'
        },
        'unknown': {
            'color': 'gray',
            'marker': '^',
            'label': 'Unknown'
        }
    }

    for json_file in json_files:
        # Load data
        data = load_experiment_data(json_file)

        # Detect sampling method
        sampling_method = data.get('parameters',
                                   {}).get('sampling_method', 'unknown')
        style = styles.get(sampling_method, styles['unknown'])

        # Extract summary statistics
        summary = data['summary_stats']
        n_values = sorted([int(n) for n in summary.keys()])

        # Get mean compression ratios
        means = [summary[str(n)]['mean'] for n in n_values]
        stds = [summary[str(n)]['std'] for n in n_values]

        # Plot with error bars
        ax.errorbar(n_values,
                    means,
                    yerr=stds,
                    marker=style['marker'],
                    markersize=8,
                    linewidth=2,
                    capsize=5,
                    label=style['label'],
                    color=style['color'],
                    alpha=0.8)

        # Print detected method for debugging
        print(
            f"📊 Detected sampling method: {sampling_method} from {Path(json_file).name}"
        )

    # Set x-axis ticks first to get all N values
    all_n_values = set()
    for json_file in json_files:
        data = load_experiment_data(json_file)
        summary = data['summary_stats']
        all_n_values.update([int(n) for n in summary.keys()])

    ax.set_xticks(sorted(all_n_values))
    ax.set_xticklabels(sorted(all_n_values))

    # Configure plot with N values in title
    n_vals_str = f" (N={{{','.join(map(str, sorted(all_n_values)))}}}" if len(
        all_n_values) <= 6 else ""
    n_vals_str += ")" if n_vals_str else ""

    ax.set_xlabel('N (Best-of-N Generations)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Compression Ratio', fontsize=14, fontweight='bold')
    ax.set_title(f'Best-of-N Compression Performance{n_vals_str}',
                 fontsize=16,
                 fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=12)

    # Add a subtle background
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('white')

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✅ Simple plot saved to: {output_file}")
    else:
        plt.show()


def plot_ratio_distribution(data: Dict, output_file: Optional[str] = None):
    """
    Plot box plots showing distribution of best ratios for each N.
    
    Args:
        data: Experiment data dictionary
        output_file: Path to save plot (if None, displays)
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Get sampling method from parameters if available
    sampling_method = data.get('parameters', {}).get('sampling_method',
                                                     'unknown')
    sampling_label = 'Iterative Diverse Sampling' if sampling_method == 'iterative_diverse' else 'Standard Sampling'

    # Collect all ratios for each N
    n_values = sorted([int(n) for n in data['summary_stats'].keys()])
    ratio_data = []

    for n in n_values:
        ratios = []
        for prompt_result in data['results']:
            if str(n) in prompt_result['results_by_n']:
                ratio = prompt_result['results_by_n'][str(n)].get('best_ratio')
                if ratio is not None:
                    ratios.append(ratio)
        ratio_data.append(ratios)

    # Create box plot
    bp = ax.boxplot(ratio_data,
                    positions=n_values,
                    widths=2,
                    patch_artist=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5))

    ax.set_xlabel('N (Number of Generations)', fontsize=14)
    ax.set_ylabel('Best Compression Ratio', fontsize=14)
    ax.set_title(
        f'Distribution of Best Compression Ratios vs N\n({sampling_label})',
        fontsize=16)
    ax.grid(True, alpha=0.3, axis='y')

    # Set x-axis
    ax.set_xticks(n_values)
    ax.set_xticklabels(n_values)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"📊 Plot saved to: {output_file}")
    else:
        plt.show()


def plot_improvement_curve(data: Dict, output_file: Optional[str] = None):
    """
    Plot relative improvement compared to N=1 baseline.
    
    Args:
        data: Experiment data dictionary
        output_file: Path to save plot (if None, displays)
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Get sampling method from parameters if available
    sampling_method = data.get('parameters', {}).get('sampling_method',
                                                     'unknown')
    sampling_label = 'Iterative Diverse Sampling' if sampling_method == 'iterative_diverse' else 'Standard Sampling'

    # Get baseline (N=1) performance
    summary = data['summary_stats']
    baseline = summary.get('1', summary.get(1))

    if not baseline:
        print("⚠️ No N=1 baseline found, skipping improvement plot")
        return

    baseline_mean = baseline['mean']

    # Calculate relative improvements
    n_values = sorted([int(n) for n in summary.keys()])
    improvements = []

    for n in n_values:
        mean_ratio = summary[str(n)]['mean']
        improvement = ((mean_ratio - baseline_mean) / baseline_mean) * 100
        improvements.append(improvement)

    # Plot improvement curve
    ax.plot(n_values,
            improvements,
            marker='o',
            markersize=8,
            linewidth=2,
            color='green')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Add percentage labels
    for n, imp in zip(n_values, improvements):
        ax.annotate(f'{imp:.1f}%',
                    xy=(n, imp),
                    xytext=(0, 5),
                    textcoords='offset points',
                    ha='center',
                    fontsize=10)

    ax.set_xlabel('N (Number of Generations)', fontsize=14)
    ax.set_ylabel('Improvement over N=1 (%)', fontsize=14)
    ax.set_title(
        f'Relative Improvement in Best Compression Ratio\n({sampling_label})',
        fontsize=16)
    ax.grid(True, alpha=0.3)

    # Set x-axis
    ax.set_xticks(n_values)
    ax.set_xticklabels(n_values)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"📊 Plot saved to: {output_file}")
    else:
        plt.show()


def plot_comparison(json_files: List[str], output_file: Optional[str] = None):
    """
    Compare results from multiple experiments (e.g., different sampling methods).
    
    Args:
        json_files: List of paths to experiment JSON files
        output_file: Path to save plot (if None, displays)
    """
    fig = plt.figure(figsize=(18, 10))

    # Create a 2x2 grid
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 3)
    ax4 = plt.subplot(2, 2, 4)

    colors = {
        'iterative_diverse': 'blue',
        'standard': 'red',
        'unknown': 'gray'
    }
    markers = {'iterative_diverse': 'o', 'standard': 's', 'unknown': '^'}

    all_data = []

    for idx, json_file in enumerate(json_files):
        data = load_experiment_data(json_file)
        summary = data['summary_stats']
        n_values = sorted([int(n) for n in summary.keys()])

        # Get sampling method
        sampling_method = data.get('parameters',
                                   {}).get('sampling_method', 'unknown')
        label = 'Iterative Diverse' if sampling_method == 'iterative_diverse' else 'Standard'

        color = colors.get(sampling_method, 'gray')
        marker = markers.get(sampling_method, '^')

        all_data.append((sampling_method, data, n_values))

        means = [summary[str(n)]['mean'] for n in n_values]
        stds = [summary[str(n)]['std'] for n in n_values]

        # Plot 1: Mean ratios with error bars
        ax1.errorbar(n_values,
                     means,
                     yerr=stds,
                     marker=marker,
                     markersize=8,
                     linewidth=2,
                     capsize=5,
                     label=label,
                     color=color,
                     alpha=0.8)

        # Plot 2: Standard deviation trends
        ax2.plot(n_values,
                 stds,
                 marker=marker,
                 markersize=8,
                 linewidth=2,
                 label=label,
                 color=color,
                 alpha=0.8)

        # Plot 3: Min-Max range
        mins = [summary[str(n)]['min'] for n in n_values]
        maxs = [summary[str(n)]['max'] for n in n_values]
        ax3.fill_between(n_values,
                         mins,
                         maxs,
                         alpha=0.3,
                         color=color,
                         label=label)
        ax3.plot(n_values,
                 means,
                 marker=marker,
                 markersize=6,
                 linewidth=2,
                 color=color,
                 alpha=0.8)

    # Plot 4: Direct difference between methods (if we have both)
    if len(all_data) == 2:
        diverse_data = next(
            (d for s, d, n in all_data if s == 'iterative_diverse'), None)
        standard_data = next((d for s, d, n in all_data if s == 'standard'),
                             None)

        if diverse_data and standard_data:
            # Find common n_values
            diverse_summary = diverse_data['summary_stats']
            standard_summary = standard_data['summary_stats']
            common_n = sorted(
                set(diverse_summary.keys()) & set(standard_summary.keys()))
            common_n = [int(n) for n in common_n]

            differences = []
            for n in common_n:
                diverse_mean = diverse_summary[str(n)]['mean']
                standard_mean = standard_summary[str(n)]['mean']
                diff_percent = (
                    (diverse_mean - standard_mean) / standard_mean) * 100
                differences.append(diff_percent)

            ax4.bar(range(len(common_n)),
                    differences,
                    color=['green' if d > 0 else 'red' for d in differences])
            ax4.set_xticks(range(len(common_n)))
            ax4.set_xticklabels([str(n) for n in common_n])
            ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax4.set_xlabel('N (Number of Generations)', fontsize=11)
            ax4.set_ylabel('Diverse vs Standard (%)', fontsize=11)
            ax4.set_title('Relative Performance: Diverse vs Standard',
                          fontsize=12)
            ax4.grid(True, alpha=0.3, axis='y')

            # Add value labels on bars
            for i, (n, diff) in enumerate(zip(common_n, differences)):
                ax4.text(i,
                         diff + (0.5 if diff > 0 else -0.5),
                         f'{diff:.1f}%',
                         ha='center',
                         va='bottom' if diff > 0 else 'top',
                         fontsize=9)

    # Configure subplots
    ax1.set_xlabel('N (Number of Generations)', fontsize=11)
    ax1.set_ylabel('Mean Compression Ratio', fontsize=11)
    ax1.set_title('Mean Compression Ratio Comparison', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=10)

    ax2.set_xlabel('N (Number of Generations)', fontsize=11)
    ax2.set_ylabel('Standard Deviation', fontsize=11)
    ax2.set_title('Variability in Compression Ratios', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=10)

    ax3.set_xlabel('N (Number of Generations)', fontsize=11)
    ax3.set_ylabel('Compression Ratio', fontsize=11)
    ax3.set_title('Range of Compression Ratios (Min-Max)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='best', fontsize=10)

    # Get all N values from all datasets for title
    all_n_vals = set()
    for _, data, n_vals in all_data:
        all_n_vals.update(n_vals)
    n_vals_str = f"N={{{','.join(map(str, sorted(all_n_vals)))}}}" if len(
        all_n_vals) <= 6 else f"N values tested"

    plt.suptitle(
        f'Sampling Method Comparison: Iterative Diverse vs Standard ({n_vals_str})',
        fontsize=14,
        y=1.02)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"📊 Comparison plot saved to: {output_file}")
    else:
        plt.show()


def create_all_plots(json_file: str, output_dir: Optional[str] = None):
    """
    Create all plots from experiment data.
    
    Args:
        json_file: Path to experiment JSON file
        output_dir: Directory to save plots (if None, displays)
    """
    # Load data
    data = load_experiment_data(json_file)

    # Create output directory if specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate output filenames
        base_name = Path(json_file).stem
        mean_plot = output_path / f"{base_name}_mean_ratio.png"
        dist_plot = output_path / f"{base_name}_distribution.png"
        imp_plot = output_path / f"{base_name}_improvement.png"
    else:
        mean_plot = None
        dist_plot = None
        imp_plot = None

    print(f"\n{'='*60}")
    print(f"📊 Creating Plots from: {json_file}")

    # Display sampling method if available
    sampling_method = data.get('parameters', {}).get('sampling_method',
                                                     'unknown')
    if sampling_method != 'unknown':
        sampling_label = 'Iterative Diverse Sampling' if sampling_method == 'iterative_diverse' else 'Standard Sampling'
        print(f"🎯 Sampling Method: {sampling_label}")

    print(f"{'='*60}")

    # Create plots
    print("\n1. Mean Ratio vs N...")
    plot_mean_ratio_vs_n(data, mean_plot, show_individual=True)

    print("\n2. Ratio Distribution...")
    plot_ratio_distribution(data, dist_plot)

    print("\n3. Improvement Curve...")
    plot_improvement_curve(data, imp_plot)

    print(f"\n{'='*60}")
    print(f"✅ All plots created successfully!")
    print(f"{'='*60}\n")


def parse_args():
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot best-of-N compression experiment results")

    parser.add_argument(
        "json_file",
        type=str,
        nargs='+',
        help=
        "Path(s) to experiment JSON file(s). Multiple files will create comparison plots."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save plots (if not specified, displays plots)")
    parser.add_argument(
        "--plot-type",
        type=str,
        choices=[
            'mean', 'distribution', 'improvement', 'all', 'comparison',
            'simple'
        ],
        default='all',
        help=
        "Type of plot to create (default: all). 'simple' creates Y=ratio, X=N plot"
    )
    parser.add_argument(
        "--no-individual",
        action="store_true",
        help="Don't show individual prompt trajectories in mean plot")

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Auto-detect latest experiment files if none specified
    if len(args.json_file) == 1 and args.json_file[0] == 'latest':
        print("🔍 Auto-detecting latest experiment files...")

        # Look for the most recent experiment files
        results_dir = Path('results/vary_n_experiment')
        if not results_dir.exists():
            print(f"❌ Error: Results directory not found: {results_dir}")
            return

        json_files = sorted(results_dir.glob('vary_n_experiment_*.json'),
                            key=lambda x: x.stat().st_mtime,
                            reverse=True)

        if len(json_files) == 0:
            print(f"❌ Error: No JSON files found in {results_dir}")
            return

        # Group files by sampling method
        files_by_method = {'iterative_diverse': [], 'standard': []}
        for file in json_files:
            try:
                data = load_experiment_data(file)
                method = data.get('parameters',
                                  {}).get('sampling_method', 'unknown')
                if method in files_by_method:
                    files_by_method[method].append(file)
            except:
                continue

        # Get the most recent file for each method
        comparison_files = []
        if files_by_method['iterative_diverse']:
            comparison_files.append(
                (files_by_method['iterative_diverse'][0], 'iterative_diverse'))
        if files_by_method['standard']:
            comparison_files.append(
                (files_by_method['standard'][0], 'standard'))

        if len(comparison_files) == 0:
            print(f"❌ Error: No valid experiment files found")
            return
        elif len(comparison_files) == 1:
            print(
                f"⚠️  Found only one sampling method: {comparison_files[0][1]}"
            )
            print(f"   File: {comparison_files[0][0].name}")
            if args.plot_type in ['comparison', 'simple']:
                print(
                    f"   For comparison plots, run the experiment with the other sampling method"
                )

        args.json_file = [str(f[0]) for f in comparison_files]

        print(f"📊 Found latest experiment files:")
        for file, method in comparison_files:
            print(f"  • {method}: {file.name}")
        print()

    # Handle multiple files for comparison or simple plot
    if len(args.json_file) > 1 or args.plot_type in ['comparison', 'simple']:
        # Check all files exist
        for json_file in args.json_file:
            if not Path(json_file).exists():
                print(f"❌ Error: File not found: {json_file}")
                return

        if args.output_dir:
            output_path = Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = None

        if args.plot_type == 'simple':
            # Create simple Y=ratio, X=N plot
            output_file = output_path / "simple_ratio_vs_n.png" if output_path else None
            plot_simple_ratio_vs_n(args.json_file, output_file)
        else:
            # Create comparison plot
            output_file = output_path / "comparison_plot.png" if output_path else None
            plot_comparison(args.json_file, output_file)

            # Also create individual plots for each file
            for json_file in args.json_file:
                print(
                    f"\n📊 Creating individual plots for: {Path(json_file).name}"
                )
                create_all_plots(json_file, args.output_dir)

        return

    # Single file processing
    json_file = args.json_file[0]

    # Check if file exists
    if not Path(json_file).exists():
        print(f"❌ Error: File not found: {json_file}")
        return

    if args.plot_type == 'all':
        create_all_plots(json_file, args.output_dir)
    else:
        # Load data
        data = load_experiment_data(json_file)

        # Create specific plot
        if args.output_dir:
            output_path = Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            base_name = Path(json_file).stem
            output_file = output_path / f"{base_name}_{args.plot_type}.png"
        else:
            output_file = None

        if args.plot_type == 'mean':
            plot_mean_ratio_vs_n(data,
                                 output_file,
                                 show_individual=not args.no_individual)
        elif args.plot_type == 'distribution':
            plot_ratio_distribution(data, output_file)
        elif args.plot_type == 'improvement':
            plot_improvement_curve(data, output_file)


if __name__ == "__main__":
    main()
