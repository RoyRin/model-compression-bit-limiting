#!/usr/bin/env python3
"""Plot compression results from YAML output."""

import argparse
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
from datetime import datetime


def load_results(yaml_path: Path) -> Dict:
    """Load compression results from YAML file."""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data


def plot_time_vs_tokens(results: List[Dict], output_dir: Path, timestamp: str):
    """Plot encoding and decoding time vs number of generated tokens."""
    # Filter successful results only
    successful = [r for r in results if r['success']]

    if not successful:
        print("No successful compressions to plot!")
        return

    # Extract data
    num_tokens = [r['num_generated_tokens'] for r in successful]
    encode_time_per_tok = [
        r['encoding_time_per_token'] * 1000 for r in successful
    ]  # Convert to ms
    decode_time_per_tok = [
        r['decoding_time_per_token'] * 1000 for r in successful
    ]  # Convert to ms

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Encoding time vs tokens
    ax1.scatter(num_tokens, encode_time_per_tok, alpha=0.6, s=30, color='blue')
    ax1.set_xlabel('Number of Generated Tokens', fontsize=12)
    ax1.set_ylabel('Encoding Time (ms/token)', fontsize=12)
    ax1.set_title('Encoding Time vs Token Count',
                  fontsize=14,
                  fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Add trend line
    z = np.polyfit(num_tokens, encode_time_per_tok, 1)
    p = np.poly1d(z)
    ax1.plot(num_tokens,
             p(num_tokens),
             "r--",
             alpha=0.8,
             linewidth=2,
             label=f'Trend: y={z[0]:.4f}x+{z[1]:.2f}')
    ax1.legend()

    # Decoding time vs tokens
    ax2.scatter(num_tokens,
                decode_time_per_tok,
                alpha=0.6,
                s=30,
                color='green')
    ax2.set_xlabel('Number of Generated Tokens', fontsize=12)
    ax2.set_ylabel('Decoding Time (ms/token)', fontsize=12)
    ax2.set_title('Decoding Time vs Token Count',
                  fontsize=14,
                  fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Add trend line
    z = np.polyfit(num_tokens, decode_time_per_tok, 1)
    p = np.poly1d(z)
    ax2.plot(num_tokens,
             p(num_tokens),
             "r--",
             alpha=0.8,
             linewidth=2,
             label=f'Trend: y={z[0]:.4f}x+{z[1]:.2f}')
    ax2.legend()

    plt.tight_layout()

    # Save plot
    output_path = output_dir / f'time_vs_tokens_{timestamp}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved time vs tokens plot to {output_path}")
    plt.close()


def plot_compression_ratio_histogram(results: List[Dict], output_dir: Path,
                                     timestamp: str):
    """Plot histogram of compression ratios."""
    # Filter successful results only
    successful = [
        r for r in results
        if r['success'] and r['compression_ratio'] is not None
    ]

    if not successful:
        print("No successful compressions to plot!")
        return

    # Extract compression ratios
    compression_ratios = [r['compression_ratio'] for r in successful]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot histogram
    n, bins, patches = ax.hist(compression_ratios,
                               bins=30,
                               alpha=0.7,
                               color='purple',
                               edgecolor='black')

    # Add mean line
    mean_ratio = np.mean(compression_ratios)
    ax.axvline(mean_ratio,
               color='red',
               linestyle='--',
               linewidth=2,
               label=f'Mean: {mean_ratio:.2f}x')

    # Add median line
    median_ratio = np.median(compression_ratios)
    ax.axvline(median_ratio,
               color='orange',
               linestyle='--',
               linewidth=2,
               label=f'Median: {median_ratio:.2f}x')

    ax.set_xlabel('Compression Ratio (x)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Compression Ratios',
                 fontsize=14,
                 fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # Add statistics text box
    stats_text = f'n = {len(compression_ratios)}\nMean: {mean_ratio:.2f}x\nMedian: {median_ratio:.2f}x\nStd: {np.std(compression_ratios):.2f}'
    ax.text(0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save plot
    output_path = output_dir / f'compression_ratio_histogram_{timestamp}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved compression ratio histogram to {output_path}")
    plt.close()


def plot_bits_per_token_histogram(results: List[Dict], output_dir: Path,
                                  timestamp: str):
    """Plot histogram of bits per token."""
    # Filter successful results only
    successful = [
        r for r in results if r['success'] and r['bits_per_token'] is not None
    ]

    if not successful:
        print("No successful compressions to plot!")
        return

    # Extract bits per token
    bits_per_token = [r['bits_per_token'] for r in successful]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot histogram
    n, bins, patches = ax.hist(bits_per_token,
                               bins=30,
                               alpha=0.7,
                               color='teal',
                               edgecolor='black')

    # Add mean line
    mean_bpt = np.mean(bits_per_token)
    ax.axvline(mean_bpt,
               color='red',
               linestyle='--',
               linewidth=2,
               label=f'Mean: {mean_bpt:.2f} bpt')

    # Add median line
    median_bpt = np.median(bits_per_token)
    ax.axvline(median_bpt,
               color='orange',
               linestyle='--',
               linewidth=2,
               label=f'Median: {median_bpt:.2f} bpt')

    ax.set_xlabel('Bits per Token', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Bits per Token',
                 fontsize=14,
                 fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # Add statistics text box
    stats_text = f'n = {len(bits_per_token)}\nMean: {mean_bpt:.2f} bpt\nMedian: {median_bpt:.2f} bpt\nStd: {np.std(bits_per_token):.2f}'
    ax.text(0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save plot
    output_path = output_dir / f'bits_per_token_histogram_{timestamp}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved bits per token histogram to {output_path}")
    plt.close()


def print_summary_stats(results: List[Dict]):
    """Print summary statistics."""
    successful = [r for r in results if r['success']]

    if not successful:
        print("\nNo successful compressions!")
        return

    print(f"\n{'='*80}")
    print(f"SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"Total samples: {len(results)}")
    print(
        f"Successful: {len(successful)} ({len(successful)/len(results)*100:.1f}%)"
    )

    # Compression ratio stats
    compression_ratios = [
        r['compression_ratio'] for r in successful
        if r['compression_ratio'] is not None
    ]
    if compression_ratios:
        print(f"\nCompression Ratio:")
        print(f"  Mean: {np.mean(compression_ratios):.2f}x")
        print(f"  Median: {np.median(compression_ratios):.2f}x")
        print(f"  Min: {np.min(compression_ratios):.2f}x")
        print(f"  Max: {np.max(compression_ratios):.2f}x")
        print(f"  Std: {np.std(compression_ratios):.2f}")

    # Bits per token stats
    bits_per_token = [
        r['bits_per_token'] for r in successful
        if r['bits_per_token'] is not None
    ]
    if bits_per_token:
        print(f"\nBits per Token:")
        print(f"  Mean: {np.mean(bits_per_token):.2f}")
        print(f"  Median: {np.median(bits_per_token):.2f}")
        print(f"  Min: {np.min(bits_per_token):.2f}")
        print(f"  Max: {np.max(bits_per_token):.2f}")
        print(f"  Std: {np.std(bits_per_token):.2f}")

    # Timing stats
    encode_times = [r['encoding_time_per_token'] * 1000 for r in successful]
    decode_times = [r['decoding_time_per_token'] * 1000 for r in successful]

    print(f"\nEncoding Time (ms/token):")
    print(f"  Mean: {np.mean(encode_times):.2f}")
    print(f"  Median: {np.median(encode_times):.2f}")
    print(f"  Min: {np.min(encode_times):.2f}")
    print(f"  Max: {np.max(encode_times):.2f}")

    print(f"\nDecoding Time (ms/token):")
    print(f"  Mean: {np.mean(decode_times):.2f}")
    print(f"  Median: {np.median(decode_times):.2f}")
    print(f"  Min: {np.min(decode_times):.2f}")
    print(f"  Max: {np.max(decode_times):.2f}")

    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Plot compression results from YAML file")
    parser.add_argument("results_file",
                        type=Path,
                        help="Path to compression results YAML file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for plots (default: data/results/plots/)")

    args = parser.parse_args()

    # Set default output directory if not specified
    if args.output_dir is None:
        base_dir = Path(__file__).parent.parent
        args.output_dir = base_dir / "data" / "results" / "plots"

    # Load results
    print(f"Loading results from {args.results_file}...")
    data = load_results(args.results_file)

    results = data['results']
    metadata = data['metadata']

    print(f"Loaded {len(results)} results")
    print(f"Model: {metadata.get('model', 'unknown')}")
    print(f"Dataset: {metadata.get('dataset', 'unknown')}")

    # Create output directory
    args.output_dir.mkdir(exist_ok=True, parents=True)

    # Print summary statistics
    print_summary_stats(results)

    # Generate timestamp for plot filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate plots
    print(f"\nGenerating plots...")
    plot_time_vs_tokens(results, args.output_dir, timestamp)
    plot_compression_ratio_histogram(results, args.output_dir, timestamp)
    plot_bits_per_token_histogram(results, args.output_dir, timestamp)

    print(f"\n✓ All plots saved to {args.output_dir}/")
    print(f"  Timestamp: {timestamp}")


if __name__ == "__main__":
    main()
