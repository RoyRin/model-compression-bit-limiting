#!/usr/bin/env python3
"""
Plot Batch Q&A Sweep results for single-dataset experiments (GSM8K, GPQA-Freeform).

Usage:
    python plot_single_dataset_sweep.py --dataset gsm8k
    python plot_single_dataset_sweep.py --dataset gpqa_freeform
    python plot_single_dataset_sweep.py --all
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from datetime import datetime
from typing import Dict, List

# Model configurations
SLM_MODELS = ['haiku', 'sonnet', 'opus']
LLM_MODELS = ['haiku', 'sonnet', 'opus', 'opus_oracle']
Q_MODELS = ['haiku', 'sonnet', 'opus']


def load_dataset_results(results_dir: Path, dataset: str) -> Dict:
    """Load results for a single dataset.

    Returns:
        Dict mapping (slm, llm, q_model) -> result dict
    """
    results = {}

    pattern = f"{dataset}_*.json"
    json_files = list(results_dir.glob(pattern))
    json_files = [f for f in json_files if 'summary' not in f.name]

    print(f"Found {len(json_files)} result files for {dataset}")

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Parse filename: {dataset}_SLM-{slm}_LLM-{llm}_Q-{q}.json
            name = json_file.stem
            parts = name.split('_')

            slm = llm = q_model = None
            for i, part in enumerate(parts):
                if part.startswith('SLM-'):
                    slm = part.replace('SLM-', '')
                elif part.startswith('LLM-'):
                    llm = part.replace('LLM-', '')
                    # Check if next part is 'oracle'
                    if i + 1 < len(parts) and parts[i + 1] == 'oracle':
                        llm = llm + '_oracle'
                elif part.startswith('Q-'):
                    q_model = part.replace('Q-', '')

            if slm and llm and q_model:
                summary = data.get('summary', {})
                key = (slm, llm, q_model)
                results[key] = {
                    'initial_accuracy': summary.get('initial_accuracy', 0),
                    'final_accuracy': summary.get('final_accuracy', 0),
                    'total': summary.get('total', 0),
                    'recovered': summary.get('recovered', 0),
                    'recovery_rate': summary.get('recovery_rate', 0),
                }
        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    return results


def build_matrix(results: Dict,
                 metric: str = 'final_accuracy') -> pd.DataFrame:
    """Build 12x3 matrix.

    Rows: 12 SLM/LLM combinations (3 SLM x 4 LLM)
    Cols: 3 Question models
    """
    slm_llm_pairs = [(slm, llm) for slm in SLM_MODELS for llm in LLM_MODELS]

    matrix = np.zeros((12, 3))

    for i, (slm, llm) in enumerate(slm_llm_pairs):
        for j, q_model in enumerate(Q_MODELS):
            key = (slm, llm, q_model)
            if key in results:
                if metric == 'improvement':
                    val = results[key]['final_accuracy'] - results[key][
                        'initial_accuracy']
                else:
                    val = results[key].get(metric, 0)
                matrix[i, j] = val * 100  # Convert to percentage
            else:
                matrix[i, j] = np.nan

    row_labels = [
        f"{slm}/{llm.replace('_oracle', '†')}" for slm, llm in slm_llm_pairs
    ]
    col_labels = [f"Q={q}" for q in Q_MODELS]

    return pd.DataFrame(matrix, index=row_labels, columns=col_labels)


def plot_heatmap(df: pd.DataFrame,
                 title: str,
                 ax,
                 metric: str,
                 show_cbar: bool = True,
                 show_ylabel: bool = True):
    """Plot a single heatmap."""

    if metric == 'improvement':
        cmap = 'RdYlGn'
        vmin, vmax = -10, 30
        fmt = '+.1f'
    else:
        cmap = 'RdYlGn'
        vmin, vmax = 0, 100
        fmt = '.1f'

    sns.heatmap(df,
                annot=True,
                fmt=fmt,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                cbar=show_cbar,
                cbar_kws={'label': f'{metric.replace("_", " ").title()} (%)'}
                if show_cbar else None,
                ax=ax,
                square=False,
                linewidths=0.5,
                linecolor='white')

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Question Model', fontsize=10)

    if show_ylabel:
        ax.set_ylabel('SLM / LLM  († = oracle)', fontsize=10)
    else:
        ax.set_ylabel('')

    ax.set_xticklabels(ax.get_xticklabels(),
                       rotation=0,
                       ha='center',
                       fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(),
                       rotation=0,
                       va='center',
                       fontsize=9)


def plot_dataset(results: Dict, dataset: str, output_path: str):
    """Create comparison plot (initial vs final) for a single dataset."""

    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    # Get problem count
    n = list(results.values())[0]['total'] if results else '?'

    # Initial accuracy
    df_init = build_matrix(results, 'initial_accuracy')
    plot_heatmap(df_init,
                 f'Initial Accuracy (n={n})',
                 axes[0],
                 'initial_accuracy',
                 show_cbar=False,
                 show_ylabel=True)

    # Final accuracy
    df_final = build_matrix(results, 'final_accuracy')
    plot_heatmap(df_final,
                 'Final Accuracy (after Q&A)',
                 axes[1],
                 'final_accuracy',
                 show_cbar=False,
                 show_ylabel=False)

    # Improvement
    df_imp = build_matrix(results, 'improvement')
    plot_heatmap(df_imp,
                 'Improvement (Final - Initial)',
                 axes[2],
                 'improvement',
                 show_cbar=True,
                 show_ylabel=False)

    dataset_name = dataset.replace('_', ' ').upper()
    fig.suptitle(
        f'{dataset_name} Batch Q&A Compression: Initial vs Final Accuracy\n(10 questions)',
        fontsize=14,
        fontweight='bold',
        y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def print_summary(results: Dict, dataset: str):
    """Print text summary of best configurations."""

    print(f"\n{'='*60}")
    print(f"{dataset.upper()} SUMMARY")
    print('=' * 60)

    df_init = build_matrix(results, 'initial_accuracy')
    df_final = build_matrix(results, 'final_accuracy')

    # Best final accuracy
    stacked = df_final.stack().dropna()
    if len(stacked) > 0:
        best_val = stacked.max()
        best_idx = stacked.idxmax()
        init_val = df_init.loc[best_idx[0], best_idx[1]]
        print(
            f"Best final: {best_idx[0]} + {best_idx[1]} = {best_val:.1f}% (was {init_val:.1f}%)"
        )

    # Best improvement
    df_imp = df_final - df_init
    stacked = df_imp.stack().dropna()
    if len(stacked) > 0:
        best_imp = stacked.max()
        best_idx = stacked.idxmax()
        print(
            f"Best improvement: {best_idx[0]} + {best_idx[1]} = +{best_imp:.1f}%"
        )

    # Average values
    print(f"\nOverall stats:")
    print(f"  Initial accuracy: {df_init.mean().mean():.1f}%")
    print(f"  Final accuracy: {df_final.mean().mean():.1f}%")
    print(f"  Average improvement: {df_imp.mean().mean():.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description='Plot single-dataset sweep results')
    parser.add_argument(
        '--results-dir',
        type=str,
        default=None,
        help=
        'Results directory containing JSON files (default: results/<dataset>_qa_sweep/data)'
    )
    parser.add_argument('--dataset',
                        type=str,
                        default=None,
                        choices=['gsm8k', 'gpqa_freeform', 'mbpp'],
                        help='Dataset to plot')
    parser.add_argument('--all', action='store_true', help='Plot all datasets')
    parser.add_argument('--summary',
                        action='store_true',
                        help='Print text summary')
    parser.add_argument('--output-dir',
                        type=str,
                        default=None,
                        help='Output directory for plots')

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    datasets = ['gsm8k', 'gpqa_freeform', 'mbpp'
                ] if args.all else [args.dataset] if args.dataset else []

    if not datasets:
        print("Please specify --dataset or --all")
        return

    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Processing {dataset}")
        print('=' * 60)

        # Determine results directory
        if args.results_dir:
            results_dir = Path(args.results_dir)
        else:
            results_dir = Path(f'results/{dataset}_qa_sweep/data')

        if not results_dir.exists():
            print(f"Error: {results_dir} not found")
            continue

        results = load_dataset_results(results_dir, dataset)
        if not results:
            print(f"No results found for {dataset}!")
            continue

        print(f"Loaded {len(results)} configurations")

        if args.summary:
            print_summary(results, dataset)

        # Determine output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = Path(f'results/{dataset}_qa_sweep/plots')
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"{dataset}_qa_sweep_{timestamp}.pdf"
        plot_dataset(results, dataset, str(output_path))

    print("\nDone!")


if __name__ == "__main__":
    main()
