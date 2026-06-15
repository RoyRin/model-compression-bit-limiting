#!/usr/bin/env python3
"""
Plot Batch Q&A Sweep results as heatmaps.

Shows initial accuracy, final accuracy, and improvement for all model combinations.

Usage:
    python plot_batch_qa_sweep.py
    python plot_batch_qa_sweep.py --results-dir results/batch_qa_sweep
    python plot_batch_qa_sweep.py --metric final  # or initial, improvement
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from datetime import datetime
from typing import Dict, Optional

# Model configurations
SLM_MODELS = ['haiku', 'sonnet', 'opus']
LLM_MODELS = ['haiku', 'sonnet', 'opus', 'opus_oracle']
Q_MODELS = ['haiku', 'sonnet', 'opus']
SUBJECTS = ['algebra', 'geometry', 'number_theory']


def load_batch_results(results_dir: Path) -> Dict:
    """Load all batch sweep results.

    Returns:
        Dict mapping (subject, slm, llm, q_model) -> result dict
    """
    results = {}

    json_files = list(results_dir.glob("*.json"))
    json_files = [f for f in json_files if f.name != 'sweep_summary.json']

    print(f"Found {len(json_files)} result files in {results_dir}")

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Parse filename: {subject}_SLM-{slm}_LLM-{llm}_Q-{q}.json
            name = json_file.stem
            parts = name.split('_')

            subject = parts[0]
            if subject == 'number':
                subject = 'number_theory'

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
                key = (subject, slm, llm, q_model)
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
                 subject: str,
                 metric: str = 'final_accuracy') -> pd.DataFrame:
    """Build 12x3 matrix for a subject.

    Rows: 12 SLM/LLM combinations (3 SLM × 4 LLM)
    Cols: 3 Question models
    """
    slm_llm_pairs = [(slm, llm) for slm in SLM_MODELS for llm in LLM_MODELS]

    matrix = np.zeros((12, 3))

    for i, (slm, llm) in enumerate(slm_llm_pairs):
        for j, q_model in enumerate(Q_MODELS):
            key = (subject, slm, llm, q_model)
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


def build_combined_matrix(results: Dict,
                          metric: str = 'final_accuracy') -> pd.DataFrame:
    """Build combined matrix averaging across subjects."""
    slm_llm_pairs = [(slm, llm) for slm in SLM_MODELS for llm in LLM_MODELS]

    matrix = np.zeros((12, 3))
    counts = np.zeros((12, 3))

    for subject in SUBJECTS:
        for i, (slm, llm) in enumerate(slm_llm_pairs):
            for j, q_model in enumerate(Q_MODELS):
                key = (subject, slm, llm, q_model)
                if key in results:
                    if metric == 'improvement':
                        val = results[key]['final_accuracy'] - results[key][
                            'initial_accuracy']
                    else:
                        val = results[key].get(metric, 0)
                    matrix[i, j] += val * 100
                    counts[i, j] += 1

    with np.errstate(invalid='ignore'):
        matrix = np.where(counts > 0, matrix / counts, np.nan)

    row_labels = [
        f"{slm}/{llm.replace('_oracle', '†')}" for slm, llm in slm_llm_pairs
    ]
    col_labels = [f"Q={q}" for q in Q_MODELS]

    return pd.DataFrame(matrix, index=row_labels, columns=col_labels)


def plot_heatmap(df: pd.DataFrame,
                 title: str,
                 ax,
                 metric: str,
                 show_cbar: bool = False,
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

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Question Model', fontsize=9)

    if show_ylabel:
        ax.set_ylabel('SLM / LLM  († = oracle)', fontsize=9)
    else:
        ax.set_ylabel('')

    ax.set_xticklabels(ax.get_xticklabels(),
                       rotation=0,
                       ha='center',
                       fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(),
                       rotation=0,
                       va='center',
                       fontsize=8)


def plot_all_subjects(results: Dict, metric: str, output_path: str):
    """Create figure with heatmaps for all subjects + combined."""

    fig, axes = plt.subplots(1, 4, figsize=(22, 10))

    # Get problem counts
    subject_counts = {}
    for subject in SUBJECTS:
        for key, val in results.items():
            if key[0] == subject:
                subject_counts[subject] = val['total']
                break

    # Plot each subject
    for idx, subject in enumerate(SUBJECTS):
        df = build_matrix(results, subject, metric)
        n = subject_counts.get(subject, '?')
        title = f"{subject.replace('_', ' ').title()}\n(n={n})"
        plot_heatmap(df,
                     title,
                     axes[idx],
                     metric,
                     show_cbar=False,
                     show_ylabel=(idx == 0))

    # Plot combined
    combined_df = build_combined_matrix(results, metric)
    total = sum(subject_counts.values())
    plot_heatmap(combined_df,
                 f"Combined\n(n={total})",
                 axes[3],
                 metric,
                 show_cbar=True,
                 show_ylabel=False)

    metric_label = metric.replace('_', ' ').title()
    fig.suptitle(
        f'MATH Batch Q&A Compression: {metric_label} by Model Configuration\n(10 questions, non-easy problems)',
        fontsize=13,
        fontweight='bold',
        y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_comparison(results: Dict, output_path: str):
    """Create side-by-side comparison of initial vs final accuracy."""

    fig, axes = plt.subplots(2, 4, figsize=(22, 16))

    subject_counts = {}
    for subject in SUBJECTS:
        for key, val in results.items():
            if key[0] == subject:
                subject_counts[subject] = val['total']
                break

    # Top row: Initial accuracy
    for idx, subject in enumerate(SUBJECTS):
        df = build_matrix(results, subject, 'initial_accuracy')
        n = subject_counts.get(subject, '?')
        title = f"{subject.replace('_', ' ').title()} - Initial\n(n={n})"
        plot_heatmap(df,
                     title,
                     axes[0, idx],
                     'initial_accuracy',
                     show_cbar=False,
                     show_ylabel=(idx == 0))

    combined_df = build_combined_matrix(results, 'initial_accuracy')
    plot_heatmap(combined_df,
                 "Combined - Initial",
                 axes[0, 3],
                 'initial_accuracy',
                 show_cbar=True,
                 show_ylabel=False)

    # Bottom row: Final accuracy
    for idx, subject in enumerate(SUBJECTS):
        df = build_matrix(results, subject, 'final_accuracy')
        title = f"{subject.replace('_', ' ').title()} - Final"
        plot_heatmap(df,
                     title,
                     axes[1, idx],
                     'final_accuracy',
                     show_cbar=False,
                     show_ylabel=(idx == 0))

    combined_df = build_combined_matrix(results, 'final_accuracy')
    total = sum(subject_counts.values())
    plot_heatmap(combined_df,
                 f"Combined - Final\n(n={total})",
                 axes[1, 3],
                 'final_accuracy',
                 show_cbar=True,
                 show_ylabel=False)

    fig.suptitle(
        'MATH Batch Q&A Compression: Initial vs Final Accuracy\n(10 questions, non-easy problems)',
        fontsize=14,
        fontweight='bold',
        y=1.01)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def print_summary(results: Dict):
    """Print text summary of best configurations."""

    print("\n" + "=" * 70)
    print("BATCH Q&A SWEEP SUMMARY")
    print("=" * 70)

    for subject in SUBJECTS + ['combined']:
        if subject == 'combined':
            df_init = build_combined_matrix(results, 'initial_accuracy')
            df_final = build_combined_matrix(results, 'final_accuracy')
            label = "COMBINED"
        else:
            df_init = build_matrix(results, subject, 'initial_accuracy')
            df_final = build_matrix(results, subject, 'final_accuracy')
            label = subject.upper()

        print(f"\n{label}")
        print("-" * 50)

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


def plot_single_subject(results: Dict, subject: str, output_path: str):
    """Create figure with all 3 metrics for a single subject."""

    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    # Get problem count
    n = '?'
    for key, val in results.items():
        if key[0] == subject:
            n = val['total']
            break

    subject_label = subject.replace('_', ' ').title()

    # Initial accuracy
    df_init = build_matrix(results, subject, 'initial_accuracy')
    plot_heatmap(df_init,
                 f'Initial Accuracy (n={n})',
                 axes[0],
                 'initial_accuracy',
                 show_cbar=False,
                 show_ylabel=True)

    # Final accuracy
    df_final = build_matrix(results, subject, 'final_accuracy')
    plot_heatmap(df_final,
                 'Final Accuracy (after Q&A)',
                 axes[1],
                 'final_accuracy',
                 show_cbar=False,
                 show_ylabel=False)

    # Improvement
    df_imp = build_matrix(results, subject, 'improvement')
    plot_heatmap(df_imp,
                 'Improvement (Final - Initial)',
                 axes[2],
                 'improvement',
                 show_cbar=True,
                 show_ylabel=False)

    fig.suptitle(
        f'MATH {subject_label}: Batch Q&A Compression Results\n(10 questions, non-easy problems)',
        fontsize=14,
        fontweight='bold',
        y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Plot Batch Q&A Sweep results')
    parser.add_argument('--results-dir',
                        type=str,
                        default='results/math_qa_sweep/data',
                        help='Results directory containing JSON files')
    parser.add_argument(
        '--metric',
        type=str,
        default='final_accuracy',
        choices=['initial_accuracy', 'final_accuracy', 'improvement'],
        help='Metric to plot')
    parser.add_argument('--comparison',
                        action='store_true',
                        help='Plot initial vs final comparison')
    parser.add_argument('--individual',
                        action='store_true',
                        help='Plot individual PDFs for each MATH subject')
    parser.add_argument('--summary',
                        action='store_true',
                        help='Print text summary')
    parser.add_argument('--output-dir',
                        type=str,
                        default='results/math_qa_sweep/plots',
                        help='Output directory for plots')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: {results_dir} not found")
        return

    results = load_batch_results(results_dir)
    if not results:
        print("No results found!")
        return

    print(f"Loaded {len(results)} configurations")

    if args.summary:
        print_summary(results)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.individual:
        # Generate individual PDFs for each subject
        for subject in SUBJECTS:
            output_path = output_dir / f"math_{subject}_qa_sweep_{timestamp}.pdf"
            plot_single_subject(results, subject, str(output_path))

    if args.comparison:
        output_path = output_dir / f"batch_qa_sweep_comparison_{timestamp}.pdf"
        plot_comparison(results, str(output_path))
    elif not args.individual:
        output_path = output_dir / f"batch_qa_sweep_{args.metric}_{timestamp}.pdf"
        plot_all_subjects(results, args.metric, str(output_path))

    print("\nDone!")


if __name__ == "__main__":
    main()
