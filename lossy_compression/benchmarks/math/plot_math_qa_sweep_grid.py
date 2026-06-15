#!/usr/bin/env python3
"""
Plot MATH Q&A sweep results in a 12x3 grid showing accuracy for different model combinations.
Creates heatmaps for each subject (algebra, geometry, number_theory) and combined.

Usage:
    python plot_math_qa_sweep_grid.py results/math_qa_sweep_20260116_000000/
    python plot_math_qa_sweep_grid.py results/math_qa_sweep_20260116_000000/ --subject algebra
    python plot_math_qa_sweep_grid.py results/math_qa_sweep_20260116_000000/ --show
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime

# Model configurations
SLM_MODELS = ['haiku', 'sonnet', 'opus']
LLM_MODELS = ['haiku', 'sonnet', 'opus',
              'opus-oracle']  # 4 options including oracle
Q_MODELS = ['haiku', 'sonnet', 'opus']

SUBJECTS = ['algebra', 'geometry', 'number_theory']


def load_sweep_results(results_dir: Path,
                       subject: Optional[str] = None) -> Dict:
    """Load all sweep results from a directory.

    Returns:
        Dict mapping (subject, slm, llm, q_model) -> result dict
    """
    results = {}

    # Find all JSON files
    pattern = f"math_qa_{subject}_*.json" if subject else "math_qa_*.json"
    json_files = list(results_dir.glob(pattern))

    print(f"Found {len(json_files)} result files in {results_dir}")

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Parse filename to get config
            # Format: math_qa_{subject}_SLM-{slm}_LLM-{llm}_Q-{q}.json
            name = json_file.stem
            parts = name.split('_')

            # Extract subject (after "math_qa_")
            subj = parts[2]

            # Extract model configs from filename
            slm = llm = q_model = None
            for part in parts:
                if part.startswith('SLM-'):
                    slm = part.replace('SLM-', '')
                elif part.startswith('LLM-'):
                    llm = part.replace('LLM-', '')
                elif part.startswith('Q-'):
                    q_model = part.replace('Q-', '')

            if slm and llm and q_model:
                key = (subj, slm, llm, q_model)
                results[key] = {
                    'accuracy': data.get('accuracy', 0),
                    'correct': data.get('correct_count', 0),
                    'total': data.get('total_problems', 0),
                    'avg_questions': data.get('avg_questions', 0),
                    'avg_time': data.get('avg_solve_time', 0),
                }
        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    return results


def calculate_accuracy_matrix(results: Dict, subject: str) -> pd.DataFrame:
    """Calculate accuracy matrix for a single subject.

    Returns a 12x3 DataFrame:
    - Rows: 12 SLM/LLM combinations (3 SLM × 4 LLM)
    - Columns: 3 Question models
    """
    # Create row labels (SLM/LLM pairs)
    slm_llm_pairs = []
    for slm in SLM_MODELS:
        for llm in LLM_MODELS:
            slm_llm_pairs.append((slm, llm))

    # Initialize matrix (12 rows × 3 cols)
    matrix = np.zeros((12, 3))

    for i, (slm, llm) in enumerate(slm_llm_pairs):
        for j, q_model in enumerate(Q_MODELS):
            key = (subject, slm, llm, q_model)
            if key in results:
                matrix[i, j] = results[key][
                    'accuracy'] * 100  # Convert to percentage
            else:
                matrix[i, j] = np.nan  # Missing data

    # Create DataFrame with proper labels
    row_labels = [f"{slm} / {llm}" for slm, llm in slm_llm_pairs]
    col_labels = [f"Q: {model}" for model in Q_MODELS]

    df = pd.DataFrame(matrix, index=row_labels, columns=col_labels)
    return df


def calculate_combined_matrix(results: Dict) -> pd.DataFrame:
    """Calculate combined accuracy matrix across all subjects.

    Averages accuracy across algebra, geometry, and number_theory.
    """
    slm_llm_pairs = [(slm, llm) for slm in SLM_MODELS for llm in LLM_MODELS]

    matrix = np.zeros((12, 3))
    counts = np.zeros((12, 3))

    for subject in SUBJECTS:
        for i, (slm, llm) in enumerate(slm_llm_pairs):
            for j, q_model in enumerate(Q_MODELS):
                key = (subject, slm, llm, q_model)
                if key in results:
                    matrix[i, j] += results[key]['accuracy'] * 100
                    counts[i, j] += 1

    # Average where we have data
    with np.errstate(invalid='ignore'):
        matrix = np.where(counts > 0, matrix / counts, np.nan)

    row_labels = [f"{slm} / {llm}" for slm, llm in slm_llm_pairs]
    col_labels = [f"Q: {model}" for model in Q_MODELS]

    return pd.DataFrame(matrix, index=row_labels, columns=col_labels)


def plot_single_heatmap(df: pd.DataFrame,
                        title: str,
                        ax,
                        show_cbar: bool = False,
                        show_ylabel: bool = True):
    """Plot a single heatmap on the given axis."""

    # Color map - RdYlGn for red (bad) to green (good)
    cmap = 'RdYlGn'

    sns.heatmap(df,
                annot=True,
                fmt='.1f',
                cmap=cmap,
                vmin=0,
                vmax=100,
                cbar=show_cbar,
                cbar_kws={'label': 'Accuracy (%)'} if show_cbar else None,
                ax=ax,
                square=False,
                linewidths=0.5,
                linecolor='white')

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Question Model', fontsize=10, fontweight='bold')

    if show_ylabel:
        ax.set_ylabel('SLM / LLM', fontsize=10, fontweight='bold')
    else:
        ax.set_ylabel('')

    # Rotate labels for readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va='center')


def plot_all_subjects(results: Dict,
                      output_path: Optional[str] = None,
                      show: bool = False):
    """Create a figure with heatmaps for all subjects plus combined."""

    # Create figure with 4 subplots (3 subjects + combined)
    fig, axes = plt.subplots(1, 4, figsize=(24, 10))

    # Get problem counts for each subject
    subject_counts = {}
    for subject in SUBJECTS:
        # Find any result for this subject to get total
        for key, val in results.items():
            if key[0] == subject:
                subject_counts[subject] = val['total']
                break

    # Plot each subject
    for idx, subject in enumerate(SUBJECTS):
        df = calculate_accuracy_matrix(results, subject)
        n_problems = subject_counts.get(subject, '?')
        title = f"{subject.replace('_', ' ').title()}\n(n={n_problems})"
        plot_single_heatmap(df,
                            title,
                            axes[idx],
                            show_cbar=False,
                            show_ylabel=(idx == 0))

    # Plot combined
    combined_df = calculate_combined_matrix(results)
    total_problems = sum(subject_counts.values())
    plot_single_heatmap(combined_df,
                        f"Combined Average\n(n={total_problems})",
                        axes[3],
                        show_cbar=True,
                        show_ylabel=False)

    # Add overall title
    fig.suptitle(
        'MATH Q&A Compression: Accuracy by Model Configuration\n(on not_easy problems where Haiku baseline = 0%)',
        fontsize=14,
        fontweight='bold',
        y=1.02)

    plt.tight_layout()

    # Save and/or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")

    if show:
        plt.show()

    plt.close()


def plot_single_subject(results: Dict,
                        subject: str,
                        output_path: Optional[str] = None,
                        show: bool = False):
    """Create a single heatmap for one subject."""

    fig, ax = plt.subplots(1, 1, figsize=(8, 10))

    df = calculate_accuracy_matrix(results, subject)

    # Get problem count
    n_problems = '?'
    for key, val in results.items():
        if key[0] == subject:
            n_problems = val['total']
            break

    title = f"MATH {subject.replace('_', ' ').title()} Q&A Compression\n(n={n_problems} not_easy problems)"
    plot_single_heatmap(df, title, ax, show_cbar=True, show_ylabel=True)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")

    if show:
        plt.show()

    plt.close()


def print_summary_table(results: Dict):
    """Print a text summary of results."""

    print("\n" + "=" * 80)
    print("MATH Q&A COMPRESSION SWEEP RESULTS")
    print("=" * 80)

    for subject in SUBJECTS:
        # Check if we have any data for this subject
        subject_results = {k: v for k, v in results.items() if k[0] == subject}
        if not subject_results:
            continue

        print(f"\n{subject.upper()}")
        print("-" * 60)

        df = calculate_accuracy_matrix(results, subject)

        # Find best configuration (handle empty/all-nan case)
        try:
            stacked = df.stack().dropna()
            if len(stacked) > 0:
                max_val = stacked.max()
                max_idx = stacked.idxmax()
                print(f"Best: {max_idx[0]} with {max_idx[1]} = {max_val:.1f}%")
        except Exception:
            pass

        print(f"\nFull matrix:")
        print(df.to_string())

    # Combined (only if we have multiple subjects)
    subjects_with_data = set(k[0] for k in results.keys())
    if len(subjects_with_data) > 1:
        print(f"\nCOMBINED AVERAGE")
        print("-" * 60)
        combined_df = calculate_combined_matrix(results)
        try:
            stacked = combined_df.stack().dropna()
            if len(stacked) > 0:
                max_val = stacked.max()
                max_idx = stacked.idxmax()
                print(f"Best: {max_idx[0]} with {max_idx[1]} = {max_val:.1f}%")
        except Exception:
            pass
        print(f"\nFull matrix:")
        print(combined_df.to_string())


def main():
    parser = argparse.ArgumentParser(
        description='Plot MATH Q&A sweep results as heatmaps')
    parser.add_argument('results_dir',
                        type=str,
                        help='Directory containing sweep result JSON files')
    parser.add_argument('--subject',
                        type=str,
                        choices=SUBJECTS,
                        help='Plot only this subject (default: all)')
    parser.add_argument('--output',
                        '-o',
                        type=str,
                        help='Output file path (default: auto-generated)')
    parser.add_argument('--show',
                        action='store_true',
                        help='Display plot interactively')
    parser.add_argument('--summary',
                        action='store_true',
                        help='Print text summary table')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return

    # Load results
    print(f"Loading results from: {results_dir}")
    results = load_sweep_results(results_dir, args.subject)

    if not results:
        print("No results found!")
        return

    print(f"Loaded {len(results)} configurations")

    # Print summary if requested
    if args.summary:
        print_summary_table(results)

    # Generate output path if not specified
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.subject:
            output_path = f"math_qa_sweep_heatmap_{args.subject}_{timestamp}.png"
        else:
            output_path = f"math_qa_sweep_heatmap_all_{timestamp}.png"

    # Plot
    if args.subject:
        plot_single_subject(results, args.subject, output_path, args.show)
    else:
        plot_all_subjects(results, output_path, args.show)

    print("\nDone!")


if __name__ == "__main__":
    main()
