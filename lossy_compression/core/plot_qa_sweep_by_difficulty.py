#!/usr/bin/env python3
"""
Plot QA Sweep results by difficulty level.

Shows final accuracy (after Q&A) broken down by:
1. Medium problems (haiku fails, sonnet/opus pass)
2. Hard + Very Hard problems (sonnet also fails)
3. All non-easy (combined)

Usage:
    python plot_qa_sweep_by_difficulty.py --dataset gsm8k
    python plot_qa_sweep_by_difficulty.py --dataset gpqa_freeform
    python plot_qa_sweep_by_difficulty.py --all
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple

# Model configurations
MODELS_CLAUDE = ['haiku', 'sonnet', 'opus']
MODELS_WITH_GPT = ['haiku', 'sonnet', 'opus', 'gpt-oss']
MODELS = MODELS_CLAUDE  # Default, updated by --include-gpt-oss flag

# Difficulty groupings
DIFFICULTY_GROUPS = {
    'medium': ['medium'],
    'hard+very_hard': ['hard', 'very_hard'],
    'all': ['medium', 'hard', 'very_hard'],
}


def load_sweep_results(results_dir: Path,
                       dataset: str,
                       version: str = None) -> Dict:
    """Load all sweep results for a dataset.

    Returns:
        Dict mapping (slm, llm, q_model) -> list of per-problem results
    """
    results = {}

    # Try multiple patterns for compatibility
    patterns = [
        f"{dataset}_SLM-*.json",  # Old format
        f"{dataset}_v*_SLM-*.json",  # New format with version
    ]

    json_files = []
    for pattern in patterns:
        json_files.extend(results_dir.glob(pattern))

    json_files = [f for f in json_files if 'summary' not in f.name]
    json_files = list(set(json_files))  # Remove duplicates

    print(f"Found {len(json_files)} result files for {dataset}")

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Parse filename: {dataset}[_v*]_SLM-{slm}_LLM-{llm}_Q-{q}.json
            name = json_file.stem
            parts = name.split('_')

            slm = llm = q_model = None
            for part in parts:
                if part.startswith('SLM-'):
                    slm = part.replace('SLM-', '')
                elif part.startswith('LLM-'):
                    llm = part.replace('LLM-', '')
                elif part.startswith('Q-'):
                    q_model = part.replace('Q-', '')

            if slm and llm and q_model:
                # Skip oracle configurations
                if 'oracle' in llm:
                    continue

                key = (slm, llm, q_model)
                # Support both 'results' (old) and 'problems' (new) keys
                results[key] = data.get('problems', data.get('results', []))

        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    return results


def load_baseline_difficulties(dataset: str,
                               baseline_dir: Path = None) -> Dict[int, str]:
    """Load problem difficulties from baseline file.

    Returns:
        Dict mapping problem_idx -> difficulty
    """
    # Try new structure first (v3.5/v4.5 directories)
    if baseline_dir:
        # New format: {baseline_dir}/{dataset}_{version}.json
        for pattern in [f"{dataset}_v*.json", f"{dataset}.json"]:
            files = list(baseline_dir.glob(pattern))
            if files:
                baseline_path = files[0]
                break
        else:
            baseline_path = None
    else:
        # Legacy paths
        baseline_files = {
            'gsm8k':
            'lossy_compression/results/gsm8k_all_models_20260115_215021.json',
            'math_algebra':
            'lossy_compression/results/math_all_models_algebra_20260115_001427.json',
            'math_geometry':
            'lossy_compression/results/math_all_models_geometry_20260114_213358.json',
            'math_number_theory':
            'lossy_compression/results/math_all_models_number_theory_20260114_213908.json',
            'gpqa_mc':
            'lossy_compression/results/gpqa_all_models_20260115_185611.json',
            'gpqa_freeform':
            'lossy_compression/results/gpqa_freeform_all_models_20260115_184911.json',
            'mbpp':
            'lossy_compression/results/mbpp_all_models_test_20260115_154846.json',
        }
        baseline_path = baseline_files.get(dataset)

    if not baseline_path or not Path(baseline_path).exists():
        print(f"Warning: baseline file not found for {dataset}")
        return {}

    print(f"Loading baseline from: {baseline_path}")
    with open(baseline_path) as f:
        data = json.load(f)

    difficulties = {}
    for r in data['results']:
        idx = r.get('problem_idx', r.get('idx'))
        diff = r.get('difficulty', 'unknown')
        if idx is not None:
            difficulties[idx] = diff

    return difficulties


def compute_accuracy_by_difficulty(
    results: Dict,
    difficulties: Dict[int, str],
    difficulty_group: List[str],
) -> Dict[Tuple[str, str, str], Tuple[float, int, int]]:
    """Compute accuracy for each model combination filtered by difficulty.

    Returns:
        Dict mapping (slm, llm, q_model) -> (accuracy, correct, total)
    """
    accuracies = {}

    for (slm, llm, q_model), problem_results in results.items():
        correct = 0
        total = 0

        for pr in problem_results:
            # Try multiple keys for problem index
            idx = pr.get('problem_idx') or pr.get('idx')

            # First try difficulty from the result itself (QA results store this)
            diff = pr.get('difficulty', 'unknown')

            # Fall back to baseline difficulties if not in result
            if diff == 'unknown' and idx is not None:
                diff = difficulties.get(idx, 'unknown')

            if diff in difficulty_group:
                total += 1
                if pr.get('final_correct', False):
                    correct += 1

        if total > 0:
            accuracies[(slm, llm, q_model)] = (correct / total, correct, total)
        else:
            accuracies[(slm, llm, q_model)] = (0.0, 0, 0)

    return accuracies


def build_heatmap_matrix(accuracies: Dict,
                         models: List[str] = None,
                         fixed_slm: str = None) -> Tuple[pd.DataFrame, int]:
    """Build matrix for heatmap.

    If fixed_slm is None:
        Rows: N^2 SLM/LLM combinations (N SLM x N LLM)
        Cols: N Question models
    If fixed_slm is set (e.g., 'haiku'):
        Rows: N LLM options
        Cols: N Question models

    Returns:
        (DataFrame, total_problems)
    """
    if models is None:
        models = MODELS

    n_models = len(models)
    total_problems = 0

    if fixed_slm:
        # 3x3 matrix: LLM x Q-model (SLM fixed)
        matrix = np.zeros((n_models, n_models))

        for i, llm in enumerate(models):
            for j, q_model in enumerate(models):
                key = (fixed_slm, llm, q_model)
                if key in accuracies:
                    acc, correct, total = accuracies[key]
                    matrix[i, j] = acc * 100
                    if total_problems == 0:
                        total_problems = total
                else:
                    matrix[i, j] = np.nan

        row_labels = [f"LLM={llm}" for llm in models]
        col_labels = [f"Q={q}" for q in models]
    else:
        # Original 9x3 matrix: SLM/LLM x Q-model
        slm_llm_pairs = [(slm, llm) for slm in models for llm in models]
        matrix = np.zeros((n_models * n_models, n_models))

        for i, (slm, llm) in enumerate(slm_llm_pairs):
            for j, q_model in enumerate(models):
                key = (slm, llm, q_model)
                if key in accuracies:
                    acc, correct, total = accuracies[key]
                    matrix[i, j] = acc * 100
                    if total_problems == 0:
                        total_problems = total
                else:
                    matrix[i, j] = np.nan

        row_labels = [f"{slm}/{llm}" for slm, llm in slm_llm_pairs]
        col_labels = [f"Q={q}" for q in models]

    return pd.DataFrame(matrix, index=row_labels,
                        columns=col_labels), total_problems


def plot_heatmap(df: pd.DataFrame,
                 title: str,
                 ax,
                 vmin: float = 0,
                 vmax: float = 100,
                 show_cbar: bool = False,
                 show_ylabel: bool = True,
                 ylabel: str = 'SLM / LLM'):
    """Plot a single heatmap."""

    sns.heatmap(df,
                annot=True,
                fmt='.1f',
                cmap='RdYlGn',
                vmin=vmin,
                vmax=vmax,
                cbar=show_cbar,
                cbar_kws={'label': 'Accuracy (%)'} if show_cbar else None,
                ax=ax,
                square=False,
                linewidths=0.5,
                linecolor='white')

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Question Model', fontsize=9)

    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
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


def plot_dataset_by_difficulty(results: Dict,
                               difficulties: Dict,
                               dataset: str,
                               output_dir: Path,
                               fixed_slm: str = None):
    """Create 3-panel plot showing accuracy by difficulty group."""

    figsize = (12, 5) if fixed_slm else (16, 8)
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    group_labels = {
        'medium': 'Medium',
        'hard+very_hard': 'Hard + Very Hard',
        'all': 'All Non-Easy',
    }

    slm_label = f" (SLM={fixed_slm})" if fixed_slm else ""
    ylabel = "LLM" if fixed_slm else "SLM / LLM"

    for idx, (group_name, diff_list) in enumerate(DIFFICULTY_GROUPS.items()):
        accuracies = compute_accuracy_by_difficulty(results, difficulties,
                                                    diff_list)
        df, n_problems = build_heatmap_matrix(accuracies, fixed_slm=fixed_slm)

        title = f"{group_labels[group_name]} (n={n_problems})"
        plot_heatmap(df,
                     title,
                     axes[idx],
                     show_cbar=(idx == 2),
                     show_ylabel=(idx == 0),
                     ylabel=ylabel)

    plt.suptitle(f"{dataset.upper()}{slm_label}",
                 fontsize=12,
                 fontweight='bold')
    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_slm_{fixed_slm}" if fixed_slm else ""
    output_path = output_dir / f"{dataset}_qa_by_difficulty{suffix}_{timestamp}.pdf"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    return output_path


def plot_dataset_all_noneasy(results: Dict,
                             difficulties: Dict,
                             dataset: str,
                             output_dir: Path,
                             fixed_slm: str = None):
    """Create single heatmap showing accuracy for all non-easy problems."""

    figsize = (6, 5) if fixed_slm else (8, 8)
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # All non-easy
    diff_list = ['medium', 'hard', 'very_hard']
    accuracies = compute_accuracy_by_difficulty(results, difficulties,
                                                diff_list)
    df, n_problems = build_heatmap_matrix(accuracies, fixed_slm=fixed_slm)

    ylabel = "LLM" if fixed_slm else "SLM / LLM"
    slm_label = f" (SLM={fixed_slm})" if fixed_slm else ""
    title = f"{dataset.upper()}{slm_label} (n={n_problems})"
    plot_heatmap(df,
                 title,
                 ax,
                 show_cbar=True,
                 show_ylabel=True,
                 ylabel=ylabel)

    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_slm_{fixed_slm}" if fixed_slm else ""
    output_path = output_dir / f"{dataset}_qa_all_noneasy{suffix}_{timestamp}.pdf"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    return output_path


def print_summary(results: Dict, difficulties: Dict, dataset: str):
    """Print text summary of results."""

    print(f"\n{'='*60}")
    print(f"{dataset.upper()} SUMMARY")
    print('=' * 60)

    for group_name, diff_list in DIFFICULTY_GROUPS.items():
        accuracies = compute_accuracy_by_difficulty(results, difficulties,
                                                    diff_list)
        df, n_problems = build_heatmap_matrix(accuracies)

        print(f"\n{group_name.upper()} (n={n_problems})")
        print("-" * 40)

        # Best configuration
        stacked = df.stack().dropna()
        if len(stacked) > 0:
            best_val = stacked.max()
            best_idx = stacked.idxmax()
            print(f"Best: {best_idx[0]} + {best_idx[1]} = {best_val:.1f}%")

            # Average
            print(f"Average: {df.mean().mean():.1f}%")


def main():
    global MODELS

    parser = argparse.ArgumentParser(
        description='Plot QA Sweep results by difficulty')
    parser.add_argument('--dataset',
                        type=str,
                        default=None,
                        choices=[
                            'gsm8k', 'math_algebra', 'math_geometry',
                            'math_number_theory', 'gpqa_mc', 'gpqa_freeform',
                            'mbpp', 'mmlu_pro', 'aime', 'hle'
                        ],
                        help='Dataset to plot')
    parser.add_argument('--all', action='store_true', help='Plot all datasets')
    parser.add_argument('--version',
                        type=str,
                        default='v4.5',
                        choices=['v3.5', 'v4.5'],
                        help='Model version (default: v4.5)')
    parser.add_argument(
        '--results-dir',
        type=str,
        default=None,
        help=
        'Results directory (default: results/qa-sweep/{version}/{dataset}_qa_sweep/data)'
    )
    parser.add_argument(
        '--baseline-dir',
        type=str,
        default=None,
        help='Baseline directory (default: results/model-baselines/{version})')
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: results/qa-sweep/{version}/plots)')
    parser.add_argument('--summary',
                        action='store_true',
                        help='Print text summary')
    parser.add_argument(
        '--single',
        action='store_true',
        help='Generate only single all-non-easy heatmap (not 3-panel)')
    parser.add_argument(
        '--both',
        action='store_true',
        help='Generate both 3-panel and single heatmap (default)')
    parser.add_argument(
        '--include-gpt-oss',
        action='store_true',
        help='Include GPT-OSS in the heatmap (4x4 grid instead of 3x3)')
    parser.add_argument(
        '--fixed-slm',
        type=str,
        default=None,
        choices=['haiku', 'sonnet', 'opus', 'gpt-oss'],
        help='Fix SLM to a specific model (3x3 plot: LLM vs Q-model)')

    args = parser.parse_args()

    # Set models based on flag
    if args.include_gpt_oss:
        MODELS = MODELS_WITH_GPT
        print(f"Using models: {MODELS} (4^3=64 combinations)")
    else:
        MODELS = MODELS_CLAUDE
        print(f"Using models: {MODELS} (3^3=27 combinations)")

    if args.fixed_slm:
        print(f"Fixed SLM: {args.fixed_slm} (3x3 heatmap: LLM x Q-model)")

    # Default to generating both if neither specified
    if not args.single:
        args.both = True

    if args.all:
        datasets = [
            'gsm8k', 'math_algebra', 'math_geometry', 'math_number_theory',
            'gpqa_mc', 'gpqa_freeform', 'mbpp', 'mmlu_pro', 'aime', 'hle'
        ]
    elif args.dataset:
        datasets = [args.dataset]
    else:
        print("Please specify --dataset or --all")
        return

    # Default baseline dir
    baseline_dir = Path(args.baseline_dir) if args.baseline_dir else Path(
        f'results/model-baselines/{args.version}')

    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Processing {dataset} ({args.version})")
        print('=' * 60)

        # Determine directories
        if args.results_dir:
            results_dir = Path(args.results_dir)
        else:
            results_dir = Path(
                f'results/qa-sweep/{args.version}/{dataset}_qa_sweep/data')

        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = Path(f'results/qa-sweep/{args.version}/plots')

        if not results_dir.exists():
            print(f"Results directory not found: {results_dir}")
            continue

        output_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        results = load_sweep_results(results_dir, dataset, args.version)
        if not results:
            print(f"No results found for {dataset}")
            continue

        print(f"Loaded {len(results)} model configurations")

        difficulties = load_baseline_difficulties(dataset, baseline_dir)
        print(f"Loaded difficulties for {len(difficulties)} problems")

        if args.summary:
            print_summary(results, difficulties, dataset)

        # Plot
        if args.both:
            plot_dataset_by_difficulty(results,
                                       difficulties,
                                       dataset,
                                       output_dir,
                                       fixed_slm=args.fixed_slm)
            plot_dataset_all_noneasy(results,
                                     difficulties,
                                     dataset,
                                     output_dir,
                                     fixed_slm=args.fixed_slm)
        elif args.single:
            plot_dataset_all_noneasy(results,
                                     difficulties,
                                     dataset,
                                     output_dir,
                                     fixed_slm=args.fixed_slm)

    print("\nDone!")


if __name__ == "__main__":
    main()
