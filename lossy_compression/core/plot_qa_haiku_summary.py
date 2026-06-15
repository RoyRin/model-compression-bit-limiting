#!/usr/bin/env python3
"""
Plot QA Sweep Summary for SLM=haiku across all datasets.

Creates a single figure with all datasets showing the 3x3 LLM×Q-model
combinations for SLM=haiku. Intended for the paper body.

Usage:
    python plot_qa_haiku_summary.py --version v4.5
    python plot_qa_haiku_summary.py --version v3.5
    python plot_qa_haiku_summary.py --both  # Generate for both versions
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Datasets to include (in order) - HLE excluded for now
DATASETS = [
    'gsm8k', 'math_algebra', 'math_geometry', 'math_number_theory', 'gpqa_mc',
    'gpqa_freeform', 'mbpp', 'aime'
]

# Pretty names for datasets (single line)
DATASET_NAMES = {
    'gsm8k': 'GSM8K',
    'math_algebra': 'MATH (Algebra)',
    'math_geometry': 'MATH (Geometry)',
    'math_number_theory': 'MATH (Num. Theory)',
    'gpqa_mc': 'GPQA (MC)',
    'gpqa_freeform': 'GPQA (Freeform)',
    'mbpp': 'MBPP',
    'mmlu_pro': 'MMLU Pro',
    'aime': 'AIME',
    'hle': 'HLE',
}

# Models for the heatmap (Claude models only for 3x3)
MODELS_CLAUDE = ['haiku', 'sonnet', 'opus']
# All models including gpt-oss (for 4x4)
MODELS_ALL = ['haiku', 'sonnet', 'opus', 'gpt-oss']


def load_sweep_results(results_dir: Path,
                       dataset: str,
                       slm_filter: str = 'haiku',
                       models: list = None) -> Dict:
    """Load sweep results for a dataset, filtering to specified SLM."""
    if models is None:
        models = MODELS_CLAUDE

    results = {}

    patterns = [
        f"{dataset}_SLM-{slm_filter}_*.json",
        f"{dataset}_v*_SLM-{slm_filter}_*.json",
    ]

    json_files = []
    for pattern in patterns:
        json_files.extend(results_dir.glob(pattern))

    json_files = [f for f in json_files if 'summary' not in f.name]
    json_files = list(set(json_files))

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

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

            # Only include specified SLM and valid LLM/Q models
            if slm == slm_filter and llm in models and q_model in models:
                key = (llm, q_model)
                problems = data.get('problems', data.get('results', []))
                results[key] = problems

        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    return results


def compute_accuracy(results: Dict, models: list = None) -> pd.DataFrame:
    """Compute accuracy matrix (LLM × Q-model)."""
    if models is None:
        models = MODELS_CLAUDE

    matrix = np.zeros((len(models), len(models)))

    for i, llm in enumerate(models):
        for j, q_model in enumerate(models):
            key = (llm, q_model)
            if key in results:
                problems = results[key]
                if problems:
                    correct = sum(1 for p in problems
                                  if p.get('final_correct', False))
                    matrix[i, j] = 100 * correct / len(problems)
                else:
                    matrix[i, j] = np.nan
            else:
                matrix[i, j] = np.nan

    row_labels = [f"LLM={m}" for m in models]
    col_labels = [f"Q={m}" for m in models]

    return pd.DataFrame(matrix, index=row_labels, columns=col_labels)


def get_problem_count(results: Dict) -> int:
    """Get the number of problems from results."""
    for key, problems in results.items():
        if problems:
            return len(problems)
    return 0


def plot_summary_figure(version: str,
                        output_dir: Path,
                        slm: str = 'haiku',
                        models: list = None,
                        results_base: Path = None):
    """Create summary figure with all datasets."""
    if models is None:
        models = MODELS_CLAUDE

    if results_base is None:
        results_base = Path(f'results/qa-sweep/{version}')

    # Collect data for all datasets
    all_data = {}
    for dataset in DATASETS:
        results_dir = results_base / f"{dataset}_qa_sweep" / "data"
        if results_dir.exists():
            results = load_sweep_results(results_dir,
                                         dataset,
                                         slm_filter=slm,
                                         models=models)
            if results:
                df = compute_accuracy(results, models=models)
                n_problems = get_problem_count(results)
                all_data[dataset] = {'df': df, 'n': n_problems}

    if not all_data:
        print(f"No data found for {version}")
        return None

    # Determine grid layout
    n_datasets = len(all_data)
    n_cols = min(4, n_datasets)
    n_rows = (n_datasets + n_cols - 1) // n_cols

    # Create figure
    fig, axes = plt.subplots(n_rows,
                             n_cols,
                             figsize=(4 * n_cols, 3.5 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Find global min/max for consistent colormap
    all_values = []
    for data in all_data.values():
        all_values.extend(data['df'].values.flatten())
    all_values = [v for v in all_values if not np.isnan(v)]
    vmin = min(all_values) if all_values else 0
    vmax = max(all_values) if all_values else 100

    # Plot each dataset
    idx = 0
    for dataset in DATASETS:
        if dataset not in all_data:
            continue

        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        data = all_data[dataset]
        df = data['df']
        n = data['n']

        # Create annotations with % symbol
        annot_data = df.map(lambda x: f'{x:.0f}%' if not np.isnan(x) else '')

        sns.heatmap(df,
                    annot=annot_data,
                    fmt='',
                    cmap='RdYlGn',
                    vmin=vmin,
                    vmax=vmax,
                    cbar=False,
                    ax=ax,
                    square=True,
                    linewidths=0.5,
                    linecolor='white',
                    annot_kws={'size': 9})

        title = DATASET_NAMES.get(dataset, dataset)
        ax.set_title(f"{title} (n={n})", fontsize=10, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticklabels(ax.get_xticklabels(),
                           rotation=45,
                           ha='right',
                           fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

        idx += 1

    # Hide empty subplots
    for i in range(idx, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')

    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap='RdYlGn',
                               norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Accuracy (%)', fontsize=10)

    # No main title - keep it clean for paper
    plt.tight_layout(rect=[0, 0, 0.9, 1])

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slm_name = slm.replace('-', '_')
    output_path = output_dir / f"qa_{slm_name}_summary_{version}_{timestamp}.pdf"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    return output_path


def plot_compact_summary(version: str,
                         output_dir: Path,
                         slm: str = 'haiku',
                         models: list = None,
                         results_base: Path = None):
    """Create a more compact bar chart summary."""
    if models is None:
        models = MODELS_CLAUDE

    if results_base is None:
        results_base = Path(f'results/qa-sweep/{version}')

    # Collect best accuracy for each dataset
    summary_data = []

    for dataset in DATASETS:
        results_dir = results_base / f"{dataset}_qa_sweep" / "data"
        if not results_dir.exists():
            continue

        results = load_sweep_results(results_dir,
                                     dataset,
                                     slm_filter=slm,
                                     models=models)
        if not results:
            continue

        # Find best LLM×Q combination
        best_acc = 0
        best_combo = None
        baseline_acc = 0  # baseline uses same model for LLM and Q as the SLM

        for (llm, q_model), problems in results.items():
            if not problems:
                continue
            correct = sum(1 for p in problems if p.get('final_correct', False))
            acc = 100 * correct / len(problems)

            # Baseline: LLM=slm and Q=slm (e.g., haiku/haiku for SLM=haiku)
            if llm == slm and q_model == slm:
                baseline_acc = acc

            if acc > best_acc:
                best_acc = acc
                best_combo = f"{llm}/{q_model}"

        n_problems = get_problem_count(results)
        summary_data.append({
            'dataset':
            DATASET_NAMES.get(dataset, dataset).replace('\n', ' '),
            'best_acc':
            best_acc,
            'best_combo':
            best_combo,
            'baseline_acc':
            baseline_acc,
            'n':
            n_problems,
        })

    if not summary_data:
        print(f"No data found for {version}")
        return None

    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 5))

    datasets = [d['dataset'] for d in summary_data]
    best_accs = [d['best_acc'] for d in summary_data]
    baseline_accs = [d['baseline_acc'] for d in summary_data]

    x = np.arange(len(datasets))
    width = 0.35

    bars1 = ax.bar(x - width / 2,
                   baseline_accs,
                   width,
                   label=f'Baseline ({slm}/{slm})',
                   color='#3498db',
                   alpha=0.8)
    bars2 = ax.bar(x + width / 2,
                   best_accs,
                   width,
                   label='Best LLM/Q combo',
                   color='#2ecc71',
                   alpha=0.8)

    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right', fontsize=9)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(0, 105)

    # Add value labels
    for bar, combo in zip(bars2, [d['best_combo'] for d in summary_data]):
        height = bar.get_height()
        ax.annotate(f'{combo}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center',
                    va='bottom',
                    fontsize=7,
                    rotation=90)

    # No main title - keep it clean for paper
    plt.tight_layout()

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slm_name = slm.replace('-', '_')
    output_path = output_dir / f"qa_{slm_name}_bar_{version}_{timestamp}.pdf"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Plot QA summary for specified SLM')
    parser.add_argument('--version',
                        type=str,
                        default='v4.5',
                        choices=['v3.5', 'v4.5'],
                        help='Model version')
    parser.add_argument('--both',
                        action='store_true',
                        help='Generate for both versions')
    parser.add_argument('--output-dir',
                        type=str,
                        default=None,
                        help='Output directory')
    parser.add_argument(
        '--results-dir',
        type=str,
        default=None,
        help=
        'Results directory (default: results/qa-sweep/{version}). Use for 100q results.'
    )
    parser.add_argument('--compact',
                        action='store_true',
                        help='Generate compact bar chart instead of heatmaps')
    parser.add_argument('--slm',
                        type=str,
                        default='haiku',
                        choices=['haiku', 'sonnet', 'opus', 'gpt-oss'],
                        help='SLM to filter results by')
    parser.add_argument(
        '--include-gpt-oss',
        action='store_true',
        help='Include gpt-oss in LLM/Q model grid (4x4 instead of 3x3)')

    args = parser.parse_args()

    versions = ['v3.5', 'v4.5'] if args.both else [args.version]
    models = MODELS_ALL if args.include_gpt_oss else MODELS_CLAUDE

    for version in versions:
        # Use custom results dir if provided, otherwise default
        if args.results_dir:
            results_base = Path(args.results_dir)
            output_dir = Path(
                args.output_dir) if args.output_dir else results_base / 'plots'
        else:
            results_base = Path(f'results/qa-sweep/{version}')
            output_dir = Path(
                args.output_dir) if args.output_dir else results_base / 'plots'

        print(f"\n{'='*60}")
        print(f"Generating SLM={args.slm} summary for {version}")
        print(f"Results dir: {results_base}")
        print(f"Models: {models}")
        print(f"{'='*60}")

        # Generate both plot types
        plot_summary_figure(version,
                            output_dir,
                            slm=args.slm,
                            models=models,
                            results_base=results_base)
        plot_compact_summary(version,
                             output_dir,
                             slm=args.slm,
                             models=models,
                             results_base=results_base)

    print("\nDone!")


if __name__ == "__main__":
    main()
