#!/usr/bin/env python3
"""
Plot results from multiple question-answer compression experiments.
Compares different question generation models.
"""

import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import seaborn as sns
from datetime import datetime

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_experiment_data(file_path: str) -> Dict[str, Any]:
    """Load experiment summary data from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def get_model_name(config: Dict[str, Any]) -> str:
    """Extract a readable model name from config."""
    if 'question_model' in config:
        model = config['question_model']
        # Extract key part of model name
        if 'haiku' in model.lower():
            return 'Haiku'
        elif 'sonnet' in model.lower():
            return 'Sonnet'
        elif 'opus' in model.lower():
            return 'Opus'
        else:
            return model.split('/')[-1].split('-')[0]
    return 'Unknown'


def get_timestamp_str() -> str:
    """Get current timestamp string for plot annotations."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_filename_timestamp() -> str:
    """Get timestamp string safe for filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def plot_start_end_scores(experiments: List[Dict[str, Any]],
                          output_file: str = None):
    """
    Plot end score vs start score for multiple experiments.
    Each experiment can have a different question model.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

    # Collect all model names for title
    all_model_names = []

    for idx, exp_data in enumerate(experiments):
        config = exp_data.get('config', {})
        model_name = get_model_name(config)
        if model_name not in all_model_names:
            all_model_names.append(model_name)

        # Collect start and end scores
        start_scores = []
        end_scores = []

        if 'results' in exp_data:
            for result in exp_data['results']:
                if 'initial_score' in result and 'final_score' in result:
                    start_scores.append(result['initial_score'])
                    end_scores.append(result['final_score'])

        if start_scores and end_scores:
            # Plot scatter points
            ax.scatter(start_scores,
                       end_scores,
                       color=colors[idx],
                       marker=markers[idx % len(markers)],
                       s=100,
                       alpha=0.6,
                       label=f'{model_name} (n={len(start_scores)})',
                       edgecolors='black',
                       linewidth=0.5)

            # Add trend line
            z = np.polyfit(start_scores, end_scores, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(start_scores), max(start_scores), 100)
            ax.plot(x_trend,
                    p(x_trend),
                    color=colors[idx],
                    alpha=0.3,
                    linestyle='--')

            # Calculate and display average improvement
            avg_improvement = np.mean(
                [e - s for s, e in zip(start_scores, end_scores)])
            ax.text(0.05,
                    0.95 - idx * 0.05,
                    f'{model_name}: Avg Δ = {avg_improvement:.2f}',
                    transform=ax.transAxes,
                    fontsize=10,
                    bbox=dict(boxstyle='round',
                              facecolor=colors[idx],
                              alpha=0.2))

    # Add diagonal line (y=x) for reference
    ax.plot([0, 10], [0, 10], 'k--', alpha=0.3, label='No improvement')

    # Add improvement zones
    ax.fill_between([0, 10], [0, 10], [10, 10],
                    alpha=0.05,
                    color='green',
                    label='Improvement zone')
    ax.fill_between([0, 10], [0, 0], [0, 10],
                    alpha=0.05,
                    color='red',
                    label='Degradation zone')

    ax.set_xlabel('Initial Score (SLM Baseline)', fontsize=12)
    ax.set_ylabel('Final Score (After Q&A Compression)', fontsize=12)

    # Create title with model names
    if all_model_names:
        model_str = ', '.join(all_model_names) if len(
            all_model_names) > 1 else all_model_names[0]
        title = f'Q&A Compression: Final vs Initial Scores\n(Question Model: {model_str})'
    else:
        title = 'Q&A Compression: Final vs Initial Scores'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')

    # Add timestamp
    fig.text(0.99,
             0.01,
             f'Generated: {get_timestamp_str()}',
             fontsize=8,
             ha='right',
             va='bottom',
             alpha=0.5)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved start-end plot to {output_file}")
    else:
        plt.show()


def plot_start_best_scores(experiments: List[Dict[str, Any]],
                           output_file: str = None):
    """
    Plot best score achieved vs start score for multiple experiments.
    Similar to start_end but shows the best score seen during the process.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

    # Collect all model names for title
    all_model_names = []

    for idx, exp_data in enumerate(experiments):
        config = exp_data.get('config', {})
        model_name = get_model_name(config)
        if model_name not in all_model_names:
            all_model_names.append(model_name)

        # Collect start and best scores
        start_scores = []
        best_scores = []

        if 'results' in exp_data:
            for result in exp_data['results']:
                if 'initial_score' in result:
                    initial = result['initial_score']
                    start_scores.append(initial)

                    # Calculate best score seen (should include initial)
                    best = initial  # Start with initial as minimum best

                    # Check if we have quality progression to find true best
                    if 'quality_progression' in result and result[
                            'quality_progression']:
                        # Best is the maximum across all scores including initial
                        best = max(result['quality_progression'])

                    # Also check explicit best_quality_score field
                    if 'best_quality_score' in result:
                        best = max(best, result['best_quality_score'])

                    # Also consider final score
                    if 'final_score' in result:
                        best = max(best, result['final_score'])
                    elif 'final_quality_score' in result:
                        best = max(best, result['final_quality_score'])

                    # Ensure best is at least as good as initial
                    best = max(best, initial)
                    best_scores.append(best)

        if start_scores and best_scores and len(start_scores) == len(
                best_scores):
            # Plot scatter points
            ax.scatter(start_scores,
                       best_scores,
                       color=colors[idx],
                       marker=markers[idx % len(markers)],
                       s=100,
                       alpha=0.6,
                       label=f'{model_name} (n={len(start_scores)})',
                       edgecolors='black',
                       linewidth=0.5)

            # Add trend line
            z = np.polyfit(start_scores, best_scores, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(start_scores), max(start_scores), 100)
            ax.plot(x_trend,
                    p(x_trend),
                    color=colors[idx],
                    alpha=0.3,
                    linestyle='--')

            # Calculate and display average improvement
            avg_improvement = np.mean(
                [b - s for s, b in zip(start_scores, best_scores)])
            ax.text(0.05,
                    0.95 - idx * 0.05,
                    f'{model_name}: Avg Δ = {avg_improvement:.2f}',
                    transform=ax.transAxes,
                    fontsize=10,
                    bbox=dict(boxstyle='round',
                              facecolor=colors[idx],
                              alpha=0.2))

    # Add diagonal line (y=x) for reference
    ax.plot([0, 10], [0, 10], 'k--', alpha=0.3, label='No improvement')

    # Add improvement zones
    ax.fill_between([0, 10], [0, 10], [10, 10],
                    alpha=0.05,
                    color='green',
                    label='Improvement zone')
    ax.fill_between([0, 10], [0, 0], [0, 10],
                    alpha=0.05,
                    color='red',
                    label='Degradation zone')

    ax.set_xlabel('Initial Score (SLM Baseline)', fontsize=12)
    ax.set_ylabel('Best Score Achieved', fontsize=12)

    # Create title with model names
    if all_model_names:
        model_str = ', '.join(all_model_names) if len(
            all_model_names) > 1 else all_model_names[0]
        title = f'Q&A Compression: Best vs Initial Scores\n(Question Model: {model_str})'
    else:
        title = 'Q&A Compression: Best vs Initial Scores'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')

    # Add timestamp
    fig.text(0.99,
             0.01,
             f'Generated: {get_timestamp_str()}',
             fontsize=8,
             ha='right',
             va='bottom',
             alpha=0.5)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved start-best plot to {output_file}")
    else:
        plt.show()


def plot_quality_progressions(experiments: List[Dict[str, Any]],
                              output_file: str = None):
    """
    Plot quality score progression over iterations for multiple experiments.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, exp_data in enumerate(experiments[:4]):  # Max 4 subplots
        ax = axes[idx]
        config = exp_data.get('config', {})
        model_name = get_model_name(config)

        progressions = []
        if 'results' in exp_data:
            for result in exp_data['results']:
                if 'quality_progression' in result and result[
                        'quality_progression']:
                    progressions.append(result['quality_progression'])

        if progressions:
            # Plot each progression as a thin line
            for prog in progressions:
                iterations = list(range(len(prog)))
                ax.plot(iterations, prog, alpha=0.3, linewidth=0.5)

            # Plot average progression as thick line
            max_len = max(len(p) for p in progressions)
            avg_progression = []
            for i in range(max_len):
                scores_at_i = [p[i] for p in progressions if i < len(p)]
                if scores_at_i:
                    avg_progression.append(np.mean(scores_at_i))

            ax.plot(range(len(avg_progression)),
                    avg_progression,
                    linewidth=2,
                    color='red',
                    label='Average')

            ax.set_xlabel('Iteration')
            ax.set_ylabel('Quality Score')
            ax.set_title(f'{model_name} Question Model', fontweight='bold')
            ax.set_ylim(0, 10)
            ax.grid(True, alpha=0.3)
            ax.legend()
        else:
            ax.text(0.5,
                    0.5,
                    f'No progression data for {model_name}',
                    transform=ax.transAxes,
                    ha='center',
                    va='center')

    # Hide unused subplots
    for idx in range(len(experiments), 4):
        axes[idx].axis('off')

    # Collect all model names for title
    all_model_names = []
    for exp_data in experiments[:4]:
        config = exp_data.get('config', {})
        model_name = get_model_name(config)
        if model_name not in all_model_names:
            all_model_names.append(model_name)

    # Create title with model names
    if all_model_names:
        model_str = ', '.join(all_model_names) if len(
            all_model_names) > 1 else all_model_names[0]
        title = f'Quality Score Progression During Q&A Compression\n(Question Model: {model_str})'
    else:
        title = 'Quality Score Progression During Q&A Compression'
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Add timestamp
    fig.text(0.99,
             0.01,
             f'Generated: {get_timestamp_str()}',
             fontsize=8,
             ha='right',
             va='bottom',
             alpha=0.5)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved progression plot to {output_file}")
    else:
        plt.show()


def plot_improvement_distribution(experiments: List[Dict[str, Any]],
                                  output_file: str = None):
    """
    Plot distribution of improvements for each experiment.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    improvements_by_model = []
    model_names = []
    all_model_names = []

    for exp_data in experiments:
        config = exp_data.get('config', {})
        model_name = get_model_name(config)
        if model_name not in all_model_names:
            all_model_names.append(model_name)

        improvements = []
        if 'results' in exp_data:
            for result in exp_data['results']:
                if 'improvement' in result:
                    improvements.append(result['improvement'])

        if improvements:
            improvements_by_model.append(improvements)
            model_names.append(model_name)

    if improvements_by_model:
        # Create violin plot
        parts = ax.violinplot(improvements_by_model,
                              positions=range(len(model_names)),
                              showmeans=True,
                              showmedians=True)

        # Customize colors
        colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.6)

        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names)
        ax.set_xlabel('Question Generation Model', fontsize=12)
        ax.set_ylabel('Score Improvement', fontsize=12)

        # Create title with model names
        if all_model_names:
            model_str = ', '.join(all_model_names) if len(
                all_model_names) > 1 else all_model_names[0]
            title = f'Distribution of Quality Score Improvements\n(Question Model: {model_str})'
        else:
            title = 'Distribution of Quality Score Improvements'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)

        # Add statistics
        for idx, (improvements,
                  name) in enumerate(zip(improvements_by_model, model_names)):
            mean_imp = np.mean(improvements)
            median_imp = np.median(improvements)
            ax.text(idx,
                    ax.get_ylim()[1] * 0.95,
                    f'μ={mean_imp:.2f}\nm={median_imp:.2f}',
                    ha='center',
                    fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Add timestamp
    fig.text(0.99,
             0.01,
             f'Generated: {get_timestamp_str()}',
             fontsize=8,
             ha='right',
             va='bottom',
             alpha=0.5)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved improvement distribution plot to {output_file}")
    else:
        plt.show()


def plot_utility_over_questions(experiments: List[Dict[str, Any]],
                                output_file: str = None):
    """
    Plot utility score (quality) over number of questions asked.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))

    for idx, exp_data in enumerate(experiments):
        config = exp_data.get('config', {})
        model_name = config.get('question_model',
                                'Unknown').split('/')[-1].split('-')[1]

        # Collect quality progressions
        all_progressions = []
        if 'results' in exp_data:
            for result in exp_data['results']:
                if 'quality_progression' in result and result[
                        'quality_progression']:
                    progression = result['quality_progression']
                    if progression:  # Make sure it's not empty
                        all_progressions.append(progression)

        if all_progressions:
            # Left plot: All trajectories
            ax_left = axes[0]
            for prog in all_progressions:
                questions = list(range(len(prog)))
                ax_left.plot(questions,
                             prog,
                             alpha=0.2,
                             color=colors[idx],
                             linewidth=0.5)

            # Calculate and plot average
            max_len = max(len(p) for p in all_progressions)
            avg_progression = []
            std_progression = []
            for i in range(max_len):
                scores_at_i = [p[i] for p in all_progressions if i < len(p)]
                if scores_at_i:
                    avg_progression.append(np.mean(scores_at_i))
                    std_progression.append(np.std(scores_at_i))

            questions = list(range(len(avg_progression)))
            ax_left.plot(questions,
                         avg_progression,
                         color=colors[idx],
                         linewidth=2,
                         label=f'{model_name} (n={len(all_progressions)})')

            # Add confidence band
            avg_array = np.array(avg_progression)
            std_array = np.array(std_progression)
            ax_left.fill_between(questions,
                                 avg_array - std_array,
                                 avg_array + std_array,
                                 color=colors[idx],
                                 alpha=0.1)

            # Right plot: Best seen so far
            ax_right = axes[1]
            for prog in all_progressions:
                best_so_far = np.maximum.accumulate(prog)
                questions = list(range(len(best_so_far)))
                ax_right.plot(questions,
                              best_so_far,
                              alpha=0.2,
                              color=colors[idx],
                              linewidth=0.5)

            # Calculate average of best-so-far
            avg_best_so_far = []
            for i in range(max_len):
                best_at_i = []
                for prog in all_progressions:
                    if i < len(prog):
                        best_at_i.append(max(prog[:i + 1]))
                if best_at_i:
                    avg_best_so_far.append(np.mean(best_at_i))

            questions = list(range(len(avg_best_so_far)))
            ax_right.plot(questions,
                          avg_best_so_far,
                          color=colors[idx],
                          linewidth=2,
                          label=f'{model_name}')

    # Configure left plot
    axes[0].set_xlabel('Number of Questions', fontsize=12)
    axes[0].set_ylabel('Utility Score (Quality)', fontsize=12)
    axes[0].set_title('Utility Score vs Number of Questions',
                      fontsize=13,
                      fontweight='bold')
    axes[0].set_ylim(0, 10)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='lower right')

    # Configure right plot
    axes[1].set_xlabel('Number of Questions', fontsize=12)
    axes[1].set_ylabel('Best Utility Score Seen', fontsize=12)
    axes[1].set_title('Best Utility Score Seen So Far',
                      fontsize=13,
                      fontweight='bold')
    axes[1].set_ylim(0, 10)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='lower right')

    # Collect all model names for title
    all_model_names = []
    for exp_data in experiments:
        config = exp_data.get('config', {})
        model_name = get_model_name(config)
        if model_name not in all_model_names:
            all_model_names.append(model_name)

    # Create title with model names
    if all_model_names:
        model_str = ', '.join(all_model_names) if len(
            all_model_names) > 1 else all_model_names[0]
        title = f'Question Efficiency Analysis\n(Question Model: {model_str})'
    else:
        title = 'Question Efficiency Analysis'
    plt.suptitle(title, fontsize=14, fontweight='bold')

    # Add timestamp
    fig.text(0.99,
             0.01,
             f'Generated: {get_timestamp_str()}',
             fontsize=8,
             ha='right',
             va='bottom',
             alpha=0.5)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved utility plot to {output_file}")
    else:
        plt.show()


def plot_individual_trajectories(experiment_dirs: List[str],
                                 output_file: str = None):
    """
    Plot quality progression for all individual questions from experiment directories.
    Each question gets its own line, and multiple experiments are shown in subplots.
    """
    # Check if we're dealing with summary files or directories
    valid_dirs = []
    for exp_path in experiment_dirs:
        path = Path(exp_path)
        if path.is_dir():
            logs_dir = path / "logs"
            if logs_dir.exists():
                valid_dirs.append(path)
        elif path.is_file() and path.suffix == '.json':
            # This is a summary file, get its parent directory
            parent = path.parent
            logs_dir = parent / "logs"
            if logs_dir.exists():
                valid_dirs.append(parent)

    if not valid_dirs:
        print("No valid experiment directories with logs found")
        return

    # Create subplots based on number of experiments
    n_experiments = len(valid_dirs)
    n_cols = min(2, n_experiments)
    n_rows = (n_experiments + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows))
    if n_experiments == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    for exp_idx, exp_dir in enumerate(valid_dirs):
        ax = axes[exp_idx] if n_experiments > 1 else axes[0]
        logs_dir = exp_dir / "logs"

        # Load all question files
        question_files = sorted(logs_dir.glob("question_*.json"))

        # Use a colormap for different questions
        colors = plt.cm.tab20(np.linspace(0, 1, len(question_files)))

        # Track statistics
        all_progressions = []
        max_iterations = 0
        lines_plotted = 0

        for idx, file_path in enumerate(question_files):
            with open(file_path, 'r') as f:
                question_data = json.load(f)

            question_id = question_data.get("question_id", idx)
            category = question_data.get("category", "unknown")

            # Get quality progression if available
            if "summary" in question_data and "quality_progression" in question_data[
                    "summary"]:
                progression = question_data["summary"]["quality_progression"]

                if progression and len(progression) > 0:
                    # Plot individual trajectory with both line and scatter points
                    iterations = list(range(len(progression)))
                    # Plot line
                    ax.plot(iterations,
                            progression,
                            color=colors[idx],
                            alpha=0.6,
                            linewidth=1.0,
                            label=f'Q{question_id}'
                            if len(question_files) <= 15 else None)
                    # Add scatter points to make individual points visible
                    ax.scatter(iterations,
                               progression,
                               color=colors[idx],
                               s=20,
                               alpha=0.8,
                               zorder=5)

                    all_progressions.append(progression)
                    max_iterations = max(max_iterations, len(progression))
                    lines_plotted += 1

        # Try to load config to get model info
        model_info = 'Unknown'
        summary_file = exp_dir / 'summary.json'
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                summary_data = json.load(f)
                if 'config' in summary_data:
                    model_info = get_model_name(summary_data['config'])

        # Add text showing number of lines plotted and model info
        info_text = f'Lines plotted: {lines_plotted}/{len(question_files)}\nQuestion Model: {model_info}'
        ax.text(0.02,
                0.98,
                info_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Configure subplot
        ax.set_xlabel('Number of Questions Asked', fontsize=11)
        ax.set_ylabel('Quality Score', fontsize=11)
        ax.set_title(f'{exp_dir.name}', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 10)
        ax.grid(True, alpha=0.3)
        if len(question_files) <= 15:
            ax.legend(fontsize=8, ncol=2 if len(question_files) > 10 else 1)

    # Hide unused subplots
    for idx in range(n_experiments, len(axes)):
        axes[idx].axis('off')

    # Collect all model names from experiments for title
    all_model_names = []
    for exp_dir in valid_dirs:
        summary_file = exp_dir / 'summary.json'
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                summary_data = json.load(f)
                if 'config' in summary_data:
                    model_name = get_model_name(summary_data['config'])
                    if model_name not in all_model_names and model_name != 'Unknown':
                        all_model_names.append(model_name)

    # Create title with model names
    if all_model_names:
        model_str = ', '.join(all_model_names) if len(
            all_model_names) > 1 else all_model_names[0]
        title = f'Individual Question Trajectories\n(Question Model: {model_str})'
    else:
        title = 'Individual Question Trajectories'
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Add timestamp
    fig.text(0.99,
             0.01,
             f'Generated: {get_timestamp_str()}',
             fontsize=8,
             ha='right',
             va='bottom',
             alpha=0.5)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved individual trajectories plot to {output_file}")
    else:
        plt.show()


def plot_summary_comparison(experiments: List[Dict[str, Any]],
                            output_file: str = None):
    """
    Create a summary comparison bar chart.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    model_names = []
    avg_initial_scores = []
    avg_final_scores = []
    avg_improvements = []
    success_rates = []

    for exp_data in experiments:
        config = exp_data.get('config', {})
        model_name = get_model_name(config)
        model_names.append(model_name)

        if 'statistics' in exp_data:
            stats = exp_data['statistics']
            avg_initial_scores.append(stats.get('avg_initial_score', 0))
            avg_final_scores.append(stats.get('avg_final_score', 0))
            avg_improvements.append(stats.get('avg_improvement', 0))
        elif 'results' in exp_data:
            # Calculate from results
            results = exp_data['results']
            initial_scores = [
                r['initial_score'] for r in results if 'initial_score' in r
            ]
            final_scores = [
                r['final_score'] for r in results if 'final_score' in r
            ]
            improvements = [
                r['improvement'] for r in results if 'improvement' in r
            ]

            avg_initial_scores.append(
                np.mean(initial_scores) if initial_scores else 0)
            avg_final_scores.append(
                np.mean(final_scores) if final_scores else 0)
            avg_improvements.append(
                np.mean(improvements) if improvements else 0)

            total = len(exp_data.get('results', []))
            successful = sum(1 for r in exp_data.get('results', [])
                             if r.get('reached_threshold', False))
            success_rates.append(successful / total * 100 if total > 0 else 0)

    x = np.arange(len(model_names))
    width = 0.35

    # Plot initial vs final scores
    ax1 = axes[0]
    ax1.bar(x - width / 2,
            avg_initial_scores,
            width,
            label='Initial',
            alpha=0.7)
    ax1.bar(x + width / 2, avg_final_scores, width, label='Final', alpha=0.7)
    ax1.set_ylabel('Quality Score')
    ax1.set_title('Average Scores')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names)
    ax1.legend()
    ax1.set_ylim(0, 10)
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot improvements
    ax2 = axes[1]
    colors = ['green' if imp > 0 else 'red' for imp in avg_improvements]
    ax2.bar(x, avg_improvements, color=colors, alpha=0.7)
    ax2.set_ylabel('Score Improvement')
    ax2.set_title('Average Improvement')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot success rates if available
    if success_rates:
        ax3 = axes[2]
        ax3.bar(x, success_rates, color='blue', alpha=0.7)
        ax3.set_ylabel('Success Rate (%)')
        ax3.set_title('Threshold Achievement Rate')
        ax3.set_xticks(x)
        ax3.set_xticklabels(model_names)
        ax3.set_ylim(0, 100)
        ax3.grid(True, alpha=0.3, axis='y')
    else:
        axes[2].axis('off')

    # Collect all model names for title
    all_model_names = []
    for exp_data in experiments:
        config = exp_data.get('config', {})
        model_name = get_model_name(config)
        if model_name not in all_model_names:
            all_model_names.append(model_name)

    # Create title with model names
    if all_model_names:
        model_str = ', '.join(all_model_names) if len(
            all_model_names) > 1 else all_model_names[0]
        title = f'Question Model Comparison Summary\n(Question Model: {model_str})'
    else:
        title = 'Question Model Comparison Summary'
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Add timestamp
    fig.text(0.99,
             0.01,
             f'Generated: {get_timestamp_str()}',
             fontsize=8,
             ha='right',
             va='bottom',
             alpha=0.5)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved summary plot to {output_file}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Plot Q&A compression experiment results')
    parser.add_argument(
        'paths',
        nargs='+',
        help='Experiment summary JSON files or experiment directories')
    parser.add_argument('--output-dir',
                        default='plots',
                        help='Output directory for plots')
    parser.add_argument('--plot-types',
                        nargs='+',
                        default=[
                            'start-end', 'start-best', 'progression',
                            'improvement', 'summary', 'utility', 'individual'
                        ],
                        choices=[
                            'start-end', 'start-best', 'progression',
                            'improvement', 'summary', 'utility', 'individual'
                        ],
                        help='Types of plots to generate')

    args = parser.parse_args()

    # Separate files and directories
    json_files = []
    experiment_dirs = []
    experiments = []

    for path_str in args.paths:
        path = Path(path_str)
        if path.exists():
            if path.is_file() and path.suffix == '.json':
                # It's a summary JSON file
                json_files.append(path_str)
                data = load_experiment_data(path_str)
                experiments.append(data)
                print(f"Loaded JSON: {path_str}")
                # Also track the directory for individual plots
                experiment_dirs.append(path_str)
            elif path.is_dir():
                # It's an experiment directory
                experiment_dirs.append(path_str)
                # Try to load summary.json from it
                summary_file = path / 'summary.json'
                if summary_file.exists():
                    data = load_experiment_data(str(summary_file))
                    experiments.append(data)
                    print(f"Loaded from directory: {path_str}/summary.json")
                else:
                    print(f"Warning: No summary.json found in {path_str}")
        else:
            print(f"Warning: Path not found: {path_str}")

    if not experiments and 'individual' not in args.plot_types:
        print("No valid experiment data found for summary plots!")
        if not experiment_dirs:
            return

    # Determine output directory based on question models
    base_output_dir = Path(args.output_dir)

    # Extract question model names from experiments
    model_names = []
    if experiments:
        for exp in experiments:
            if 'config' in exp:
                model_name = get_model_name(exp['config']).lower()
                if model_name not in model_names:
                    model_names.append(model_name)

    # Get timestamp for both directory and filenames
    timestamp = get_filename_timestamp()

    # Create subfolder name based on models and timestamp
    if model_names:
        if len(model_names) == 1:
            # Single model - use simple folder name
            subfolder = f"qa_model__{model_names[0]}__{timestamp}"
        else:
            # Multiple models - combine names
            subfolder = f"qa_models__{'_vs_'.join(sorted(model_names))}__{timestamp}"
    else:
        # Fallback if no model info found
        subfolder = f"qa_model__unknown__{timestamp}"

    output_dir = base_output_dir / subfolder
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📁 Output directory: {output_dir}")

    # Generate plots that need summary data
    if experiments:
        if 'start-end' in args.plot_types:
            plot_start_end_scores(
                experiments, output_dir / f'start_end_scores_{timestamp}.png')

        if 'start-best' in args.plot_types:
            plot_start_best_scores(
                experiments, output_dir / f'start_best_scores_{timestamp}.png')

        if 'progression' in args.plot_types:
            plot_quality_progressions(
                experiments,
                output_dir / f'quality_progressions_{timestamp}.png')

        if 'improvement' in args.plot_types:
            plot_improvement_distribution(
                experiments,
                output_dir / f'improvement_distribution_{timestamp}.png')

        if 'summary' in args.plot_types:
            plot_summary_comparison(
                experiments,
                output_dir / f'summary_comparison_{timestamp}.png')

        if 'utility' in args.plot_types:
            plot_utility_over_questions(
                experiments,
                output_dir / f'utility_over_questions_{timestamp}.png')

    # Generate individual trajectory plots from directories
    if 'individual' in args.plot_types and experiment_dirs:
        plot_individual_trajectories(
            experiment_dirs,
            output_dir / f'individual_trajectories_{timestamp}.png')

    print(f"\n✅ All plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
