#!/usr/bin/env python3
"""
Analyze and visualize comparison between experiments with and without open-ended guidance.
Combines comparison JSON creation and plotting functionality.
"""

import json
import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def create_comparison(exp1_name, exp2_name, base_name, config):
    """Create comparison data from two experiments."""

    # Load results from both experiments
    exp1_path = Path('experiments') / exp1_name / 'summary.json'
    exp2_path = Path('experiments') / exp2_name / 'summary.json'

    if not exp1_path.exists():
        print(f"❌ Error: {exp1_path} not found")
        sys.exit(1)

    if not exp2_path.exists():
        print(f"❌ Error: {exp2_path} not found")
        sys.exit(1)

    with open(exp1_path, 'r') as f:
        data_with_guidance = json.load(f)

    with open(exp2_path, 'r') as f:
        data_without_guidance = json.load(f)

    # Create comparison data structure
    comparison = {
        'experiment_name': base_name,
        'timestamp': base_name.split('_')[-1],
        'config': config,
        'questions': []
    }

    # Match questions from both experiments
    with_guidance_results = {
        r['question_id']: r
        for r in data_with_guidance.get('results', [])
    }
    without_guidance_results = {
        r['question_id']: r
        for r in data_without_guidance.get('results', [])
    }

    # Combine results for each question
    all_question_ids = set(with_guidance_results.keys()) | set(
        without_guidance_results.keys())

    for qid in sorted(all_question_ids):
        question_data = {
            'question_id':
            qid,
            'category':
            with_guidance_results.get(qid, without_guidance_results.get(
                qid, {})).get('category', 'unknown'),
            'initial_score':
            None,
            'with_guidance_score':
            None,
            'without_guidance_score':
            None,
            'with_guidance_iterations':
            None,
            'without_guidance_iterations':
            None
        }

        if qid in with_guidance_results:
            question_data['initial_score'] = with_guidance_results[qid][
                'initial_score']
            question_data['with_guidance_score'] = with_guidance_results[qid][
                'final_score']
            question_data['with_guidance_iterations'] = with_guidance_results[
                qid]['iterations']

        if qid in without_guidance_results:
            if question_data['initial_score'] is None:
                question_data['initial_score'] = without_guidance_results[qid][
                    'initial_score']
            question_data['without_guidance_score'] = without_guidance_results[
                qid]['final_score']
            question_data[
                'without_guidance_iterations'] = without_guidance_results[qid][
                    'iterations']

        comparison['questions'].append(question_data)

    # Calculate summary statistics
    with_guidance_scores = [
        q['with_guidance_score'] for q in comparison['questions']
        if q['with_guidance_score'] is not None
    ]
    without_guidance_scores = [
        q['without_guidance_score'] for q in comparison['questions']
        if q['without_guidance_score'] is not None
    ]
    initial_scores = [
        q['initial_score'] for q in comparison['questions']
        if q['initial_score'] is not None
    ]

    comparison['summary'] = {
        'avg_initial_score':
        sum(initial_scores) / len(initial_scores) if initial_scores else 0,
        'avg_with_guidance_score':
        sum(with_guidance_scores) /
        len(with_guidance_scores) if with_guidance_scores else 0,
        'avg_without_guidance_score':
        sum(without_guidance_scores) /
        len(without_guidance_scores) if without_guidance_scores else 0,
        'avg_improvement_with_guidance':
        (sum(with_guidance_scores) / len(with_guidance_scores) -
         sum(initial_scores) / len(initial_scores))
        if with_guidance_scores and initial_scores else 0,
        'avg_improvement_without_guidance':
        (sum(without_guidance_scores) / len(without_guidance_scores) -
         sum(initial_scores) / len(initial_scores))
        if without_guidance_scores and initial_scores else 0,
        'num_questions':
        len(comparison['questions']),
        'num_improved_with_guidance':
        sum(1 for q in comparison['questions']
            if q['with_guidance_score'] and q['initial_score']
            and q['with_guidance_score'] > q['initial_score']),
        'num_improved_without_guidance':
        sum(1 for q in comparison['questions']
            if q['without_guidance_score'] and q['initial_score']
            and q['without_guidance_score'] > q['initial_score'])
    }

    # Save comparison JSON
    comparison_file = Path('experiments') / base_name / 'comparison.json'
    comparison_file.parent.mkdir(parents=True, exist_ok=True)

    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=2)

    print(f'✅ Comparison data saved to: {comparison_file}')
    print()
    print('📊 Summary Statistics:')
    print(
        f'  Average initial score: {comparison["summary"]["avg_initial_score"]:.2f}/10'
    )
    print(
        f'  Average with guidance: {comparison["summary"]["avg_with_guidance_score"]:.2f}/10'
    )
    print(
        f'  Average without guidance: {comparison["summary"]["avg_without_guidance_score"]:.2f}/10'
    )
    print(
        f'  Improvement with guidance: +{comparison["summary"]["avg_improvement_with_guidance"]:.2f}'
    )
    print(
        f'  Improvement without guidance: +{comparison["summary"]["avg_improvement_without_guidance"]:.2f}'
    )
    print(
        f'  Questions improved with guidance: {comparison["summary"]["num_improved_with_guidance"]}/{comparison["summary"]["num_questions"]}'
    )
    print(
        f'  Questions improved without guidance: {comparison["summary"]["num_improved_without_guidance"]}/{comparison["summary"]["num_questions"]}'
    )

    return comparison_file, comparison


def plot_comparison(comparison_data, output_dir='.'):
    """Create bar charts comparing experiment results."""

    # Prepare data for plotting
    questions = comparison_data['questions']
    question_ids = [q['question_id'] for q in questions]
    categories = [q['category'] for q in questions]
    initial_scores = [
        q['initial_score'] if q['initial_score'] is not None else 0
        for q in questions
    ]
    with_guidance_scores = [
        q['with_guidance_score'] if q['with_guidance_score'] is not None else 0
        for q in questions
    ]
    without_guidance_scores = [
        q['without_guidance_score']
        if q['without_guidance_score'] is not None else 0 for q in questions
    ]

    # Create figure with larger size
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

    # Plot 1: Bar chart comparison
    x = np.arange(len(question_ids))
    width = 0.25

    bars1 = ax1.bar(x - width,
                    initial_scores,
                    width,
                    label='Initial',
                    color='#ff9999')
    bars2 = ax1.bar(x,
                    with_guidance_scores,
                    width,
                    label='With Open-Ended Guidance',
                    color='#66b3ff')
    bars3 = ax1.bar(x + width,
                    without_guidance_scores,
                    width,
                    label='Binary Q&A Only',
                    color='#99ff99')

    ax1.set_xlabel('Question ID')
    ax1.set_ylabel('Score (out of 10)')
    ax1.set_title('LLM-SLM Compression: Score Comparison Across Questions')
    ax1.set_xticks(x)
    ax1.set_xticklabels(
        [f'Q{qid}\n{cat[:4]}' for qid, cat in zip(question_ids, categories)],
        rotation=45,
        ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 10])

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width() / 2.,
                         height + 0.1,
                         f'{height:.1f}',
                         ha='center',
                         va='bottom',
                         fontsize=8)

    # Plot 2: Improvement comparison
    improvements_with = [
        w - i if w is not None and i is not None else 0
        for w, i in zip(with_guidance_scores, initial_scores)
    ]
    improvements_without = [
        w - i if w is not None and i is not None else 0
        for w, i in zip(without_guidance_scores, initial_scores)
    ]

    bars4 = ax2.bar(x - width / 2,
                    improvements_with,
                    width,
                    label='With Open-Ended Guidance',
                    color='#66b3ff')
    bars5 = ax2.bar(x + width / 2,
                    improvements_without,
                    width,
                    label='Binary Q&A Only',
                    color='#99ff99')

    ax2.set_xlabel('Question ID')
    ax2.set_ylabel('Improvement from Initial Score')
    ax2.set_title('Score Improvements by Guidance Type')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Q{qid}' for qid in question_ids], rotation=45)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Add value labels
    for bars in [bars4, bars5]:
        for bar in bars:
            height = bar.get_height()
            if abs(height) > 0.01:
                y_pos = height + 0.05 if height > 0 else height - 0.15
                ax2.text(bar.get_x() + bar.get_width() / 2.,
                         y_pos,
                         f'{height:+.1f}',
                         ha='center',
                         va='bottom' if height > 0 else 'top',
                         fontsize=8)

    # Add summary statistics as text
    summary = comparison_data.get('summary', {})
    textstr = f'Avg Initial: {summary.get("avg_initial_score", 0):.2f}\n'
    textstr += f'Avg w/ Guidance: {summary.get("avg_with_guidance_score", 0):.2f}\n'
    textstr += f'Avg w/o Guidance: {summary.get("avg_without_guidance_score", 0):.2f}'

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.02,
             0.98,
             textstr,
             transform=ax1.transAxes,
             fontsize=10,
             verticalalignment='top',
             bbox=props)

    plt.tight_layout()

    # Save plots
    output_path = Path(output_dir)
    plt.savefig(output_path / 'comparison_plot.png',
                dpi=150,
                bbox_inches='tight')
    plt.savefig(output_path / 'comparison_plot.pdf', bbox_inches='tight')
    print(f'✅ Plots saved as comparison_plot.png and comparison_plot.pdf')

    # Don't show in non-interactive mode
    if sys.stdout.isatty():
        plt.show()
    else:
        plt.close()


def main():
    """Main function for command-line usage."""
    if len(sys.argv) < 2:
        print("Usage:")
        print(
            "  Create comparison: python analyze_comparison.py create <exp1> <exp2> <base_name> [config_json]"
        )
        print(
            "  Plot existing:     python analyze_comparison.py plot <comparison.json> [output_dir]"
        )
        print(
            "  Both:             python analyze_comparison.py both <exp1> <exp2> <base_name> [config_json]"
        )
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "create":
        if len(sys.argv) < 5:
            print(
                "Usage: python analyze_comparison.py create <exp1> <exp2> <base_name> [config_json]"
            )
            sys.exit(1)

        exp1_name = sys.argv[2]
        exp2_name = sys.argv[3]
        base_name = sys.argv[4]
        config = json.loads(sys.argv[5]) if len(sys.argv) > 5 else {}

        create_comparison(exp1_name, exp2_name, base_name, config)

    elif mode == "plot":
        if len(sys.argv) < 3:
            print(
                "Usage: python analyze_comparison.py plot <comparison.json> [output_dir]"
            )
            sys.exit(1)

        comparison_file = sys.argv[2]
        output_dir = sys.argv[3] if len(sys.argv) > 3 else '.'

        with open(comparison_file, 'r') as f:
            comparison_data = json.load(f)

        plot_comparison(comparison_data, output_dir)

    elif mode == "both":
        if len(sys.argv) < 5:
            print(
                "Usage: python analyze_comparison.py both <exp1> <exp2> <base_name> [config_json]"
            )
            sys.exit(1)

        exp1_name = sys.argv[2]
        exp2_name = sys.argv[3]
        base_name = sys.argv[4]
        config = json.loads(sys.argv[5]) if len(sys.argv) > 5 else {}

        # Create comparison
        comparison_file, comparison_data = create_comparison(
            exp1_name, exp2_name, base_name, config)

        # Plot results
        output_dir = comparison_file.parent
        plot_comparison(comparison_data, output_dir)

    else:
        print(f"Unknown mode: {mode}")
        print("Use 'create', 'plot', or 'both'")
        sys.exit(1)


if __name__ == "__main__":
    main()
