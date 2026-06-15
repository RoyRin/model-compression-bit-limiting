#!/usr/bin/env python3
"""
Merge Opus rerun results into existing baseline files and recalculate difficulty.

Usage:
    python scripts/merge_opus_results.py --opus-dir results/opus_rerun --baseline-dir lossy_compression/results
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter


def find_matching_baseline(opus_dataset: str, baseline_dir: Path) -> Path:
    """Find the most recent baseline file for a dataset."""

    # Map opus dataset names to baseline patterns
    patterns = {
        'math_algebra': 'math_all_models_algebra_*.json',
        'math_geometry': 'math_all_models_geometry_*.json',
        'math_number_theory': 'math_all_models_number_theory_*.json',
        'gsm8k': 'gsm8k_all_models_*.json',
        'gpqa_mc': 'gpqa_all_models_*.json',
        'gpqa_freeform': 'gpqa_freeform_all_models_*.json',
        'mbpp': 'mbpp_all_models_*.json',
    }

    pattern = patterns.get(opus_dataset)
    if not pattern:
        return None

    candidates = sorted(baseline_dir.glob(pattern), reverse=True)
    if candidates:
        return candidates[0]
    return None


def merge_opus_into_baseline(baseline_path: Path, opus_path: Path,
                             output_path: Path):
    """Merge opus results into baseline and save updated file."""

    print(f"\nMerging: {opus_path.name}")
    print(f"  Into: {baseline_path.name}")

    with open(baseline_path) as f:
        baseline = json.load(f)

    with open(opus_path) as f:
        opus_data = json.load(f)

    # Create lookup for opus results
    opus_lookup = {r['problem_idx']: r for r in opus_data['results']}

    # Track changes
    opus_changed = 0
    difficulty_changed = 0

    # Update each result
    for result in baseline['results']:
        idx = result['problem_idx']
        if idx in opus_lookup:
            opus_result = opus_lookup[idx]

            old_opus_correct = result.get('models',
                                          {}).get('opus',
                                                  {}).get('correct', False)
            new_opus_correct = opus_result.get('opus_correct', False)

            if old_opus_correct != new_opus_correct:
                opus_changed += 1

            # Update opus data
            result['models']['opus'] = {
                'answer': opus_result.get('opus_answer'),
                'correct': new_opus_correct,
                'solve_time': opus_result.get('opus_time', 0),
            }

        # Recalculate difficulty
        haiku_ok = result.get('models', {}).get('haiku',
                                                {}).get('correct', False)
        sonnet_ok = result.get('models', {}).get('sonnet',
                                                 {}).get('correct', False)
        opus_ok = result.get('models', {}).get('opus',
                                               {}).get('correct', False)

        old_difficulty = result.get('difficulty', 'unknown')

        if haiku_ok and sonnet_ok and opus_ok:
            new_difficulty = 'easy'
        elif not haiku_ok and (sonnet_ok or opus_ok):
            new_difficulty = 'medium'
        elif not haiku_ok and not sonnet_ok and opus_ok:
            new_difficulty = 'hard'
        else:
            new_difficulty = 'very_hard'

        if old_difficulty != new_difficulty:
            difficulty_changed += 1

        result['difficulty'] = new_difficulty

    # Recalculate summary stats
    difficulties = Counter(r['difficulty'] for r in baseline['results'])
    baseline['difficulty_counts'] = dict(difficulties)

    total = len(baseline['results'])
    baseline['model_accuracy'] = {
        'haiku':
        sum(1 for r in baseline['results']
            if r['models'].get('haiku', {}).get('correct', False)) / total,
        'sonnet':
        sum(1 for r in baseline['results']
            if r['models'].get('sonnet', {}).get('correct', False)) / total,
        'opus':
        sum(1 for r in baseline['results']
            if r['models'].get('opus', {}).get('correct', False)) / total,
    }

    # Add merge metadata
    baseline['metadata'] = baseline.get('metadata', {})
    baseline['metadata']['opus_rerun_merged'] = {
        'timestamp': datetime.now().isoformat(),
        'opus_file': str(opus_path),
        'opus_changed': opus_changed,
        'difficulty_changed': difficulty_changed,
    }

    # Save updated baseline
    with open(output_path, 'w') as f:
        json.dump(baseline, f, indent=2)

    print(f"  Opus answers changed: {opus_changed}")
    print(f"  Difficulty changed: {difficulty_changed}")
    print(f"  Saved to: {output_path}")

    return {
        'opus_changed': opus_changed,
        'difficulty_changed': difficulty_changed,
        'model_accuracy': baseline['model_accuracy'],
        'difficulty_counts': baseline['difficulty_counts'],
    }


def main():
    parser = argparse.ArgumentParser(
        description='Merge Opus results into baselines')
    parser.add_argument('--opus-dir',
                        type=str,
                        default='results/opus_rerun',
                        help='Directory with opus rerun results')
    parser.add_argument('--baseline-dir',
                        type=str,
                        default='lossy_compression/results',
                        help='Directory with baseline results')
    parser.add_argument('--output-dir',
                        type=str,
                        default=None,
                        help='Output directory (default: overwrite baselines)')

    args = parser.parse_args()

    opus_dir = Path(args.opus_dir)
    baseline_dir = Path(args.baseline_dir)
    output_dir = Path(args.output_dir) if args.output_dir else baseline_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MERGING OPUS RESULTS INTO BASELINES")
    print("=" * 60)
    print(f"Opus dir: {opus_dir}")
    print(f"Baseline dir: {baseline_dir}")
    print(f"Output dir: {output_dir}")
    print("=" * 60)

    # Find all opus result files
    opus_files = list(opus_dir.glob("opus_*.json"))
    opus_files = [f for f in opus_files if 'all_datasets' not in f.name]

    if not opus_files:
        print("No opus result files found!")
        return

    summary = {}

    for opus_file in opus_files:
        # Parse dataset name from filename
        # e.g., opus_math_algebra_20260116_150000.json -> math_algebra
        parts = opus_file.stem.split('_')
        if len(parts) >= 3:
            dataset = '_'.join(
                parts[1:-2])  # Remove 'opus_' prefix and timestamp
        else:
            continue

        baseline_file = find_matching_baseline(dataset, baseline_dir)
        if baseline_file:
            output_file = output_dir / baseline_file.name
            result = merge_opus_into_baseline(baseline_file, opus_file,
                                              output_file)
            summary[dataset] = result
        else:
            print(f"\nNo baseline found for: {dataset}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Dataset':<25} | {'Haiku':>8} | {'Sonnet':>8} | {'Opus':>8}")
    print("-" * 60)

    for dataset, result in summary.items():
        acc = result['model_accuracy']
        print(
            f"{dataset:<25} | {100*acc['haiku']:>7.1f}% | {100*acc['sonnet']:>7.1f}% | {100*acc['opus']:>7.1f}%"
        )

    print("-" * 60)
    print("\nDifficulty distribution after merge:")
    print(
        f"{'Dataset':<25} | {'Easy':>8} | {'Medium':>8} | {'Hard':>8} | {'V.Hard':>8}"
    )
    print("-" * 70)

    for dataset, result in summary.items():
        d = result['difficulty_counts']
        total = sum(d.values())
        print(
            f"{dataset:<25} | {d.get('easy',0):>8} | {d.get('medium',0):>8} | {d.get('hard',0):>8} | {d.get('very_hard',0):>8}"
        )


if __name__ == "__main__":
    main()
