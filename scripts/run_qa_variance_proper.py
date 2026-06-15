#!/usr/bin/env python3
"""
QA Variance Computation using the exact run_qa_sweep.py protocol.

Runs K trials of each configuration to compute standard deviations
and confidence intervals. Reuses the same pipeline, prompts, and
answer checking as the main QA sweep, ensuring numbers are comparable.

Configurations:
  BLC  (haiku->haiku->haiku): self-refinement baseline
  QA   (haiku->opus->haiku):  Haiku asks, Opus answers
  QA+  (haiku->opus->opus):   Opus asks, Opus answers

Usage:
    python scripts/run_qa_variance_proper.py --all --trials 3 --parallel 10
    python scripts/run_qa_variance_proper.py --dataset gsm8k --trials 5 --parallel 10
    python scripts/run_qa_variance_proper.py --all --trials 3 --parallel 10 --limit 50

Results saved to: results/qa-variance/{version}/
"""

import glob
import json
import time
import argparse
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Any
import anthropic

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.llm_api import get_anthropic_key
from lossy_compression.core.run_qa_sweep import (
    Problem,
    load_problems,
    run_qa_pipeline_single,
    check_answer,
    MODEL_IDS_new,
    MODEL_IDS_old,
    DIFFICULTIES,
)
import lossy_compression.core.run_qa_sweep as qa_sweep_module

# Configurations matching the paper table
SLM = 'haiku'
CONFIGS = [
    {
        'name': 'BLC',
        'llm': 'haiku',
        'q': 'haiku'
    },
    {
        'name': 'QA',
        'llm': 'opus',
        'q': 'haiku'
    },
    {
        'name': 'QA+',
        'llm': 'opus',
        'q': 'opus'
    },
]

DATASETS = [
    'gsm8k',
    'math_algebra',
    'math_geometry',
    'math_number_theory',
    'gpqa_mc',
    'mbpp',
    'aime',
    'hle',
]


def run_single_trial(
    problems: List[Problem],
    slm: str,
    llm: str,
    q_model: str,
    client: anthropic.Anthropic,
    parallel: int = 1,
) -> List[Dict]:
    """Run one trial of the QA protocol on all problems. Returns per-problem results."""
    results = []

    def process_one(problem):
        state = run_qa_pipeline_single(problem, slm, llm, q_model, client)
        return {
            'idx': problem.idx,
            'difficulty': problem.difficulty,
            'initial_correct': state.initial_correct,
            'final_correct': state.final_correct,
            'error': state.error,
        }

    if parallel > 1:
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {executor.submit(process_one, p): p for p in problems}
            done = 0
            for future in as_completed(futures):
                results.append(future.result())
                done += 1
                if done % 50 == 0:
                    print(f"      {done}/{len(problems)} problems done")
    else:
        for i, p in enumerate(problems):
            results.append(process_one(p))
            if (i + 1) % 50 == 0:
                print(f"      {i+1}/{len(problems)} problems done")

    return results


def find_existing_result(output_dir: Path, dataset: str, version: str,
                         num_trials: int) -> Optional[Path]:
    """Check if a valid result file already exists for this dataset."""
    pattern = str(output_dir / f"{dataset}_variance_{version}_*.json")
    for path in sorted(glob.glob(pattern), reverse=True):  # newest first
        try:
            with open(path) as f:
                data = json.load(f)
            if (data.get('protocol',
                         '') == 'run_qa_sweep (gold-answer, batch questions)'
                    and data.get('num_trials', 0) >= num_trials
                    and len(data.get('combinations', {})) == len(CONFIGS)):
                return Path(path)
        except (json.JSONDecodeError, KeyError):
            continue
    return None


def run_variance_for_dataset(
    dataset: str,
    baseline_dir: Path,
    output_dir: Path,
    num_trials: int = 3,
    parallel: int = 1,
    limit: Optional[int] = None,
    hle_very_hard_limit: Optional[int] = 100,
) -> Optional[Dict]:
    """Run variance experiment for one dataset."""
    version = qa_sweep_module.MODEL_VERSION

    # Resume: skip if valid result already exists
    existing = find_existing_result(output_dir, dataset, version, num_trials)
    if existing:
        print(f"\n  Skipping {dataset}: found existing result at {existing}")
        with open(existing) as f:
            return json.load(f)

    print(f"\n{'='*70}")
    print(f"QA Variance (proper protocol): {dataset} ({version})")
    print(f"Trials: {num_trials}, Parallel: {parallel}")
    print(f"Configs: {', '.join(c['name'] for c in CONFIGS)}")
    print(f"{'='*70}")

    # Load problems (same as main sweep)
    problems = load_problems(
        dataset,
        baseline_dir=baseline_dir,
        hle_very_hard_limit=hle_very_hard_limit,
    )
    if limit:
        problems = problems[:limit]

    if not problems:
        print(f"  No problems found for {dataset}!")
        return None

    print(f"  Problems: {len(problems)}")

    client = anthropic.Anthropic(api_key=get_anthropic_key())

    all_results = {}

    for config in CONFIGS:
        config_name = config['name']
        llm = config['llm']
        q_model = config['q']
        combo_key = f"{config_name}_LLM-{llm}_Q-{q_model}"

        print(f"\n  --- {config_name} (SLM={SLM}, LLM={llm}, Q={q_model}) ---")

        trial_results = []
        trial_accuracies = []

        for trial in range(num_trials):
            t0 = time.time()
            print(f"    Trial {trial + 1}/{num_trials}...")

            results = run_single_trial(problems,
                                       SLM,
                                       llm,
                                       q_model,
                                       client,
                                       parallel=parallel)
            trial_results.append(results)

            # Compute accuracy for this trial
            correct = sum(1 for r in results if r['final_correct'])
            acc = correct / len(problems)
            elapsed = time.time() - t0
            trial_accuracies.append(acc)
            print(
                f"      Result: {correct}/{len(problems)} = {100*acc:.1f}% ({elapsed:.0f}s)"
            )

        mean_acc = float(np.mean(trial_accuracies))
        std_acc = float(np.std(trial_accuracies))
        print(f"    Mean: {100*mean_acc:.1f}% +/- {100*std_acc:.1f}%")

        all_results[combo_key] = {
            'config_name': config_name,
            'slm': SLM,
            'llm': llm,
            'q_model': q_model,
            'n_problems': len(problems),
            'num_trials': num_trials,
            'trial_accuracies': [float(a) for a in trial_accuracies],
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'per_trial': {
                str(t): [{
                    'idx': r['idx'],
                    'difficulty': r['difficulty'],
                    'initial_correct': r['initial_correct'],
                    'final_correct': r['final_correct'],
                    'error': r['error'],
                } for r in trial_results[t]]
                for t in range(num_trials)
            },
        }

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{dataset}_variance_{version}_{timestamp}.json"

    output_data = {
        'dataset': dataset,
        'version': version,
        'slm': SLM,
        'configs': [c['name'] for c in CONFIGS],
        'num_trials': num_trials,
        'n_problems': len(problems),
        'hle_very_hard_limit': hle_very_hard_limit,
        'timestamp': timestamp,
        'protocol': 'run_qa_sweep (gold-answer, batch questions)',
        'combinations': all_results,
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\n  Saved: {output_file}")

    # Summary table
    print(f"\n  {'Config':<8} {'LLM':<8} {'Q':<8} {'Mean':>8} {'Std':>8}")
    print(f"  {'-'*44}")
    for combo_key, result in all_results.items():
        print(
            f"  {result['config_name']:<8} {result['llm']:<8} {result['q_model']:<8} "
            f"{100*result['mean_accuracy']:>6.1f}% {100*result['std_accuracy']:>6.1f}%"
        )

    return output_data


def main():
    parser = argparse.ArgumentParser(
        description='QA variance using the exact run_qa_sweep.py protocol')
    parser.add_argument('--dataset',
                        type=str,
                        choices=DATASETS,
                        help='Single dataset to evaluate')
    parser.add_argument('--all', action='store_true', help='Run all datasets')
    parser.add_argument('--trials',
                        type=int,
                        default=3,
                        help='Number of trials per configuration (default: 3)')
    parser.add_argument('--parallel',
                        type=int,
                        default=1,
                        help='Parallel workers per trial')
    parser.add_argument('--limit',
                        type=int,
                        default=None,
                        help='Limit problems per dataset (for testing)')
    parser.add_argument(
        '--hle-very-hard-limit',
        type=int,
        default=100,
        help='Max very-hard HLE problems (default: 100, matching paper)')
    parser.add_argument('--baseline-dir',
                        type=str,
                        default=None,
                        help='Baseline directory override')
    parser.add_argument('--output-dir',
                        type=str,
                        default=None,
                        help='Output directory override')
    parser.add_argument('--use-old-models',
                        action='store_true',
                        help='Use Claude 3.5/4 models instead of 4.5')
    args = parser.parse_args()

    # Set model version
    if args.use_old_models:
        qa_sweep_module.MODEL_IDS = MODEL_IDS_old
        qa_sweep_module.MODEL_VERSION = 'v3.5'
    else:
        qa_sweep_module.MODEL_IDS = MODEL_IDS_new
        qa_sweep_module.MODEL_VERSION = 'v4.5'

    version = qa_sweep_module.MODEL_VERSION
    baseline_dir = Path(args.baseline_dir) if args.baseline_dir else Path(
        f'results/model-baselines/{version}')
    output_dir = Path(args.output_dir) if args.output_dir else Path(
        f'results/qa-variance/{version}')

    if args.all:
        datasets = DATASETS
    elif args.dataset:
        datasets = [args.dataset]
    else:
        parser.error("Specify --dataset or --all")

    print(f"QA Variance Experiment (proper protocol)")
    print(f"Version: {version}")
    print(f"Trials: {args.trials}")
    print(f"SLM: {SLM}")
    print(f"Configs:")
    for c in CONFIGS:
        print(f"  {c['name']}: SLM={SLM}, LLM={c['llm']}, Q={c['q']}")
    print(f"Datasets: {datasets}")
    print(f"Parallel: {args.parallel}")
    print(f"HLE very-hard limit: {args.hle_very_hard_limit}")
    print(f"Baseline: {baseline_dir}")
    print(f"Output: {output_dir}")

    for dataset in datasets:
        try:
            run_variance_for_dataset(
                dataset=dataset,
                baseline_dir=baseline_dir,
                output_dir=output_dir,
                num_trials=args.trials,
                parallel=args.parallel,
                limit=args.limit,
                hle_very_hard_limit=args.hle_very_hard_limit,
            )
        except Exception as e:
            print(f"\nERROR on {dataset}: {e}")
            import traceback
            traceback.print_exc()

    print("\nDone!")


if __name__ == '__main__':
    main()
