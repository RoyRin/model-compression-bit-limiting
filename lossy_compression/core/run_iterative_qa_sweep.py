#!/usr/bin/env python3
"""
Iterative QA Compression Sweep with Judge-Based Thresholding.

Runs the iterative QA protocol where:
1. SLM generates initial response
2. LLM generates its own response (used as reference)
3. LLM grades SLM's response quality (1-10 scale)
4. If below threshold: generate 5 questions, LLM answers, SLM updates
5. Repeat (2 rounds × 5 questions = 10 questions max)

Key differences from run_qa_sweep.py (gold-answer ablation):
- LLM uses its OWN solution as reference (not gold answer)
- Quality thresholding with judge (can stop early if quality >= 7)
- Questions generated in 2 batches of 5 (not 1 batch of 10)
- Temperature = 0 for all API calls (deterministic)
- Only runs 2 configurations: BLC and QA

Configurations:
- BLC (Bit-Limited COT): haiku→haiku→haiku (self-refinement baseline)
- QA: haiku→opus→haiku (knowledge transfer from Opus)

Usage:
    python lossy_compression/core/run_iterative_qa_sweep.py --all --parallel 10
    python lossy_compression/core/run_iterative_qa_sweep.py --dataset gsm8k --parallel 10
    python lossy_compression/core/run_iterative_qa_sweep.py --dataset math --subject algebra
    python lossy_compression/core/run_iterative_qa_sweep.py --all --limit 5  # quick test
"""

import json
import time
import argparse
import sys
import threading
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Reuse problem loading, answer checking, and prompts from existing sweep
from lossy_compression.core.run_qa_sweep import (
    load_problems,
    check_answer,
    Problem,
    get_system_prompt,
    make_proposal_prompt,
    MODEL_IDS_new,
    MODEL_IDS_old,
    BASELINE_PATTERNS,
    find_baseline_file,
    DIFFICULTIES,
)
# Reuse the iterative QA protocol
from lossy_compression.core.qa_compression import (
    iterative_SLM_loop,
    EVAL_MODE_DEFAULT,
    EVAL_MODE_MATH,
    EVAL_MODE_CODE,
    EVAL_MODE_SCIENCE,
    JUDGE_MODE_COMPARISON,
    JUDGE_MODE_OBJECTIVE,
)

# =============================================================================
# Configuration
# =============================================================================

MODEL_IDS = MODEL_IDS_new  # Default to 4.5 models
MODEL_VERSION = "v4.5"

# Only two configurations
CONFIGS = [
    {
        'name': 'BLC',
        'slm': 'haiku',
        'llm': 'haiku',
        'q': 'haiku'
    },
    {
        'name': 'QA',
        'slm': 'haiku',
        'llm': 'opus',
        'q': 'haiku'
    },
]

# Map datasets to evaluation modes
EVAL_MODES = {
    'gsm8k': EVAL_MODE_MATH,
    'math_algebra': EVAL_MODE_MATH,
    'math_geometry': EVAL_MODE_MATH,
    'math_number_theory': EVAL_MODE_MATH,
    'gpqa_mc': EVAL_MODE_SCIENCE,
    'gpqa_freeform': EVAL_MODE_SCIENCE,
    'mbpp': EVAL_MODE_CODE,
    'aime': EVAL_MODE_MATH,
    'hle': EVAL_MODE_DEFAULT,
}

# Protocol parameters
MAX_QUESTIONS = 10  # Total questions (2 rounds × 5)
BATCH_SIZE = 5  # Questions per round
QUALITY_THRESHOLD = 7  # Stop if quality >= 7/10
LLM_TEMPERATURE = 0.0  # Deterministic
SLM_TEMPERATURE = 0.0  # Deterministic

ALL_DATASETS = [
    'gsm8k',
    'math_algebra',
    'math_geometry',
    'math_number_theory',
    'gpqa_mc',
    'mbpp',
    'aime',
    'hle',
]

# =============================================================================
# Single problem runner
# =============================================================================


def run_single_problem(problem: Problem,
                       config: Dict,
                       eval_mode: str,
                       system_prompt: str,
                       judge_mode: str = JUDGE_MODE_COMPARISON,
                       quality_threshold: int = 7,
                       gold_judge: bool = False) -> Dict:
    """Run iterative QA on a single problem with one configuration."""

    slm_model_id = MODEL_IDS[config['slm']]
    llm_model_id = MODEL_IDS[config['llm']]
    q_model_id = MODEL_IDS[config['q']]

    # Build the prompt (same as run_qa_sweep.py)
    prompt = make_proposal_prompt(problem)

    try:
        best_answer, qa_pairs, metrics = iterative_SLM_loop(
            prompt=prompt,
            system_prompt=system_prompt,
            large_model_name=llm_model_id,
            small_model_name=slm_model_id,
            question_model_name=q_model_id,
            max_iterations=MAX_QUESTIONS,
            quality_threshold=quality_threshold,
            llm_temperature=LLM_TEMPERATURE,
            slm_temperature=SLM_TEMPERATURE,
            batch_mode=True,
            batch_size=BATCH_SIZE,
            evaluation_mode=eval_mode,
            gold_answer=problem.gold_answer
            if gold_judge else None,  # gold_judge: judge gets gold answer
            judge_mode=judge_mode,
            verbose=False,
        )

        guiding_questions, guiding_answers = qa_pairs
        final_correct = check_answer(problem, best_answer)

        return {
            'idx':
            problem.idx,
            'difficulty':
            problem.difficulty,
            'initial_correct':
            problem.baseline_correct.get(config['slm'], False),
            'final_correct':
            final_correct,
            'questions':
            guiding_questions,
            'answers': [str(a) for a in guiding_answers],
            'quality_scores':
            metrics.get('quality_scores', []),
            'n_questions_used':
            metrics.get('total_bits_uniform', len(guiding_questions)),
            'early_stopped':
            metrics.get('quality_scores', [None])[-1] is not None
            and len(metrics.get('quality_scores', [])) > 1
            and metrics['quality_scores'][-1] >= quality_threshold,
            'error':
            None,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            'idx': problem.idx,
            'difficulty': problem.difficulty,
            'initial_correct':
            problem.baseline_correct.get(config['slm'], False),
            'final_correct': False,
            'questions': [],
            'answers': [],
            'quality_scores': [],
            'n_questions_used': 0,
            'early_stopped': False,
            'error': str(e),
        }


# =============================================================================
# Dataset runner
# =============================================================================


def run_config_on_dataset(
    problems: List[Problem],
    config: Dict,
    dataset: str,
    output_dir: Path,
    parallel_workers: int = 1,
    judge_mode: str = JUDGE_MODE_COMPARISON,
    quality_threshold: int = 7,
    gold_judge: bool = False,
    easy_only: bool = False,
) -> Dict[str, Any]:
    """Run one config (BLC or QA) on all problems for a dataset."""

    config_name = config['name']
    eval_mode = EVAL_MODES.get(dataset, EVAL_MODE_DEFAULT)
    system_prompt = get_system_prompt(dataset, 'slm')

    print(f"\n{'='*60}")
    print(
        f"  {config_name}: SLM={config['slm']}, LLM={config['llm']}, Q={config['q']}"
    )
    print(f"  Dataset: {dataset} ({len(problems)} problems)")
    print(f"  Eval mode: {eval_mode}, Threshold: {quality_threshold}")
    print(f"  Judge mode: {judge_mode}, Gold judge: {gold_judge}")
    print(f"  Batch size: {BATCH_SIZE}, Max questions: {MAX_QUESTIONS}")
    print(f"  Temperature: LLM={LLM_TEMPERATURE}, SLM={SLM_TEMPERATURE}")
    print(f"  Parallel workers: {parallel_workers}")
    print(f"{'='*60}")

    start_time = time.time()
    results = []
    lock = threading.Lock()
    completed = [0]
    errors = [0]

    def process_one(problem):
        result = run_single_problem(problem,
                                    config,
                                    eval_mode,
                                    system_prompt,
                                    judge_mode=judge_mode,
                                    quality_threshold=quality_threshold,
                                    gold_judge=gold_judge)
        with lock:
            completed[0] += 1
            if result['error']:
                errors[0] += 1
            status = "OK" if result['final_correct'] else (
                "ERR" if result['error'] else "WRONG")
            print(f"  [{completed[0]}/{len(problems)}] "
                  f"Problem {problem.idx} ({problem.difficulty}): {status} "
                  f"(Qs={result['n_questions_used']}, "
                  f"scores={result['quality_scores']})")
        return result

    if parallel_workers > 1:
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            futures = {executor.submit(process_one, p): p for p in problems}
            for future in as_completed(futures):
                results.append(future.result())
    else:
        for p in problems:
            results.append(process_one(p))

    elapsed = time.time() - start_time

    # Compute summary statistics
    n = len(results)
    initial_correct = sum(1 for r in results if r['initial_correct'])
    final_correct = sum(1 for r in results if r['final_correct'])
    recovered = sum(1 for r in results
                    if not r['initial_correct'] and r['final_correct'])
    regressed = sum(1 for r in results
                    if r['initial_correct'] and not r['final_correct'])
    early_stopped = sum(1 for r in results if r['early_stopped'])
    n_errors = sum(1 for r in results if r['error'])

    # Per-difficulty breakdown
    difficulty_stats = {}
    for diff in DIFFICULTIES:
        diff_results = [r for r in results if r['difficulty'] == diff]
        if diff_results:
            diff_init = sum(1 for r in diff_results if r['initial_correct'])
            diff_final = sum(1 for r in diff_results if r['final_correct'])
            diff_recovered = sum(
                1 for r in diff_results
                if not r['initial_correct'] and r['final_correct'])
            diff_n = len(diff_results)
            # Recovery rate: of those initially wrong, how many became correct
            initially_wrong = diff_n - diff_init
            recovery_rate = diff_recovered / initially_wrong if initially_wrong > 0 else 0.0
            difficulty_stats[diff] = {
                'n': diff_n,
                'initial_correct': diff_init,
                'final_correct': diff_final,
                'recovered': diff_recovered,
                'recovery_rate': recovery_rate,
            }

    # Overall recovery rate
    initially_wrong_total = n - initial_correct
    recovery_rate_total = recovered / initially_wrong_total if initially_wrong_total > 0 else 0.0

    summary = {
        'config': config_name,
        'slm': config['slm'],
        'llm': config['llm'],
        'q_model': config['q'],
        'dataset': dataset,
        'n_problems': n,
        'initial_correct': initial_correct,
        'final_correct': final_correct,
        'recovered': recovered,
        'regressed': regressed,
        'recovery_rate': recovery_rate_total,
        'early_stopped': early_stopped,
        'errors': n_errors,
        'time_seconds': elapsed,
        'difficulty_breakdown': difficulty_stats,
    }

    print(f"\n--- {config_name} Results ({dataset}) ---")
    print(f"  Initial: {initial_correct}/{n} ({100*initial_correct/n:.1f}%)")
    print(f"  Final:   {final_correct}/{n} ({100*final_correct/n:.1f}%)")
    print(f"  Recovered: {recovered}, Regressed: {regressed}")
    print(f"  Recovery rate: {100*recovery_rate_total:.1f}% "
          f"({recovered}/{initially_wrong_total} initially wrong)")
    print(f"  Early stopped: {early_stopped}/{n}")
    if n_errors:
        print(f"  Errors: {n_errors}")
    print(f"  Time: {int(elapsed)}s")

    for diff in DIFFICULTIES:
        if diff in difficulty_stats:
            ds = difficulty_stats[diff]
            print(
                f"    {diff}: {ds['recovered']}/{ds['n'] - ds['initial_correct']} recovered "
                f"({100*ds['recovery_rate']:.1f}%)")

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    # Build filename with ablation suffixes
    ablation_suffix = ""
    if easy_only:
        ablation_suffix += "_easy"
    if quality_threshold != 7:
        ablation_suffix += f"_t{quality_threshold}"
    if gold_judge:
        ablation_suffix += "_goldjudge"
    output_file = output_dir / f"{dataset}_{config_name}_{judge_mode}{ablation_suffix}_{MODEL_VERSION}_{ts}.json"

    output_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'dataset': dataset,
            'model_version': MODEL_VERSION,
            'model_ids': MODEL_IDS,
            'judge_mode': judge_mode,
            'protocol': {
                'max_questions':
                MAX_QUESTIONS,
                'batch_size':
                BATCH_SIZE,
                'quality_threshold':
                quality_threshold,
                'llm_temperature':
                LLM_TEMPERATURE,
                'slm_temperature':
                SLM_TEMPERATURE,
                'gold_judge':
                gold_judge,
                'easy_only':
                easy_only,
                'judge_mode':
                judge_mode,
                'description':
                f'Iterative QA with LLM-as-judge thresholding ({judge_mode} mode). '
                + ('Easy problems only. ' if easy_only else '') +
                ('Judge given gold answer. ' if gold_judge else '') +
                'LLM uses its own solution as reference for answering questions.',
            },
        },
        'summary': summary,
        'problems': sorted(results, key=lambda r: r['idx']),
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"  Saved: {output_file}")

    return summary


def run_dataset(dataset: str,
                output_dir: Path,
                baseline_dir: Optional[Path],
                parallel_workers: int,
                limit: Optional[int],
                hle_very_hard_limit: Optional[int],
                judge_mode: str = JUDGE_MODE_COMPARISON,
                quality_threshold: int = 7,
                gold_judge: bool = False,
                easy_only: bool = False) -> Dict:
    """Run both BLC and QA configs on a dataset."""

    mode_label = "easy" if easy_only else "non-easy"
    print(f"\n{'#'*60}")
    print(f"# Dataset: {dataset} ({mode_label} problems)")
    print(f"{'#'*60}")

    difficulties = ['easy'] if easy_only else None
    problems = load_problems(dataset,
                             baseline_dir=baseline_dir,
                             hle_very_hard_limit=hle_very_hard_limit,
                             difficulties=difficulties)
    if limit:
        problems = problems[:limit]

    print(f"Loaded {len(problems)} {mode_label} problems")

    if not problems:
        print("No problems to run!")
        return {}

    all_summaries = {}
    for config in CONFIGS:
        config_name = config['name']

        # Check if result already exists
        ablation_suffix = ""
        if easy_only:
            ablation_suffix += "_easy"
        if quality_threshold != 7:
            ablation_suffix += f"_t{quality_threshold}"
        if gold_judge:
            ablation_suffix += "_goldjudge"
        existing = list(
            output_dir.glob(
                f"{dataset}_{config_name}_{judge_mode}{ablation_suffix}_{MODEL_VERSION}_*.json"
            ))
        if existing:
            print(f"\nSKIP {config_name} (exists): {existing[-1].name}")
            with open(existing[-1]) as f:
                data = json.load(f)
            all_summaries[config_name] = data.get('summary', {})
            continue

        summary = run_config_on_dataset(
            problems,
            config,
            dataset,
            output_dir,
            parallel_workers=parallel_workers,
            judge_mode=judge_mode,
            quality_threshold=quality_threshold,
            gold_judge=gold_judge,
            easy_only=easy_only,
        )
        all_summaries[config_name] = summary

    return all_summaries


# =============================================================================
# Main
# =============================================================================


def main():
    global MODEL_IDS, MODEL_VERSION

    parser = argparse.ArgumentParser(
        description='Run Iterative QA Sweep with Judge-Based Thresholding')
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        choices=['gsm8k', 'math', 'gpqa', 'mbpp', 'aime', 'hle'],
        help='Dataset to run')
    parser.add_argument(
        '--subject',
        type=str,
        default='all',
        choices=['all', 'algebra', 'geometry', 'number_theory'],
        help='MATH subject')
    parser.add_argument('--format',
                        type=str,
                        default='mc',
                        choices=['mc', 'freeform'],
                        help='GPQA format')
    parser.add_argument('--all', action='store_true', help='Run all datasets')
    parser.add_argument('--limit',
                        type=int,
                        default=None,
                        help='Limit problems per dataset (for testing)')
    parser.add_argument('--hle-very-hard-limit',
                        type=int,
                        default=None,
                        help='Limit very_hard problems for HLE')
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help=
        'Output directory (default: results/iterative-qa-sweep/MODEL_VERSION)')
    parser.add_argument('--baseline-dir',
                        type=str,
                        default=None,
                        help='Directory containing baseline files')
    parser.add_argument('--parallel',
                        type=int,
                        default=1,
                        help='Number of parallel workers (default: 1)')
    parser.add_argument(
        '--judge-mode',
        type=str,
        default='comparison',
        choices=['comparison', 'objective'],
        help=
        'Judge mode: "comparison" = judge compares SLM answer vs LLM answer; '
        '"objective" = standalone quality evaluation (default: comparison)')
    parser.add_argument(
        '--quality-threshold',
        type=int,
        default=7,
        help='Quality threshold for judge early stopping (default: 7). '
        'If judge score >= threshold, accept answer and stop.')
    parser.add_argument('--gold-judge',
                        action='store_true',
                        help='Give the judge the gold answer for evaluation. '
                        'LLM still answers questions using its own solution.')
    parser.add_argument(
        '--easy-only',
        action='store_true',
        help='Run on easy problems only (all models correct at baseline). '
        'Useful for measuring regression rates with judge.')
    parser.add_argument(
        '--use-old-models',
        action='store_true',
        help='Use old model versions (3.5 haiku, sonnet 4, opus 4)')

    args = parser.parse_args()

    # Set model versions
    if args.use_old_models:
        MODEL_IDS = MODEL_IDS_old
        MODEL_VERSION = "v3.5"
        print(
            "Using OLD models: claude-3-5-haiku, claude-sonnet-4, claude-opus-4"
        )
    else:
        MODEL_IDS = MODEL_IDS_new
        MODEL_VERSION = "v4.5"
        print(
            "Using 4.5 models: claude-haiku-4-5, claude-sonnet-4-5, claude-opus-4-5"
        )

    # Output directory
    output_dir = Path(args.output_dir) if args.output_dir else Path(
        f'results/iterative-qa-sweep/{MODEL_VERSION}')
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    # Baseline directory
    baseline_dir = Path(args.baseline_dir) if args.baseline_dir else None
    if baseline_dir:
        print(f"Baselines: {baseline_dir}")

    # Determine datasets
    datasets_to_run = []
    if args.all:
        datasets_to_run = ALL_DATASETS
    elif args.dataset == 'gsm8k':
        datasets_to_run = ['gsm8k']
    elif args.dataset == 'math':
        if args.subject == 'all':
            datasets_to_run = [
                'math_algebra', 'math_geometry', 'math_number_theory'
            ]
        else:
            datasets_to_run = [f'math_{args.subject}']
    elif args.dataset == 'gpqa':
        datasets_to_run = [f'gpqa_{args.format}']
    elif args.dataset == 'mbpp':
        datasets_to_run = ['mbpp']
    elif args.dataset == 'aime':
        datasets_to_run = ['aime']
    elif args.dataset == 'hle':
        datasets_to_run = ['hle']

    if not datasets_to_run:
        parser.error("No datasets specified. Use --all or --dataset")

    judge_mode = args.judge_mode
    quality_threshold = args.quality_threshold
    gold_judge = args.gold_judge
    easy_only = args.easy_only

    print(f"\nProtocol: Iterative QA with judge-based thresholding")
    print(f"  Max questions: {MAX_QUESTIONS} (in batches of {BATCH_SIZE})")
    print(f"  Quality threshold: {quality_threshold}/10")
    print(f"  Judge mode: {judge_mode}")
    print(f"  Temperature: LLM={LLM_TEMPERATURE}, SLM={SLM_TEMPERATURE}")
    print(
        f"  Gold answer for judge: {'YES' if gold_judge else 'NO'} (LLM always uses own solution for Q&A)"
    )
    print(f"  Easy only: {'YES' if easy_only else 'NO'}")
    print(f"  Configs: {', '.join(c['name'] for c in CONFIGS)}")
    print(f"  Datasets: {', '.join(datasets_to_run)}")
    print(f"  Parallel: {args.parallel} workers")

    # Run
    all_results = {}
    for dataset in datasets_to_run:
        try:
            summaries = run_dataset(
                dataset,
                output_dir,
                baseline_dir,
                parallel_workers=args.parallel,
                limit=args.limit,
                hle_very_hard_limit=args.hle_very_hard_limit,
                judge_mode=judge_mode,
                quality_threshold=quality_threshold,
                gold_judge=gold_judge,
                easy_only=easy_only,
            )
            all_results[dataset] = summaries
        except Exception as e:
            print(f"\nERROR on {dataset}: {e}")
            import traceback
            traceback.print_exc()

    # Print final summary table
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY")
    print(f"{'='*80}")
    print(
        f"{'Dataset':<20s} {'n':>4s}  {'BLC Recovery':>14s}  {'QA Recovery':>14s}  {'Delta':>8s}"
    )
    print('-' * 70)

    for dataset in datasets_to_run:
        if dataset not in all_results:
            continue
        summaries = all_results[dataset]
        blc = summaries.get('BLC', {})
        qa = summaries.get('QA', {})

        n = blc.get('n_problems', qa.get('n_problems', 0))
        blc_rate = blc.get('recovery_rate', 0)
        qa_rate = qa.get('recovery_rate', 0)
        delta = qa_rate - blc_rate

        print(
            f"{dataset:<20s} {n:>4d}  "
            f"{100*blc_rate:>5.1f}% ({blc.get('recovered', 0):>3d}/{n - blc.get('initial_correct', 0):>3d})  "
            f"{100*qa_rate:>5.1f}% ({qa.get('recovered', 0):>3d}/{n - qa.get('initial_correct', 0):>3d})  "
            f"{100*delta:>+6.1f}%")

    print(f"\nDone! Results in: {output_dir}")


if __name__ == '__main__':
    main()
