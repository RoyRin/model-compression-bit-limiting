#!/usr/bin/env python3
"""
Best-of-N + QA Compression Sweep.

Protocol:
1. SLM generates N candidates (temperature > 0 for diversity)
2. LLM evaluates candidates using gold answer:
   - If one is correct → select it (0 QA bits)
   - If none correct → pick the best, then run normal QA (10 questions)
3. Check final answer against gold

This is designed to reduce regression on easy problems (LLM can identify
the correct candidate) while still improving on hard problems (QA fallback).

Only runs haiku→haiku→opus (SLM=haiku, Q=haiku, LLM=opus).

Usage:
    # Quick test
    python lossy_compression/core/run_bestofn_qa_sweep.py --dataset gsm8k --limit 5

    # Full run on all datasets
    python lossy_compression/core/run_bestofn_qa_sweep.py --all --parallel 6

    # Easy-only regression test
    python lossy_compression/core/run_bestofn_qa_sweep.py --all --easy-only --parallel 6
"""

import json
import time
import argparse
import re
import sys
import threading
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import anthropic
from utils.llm_api import get_anthropic_key
import lossy_compression.core.run_qa_sweep as qa_sweep_module
from lossy_compression.core.run_qa_sweep import (
    load_problems,
    check_answer,
    Problem,
    get_system_prompt,
    make_proposal_prompt,
    make_question_prompt,
    make_answer_prompt,
    make_update_prompt,
    get_max_tokens,
    MODEL_IDS_new,
    MODEL_IDS_old,
    BASELINE_PATTERNS,
    find_baseline_file,
    DIFFICULTIES,
    NUM_QUESTIONS,
    MAX_RETRIES,
    INITIAL_BACKOFF,
    MAX_BACKOFF,
)

# =============================================================================
# Configuration
# =============================================================================

MODEL_IDS = MODEL_IDS_new
MODEL_VERSION = "v4.5"

N_CANDIDATES = 4  # Number of SLM candidates to generate
CANDIDATE_TEMPERATURE = 0.7  # Temperature for diverse candidates
LLM_TEMPERATURE = 0.0  # Deterministic LLM

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
# API call with temperature support
# =============================================================================


def call_api(client: anthropic.Anthropic,
             model: str,
             messages: List[Dict],
             system: str = None,
             max_tokens: int = 1024,
             temperature: float = None) -> str:
    """Make a single API call with retry logic and optional temperature."""
    model_id = MODEL_IDS.get(model, model)

    for attempt in range(MAX_RETRIES):
        try:
            kwargs = {
                'model': model_id,
                'max_tokens': max_tokens,
                'messages': messages,
            }
            if system:
                kwargs['system'] = system
            if temperature is not None:
                kwargs['temperature'] = temperature

            response = client.messages.create(**kwargs)
            return response.content[0].text if response.content else ""

        except (anthropic.RateLimitError, anthropic.APIConnectionError) as e:
            wait_time = min(INITIAL_BACKOFF * (2**attempt), MAX_BACKOFF)
            time.sleep(wait_time)
        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                wait_time = min(INITIAL_BACKOFF * (2**attempt), MAX_BACKOFF)
                time.sleep(wait_time)
            else:
                raise

    return ""


# =============================================================================
# LLM Selection Prompt
# =============================================================================


def make_selection_prompt(problem: Problem,
                          candidates: List[str],
                          reference_answer: str = None) -> str:
    """Create prompt for LLM to select best candidate.

    Args:
        reference_answer: The answer to compare against. If None, uses problem.gold_answer.
    """
    ref = reference_answer if reference_answer is not None else problem.gold_answer
    candidates_text = ""
    for i, c in enumerate(candidates):
        # Truncate very long candidates for the selection prompt
        truncated = c[:3000] if len(c) > 3000 else c
        candidates_text += f"\n--- Candidate {i+1} ---\n{truncated}\n"

    return f"""You are evaluating candidate solutions to a problem. The correct answer is provided.

Problem: {problem.question}

Correct answer: {ref}

Here are {len(candidates)} candidate solutions:
{candidates_text}
---

First, determine if any candidate arrives at the correct answer. If so, respond with:
CORRECT: <number>

If none is correct, pick the one that is closest to correct or most promising to fix. Respond with:
BEST: <number>

Respond with ONLY one line: either "CORRECT: <number>" or "BEST: <number>" where <number> is 1-{len(candidates)}."""


def make_answer_prompt_with_ref(problem: Problem, question: str,
                                reference_answer: str) -> str:
    """Create prompt for LLM to answer a question using a reference answer."""
    return f"""Problem: {problem.question}

Correct answer: {reference_answer}

Question: {question}

Answer with only "Yes" or "No"."""


def parse_selection_response(response: str, n_candidates: int) -> tuple:
    """Parse the LLM's selection response.

    Returns:
        (selected_idx, is_correct): 0-based index and whether LLM thinks it's correct
    """
    response = response.strip().upper()

    # Try CORRECT: N
    match = re.search(r'CORRECT:\s*(\d+)', response)
    if match:
        idx = int(match.group(1)) - 1  # Convert to 0-based
        if 0 <= idx < n_candidates:
            return idx, True

    # Try BEST: N
    match = re.search(r'BEST:\s*(\d+)', response)
    if match:
        idx = int(match.group(1)) - 1
        if 0 <= idx < n_candidates:
            return idx, False

    # Fallback: try to find any number
    match = re.search(r'(\d+)', response)
    if match:
        idx = int(match.group(1)) - 1
        if 0 <= idx < n_candidates:
            return idx, False

    # Default to first candidate
    return 0, False


# =============================================================================
# Single problem runner
# =============================================================================


def run_single_problem(problem: Problem,
                       client: anthropic.Anthropic,
                       n_candidates: int = N_CANDIDATES,
                       use_gold: bool = True) -> Dict:
    """Run best-of-N + QA pipeline for a single problem.

    Args:
        use_gold: If True, LLM uses gold answer for selection and QA.
                  If False, LLM generates its own solution and uses that as reference.
    """

    slm = 'haiku'
    llm = 'opus'
    q_model = 'haiku'
    slm_id = MODEL_IDS[slm]
    llm_id = MODEL_IDS[llm]
    q_id = MODEL_IDS[q_model]

    system = get_system_prompt(problem.dataset, 'slm')
    proposal_prompt = make_proposal_prompt(problem)

    result = {
        'idx': problem.idx,
        'difficulty': problem.difficulty,
        'initial_correct': problem.baseline_correct.get(slm, False),
        'n_candidates': n_candidates,
        'use_gold': use_gold,
        'candidates_generated': 0,
        'selected_idx': -1,
        'llm_said_correct': False,
        'actually_correct_after_selection': False,
        'qa_used': False,
        'n_questions_used': 0,
        'final_correct': False,
        'error': None,
    }

    try:
        # =====================================================================
        # Phase 0 (no-gold only): Generate LLM's own solution as reference
        # =====================================================================
        if use_gold:
            reference_answer = problem.gold_answer
        else:
            llm_system = get_system_prompt(problem.dataset, 'slm')
            llm_solution = call_api(
                client,
                llm_id,
                [{
                    'role': 'user',
                    'content': proposal_prompt
                }],
                system=llm_system,
                max_tokens=get_max_tokens(llm, 'proposal'),
                temperature=LLM_TEMPERATURE,
            )
            reference_answer = llm_solution

        # =====================================================================
        # Phase 1: Generate N candidates from SLM
        # =====================================================================
        candidates = []
        for i in range(n_candidates):
            candidate = call_api(
                client,
                slm_id,
                [{
                    'role': 'user',
                    'content': proposal_prompt
                }],
                system=system,
                max_tokens=get_max_tokens(slm, 'proposal'),
                temperature=CANDIDATE_TEMPERATURE,
            )
            candidates.append(candidate)
        result['candidates_generated'] = len(candidates)

        # =====================================================================
        # Phase 2: LLM selects best candidate
        # =====================================================================
        selection_prompt = make_selection_prompt(
            problem, candidates, reference_answer=reference_answer)
        selection_response = call_api(
            client,
            llm_id,
            [{
                'role': 'user',
                'content': selection_prompt
            }],
            max_tokens=50,
            temperature=LLM_TEMPERATURE,
        )

        selected_idx, llm_said_correct = parse_selection_response(
            selection_response, len(candidates))
        result['selected_idx'] = selected_idx
        result['llm_said_correct'] = llm_said_correct

        selected_answer = candidates[selected_idx]

        # Check if selected answer is actually correct (always against gold)
        actually_correct = check_answer(problem, selected_answer)
        result['actually_correct_after_selection'] = actually_correct

        # =====================================================================
        # Phase 3: If LLM says correct → done; otherwise → QA
        # =====================================================================
        if llm_said_correct:
            # LLM thinks this candidate is correct — use it directly
            result['qa_used'] = False
            result['n_questions_used'] = 0
            result['final_correct'] = actually_correct
        else:
            # None correct — run QA from the selected (best) candidate
            result['qa_used'] = True

            # Generate questions about the selected candidate
            question_prompt = make_question_prompt(problem, selected_answer)
            questions_response = call_api(
                client,
                q_id,
                [{
                    'role': 'user',
                    'content': question_prompt
                }],
                max_tokens=get_max_tokens(q_model, 'questions'),
                temperature=LLM_TEMPERATURE,
            )

            questions = re.findall(r'\d+\.\s*(.+?)(?=\n\d+\.|\Z)',
                                   questions_response, re.DOTALL)
            questions = [q.strip() for q in questions[:NUM_QUESTIONS]]
            while len(questions) < NUM_QUESTIONS:
                questions.append("Is the answer correct?")

            # LLM answers questions using reference answer
            answers = []
            for q in questions:
                if use_gold:
                    ap = make_answer_prompt(problem, q)
                else:
                    ap = make_answer_prompt_with_ref(problem, q,
                                                     reference_answer)
                resp = call_api(
                    client,
                    llm_id,
                    [{
                        'role': 'user',
                        'content': ap
                    }],
                    max_tokens=get_max_tokens(llm, 'answer'),
                    temperature=LLM_TEMPERATURE,
                )
                response_upper = resp.strip().upper()
                if response_upper.startswith('YES'):
                    answers.append('Yes')
                elif response_upper.startswith('NO'):
                    answers.append('No')
                else:
                    answers.append('Unknown')

            result['n_questions_used'] = len(questions)
            result['questions'] = questions
            result['answers'] = answers

            # SLM updates based on Q&A (starting from selected candidate)
            update_prompt = make_update_prompt(problem, questions, answers)
            final_reasoning = call_api(
                client,
                slm_id,
                [{
                    'role': 'user',
                    'content': update_prompt
                }],
                system=system,
                max_tokens=get_max_tokens(slm, 'update'),
                temperature=LLM_TEMPERATURE,
            )

            result['final_correct'] = check_answer(problem, final_reasoning)

    except Exception as e:
        import traceback
        traceback.print_exc()
        result['error'] = str(e)

    return result


# =============================================================================
# Dataset runner
# =============================================================================


def run_dataset(dataset: str,
                output_dir: Path,
                baseline_dir: Optional[Path],
                parallel_workers: int,
                limit: Optional[int],
                easy_only: bool = False,
                n_candidates: int = N_CANDIDATES,
                use_gold: bool = True) -> Dict:
    """Run best-of-N + QA on a dataset."""

    ref_label = "gold" if use_gold else "nogold"
    mode_label = "easy" if easy_only else "all"
    print(f"\n{'#'*60}")
    print(f"# Dataset: {dataset} ({mode_label} problems, ref={ref_label})")
    print(f"# Protocol: Best-of-{n_candidates} + QA (haiku→haiku→opus)")
    print(f"{'#'*60}")

    # Load problems — include ALL difficulties so we can measure both
    # regression (easy) and recovery (non-easy)
    if easy_only:
        difficulties = ['easy']
    else:
        difficulties = ['easy', 'medium', 'hard', 'very_hard']

    problems = load_problems(dataset,
                             baseline_dir=baseline_dir,
                             difficulties=difficulties)
    if limit:
        problems = problems[:limit]

    print(f"Loaded {len(problems)} problems")
    if not problems:
        print("No problems to run!")
        return {}

    # Check for existing results
    suffix = "_easy" if easy_only else ""
    gold_suffix = "" if use_gold else "_nogold"
    existing = list(
        output_dir.glob(
            f"{dataset}_bestof{n_candidates}_qa{suffix}{gold_suffix}_{MODEL_VERSION}_*.json"
        ))
    if existing:
        print(f"\nSKIP (exists): {existing[-1].name}")
        with open(existing[-1]) as f:
            data = json.load(f)
        return data.get('summary', {})

    client = anthropic.Anthropic(api_key=get_anthropic_key())
    start_time = time.time()
    results = []
    lock = threading.Lock()
    completed = [0]

    def process_one(problem):
        result = run_single_problem(problem,
                                    client,
                                    n_candidates=n_candidates,
                                    use_gold=use_gold)
        with lock:
            completed[0] += 1
            status = "OK" if result['final_correct'] else (
                "ERR" if result['error'] else "WRONG")
            qa_info = f"QA={result['n_questions_used']}" if result[
                'qa_used'] else "selected"
            print(f"  [{completed[0]}/{len(problems)}] "
                  f"Problem {problem.idx} ({problem.difficulty}): {status} "
                  f"({qa_info}, llm_correct={result['llm_said_correct']})")
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
    qa_used_count = sum(1 for r in results if r['qa_used'])
    llm_said_correct_count = sum(1 for r in results if r['llm_said_correct'])
    selection_actually_correct = sum(1 for r in results
                                     if r['actually_correct_after_selection'])

    # Per-difficulty breakdown
    difficulty_stats = {}
    for diff in ['easy', 'medium', 'hard', 'very_hard']:
        diff_results = [r for r in results if r['difficulty'] == diff]
        if diff_results:
            d_n = len(diff_results)
            d_init = sum(1 for r in diff_results if r['initial_correct'])
            d_final = sum(1 for r in diff_results if r['final_correct'])
            d_recovered = sum(
                1 for r in diff_results
                if not r['initial_correct'] and r['final_correct'])
            d_regressed = sum(
                1 for r in diff_results
                if r['initial_correct'] and not r['final_correct'])
            d_qa = sum(1 for r in diff_results if r['qa_used'])
            d_llm_correct = sum(1 for r in diff_results
                                if r['llm_said_correct'])
            initially_wrong = d_n - d_init
            recovery_rate = d_recovered / initially_wrong if initially_wrong > 0 else 0.0
            regression_rate = d_regressed / d_init if d_init > 0 else 0.0
            difficulty_stats[diff] = {
                'n': d_n,
                'initial_correct': d_init,
                'final_correct': d_final,
                'recovered': d_recovered,
                'regressed': d_regressed,
                'recovery_rate': recovery_rate,
                'regression_rate': regression_rate,
                'qa_used': d_qa,
                'llm_said_correct': d_llm_correct,
            }

    initially_wrong_total = n - initial_correct
    recovery_rate_total = recovered / initially_wrong_total if initially_wrong_total > 0 else 0.0
    regression_rate_total = regressed / initial_correct if initial_correct > 0 else 0.0

    summary = {
        'dataset': dataset,
        'n_problems': n,
        'n_candidates': n_candidates,
        'initial_correct': initial_correct,
        'final_correct': final_correct,
        'recovered': recovered,
        'regressed': regressed,
        'recovery_rate': recovery_rate_total,
        'regression_rate': regression_rate_total,
        'qa_used': qa_used_count,
        'llm_said_correct': llm_said_correct_count,
        'selection_actually_correct': selection_actually_correct,
        'time_seconds': elapsed,
        'difficulty_breakdown': difficulty_stats,
    }

    print(f"\n--- Best-of-{n_candidates} + QA Results ({dataset}) ---")
    print(f"  Initial: {initial_correct}/{n} ({100*initial_correct/n:.1f}%)")
    print(f"  Final:   {final_correct}/{n} ({100*final_correct/n:.1f}%)")
    print(f"  Recovered: {recovered}, Regressed: {regressed}")
    print(f"  Recovery rate: {100*recovery_rate_total:.1f}% "
          f"({recovered}/{initially_wrong_total} initially wrong)")
    print(f"  Regression rate: {100*regression_rate_total:.1f}% "
          f"({regressed}/{initial_correct} initially correct)")
    print(f"  LLM selected as correct: {llm_said_correct_count}/{n} "
          f"(actually correct: {selection_actually_correct})")
    print(f"  QA fallback used: {qa_used_count}/{n}")
    print(f"  Time: {int(elapsed)}s")

    for diff in ['easy', 'medium', 'hard', 'very_hard']:
        if diff in difficulty_stats:
            ds = difficulty_stats[diff]
            print(
                f"    {diff} (n={ds['n']}): "
                f"recovered={ds['recovered']}, regressed={ds['regressed']}, "
                f"llm_correct={ds['llm_said_correct']}, qa_used={ds['qa_used']}"
            )

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f"{dataset}_bestof{n_candidates}_qa{suffix}{gold_suffix}_{MODEL_VERSION}_{ts}.json"

    output_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'dataset': dataset,
            'model_version': MODEL_VERSION,
            'model_ids': MODEL_IDS,
            'protocol': {
                'n_candidates':
                n_candidates,
                'candidate_temperature':
                CANDIDATE_TEMPERATURE,
                'llm_temperature':
                LLM_TEMPERATURE,
                'n_questions':
                NUM_QUESTIONS,
                'slm':
                'haiku',
                'llm':
                'opus',
                'q_model':
                'haiku',
                'easy_only':
                easy_only,
                'use_gold':
                use_gold,
                'description':
                (f'Best-of-{n_candidates} + QA. SLM generates {n_candidates} candidates '
                 f'(temp={CANDIDATE_TEMPERATURE}), LLM selects best/correct using '
                 f'{"gold answer" if use_gold else "LLM own solution"}. '
                 f'If correct → done. If not → {NUM_QUESTIONS} QA questions.'),
            },
        },
        'summary': summary,
        'problems': sorted(results, key=lambda r: r['idx']),
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"  Saved: {output_file}")

    return summary


# =============================================================================
# Main
# =============================================================================


def main():
    global MODEL_IDS, MODEL_VERSION

    parser = argparse.ArgumentParser(
        description='Run Best-of-N + QA Compression Sweep')
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
    parser.add_argument('--all', action='store_true', help='Run all datasets')
    parser.add_argument('--limit',
                        type=int,
                        default=None,
                        help='Limit problems per dataset (for testing)')
    parser.add_argument('--output-dir',
                        type=str,
                        default=None,
                        help='Output directory')
    parser.add_argument('--baseline-dir',
                        type=str,
                        default=None,
                        help='Directory containing baseline files')
    parser.add_argument('--parallel',
                        type=int,
                        default=1,
                        help='Number of parallel workers')
    parser.add_argument('--easy-only',
                        action='store_true',
                        help='Run on easy problems only (regression test)')
    parser.add_argument(
        '--n-candidates',
        type=int,
        default=N_CANDIDATES,
        help=f'Number of SLM candidates (default: {N_CANDIDATES})')
    parser.add_argument(
        '--no-gold',
        action='store_true',
        help='LLM uses its own solution instead of gold answer')
    parser.add_argument('--use-old-models',
                        action='store_true',
                        help='Use old model versions')

    args = parser.parse_args()

    if args.use_old_models:
        MODEL_IDS = MODEL_IDS_old
        MODEL_VERSION = "v3.5"
    else:
        MODEL_IDS = MODEL_IDS_new
        MODEL_VERSION = "v4.5"

    # Sync with imported module so load_problems etc. use the right IDs
    qa_sweep_module.MODEL_IDS = MODEL_IDS

    output_dir = (Path(args.output_dir) if args.output_dir else
                  Path(f'results/bestofn-qa-sweep/{MODEL_VERSION}'))
    baseline_dir = Path(args.baseline_dir) if args.baseline_dir else None

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
        datasets_to_run = ['gpqa_mc']
    elif args.dataset == 'mbpp':
        datasets_to_run = ['mbpp']
    elif args.dataset == 'aime':
        datasets_to_run = ['aime']
    elif args.dataset == 'hle':
        datasets_to_run = ['hle']

    if not datasets_to_run:
        parser.error("No datasets specified. Use --all or --dataset")

    n_candidates = args.n_candidates
    easy_only = args.easy_only
    use_gold = not args.no_gold

    print(f"Protocol: Best-of-{n_candidates} + QA")
    print(f"  SLM: haiku, LLM: opus, Q: haiku")
    print(f"  Reference: {'gold answer' if use_gold else 'LLM own solution'}")
    print(f"  Candidate temperature: {CANDIDATE_TEMPERATURE}")
    print(f"  QA questions: {NUM_QUESTIONS}")
    print(f"  Easy only: {'YES' if easy_only else 'NO'}")
    print(f"  Parallel: {args.parallel} workers")
    print(f"  Output: {output_dir}")
    print(f"  Datasets: {', '.join(datasets_to_run)}")

    all_results = {}
    for dataset in datasets_to_run:
        try:
            summary = run_dataset(
                dataset,
                output_dir,
                baseline_dir,
                parallel_workers=args.parallel,
                limit=args.limit,
                easy_only=easy_only,
                n_candidates=n_candidates,
                use_gold=use_gold,
            )
            all_results[dataset] = summary
        except Exception as e:
            print(f"\nERROR on {dataset}: {e}")
            import traceback
            traceback.print_exc()

    # Print final summary
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY (Best-of-{n_candidates} + QA)")
    print(f"{'='*80}")
    print(f"{'Dataset':<20s} {'n':>4s}  {'Init':>5s}  {'Final':>5s}  "
          f"{'Recov':>6s}  {'Regr':>6s}  {'LLM✓':>5s}  {'QA':>4s}")
    print('-' * 80)

    for dataset in datasets_to_run:
        if dataset not in all_results:
            continue
        s = all_results[dataset]
        if not s:
            continue
        n = s.get('n_problems', 0)
        print(f"{dataset:<20s} {n:>4d}  "
              f"{s.get('initial_correct', 0):>5d}  "
              f"{s.get('final_correct', 0):>5d}  "
              f"{s.get('recovered', 0):>6d}  "
              f"{s.get('regressed', 0):>6d}  "
              f"{s.get('llm_said_correct', 0):>5d}  "
              f"{s.get('qa_used', 0):>4d}")

    print(f"\nDone! Results in: {output_dir}")


if __name__ == '__main__':
    main()
