#!/usr/bin/env python3
"""
QA Variance Computation - Multiple trials for paper configurations.

Runs K trials of each configuration to compute standard deviations
and confidence intervals for QA compression results.

Configurations:
  BLC  (haiku->haiku->haiku): self-refinement baseline
  QA   (haiku->opus->haiku):  Haiku asks, Opus answers
  QA+  (haiku->opus->opus):   Opus asks, Opus answers

Usage:
    python scripts/run_qa_variance.py --dataset gsm8k --trials 3
    python scripts/run_qa_variance.py --all --trials 3 --parallel 10
    python scripts/run_qa_variance.py --all --version v3.5

Results saved to: results/qa-variance/{version}/{dataset}/
"""

import json
import time
import argparse
import re
import os
import sys
import threading
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import anthropic

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
from utils.llm_api import get_anthropic_key
from lossy_compression.judge import judge_freeform_answer
from lossy_compression.benchmarks.hle import extract_hle_answer, check_hle_answer
from lossy_compression.benchmarks.aime import extract_aime_answer, check_aime_answer

# Model IDs
MODEL_IDS_new = {
    'haiku': 'claude-haiku-4-5-20251001',
    'sonnet': 'claude-sonnet-4-5-20250929',
    'opus': 'claude-opus-4-5-20251101',
}
MODEL_IDS_old = {
    'haiku': 'claude-3-5-haiku-20241022',
    'sonnet': 'claude-sonnet-4-20250514',
    'opus': 'claude-opus-4-20250514',
}

MODEL_IDS = MODEL_IDS_new  # Default, switched by --use-old-models
MODEL_VERSION = "v4.5"

# Configurations matching the paper table
SLM = 'haiku'  # Fixed - SLM is always haiku
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

NUM_QUESTIONS = 10
DIFFICULTIES = ['medium', 'hard', 'very_hard']

# Retry configuration
MAX_RETRIES = 5
INITIAL_BACKOFF = 5
MAX_BACKOFF = 300

# Datasets (matching paper table - gpqa_freeform excluded)
DATASETS = [
    'gsm8k', 'math_algebra', 'math_geometry', 'math_number_theory', 'gpqa_mc',
    'mbpp', 'aime', 'hle'
]


@dataclass
class Problem:
    idx: int
    dataset: str
    question: str
    gold_answer: str
    difficulty: str
    choices: Optional[List[str]] = None
    function_name: Optional[str] = None
    answer_type: str = 'exactMatch'


@dataclass
class TrialResult:
    trial: int
    problem_idx: int
    initial_correct: bool
    final_correct: bool
    questions: List[str] = field(default_factory=list)
    answers: List[str] = field(default_factory=list)
    error: Optional[str] = None


def call_api_with_retry(client,
                        model: str,
                        messages: List[Dict],
                        system: str = None,
                        max_tokens: int = 1024) -> str:
    """Call API with exponential backoff retry."""
    model_id = MODEL_IDS[model]
    backoff = INITIAL_BACKOFF

    for attempt in range(MAX_RETRIES):
        try:
            kwargs = {
                'model': model_id,
                'max_tokens': max_tokens,
                'messages': messages,
                'temperature': 1.0,  # Use temperature for variance
            }
            if system:
                kwargs['system'] = system

            response = client.messages.create(**kwargs)
            return response.content[0].text

        except anthropic.RateLimitError:
            if attempt < MAX_RETRIES - 1:
                time.sleep(backoff)
                backoff = min(backoff * 2, MAX_BACKOFF)
            else:
                raise
        except Exception as e:
            if attempt < MAX_RETRIES - 1 and 'overloaded' in str(e).lower():
                time.sleep(backoff)
                backoff = min(backoff * 2, MAX_BACKOFF)
            else:
                raise

    return ""


def get_system_prompt(dataset: str) -> str:
    """Get system prompt for SLM."""
    prompts = {
        'gsm8k':
        "You are solving math problems. Show your work and put the final numerical answer after ####.",
        'math_algebra':
        "You are solving math problems. Show your work and put your final answer in \\boxed{}.",
        'math_geometry':
        "You are solving math problems. Show your work and put your final answer in \\boxed{}.",
        'math_number_theory':
        "You are solving math problems. Show your work and put your final answer in \\boxed{}.",
        'gpqa_mc':
        "You are answering science questions. Analyze carefully and give your answer as A, B, C, or D.",
        'gpqa_freeform':
        "You are answering science questions. Provide a clear, detailed answer.",
        'mbpp':
        "You are writing Python code. Provide only the function definition.",
        'hle':
        "You are a highly knowledgeable expert. Answer questions precisely and concisely.",
        'aime':
        "You are an expert competition mathematician. Solve AIME problems carefully. Show your work and give the final answer as an integer from 0 to 999.",
    }
    return prompts.get(dataset, "You are a helpful assistant.")


def check_answer(problem: Problem, response: str) -> bool:
    """Check if response is correct."""
    dataset = problem.dataset

    if dataset == 'gsm8k':
        match = re.search(r'####\s*(-?[\d,]+(?:\.\d+)?)', response)
        if match:
            extracted = match.group(1).replace(',', '')
            try:
                return abs(
                    float(extracted) -
                    float(problem.gold_answer.replace(',', ''))) < 0.01
            except:
                return False
        return False

    elif dataset.startswith('math_'):
        match = re.search(r'\\boxed\{([^}]+)\}', response)
        if match:
            return match.group(
                1).strip().lower() == problem.gold_answer.strip().lower()
        return False

    elif dataset == 'gpqa_mc':
        letters = re.findall(r'\b([A-D])\b', response.upper())
        if letters:
            return letters[-1] == problem.gold_answer.upper()
        return False

    elif dataset == 'gpqa_freeform':
        return judge_freeform_answer(problem.question, response,
                                     problem.gold_answer)

    elif dataset == 'mbpp':
        # Simplified - just check syntax
        try:
            compile(response, '<string>', 'exec')
            return True
        except:
            return False

    elif dataset == 'hle':
        extracted = extract_hle_answer(response, problem.answer_type)
        return check_hle_answer(extracted, problem.gold_answer,
                                problem.answer_type)

    elif dataset == 'aime':
        extracted = extract_aime_answer(response)
        return check_aime_answer(extracted, problem.gold_answer)

    return False


def run_single_qa_trial(
    problem: Problem,
    llm: str,
    q_model: str,
    trial: int,
    client: anthropic.Anthropic,
) -> TrialResult:
    """Run a single QA trial for one problem."""
    result = TrialResult(
        trial=trial,
        problem_idx=problem.idx,
        initial_correct=False,
        final_correct=False,
    )

    system = get_system_prompt(problem.dataset)

    try:
        # Step 1: SLM proposes initial answer
        proposal_prompt = f"Solve this problem:\n\n{problem.question}"
        if problem.choices:
            proposal_prompt += "\n\nChoices:\n"
            for i, c in enumerate(problem.choices):
                proposal_prompt += f"{chr(65+i)}. {c}\n"

        initial_response = call_api_with_retry(client,
                                               SLM, [{
                                                   'role': 'user',
                                                   'content': proposal_prompt
                                               }],
                                               system=system,
                                               max_tokens=2048)
        result.initial_correct = check_answer(problem, initial_response)

        # Step 2: Q-model generates questions
        question_prompt = f"""Given this problem and an initial attempt, generate {NUM_QUESTIONS} yes/no questions that would help clarify the correct solution approach.

Problem: {problem.question}

Initial attempt: {initial_response}

Generate exactly {NUM_QUESTIONS} yes/no questions, numbered 1-{NUM_QUESTIONS}. Each question should:
- Be answerable with just "Yes" or "No"
- Help identify if the approach or answer is correct
- Focus on key steps, calculations, or concepts

Questions:"""

        questions_response = call_api_with_retry(client,
                                                 q_model,
                                                 [{
                                                     'role': 'user',
                                                     'content': question_prompt
                                                 }],
                                                 max_tokens=1024)
        questions = re.findall(r'\d+\.\s*(.+?)(?=\n\d+\.|\Z)',
                               questions_response, re.DOTALL)
        result.questions = [q.strip() for q in questions[:NUM_QUESTIONS]]
        while len(result.questions) < NUM_QUESTIONS:
            result.questions.append("Is the answer correct?")

        # Step 3: LLM answers questions
        result.answers = []
        for q in result.questions:
            answer_prompt = f"""Problem: {problem.question}

Question: {q}

Answer with just "Yes" or "No"."""

            response = call_api_with_retry(client,
                                           llm, [{
                                               'role': 'user',
                                               'content': answer_prompt
                                           }],
                                           max_tokens=10)
            response_upper = response.strip().upper()
            if response_upper.startswith('YES'):
                result.answers.append('Yes')
            elif response_upper.startswith('NO'):
                result.answers.append('No')
            else:
                result.answers.append('Unknown')

        # Step 4: SLM updates based on Q&A
        qa_pairs = "\n".join([
            f"Q: {q}\nA: {a}" for q, a in zip(result.questions, result.answers)
        ])
        update_prompt = f"""You previously attempted this problem. Based on the feedback below, provide an improved solution.

Problem: {problem.question}

Your initial attempt: {initial_response}

Feedback (questions and answers about your solution):
{qa_pairs}

Based on this feedback, solve the problem again. Show your work and provide the final answer."""

        final_response = call_api_with_retry(client,
                                             SLM, [{
                                                 'role': 'user',
                                                 'content': update_prompt
                                             }],
                                             system=system,
                                             max_tokens=2048)
        result.final_correct = check_answer(problem, final_response)

    except Exception as e:
        result.error = str(e)

    return result


def load_problems_from_baseline(
        dataset: str,
        baseline_dir: Path,
        use_strict_difficulty: bool = False) -> List[Problem]:
    """Load non-easy problems from baseline file.

    Args:
        use_strict_difficulty: If True, use 'strict_difficulty' from robust baselines
    """
    baseline_file = baseline_dir / f"{dataset}_{MODEL_VERSION}.json"
    if not baseline_file.exists():
        # Try alternative patterns
        patterns = [f"{dataset}_*.json", f"{dataset}.json"]
        for pattern in patterns:
            files = list(baseline_dir.glob(pattern))
            if files:
                baseline_file = max(files, key=lambda f: f.stat().st_mtime)
                break
        else:
            print(f"No baseline file found for {dataset} in {baseline_dir}")
            return []

    print(f"Loading from: {baseline_file}")
    with open(baseline_file) as f:
        data = json.load(f)

    problems = []
    for r in data['results']:
        # Use strict_difficulty for robust baselines if requested
        if use_strict_difficulty:
            difficulty = r.get('strict_difficulty',
                               r.get('difficulty', 'unknown'))
        else:
            difficulty = r.get('difficulty', 'unknown')
        if difficulty not in DIFFICULTIES:
            continue

        idx = r.get('problem_idx', r.get('idx', len(problems)))

        problem = Problem(
            idx=idx,
            dataset=dataset,
            question=r.get('question', r.get('problem', '')),
            gold_answer=str(r.get('gold_answer', r.get('answer', ''))),
            difficulty=difficulty,
            choices=r.get('choices'),
            function_name=r.get('function_name'),
            answer_type=r.get('answer_type', 'exactMatch'),
        )
        problems.append(problem)

    return problems


def run_variance_experiment(
    dataset: str,
    baseline_dir: Path,
    output_dir: Path,
    num_trials: int = 3,
    parallel: int = 1,
    limit: Optional[int] = None,
    use_strict_difficulty: bool = False,
) -> Dict[str, Any]:
    """Run variance experiment for one dataset."""

    print(f"\n{'='*60}")
    print(f"QA Variance: {dataset} ({MODEL_VERSION})" +
          (" (ROBUST)" if use_strict_difficulty else ""))
    print(f"Trials: {num_trials}, SLM: {SLM}")
    print(f"Configs: {', '.join(c['name'] for c in CONFIGS)}")
    print(f"{'='*60}")

    # Load problems
    problems = load_problems_from_baseline(
        dataset, baseline_dir, use_strict_difficulty=use_strict_difficulty)
    if limit:
        problems = problems[:limit]

    if not problems:
        print("No problems to evaluate!")
        return {}

    print(f"Problems: {len(problems)}")

    client = anthropic.Anthropic(api_key=get_anthropic_key())

    # Results structure: {config_name: {trial: [results]}}
    all_results = {}

    for config in CONFIGS:
        config_name = config['name']
        llm = config['llm']
        q_model = config['q']
        combo_key = f"{config_name}_LLM-{llm}_Q-{q_model}"
        print(f"\n--- {config_name} ({SLM}->{llm}->{SLM}, Q={q_model}) ---")

        combo_results = {t: [] for t in range(num_trials)}

        # Run trials
        for trial in range(num_trials):
            print(f"  Trial {trial + 1}/{num_trials}...")

            if parallel > 1:
                with ThreadPoolExecutor(max_workers=parallel) as executor:
                    futures = {
                        executor.submit(run_single_qa_trial, p, llm, q_model, trial, client):
                        p
                        for p in problems
                    }
                    for future in tqdm(as_completed(futures),
                                       total=len(problems),
                                       desc=f"    T{trial+1}",
                                       leave=False):
                        combo_results[trial].append(future.result())
            else:
                for p in tqdm(problems, desc=f"    T{trial+1}", leave=False):
                    combo_results[trial].append(
                        run_single_qa_trial(p, llm, q_model, trial, client))

        # Compute per-trial accuracy
        trial_accuracies = []
        for trial in range(num_trials):
            correct = sum(1 for r in combo_results[trial] if r.final_correct)
            acc = correct / len(problems) if problems else 0
            trial_accuracies.append(acc)
            print(
                f"    Trial {trial+1}: {correct}/{len(problems)} = {100*acc:.1f}%"
            )

        mean_acc = np.mean(trial_accuracies)
        std_acc = np.std(trial_accuracies)
        print(f"    Mean: {100*mean_acc:.1f}% ± {100*std_acc:.1f}%")

        all_results[combo_key] = {
            'config_name': config_name,
            'llm': llm,
            'q_model': q_model,
            'slm': SLM,
            'trials': num_trials,
            'n_problems': len(problems),
            'trial_accuracies': trial_accuracies,
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'per_trial': {
                t: [{
                    'problem_idx': r.problem_idx,
                    'initial_correct': r.initial_correct,
                    'final_correct': r.final_correct,
                    'error': r.error,
                } for r in combo_results[t]]
                for t in range(num_trials)
            }
        }

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{dataset}_variance_{MODEL_VERSION}_{timestamp}.json"

    output_data = {
        'dataset': dataset,
        'version': MODEL_VERSION,
        'slm': SLM,
        'configs': [c['name'] for c in CONFIGS],
        'num_trials': num_trials,
        'n_problems': len(problems),
        'timestamp': timestamp,
        'combinations': all_results,
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved: {output_file}")

    # Print summary table
    print(f"\n{'='*60}")
    print(f"SUMMARY: {dataset} (SLM={SLM})")
    print(f"{'='*60}")
    print(f"{'Config':<8} {'LLM':<8} {'Q':<8} {'Mean':<12} {'Std':<10}")
    print("-" * 50)
    for combo_key, result in all_results.items():
        print(
            f"{result['config_name']:<8} {result['llm']:<8} {result['q_model']:<8} "
            f"{100*result['mean_accuracy']:>6.1f}%      {100*result['std_accuracy']:>6.1f}%"
        )

    return output_data


def main():
    global MODEL_IDS, MODEL_VERSION

    parser = argparse.ArgumentParser(description='Run QA variance experiment')
    parser.add_argument('--dataset',
                        type=str,
                        choices=DATASETS,
                        help='Dataset to evaluate')
    parser.add_argument('--all', action='store_true', help='Run all datasets')
    parser.add_argument('--version',
                        type=str,
                        default='v4.5',
                        choices=['v3.5', 'v4.5'],
                        help='Model version (default: v4.5)')
    parser.add_argument('--use-old-models',
                        action='store_true',
                        help='Use old model IDs (v3.5)')
    parser.add_argument('--baseline-dir',
                        type=str,
                        default=None,
                        help='Baseline directory')
    parser.add_argument('--output-dir',
                        type=str,
                        default=None,
                        help='Output directory')
    parser.add_argument('--trials',
                        type=int,
                        default=3,
                        help='Number of trials per combination (default: 3)')
    parser.add_argument('--parallel',
                        type=int,
                        default=1,
                        help='Parallel workers')
    parser.add_argument('--limit',
                        type=int,
                        default=None,
                        help='Limit problems (for testing)')
    parser.add_argument('--robust',
                        action='store_true',
                        help='Use robust baselines with strict difficulty')

    args = parser.parse_args()

    # Set model version
    if args.use_old_models or args.version == 'v3.5':
        MODEL_IDS = MODEL_IDS_old
        MODEL_VERSION = 'v3.5'
    else:
        MODEL_IDS = MODEL_IDS_new
        MODEL_VERSION = 'v4.5'

    # Set directories - use robust paths if --robust flag
    if args.robust:
        default_baseline = f'results/model-baselines-robust/{MODEL_VERSION}'
        default_output = f'results/robust-qa-variance/{MODEL_VERSION}'
    else:
        default_baseline = f'results/model-baselines/{MODEL_VERSION}'
        default_output = f'results/qa-variance/{MODEL_VERSION}'

    baseline_dir = Path(
        args.baseline_dir) if args.baseline_dir else Path(default_baseline)
    output_dir = Path(
        args.output_dir) if args.output_dir else Path(default_output)

    # Determine datasets
    if args.all:
        datasets = DATASETS
    elif args.dataset:
        datasets = [args.dataset]
    else:
        print("Please specify --dataset or --all")
        return

    print(f"QA Variance Experiment")
    print(f"Version: {MODEL_VERSION}")
    print(f"Trials: {args.trials}")
    print(f"SLM: {SLM} (fixed)")
    print(f"Configs: {', '.join(c['name'] for c in CONFIGS)}")
    print(f"  BLC:  {SLM}->{CONFIGS[0]['llm']}->{SLM} (Q={CONFIGS[0]['q']})")
    print(f"  QA:   {SLM}->{CONFIGS[1]['llm']}->{SLM} (Q={CONFIGS[1]['q']})")
    print(f"  QA+:  {SLM}->{CONFIGS[2]['llm']}->{SLM} (Q={CONFIGS[2]['q']})")
    print(f"Combinations per dataset: {len(CONFIGS)}")
    print(f"Datasets: {datasets}")

    for dataset in datasets:
        try:
            run_variance_experiment(
                dataset=dataset,
                baseline_dir=baseline_dir,
                output_dir=output_dir,
                num_trials=args.trials,
                parallel=args.parallel,
                limit=args.limit,
                use_strict_difficulty=args.robust,
            )
        except Exception as e:
            print(f"Error on {dataset}: {e}")
            import traceback
            traceback.print_exc()

    print("\nDone!")


if __name__ == "__main__":
    main()
