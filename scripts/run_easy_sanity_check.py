#!/usr/bin/env python3
"""
Easy Questions Sanity Check - Verify retry success rate on easy problems.

Runs SLM=haiku, LLM=opus, Q={haiku, sonnet, opus} on EASY problems only
to check how many easy problems Haiku still gets right on retry.

This measures stochastic variance: if Haiku got a problem right in the baseline,
does it still get it right when we retry?

Usage:
    python scripts/run_easy_sanity_check.py --dataset gsm8k
    python scripts/run_easy_sanity_check.py --all --parallel 10
    python scripts/run_easy_sanity_check.py --all --version v3.5

Results saved to: results/easy-sanity-check/{version}/
"""

import json
import time
import argparse
import re
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import anthropic

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

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

MODEL_IDS = MODEL_IDS_new
MODEL_VERSION = "v4.5"

# Configuration - Fixed SLM=haiku, LLM=opus, vary Q only
LLM = 'opus'
Q_MODELS = ['haiku', 'sonnet', 'opus']
SLM = 'haiku'

NUM_QUESTIONS = 10

# Retry configuration
MAX_RETRIES = 5
INITIAL_BACKOFF = 5
MAX_BACKOFF = 300

# Datasets
DATASETS = [
    'gsm8k', 'math_algebra', 'math_geometry', 'math_number_theory', 'gpqa_mc',
    'gpqa_freeform', 'mbpp', 'aime', 'hle'
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
class Result:
    problem_idx: int
    initial_correct: bool
    final_correct: bool
    q_model: str
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
                'temperature': 1.0,  # Use temperature for variance measurement
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


def run_single_problem(
    problem: Problem,
    q_model: str,
    client: anthropic.Anthropic,
) -> Result:
    """Run QA on a single easy problem."""
    result = Result(
        problem_idx=problem.idx,
        initial_correct=False,
        final_correct=False,
        q_model=q_model,
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

Generate exactly {NUM_QUESTIONS} yes/no questions, numbered 1-{NUM_QUESTIONS}."""

        questions_response = call_api_with_retry(client,
                                                 q_model,
                                                 [{
                                                     'role': 'user',
                                                     'content': question_prompt
                                                 }],
                                                 max_tokens=1024)
        questions = re.findall(r'\d+\.\s*(.+?)(?=\n\d+\.|\Z)',
                               questions_response, re.DOTALL)
        questions = [q.strip() for q in questions[:NUM_QUESTIONS]]
        while len(questions) < NUM_QUESTIONS:
            questions.append("Is the answer correct?")

        # Step 3: LLM answers questions (with gold answer, matching QA sweep protocol)
        answers = []
        for q in questions:
            answer_prompt = f"""Problem: {problem.question}

Correct answer: {problem.gold_answer}

Question: {q}

Answer with only "Yes" or "No"."""

            response = call_api_with_retry(client,
                                           LLM, [{
                                               'role': 'user',
                                               'content': answer_prompt
                                           }],
                                           max_tokens=10)
            response_upper = response.strip().upper()
            if response_upper.startswith('YES'):
                answers.append('Yes')
            elif response_upper.startswith('NO'):
                answers.append('No')
            else:
                answers.append('Unknown')

        # Step 4: SLM updates based on Q&A
        qa_pairs = "\n".join(
            [f"Q: {q}\nA: {a}" for q, a in zip(questions, answers)])
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


def load_easy_problems(dataset: str,
                       baseline_dir: Path,
                       limit: Optional[int] = None) -> List[Problem]:
    """Load EASY problems from baseline file."""
    baseline_file = baseline_dir / f"{dataset}_{MODEL_VERSION}.json"
    if not baseline_file.exists():
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
        difficulty = r.get('difficulty', 'unknown')
        # Only include EASY problems
        if difficulty != 'easy':
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

    if limit:
        problems = problems[:limit]

    return problems


def run_sanity_check(
    dataset: str,
    baseline_dir: Path,
    output_dir: Path,
    parallel: int = 1,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Run sanity check on easy problems for one dataset."""

    print(f"\n{'='*60}")
    print(f"Easy Sanity Check: {dataset} ({MODEL_VERSION})")
    print(f"SLM: {SLM}, LLM: {LLM}")
    print(f"{'='*60}")

    # Load EASY problems only
    problems = load_easy_problems(dataset, baseline_dir, limit)

    if not problems:
        print("No easy problems found!")
        return {}

    print(f"Easy problems: {len(problems)}")

    client = anthropic.Anthropic(api_key=get_anthropic_key())

    all_results = {}

    for q_model in Q_MODELS:
        combo_key = f"Q-{q_model}"
        print(f"\n--- {combo_key} ---")

        results = []

        if parallel > 1:
            with ThreadPoolExecutor(max_workers=parallel) as executor:
                futures = {
                    executor.submit(run_single_problem, p, q_model, client): p
                    for p in problems
                }
                for future in tqdm(as_completed(futures),
                                   total=len(problems),
                                   desc=f"  {combo_key}",
                                   leave=False):
                    results.append(future.result())
        else:
            for p in tqdm(problems, desc=f"  {combo_key}", leave=False):
                results.append(run_single_problem(p, q_model, client))

        # Compute stats
        initial_correct = sum(1 for r in results if r.initial_correct)
        final_correct = sum(1 for r in results if r.final_correct)
        initial_rate = initial_correct / len(problems) if problems else 0
        final_rate = final_correct / len(problems) if problems else 0

        # Failure rate = how many easy problems we now get WRONG
        initial_failure_rate = 1 - initial_rate
        final_failure_rate = 1 - final_rate

        print(
            f"  Initial: {initial_correct}/{len(problems)} correct ({100*initial_rate:.1f}%), "
            f"failure rate: {100*initial_failure_rate:.1f}%")
        print(
            f"  Final:   {final_correct}/{len(problems)} correct ({100*final_rate:.1f}%), "
            f"failure rate: {100*final_failure_rate:.1f}%")

        all_results[combo_key] = {
            'q_model':
            q_model,
            'n_problems':
            len(problems),
            'initial_correct':
            initial_correct,
            'final_correct':
            final_correct,
            'initial_accuracy':
            initial_rate,
            'final_accuracy':
            final_rate,
            'initial_failure_rate':
            initial_failure_rate,
            'final_failure_rate':
            final_failure_rate,
            'per_problem': [{
                'problem_idx': r.problem_idx,
                'initial_correct': r.initial_correct,
                'final_correct': r.final_correct,
                'error': r.error,
            } for r in results]
        }

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{dataset}_easy_sanity_{MODEL_VERSION}_{timestamp}.json"

    output_data = {
        'dataset': dataset,
        'version': MODEL_VERSION,
        'slm': SLM,
        'llm': LLM,
        'n_easy_problems': len(problems),
        'timestamp': timestamp,
        'results': all_results,
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved: {output_file}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {dataset} (Easy Problems Sanity Check)")
    print(f"{'='*60}")
    print(
        f"{'Q-Model':<10} {'Initial':<15} {'Final':<15} {'Init Fail':<12} {'Final Fail':<12}"
    )
    print("-" * 65)
    for combo_key, result in all_results.items():
        print(f"{result['q_model']:<10} "
              f"{100*result['initial_accuracy']:>6.1f}%        "
              f"{100*result['final_accuracy']:>6.1f}%        "
              f"{100*result['initial_failure_rate']:>6.1f}%       "
              f"{100*result['final_failure_rate']:>6.1f}%")

    return output_data


def main():
    global MODEL_IDS, MODEL_VERSION

    parser = argparse.ArgumentParser(
        description='Run easy problems sanity check')
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
    parser.add_argument('--baseline-dir',
                        type=str,
                        default=None,
                        help='Baseline directory')
    parser.add_argument('--output-dir',
                        type=str,
                        default=None,
                        help='Output directory')
    parser.add_argument('--parallel',
                        type=int,
                        default=1,
                        help='Parallel workers')
    parser.add_argument('--limit',
                        type=int,
                        default=None,
                        help='Limit problems per dataset (for testing)')

    args = parser.parse_args()

    # Set model version
    if args.version == 'v3.5':
        MODEL_IDS = MODEL_IDS_old
        MODEL_VERSION = 'v3.5'
    else:
        MODEL_IDS = MODEL_IDS_new
        MODEL_VERSION = 'v4.5'

    # Set directories
    baseline_dir = Path(args.baseline_dir) if args.baseline_dir else Path(
        f'results/model-baselines/{MODEL_VERSION}')
    output_dir = Path(args.output_dir) if args.output_dir else Path(
        f'results/easy-sanity-check/{MODEL_VERSION}')

    # Determine datasets
    if args.all:
        datasets = DATASETS
    elif args.dataset:
        datasets = [args.dataset]
    else:
        print("Please specify --dataset or --all")
        return

    print(f"Easy Problems Sanity Check")
    print(f"Version: {MODEL_VERSION}")
    print(f"SLM: {SLM} (fixed)")
    print(f"LLM: {LLM} (fixed)")
    print(f"Q-Models: {Q_MODELS}")
    print(f"Datasets: {datasets}")
    print(f"Baseline dir: {baseline_dir}")
    print(f"Output dir: {output_dir}")

    for dataset in datasets:
        try:
            run_sanity_check(
                dataset=dataset,
                baseline_dir=baseline_dir,
                output_dir=output_dir,
                parallel=args.parallel,
                limit=args.limit,
            )
        except Exception as e:
            print(f"Error on {dataset}: {e}")
            import traceback
            traceback.print_exc()

    print("\nDone!")


if __name__ == "__main__":
    main()
