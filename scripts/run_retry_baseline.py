#!/usr/bin/env python3
"""
Retry Baseline Experiment

Measures the stochastic variance in model performance by re-running haiku
on problems it got wrong in the baseline. This provides a null hypothesis
for QA compression - if QA only matches retry rate, it's not adding value.

Usage:
    python scripts/run_retry_baseline.py --dataset gsm8k
    python scripts/run_retry_baseline.py --all --parallel 10
    python scripts/run_retry_baseline.py --all --version v3.5
"""

import json
import argparse
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import anthropic

# Add project root to path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

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

# Initialize client
client = anthropic.Anthropic()

# Datasets to support
DATASETS = [
    'gsm8k', 'math_algebra', 'math_geometry', 'math_number_theory', 'gpqa_mc',
    'gpqa_freeform', 'aime', 'hle'
]


@dataclass
class RetryProblem:
    idx: int
    dataset: str
    question: str
    gold_answer: str
    answer_type: str = ''  # For HLE


def generate_response(prompt: str, system_prompt: str, model_name: str,
                      temperature: float, max_tokens: int) -> str:
    """Generate a response from the model."""
    response = client.messages.create(model=model_name,
                                      max_tokens=max_tokens,
                                      temperature=temperature,
                                      system=system_prompt,
                                      messages=[{
                                          "role": "user",
                                          "content": prompt
                                      }])
    return response.content[0].text


def load_haiku_wrong_problems(dataset: str, baseline_dir: Path,
                              version: str) -> List[RetryProblem]:
    """Load problems where haiku got wrong in baseline."""

    baseline_file = baseline_dir / f"{dataset}_{version}.json"
    if not baseline_file.exists():
        print(f"Baseline file not found: {baseline_file}")
        return []

    with open(baseline_file) as f:
        data = json.load(f)

    problems = []
    for r in data['results']:
        # Check if haiku got it wrong
        haiku_correct = r.get('models', {}).get('haiku',
                                                {}).get('correct', False)
        if haiku_correct:
            continue  # Skip problems haiku got right

        idx = r.get('problem_idx', len(problems))

        if dataset == 'gsm8k':
            problem = RetryProblem(
                idx=idx,
                dataset=dataset,
                question=r.get('question', ''),
                gold_answer=str(r.get('gold_answer', '')),
            )
        elif dataset.startswith('math_'):
            problem = RetryProblem(
                idx=idx,
                dataset=dataset,
                question=r.get('question', r.get('problem', '')),
                gold_answer=str(r.get('gold_answer', '')),
            )
        elif dataset == 'gpqa_mc':
            problem = RetryProblem(
                idx=idx,
                dataset=dataset,
                question=r.get('question', ''),
                gold_answer=str(r.get('gold_answer', '')),
            )
        elif dataset == 'gpqa_freeform':
            problem = RetryProblem(
                idx=idx,
                dataset=dataset,
                question=r.get('question', ''),
                gold_answer=str(r.get('gold_answer', '')),
            )
        elif dataset == 'aime':
            problem = RetryProblem(
                idx=idx,
                dataset=dataset,
                question=r.get('problem', r.get('question', '')),
                gold_answer=str(r.get('gold_answer', '')),
            )
        elif dataset == 'hle':
            problem = RetryProblem(
                idx=idx,
                dataset=dataset,
                question=r.get('question', ''),
                gold_answer=str(r.get('answer', r.get('gold_answer', ''))),
                answer_type=r.get('answer_type', 'exactMatch'),
            )
        else:
            continue

        problems.append(problem)

    return problems


def get_system_prompt(dataset: str) -> str:
    """Get system prompt for dataset."""
    if dataset == 'gsm8k':
        return "You are solving math word problems. Show your work and put your final numerical answer after '#### '."
    elif dataset.startswith('math_'):
        return "You are solving competition math problems. Show your work and put your final answer in \\boxed{}."
    elif dataset == 'gpqa_mc':
        return "You are answering multiple choice science questions. Provide your reasoning, then give your final answer as a single letter (A, B, C, or D)."
    elif dataset == 'gpqa_freeform':
        return "You are answering science questions. Provide a clear, detailed answer."
    elif dataset == 'aime':
        return "You are solving AIME competition math problems. Show your work and give the final answer as an integer from 0 to 999."
    elif dataset == 'hle':
        return "You are a knowledgeable expert. Answer precisely and concisely."
    return "You are a helpful assistant."


# ============= Answer Extraction and Checking =============


def extract_gsm8k_answer(response: str) -> Optional[str]:
    """Extract answer from GSM8K format (#### NUMBER)."""
    match = re.search(r'####\s*(-?[\d,]+(?:\.\d+)?)', response)
    if match:
        return match.group(1).replace(',', '')
    # Fallback: find last number
    numbers = re.findall(r'-?[\d,]+(?:\.\d+)?', response)
    return numbers[-1].replace(',', '') if numbers else None


def check_gsm8k_answer(extracted: Optional[str], gold: str) -> bool:
    """Check GSM8K answer."""
    if not extracted:
        return False
    try:
        return abs(float(extracted) - float(gold.replace(',', ''))) < 0.01
    except:
        return False


def extract_math_answer(response: str) -> Optional[str]:
    """Extract answer from \\boxed{} format."""
    match = re.search(r'\\boxed\{([^}]+)\}', response)
    if match:
        return match.group(1).strip()
    return None


def check_math_answer(extracted: Optional[str], gold: str) -> bool:
    """Check MATH answer (simplified)."""
    if not extracted:
        return False
    # Normalize both
    extracted = extracted.strip().lower()
    gold = gold.strip().lower()
    return extracted == gold


def check_gpqa_mc_answer(response: str, gold: str) -> bool:
    """Check GPQA MC answer (letter matching)."""
    # Find last letter A-D in response
    letters = re.findall(r'\b([A-D])\b', response.upper())
    if not letters:
        return False
    return letters[-1] == gold.upper()


def extract_aime_answer(response: str) -> Optional[int]:
    """Extract AIME answer (integer 0-999)."""
    # Look for common patterns
    patterns = [
        r'(?:answer|final answer|result)[\s:]*(\d+)',
        r'\\boxed\{(\d+)\}',
        r'####\s*(\d+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            val = int(match.group(1))
            if 0 <= val <= 999:
                return val
    # Fallback: last 3-digit or less number
    numbers = re.findall(r'\b(\d{1,3})\b', response)
    if numbers:
        return int(numbers[-1])
    return None


def check_aime_answer(extracted: Optional[int], gold: str) -> bool:
    """Check AIME answer."""
    if extracted is None:
        return False
    try:
        return extracted == int(gold)
    except:
        return False


def extract_hle_answer(response: str, answer_type: str) -> str:
    """Extract HLE answer based on type."""
    if answer_type == 'multipleChoice':
        letters = re.findall(r'\b([A-E])\b', response.upper())
        return letters[-1] if letters else ''
    # For exactMatch, return the full response (will be compared)
    return response.strip()


def check_hle_answer(extracted: str, gold: str, answer_type: str) -> bool:
    """Check HLE answer."""
    if not extracted:
        return False
    if answer_type == 'multipleChoice':
        return extracted.upper() == gold.upper()
    # For exactMatch, do simple string comparison
    return extracted.lower().strip() == gold.lower().strip()


def check_answer(problem: RetryProblem, response: str) -> bool:
    """Check if response is correct for the given problem."""
    dataset = problem.dataset

    if dataset == 'gsm8k':
        extracted = extract_gsm8k_answer(response)
        return check_gsm8k_answer(extracted, problem.gold_answer)

    elif dataset.startswith('math_'):
        extracted = extract_math_answer(response)
        return check_math_answer(extracted, problem.gold_answer)

    elif dataset == 'gpqa_mc':
        return check_gpqa_mc_answer(response, problem.gold_answer)

    elif dataset == 'gpqa_freeform':
        # Skip freeform - requires LLM judge
        return False

    elif dataset == 'aime':
        extracted = extract_aime_answer(response)
        return check_aime_answer(extracted, problem.gold_answer)

    elif dataset == 'hle':
        extracted = extract_hle_answer(response, problem.answer_type)
        return check_hle_answer(extracted, problem.gold_answer,
                                problem.answer_type)

    return False


def run_single_retry(problem: RetryProblem, model_name: str,
                     temperature: float) -> Tuple[int, bool, str]:
    """Run a single retry and return (idx, correct, response)."""

    system_prompt = get_system_prompt(problem.dataset)

    try:
        response = generate_response(
            prompt=problem.question,
            system_prompt=system_prompt,
            model_name=model_name,
            temperature=temperature,
            max_tokens=2048,
        )

        correct = check_answer(problem, response)
        return (problem.idx, correct, response)

    except Exception as e:
        print(f"Error on problem {problem.idx}: {e}")
        return (problem.idx, False, f"ERROR: {e}")


def run_retry_baseline(
    dataset: str,
    baseline_dir: Path,
    output_dir: Path,
    version: str = 'v4.5',
    temperature: float = 1.0,
    parallel: int = 1,
    limit: int = None,
) -> Dict:
    """Run retry baseline for a dataset."""

    print(f"\n{'='*60}")
    print(f"Retry Baseline: {dataset} ({version})")
    print(f"{'='*60}")

    # Get model map
    model_map = MODEL_IDS_new if version == 'v4.5' else MODEL_IDS_old
    haiku_model = model_map['haiku']

    # Load problems haiku got wrong
    problems = load_haiku_wrong_problems(dataset, baseline_dir, version)

    if limit:
        problems = problems[:limit]

    print(f"Haiku wrong problems: {len(problems)}")
    print(f"Model: {haiku_model}")
    print(f"Temperature: {temperature}")

    if not problems:
        print("No problems to retry!")
        return {}

    # Run retries
    results = []
    correct_count = 0

    if parallel > 1:
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {
                executor.submit(run_single_retry, p, haiku_model, temperature):
                p
                for p in problems
            }

            with tqdm(total=len(problems), desc="Retrying") as pbar:
                for future in as_completed(futures):
                    idx, correct, response = future.result()
                    results.append({
                        'problem_idx': idx,
                        'retry_correct': correct,
                        'retry_response':
                        response[:500],  # Truncate for storage
                    })
                    if correct:
                        correct_count += 1
                    pbar.update(1)
                    pbar.set_postfix(
                        {'recovered': f"{correct_count}/{len(results)}"})
    else:
        for problem in tqdm(problems, desc="Retrying"):
            idx, correct, response = run_single_retry(problem, haiku_model,
                                                      temperature)
            results.append({
                'problem_idx': idx,
                'retry_correct': correct,
                'retry_response': response[:500],
            })
            if correct:
                correct_count += 1

    # Compute stats
    retry_rate = correct_count / len(problems) if problems else 0

    summary = {
        'dataset': dataset,
        'version': version,
        'model': haiku_model,
        'temperature': temperature,
        'total_wrong': len(problems),
        'retry_correct': correct_count,
        'retry_rate': retry_rate,
    }

    print(f"\nResults:")
    print(f"  Total haiku wrong: {len(problems)}")
    print(f"  Retry correct: {correct_count}")
    print(f"  Retry rate: {retry_rate:.1%}")

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"retry_baseline_{dataset}_{version}_{timestamp}.json"

    output_data = {
        'summary': summary,
        'results': results,
        'timestamp': timestamp,
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Saved: {output_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Run retry baseline experiment')
    parser.add_argument('--dataset',
                        type=str,
                        default=None,
                        choices=DATASETS,
                        help='Dataset to run')
    parser.add_argument('--all', action='store_true', help='Run all datasets')
    parser.add_argument('--version',
                        type=str,
                        default='v4.5',
                        choices=['v3.5', 'v4.5'],
                        help='Model version (default: v4.5)')
    parser.add_argument(
        '--baseline-dir',
        type=str,
        default=None,
        help='Baseline directory (default: results/model-baselines/{version})')
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: results/retry_baseline/{version})')
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='Temperature for retry (default: 1.0, matching QA conditions)')
    parser.add_argument('--parallel',
                        type=int,
                        default=1,
                        help='Number of parallel workers')
    parser.add_argument('--limit',
                        type=int,
                        default=None,
                        help='Limit problems per dataset (for testing)')

    args = parser.parse_args()

    # Set directories
    baseline_dir = Path(args.baseline_dir) if args.baseline_dir else Path(
        f'results/model-baselines/{args.version}')
    output_dir = Path(args.output_dir) if args.output_dir else Path(
        f'results/retry_baseline/{args.version}')

    # Determine datasets
    if args.all:
        datasets = DATASETS
    elif args.dataset:
        datasets = [args.dataset]
    else:
        print("Please specify --dataset or --all")
        return

    print(f"Retry Baseline Experiment")
    print(f"Version: {args.version}")
    print(f"Temperature: {args.temperature}")
    print(f"Datasets: {datasets}")

    all_results = {}

    for dataset in datasets:
        try:
            result = run_retry_baseline(
                dataset=dataset,
                baseline_dir=baseline_dir,
                output_dir=output_dir,
                version=args.version,
                temperature=args.temperature,
                parallel=args.parallel,
                limit=args.limit,
            )
            all_results[dataset] = result
        except Exception as e:
            print(f"Error on {dataset}: {e}")
            import traceback
            traceback.print_exc()

    # Print summary
    print(f"\n{'='*60}")
    print("RETRY BASELINE SUMMARY")
    print(f"{'='*60}")

    for dataset, result in all_results.items():
        if result:
            print(
                f"{dataset:20} {result['retry_correct']:4}/{result['total_wrong']:4} = {result['retry_rate']:.1%}"
            )

    print(f"\nDone!")


if __name__ == "__main__":
    main()
