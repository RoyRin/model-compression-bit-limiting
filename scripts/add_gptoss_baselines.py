#!/usr/bin/env python3
"""
Add GPT-OSS-120B results to existing baseline files.

Reads each baseline JSON, runs GPT-OSS on problems where it's missing,
and merges results back without re-running Claude models.

Usage:
    python scripts/add_gptoss_baselines.py --all --parallel 6
    python scripts/add_gptoss_baselines.py --dataset aime --parallel 6
    python scripts/add_gptoss_baselines.py --all --parallel 6 --limit 5  # test run

Results saved to: results/model-baselines/{version}-gptoss/ (copies, does not overwrite originals)
"""

import json
import os
import re
import random
import subprocess
import sys
import tempfile
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
from lossy_compression import MODEL_ALIAS_MAP, model_completion
from lossy_compression.benchmarks.hle import (build_hle_prompt,
                                              extract_hle_answer,
                                              check_hle_answer)
from lossy_compression.benchmarks.aime import (extract_aime_answer,
                                               check_aime_answer)
from utils.llm_api import get_openrouter_key

# Model config
MODEL = 'gpt-oss'
MODEL_FULL = MODEL_ALIAS_MAP[MODEL]  # "openai/gpt-oss-120b"
TEMPERATURE = 0.0

# Rate limiting
RATE_LIMIT_BACKOFFS = [30, 60, 120, 240, 480, 480, 480, 480, 480, 480]


def check_rate_limit(error_msg: str) -> bool:
    """Check if error is a rate limit."""
    keywords = [
        'rate_limit', 'rate limit', 'too many requests', '429', 'overloaded',
        'capacity', 'throttl'
    ]
    return any(k in error_msg.lower() for k in keywords)


def call_gptoss(prompt: str, system: str, max_tokens: int = 2000) -> Dict:
    """Call GPT-OSS with retry logic."""
    for attempt in range(10):
        try:
            start = time.time()
            response = model_completion(
                model=MODEL_FULL,
                system=system,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=TEMPERATURE,
            )
            return {
                'response': response,
                'success': True,
                'time': time.time() - start,
                'error': None,
            }
        except Exception as e:
            error_str = str(e)
            if check_rate_limit(error_str):
                wait = RATE_LIMIT_BACKOFFS[min(attempt,
                                               len(RATE_LIMIT_BACKOFFS) - 1)]
                print(
                    f"\n  Rate limited, waiting {wait}s (attempt {attempt + 1}/10)..."
                )
                time.sleep(wait)
            else:
                return {
                    'response': None,
                    'success': False,
                    'time': 0,
                    'error': error_str[:200],
                }
    return {
        'response': None,
        'success': False,
        'time': 0,
        'error': 'max retries'
    }


# =============================================================================
# Dataset-specific prompt/extraction/check logic
# =============================================================================


def extract_gsm8k_answer(text: str) -> Optional[str]:
    if not text:
        return None
    match = re.search(r'####\s*(\-?\d[\d,]*)', text)
    if match:
        return match.group(1).replace(',', '')
    boxed = re.findall(r'\\boxed\{([^{}]*)\}', text)
    if boxed:
        nums = re.findall(r'\-?\d+', boxed[-1])
        return nums[-1] if nums else None
    nums = re.findall(r'\-?\d[\d,]*', text)
    return nums[-1].replace(',', '') if nums else None


def extract_boxed_answer(text: str) -> Optional[str]:
    if not text:
        return None
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)
    return matches[-1].strip() if matches else None


def normalize_math_answer(answer: str) -> str:
    if not answer:
        return ''
    ans = answer.strip().lower()
    ans = ans.replace('$', '')
    ans = re.sub(r'\\frac\{(\d+)\}\{(\d+)\}', r'\1/\2', ans)
    return ans


def extract_mc_answer(text: str) -> Optional[str]:
    if not text:
        return None
    match = re.search(r'\b([A-D])\b', text)
    return match.group(1) if match else None


def extract_python_code(response: str) -> Optional[str]:
    if not response:
        return None
    code_match = re.findall(r'```(?:python)?\s*(.*?)```', response, re.DOTALL)
    if code_match:
        return code_match[0].strip()
    func_match = re.findall(r'(def\s+\w+.*?)(?=\ndef\s|\Z)', response,
                            re.DOTALL)
    if func_match:
        return func_match[0].strip()
    return response.strip()


def run_code_tests(code: str,
                   tests: List[str],
                   timeout: int = 5) -> Tuple[int, int]:
    if not code:
        return 0, len(tests)
    test_code = code + '\n\n' + '\n'.join(tests)
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py',
                                         delete=False) as f:
            f.write(test_code)
            f.flush()
            temp_path = f.name
        result = subprocess.run(['python3', temp_path],
                                capture_output=True,
                                timeout=timeout)
        os.unlink(temp_path)
        if result.returncode == 0:
            return len(tests), len(tests)
        return 0, len(tests)
    except subprocess.TimeoutExpired:
        return 0, len(tests)
    except Exception:
        return 0, len(tests)
    finally:
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)


# =============================================================================
# Per-dataset handlers
# =============================================================================


def process_gsm8k(result: Dict) -> Dict:
    """Process a single GSM8K problem for GPT-OSS."""
    system = "You are a math tutor. Solve problems step by step. End with #### followed by the numerical answer."
    question = result['question']
    prompt = f"Solve this math problem:\n\n{question}\n\nShow your work, then give the final answer after ####"

    resp = call_gptoss(prompt, system, max_tokens=2000)
    response_text = resp['response'] or ''
    extracted = extract_gsm8k_answer(response_text)
    gold = str(result['gold_answer'])
    correct = str(extracted) == gold if extracted else False

    return {
        'answer': extracted,
        'correct': correct,
        'time': resp['time'],
        'error': resp['error'],
    }


def process_math(result: Dict) -> Dict:
    """Process a single MATH problem for GPT-OSS."""
    system = "You are a math expert. Solve problems step by step. Put your final answer in \\boxed{}."
    question = result['question']
    prompt = f"Solve this problem:\n\n{question}\n\nPut your final answer in \\boxed{{}}"

    resp = call_gptoss(prompt, system, max_tokens=2048)
    response_text = resp['response'] or ''
    extracted = extract_boxed_answer(response_text)
    gold = result['gold_answer']
    correct = normalize_math_answer(str(extracted)) == normalize_math_answer(
        str(gold)) if extracted else False

    return {
        'answer': extracted,
        'correct': correct,
        'time': resp['time'],
        'error': resp['error'],
    }


def process_gpqa_mc(result: Dict, original_problem: Dict,
                    problem_id: int) -> Dict:
    """Process a single GPQA MC problem for GPT-OSS."""
    system = "You are an expert scientist with deep knowledge in physics, chemistry, and biology."

    answers = [
        original_problem['Incorrect Answer 1'],
        original_problem['Incorrect Answer 2'],
        original_problem['Incorrect Answer 3'],
        original_problem['Correct Answer'],
    ]
    rng = random.Random(42 + problem_id)
    indices = [0, 1, 2, 3]
    rng.shuffle(indices)
    letters = ['A', 'B', 'C', 'D']
    choices = []
    correct_letter = None
    for i, idx in enumerate(indices):
        choices.append((letters[i], answers[idx]))
        if idx == 3:
            correct_letter = letters[i]

    prompt = f"""{original_problem['Question']}

Choices:
A) {choices[0][1]}
B) {choices[1][1]}
C) {choices[2][1]}
D) {choices[3][1]}

Analyze this question carefully and select the best answer. State your answer as A, B, C, or D."""

    resp = call_gptoss(prompt, system, max_tokens=1250)
    response_text = resp['response'] or ''
    extracted = extract_mc_answer(response_text)
    correct = extracted == correct_letter if extracted else False

    return {
        'answer': extracted,
        'correct': correct,
        'time': resp['time'],
        'error': resp['error'],
    }


def process_gpqa_freeform(result: Dict) -> Dict:
    """Process a single GPQA freeform problem for GPT-OSS."""
    system = "You are an expert scientist with deep knowledge in physics, chemistry, and biology."
    prompt = f"Question: {result['question']}\n\nProvide a clear, detailed answer."

    resp = call_gptoss(prompt, system, max_tokens=2000)
    response_text = resp['response'] or ''

    # For freeform, we need LLM-as-judge. Store the response and mark as needing evaluation.
    # Use a simple heuristic: check if gold answer appears in response
    gold = result.get('gold_answer', '')
    correct = gold.lower().strip() in response_text.lower(
    ) if gold and response_text else False

    return {
        'answer': response_text[:500] if response_text else None,
        'correct': correct,
        'time': resp['time'],
        'error': resp['error'],
    }


def process_mbpp(result: Dict, original_problem: Dict) -> Dict:
    """Process a single MBPP problem for GPT-OSS."""
    system = "You are an expert Python programmer. Write clean, efficient code."
    task_desc = result['prompt']
    func_name = None
    tests = original_problem.get('test_list', [])
    if tests:
        match = re.search(r'assert\s+(\w+)\s*\(', tests[0])
        if match:
            func_name = match.group(1)

    prompt = f"""Write a Python function that solves this problem:

{task_desc}

{"The function should be named: " + func_name if func_name else ""}
Write ONLY the function code, no examples or test cases."""

    resp = call_gptoss(prompt, system, max_tokens=2048)
    response_text = resp['response'] or ''
    code = extract_python_code(response_text)

    tests_passed, tests_total = 0, len(tests)
    if code and tests:
        tests_passed, tests_total = run_code_tests(code, tests)

    correct = tests_passed == tests_total and tests_total > 0

    return {
        'code': code[:500] if code else None,
        'tests_passed': tests_passed,
        'tests_total': tests_total,
        'correct': correct,
        'time': resp['time'],
        'error': resp['error'],
    }


def process_aime(result: Dict) -> Dict:
    """Process a single AIME problem for GPT-OSS."""
    system = "You are an expert competition mathematician. Solve problems carefully and show your work."
    problem_text = result['problem']
    prompt = f"""Solve this AIME problem. The answer is an integer from 000 to 999.

{problem_text}

Show your work step by step. State your final answer as a single integer (000-999)."""

    resp = call_gptoss(prompt, system, max_tokens=2048)
    response_text = resp['response'] or ''
    extracted = extract_aime_answer(response_text)
    correct = check_aime_answer(extracted, result['gold_answer'])

    return {
        'response': response_text[:500] if response_text else None,
        'extracted': extracted,
        'correct': correct,
        'time': resp['time'],
        'error': resp['error'],
    }


def process_hle(result: Dict) -> Dict:
    """Process a single HLE problem for GPT-OSS."""
    system = "You are a highly knowledgeable assistant. Answer questions carefully and precisely."
    prompt = build_hle_prompt(result['question'], result['answer_type'])

    resp = call_gptoss(prompt, system, max_tokens=2048)
    response_text = resp['response'] or ''
    extracted = extract_hle_answer(response_text, result['answer_type'])
    correct = check_hle_answer(extracted, result['answer'],
                               result['answer_type'])

    return {
        'response': response_text[:500] if response_text else None,
        'extracted': extracted,
        'correct': correct,
        'time': resp['time'],
        'error': resp['error'],
    }


def process_mmlu_pro(result: Dict) -> Dict:
    """Process a single MMLU-Pro problem for GPT-OSS."""
    system = "You are an expert across many academic disciplines. Provide clear, accurate answers."
    prompt = f"Question: {result['question']}\n\nProvide a clear, accurate answer."

    resp = call_gptoss(prompt, system, max_tokens=2048)
    response_text = resp['response'] or ''

    gold = result.get('gold_answer', '')
    correct = gold.lower().strip() in response_text.lower(
    ) if gold and response_text else False

    return {
        'answer': response_text[:500] if response_text else None,
        'correct': correct,
        'time': resp['time'],
        'error': resp['error'],
    }


# =============================================================================
# Main processing
# =============================================================================

DATASET_HANDLERS = {
    'gsm8k': process_gsm8k,
    'math_algebra': process_math,
    'math_geometry': process_math,
    'math_number_theory': process_math,
    'gpqa_mc': None,  # needs original dataset, handled separately
    'gpqa_freeform': process_gpqa_freeform,
    'mbpp': None,  # needs original dataset, handled separately
    'aime': process_aime,
    'hle': process_hle,
    'mmlu_pro': process_mmlu_pro,
}

DATASETS = list(DATASET_HANDLERS.keys())


def add_gptoss_to_file(
    dataset: str,
    baseline_dir: Path,
    output_dir: Path,
    version: str,
    parallel: int = 1,
    limit: Optional[int] = None,
) -> Optional[Dict]:
    """Add GPT-OSS results to a copy of the existing baseline file."""
    filename = f"{dataset}_{version}.json"
    src_path = baseline_dir / filename
    out_path = output_dir / filename

    if not src_path.exists():
        print(f"  SKIP {dataset}: no baseline file found at {src_path}")
        return None

    # Check output file first (resume support)
    if out_path.exists():
        with open(out_path) as f:
            data = json.load(f)
        if MODEL in data.get('model_accuracy', {}):
            acc = data['model_accuracy'][MODEL]
            print(
                f"  SKIP {dataset}: gpt-oss already present in {out_path} ({100*acc:.1f}%)"
            )
            return data

    # Load from source
    with open(src_path) as f:
        data = json.load(f)

    # Check if source already has gpt-oss
    if MODEL in data.get('model_accuracy', {}):
        acc = data['model_accuracy'][MODEL]
        print(
            f"  SKIP {dataset}: gpt-oss already present in source ({100*acc:.1f}%)"
        )
        return data

    results = data['results']
    if limit:
        results = results[:limit]

    n_total = len(results)
    print(f"\n{'='*60}")
    print(f"Adding GPT-OSS to {dataset} ({n_total} problems)")
    print(f"{'='*60}")

    # Load original datasets if needed
    original_data = None
    if dataset == 'gpqa_mc':
        ds = load_dataset('Idavidrein/gpqa', 'gpqa_diamond')
        original_data = list(ds['train'])
    elif dataset == 'mbpp':
        ds = load_dataset('google-research-datasets/mbpp', 'sanitized')
        original_data = list(ds['test'])

    # Process problems
    def process_one(idx_result):
        idx, result = idx_result
        if dataset == 'gpqa_mc':
            return idx, process_gpqa_mc(result,
                                        original_data[result['problem_idx']],
                                        result['problem_idx'])
        elif dataset == 'mbpp':
            return idx, process_mbpp(result,
                                     original_data[result['problem_idx']])
        else:
            handler = DATASET_HANDLERS[dataset]
            return idx, handler(result)

    gptoss_results = {}
    if parallel > 1:
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {
                executor.submit(process_one, (i, r)): i
                for i, r in enumerate(results)
            }
            done = 0
            for future in as_completed(futures):
                idx, model_data = future.result()
                gptoss_results[idx] = model_data
                done += 1
                if done % 20 == 0 or done == n_total:
                    correct = sum(1 for r in gptoss_results.values()
                                  if r.get('correct'))
                    print(
                        f"  [{done}/{n_total}] Running accuracy: {100*correct/done:.1f}%"
                    )
    else:
        for i, r in enumerate(results):
            _, model_data = process_one((i, r))
            gptoss_results[i] = model_data
            if (i + 1) % 20 == 0 or (i + 1) == n_total:
                correct = sum(1 for r in gptoss_results.values()
                              if r.get('correct'))
                print(
                    f"  [{i+1}/{n_total}] Running accuracy: {100*correct/(i+1):.1f}%"
                )

    # Merge results back
    for i, result in enumerate(results):
        if i in gptoss_results:
            result['models'][MODEL] = gptoss_results[i]

    # Update model_accuracy
    all_results = data['results']
    correct_count = sum(
        1 for r in all_results
        if r.get('models', {}).get(MODEL, {}).get('correct', False))
    data['model_accuracy'][MODEL] = correct_count / len(
        all_results) if all_results else 0

    # Save to output directory (not in-place)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(data, f, indent=2)

    acc = data['model_accuracy'][MODEL]
    print(
        f"\n  GPT-OSS accuracy: {100*acc:.1f}% ({correct_count}/{len(all_results)})"
    )
    print(f"  Saved: {out_path}")

    return data


def main():
    parser = argparse.ArgumentParser(
        description='Add GPT-OSS results to existing baseline files')
    parser.add_argument('--dataset',
                        type=str,
                        choices=DATASETS,
                        help='Single dataset to process')
    parser.add_argument('--all',
                        action='store_true',
                        help='Process all datasets')
    parser.add_argument('--parallel',
                        type=int,
                        default=1,
                        help='Parallel workers (default: 1)')
    parser.add_argument('--limit',
                        type=int,
                        default=None,
                        help='Limit problems per dataset (for testing)')
    parser.add_argument(
        '--baseline-dir',
        type=str,
        default=None,
        help='Baseline directory (default: results/model-baselines/v4.5)')
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help=
        'Output directory for updated files (default: results/model-baselines/v4.5-gptoss)'
    )
    args = parser.parse_args()

    version = 'v4.5'
    baseline_dir = Path(args.baseline_dir) if args.baseline_dir else Path(
        f'results/model-baselines/{version}')
    output_dir = Path(args.output_dir) if args.output_dir else Path(
        f'results/model-baselines/{version}-gptoss')

    if args.all:
        datasets = DATASETS
    elif args.dataset:
        datasets = [args.dataset]
    else:
        parser.error("Specify --dataset or --all")

    print(f"Adding GPT-OSS ({MODEL_FULL}) to baselines")
    print(f"Baseline dir: {baseline_dir}")
    print(f"Output dir:   {output_dir}")
    print(f"Parallel: {args.parallel}")
    if args.limit:
        print(f"Limit: {args.limit} problems per dataset")
    print()

    for dataset in datasets:
        try:
            add_gptoss_to_file(
                dataset=dataset,
                baseline_dir=baseline_dir,
                output_dir=output_dir,
                version=version,
                parallel=args.parallel,
                limit=args.limit,
            )
        except Exception as e:
            print(f"\nERROR on {dataset}: {e}")
            import traceback
            traceback.print_exc()

    print("\nDone!")


if __name__ == '__main__':
    main()
