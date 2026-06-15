#!/usr/bin/env python3
"""
Rerun Opus evaluation on ALL datasets using Anthropic Message Batches API.

This is MUCH faster than sequential calls:
- 50% cheaper than regular API
- Parallel processing on Anthropic's side
- Up to 10,000 requests per batch

Usage:
    # Submit batch for all datasets
    python scripts/rerun_opus_batch.py --submit

    # Check batch status
    python scripts/rerun_opus_batch.py --status --batch-id <batch_id>

    # Download results when complete
    python scripts/rerun_opus_batch.py --download --batch-id <batch_id>

    # All-in-one: submit, poll, download
    python scripts/rerun_opus_batch.py --run
"""

import json
import time
import argparse
import re
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import anthropic

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
from utils.llm_api import get_anthropic_key

# Opus model
OPUS_MODEL = "claude-opus-4-20250514"

# Output directory
OUTPUT_DIR = Path("results/opus_batch")


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract answer from \\boxed{...} format."""
    if not text:
        return None
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    return None


def extract_numerical_answer(text: str) -> Optional[str]:
    """Extract numerical answer for GSM8K."""
    if not text:
        return None
    match = re.search(r'####\s*(\d+)', text)
    if match:
        return match.group(1)
    boxed = extract_boxed_answer(text)
    if boxed:
        nums = re.findall(r'\d+', boxed)
        if nums:
            return nums[-1]
    nums = re.findall(r'\b(\d+)\b', text)
    if nums:
        return nums[-1]
    return None


def normalize_answer(answer: str) -> Optional[str]:
    """Normalize answer for comparison."""
    if answer is None:
        return None
    ans = str(answer).strip().lower()
    ans = ans.replace('\\$', '').replace('$', '')
    ans = ans.replace('\\text{', '').replace('}', '')
    ans = ans.replace('\\', '').replace(' ', '')
    return ans


def extract_mc_answer(text: str) -> Optional[str]:
    """Extract multiple choice answer (A/B/C/D) from response text.

    Tries multiple patterns in order of specificity:
    1. "The answer is X" or "Answer: X"
    2. "X is correct" or "X is the correct answer"
    3. Starts with just the letter
    4. Letter in parentheses or followed by period at start
    5. Last single letter A-D in the response
    """
    if not text:
        return None

    text_upper = text.upper().strip()

    # Pattern 1: "The answer is X" or "Answer: X" or "Answer is X"
    patterns = [
        r'(?:THE\s+)?ANSWER\s*(?:IS|:)\s*\(?([A-D])\)?',
        r'(?:THE\s+)?CORRECT\s+ANSWER\s*(?:IS|:)\s*\(?([A-D])\)?',
        r'\(?([A-D])\)?\s+IS\s+(?:THE\s+)?CORRECT',
        r'I\s+(?:WOULD\s+)?(?:CHOOSE|SELECT|PICK)\s+\(?([A-D])\)?',
        r'(?:OPTION|CHOICE)\s+\(?([A-D])\)?',
    ]

    for pattern in patterns:
        match = re.search(pattern, text_upper)
        if match:
            return match.group(1)

    # Pattern 2: Response starts with just the letter (possibly with punctuation)
    match = re.match(r'^\s*\(?([A-D])\)?[\.\):\s]', text_upper)
    if match:
        return match.group(1)

    # Pattern 3: Look for letter at the very end
    match = re.search(r'\(?([A-D])\)?[\.\s]*$', text_upper)
    if match:
        return match.group(1)

    # Pattern 4: Last resort - find the last standalone A-D letter
    matches = re.findall(r'(?<![A-Z])\(?([A-D])\)?(?![A-Z])', text_upper)
    if matches:
        return matches[-1]

    return None


def prepare_math_requests(subject: str) -> List[Dict]:
    """Prepare batch requests for MATH dataset."""
    print(f"  Loading MATH ({subject})...")
    ds = load_dataset('EleutherAI/hendrycks_math', subject)
    problems = list(ds['test'])

    system = """You are a mathematical problem solver. Show clear step-by-step reasoning.
Always put your final answer in \\boxed{} format at the end."""

    requests = []
    for idx, problem in enumerate(problems):
        problem_text = problem['problem']
        gold_answer = extract_boxed_answer(problem['solution'])

        prompt = f"""Solve this math problem step by step. Put your final answer in \\boxed{{}}.

Problem: {problem_text}

Solution:"""

        requests.append({
            'custom_id': f"math_{subject}_{idx}",
            'params': {
                'model': OPUS_MODEL,
                'max_tokens': 2000,
                'temperature': 0,
                'system': system,
                'messages': [{
                    'role': 'user',
                    'content': prompt
                }],
            },
            'metadata': {
                'dataset': f'math_{subject}',
                'problem_idx': idx,
                'gold_answer': gold_answer,
            }
        })

    print(f"    Prepared {len(requests)} requests")
    return requests


def prepare_gsm8k_requests() -> List[Dict]:
    """Prepare batch requests for GSM8K dataset."""
    print(f"  Loading GSM8K...")
    ds = load_dataset('openai/gsm8k', 'main')
    problems = list(ds['test'])

    system = """You are a math tutor helping with grade school math problems.
Show your work step by step and give the final numerical answer."""

    requests = []
    for idx, problem in enumerate(problems):
        question = problem['question']
        gold = extract_numerical_answer(problem['answer'])

        prompt = f"""Solve this math problem step by step:

{question}

Show your reasoning and give the final answer as a number."""

        requests.append({
            'custom_id': f"gsm8k_{idx}",
            'params': {
                'model': OPUS_MODEL,
                'max_tokens': 2000,
                'temperature': 0,
                'system': system,
                'messages': [{
                    'role': 'user',
                    'content': prompt
                }],
            },
            'metadata': {
                'dataset': 'gsm8k',
                'problem_idx': idx,
                'gold_answer': gold,
            }
        })

    print(f"    Prepared {len(requests)} requests")
    return requests


def prepare_gpqa_requests(format_type: str = 'mc') -> List[Dict]:
    """Prepare batch requests for GPQA dataset."""
    print(f"  Loading GPQA ({format_type})...")
    ds = load_dataset('Idavidrein/gpqa', 'gpqa_diamond')
    problems = list(ds['train'])

    if format_type == 'mc':
        system = """You are an expert scientist answering graduate-level questions.
Analyze the question carefully and select the best answer from the options.
Reply with ONLY the letter (A, B, C, or D)."""
    else:
        system = """You are an expert scientist answering graduate-level questions.
Provide a clear, well-reasoned answer."""

    requests = []
    import random

    for idx, problem in enumerate(problems):
        question = problem['Question']
        correct_answer = problem['Correct Answer']

        if format_type == 'mc':
            choices = [
                problem['Correct Answer'],
                problem['Incorrect Answer 1'],
                problem['Incorrect Answer 2'],
                problem['Incorrect Answer 3'],
            ]
            random.seed(idx)
            random.shuffle(choices)
            correct_idx = choices.index(correct_answer)
            letters = ['A', 'B', 'C', 'D']

            options_text = "\n".join(
                [f"{letters[i]}. {c}" for i, c in enumerate(choices)])
            prompt = f"""Question: {question}

{options_text}

Which answer is correct? Reply with just the letter (A, B, C, or D)."""
            gold = letters[correct_idx]
            max_tokens = 100
        else:
            prompt = f"""Question: {question}

Provide a clear, detailed answer."""
            gold = correct_answer
            max_tokens = 1000

        requests.append({
            'custom_id': f"gpqa_{format_type}_{idx}",
            'params': {
                'model': OPUS_MODEL,
                'max_tokens': max_tokens,
                'temperature': 0,
                'system': system,
                'messages': [{
                    'role': 'user',
                    'content': prompt
                }],
            },
            'metadata': {
                'dataset': f'gpqa_{format_type}',
                'problem_idx': idx,
                'gold_answer': gold,
            }
        })

    print(f"    Prepared {len(requests)} requests")
    return requests


def extract_function_name(test_list):
    """Extract expected function name from test cases."""
    if not test_list:
        return None
    match = re.search(r'assert\s+(\w+)\s*\(', test_list[0])
    return match.group(1) if match else None


def prepare_mbpp_requests() -> List[Dict]:
    """Prepare batch requests for MBPP dataset."""
    print(f"  Loading MBPP...")
    ds = load_dataset('mbpp', 'sanitized')
    problems = list(ds['test'])

    system = """You are a Python code generation assistant.
Write clean, correct Python code that solves the given problem.
Return only the code without any markdown formatting or explanations."""

    requests = []
    for idx, problem in enumerate(problems):
        prompt_text = problem['prompt']
        test_list = problem['test_list']

        # Extract expected function name from tests
        func_name = extract_function_name(test_list)
        func_hint = f"\n\nName your function: {func_name}" if func_name else ""

        prompt = f"""Write a Python function to solve this problem:

{prompt_text}{func_hint}

Provide only the Python code without any explanation. Include the complete function definition."""

        requests.append({
            'custom_id': f"mbpp_{idx}",
            'params': {
                'model': OPUS_MODEL,
                'max_tokens': 1000,
                'temperature': 0,
                'system': system,
                'messages': [{
                    'role': 'user',
                    'content': prompt
                }],
            },
            'metadata': {
                'dataset': 'mbpp',
                'problem_idx': idx,
                'test_list': test_list,
                'func_name': func_name,
            }
        })

    print(f"    Prepared {len(requests)} requests")
    return requests


def prepare_all_requests() -> tuple[List[Dict], Dict]:
    """Prepare all batch requests."""
    print("Preparing batch requests...")

    all_requests = []
    metadata_map = {}

    # MATH datasets
    for subject in ['algebra', 'geometry', 'number_theory']:
        requests = prepare_math_requests(subject)
        for r in requests:
            metadata_map[r['custom_id']] = r['metadata']
        all_requests.extend(requests)

    # GSM8K
    requests = prepare_gsm8k_requests()
    for r in requests:
        metadata_map[r['custom_id']] = r['metadata']
    all_requests.extend(requests)

    # GPQA
    for fmt in ['mc', 'freeform']:
        requests = prepare_gpqa_requests(fmt)
        for r in requests:
            metadata_map[r['custom_id']] = r['metadata']
        all_requests.extend(requests)

    # MBPP
    requests = prepare_mbpp_requests()
    for r in requests:
        metadata_map[r['custom_id']] = r['metadata']
    all_requests.extend(requests)

    print(f"\nTotal requests: {len(all_requests)}")
    return all_requests, metadata_map


def submit_batch(requests: List[Dict], metadata_map: Dict) -> str:
    """Submit batch to Anthropic API."""
    print("\nSubmitting batch to Anthropic...")

    client = anthropic.Anthropic(api_key=get_anthropic_key())

    # Format requests for Anthropic batch API
    batch_requests = []
    for r in requests:
        batch_requests.append({
            'custom_id': r['custom_id'],
            'params': r['params'],
        })

    # Submit batch
    batch = client.messages.batches.create(requests=batch_requests)

    batch_id = batch.id
    print(f"Batch submitted!")
    print(f"  Batch ID: {batch_id}")
    print(f"  Status: {batch.processing_status}")

    # Save batch info
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    batch_info = {
        'batch_id': batch_id,
        'timestamp': timestamp,
        'num_requests': len(requests),
        'status': batch.processing_status,
        'metadata_map': metadata_map,
    }

    info_path = OUTPUT_DIR / f"batch_info_{timestamp}.json"
    with open(info_path, 'w') as f:
        json.dump(batch_info, f, indent=2)
    print(f"  Batch info saved: {info_path}")

    return batch_id


def check_batch_status(batch_id: str) -> Dict:
    """Check batch status."""
    client = anthropic.Anthropic(api_key=get_anthropic_key())

    batch = client.messages.batches.retrieve(batch_id)

    status = {
        'batch_id': batch_id,
        'processing_status': batch.processing_status,
        'request_counts': {
            'processing': batch.request_counts.processing,
            'succeeded': batch.request_counts.succeeded,
            'errored': batch.request_counts.errored,
            'canceled': batch.request_counts.canceled,
            'expired': batch.request_counts.expired,
        },
        'created_at':
        batch.created_at.isoformat() if batch.created_at else None,
        'ended_at': batch.ended_at.isoformat() if batch.ended_at else None,
    }

    return status


def download_results(batch_id: str, metadata_map: Dict = None) -> Dict:
    """Download and process batch results."""
    print(f"\nDownloading results for batch: {batch_id}")

    client = anthropic.Anthropic(api_key=get_anthropic_key())

    # Check status first
    batch = client.messages.batches.retrieve(batch_id)
    if batch.processing_status != 'ended':
        print(f"Batch not complete. Status: {batch.processing_status}")
        return None

    # Load metadata map if not provided
    if metadata_map is None:
        # Try to find it from saved batch info
        info_files = sorted(OUTPUT_DIR.glob("batch_info_*.json"), reverse=True)
        for info_file in info_files:
            with open(info_file) as f:
                info = json.load(f)
            if info.get('batch_id') == batch_id:
                metadata_map = info.get('metadata_map', {})
                break

    # Stream results
    results_by_dataset = {}

    for result in client.messages.batches.results(batch_id):
        custom_id = result.custom_id
        metadata = metadata_map.get(custom_id, {})
        dataset = metadata.get('dataset', 'unknown')

        if dataset not in results_by_dataset:
            results_by_dataset[dataset] = []

        if result.result.type == 'succeeded':
            response = result.result.message
            response_text = response.content[0].text if response.content else ''

            # Extract answer based on dataset type
            if dataset.startswith('math_'):
                extracted = extract_boxed_answer(response_text)
                gold = metadata.get('gold_answer')
                is_correct = normalize_answer(extracted) == normalize_answer(
                    gold)
            elif dataset == 'gsm8k':
                extracted = extract_numerical_answer(response_text)
                gold = metadata.get('gold_answer')
                is_correct = str(extracted) == str(gold)
            elif dataset.startswith('gpqa_'):
                if dataset == 'gpqa_mc':
                    extracted = extract_mc_answer(response_text)
                    gold = metadata.get('gold_answer')
                    is_correct = extracted == gold
                else:
                    # Freeform - store response, will need manual review or LLM judge
                    gold = metadata.get('gold_answer', '')
                    extracted = response_text[:500] if response_text else None
                    # More lenient matching: check key phrases from gold answer
                    gold_words = set(gold.lower().split())
                    response_words = set(response_text.lower().split()
                                         ) if response_text else set()
                    # Consider correct if significant overlap in key terms
                    overlap = len(gold_words & response_words)
                    is_correct = overlap >= len(
                        gold_words) * 0.5 if gold_words else False
            elif dataset == 'mbpp':
                # Extract code - try multiple patterns
                code = None
                # Pattern 1: ```python ... ```
                code_match = re.search(r'```python\s*(.*?)```', response_text,
                                       re.DOTALL)
                if code_match:
                    code = code_match.group(1)
                else:
                    # Pattern 2: ``` ... ``` (any code block)
                    code_match = re.search(r'```\s*(.*?)```', response_text,
                                           re.DOTALL)
                    if code_match:
                        code = code_match.group(1)
                    else:
                        # Pattern 3: Raw code (no markdown)
                        code = response_text.strip()

                test_list = metadata.get('test_list', [])

                is_correct = False
                extracted = code[:200] if code else None

                # Run tests using subprocess (like the baseline)
                import tempfile
                import subprocess

                passed = 0
                for test in test_list:
                    test_code = f"{code}\n\n{test}"
                    try:
                        with tempfile.NamedTemporaryFile(mode='w',
                                                         suffix='.py',
                                                         delete=False) as f:
                            f.write(test_code)
                            temp_path = f.name
                        result = subprocess.run(['python', temp_path],
                                                capture_output=True,
                                                text=True,
                                                timeout=10)
                        os.unlink(temp_path)
                        if result.returncode == 0:
                            passed += 1
                    except:
                        pass

                is_correct = passed == len(test_list)
            else:
                extracted = None
                is_correct = False

            results_by_dataset[dataset].append({
                'problem_idx':
                metadata.get('problem_idx'),
                'opus_answer':
                extracted,
                'opus_correct':
                is_correct,
                'gold_answer':
                metadata.get('gold_answer'),
            })
        else:
            # Failed request
            results_by_dataset[dataset].append({
                'problem_idx':
                metadata.get('problem_idx'),
                'opus_answer':
                None,
                'opus_correct':
                False,
                'error':
                str(result.result.error)
                if hasattr(result.result, 'error') else 'Unknown error',
            })

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    all_results = {}

    for dataset, results in results_by_dataset.items():
        # Sort by problem_idx
        results.sort(key=lambda x: x.get('problem_idx', 0))

        total = len(results)
        correct = sum(1 for r in results if r.get('opus_correct', False))

        dataset_result = {
            'dataset': dataset,
            'total': total,
            'opus_correct': correct,
            'opus_accuracy': correct / total if total > 0 else 0,
            'results': results,
        }

        all_results[dataset] = dataset_result

        # Save individual dataset result
        out_path = OUTPUT_DIR / f"opus_{dataset}_{timestamp}.json"
        with open(out_path, 'w') as f:
            json.dump(dataset_result, f, indent=2)
        print(f"  Saved: {out_path}")

    # Save combined results
    combined_path = OUTPUT_DIR / f"opus_all_datasets_{timestamp}.json"
    with open(combined_path, 'w') as f:
        json.dump(
            {
                'timestamp': timestamp,
                'batch_id': batch_id,
                'datasets': all_results,
                'summary': {
                    name: {
                        'total': r['total'],
                        'opus_correct': r['opus_correct'],
                        'opus_accuracy': r['opus_accuracy'],
                    }
                    for name, r in all_results.items()
                }
            },
            f,
            indent=2)
    print(f"\nCombined results: {combined_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY - Opus Accuracy (Batch)")
    print("=" * 60)
    for name, result in sorted(all_results.items()):
        acc = 100 * result['opus_accuracy']
        print(
            f"{name:<25}: {acc:>6.1f}% ({result['opus_correct']}/{result['total']})"
        )
    print("=" * 60)

    return all_results


def run_batch_pipeline():
    """Full pipeline: prepare, submit, poll, download."""
    print("=" * 60)
    print("OPUS BATCH RERUN - Full Pipeline")
    print("=" * 60)

    # Prepare requests
    requests, metadata_map = prepare_all_requests()

    # Submit batch
    batch_id = submit_batch(requests, metadata_map)

    # Poll for completion
    print("\nPolling for completion...")
    poll_interval = 30  # seconds

    while True:
        status = check_batch_status(batch_id)
        counts = status['request_counts']

        succeeded = counts['succeeded']
        errored = counts['errored']
        processing = counts['processing']
        total = succeeded + errored + processing

        print(f"  [{datetime.now().strftime('%H:%M:%S')}] "
              f"Status: {status['processing_status']} | "
              f"Succeeded: {succeeded}/{total} | "
              f"Errored: {errored} | "
              f"Processing: {processing}")

        if status['processing_status'] == 'ended':
            print("\nBatch complete!")
            break

        time.sleep(poll_interval)

    # Download results
    results = download_results(batch_id, metadata_map)

    return results


def main():
    parser = argparse.ArgumentParser(description='Rerun Opus using batch API')
    parser.add_argument('--submit',
                        action='store_true',
                        help='Submit batch (without waiting)')
    parser.add_argument('--status',
                        action='store_true',
                        help='Check batch status')
    parser.add_argument('--download',
                        action='store_true',
                        help='Download completed batch results')
    parser.add_argument('--run',
                        action='store_true',
                        help='Full pipeline: submit, poll, download')
    parser.add_argument('--batch-id',
                        type=str,
                        help='Batch ID (for --status or --download)')

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.run:
        run_batch_pipeline()
    elif args.submit:
        requests, metadata_map = prepare_all_requests()
        batch_id = submit_batch(requests, metadata_map)
        print(f"\nBatch ID: {batch_id}")
        print("Use --status --batch-id <id> to check status")
        print("Use --download --batch-id <id> when complete")
    elif args.status:
        if not args.batch_id:
            print("Error: --batch-id required with --status")
            sys.exit(1)
        status = check_batch_status(args.batch_id)
        print(json.dumps(status, indent=2))
    elif args.download:
        if not args.batch_id:
            print("Error: --batch-id required with --download")
            sys.exit(1)
        download_results(args.batch_id)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
