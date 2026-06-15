#!/usr/bin/env python3
"""
Robust Baseline Evaluation - Multiple trials per model.

Runs each model K times per problem to get robust difficulty classifications.
A problem is classified as "not-easy" only if the model got it wrong K/K times.

This provides more reliable difficulty labels by filtering out stochastic variance.

Usage:
    python scripts/run_robust_baselines.py --dataset gsm8k --trials 3
    python scripts/run_robust_baselines.py --all --trials 3 --parallel 10
    python scripts/run_robust_baselines.py --all --version v3.5

Results saved to: results/model-baselines-robust/{version}/
"""

import json
import time
import argparse
import re
import os
import sys
import random
import threading
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

import anthropic
from datasets import load_dataset
from lossy_compression import MODEL_ALIAS_MAP, MODEL_ALIAS_MAP_old, model_completion
from lossy_compression.judge import judge_freeform_answer
from lossy_compression.benchmarks.hle import (load_hle_dataset,
                                              get_hle_problem,
                                              build_hle_prompt,
                                              extract_hle_answer,
                                              check_hle_answer)
from lossy_compression.benchmarks.aime import (load_aime_dataset,
                                               get_aime_problem,
                                               build_aime_prompt,
                                               extract_aime_answer,
                                               check_aime_answer)
from utils.llm_api import get_anthropic_key

# Model configuration
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

ACTIVE_MODEL_MAP = MODEL_ALIAS_MAP  # Default to 4.5 models
MODEL_VERSION = "v4.5"

MODELS = ['haiku', 'sonnet', 'opus']  # Can be overridden via --models flag

# Datasets
DATASETS = [
    'gsm8k', 'math_algebra', 'math_geometry', 'math_number_theory', 'gpqa_mc',
    'gpqa_freeform', 'mbpp', 'aime', 'hle'
]

# Rate limit handling
MAX_RETRIES = 10
INITIAL_BACKOFF = 5
MAX_BACKOFF = 300


def call_model_with_retry(model: str,
                          prompt: str,
                          system: str = None,
                          max_tokens: int = 2048) -> Dict:
    """Call model with exponential backoff retry."""
    backoff = INITIAL_BACKOFF
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            start = time.time()
            response = model_completion(
                prompt,
                model,
                system=system,
                max_tokens=max_tokens,
                temperature=1.0,  # Use temperature for variance
            )
            elapsed = time.time() - start
            return {'response': response, 'time': elapsed, 'error': None}

        except Exception as e:
            last_error = str(e)
            error_lower = last_error.lower()
            if any(x in error_lower
                   for x in ['rate', 'overload', '429', '529', 'capacity']):
                wait = min(backoff * (2**attempt), MAX_BACKOFF)
                time.sleep(wait)
            else:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(backoff)
                else:
                    return {'response': None, 'time': 0, 'error': last_error}

    return {'response': None, 'time': 0, 'error': last_error}


# =============================================================================
# Answer extraction functions
# =============================================================================


def extract_gsm8k_answer(text: str) -> Optional[str]:
    if not text:
        return None
    match = re.search(r'####\s*(\-?\d[\d,]*)', text)
    if match:
        return match.group(1).replace(',', '')
    nums = re.findall(r'\-?\d[\d,]*', text)
    return nums[-1].replace(',', '') if nums else None


def extract_math_answer(text: str) -> Optional[str]:
    if not text:
        return None
    matches = re.findall(r'\\boxed\{([^{}]*)\}', text)
    return matches[-1].strip() if matches else None


def extract_gpqa_mc_answer(text: str) -> Optional[str]:
    if not text:
        return None
    letters = re.findall(r'\b([A-D])\b', text.upper())
    return letters[-1] if letters else None


def extract_mbpp_code(text: str) -> Optional[str]:
    if not text:
        return None
    # Extract code from markdown blocks
    blocks = re.findall(r'```(?:python)?\s*(.*?)```', text, re.DOTALL)
    if blocks:
        return blocks[0].strip()
    # Or just return the text if it looks like code
    if 'def ' in text:
        return text
    return None


# =============================================================================
# Dataset runners
# =============================================================================


def run_gsm8k_robust(output_dir: Path,
                     num_trials: int,
                     parallel: int,
                     limit: Optional[int] = None) -> Dict:
    """Run GSM8K with multiple trials per model."""
    print(f"\n{'='*60}\nGSM8K Robust Baseline ({num_trials} trials)\n{'='*60}")

    ds = load_dataset('openai/gsm8k', 'main')
    problems = list(ds['test'])
    if limit:
        problems = problems[:limit]

    print(f"Problems: {len(problems)}")
    system = "You are a math tutor. Solve problems step by step. End with #### followed by the numerical answer."

    # Initialize results structure
    results = []
    for idx, problem in enumerate(problems):
        results.append({
            'problem_idx': idx,
            'question': problem['question'],
            'gold_answer': extract_gsm8k_answer(problem['answer']),
            'models': {},
        })

    # Process ONE MODEL AT A TIME to avoid rate limit conflicts
    for model in MODELS:
        print(f"\n  Processing {model}...")

        def process_problem_for_model(idx):
            """Run all trials for one problem with the current model."""
            problem = problems[idx]
            gold = results[idx]['gold_answer']
            prompt = f"Solve this math problem:\n\n{problem['question']}\n\nShow your work, then give the final answer after ####"

            trial_results = []
            for trial_num in range(num_trials):
                result = call_model_with_retry(model, prompt, system)
                extracted = extract_gsm8k_answer(
                    result['response']) if result['response'] else None
                correct = str(extracted) == str(
                    gold) if extracted and gold else False
                trial_results.append({
                    'trial': trial_num,
                    'answer': extracted,
                    'correct': correct,
                    'error': result['error'],
                })

            correct_count = sum(1 for t in trial_results if t['correct'])
            return idx, {
                'trials': trial_results,
                'correct_count': correct_count,
                'total_trials': num_trials,
                'success_rate': correct_count / num_trials,
            }

        # Parallelize across problems for this model
        if parallel > 1:
            with ThreadPoolExecutor(max_workers=parallel) as executor:
                futures = [
                    executor.submit(process_problem_for_model, idx)
                    for idx in range(len(problems))
                ]
                for future in tqdm(as_completed(futures),
                                   total=len(futures),
                                   desc=f"  {model}"):
                    idx, model_result = future.result()
                    results[idx]['models'][model] = model_result
        else:
            for idx in tqdm(range(len(problems)), desc=f"  {model}"):
                _, model_result = process_problem_for_model(idx)
                results[idx]['models'][model] = model_result

    # Classify difficulty after all models complete
    for r in results:
        r['difficulty'] = classify_robust_difficulty(r['models'], num_trials)
        r['strict_difficulty'] = classify_strict_difficulty(
            r['models'], num_trials)

    return save_robust_results('gsm8k', results, output_dir, num_trials)


def run_math_robust(subject: str,
                    output_dir: Path,
                    num_trials: int,
                    parallel: int,
                    limit: Optional[int] = None) -> Dict:
    """Run MATH with multiple trials per model."""
    dataset_name = f"math_{subject}"
    print(
        f"\n{'='*60}\n{dataset_name.upper()} Robust Baseline ({num_trials} trials)\n{'='*60}"
    )

    ds = load_dataset('EleutherAI/hendrycks_math', subject)
    problems = list(ds['test'])
    if limit:
        problems = problems[:limit]

    print(f"Problems: {len(problems)}")
    system = "You are a math expert. Solve problems step by step. Put your final answer in \\boxed{}."

    # Initialize results structure
    results = []
    for idx, problem in enumerate(problems):
        gold = problem['solution']
        gold_match = re.findall(r'\\boxed\{([^{}]*)\}', gold)
        gold_answer = gold_match[-1].strip() if gold_match else gold
        results.append({
            'problem_idx': idx,
            'question': problem['problem'],
            'gold_answer': gold_answer,
            'models': {},
        })

    # Process ONE MODEL AT A TIME
    for model in MODELS:
        print(f"\n  Processing {model}...")

        def process_problem_for_model(idx):
            problem = problems[idx]
            gold_answer = results[idx]['gold_answer']
            prompt = f"Solve this math problem:\n\n{problem['problem']}\n\nShow your work and put your final answer in \\boxed{{}}"

            trial_results = []
            for trial_num in range(num_trials):
                result = call_model_with_retry(model, prompt, system)
                extracted = extract_math_answer(
                    result['response']) if result['response'] else None
                correct = extracted and extracted.strip().lower(
                ) == gold_answer.strip().lower()
                trial_results.append({
                    'trial': trial_num,
                    'answer': extracted,
                    'correct': correct,
                    'error': result['error'],
                })

            correct_count = sum(1 for t in trial_results if t['correct'])
            return idx, {
                'trials': trial_results,
                'correct_count': correct_count,
                'total_trials': num_trials,
                'success_rate': correct_count / num_trials,
            }

        if parallel > 1:
            with ThreadPoolExecutor(max_workers=parallel) as executor:
                futures = [
                    executor.submit(process_problem_for_model, idx)
                    for idx in range(len(problems))
                ]
                for future in tqdm(as_completed(futures),
                                   total=len(futures),
                                   desc=f"  {model}"):
                    idx, model_result = future.result()
                    results[idx]['models'][model] = model_result
        else:
            for idx in tqdm(range(len(problems)), desc=f"  {model}"):
                _, model_result = process_problem_for_model(idx)
                results[idx]['models'][model] = model_result

    for r in results:
        r['difficulty'] = classify_robust_difficulty(r['models'], num_trials)
        r['strict_difficulty'] = classify_strict_difficulty(
            r['models'], num_trials)

    return save_robust_results(dataset_name, results, output_dir, num_trials)


def run_gpqa_mc_robust(output_dir: Path,
                       num_trials: int,
                       parallel: int,
                       limit: Optional[int] = None) -> Dict:
    """Run GPQA MC with multiple trials per model."""
    print(
        f"\n{'='*60}\nGPQA MC Robust Baseline ({num_trials} trials)\n{'='*60}")

    ds = load_dataset('Idavidrein/gpqa', 'gpqa_diamond')
    problems = list(ds['train'])
    if limit:
        problems = problems[:limit]

    print(f"Problems: {len(problems)}")
    system = "You are a science expert. Answer the multiple choice question. Give your final answer as a single letter (A, B, C, or D)."

    # Initialize results with shuffled choices
    results = []
    prompts = []
    for idx, problem in enumerate(problems):
        random.seed(42 + idx)
        choices = [
            problem['Correct Answer'],
            problem['Incorrect Answer 1'],
            problem['Incorrect Answer 2'],
            problem['Incorrect Answer 3'],
        ]
        random.shuffle(choices)
        gold_idx = choices.index(problem['Correct Answer'])
        gold_letter = chr(65 + gold_idx)

        prompt = f"{problem['Question']}\n\n"
        for i, c in enumerate(choices):
            prompt += f"{chr(65+i)}. {c}\n"
        prompt += "\nAnswer with the letter only."
        prompts.append(prompt)

        results.append({
            'problem_idx': idx,
            'question': problem['Question'],
            'gold_answer': gold_letter,
            'choices': choices,
            'models': {},
        })

    # Process ONE MODEL AT A TIME
    for model in MODELS:
        print(f"\n  Processing {model}...")

        def process_problem_for_model(idx):
            gold_letter = results[idx]['gold_answer']
            trial_results = []
            for trial_num in range(num_trials):
                result = call_model_with_retry(model,
                                               prompts[idx],
                                               system,
                                               max_tokens=100)
                extracted = extract_gpqa_mc_answer(
                    result['response']) if result['response'] else None
                correct = extracted == gold_letter if extracted else False
                trial_results.append({
                    'trial': trial_num,
                    'answer': extracted,
                    'correct': correct,
                    'error': result['error'],
                })

            correct_count = sum(1 for t in trial_results if t['correct'])
            return idx, {
                'trials': trial_results,
                'correct_count': correct_count,
                'total_trials': num_trials,
                'success_rate': correct_count / num_trials,
            }

        if parallel > 1:
            with ThreadPoolExecutor(max_workers=parallel) as executor:
                futures = [
                    executor.submit(process_problem_for_model, idx)
                    for idx in range(len(problems))
                ]
                for future in tqdm(as_completed(futures),
                                   total=len(futures),
                                   desc=f"  {model}"):
                    idx, model_result = future.result()
                    results[idx]['models'][model] = model_result
        else:
            for idx in tqdm(range(len(problems)), desc=f"  {model}"):
                _, model_result = process_problem_for_model(idx)
                results[idx]['models'][model] = model_result

    for r in results:
        r['difficulty'] = classify_robust_difficulty(r['models'], num_trials)
        r['strict_difficulty'] = classify_strict_difficulty(
            r['models'], num_trials)

    return save_robust_results('gpqa_mc', results, output_dir, num_trials)


def run_gpqa_freeform_robust(output_dir: Path,
                             num_trials: int,
                             parallel: int,
                             limit: Optional[int] = None) -> Dict:
    """Run GPQA Freeform with multiple trials per model."""
    print(
        f"\n{'='*60}\nGPQA Freeform Robust Baseline ({num_trials} trials)\n{'='*60}"
    )

    ds = load_dataset('Idavidrein/gpqa', 'gpqa_diamond')
    problems = list(ds['train'])
    if limit:
        problems = problems[:limit]

    print(f"Problems: {len(problems)}")
    system = "You are a science expert. Answer the question directly and concisely."

    # Initialize results
    results = []
    for idx, problem in enumerate(problems):
        results.append({
            'problem_idx': idx,
            'question': problem['Question'],
            'gold_answer': problem['Correct Answer'],
            'models': {},
        })

    # Process ONE MODEL AT A TIME
    for model in MODELS:
        print(f"\n  Processing {model}...")

        def process_problem_for_model(idx):
            problem = problems[idx]
            gold = results[idx]['gold_answer']
            prompt = f"Answer this question:\n\n{problem['Question']}"

            trial_results = []
            for trial_num in range(num_trials):
                result = call_model_with_retry(model, prompt, system)
                response = result['response']
                correct = judge_freeform_answer(problem['Question'], response,
                                                gold) if response else False
                trial_results.append({
                    'trial': trial_num,
                    'answer': response[:200] if response else None,
                    'correct': correct,
                    'error': result['error'],
                })

            correct_count = sum(1 for t in trial_results if t['correct'])
            return idx, {
                'trials': trial_results,
                'correct_count': correct_count,
                'total_trials': num_trials,
                'success_rate': correct_count / num_trials,
            }

        if parallel > 1:
            with ThreadPoolExecutor(max_workers=parallel) as executor:
                futures = [
                    executor.submit(process_problem_for_model, idx)
                    for idx in range(len(problems))
                ]
                for future in tqdm(as_completed(futures),
                                   total=len(futures),
                                   desc=f"  {model}"):
                    idx, model_result = future.result()
                    results[idx]['models'][model] = model_result
        else:
            for idx in tqdm(range(len(problems)), desc=f"  {model}"):
                _, model_result = process_problem_for_model(idx)
                results[idx]['models'][model] = model_result

    for r in results:
        r['difficulty'] = classify_robust_difficulty(r['models'], num_trials)
        r['strict_difficulty'] = classify_strict_difficulty(
            r['models'], num_trials)

    return save_robust_results('gpqa_freeform', results, output_dir,
                               num_trials)


def run_aime_robust(output_dir: Path,
                    num_trials: int,
                    parallel: int,
                    limit: Optional[int] = None) -> Dict:
    """Run AIME with multiple trials per model."""
    print(f"\n{'='*60}\nAIME Robust Baseline ({num_trials} trials)\n{'='*60}")

    problems = load_aime_dataset()
    if limit:
        problems = problems[:limit]

    print(f"Problems: {len(problems)}")
    system = "You are an expert competition mathematician. Solve AIME problems. Give the final answer as an integer from 0 to 999."

    # Initialize results
    results = []
    prompts = []
    for idx, problem in enumerate(problems):
        prompts.append(build_aime_prompt(problem))
        results.append({
            'problem_idx': idx,
            'question': get_aime_problem(problem),
            'gold_answer': str(problem.get('answer', '')),
            'models': {},
        })

    # Process ONE MODEL AT A TIME
    for model in MODELS:
        print(f"\n  Processing {model}...")

        def process_problem_for_model(idx):
            gold = results[idx]['gold_answer']
            trial_results = []
            for trial_num in range(num_trials):
                result = call_model_with_retry(model, prompts[idx], system)
                extracted = extract_aime_answer(
                    result['response']) if result['response'] else None
                correct = check_aime_answer(extracted, gold)
                trial_results.append({
                    'trial':
                    trial_num,
                    'answer':
                    str(extracted) if extracted is not None else None,
                    'correct':
                    correct,
                    'error':
                    result['error'],
                })

            correct_count = sum(1 for t in trial_results if t['correct'])
            return idx, {
                'trials': trial_results,
                'correct_count': correct_count,
                'total_trials': num_trials,
                'success_rate': correct_count / num_trials,
            }

        if parallel > 1:
            with ThreadPoolExecutor(max_workers=parallel) as executor:
                futures = [
                    executor.submit(process_problem_for_model, idx)
                    for idx in range(len(problems))
                ]
                for future in tqdm(as_completed(futures),
                                   total=len(futures),
                                   desc=f"  {model}"):
                    idx, model_result = future.result()
                    results[idx]['models'][model] = model_result
        else:
            for idx in tqdm(range(len(problems)), desc=f"  {model}"):
                _, model_result = process_problem_for_model(idx)
                results[idx]['models'][model] = model_result

    for r in results:
        r['difficulty'] = classify_robust_difficulty(r['models'], num_trials)
        r['strict_difficulty'] = classify_strict_difficulty(
            r['models'], num_trials)

    return save_robust_results('aime', results, output_dir, num_trials)


def run_hle_robust(output_dir: Path,
                   num_trials: int,
                   parallel: int,
                   limit: Optional[int] = None) -> Dict:
    """Run HLE with multiple trials per model."""
    print(f"\n{'='*60}\nHLE Robust Baseline ({num_trials} trials)\n{'='*60}")

    problems = load_hle_dataset()
    if limit:
        problems = problems[:limit]

    print(f"Problems: {len(problems)}")
    system = "You are a highly knowledgeable expert. Answer questions precisely and concisely."

    # Initialize results
    results = []
    prompts = []
    for idx, problem in enumerate(problems):
        prompts.append(build_hle_prompt(problem))
        results.append({
            'problem_idx': idx,
            'question': get_hle_problem(problem)[:500],
            'gold_answer': problem.get('answer', ''),
            'answer_type': problem.get('answer_type', 'exactMatch'),
            'models': {},
        })

    # Process ONE MODEL AT A TIME
    for model in MODELS:
        print(f"\n  Processing {model}...")

        def process_problem_for_model(idx):
            gold = results[idx]['gold_answer']
            answer_type = results[idx]['answer_type']
            trial_results = []
            for trial_num in range(num_trials):
                result = call_model_with_retry(model, prompts[idx], system)
                extracted = extract_hle_answer(
                    result['response'],
                    answer_type) if result['response'] else None
                correct = check_hle_answer(extracted, gold, answer_type)
                trial_results.append({
                    'trial': trial_num,
                    'answer': extracted,
                    'correct': correct,
                    'error': result['error'],
                })

            correct_count = sum(1 for t in trial_results if t['correct'])
            return idx, {
                'trials': trial_results,
                'correct_count': correct_count,
                'total_trials': num_trials,
                'success_rate': correct_count / num_trials,
            }

        if parallel > 1:
            with ThreadPoolExecutor(max_workers=parallel) as executor:
                futures = [
                    executor.submit(process_problem_for_model, idx)
                    for idx in range(len(problems))
                ]
                for future in tqdm(as_completed(futures),
                                   total=len(futures),
                                   desc=f"  {model}"):
                    idx, model_result = future.result()
                    results[idx]['models'][model] = model_result
        else:
            for idx in tqdm(range(len(problems)), desc=f"  {model}"):
                _, model_result = process_problem_for_model(idx)
                results[idx]['models'][model] = model_result

    for r in results:
        r['difficulty'] = classify_robust_difficulty(r['models'], num_trials)
        r['strict_difficulty'] = classify_strict_difficulty(
            r['models'], num_trials)

    return save_robust_results('hle', results, output_dir, num_trials)


def run_mbpp_robust(output_dir: Path,
                    num_trials: int,
                    parallel: int,
                    limit: Optional[int] = None) -> Dict:
    """Run MBPP with multiple trials per model."""
    print(f"\n{'='*60}\nMBPP Robust Baseline ({num_trials} trials)\n{'='*60}")

    ds = load_dataset('google-research-datasets/mbpp', 'sanitized')
    problems = list(ds['test'])
    if limit:
        problems = problems[:limit]

    print(f"Problems: {len(problems)}")
    system = "You are a Python expert. Write only the function definition, no explanations."

    # Initialize results
    results = []
    for idx, problem in enumerate(problems):
        results.append({
            'problem_idx': idx,
            'question': problem['prompt'],
            'test_cases': problem['test_list'],
            'models': {},
        })

    # Process ONE MODEL AT A TIME
    for model in MODELS:
        print(f"\n  Processing {model}...")

        def process_problem_for_model(idx):
            problem = problems[idx]
            prompt_text = problem['prompt']
            test_cases = problem['test_list']

            trial_results = []
            for trial_num in range(num_trials):
                result = call_model_with_retry(model, prompt_text, system)
                response = result['response']
                code_extracted = extract_mbpp_code(
                    response) if response else None

                correct = False
                if code_extracted:
                    try:
                        exec_globals = {}
                        exec(code_extracted, exec_globals)
                        for test in test_cases:
                            exec(test, exec_globals)
                        correct = True
                    except:
                        correct = False

                trial_results.append({
                    'trial':
                    trial_num,
                    'answer':
                    code_extracted[:200] if code_extracted else None,
                    'correct':
                    correct,
                    'error':
                    result['error'],
                })

            correct_count = sum(1 for t in trial_results if t['correct'])
            return idx, {
                'trials': trial_results,
                'correct_count': correct_count,
                'total_trials': num_trials,
                'success_rate': correct_count / num_trials,
            }

        if parallel > 1:
            with ThreadPoolExecutor(max_workers=parallel) as executor:
                futures = [
                    executor.submit(process_problem_for_model, idx)
                    for idx in range(len(problems))
                ]
                for future in tqdm(as_completed(futures),
                                   total=len(futures),
                                   desc=f"  {model}"):
                    idx, model_result = future.result()
                    results[idx]['models'][model] = model_result
        else:
            for idx in tqdm(range(len(problems)), desc=f"  {model}"):
                _, model_result = process_problem_for_model(idx)
                results[idx]['models'][model] = model_result

    for r in results:
        r['difficulty'] = classify_robust_difficulty(r['models'], num_trials)
        r['strict_difficulty'] = classify_strict_difficulty(
            r['models'], num_trials)

    return save_robust_results('mbpp', results, output_dir, num_trials)


# =============================================================================
# Helper functions
# =============================================================================


def classify_robust_difficulty(models_data: Dict, num_trials: int) -> str:
    """Classify difficulty based on robust results.

    A model is considered to "pass" if it got correct > 0 times.
    A model is considered to "fail" if it got correct 0 times (all trials wrong).
    """
    haiku_pass = models_data.get('haiku', {}).get('correct_count', 0) > 0
    sonnet_pass = models_data.get('sonnet', {}).get('correct_count', 0) > 0
    opus_pass = models_data.get('opus', {}).get('correct_count', 0) > 0

    # If only haiku was run, use simple classification
    if 'sonnet' not in models_data and 'opus' not in models_data:
        return 'easy' if haiku_pass else 'not_easy'

    if haiku_pass and sonnet_pass and opus_pass:
        return 'easy'
    elif not haiku_pass and sonnet_pass and opus_pass:
        return 'medium'
    elif not haiku_pass and not sonnet_pass and opus_pass:
        return 'hard'
    else:
        return 'very_hard'


def classify_strict_difficulty(models_data: Dict, num_trials: int) -> str:
    """Stricter classification: model "passes" only if correct ALL trials."""
    haiku_pass = models_data.get('haiku', {}).get('correct_count',
                                                  0) == num_trials
    sonnet_pass = models_data.get('sonnet', {}).get('correct_count',
                                                    0) == num_trials
    opus_pass = models_data.get('opus', {}).get('correct_count',
                                                0) == num_trials

    if haiku_pass and sonnet_pass and opus_pass:
        return 'easy'
    elif not haiku_pass and sonnet_pass and opus_pass:
        return 'medium'
    elif not haiku_pass and not sonnet_pass and opus_pass:
        return 'hard'
    else:
        return 'very_hard'


def save_robust_results(dataset: str, results: List[Dict], output_dir: Path,
                        num_trials: int) -> Dict:
    """Save robust baseline results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{dataset}_{MODEL_VERSION}_trials{num_trials}_{timestamp}.json"

    # Compute summary statistics
    difficulties = [r['difficulty'] for r in results]
    strict_difficulties = [
        classify_strict_difficulty(r['models'], num_trials) for r in results
    ]

    # Per-model accuracy stats
    model_stats = {}
    for model in MODELS:
        success_rates = [r['models'][model]['success_rate'] for r in results]
        model_stats[model] = {
            'mean_success_rate':
            np.mean(success_rates),
            'std_success_rate':
            np.std(success_rates),
            'problems_any_correct':
            sum(1 for r in results if r['models'][model]['correct_count'] > 0),
            'problems_all_correct':
            sum(1 for r in results
                if r['models'][model]['correct_count'] == num_trials),
            'problems_all_wrong':
            sum(1 for r in results
                if r['models'][model]['correct_count'] == 0),
        }

    summary = {
        'dataset': dataset,
        'version': MODEL_VERSION,
        'num_trials': num_trials,
        'n_problems': len(results),
        'difficulty_distribution': {
            'easy': difficulties.count('easy'),
            'medium': difficulties.count('medium'),
            'hard': difficulties.count('hard'),
            'very_hard': difficulties.count('very_hard'),
        },
        'strict_difficulty_distribution': {
            'easy': strict_difficulties.count('easy'),
            'medium': strict_difficulties.count('medium'),
            'hard': strict_difficulties.count('hard'),
            'very_hard': strict_difficulties.count('very_hard'),
        },
        'model_stats': model_stats,
        'timestamp': timestamp,
    }

    output_data = {
        'summary': summary,
        'results': results,
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved: {output_file}")
    print(f"\nSummary:")
    print(f"  Difficulty (any correct = pass):")
    for d in ['easy', 'medium', 'hard', 'very_hard']:
        print(f"    {d}: {summary['difficulty_distribution'][d]}")
    print(f"  Difficulty (all correct = pass):")
    for d in ['easy', 'medium', 'hard', 'very_hard']:
        print(f"    {d}: {summary['strict_difficulty_distribution'][d]}")
    print(f"\n  Model Stats:")
    for model, stats in model_stats.items():
        print(
            f"    {model}: {100*stats['mean_success_rate']:.1f}% ± {100*stats['std_success_rate']:.1f}%"
        )
        print(
            f"      any_correct: {stats['problems_any_correct']}, all_correct: {stats['problems_all_correct']}, all_wrong: {stats['problems_all_wrong']}"
        )

    return output_data


def main():
    global ACTIVE_MODEL_MAP, MODEL_VERSION

    parser = argparse.ArgumentParser(
        description='Run robust baseline evaluation')
    parser.add_argument('--dataset',
                        type=str,
                        choices=DATASETS + ['math'],
                        help='Dataset to evaluate')
    parser.add_argument('--subject',
                        type=str,
                        choices=['algebra', 'geometry', 'number_theory'],
                        help='MATH subject (required if --dataset math)')
    parser.add_argument('--all', action='store_true', help='Run all datasets')
    parser.add_argument('--version',
                        type=str,
                        default='v4.5',
                        choices=['v3.5', 'v4.5'],
                        help='Model version (default: v4.5)')
    parser.add_argument('--use-old-models',
                        action='store_true',
                        help='Use old model IDs (v3.5)')
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help=
        'Output directory (default: results/model-baselines-robust/{version})')
    parser.add_argument('--trials',
                        type=int,
                        default=3,
                        help='Number of trials per model (default: 3)')
    parser.add_argument('--parallel',
                        type=int,
                        default=1,
                        help='Parallel workers for trials')
    parser.add_argument('--limit',
                        type=int,
                        default=None,
                        help='Limit problems (for testing)')
    parser.add_argument(
        '--models',
        type=str,
        default=None,
        help=
        'Comma-separated list of models to run (default: haiku,sonnet,opus)')

    args = parser.parse_args()

    # Override MODELS if specified
    global MODELS
    if args.models:
        MODELS = [m.strip() for m in args.models.split(',')]

    # Set model version
    if args.use_old_models or args.version == 'v3.5':
        ACTIVE_MODEL_MAP = MODEL_ALIAS_MAP_old
        MODEL_VERSION = 'v3.5'
    else:
        ACTIVE_MODEL_MAP = MODEL_ALIAS_MAP
        MODEL_VERSION = 'v4.5'

    # Set output directory
    output_dir = Path(args.output_dir) if args.output_dir else Path(
        f'results/model-baselines-robust/{MODEL_VERSION}')

    # Determine datasets
    if args.all:
        datasets = DATASETS
    elif args.dataset == 'math':
        if args.subject:
            datasets = [f'math_{args.subject}']
        else:
            datasets = ['math_algebra', 'math_geometry', 'math_number_theory']
    elif args.dataset:
        datasets = [args.dataset]
    else:
        print("Please specify --dataset or --all")
        return

    print(f"Robust Baseline Experiment")
    print(f"Version: {MODEL_VERSION}")
    print(f"Trials: {args.trials}")
    print(f"Datasets: {datasets}")
    print(f"Output: {output_dir}")

    for dataset in datasets:
        try:
            if dataset == 'gsm8k':
                run_gsm8k_robust(output_dir, args.trials, args.parallel,
                                 args.limit)
            elif dataset.startswith('math_'):
                subject = dataset.replace('math_', '')
                run_math_robust(subject, output_dir, args.trials,
                                args.parallel, args.limit)
            elif dataset == 'gpqa_mc':
                run_gpqa_mc_robust(output_dir, args.trials, args.parallel,
                                   args.limit)
            elif dataset == 'gpqa_freeform':
                run_gpqa_freeform_robust(output_dir, args.trials,
                                         args.parallel, args.limit)
            elif dataset == 'aime':
                run_aime_robust(output_dir, args.trials, args.parallel,
                                args.limit)
            elif dataset == 'hle':
                run_hle_robust(output_dir, args.trials, args.parallel,
                               args.limit)
            elif dataset == 'mbpp':
                run_mbpp_robust(output_dir, args.trials, args.parallel,
                                args.limit)
            else:
                print(
                    f"Dataset {dataset} not yet implemented for robust baseline"
                )

        except Exception as e:
            print(f"Error on {dataset}: {e}")
            import traceback
            traceback.print_exc()

    print("\nDone!")


if __name__ == "__main__":
    main()
