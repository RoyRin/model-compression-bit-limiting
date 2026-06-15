#!/usr/bin/env python3
"""
Clean QA Compression Sweep - 3^3 model combinations per dataset.

Runs the iterative Q&A compression approach across all combinations of:
- SLM (Small Language Model): haiku, sonnet, opus
- LLM (Large Language Model for answers): haiku, sonnet, opus
- Q (Question generator): haiku, sonnet, opus

Total: 27 combinations per dataset.

Uses baseline files for initial correctness (no re-evaluation needed).
Only evaluates on non-easy problems (medium, hard, very_hard).
Uses Anthropic Message Batches API for efficiency.

Usage:
    python run_qa_sweep.py --dataset gsm8k
    python run_qa_sweep.py --dataset math --subject algebra
    python run_qa_sweep.py --dataset gpqa --format mc
    python run_qa_sweep.py --all
"""

import json
import time
import argparse
import re
import os
import sys
import tempfile
import subprocess
import threading
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import anthropic

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datasets import load_dataset
from utils.llm_api import get_anthropic_key
from lossy_compression.judge import judge_freeform_answer
from lossy_compression.benchmarks.hle import (extract_hle_answer,
                                              check_hle_answer)
from lossy_compression.benchmarks.aime import (extract_aime_answer,
                                               check_aime_answer)

# Configuration
MODELS = ['haiku', 'sonnet', 'opus']
MODELS_WITH_GPT_OSS = ['haiku', 'sonnet', 'opus', 'gpt-oss']
MODEL_IDS_old = {
    'haiku': 'claude-3-5-haiku-20241022',
    'sonnet': 'claude-sonnet-4-20250514',
    'opus': 'claude-opus-4-20250514',
    'gpt-oss': 'openai/gpt-oss-120b',
}
MODEL_IDS_new = {
    'haiku': 'claude-haiku-4-5-20251001',
    'sonnet': 'claude-sonnet-4-5-20250929',
    'opus': 'claude-opus-4-5-20251101',
    'gpt-oss': 'openai/gpt-oss-120b',
}
# Active model map (switched via --use-old-models flag)
MODEL_IDS = MODEL_IDS_new  # Default to 4.5 models
MODEL_VERSION = "v4.5"  # For output filenames

# Reasoning models need extra tokens for chain-of-thought
REASONING_MODELS = {'gpt-oss'}


def get_max_tokens(model: str, task: str) -> int:
    """Get appropriate max_tokens for a model and task.

    Reasoning models (like gpt-oss) need extra tokens for internal reasoning.
    """
    base_tokens = {
        'proposal': 2048,
        'questions': 1024,
        'answer': 10,  # Yes/No answers
        'update': 2048,
    }

    tokens = base_tokens.get(task, 1024)

    # Reasoning models need extra tokens for chain-of-thought
    # They can use 150-200 tokens just for reasoning before output
    if model in REASONING_MODELS:
        if task == 'answer':
            tokens = 500  # Yes/No answers need room for reasoning
        else:
            tokens = tokens + 250  # Other tasks get 250 extra

    return tokens


NUM_QUESTIONS = 10
DIFFICULTIES = ['medium', 'hard', 'very_hard']  # Non-easy only
POLL_INTERVAL = 30  # seconds

# Retry configuration
MAX_RETRIES = 5
INITIAL_BACKOFF = 5  # seconds
MAX_BACKOFF = 300  # 5 minutes

# Global flags for execution mode
USE_ITERATIVE = False  # If True, use iterative API calls instead of batch
PARALLEL_WORKERS = 1  # Number of parallel workers for iterative mode
TRIAL = None  # Trial number for variance experiments (changes seed and output filenames)
BLC_QA_ONLY = False  # If True, only run BLC/QA/QA+ for haiku SLM (3 combinations)


def get_output_filename(dataset: str, combo_name: str) -> str:
    """Generate output filename with optional trial suffix."""
    if TRIAL is not None:
        return f"{dataset}_{MODEL_VERSION}_{combo_name}_trial{TRIAL}.json"
    return f"{dataset}_{MODEL_VERSION}_{combo_name}.json"


# Default baseline files (old models)
BASELINE_FILES = {
    'gsm8k': 'lossy_compression/results/gsm8k_all_models_20260115_215021.json',
    'math_algebra':
    'lossy_compression/results/math_all_models_algebra_20260115_001427.json',
    'math_geometry':
    'lossy_compression/results/math_all_models_geometry_20260114_213358.json',
    'math_number_theory':
    'lossy_compression/results/math_all_models_number_theory_20260114_213908.json',
    'gpqa_mc':
    'lossy_compression/results/gpqa_all_models_20260115_185611.json',
    'gpqa_freeform':
    'lossy_compression/results/gpqa_freeform_all_models_20260115_184911.json',
    'mbpp':
    'lossy_compression/results/mbpp_all_models_test_20260115_154846.json',
}

# Baseline file patterns for auto-discovery
BASELINE_PATTERNS = {
    # Matches both old format (gsm8k_all_models_*.json) and new format (gsm8k_v3.5_*.json)
    'gsm8k': 'gsm8k_*.json',
    'math_algebra': 'math_algebra_*.json',
    'math_geometry': 'math_geometry_*.json',
    'math_number_theory': 'math_number_theory_*.json',
    'gpqa_mc': 'gpqa_mc_*.json',
    'gpqa_freeform': 'gpqa_freeform_*.json',
    'mbpp': 'mbpp_*.json',
    'mmlu_pro': 'mmlu_pro_*.json',
    'hle': 'hle_*.json',
    'aime': 'aime_*.json',
}


def find_baseline_file(dataset: str, baseline_dir: Path) -> Optional[Path]:
    """Find the most recent baseline file for a dataset in the given directory."""
    pattern = BASELINE_PATTERNS.get(dataset)
    if not pattern:
        return None

    files = list(baseline_dir.glob(pattern))
    if not files:
        return None

    # Return the most recent file
    return max(files, key=lambda f: f.stat().st_mtime)


@dataclass
class Problem:
    """A single problem for QA evaluation."""
    idx: int
    dataset: str
    question: str
    gold_answer: str
    difficulty: str
    baseline_correct: Dict[str, bool]  # {model: correct} from baseline
    # For MBPP
    function_name: str = ''
    test_list: List[str] = field(default_factory=list)
    test_setup_code: str = ''
    # For GPQA MC
    choices: List[str] = field(default_factory=list)
    # For HLE (Humanity's Last Exam)
    answer_type: str = 'exactMatch'  # 'exactMatch' or 'multipleChoice'


@dataclass
class ProblemState:
    """State for a problem through the QA pipeline."""
    problem: Problem
    slm: str
    llm: str
    q_model: str

    # Pipeline state
    initial_answer: str = ''
    initial_reasoning: str = ''
    initial_correct: bool = False  # From baseline

    questions: List[str] = field(default_factory=list)
    answers: List[str] = field(default_factory=list)

    final_answer: str = ''
    final_reasoning: str = ''
    final_correct: bool = False

    error: str = ''


def load_problems(dataset: str,
                  baseline_dir: Optional[Path] = None,
                  hle_very_hard_limit: Optional[int] = None,
                  use_strict_difficulty: bool = False,
                  difficulties: Optional[List[str]] = None,
                  slm_model: Optional[str] = None) -> List[Problem]:
    """Load problems from baseline file, filtering by difficulty.

    Args:
        dataset: Dataset name
        baseline_dir: Directory containing baseline files
        hle_very_hard_limit: For HLE only - limit very_hard problems to this number
                            (keeps all medium and hard problems)
        use_strict_difficulty: If True, use 'strict_difficulty' field from robust baselines
                              (all correct = pass, vs any correct = pass)
        difficulties: List of difficulty levels to include. Defaults to DIFFICULTIES
                     (non-easy: medium, hard, very_hard).
        slm_model: If set to a non-haiku model (e.g., 'gpt-oss'), reclassify difficulty
                   relative to that model. 'easy' = slm correct, 'hard' = slm wrong + opus
                   correct, 'very_hard' = both wrong. Loads all problems first, then
                   reclassifies and filters.
    """
    final_difficulties = difficulties if difficulties is not None else DIFFICULTIES
    # When reclassifying, load ALL problems first, then filter after reclassification
    if slm_model and slm_model != 'haiku':
        allowed_difficulties = ['easy', 'medium', 'hard', 'very_hard']
    else:
        allowed_difficulties = final_difficulties

    if baseline_dir:
        # Try to find baseline file in the specified directory
        baseline_path = find_baseline_file(dataset, baseline_dir)
        if not baseline_path:
            raise ValueError(
                f"No baseline file found for {dataset} in {baseline_dir}")
    else:
        # Use default baseline files
        baseline_path = BASELINE_FILES.get(dataset)
        if not baseline_path or not Path(baseline_path).exists():
            raise ValueError(
                f"Baseline file not found for {dataset}: {baseline_path}")
        baseline_path = Path(baseline_path)

    with open(baseline_path) as f:
        data = json.load(f)

    # For MBPP, also load the dataset to get test cases
    mbpp_data = {}
    if dataset == 'mbpp':
        ds = load_dataset('google-research-datasets/mbpp', 'sanitized')
        mbpp_data = {p['task_id']: p for p in ds['test']}

    # For GPQA MC, load the dataset for choices
    gpqa_data = {}
    if dataset == 'gpqa_mc':
        ds = load_dataset('Idavidrein/gpqa', 'gpqa_diamond')
        gpqa_data = {i: p for i, p in enumerate(ds['train'])}

    problems = []
    for r in data['results']:
        # Use strict_difficulty for robust baselines if requested
        if use_strict_difficulty:
            difficulty = r.get('strict_difficulty',
                               r.get('difficulty', 'unknown'))
        else:
            difficulty = r.get('difficulty', 'unknown')
        if difficulty not in allowed_difficulties:
            continue

        # Get baseline correctness for each model
        baseline_correct = {}
        for model in MODELS:
            baseline_correct[model] = r.get('models',
                                            {}).get(model,
                                                    {}).get('correct', False)
        # Also load any extra models (e.g., gpt-oss) present in the data
        for model_name in r.get('models', {}):
            if model_name not in baseline_correct:
                baseline_correct[model_name] = r['models'][model_name].get(
                    'correct', False)

        # Extract problem data based on dataset
        if dataset == 'gsm8k':
            problem = Problem(
                idx=r.get('problem_idx', len(problems)),
                dataset=dataset,
                question=r.get('question', ''),
                gold_answer=str(r.get('gold_answer', '')),
                difficulty=difficulty,
                baseline_correct=baseline_correct,
            )
        elif dataset.startswith('math_'):
            problem = Problem(
                idx=r.get('problem_idx', len(problems)),
                dataset=dataset,
                question=r.get('question', r.get('problem', '')),
                gold_answer=str(r.get('gold_answer', '')),
                difficulty=difficulty,
                baseline_correct=baseline_correct,
            )
        elif dataset == 'gpqa_mc':
            idx = r.get('problem_idx', len(problems))
            choices = []
            if idx in gpqa_data:
                gp = gpqa_data[idx]
                choices = [
                    gp.get('Correct Answer', ''),
                    gp.get('Incorrect Answer 1', ''),
                    gp.get('Incorrect Answer 2', ''),
                    gp.get('Incorrect Answer 3', ''),
                ]
            problem = Problem(
                idx=idx,
                dataset=dataset,
                question=r.get('question', ''),
                gold_answer=str(r.get('gold_answer', '')),
                difficulty=difficulty,
                baseline_correct=baseline_correct,
                choices=choices,
            )
        elif dataset == 'gpqa_freeform':
            problem = Problem(
                idx=r.get('problem_idx', len(problems)),
                dataset=dataset,
                question=r.get('question', ''),
                gold_answer=str(r.get('gold_answer', '')),
                difficulty=difficulty,
                baseline_correct=baseline_correct,
            )
        elif dataset == 'mmlu_pro':
            problem = Problem(
                idx=r.get('problem_idx', len(problems)),
                dataset=dataset,
                question=r.get('question', ''),
                gold_answer=str(r.get('gold_answer', '')),
                difficulty=difficulty,
                baseline_correct=baseline_correct,
            )
        elif dataset == 'hle':
            # HLE stores answer_type for MC vs exactMatch
            problem = Problem(
                idx=r.get('problem_idx', len(problems)),
                dataset=dataset,
                question=r.get('question', ''),
                gold_answer=str(r.get('answer', '')),
                difficulty=difficulty,
                baseline_correct=baseline_correct,
            )
            # Store answer_type in the problem for evaluation
            problem.answer_type = r.get('answer_type', 'exactMatch')
        elif dataset == 'aime':
            # AIME problems have integer answers 0-999
            problem = Problem(
                idx=r.get('problem_idx', len(problems)),
                dataset=dataset,
                question=r.get('problem', ''),
                gold_answer=str(r.get('gold_answer', '')),
                difficulty=difficulty,
                baseline_correct=baseline_correct,
            )
        elif dataset == 'mbpp':
            task_id = r.get('task_id')
            if task_id and task_id in mbpp_data:
                mbpp_p = mbpp_data[task_id]
                func_name = extract_function_name(mbpp_p.get('test_list', []))
                problem = Problem(
                    idx=r.get('problem_idx', len(problems)),
                    dataset=dataset,
                    question=mbpp_p['prompt'],
                    gold_answer='[code]',
                    difficulty=difficulty,
                    baseline_correct=baseline_correct,
                    function_name=func_name or '',
                    test_list=mbpp_p.get('test_list', []),
                    test_setup_code=mbpp_p.get('test_setup_code', ''),
                )
            else:
                continue
        else:
            continue

        problems.append(problem)

    # For HLE, optionally subsample very_hard problems
    if dataset == 'hle' and hle_very_hard_limit is not None:
        import random
        seed = 42 if TRIAL is None else 42 + TRIAL
        random.seed(seed)  # Reproducible sampling

        medium_hard = [
            p for p in problems if p.difficulty in ('medium', 'hard')
        ]
        very_hard = [p for p in problems if p.difficulty == 'very_hard']

        if len(very_hard) > hle_very_hard_limit:
            very_hard_sampled = random.sample(very_hard, hle_very_hard_limit)
            print(f"HLE subsampling: keeping {len(medium_hard)} medium+hard, "
                  f"{len(very_hard_sampled)}/{len(very_hard)} very_hard")
            problems = medium_hard + very_hard_sampled
        else:
            print(
                f"HLE: {len(medium_hard)} medium+hard, {len(very_hard)} very_hard (no subsampling needed)"
            )

    # Reclassify difficulty relative to a non-haiku SLM (e.g., gpt-oss)
    if slm_model and slm_model != 'haiku':
        before = len(problems)
        reclassified = []
        for p in problems:
            slm_correct = p.baseline_correct.get(slm_model, False)
            opus_correct = p.baseline_correct.get('opus', False)

            if slm_correct:
                p.difficulty = 'easy'
            elif opus_correct:
                p.difficulty = 'hard'
            else:
                p.difficulty = 'very_hard'

            if p.difficulty in final_difficulties:
                reclassified.append(p)

        problems = reclassified
        print(
            f"  Reclassified for SLM={slm_model}: {before} total -> "
            f"{len(problems)} non-easy ({sum(1 for p in problems if p.difficulty == 'hard')} hard, "
            f"{sum(1 for p in problems if p.difficulty == 'very_hard')} very_hard)"
        )

    return problems


def extract_function_name(test_list: List[str]) -> Optional[str]:
    """Extract function name from test cases."""
    if not test_list:
        return None
    match = re.search(r'assert\s+(\w+)\s*\(', test_list[0])
    return match.group(1) if match else None


# =============================================================================
# Batch API helpers with retry logic
# =============================================================================


def retry_with_backoff(func, *args, **kwargs):
    """Retry a function with exponential backoff."""
    backoff = INITIAL_BACKOFF
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            return func(*args, **kwargs)
        except anthropic.RateLimitError as e:
            last_error = e
            wait_time = min(backoff * (2**attempt), MAX_BACKOFF)
            print(
                f"\n    Rate limited, waiting {wait_time}s (attempt {attempt + 1}/{MAX_RETRIES})..."
            )
            time.sleep(wait_time)
        except anthropic.APIConnectionError as e:
            last_error = e
            wait_time = min(backoff * (2**attempt), MAX_BACKOFF)
            print(
                f"\n    Connection error, waiting {wait_time}s (attempt {attempt + 1}/{MAX_RETRIES})..."
            )
            time.sleep(wait_time)
        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                # Server error, retry
                last_error = e
                wait_time = min(backoff * (2**attempt), MAX_BACKOFF)
                print(
                    f"\n    Server error {e.status_code}, waiting {wait_time}s (attempt {attempt + 1}/{MAX_RETRIES})..."
                )
                time.sleep(wait_time)
            else:
                # Client error, don't retry
                raise

    raise last_error


def submit_batch(client: anthropic.Anthropic, requests: List[Dict]) -> str:
    """Submit batch to Anthropic API and return batch ID with retry."""

    def _submit():
        batch = client.messages.batches.create(requests=requests)
        return batch.id

    return retry_with_backoff(_submit)


def poll_batch(client: anthropic.Anthropic, batch_id: str) -> Dict:
    """Poll batch status until complete with retry on transient errors."""
    consecutive_errors = 0
    max_consecutive_errors = 10

    while True:
        try:
            batch = client.messages.batches.retrieve(batch_id)
            consecutive_errors = 0  # Reset on success

            status = batch.processing_status
            counts = batch.request_counts

            print(
                f"    [{counts.succeeded + counts.errored}/{counts.processing + counts.succeeded + counts.errored}] "
                f"processing: {counts.processing}",
                end='\r')

            if status == 'ended':
                print()
                return {
                    'status': status,
                    'succeeded': counts.succeeded,
                    'errored': counts.errored,
                }

        except (anthropic.RateLimitError, anthropic.APIConnectionError,
                anthropic.APIStatusError) as e:
            consecutive_errors += 1
            if consecutive_errors >= max_consecutive_errors:
                raise RuntimeError(
                    f"Too many consecutive errors polling batch: {e}")
            wait_time = min(INITIAL_BACKOFF * (2**consecutive_errors),
                            MAX_BACKOFF)
            print(
                f"\n    Poll error, waiting {wait_time}s ({consecutive_errors}/{max_consecutive_errors})..."
            )
            time.sleep(wait_time)
            continue

        time.sleep(POLL_INTERVAL)


def download_batch_results(client: anthropic.Anthropic,
                           batch_id: str) -> Dict[str, str]:
    """Download and parse batch results with retry."""

    def _download():
        results = {}
        for result in client.messages.batches.results(batch_id):
            if result.result.type == 'succeeded':
                content = result.result.message.content[
                    0].text if result.result.message.content else ""
                results[result.custom_id] = content
        return results

    return retry_with_backoff(_download)


def run_batch_step(client: anthropic.Anthropic, requests: List[Dict],
                   step_name: str) -> Dict[str, str]:
    """Run a batch step: submit, poll, download with full error handling."""
    if not requests:
        return {}

    print(f"  {step_name}: {len(requests)} requests...")

    try:
        batch_id = submit_batch(client, requests)
        print(f"    Batch ID: {batch_id}")

        poll_batch(client, batch_id)
        results = download_batch_results(client, batch_id)
        print(f"    Got {len(results)} results")

        return results

    except Exception as e:
        print(f"\n    ERROR in {step_name}: {e}")
        raise


# =============================================================================
# Iterative API helpers (for parallel execution without batch API)
# =============================================================================


def _call_openrouter(model_id: str,
                     messages: List[Dict],
                     system: str = None,
                     max_tokens: int = 1024) -> str:
    """Call a non-Anthropic model via OpenRouter with retry logic."""
    from utils.llm_api import openrouter_messages

    # OpenRouter uses system message in the messages list (not a separate param)
    full_messages = []
    if system:
        full_messages.append({'role': 'system', 'content': system})
    full_messages.extend(messages)

    for attempt in range(MAX_RETRIES):
        try:
            return openrouter_messages(
                messages=full_messages,
                model=model_id,
                max_tokens=max_tokens,
            )
        except Exception as e:
            error_msg = str(e).lower()
            if any(
                    term in error_msg for term in
                ['rate_limit', 'rate limit', '429', 'too many', 'overloaded']):
                wait_time = min(INITIAL_BACKOFF * (2**attempt), MAX_BACKOFF)
                time.sleep(wait_time)
            else:
                raise

    return ""


def call_api_iterative(client: anthropic.Anthropic,
                       model: str,
                       messages: List[Dict],
                       system: str = None,
                       max_tokens: int = 1024) -> str:
    """Make a single API call with retry logic.

    Routes to OpenRouter for non-Anthropic models (e.g., gpt-oss).
    The `client` parameter is only used for Anthropic models.
    """
    model_id = MODEL_IDS.get(model, model)

    # Route non-Anthropic models to OpenRouter
    if model_id.startswith("openai/") or model_id.startswith("openrouter/"):
        return _call_openrouter(model_id,
                                messages,
                                system=system,
                                max_tokens=max_tokens)

    for attempt in range(MAX_RETRIES):
        try:
            kwargs = {
                'model': model_id,
                'max_tokens': max_tokens,
                'messages': messages,
            }
            if system:
                kwargs['system'] = system

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


def run_qa_pipeline_single(problem: Problem, slm: str, llm: str, q_model: str,
                           client: anthropic.Anthropic) -> Dict:
    """Run the full QA pipeline for a single problem iteratively."""
    state = ProblemState(
        problem=problem,
        slm=slm,
        llm=llm,
        q_model=q_model,
        initial_correct=problem.baseline_correct.get(slm, False),
    )

    try:
        # Step 1: Initial proposal from SLM
        system = get_system_prompt(problem.dataset, 'slm')
        proposal_prompt = make_proposal_prompt(problem)
        state.initial_reasoning = call_api_iterative(
            client,
            slm, [{
                'role': 'user',
                'content': proposal_prompt
            }],
            system=system,
            max_tokens=get_max_tokens(slm, 'proposal'))
        state.initial_answer = state.initial_reasoning

        # Step 2: Generate questions from Q model
        question_prompt = make_question_prompt(problem, state.initial_answer)
        questions_response = call_api_iterative(client,
                                                q_model,
                                                [{
                                                    'role': 'user',
                                                    'content': question_prompt
                                                }],
                                                max_tokens=get_max_tokens(
                                                    q_model, 'questions'))
        questions = re.findall(r'\d+\.\s*(.+?)(?=\n\d+\.|\Z)',
                               questions_response, re.DOTALL)
        state.questions = [q.strip() for q in questions[:NUM_QUESTIONS]]
        while len(state.questions) < NUM_QUESTIONS:
            state.questions.append("Is the answer correct?")

        # Step 3: LLM answers questions
        state.answers = []
        for q in state.questions:
            answer_prompt = make_answer_prompt(problem, q)
            response = call_api_iterative(client,
                                          llm, [{
                                              'role': 'user',
                                              'content': answer_prompt
                                          }],
                                          max_tokens=get_max_tokens(
                                              llm, 'answer'))
            response_upper = response.strip().upper()
            if response_upper.startswith('YES'):
                state.answers.append('Yes')
            elif response_upper.startswith('NO'):
                state.answers.append('No')
            else:
                state.answers.append('Unknown')

        # Step 4: SLM updates based on Q&A
        update_prompt = make_update_prompt(problem, state.questions,
                                           state.answers)
        state.final_reasoning = call_api_iterative(client,
                                                   slm,
                                                   [{
                                                       'role': 'user',
                                                       'content': update_prompt
                                                   }],
                                                   system=system,
                                                   max_tokens=get_max_tokens(
                                                       slm, 'update'))
        state.final_answer = state.final_reasoning

        # Evaluate final answer
        state.final_correct = check_answer(problem, state.final_answer)

    except Exception as e:
        state.error = str(e)

    return state


def run_qa_sweep_iterative(
    problems: List[Problem],
    slm: str,
    llm: str,
    q_model: str,
    output_dir: Path,
) -> Dict[str, Any]:
    """Run iterative QA pipeline for one model combination with parallel workers."""

    print(f"\n{'='*60}")
    print(
        f"SLM: {slm}, LLM: {llm}, Q: {q_model} (ITERATIVE, {PARALLEL_WORKERS} workers)"
    )
    print(f"Problems: {len(problems)}")
    print(f"{'='*60}")

    client = anthropic.Anthropic(api_key=get_anthropic_key())
    start_time = time.time()

    states = []
    lock = threading.Lock()
    completed_count = [0]

    def process_problem(problem):
        state = run_qa_pipeline_single(problem, slm, llm, q_model, client)
        with lock:
            completed_count[0] += 1
            status = "✓" if state.final_correct else "✗"
            print(
                f"  [{completed_count[0]}/{len(problems)}] Problem {problem.idx}: {status}",
                end='\r')
        return state

    if PARALLEL_WORKERS > 1:
        with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
            futures = {
                executor.submit(process_problem, p): p
                for p in problems
            }
            for future in as_completed(futures):
                states.append(future.result())
    else:
        for p in problems:
            states.append(process_problem(p))

    print()  # Newline after progress

    elapsed = time.time() - start_time

    # Compute summary
    initial_correct = sum(1 for s in states if s.initial_correct)
    final_correct = sum(1 for s in states if s.final_correct)
    recovered = sum(1 for s in states
                    if not s.initial_correct and s.final_correct)
    lost = sum(1 for s in states if s.initial_correct and not s.final_correct)

    summary = {
        'slm': slm,
        'llm': llm,
        'q_model': q_model,
        'n_problems': len(problems),
        'initial_correct': initial_correct,
        'final_correct': final_correct,
        'recovered': recovered,
        'lost': lost,
        'initial_accuracy': initial_correct / len(problems) if problems else 0,
        'final_accuracy': final_correct / len(problems) if problems else 0,
        'time_seconds': elapsed,
    }

    print(f"\nResults:")
    print(
        f"  Initial: {initial_correct}/{len(problems)} ({100*summary['initial_accuracy']:.1f}%)"
    )
    print(
        f"  Final: {final_correct}/{len(problems)} ({100*summary['final_accuracy']:.1f}%)"
    )
    print(f"  Recovered: {recovered}, Lost: {lost}")
    print(f"  Time: {int(elapsed)}s")

    # Save results
    combo_name = f"SLM-{slm}_LLM-{llm}_Q-{q_model}"
    output_file = output_dir / get_output_filename(problems[0].dataset,
                                                   combo_name)

    results_data = {
        'summary':
        summary,
        'problems': [{
            'idx': s.problem.idx,
            'question': s.problem.question[:200],
            'gold': s.problem.gold_answer,
            'difficulty': s.problem.difficulty,
            'initial_correct': s.initial_correct,
            'final_correct': s.final_correct,
            'questions': s.questions,
            'answers': s.answers,
            'error': s.error,
        } for s in states],
    }

    with open(output_file, 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"  Saved: {output_file.name}")

    return summary


# =============================================================================
# Prompt templates
# =============================================================================


def get_system_prompt(dataset: str, role: str) -> str:
    """Get system prompt based on dataset and role."""
    if role == 'slm':
        if dataset == 'gsm8k':
            return "You are solving math problems. Show your work and put the final numerical answer after ####."
        elif dataset.startswith('math_'):
            return "You are solving math problems. Show your work and put your final answer in \\boxed{}."
        elif dataset == 'gpqa_mc':
            return "You are answering science questions. Analyze carefully and give your answer as A, B, C, or D."
        elif dataset == 'gpqa_freeform':
            return "You are answering science questions. Provide a clear, detailed answer."
        elif dataset == 'mmlu_pro':
            return "You are an expert across many academic disciplines. Provide clear, accurate answers."
        elif dataset == 'mbpp':
            return "You are writing Python code. Provide only the function definition."
        elif dataset == 'hle':
            return "You are a highly knowledgeable expert. Answer questions precisely and concisely. For multiple choice, give only the letter. For open-ended questions, give a brief, direct answer."
        elif dataset == 'aime':
            return "You are an expert competition mathematician. Solve AIME problems carefully. Show your work and give the final answer as an integer from 0 to 999."
    return "You are a helpful assistant."


def make_custom_id(prefix: str, state: ProblemState) -> str:
    """Create unique custom_id."""
    return f"{prefix}_{state.problem.dataset}_{state.problem.idx}_{state.slm}_{state.llm}_{state.q_model}"


def make_proposal_prompt(problem: Problem) -> str:
    """Create prompt for SLM to propose initial answer."""
    prompt = f"Solve this problem:\n\n{problem.question}"
    if problem.function_name:
        prompt += f"\n\nName your function: {problem.function_name}"
    if problem.choices:
        prompt += "\n\nChoices:\n"
        for i, c in enumerate(problem.choices):
            prompt += f"{chr(65+i)}. {c}\n"
    return prompt


def make_question_prompt(problem: Problem, initial_answer: str) -> str:
    """Create prompt for Q model to generate questions."""
    return f"""Given this problem and an initial attempt, generate {NUM_QUESTIONS} yes/no questions that would help clarify the correct solution approach.

Problem: {problem.question}

Initial attempt: {initial_answer}

Generate exactly {NUM_QUESTIONS} yes/no questions, numbered 1-{NUM_QUESTIONS}. Each question should:
- Be answerable with just "Yes" or "No"
- Help identify if the approach or answer is correct
- Focus on key steps, calculations, or concepts

Questions:"""


def make_answer_prompt(problem: Problem, question: str) -> str:
    """Create prompt for LLM to answer a question."""
    return f"""Problem: {problem.question}

Correct answer: {problem.gold_answer}

Question: {question}

Answer with only "Yes" or "No"."""


def make_update_prompt(problem: Problem, questions: List[str],
                       answers: List[str]) -> str:
    """Create prompt for SLM to update answer based on Q&A."""
    qa_text = "\n".join(
        [f"Q: {q}\nA: {a}" for q, a in zip(questions, answers)])

    prompt = f"""Problem: {problem.question}

You asked some clarifying questions and got these answers:
{qa_text}

Based on this information, provide your final answer to the problem."""

    if problem.function_name:
        prompt += f"\n\nName your function: {problem.function_name}"
    if problem.choices:
        prompt += "\n\nChoices:\n"
        for i, c in enumerate(problem.choices):
            prompt += f"{chr(65+i)}. {c}\n"

    return prompt


# =============================================================================
# Answer checking
# =============================================================================


def check_answer(problem: Problem, response: str) -> bool:
    """Check if the answer is correct."""
    dataset = problem.dataset

    if dataset == 'gsm8k':
        # Extract number after ####
        match = re.search(r'####\s*(\-?[\d,]+)', response)
        if match:
            extracted = match.group(1).replace(',', '')
        else:
            numbers = re.findall(r'\-?[\d,]+', response)
            extracted = numbers[-1].replace(',', '') if numbers else ''

        try:
            return float(extracted) == float(problem.gold_answer)
        except:
            return extracted == problem.gold_answer

    elif dataset.startswith('math_'):
        # Extract boxed answer
        matches = re.findall(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}',
                             response)
        extracted = matches[-1].strip() if matches else ''

        def normalize(s):
            return s.strip().lower().replace(' ', '').replace('$', '')

        return normalize(extracted) == normalize(problem.gold_answer)

    elif dataset == 'gpqa_mc':
        # Extract letter
        match = re.search(r'\b([A-D])\b', response.upper())
        extracted = match.group(1) if match else ''
        return extracted == problem.gold_answer

    elif dataset == 'gpqa_freeform':
        # Use LLM-as-judge for semantic evaluation
        return judge_freeform_answer(problem.question,
                                     problem.gold_answer,
                                     response,
                                     model_map=MODEL_IDS)

    elif dataset == 'mmlu_pro':
        # Use LLM-as-judge for semantic evaluation
        return judge_freeform_answer(problem.question,
                                     problem.gold_answer,
                                     response,
                                     model_map=MODEL_IDS)

    elif dataset == 'mbpp':
        return check_mbpp_code(response, problem)

    elif dataset == 'hle':
        # Use HLE-specific answer extraction and checking
        extracted = extract_hle_answer(response, problem.answer_type)
        return check_hle_answer(extracted, problem.gold_answer,
                                problem.answer_type)

    elif dataset == 'aime':
        # AIME answers are integers 0-999
        extracted = extract_aime_answer(response)
        return check_aime_answer(extracted, problem.gold_answer)

    return False


def check_mbpp_code(response: str, problem: Problem) -> bool:
    """Check MBPP code by running tests."""
    # Extract code
    code_match = re.search(r'```python\n(.*?)```', response, re.DOTALL)
    if code_match:
        code = code_match.group(1).strip()
    else:
        def_match = re.search(r'(def\s+\w+.*?)(?=\ndef\s|\Z)', response,
                              re.DOTALL)
        code = def_match.group(1).strip() if def_match else response.strip()

    if not code:
        return False

    # Run tests
    test_code = code + '\n\n' + '\n'.join(problem.test_list)

    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py',
                                         delete=False) as f:
            f.write(test_code)
            temp_path = f.name

        result = subprocess.run(['python3', temp_path],
                                capture_output=True,
                                timeout=5)
        os.unlink(temp_path)

        return result.returncode == 0
    except:
        return False


# =============================================================================
# Main pipeline
# =============================================================================


def run_qa_sweep_batch(
    problems: List[Problem],
    slm: str,
    llm: str,
    q_model: str,
    output_dir: Path,
) -> Dict[str, Any]:
    """Run batch QA pipeline for one model combination."""

    print(f"\n{'='*60}")
    print(f"SLM: {slm}, LLM: {llm}, Q: {q_model}")
    print(f"Problems: {len(problems)}")
    print(f"{'='*60}")

    # Initialize states
    states = []
    for p in problems:
        state = ProblemState(
            problem=p,
            slm=slm,
            llm=llm,
            q_model=q_model,
            initial_correct=p.baseline_correct.get(slm, False),
        )
        states.append(state)

    client = anthropic.Anthropic(api_key=get_anthropic_key())
    start_time = time.time()

    # Step 1: Initial proposals from SLM
    print("\nStep 1: Initial proposals")
    requests = []
    for state in states:
        system = get_system_prompt(state.problem.dataset, 'slm')
        requests.append({
            'custom_id': make_custom_id('proposal', state),
            'params': {
                'model':
                MODEL_IDS[slm],
                'max_tokens':
                get_max_tokens(slm, 'proposal'),
                'system':
                system,
                'messages': [{
                    'role': 'user',
                    'content': make_proposal_prompt(state.problem)
                }],
            },
        })

    results = run_batch_step(client, requests, 'Proposals')

    for state in states:
        key = make_custom_id('proposal', state)
        if key in results:
            state.initial_reasoning = results[key]
            # Extract answer (simple: use the full response)
            state.initial_answer = results[key]

    # Step 2: Generate questions from Q model
    print("\nStep 2: Generating questions")
    requests = []
    for state in states:
        requests.append({
            'custom_id': make_custom_id('questions', state),
            'params': {
                'model':
                MODEL_IDS[q_model],
                'max_tokens':
                get_max_tokens(q_model, 'questions'),
                'messages': [{
                    'role':
                    'user',
                    'content':
                    make_question_prompt(state.problem, state.initial_answer)
                }],
            },
        })

    results = run_batch_step(client, requests, 'Questions')

    for state in states:
        key = make_custom_id('questions', state)
        if key in results:
            # Parse numbered questions
            questions = re.findall(r'\d+\.\s*(.+?)(?=\n\d+\.|\Z)',
                                   results[key], re.DOTALL)
            state.questions = [q.strip() for q in questions[:NUM_QUESTIONS]]
            # Pad if needed
            while len(state.questions) < NUM_QUESTIONS:
                state.questions.append("Is the answer correct?")

    # Step 3: LLM answers questions (batch all questions)
    print("\nStep 3: Answering questions")
    requests = []
    for state in states:
        for i, q in enumerate(state.questions):
            requests.append({
                'custom_id': f"{make_custom_id('answer', state)}_{i}",
                'params': {
                    'model':
                    MODEL_IDS[llm],
                    'max_tokens':
                    get_max_tokens(llm, 'answer'),
                    'messages': [{
                        'role':
                        'user',
                        'content':
                        make_answer_prompt(state.problem, q)
                    }],
                },
            })

    results = run_batch_step(client, requests, 'Answers')

    for state in states:
        state.answers = []
        for i in range(len(state.questions)):
            key = f"{make_custom_id('answer', state)}_{i}"
            if key in results:
                response_upper = results[key].strip().upper()
                if response_upper.startswith('YES'):
                    state.answers.append('Yes')
                elif response_upper.startswith('NO'):
                    state.answers.append('No')
                else:
                    state.answers.append('Unknown')
            else:
                state.answers.append('Unknown')

    # Step 4: SLM updates based on Q&A
    print("\nStep 4: Updating answers")
    requests = []
    for state in states:
        system = get_system_prompt(state.problem.dataset, 'slm')
        requests.append({
            'custom_id': make_custom_id('update', state),
            'params': {
                'model':
                MODEL_IDS[slm],
                'max_tokens':
                get_max_tokens(slm, 'update'),
                'system':
                system,
                'messages': [{
                    'role':
                    'user',
                    'content':
                    make_update_prompt(state.problem, state.questions,
                                       state.answers)
                }],
            },
        })

    results = run_batch_step(client, requests, 'Updates')

    for state in states:
        key = make_custom_id('update', state)
        if key in results:
            state.final_reasoning = results[key]
            state.final_answer = results[key]
            state.final_correct = check_answer(state.problem,
                                               state.final_answer)

    elapsed = time.time() - start_time

    # Calculate summary
    initial_correct = sum(1 for s in states if s.initial_correct)
    final_correct = sum(1 for s in states if s.final_correct)
    recovered = sum(1 for s in states
                    if s.final_correct and not s.initial_correct)
    lost = sum(1 for s in states if not s.final_correct and s.initial_correct)

    print(f"\nResults:")
    print(
        f"  Initial: {initial_correct}/{len(states)} ({100*initial_correct/len(states):.1f}%)"
    )
    print(
        f"  Final: {final_correct}/{len(states)} ({100*final_correct/len(states):.1f}%)"
    )
    print(f"  Recovered: {recovered}, Lost: {lost}")
    print(f"  Time: {elapsed:.0f}s")

    summary = {
        'dataset':
        problems[0].dataset if problems else '',
        'slm':
        slm,
        'llm':
        llm,
        'qa':
        q_model,
        'total':
        len(states),
        'initial_correct':
        initial_correct,
        'final_correct':
        final_correct,
        'recovered':
        recovered,
        'lost':
        lost,
        'initial_accuracy':
        initial_correct / len(states) if states else 0,
        'final_accuracy':
        final_correct / len(states) if states else 0,
        'recovery_rate':
        recovered / (len(states) - initial_correct) if
        (len(states) - initial_correct) > 0 else 0,
        'time_seconds':
        elapsed,
    }

    # Save results
    combo_name = f"SLM-{slm}_LLM-{llm}_Q-{q_model}"
    output_file = output_dir / get_output_filename(problems[0].dataset,
                                                   combo_name)

    output_data = {
        'summary':
        summary,
        'results': [{
            'problem_idx': s.problem.idx,
            'initial_correct': s.initial_correct,
            'final_correct': s.final_correct,
            'questions': s.questions,
            'answers': s.answers,
        } for s in states],
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"  Saved: {output_file.name}")

    return summary


def run_sweep(dataset: str,
              output_dir: Path,
              limit: int = None,
              baseline_dir: Optional[Path] = None,
              hle_very_hard_limit: int = None,
              use_strict_difficulty: bool = False) -> Dict:
    """Run full 3^3 sweep for a dataset."""

    print(f"\n{'='*60}")
    print(f"Running QA Sweep: {dataset}" +
          (" (ROBUST)" if use_strict_difficulty else ""))
    print(f"{'='*60}")

    # Load problems
    problems = load_problems(dataset,
                             baseline_dir=baseline_dir,
                             hle_very_hard_limit=hle_very_hard_limit,
                             use_strict_difficulty=use_strict_difficulty)
    if limit:
        problems = problems[:limit]

    print(f"Loaded {len(problems)} non-easy problems")

    if not problems:
        print("No problems to run!")
        return {}

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate model combinations
    # For HLE, skip gpt-oss (too slow, 64->27 combinations)
    if dataset == 'hle' and 'gpt-oss' in MODELS:
        models_to_use = [m for m in MODELS if m != 'gpt-oss']
        print(f"HLE: Skipping gpt-oss (using {models_to_use})")
    else:
        models_to_use = MODELS

    combinations = [(slm, llm, q) for slm in models_to_use
                    for llm in models_to_use for q in models_to_use]

    # Filter to BLC/QA/QA+ only if requested
    if BLC_QA_ONLY:
        blc_qa_combos = [
            ('haiku', 'haiku', 'haiku'),  # BLC
            ('haiku', 'opus', 'haiku'),  # QA
            ('haiku', 'opus', 'opus'),  # QA+
        ]
        combinations = [c for c in combinations if c in blc_qa_combos]
        print(f"BLC/QA/QA+ only mode: {len(combinations)} combinations")
    else:
        print(f"Running {len(combinations)} model combinations")

    all_results = {}

    for combo_idx, (slm, llm, q_model) in enumerate(combinations):
        combo_name = f"SLM-{slm}_LLM-{llm}_Q-{q_model}"

        # Check if already exists
        output_file = output_dir / get_output_filename(dataset, combo_name)
        if output_file.exists():
            print(
                f"\n[{combo_idx+1}/{len(combinations)}] SKIP (exists): {combo_name}"
            )
            # Load existing summary
            with open(output_file) as f:
                data = json.load(f)
            all_results[combo_name] = data.get('summary', {})
            continue

        print(f"\n[{combo_idx+1}/{len(combinations)}] Running: {combo_name}")

        try:
            if USE_ITERATIVE:
                summary = run_qa_sweep_iterative(problems, slm, llm, q_model,
                                                 output_dir)
            else:
                summary = run_qa_sweep_batch(problems, slm, llm, q_model,
                                             output_dir)
            all_results[combo_name] = summary
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Save sweep summary
    summary_file = output_dir / f"sweep_summary_{MODEL_VERSION}.json"
    with open(summary_file, 'w') as f:
        json.dump(
            {
                'timestamp': datetime.now().isoformat(),
                'dataset': dataset,
                'model_version': MODEL_VERSION,
                'model_ids': MODEL_IDS,
                'n_problems': len(problems),
                'n_combinations': len(combinations),
                'results': all_results,
            },
            f,
            indent=2)

    print(f"\nSweep summary saved: {summary_file}")

    return all_results


def main():
    global USE_ITERATIVE, PARALLEL_WORKERS

    parser = argparse.ArgumentParser(description='Run QA Compression Sweep')
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        choices=['gsm8k', 'math', 'gpqa', 'mbpp', 'mmlu_pro', 'hle', 'aime'],
        help='Dataset to run')
    parser.add_argument(
        '--subject',
        type=str,
        default='all',
        choices=['all', 'algebra', 'geometry', 'number_theory'],
        help='MATH subject')
    parser.add_argument('--format',
                        type=str,
                        default='all',
                        choices=['all', 'mc', 'freeform'],
                        help='GPQA format')
    parser.add_argument('--all', action='store_true', help='Run all datasets')
    parser.add_argument('--limit',
                        type=int,
                        default=None,
                        help='Limit problems per dataset (for testing)')
    parser.add_argument(
        '--hle-very-hard-limit',
        type=int,
        default=None,
        help='For HLE: limit very_hard problems (keeps all medium+hard). '
        'E.g., --hle-very-hard-limit 200 keeps 50 medium + 156 hard + 200 very_hard = 406 total'
    )
    parser.add_argument('--output-dir',
                        type=str,
                        default='results',
                        help='Base output directory')
    parser.add_argument(
        '--baseline-dir',
        type=str,
        default=None,
        help='Directory containing baseline files (auto-discovers most recent)'
    )
    parser.add_argument('--iterative',
                        action='store_true',
                        help='Use iterative API calls instead of batch API')
    parser.add_argument(
        '--parallel',
        type=int,
        default=1,
        help='Number of parallel workers for iterative mode (default: 1)')
    parser.add_argument(
        '--use-old-models',
        action='store_true',
        help=
        'Use old model versions (3.5 haiku, sonnet 4, opus 4) instead of 4.5 models'
    )
    parser.add_argument(
        '--include-gpt-oss',
        action='store_true',
        help=
        'Include GPT-OSS-120B via OpenRouter (expands to 4^3=64 combinations)')
    parser.add_argument(
        '--robust',
        action='store_true',
        help=
        'Use robust baselines with strict difficulty (all trials correct = pass). '
        'Reads from model-baselines-robust/ and outputs to robust-qa-sweep/')
    parser.add_argument(
        '--gpt-oss-slm',
        action='store_true',
        help=
        'Run GPT-OSS as SLM: BLC (gpt-oss/gpt-oss/gpt-oss), QA (gpt-oss/opus/gpt-oss), '
        'QA+ (gpt-oss/opus/opus). Difficulty relative to GPT-OSS vs Opus.')
    parser.add_argument(
        '--trial',
        type=int,
        default=None,
        help=
        'Trial number for variance experiments. Changes random seed and adds _trial{N} to output filenames.'
    )
    parser.add_argument(
        '--num-questions',
        type=int,
        default=10,
        help='Number of yes/no questions to ask per problem (default: 10)')
    parser.add_argument(
        '--blc-qa-only',
        action='store_true',
        help='Only run BLC/QA/QA+ for haiku SLM (3 combinations instead of 27): '
        'BLC (haiku/haiku/haiku), QA (haiku/opus/haiku), QA+ (haiku/opus/opus)'
    )

    args = parser.parse_args()

    # Set global flags
    global MODEL_IDS, MODEL_VERSION, MODELS, TRIAL, NUM_QUESTIONS, BLC_QA_ONLY
    USE_ITERATIVE = args.iterative
    PARALLEL_WORKERS = args.parallel
    TRIAL = args.trial
    NUM_QUESTIONS = args.num_questions
    BLC_QA_ONLY = args.blc_qa_only

    if TRIAL is not None:
        print(f"Trial {TRIAL}: seed={42 + TRIAL}, output suffix=_trial{TRIAL}")

    if NUM_QUESTIONS != 10:
        print(f"Using {NUM_QUESTIONS} questions per problem (default: 10)")

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

    # Include GPT-OSS if requested
    if args.include_gpt_oss:
        MODELS = MODELS_WITH_GPT_OSS
        print("Including GPT-OSS-120B (4 models, 64 combinations)")
    else:
        MODELS = ['haiku', 'sonnet', 'opus']

    # Handle --robust mode: use robust baselines and output directory
    if args.robust:
        if args.baseline_dir:
            baseline_dir = Path(args.baseline_dir)
        else:
            baseline_dir = Path(
                f'results/model-baselines-robust/{MODEL_VERSION}')
        base_output = Path(
            args.output_dir) if args.output_dir != 'results' else Path(
                f'results/robust-qa-sweep/{MODEL_VERSION}')
        print(f"ROBUST MODE: Using strict difficulty from {baseline_dir}")
        print(f"ROBUST MODE: Output to {base_output}")
    else:
        base_output = Path(args.output_dir)
        baseline_dir = Path(args.baseline_dir) if args.baseline_dir else None

    # GPT-OSS SLM mode: override defaults
    if args.gpt_oss_slm:
        USE_ITERATIVE = True  # Force iterative (batch API is Anthropic-only)
        if not baseline_dir:
            baseline_dir = Path('results/model-baselines/v4.5-gptoss')
        if args.output_dir == 'results':
            base_output = Path(f'results/qa_sweep_gptoss_slm/{MODEL_VERSION}')

    datasets_to_run = []

    if args.all:
        datasets_to_run = [
            'gpqa_freeform',
            'mbpp',
            'mmlu_pro',
            'gsm8k',
            'math_algebra',
            'math_geometry',
            'math_number_theory',
            'gpqa_mc',
            'aime',
            'hle',
        ]
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
        if args.format == 'all':
            datasets_to_run = ['gpqa_mc', 'gpqa_freeform']
        else:
            datasets_to_run = [f'gpqa_{args.format}']
    elif args.dataset == 'mbpp':
        datasets_to_run = ['mbpp']
    elif args.dataset == 'mmlu_pro':
        datasets_to_run = ['mmlu_pro']
    elif args.dataset == 'hle':
        datasets_to_run = ['hle']
    elif args.dataset == 'aime':
        datasets_to_run = ['aime']

    if not datasets_to_run:
        print("Please specify --dataset or --all")
        return

    print(f"Datasets to run: {datasets_to_run}")
    print(f"Model combinations: {len(MODELS)}^3 = {len(MODELS)**3}")
    if USE_ITERATIVE:
        print(f"Mode: ITERATIVE PARALLEL ({PARALLEL_WORKERS} workers)")
    else:
        print("Mode: BATCH API")
    if baseline_dir:
        print(f"Baseline directory: {baseline_dir}")

    # GPT-OSS SLM mode: run only 3 targeted configs with GPT-OSS difficulty
    if args.gpt_oss_slm:
        GPT_OSS_CONFIGS = [
            ('gpt-oss', 'gpt-oss', 'gpt-oss'),  # BLC
            ('gpt-oss', 'opus', 'gpt-oss'),  # QA
            ('gpt-oss', 'opus', 'opus'),  # QA+
        ]

        print(f"\nGPT-OSS SLM mode:")
        print(
            f"  Configs: BLC (gpt-oss/gpt-oss/gpt-oss), QA (gpt-oss/opus/gpt-oss), QA+ (gpt-oss/opus/opus)"
        )
        print(f"  Baseline dir: {baseline_dir}")
        print(f"  Output dir: {base_output}")

        for dataset in datasets_to_run:
            print(f"\n{'='*60}")
            print(f"Dataset: {dataset} (GPT-OSS SLM)")
            print(f"{'='*60}")

            problems = load_problems(
                dataset,
                baseline_dir=baseline_dir,
                hle_very_hard_limit=args.hle_very_hard_limit,
                slm_model='gpt-oss')
            if args.limit:
                problems = problems[:args.limit]

            if not problems:
                print("No problems to run!")
                continue

            print(
                f"Loaded {len(problems)} non-easy problems (GPT-OSS difficulty)"
            )

            output_dir = base_output / f"{dataset}_qa_sweep" / "data"
            output_dir.mkdir(parents=True, exist_ok=True)

            for slm, llm, q_model in GPT_OSS_CONFIGS:
                combo_name = f"SLM-{slm}_LLM-{llm}_Q-{q_model}"
                output_file = output_dir / get_output_filename(
                    dataset, combo_name)
                if output_file.exists():
                    print(f"\n  SKIP (exists): {combo_name}")
                    continue

                try:
                    run_qa_sweep_iterative(problems, slm, llm, q_model,
                                           output_dir)
                except Exception as e:
                    print(f"ERROR: {e}")
                    import traceback
                    traceback.print_exc()

        print("\n" + "=" * 60)
        print("GPT-OSS SLM SWEEP COMPLETE")
        print("=" * 60)
        return

    for dataset in datasets_to_run:
        output_dir = base_output / f"{dataset}_qa_sweep" / "data"
        run_sweep(dataset,
                  output_dir,
                  args.limit,
                  baseline_dir=baseline_dir,
                  hle_very_hard_limit=args.hle_very_hard_limit,
                  use_strict_difficulty=args.robust)

    print("\n" + "=" * 60)
    print("ALL SWEEPS COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
