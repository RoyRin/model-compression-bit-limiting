#!/usr/bin/env python3
"""
Batch Q&A Compression Sweep for All Datasets.

Runs the 6-step batch Q&A pipeline across all model combinations:
- 3 SLM models (haiku, sonnet, opus)
- 4 LLM options (haiku, sonnet, opus, opus-oracle)
- 3 Q models (haiku, sonnet, opus)
= 36 combinations per dataset

Supported datasets:
- GSM8K: Grade school math (500 non-easy problems)
- GPQA-Freeform: Graduate-level science (83 non-easy problems)
- MBPP: Python programming (94 non-easy problems)
- HumanEval: Python programming (164 problems)

Usage:
    # Run all datasets, all combinations
    python batch_qa_sweep_all_datasets.py

    # Run specific dataset
    python batch_qa_sweep_all_datasets.py --dataset gsm8k

    # Test with limited problems
    python batch_qa_sweep_all_datasets.py --dataset gsm8k --limit 10

    # Run specific combination
    python batch_qa_sweep_all_datasets.py --dataset gsm8k --slm haiku --llm opus --qa haiku
"""

import json
import time
import argparse
import tempfile
import subprocess
import os
import re
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
import anthropic
from datasets import load_dataset

# Model configurations
SLM_MODELS = ['haiku', 'sonnet', 'opus']
LLM_MODELS = ['haiku', 'sonnet', 'opus']  # oracle handled separately
Q_MODELS = ['haiku', 'sonnet', 'opus']

MODEL_IDS = {
    'haiku': 'claude-3-5-haiku-20241022',
    'sonnet': 'claude-3-5-sonnet-20241022',
    'opus': 'claude-3-opus-20240229',
}

# Dataset configurations
DATASET_CONFIGS = {
    'gsm8k': {
        'baseline_file':
        'lossy_compression/results/gsm8k_all_models_20260115_215021.json',
        'difficulties': ['medium', 'hard', 'very_hard'],
        'answer_type': 'numeric',
        'description': 'Grade school math word problems',
    },
    'gpqa_freeform': {
        'baseline_file':
        'lossy_compression/results/gpqa_freeform_all_models_20260115_184911.json',
        'difficulties': ['medium', 'hard', 'very_hard'],
        'answer_type': 'freeform',
        'description': 'Graduate-level science questions (free-form)',
    },
    'mbpp': {
        'baseline_file':
        'lossy_compression/results/mbpp_all_models_test_20260115_172240.json',
        'difficulties': ['medium', 'hard', 'very_hard'],
        'answer_type': 'code',
        'description': 'Python programming problems (MBPP sanitized)',
    },
}

# =============================================================================
# Code execution helpers for MBPP
# =============================================================================


def extract_function_name(test_list: List[str]) -> Optional[str]:
    """Extract the expected function name from test cases."""
    if not test_list:
        return None
    match = re.search(r'assert\s+(\w+)\s*\(', test_list[0])
    return match.group(1) if match else None


def run_code_tests(code: str,
                   test_list: List[str],
                   test_setup_code: str = '',
                   timeout: int = 10) -> Dict:
    """Run test cases against the generated code.

    Returns dict with 'passed', 'total', 'all_passed' keys.
    """
    full_code = f"{test_setup_code}\n\n{code}\n\n"
    passed = 0

    for test in test_list:
        test_code = full_code + f"\n{test}"
        try:
            with tempfile.NamedTemporaryFile(mode='w',
                                             suffix='.py',
                                             delete=False) as f:
                f.write(test_code)
                temp_path = f.name
            result = subprocess.run(['python', temp_path],
                                    capture_output=True,
                                    text=True,
                                    timeout=timeout)
            os.unlink(temp_path)
            if result.returncode == 0:
                passed += 1
        except Exception:
            try:
                os.unlink(temp_path)
            except:
                pass

    return {
        'passed': passed,
        'total': len(test_list),
        'all_passed': passed == len(test_list)
    }


@dataclass
class ProblemState:
    """State for a single problem through the Q&A pipeline."""
    problem_idx: int
    dataset: str
    problem: str
    correct_answer: str
    difficulty: str

    # Model assignments
    slm_model: str = ''
    qa_model: str = ''
    llm_model: str = ''
    is_oracle: bool = False

    # Pipeline state
    slm_initial_answer: str = ''
    slm_initial_reasoning: str = ''
    initial_correct: bool = False

    questions: List[str] = field(default_factory=list)
    answers: List[str] = field(default_factory=list)

    slm_final_answer: str = ''
    slm_final_reasoning: str = ''
    final_correct: bool = False

    # For code problems (MBPP)
    function_name: str = ''
    test_list: List[str] = field(default_factory=list)
    test_setup_code: str = ''


def load_problems(dataset: str,
                  difficulties: List[str],
                  limit: Optional[int] = None) -> List[ProblemState]:
    """Load problems from baseline results file."""
    config = DATASET_CONFIGS[dataset]

    with open(config['baseline_file'], 'r') as f:
        data = json.load(f)

    results = data.get('results', data.get('problems', []))

    # For MBPP, load the actual dataset to get test cases
    mbpp_dataset = None
    if dataset == 'mbpp':
        ds = load_dataset('google-research-datasets/mbpp', 'sanitized')
        mbpp_dataset = {p['task_id']: p for p in ds['test']}

    problems = []
    for r in results:
        diff = r.get('difficulty', 'unknown')
        if diff not in difficulties:
            continue

        # Extract problem text and answer based on dataset
        if dataset == 'gsm8k':
            problem_text = r.get('question', r.get('problem', ''))
            answer = str(r.get('gold_answer', r.get('answer', '')))
            state = ProblemState(
                problem_idx=r.get('problem_idx', r.get('idx', len(problems))),
                dataset=dataset,
                problem=problem_text,
                correct_answer=answer,
                difficulty=diff,
            )

        elif dataset == 'gpqa_freeform':
            problem_text = r.get('question', r.get('problem', ''))
            answer = str(r.get('gold_answer', r.get('answer', '')))
            state = ProblemState(
                problem_idx=r.get('problem_idx', r.get('idx', len(problems))),
                dataset=dataset,
                problem=problem_text,
                correct_answer=answer,
                difficulty=diff,
            )

        elif dataset == 'mbpp':
            task_id = r.get('task_id')
            if mbpp_dataset and task_id in mbpp_dataset:
                mbpp_problem = mbpp_dataset[task_id]
                problem_text = mbpp_problem['prompt']
                test_list = mbpp_problem['test_list']
                test_setup_code = mbpp_problem.get('test_setup_code', '')
                func_name = extract_function_name(test_list)

                state = ProblemState(
                    problem_idx=r.get('problem_idx',
                                      r.get('idx', len(problems))),
                    dataset=dataset,
                    problem=problem_text,
                    correct_answer=
                    '[code]',  # Code problems don't have simple answers
                    difficulty=diff,
                    function_name=func_name or '',
                    test_list=test_list,
                    test_setup_code=test_setup_code,
                )
            else:
                continue  # Skip if can't find in dataset
        else:
            problem_text = r.get('question', r.get('problem', ''))
            answer = str(r.get('gold_answer', r.get('answer', '')))
            state = ProblemState(
                problem_idx=r.get('problem_idx', r.get('idx', len(problems))),
                dataset=dataset,
                problem=problem_text,
                correct_answer=answer,
                difficulty=diff,
            )

        problems.append(state)

        if limit and len(problems) >= limit:
            break

    return problems


def get_system_prompt(dataset: str) -> str:
    """Get system prompt based on dataset type."""
    if dataset == 'gsm8k':
        return """You are a skilled mathematician solving grade school math problems.
Provide a clear, step-by-step solution and end with the final numerical answer.
Format your final answer as: **Answer: [number]**"""

    elif dataset == 'gpqa_freeform':
        return """You are an expert scientist answering graduate-level questions.
Provide a clear, well-reasoned answer with supporting explanation.
Format your final answer as: **Answer: [your answer]**"""

    elif dataset in ['mbpp', 'humaneval']:
        return """You are an expert Python programmer.
Write clean, correct Python code that solves the given problem.
Include only the function definition, no test code or examples."""

    return "You are a helpful assistant."


def extract_answer(response: str,
                   dataset: str,
                   function_name: str = '') -> str:
    """Extract answer from response based on dataset type."""
    import re

    if dataset == 'gsm8k':
        # Look for **Answer: X** pattern
        match = re.search(r'\*\*Answer:\s*\$?([0-9,.-]+)', response)
        if match:
            return match.group(1).replace(',', '').strip()
        # Fallback: last number in response
        numbers = re.findall(r'[-]?\d+(?:,\d{3})*(?:\.\d+)?', response)
        if numbers:
            return numbers[-1].replace(',', '')
        return ''

    elif dataset == 'gpqa_freeform':
        # Look for **Answer: X** pattern
        match = re.search(r'\*\*Answer:\s*(.+?)(?:\*\*|$)', response,
                          re.DOTALL)
        if match:
            return match.group(1).strip()
        # Return last paragraph as fallback
        paragraphs = response.strip().split('\n\n')
        return paragraphs[-1].strip() if paragraphs else response.strip()

    elif dataset in ['mbpp', 'humaneval']:
        # Extract code block or function definition
        code_match = re.search(r'```python\n(.*?)```', response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        # Look for def statement
        def_match = re.search(r'(def\s+\w+.*?)(?=\ndef\s|\Z)', response,
                              re.DOTALL)
        if def_match:
            return def_match.group(1).strip()
        return response.strip()

    return response.strip()


def check_answer(response: str,
                 correct_answer: str,
                 dataset: str,
                 function_name: str = '',
                 test_list: List[str] = None,
                 test_setup_code: str = '') -> bool:
    """Check if answer is correct based on dataset type."""

    if dataset == 'gsm8k':
        extracted = extract_answer(response, dataset)
        try:
            return float(extracted) == float(correct_answer)
        except:
            return extracted == correct_answer

    elif dataset == 'gpqa_freeform':
        # For freeform, we'll use LLM evaluation later
        # Here just return False as placeholder
        return False

    elif dataset == 'mbpp':
        # Execute code and run tests
        code = extract_answer(response, dataset, function_name)
        if not code or not test_list:
            return False

        result = run_code_tests(code, test_list, test_setup_code)
        return result['all_passed']

    return False


def make_custom_id(prefix: str, state: ProblemState) -> str:
    """Create unique custom_id for batch API."""
    return f"{prefix}_{state.dataset}_{state.problem_idx}"


def create_batch_requests(states: List[ProblemState], step: str,
                          model: str) -> List[Dict]:
    """Create batch API requests for a given step."""
    requests = []
    model_id = MODEL_IDS[model]

    for state in states:
        custom_id = make_custom_id(step, state)
        system_prompt = get_system_prompt(state.dataset)

        if step == 'proposal':
            content = f"Solve this problem:\n\n{state.problem}"
            if state.function_name:
                content += f"\n\nName your function: {state.function_name}"

            messages = [{"role": "user", "content": content}]

        elif step == 'eval_initial':
            if state.dataset == 'gpqa_freeform':
                # LLM-as-judge evaluation
                messages = [{
                    "role":
                    "user",
                    "content":
                    f"""Evaluate if this answer is correct.

Question: {state.problem}

Correct answer: {state.correct_answer}

Student answer: {state.slm_initial_answer}

Is the student's answer essentially correct? Reply with exactly "CORRECT" or "INCORRECT"."""
                }]
            else:
                continue  # Skip for datasets with exact matching

        elif step == 'questions':
            # Q model generates questions
            qa_context = f"""Problem: {state.problem}

Initial solution attempt:
{state.slm_initial_reasoning}

Answer: {state.slm_initial_answer}"""

            messages = [{
                "role":
                "user",
                "content":
                f"""{qa_context}

Generate 10 clarifying questions that would help verify or improve this solution.
Focus on:
1. Checking key assumptions
2. Verifying calculations
3. Identifying potential errors
4. Exploring edge cases

Format as a numbered list (1-10)."""
            }]

        elif step == 'answers':
            # LLM answers the questions
            if state.is_oracle:
                # Oracle mode: give direct answer
                messages = [{
                    "role":
                    "user",
                    "content":
                    f"""Problem: {state.problem}

The correct answer is: {state.correct_answer}

Questions about the solution:
{chr(10).join(state.questions)}

Answer each question, keeping in mind the correct answer above."""
                }]
            else:
                messages = [{
                    "role":
                    "user",
                    "content":
                    f"""Problem: {state.problem}

Solution attempt:
{state.slm_initial_reasoning}

Questions:
{chr(10).join(state.questions)}

Answer each question to help verify or improve the solution."""
                }]

        elif step == 'update':
            # SLM updates based on Q&A
            qa_text = "\n".join([
                f"Q: {q}\nA: {a}"
                for q, a in zip(state.questions, state.answers)
            ])

            content = f"""Problem: {state.problem}

Your initial solution:
{state.slm_initial_reasoning}

Your initial answer: {state.slm_initial_answer}

Q&A feedback:
{qa_text}

Based on this feedback, provide your final solution and answer.
If your original answer was correct, you can keep it.
If you found errors, correct them."""

            if state.function_name:
                content += f"\n\nName your function: {state.function_name}"

            messages = [{"role": "user", "content": content}]

        elif step == 'eval_final':
            if state.dataset == 'gpqa_freeform':
                messages = [{
                    "role":
                    "user",
                    "content":
                    f"""Evaluate if this answer is correct.

Question: {state.problem}

Correct answer: {state.correct_answer}

Student answer: {state.slm_final_answer}

Is the student's answer essentially correct? Reply with exactly "CORRECT" or "INCORRECT"."""
                }]
            else:
                continue

        else:
            continue

        requests.append({
            "custom_id": custom_id,
            "params": {
                "model":
                model_id,
                "max_tokens":
                4096,
                "messages":
                messages,
                "system":
                system_prompt if step in ['proposal', 'update'] else None,
            }
        })

    # Filter out None system prompts
    for req in requests:
        if req["params"]["system"] is None:
            del req["params"]["system"]

    return requests


def submit_and_wait_batch(client: anthropic.Anthropic, requests: List[Dict],
                          step_name: str) -> Dict[str, str]:
    """Submit batch and wait for completion."""
    if not requests:
        return {}

    print(f"  Submitting {len(requests)} requests for {step_name}...")

    batch = client.messages.batches.create(requests=requests)
    batch_id = batch.id
    print(f"  Batch ID: {batch_id}")

    # Poll for completion
    while True:
        status = client.messages.batches.retrieve(batch_id)

        if status.processing_status == 'ended':
            print(
                f"  Batch complete: {status.request_counts.succeeded} succeeded, "
                f"{status.request_counts.errored} errors")
            break

        # Show progress
        counts = status.request_counts
        print(
            f"  Progress: {counts.succeeded + counts.errored}/{counts.processing + counts.succeeded + counts.errored} "
            f"(processing: {counts.processing})",
            end='\r')
        time.sleep(5)

    # Retrieve results
    results = {}
    for result in client.messages.batches.results(batch_id):
        if result.result.type == 'succeeded':
            content = result.result.message.content[0].text
            results[result.custom_id] = content

    return results


def run_batch_qa_sweep(
    dataset: str,
    slm_model: str,
    qa_model: str,
    llm_model: str,
    is_oracle: bool,
    output_dir: Path,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Run batch Q&A pipeline for one model combination."""

    config = DATASET_CONFIGS[dataset]
    llm_label = f"{llm_model}_oracle" if is_oracle else llm_model

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset}")
    print(f"SLM: {slm_model}, LLM: {llm_label}, Q: {qa_model}")
    print(f"{'='*60}")

    # Load problems
    states = load_problems(dataset, config['difficulties'], limit)
    print(f"Loaded {len(states)} problems")

    if not states:
        return {'error': 'No problems loaded'}

    # Assign models to states
    for s in states:
        s.slm_model = slm_model
        s.qa_model = qa_model
        s.llm_model = llm_model
        s.is_oracle = is_oracle

    client = anthropic.Anthropic()
    start_time = time.time()

    # Step 1: Initial proposals from SLM
    print("\nStep 1: Initial proposals")
    requests = create_batch_requests(states, 'proposal', slm_model)
    results = submit_and_wait_batch(client, requests, 'proposal')

    for s in states:
        custom_id = make_custom_id('proposal', s)
        if custom_id in results:
            response = results[custom_id]
            s.slm_initial_reasoning = response
            s.slm_initial_answer = extract_answer(response, dataset,
                                                  s.function_name)

            # Check if correct (for non-LLM-judged datasets)
            if dataset != 'gpqa_freeform':
                s.initial_correct = check_answer(response, s.correct_answer,
                                                 dataset, s.function_name,
                                                 s.test_list,
                                                 s.test_setup_code)

    # Step 2: Evaluate initial answers (for GPQA freeform)
    if dataset == 'gpqa_freeform':
        print("\nStep 2: Evaluating initial answers")
        requests = create_batch_requests(states, 'eval_initial', llm_model)
        results = submit_and_wait_batch(client, requests, 'eval_initial')

        for s in states:
            custom_id = make_custom_id('eval_initial', s)
            if custom_id in results:
                s.initial_correct = 'CORRECT' in results[custom_id].upper()

    initial_correct = sum(1 for s in states if s.initial_correct)
    print(
        f"Initial accuracy: {initial_correct}/{len(states)} ({100*initial_correct/len(states):.1f}%)"
    )

    # Step 3: Generate questions from Q model
    print("\nStep 3: Generating questions")
    requests = create_batch_requests(states, 'questions', qa_model)
    results = submit_and_wait_batch(client, requests, 'questions')

    import re
    for s in states:
        custom_id = make_custom_id('questions', s)
        if custom_id in results:
            # Parse numbered questions
            questions = re.findall(r'\d+\.\s*(.+?)(?=\n\d+\.|\Z)',
                                   results[custom_id], re.DOTALL)
            s.questions = [q.strip() for q in questions[:10]]

    # Step 4: LLM answers questions
    print("\nStep 4: Answering questions")
    requests = create_batch_requests(states, 'answers', llm_model)
    results = submit_and_wait_batch(client, requests, 'answers')

    for s in states:
        custom_id = make_custom_id('answers', s)
        if custom_id in results:
            # Parse answers (match to questions)
            response = results[custom_id]
            # Simple split by question numbers
            answers = re.split(r'\n\d+\.', response)
            s.answers = [a.strip() for a in answers[1:]][:len(s.questions)]
            # Pad if needed
            while len(s.answers) < len(s.questions):
                s.answers.append("")

    # Step 5: SLM updates answer
    print("\nStep 5: Updating answers")
    requests = create_batch_requests(states, 'update', slm_model)
    results = submit_and_wait_batch(client, requests, 'update')

    for s in states:
        custom_id = make_custom_id('update', s)
        if custom_id in results:
            response = results[custom_id]
            s.slm_final_reasoning = response
            s.slm_final_answer = extract_answer(response, dataset,
                                                s.function_name)

            if dataset != 'gpqa_freeform':
                s.final_correct = check_answer(response, s.correct_answer,
                                               dataset, s.function_name,
                                               s.test_list, s.test_setup_code)

    # Step 6: Evaluate final answers (for GPQA freeform)
    if dataset == 'gpqa_freeform':
        print("\nStep 6: Evaluating final answers")
        requests = create_batch_requests(states, 'eval_final', llm_model)
        results = submit_and_wait_batch(client, requests, 'eval_final')

        for s in states:
            custom_id = make_custom_id('eval_final', s)
            if custom_id in results:
                s.final_correct = 'CORRECT' in results[custom_id].upper()

    elapsed = time.time() - start_time
    final_correct = sum(1 for s in states if s.final_correct)
    recovered = sum(1 for s in states
                    if s.final_correct and not s.initial_correct)

    print(
        f"\nFinal accuracy: {final_correct}/{len(states)} ({100*final_correct/len(states):.1f}%)"
    )
    print(f"Recovered: {recovered}/{len(states) - initial_correct}")
    print(f"Time: {elapsed:.1f}s")

    # Build results
    summary = {
        'dataset':
        dataset,
        'slm':
        slm_model,
        'llm':
        llm_label,
        'qa':
        qa_model,
        'oracle':
        is_oracle,
        'total':
        len(states),
        'initial_correct':
        initial_correct,
        'final_correct':
        final_correct,
        'recovered':
        recovered,
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

    output_data = {
        'config': {
            'dataset': dataset,
            'slm': slm_model,
            'llm': llm_label,
            'qa': qa_model,
            'oracle': is_oracle,
            'difficulties': config['difficulties'],
        },
        'summary': summary,
        'results': [asdict(s) for s in states],
    }

    # Save results
    output_file = output_dir / f"{dataset}_SLM-{slm_model}_LLM-{llm_label}_Q-{qa_model}.json"
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"Saved: {output_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Batch Q&A Sweep for All Datasets')
    parser.add_argument('--dataset',
                        type=str,
                        choices=list(DATASET_CONFIGS.keys()),
                        help='Run specific dataset (default: all)')
    parser.add_argument('--slm',
                        type=str,
                        choices=SLM_MODELS,
                        help='Run specific SLM (default: all)')
    parser.add_argument('--llm',
                        type=str,
                        choices=LLM_MODELS + ['opus-oracle'],
                        help='Run specific LLM (default: all)')
    parser.add_argument('--qa',
                        type=str,
                        choices=Q_MODELS,
                        help='Run specific Q model (default: all)')
    parser.add_argument('--limit', type=int, help='Limit problems per dataset')
    parser.add_argument('--output-dir',
                        type=str,
                        default='results/batch_qa_sweep_all',
                        help='Output directory')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which datasets to run
    datasets = [args.dataset] if args.dataset else list(DATASET_CONFIGS.keys())

    # Determine model combinations
    if args.slm:
        slm_models = [args.slm]
    else:
        slm_models = SLM_MODELS

    if args.llm:
        if args.llm == 'opus-oracle':
            llm_configs = [('opus', True)]
        else:
            llm_configs = [(args.llm, False)]
    else:
        llm_configs = [(m, False) for m in LLM_MODELS] + [('opus', True)]

    if args.qa:
        qa_models = [args.qa]
    else:
        qa_models = Q_MODELS

    # Count total combinations
    total_combos = len(datasets) * len(slm_models) * len(llm_configs) * len(
        qa_models)
    print(
        f"Running {total_combos} combinations across {len(datasets)} datasets")

    all_summaries = []
    combo_idx = 0

    for dataset in datasets:
        for slm in slm_models:
            for llm, is_oracle in llm_configs:
                for qa in qa_models:
                    combo_idx += 1
                    llm_label = f"{llm}_oracle" if is_oracle else llm

                    # Check if already exists
                    output_file = output_dir / f"{dataset}_SLM-{slm}_LLM-{llm_label}_Q-{qa}.json"
                    if output_file.exists():
                        print(
                            f"\n[{combo_idx}/{total_combos}] SKIP (exists): {dataset} SLM={slm} LLM={llm_label} Q={qa}"
                        )
                        continue

                    print(
                        f"\n[{combo_idx}/{total_combos}] Running: {dataset} SLM={slm} LLM={llm_label} Q={qa}"
                    )

                    try:
                        summary = run_batch_qa_sweep(
                            dataset=dataset,
                            slm_model=slm,
                            qa_model=qa,
                            llm_model=llm,
                            is_oracle=is_oracle,
                            output_dir=output_dir,
                            limit=args.limit,
                        )
                        all_summaries.append(summary)
                    except Exception as e:
                        print(f"ERROR: {e}")
                        import traceback
                        traceback.print_exc()

    # Save overall summary
    summary_file = output_dir / 'sweep_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(
            {
                'timestamp': datetime.now().isoformat(),
                'datasets': datasets,
                'total_combinations': total_combos,
                'summaries': all_summaries,
            },
            f,
            indent=2)

    print(f"\n{'='*60}")
    print("SWEEP COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
