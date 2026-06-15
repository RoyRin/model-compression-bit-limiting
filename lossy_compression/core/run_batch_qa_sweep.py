#!/usr/bin/env python3
"""
Batch QA Compression Sweep

Runs the 6-step batch QA pipeline across all model combinations:
- 3 SLM options (haiku, sonnet, opus)
- 4 LLM options (haiku, sonnet, opus, opus-oracle)
- 3 Q options (haiku, sonnet, opus)
= 36 combinations per subject

Uses Anthropic Message Batches API for fast parallel processing.
Each combination runs through 6 batch steps (no iterative loop, 10 questions).

Usage:
    # Run all combinations for all subjects
    python run_batch_qa_sweep.py

    # Run specific subject
    python run_batch_qa_sweep.py --subject algebra

    # Test with limit
    python run_batch_qa_sweep.py --limit 10
"""

import json
import time
import argparse
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from copy import deepcopy
import anthropic

from utils.llm_api import get_anthropic_key

# Model configurations
MODELS = {
    'haiku': 'claude-3-5-haiku-20241022',
    'sonnet': 'claude-sonnet-4-20250514',
    'opus': 'claude-opus-4-20250514',
}

# Sweep configuration
SLM_OPTIONS = ['haiku', 'sonnet', 'opus']
LLM_OPTIONS = ['haiku', 'sonnet', 'opus']  # oracle handled separately
Q_OPTIONS = ['haiku', 'sonnet', 'opus']
SUBJECTS = ['algebra', 'geometry', 'number_theory']

NUM_QUESTIONS = 10
POLL_INTERVAL = 30

# Baseline results directory
BASELINE_DIR = Path(__file__).parent.parent / 'results'


@dataclass
class ProblemState:
    """Track state for each problem through the pipeline."""
    problem_idx: int
    problem: str
    correct_answer: str
    dataset: str = ""
    difficulty: str = ""

    # Config for this run
    slm_model: str = ""
    qa_model: str = ""
    llm_model: str = ""
    is_oracle: bool = False

    # Results
    slm_initial_answer: Optional[str] = None
    slm_initial_reasoning: Optional[str] = None
    initial_correct: Optional[bool] = None
    questions: List[str] = field(default_factory=list)
    answers: List[str] = field(default_factory=list)
    slm_final_answer: Optional[str] = None
    slm_final_reasoning: Optional[str] = None
    final_correct: Optional[bool] = None


def make_custom_id(prefix: str, state: ProblemState) -> str:
    """Create unique custom_id."""
    return f"{prefix}_{state.dataset}_{state.problem_idx}_{state.slm_model}_{state.llm_model}_{state.qa_model}"


def submit_batch(requests: List[Dict]) -> str:
    """Submit batch to Anthropic API."""
    client = anthropic.Anthropic(api_key=get_anthropic_key())
    batch = client.messages.batches.create(requests=requests)
    return batch.id


def poll_batch(batch_id: str, silent: bool = False) -> Dict:
    """Poll batch until complete."""
    client = anthropic.Anthropic(api_key=get_anthropic_key())

    while True:
        batch = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts

        if not silent:
            print(
                f"    {batch.processing_status} | {counts.succeeded}/{counts.processing + counts.succeeded}"
            )

        if batch.processing_status == 'ended':
            return {'succeeded': counts.succeeded, 'errored': counts.errored}

        time.sleep(POLL_INTERVAL)


def download_results(batch_id: str) -> Dict[str, Any]:
    """Download batch results."""
    client = anthropic.Anthropic(api_key=get_anthropic_key())

    results = {}
    for result in client.messages.batches.results(batch_id):
        if result.result.type == 'succeeded':
            content = result.result.message.content[
                0].text if result.result.message.content else ""
            results[result.custom_id] = {'success': True, 'content': content}
        else:
            results[result.custom_id] = {
                'success': False,
                'error': str(result.result)
            }

    return results


def run_batch_step(requests: List[Dict], step_name: str) -> Dict[str, Any]:
    """Run a batch step."""
    if not requests:
        return {}

    print(f"  {step_name}: {len(requests)} requests...", end=" ", flush=True)
    batch_id = submit_batch(requests)
    poll_batch(batch_id, silent=True)
    results = download_results(batch_id)
    print(f"done")
    return results


# =============================================================================
# Prompts and parsing
# =============================================================================


def make_proposal_prompt(problem: str) -> str:
    return f"""Solve this problem step by step. At the end, provide your final answer on a new line starting with "ANSWER: ".

Problem:
{problem}"""


def make_evaluation_prompt(problem: str, proposed: str, correct: str) -> str:
    return f"""You are evaluating if a proposed answer to a math problem is correct.

Problem:
{problem}

Proposed Answer: {proposed}
Correct Answer: {correct}

Is the proposed answer mathematically equivalent to the correct answer?
Respond with only "CORRECT" or "INCORRECT"."""


def make_question_prompt(problem: str, proposed: str, correct: str) -> str:
    return f"""A student attempted this problem but got the wrong answer. Generate {NUM_QUESTIONS} specific questions that would help them arrive at the correct answer.

Problem:
{problem}

Student's Wrong Answer: {proposed}
Correct Answer: {correct}

Format as a numbered list:
1. [Question 1]
...
{NUM_QUESTIONS}. [Question {NUM_QUESTIONS}]"""


def make_answer_prompt(problem: str, questions: List[str]) -> str:
    q_text = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
    return f"""Answer each question about this problem. Give brief, direct answers.

Problem:
{problem}

Questions:
{q_text}

Format as:
1. [Answer 1]
...
{len(questions)}. [Answer {len(questions)}]"""


def make_update_prompt(problem: str, initial: str,
                       qa_pairs: List[Tuple[str, str]]) -> str:
    qa_text = "\n".join(f"Q: {q}\nA: {a}\n" for q, a in qa_pairs)
    return f"""You previously attempted this problem and got: {initial}

Here are some hints:
{qa_text}

Reconsider the problem and provide your final answer.

Problem:
{problem}

Think step by step, then provide your final answer starting with "ANSWER: "."""


def extract_answer(content: str) -> Optional[str]:
    """Extract answer from response."""
    if not content:
        return None
    if 'ANSWER:' in content:
        return content.split('ANSWER:')[-1].strip()
    if '\\boxed{' in content:
        match = re.search(r'\\boxed\{([^}]+)\}', content)
        if match:
            return match.group(1)
    lines = [l.strip() for l in content.strip().split('\n') if l.strip()]
    return lines[-1] if lines else None


def parse_numbered_list(content: str) -> List[str]:
    """Parse numbered list from response."""
    items = []
    for line in content.split('\n'):
        match = re.match(r'^\d+[\.\)]\s*(.+)$', line.strip())
        if match:
            items.append(match.group(1))
    return items


# =============================================================================
# Pipeline steps
# =============================================================================


def run_proposals(states: List[ProblemState]) -> None:
    """Step 1: SLM proposals."""
    requests = []
    for s in states:
        requests.append({
            'custom_id': make_custom_id("prop", s),
            'params': {
                'model':
                MODELS[s.slm_model],
                'max_tokens':
                2048,
                'messages': [{
                    'role': 'user',
                    'content': make_proposal_prompt(s.problem)
                }],
            },
        })

    results = run_batch_step(requests, "Proposals")

    for s in states:
        key = make_custom_id("prop", s)
        if key in results and results[key]['success']:
            s.slm_initial_reasoning = results[key]['content']
            s.slm_initial_answer = extract_answer(results[key]['content'])


def run_evaluations(states: List[ProblemState],
                    use_oracle: bool = False) -> None:
    """Step 2: LLM evaluations."""
    requests = []
    for s in states:
        if s.slm_initial_answer:
            if use_oracle or s.is_oracle:
                # Oracle mode: directly compare
                continue
            requests.append({
                'custom_id': make_custom_id("eval", s),
                'params': {
                    'model':
                    MODELS[s.llm_model],
                    'max_tokens':
                    64,
                    'messages': [{
                        'role':
                        'user',
                        'content':
                        make_evaluation_prompt(s.problem, s.slm_initial_answer,
                                               s.correct_answer)
                    }],
                },
            })

    results = run_batch_step(requests, "Evaluations")

    for s in states:
        if s.is_oracle:
            # Oracle: exact match
            s.initial_correct = s.slm_initial_answer == s.correct_answer
        else:
            key = make_custom_id("eval", s)
            if key in results and results[key]['success']:
                s.initial_correct = 'CORRECT' in results[key]['content'].upper(
                )
            else:
                s.initial_correct = False


def run_questions(states: List[ProblemState]) -> None:
    """Step 3: Generate questions for wrong answers."""
    wrong = [s for s in states if s.initial_correct == False]
    if not wrong:
        return

    requests = []
    for s in wrong:
        requests.append({
            'custom_id': make_custom_id("q", s),
            'params': {
                'model':
                MODELS[s.qa_model],
                'max_tokens':
                1024,
                'messages': [{
                    'role':
                    'user',
                    'content':
                    make_question_prompt(s.problem, s.slm_initial_answer,
                                         s.correct_answer)
                }],
            },
        })

    results = run_batch_step(requests, "Questions")

    for s in wrong:
        key = make_custom_id("q", s)
        if key in results and results[key]['success']:
            s.questions = parse_numbered_list(
                results[key]['content'])[:NUM_QUESTIONS]


def run_answers(states: List[ProblemState]) -> None:
    """Step 4: LLM answers questions."""
    with_q = [s for s in states if s.questions]
    if not with_q:
        return

    requests = []
    for s in with_q:
        requests.append({
            'custom_id': make_custom_id("a", s),
            'params': {
                'model':
                MODELS[s.llm_model],
                'max_tokens':
                2048,
                'messages': [{
                    'role':
                    'user',
                    'content':
                    make_answer_prompt(s.problem, s.questions)
                }],
            },
        })

    results = run_batch_step(requests, "Answers")

    for s in with_q:
        key = make_custom_id("a", s)
        if key in results and results[key]['success']:
            s.answers = parse_numbered_list(
                results[key]['content'])[:len(s.questions)]


def run_updates(states: List[ProblemState]) -> None:
    """Step 5: SLM updates with Q&A."""
    with_qa = [s for s in states if s.questions and s.answers]
    if not with_qa:
        return

    requests = []
    for s in with_qa:
        qa_pairs = list(zip(s.questions, s.answers))
        requests.append({
            'custom_id': make_custom_id("u", s),
            'params': {
                'model':
                MODELS[s.slm_model],
                'max_tokens':
                2048,
                'messages': [{
                    'role':
                    'user',
                    'content':
                    make_update_prompt(s.problem, s.slm_initial_answer,
                                       qa_pairs)
                }],
            },
        })

    results = run_batch_step(requests, "Updates")

    for s in with_qa:
        key = make_custom_id("u", s)
        if key in results and results[key]['success']:
            s.slm_final_reasoning = results[key]['content']
            s.slm_final_answer = extract_answer(results[key]['content'])
        else:
            s.slm_final_answer = s.slm_initial_answer


def run_final_evals(states: List[ProblemState]) -> None:
    """Step 6: Final evaluation."""
    need_eval = [
        s for s in states if s.initial_correct == False and s.slm_final_answer
    ]
    if not need_eval:
        for s in states:
            if s.initial_correct:
                s.final_correct = True
        return

    requests = []
    for s in need_eval:
        if s.is_oracle:
            continue
        requests.append({
            'custom_id': make_custom_id("fe", s),
            'params': {
                'model':
                MODELS[s.llm_model],
                'max_tokens':
                64,
                'messages': [{
                    'role':
                    'user',
                    'content':
                    make_evaluation_prompt(s.problem, s.slm_final_answer,
                                           s.correct_answer)
                }],
            },
        })

    results = run_batch_step(requests, "Final Evals")

    for s in states:
        if s.initial_correct:
            s.final_correct = True
        elif s.is_oracle:
            s.final_correct = s.slm_final_answer == s.correct_answer
        else:
            key = make_custom_id("fe", s)
            if key in results and results[key]['success']:
                s.final_correct = 'CORRECT' in results[key]['content'].upper()
            else:
                s.final_correct = False


# =============================================================================
# Main sweep
# =============================================================================


def load_problems(subject: str,
                  difficulties: List[str],
                  limit: Optional[int] = None) -> List[Dict]:
    """Load problems from baseline results."""
    pattern = f'math_all_models_{subject}_*.json'
    candidates = sorted(BASELINE_DIR.glob(pattern), reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No baseline for {subject}")

    with open(candidates[0]) as f:
        data = json.load(f)

    problems = []
    for r in data['results']:
        if r.get('difficulty') in difficulties:
            problems.append({
                'problem_idx': r['problem_idx'],
                'problem': r['problem'],
                'gold_answer': r['gold_answer'],
                'difficulty': r['difficulty'],
            })
            if limit and len(problems) >= limit:
                break

    return problems


def run_sweep(
    subjects: List[str],
    difficulties: List[str],
    limit: Optional[int] = None,
    output_dir: str = 'results/batch_qa_sweep',
) -> Dict[str, Any]:
    """Run full sweep across all model combinations."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate all combinations
    combinations = []
    for slm in SLM_OPTIONS:
        for llm in LLM_OPTIONS:
            for qa in Q_OPTIONS:
                combinations.append({
                    'slm': slm,
                    'llm': llm,
                    'qa': qa,
                    'oracle': False
                })
        # Add oracle variant (llm=opus, oracle=True)
        for qa in Q_OPTIONS:
            combinations.append({
                'slm': slm,
                'llm': 'opus',
                'qa': qa,
                'oracle': True
            })

    print(f"\n{'='*60}")
    print("BATCH QA SWEEP")
    print(f"{'='*60}")
    print(f"Subjects: {subjects}")
    print(f"Difficulties: {difficulties}")
    print(f"Combinations per subject: {len(combinations)}")
    print(f"Total combinations: {len(subjects) * len(combinations)}")
    print(f"{'='*60}\n")

    all_results = {}
    sweep_start = time.time()

    for subject in subjects:
        print(f"\n{'='*60}")
        print(f"Subject: {subject}")
        print(f"{'='*60}")

        # Load problems once per subject
        problems = load_problems(subject, difficulties, limit)
        print(f"Loaded {len(problems)} problems")

        for combo_idx, combo in enumerate(combinations):
            combo_name = f"SLM-{combo['slm']}_LLM-{combo['llm']}{'_oracle' if combo['oracle'] else ''}_Q-{combo['qa']}"
            print(f"\n[{combo_idx+1}/{len(combinations)}] {combo_name}")

            # Create states for this combination
            states = []
            for p in problems:
                states.append(
                    ProblemState(
                        problem_idx=p['problem_idx'],
                        problem=p['problem'],
                        correct_answer=p['gold_answer'],
                        dataset=f"math_{subject}",
                        difficulty=p['difficulty'],
                        slm_model=combo['slm'],
                        qa_model=combo['qa'],
                        llm_model=combo['llm'],
                        is_oracle=combo['oracle'],
                    ))

            # Run 6-step pipeline
            combo_start = time.time()
            run_proposals(states)
            run_evaluations(states)
            run_questions(states)
            run_answers(states)
            run_updates(states)
            run_final_evals(states)
            combo_time = time.time() - combo_start

            # Calculate results
            initial_correct = sum(1 for s in states if s.initial_correct)
            final_correct = sum(1 for s in states if s.final_correct)
            recovered = sum(1 for s in states
                            if not s.initial_correct and s.final_correct)

            print(
                f"  Initial: {initial_correct}/{len(states)} ({100*initial_correct/len(states):.1f}%)"
            )
            print(
                f"  Final:   {final_correct}/{len(states)} ({100*final_correct/len(states):.1f}%)"
            )
            print(f"  Time:    {combo_time:.1f}s")

            # Save results for this combination
            result_key = f"{subject}_{combo_name}"
            all_results[result_key] = {
                'subject':
                subject,
                'slm':
                combo['slm'],
                'llm':
                combo['llm'],
                'qa':
                combo['qa'],
                'oracle':
                combo['oracle'],
                'total':
                len(states),
                'initial_correct':
                initial_correct,
                'final_correct':
                final_correct,
                'recovered':
                recovered,
                'initial_accuracy':
                initial_correct / len(states),
                'final_accuracy':
                final_correct / len(states),
                'recovery_rate':
                recovered / (len(states) - initial_correct)
                if initial_correct < len(states) else 0,
                'time_seconds':
                combo_time,
            }

            # Save individual result file
            result_file = output_path / f"{result_key}.json"
            with open(result_file, 'w') as f:
                json.dump(
                    {
                        'config': combo,
                        'summary': all_results[result_key],
                        'results': [asdict(s) for s in states],
                    },
                    f,
                    indent=2)

    sweep_time = time.time() - sweep_start

    # Save summary
    summary_file = output_path / 'sweep_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(
            {
                'timestamp': datetime.now().isoformat(),
                'subjects': subjects,
                'difficulties': difficulties,
                'num_questions': NUM_QUESTIONS,
                'total_time_seconds': sweep_time,
                'results': all_results,
            },
            f,
            indent=2)

    print(f"\n{'='*60}")
    print("SWEEP COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {sweep_time/60:.1f} minutes")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description='Batch QA Compression Sweep')
    parser.add_argument('--subject',
                        type=str,
                        default=None,
                        help='Run specific subject only')
    parser.add_argument('--limit',
                        type=int,
                        default=None,
                        help='Limit problems per subject')
    parser.add_argument('--output-dir',
                        type=str,
                        default='results/batch_qa_sweep',
                        help='Output directory')

    args = parser.parse_args()

    subjects = [args.subject] if args.subject else SUBJECTS
    difficulties = ['medium', 'hard', 'very_hard']

    run_sweep(subjects, difficulties, args.limit, args.output_dir)


if __name__ == "__main__":
    main()
