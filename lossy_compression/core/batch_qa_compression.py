#!/usr/bin/env python3
"""
Batch QA Compression Pipeline

Runs the QA compression approach in 6 batched steps (no iterative loop):
1. SLM Proposal: Generate initial answers for all problems
2. LLM Evaluation: Check which answers are correct
3. Question Generation: QA model generates 10 questions for wrong answers
4. LLM Answers: Answer all questions for each problem
5. SLM Update: Update answers using all Q&A pairs
6. Final Evaluation: Check if updated answers are correct

Uses Anthropic Message Batches API for efficient parallel processing.

Model roles:
- SLM: Small model for proposal and update (e.g., haiku)
- QA: Model for generating questions (e.g., haiku)
- LLM: Large model for evaluation and answers (e.g., opus)
"""

import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
import anthropic

from utils.llm_api import get_anthropic_key

# Model configurations
MODELS = {
    'haiku': 'claude-3-5-haiku-20241022',
    'sonnet': 'claude-sonnet-4-20250514',
    'opus': 'claude-opus-4-20250514',
}

NUM_QUESTIONS = 10
POLL_INTERVAL = 30  # seconds

# Baseline results directory
BASELINE_DIR = Path(__file__).parent.parent / 'results'


@dataclass
class ProblemState:
    """Track state for each problem through the pipeline."""
    problem_idx: int
    problem: str
    correct_answer: str
    dataset: str = ""
    difficulty: str = ""  # easy, medium, hard, very_hard

    # Step 1: SLM Proposal
    slm_initial_answer: Optional[str] = None
    slm_initial_reasoning: Optional[str] = None

    # Step 2: LLM Evaluation
    initial_correct: Optional[bool] = None

    # Step 3: Questions
    questions: List[str] = field(default_factory=list)

    # Step 4: LLM Answers
    answers: List[str] = field(default_factory=list)

    # Step 5: SLM Update
    slm_final_answer: Optional[str] = None
    slm_final_reasoning: Optional[str] = None

    # Final evaluation
    final_correct: Optional[bool] = None

    # Timing
    timestamps: Dict[str, str] = field(default_factory=dict)


@dataclass
class StepTiming:
    """Track timing for each pipeline step."""
    step_name: str
    start_time: float = 0
    end_time: float = 0
    num_requests: int = 0

    @property
    def duration_seconds(self) -> float:
        return self.end_time - self.start_time

    @property
    def duration_str(self) -> str:
        d = self.duration_seconds
        if d < 60:
            return f"{d:.1f}s"
        elif d < 3600:
            return f"{d/60:.1f}m"
        else:
            return f"{d/3600:.1f}h"


def submit_batch(requests: List[Dict]) -> str:
    """Submit batch to Anthropic API and return batch ID."""
    client = anthropic.Anthropic(api_key=get_anthropic_key())

    batch = client.messages.batches.create(requests=requests)
    print(f"    Submitted batch {batch.id} with {len(requests)} requests")
    return batch.id


def poll_batch(batch_id: str) -> Dict:
    """Poll batch status until complete."""
    client = anthropic.Anthropic(api_key=get_anthropic_key())

    while True:
        batch = client.messages.batches.retrieve(batch_id)
        status = batch.processing_status

        counts = batch.request_counts
        print(f"    Status: {status} | Processing: {counts.processing} | "
              f"Succeeded: {counts.succeeded} | Errored: {counts.errored}")

        if status == 'ended':
            return {
                'status': status,
                'succeeded': counts.succeeded,
                'errored': counts.errored,
                'results_url': batch.results_url,
            }

        time.sleep(POLL_INTERVAL)


def download_batch_results(batch_id: str) -> Dict[str, Any]:
    """Download and parse batch results."""
    client = anthropic.Anthropic(api_key=get_anthropic_key())

    results = {}
    for result in client.messages.batches.results(batch_id):
        custom_id = result.custom_id
        if result.result.type == 'succeeded':
            message = result.result.message
            content = message.content[0].text if message.content else ""
            results[custom_id] = {
                'success': True,
                'content': content,
            }
        else:
            results[custom_id] = {
                'success': False,
                'error': str(result.result),
            }

    return results


def run_batch_step(requests: List[Dict],
                   step_name: str) -> Tuple[Dict[str, Any], StepTiming]:
    """Run a batch step: submit, poll, download. Returns results and timing."""
    timing = StepTiming(step_name=step_name, num_requests=len(requests))
    timing.start_time = time.time()

    print(f"\n  {step_name}: Submitting {len(requests)} requests...")

    batch_id = submit_batch(requests)
    print(f"    Polling for completion...")

    status = poll_batch(batch_id)
    print(f"    Downloading results...")

    results = download_batch_results(batch_id)

    timing.end_time = time.time()
    print(f"    Got {len(results)} results in {timing.duration_str}")

    return results, timing


# =============================================================================
# Step 1: SLM Proposal
# =============================================================================


def make_proposal_prompt(problem: str) -> str:
    """Create prompt for SLM to propose initial answer."""
    return f"""Solve this problem step by step. At the end, provide your final answer on a new line starting with "ANSWER: ".

Problem:
{problem}"""


def make_custom_id(prefix: str, state: ProblemState) -> str:
    """Create unique custom_id including dataset and problem_idx."""
    return f"{prefix}_{state.dataset}_{state.problem_idx}"


def prepare_proposal_requests(states: List[ProblemState],
                              slm_model: str) -> List[Dict]:
    """Prepare batch requests for SLM proposals."""
    requests = []
    for state in states:
        requests.append({
            'custom_id': make_custom_id("proposal", state),
            'params': {
                'model':
                MODELS[slm_model],
                'max_tokens':
                2048,
                'messages': [{
                    'role': 'user',
                    'content': make_proposal_prompt(state.problem)
                }],
            },
        })
    return requests


def parse_proposal_results(states: List[ProblemState], results: Dict[str,
                                                                     Any]):
    """Parse proposal results and update states."""
    for state in states:
        key = make_custom_id("proposal", state)
        if key in results and results[key]['success']:
            content = results[key]['content']
            state.slm_initial_reasoning = content

            # Extract answer
            if 'ANSWER:' in content:
                state.slm_initial_answer = content.split('ANSWER:')[-1].strip()
            else:
                # Try to find boxed answer
                if '\\boxed{' in content:
                    import re
                    match = re.search(r'\\boxed\{([^}]+)\}', content)
                    if match:
                        state.slm_initial_answer = match.group(1)
                else:
                    # Take last line as answer
                    lines = [
                        l.strip() for l in content.strip().split('\n')
                        if l.strip()
                    ]
                    state.slm_initial_answer = lines[-1] if lines else ""

        state.timestamps['proposal'] = datetime.now().isoformat()


# =============================================================================
# Step 2: LLM Evaluation
# =============================================================================


def make_evaluation_prompt(problem: str, proposed_answer: str,
                           correct_answer: str) -> str:
    """Create prompt for LLM to evaluate if answer is correct."""
    return f"""You are evaluating if a proposed answer to a math problem is correct.

Problem:
{problem}

Proposed Answer: {proposed_answer}
Correct Answer: {correct_answer}

Is the proposed answer mathematically equivalent to the correct answer?
Consider that answers may be written in different forms (e.g., fractions vs decimals, simplified vs unsimplified).

Respond with only "CORRECT" or "INCORRECT"."""


def prepare_evaluation_requests(states: List[ProblemState],
                                llm_model: str) -> List[Dict]:
    """Prepare batch requests for LLM evaluation."""
    requests = []
    for state in states:
        if state.slm_initial_answer:
            requests.append({
                'custom_id': make_custom_id("eval", state),
                'params': {
                    'model':
                    MODELS[llm_model],
                    'max_tokens':
                    64,
                    'messages': [{
                        'role':
                        'user',
                        'content':
                        make_evaluation_prompt(state.problem,
                                               state.slm_initial_answer,
                                               state.correct_answer)
                    }],
                },
            })
    return requests


def parse_evaluation_results(states: List[ProblemState], results: Dict[str,
                                                                       Any]):
    """Parse evaluation results and update states."""
    for state in states:
        key = make_custom_id("eval", state)
        if key in results and results[key]['success']:
            content = results[key]['content'].strip().upper()
            state.initial_correct = 'CORRECT' in content
        else:
            state.initial_correct = False

        state.timestamps['evaluation'] = datetime.now().isoformat()


# =============================================================================
# Step 3: Question Generation
# =============================================================================


def make_question_prompt(problem: str, proposed_answer: str,
                         correct_answer: str) -> str:
    """Create prompt to generate questions that would help solve the problem."""
    return f"""A student attempted this problem but got the wrong answer. Generate {NUM_QUESTIONS} specific questions that, if answered, would help the student arrive at the correct answer.

Problem:
{problem}

Student's Wrong Answer: {proposed_answer}
Correct Answer: {correct_answer}

Generate {NUM_QUESTIONS} targeted questions. Each question should:
- Address a specific aspect of the problem the student may have misunderstood
- Be answerable with a brief, factual response
- Help guide the student toward the correct solution

Format your response as a numbered list:
1. [Question 1]
2. [Question 2]
...
{NUM_QUESTIONS}. [Question {NUM_QUESTIONS}]"""


def prepare_question_requests(states: List[ProblemState],
                              llm_model: str) -> List[Dict]:
    """Prepare batch requests for question generation (only for wrong answers)."""
    requests = []
    for state in states:
        if state.initial_correct == False:  # Only for wrong answers
            requests.append({
                'custom_id': make_custom_id("questions", state),
                'params': {
                    'model':
                    MODELS[llm_model],
                    'max_tokens':
                    1024,
                    'messages': [{
                        'role':
                        'user',
                        'content':
                        make_question_prompt(state.problem,
                                             state.slm_initial_answer,
                                             state.correct_answer)
                    }],
                },
            })
    return requests


def parse_question_results(states: List[ProblemState], results: Dict[str,
                                                                     Any]):
    """Parse question results and update states."""
    import re

    for state in states:
        key = make_custom_id("questions", state)
        if key in results and results[key]['success']:
            content = results[key]['content']

            # Parse numbered questions
            questions = []
            for line in content.split('\n'):
                line = line.strip()
                # Match lines starting with number + period or parenthesis
                match = re.match(r'^\d+[\.\)]\s*(.+)$', line)
                if match:
                    questions.append(match.group(1))

            state.questions = questions[:NUM_QUESTIONS]

        state.timestamps['questions'] = datetime.now().isoformat()


# =============================================================================
# Step 4: LLM Answers
# =============================================================================


def make_answer_prompt(problem: str, questions: List[str]) -> str:
    """Create prompt for LLM to answer all questions."""
    questions_text = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))

    return f"""Answer each of these questions about the following problem. Give brief, direct answers.

Problem:
{problem}

Questions:
{questions_text}

Provide your answers in the same numbered format:
1. [Answer 1]
2. [Answer 2]
...
{len(questions)}. [Answer {len(questions)}]"""


def prepare_answer_requests(states: List[ProblemState],
                            llm_model: str) -> List[Dict]:
    """Prepare batch requests for LLM to answer questions."""
    requests = []
    for state in states:
        if state.questions:  # Only if we have questions
            requests.append({
                'custom_id': make_custom_id("answers", state),
                'params': {
                    'model':
                    MODELS[llm_model],
                    'max_tokens':
                    2048,
                    'messages': [{
                        'role':
                        'user',
                        'content':
                        make_answer_prompt(state.problem, state.questions)
                    }],
                },
            })
    return requests


def parse_answer_results(states: List[ProblemState], results: Dict[str, Any]):
    """Parse answer results and update states."""
    import re

    for state in states:
        key = make_custom_id("answers", state)
        if key in results and results[key]['success']:
            content = results[key]['content']

            # Parse numbered answers
            answers = []
            for line in content.split('\n'):
                line = line.strip()
                match = re.match(r'^\d+[\.\)]\s*(.+)$', line)
                if match:
                    answers.append(match.group(1))

            state.answers = answers[:len(state.questions)]

        state.timestamps['answers'] = datetime.now().isoformat()


# =============================================================================
# Step 5: SLM Update
# =============================================================================


def make_update_prompt(problem: str, initial_answer: str,
                       qa_pairs: List[Tuple[str, str]]) -> str:
    """Create prompt for SLM to update answer based on Q&A."""
    qa_text = "\n".join(f"Q: {q}\nA: {a}\n" for q, a in qa_pairs)

    return f"""You previously attempted this problem and got: {initial_answer}

Here are some hints in Q&A form:
{qa_text}

Using these hints, reconsider the problem and provide your final answer.

Problem:
{problem}

Think step by step, then provide your final answer on a new line starting with "ANSWER: "."""


def prepare_update_requests(states: List[ProblemState],
                            slm_model: str) -> List[Dict]:
    """Prepare batch requests for SLM to update answers."""
    requests = []
    for state in states:
        if state.questions and state.answers:
            qa_pairs = list(zip(state.questions, state.answers))
            requests.append({
                'custom_id': make_custom_id("update", state),
                'params': {
                    'model':
                    MODELS[slm_model],
                    'max_tokens':
                    2048,
                    'messages': [{
                        'role':
                        'user',
                        'content':
                        make_update_prompt(state.problem,
                                           state.slm_initial_answer, qa_pairs)
                    }],
                },
            })
    return requests


def parse_update_results(states: List[ProblemState], results: Dict[str, Any]):
    """Parse update results and update states."""
    for state in states:
        key = make_custom_id("update", state)
        if key in results and results[key]['success']:
            content = results[key]['content']
            state.slm_final_reasoning = content

            # Extract answer
            if 'ANSWER:' in content:
                state.slm_final_answer = content.split('ANSWER:')[-1].strip()
            else:
                if '\\boxed{' in content:
                    import re
                    match = re.search(r'\\boxed\{([^}]+)\}', content)
                    if match:
                        state.slm_final_answer = match.group(1)
                else:
                    lines = [
                        l.strip() for l in content.strip().split('\n')
                        if l.strip()
                    ]
                    state.slm_final_answer = lines[-1] if lines else ""
        else:
            # Keep initial answer if update failed
            state.slm_final_answer = state.slm_initial_answer
            state.slm_final_reasoning = state.slm_initial_reasoning

        state.timestamps['update'] = datetime.now().isoformat()


# =============================================================================
# Step 6: Final Evaluation
# =============================================================================


def prepare_final_eval_requests(states: List[ProblemState],
                                llm_model: str) -> List[Dict]:
    """Prepare batch requests for final evaluation."""
    requests = []
    for state in states:
        # Only evaluate problems that went through Q&A
        if state.slm_final_answer and state.initial_correct == False:
            requests.append({
                'custom_id': make_custom_id("final_eval", state),
                'params': {
                    'model':
                    MODELS[llm_model],
                    'max_tokens':
                    64,
                    'messages': [{
                        'role':
                        'user',
                        'content':
                        make_evaluation_prompt(state.problem,
                                               state.slm_final_answer,
                                               state.correct_answer)
                    }],
                },
            })
    return requests


def parse_final_eval_results(states: List[ProblemState], results: Dict[str,
                                                                       Any]):
    """Parse final evaluation results."""
    for state in states:
        # Problems that were initially correct stay correct
        if state.initial_correct:
            state.final_correct = True
            continue

        key = make_custom_id("final_eval", state)
        if key in results and results[key]['success']:
            content = results[key]['content'].strip().upper()
            state.final_correct = 'CORRECT' in content
        else:
            state.final_correct = False

        state.timestamps['final_eval'] = datetime.now().isoformat()


# =============================================================================
# Main Pipeline
# =============================================================================


def run_batch_qa_pipeline(
    states: List[ProblemState],
    slm_model: str = 'haiku',
    qa_model: str = 'haiku',
    llm_model: str = 'opus',
) -> Tuple[List[ProblemState], Dict[str, Any]]:
    """Run the full batch QA pipeline.

    Args:
        states: List of problems to process
        slm_model: Model for proposal and update (e.g., haiku)
        qa_model: Model for question generation (e.g., haiku)
        llm_model: Model for evaluation and answers (e.g., opus)

    Returns:
        Tuple of (states, timing_info)
    """
    pipeline_start = time.time()
    timings: List[StepTiming] = []

    print(f"\n{'='*60}")
    print(f"BATCH QA COMPRESSION PIPELINE")
    print(f"{'='*60}")
    print(f"Problems: {len(states)}")
    print(f"SLM (proposal/update): {slm_model}")
    print(f"QA (questions): {qa_model}")
    print(f"LLM (eval/answers): {llm_model}")
    print(f"Questions per problem: {NUM_QUESTIONS}")
    print(f"{'='*60}")

    # Step 1: SLM Proposal
    print("\n[Step 1/6] SLM Proposal")
    requests = prepare_proposal_requests(states, slm_model)
    results, timing = run_batch_step(requests, "Proposal")
    timings.append(timing)
    parse_proposal_results(states, results)

    # Step 2: LLM Evaluation
    print("\n[Step 2/6] LLM Evaluation")
    requests = prepare_evaluation_requests(states, llm_model)
    results, timing = run_batch_step(requests, "Evaluation")
    timings.append(timing)
    parse_evaluation_results(states, results)

    # Count initial accuracy
    initial_correct = sum(1 for s in states if s.initial_correct)
    print(
        f"    Initial accuracy: {initial_correct}/{len(states)} ({100*initial_correct/len(states):.1f}%)"
    )

    # Step 3: Question Generation (only for wrong answers)
    wrong_states = [s for s in states if not s.initial_correct]
    print(
        f"\n[Step 3/6] Question Generation ({len(wrong_states)} problems need Q&A)"
    )

    if wrong_states:
        requests = prepare_question_requests(states, qa_model)  # Use QA model
        results, timing = run_batch_step(requests, "Questions")
        timings.append(timing)
        parse_question_results(states, results)

        # Step 4: LLM Answers
        print("\n[Step 4/6] LLM Answers")
        requests = prepare_answer_requests(states, llm_model)
        results, timing = run_batch_step(requests, "Answers")
        timings.append(timing)
        parse_answer_results(states, results)

        # Step 5: SLM Update
        print("\n[Step 5/6] SLM Update")
        requests = prepare_update_requests(states, slm_model)
        results, timing = run_batch_step(requests, "Update")
        timings.append(timing)
        parse_update_results(states, results)

        # Step 6: Final Evaluation
        print("\n[Step 6/6] Final Evaluation")
        requests = prepare_final_eval_requests(states, llm_model)
        results, timing = run_batch_step(requests, "Final Eval")
        timings.append(timing)
        parse_final_eval_results(states, results)
    else:
        print("    All problems correct - skipping Q&A steps")
        for state in states:
            state.final_correct = state.initial_correct

    pipeline_end = time.time()
    total_duration = pipeline_end - pipeline_start

    # Summary
    final_correct = sum(1 for s in states if s.final_correct)
    recovered = sum(1 for s in states
                    if not s.initial_correct and s.final_correct)

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(
        f"Initial accuracy: {initial_correct}/{len(states)} ({100*initial_correct/len(states):.1f}%)"
    )
    print(
        f"Final accuracy:   {final_correct}/{len(states)} ({100*final_correct/len(states):.1f}%)"
    )
    if wrong_states:
        print(
            f"Recovered:        {recovered}/{len(wrong_states)} ({100*recovered/len(wrong_states):.1f}%)"
        )

    print(f"\n{'='*60}")
    print("TIMING")
    print(f"{'='*60}")
    for t in timings:
        print(
            f"  {t.step_name:<20} {t.num_requests:>5} reqs  {t.duration_str:>10}"
        )
    print(f"  {'-'*40}")
    total_reqs = sum(t.num_requests for t in timings)
    if total_duration < 60:
        total_str = f"{total_duration:.1f}s"
    elif total_duration < 3600:
        total_str = f"{total_duration/60:.1f}m"
    else:
        total_str = f"{total_duration/3600:.1f}h"
    print(f"  {'TOTAL':<20} {total_reqs:>5} reqs  {total_str:>10}")
    print(f"{'='*60}")

    timing_info = {
        'total_duration_seconds':
        total_duration,
        'steps': [{
            'name': t.step_name,
            'num_requests': t.num_requests,
            'duration_seconds': t.duration_seconds,
        } for t in timings],
    }

    return states, timing_info


def load_math_problems(
    subject: str,
    limit: Optional[int] = None,
    difficulties: Optional[List[str]] = None,
) -> List[ProblemState]:
    """Load MATH problems from baseline results file.

    Args:
        subject: MATH subject (algebra, geometry, number_theory)
        limit: Max number of problems to load
        difficulties: Filter to these difficulty levels (e.g., ['medium', 'hard', 'very_hard'])
    """
    dataset_name = f'math_{subject}'
    print(f"Loading MATH {subject} from baseline...")

    # Find most recent baseline file
    pattern = f'math_all_models_{subject}_*.json'
    candidates = sorted(BASELINE_DIR.glob(pattern), reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No baseline file found for {dataset_name}")

    baseline_path = candidates[0]
    print(f"  Using: {baseline_path.name}")

    with open(baseline_path) as f:
        data = json.load(f)

    states = []
    for result in data.get('results', []):
        difficulty = result.get('difficulty', 'unknown')

        # Check difficulty filter
        if difficulties and difficulty not in difficulties:
            continue

        if limit and len(states) >= limit:
            break

        states.append(
            ProblemState(
                problem_idx=result['problem_idx'],
                problem=result['problem'],
                correct_answer=result['gold_answer'],
                dataset=dataset_name,
                difficulty=difficulty,
            ))

    print(f"  Loaded {len(states)} problems")
    if difficulties:
        print(f"  Filtered to difficulties: {difficulties}")

    return states


def load_gsm8k_problems(
    limit: Optional[int] = None,
    difficulties: Optional[List[str]] = None,
) -> List[ProblemState]:
    """Load GSM8K problems from baseline results file."""
    print("Loading GSM8K from baseline...")

    # Find most recent baseline file
    pattern = 'gsm8k_all_models_*.json'
    candidates = sorted(BASELINE_DIR.glob(pattern), reverse=True)
    if not candidates:
        raise FileNotFoundError("No baseline file found for gsm8k")

    baseline_path = candidates[0]
    print(f"  Using: {baseline_path.name}")

    with open(baseline_path) as f:
        data = json.load(f)

    states = []
    for result in data.get('results', []):
        difficulty = result.get('difficulty', 'unknown')

        # Check difficulty filter
        if difficulties and difficulty not in difficulties:
            continue

        if limit and len(states) >= limit:
            break

        states.append(
            ProblemState(
                problem_idx=result['problem_idx'],
                problem=result['problem'],
                correct_answer=result['gold_answer'],
                dataset='gsm8k',
                difficulty=difficulty,
            ))

    print(f"  Loaded {len(states)} problems")
    if difficulties:
        print(f"  Filtered to difficulties: {difficulties}")

    return states


def save_results(states: List[ProblemState], output_path: Path, config: Dict,
                 timing_info: Dict):
    """Save results to JSON."""
    results = []
    for state in states:
        results.append(asdict(state))

    # Difficulty breakdown
    diff_counts = {}
    for state in states:
        d = state.difficulty or 'unknown'
        diff_counts[d] = diff_counts.get(d, 0) + 1

    output = {
        'config': config,
        'timing': timing_info,
        'summary': {
            'total':
            len(states),
            'initial_correct':
            sum(1 for s in states if s.initial_correct),
            'final_correct':
            sum(1 for s in states if s.final_correct),
            'recovered':
            sum(1 for s in states
                if not s.initial_correct and s.final_correct),
            'difficulty_counts':
            diff_counts,
        },
        'results': results,
        'timestamp': datetime.now().isoformat(),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved results to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Batch QA Compression Pipeline')
    parser.add_argument(
        '--dataset',
        type=str,
        default='math_all',
        choices=[
            'gsm8k', 'math_algebra', 'math_geometry', 'math_number_theory',
            'math_all'
        ],
        help='Dataset to use (math_all runs all MATH subjects)')
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of problems per subject (for testing)')
    parser.add_argument('--slm',
                        type=str,
                        default='haiku',
                        choices=['haiku', 'sonnet'],
                        help='Small language model for proposals and updates')
    parser.add_argument('--qa',
                        type=str,
                        default='haiku',
                        choices=['haiku', 'sonnet', 'opus'],
                        help='Model for question generation')
    parser.add_argument('--llm',
                        type=str,
                        default='opus',
                        choices=['sonnet', 'opus'],
                        help='Large language model for evaluation and answers')
    parser.add_argument(
        '--difficulties',
        type=str,
        default='medium,hard,very_hard',
        help=
        'Comma-separated difficulties to include (default: medium,hard,very_hard)'
    )
    parser.add_argument('--output-dir',
                        type=str,
                        default='results/batch_qa',
                        help='Output directory')

    args = parser.parse_args()

    # Parse difficulties
    difficulties = [
        d.strip() for d in args.difficulties.split(',') if d.strip()
    ]
    if 'all' in difficulties:
        difficulties = None  # No filtering

    print(f"\n{'='*60}")
    print("LOADING PROBLEMS")
    print(f"{'='*60}")

    # Load problems
    all_states = []

    if args.dataset == 'math_all':
        for subject in ['algebra', 'geometry', 'number_theory']:
            states = load_math_problems(subject, args.limit, difficulties)
            all_states.extend(states)
    elif args.dataset == 'gsm8k':
        all_states = load_gsm8k_problems(args.limit, difficulties)
    elif args.dataset.startswith('math_'):
        subject = args.dataset.replace('math_', '')
        all_states = load_math_problems(subject, args.limit, difficulties)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print(f"\nTotal problems to process: {len(all_states)}")

    if not all_states:
        print("No problems to process!")
        return

    # Run pipeline
    states, timing_info = run_batch_qa_pipeline(
        all_states,
        slm_model=args.slm,
        qa_model=args.qa,
        llm_model=args.llm,
    )

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    diff_str = '_'.join(difficulties) if difficulties else 'all'
    output_path = Path(
        args.output_dir
    ) / f"{args.dataset}_{args.slm}_{args.qa}_{args.llm}_{diff_str}_{timestamp}.json"

    config = {
        'dataset': args.dataset,
        'slm': args.slm,
        'qa': args.qa,
        'llm': args.llm,
        'num_questions': NUM_QUESTIONS,
        'difficulties': difficulties,
        'limit': args.limit,
    }

    save_results(states, output_path, config, timing_info)


if __name__ == "__main__":
    main()
