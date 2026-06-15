"""AIME (American Invitational Mathematics Examination) benchmark utilities.

This module provides utilities for loading, prompting, and evaluating
the AIME benchmark dataset.

Dataset: AI-MO/aimo-validation-aime on HuggingFace
- 90 problems from AIME competitions
- Answers are integers from 0 to 999
- Problems are competition-level high school math
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from datasets import load_dataset


def load_aime_dataset() -> Tuple[Any, List[int]]:
    """Load AIME dataset and return (dataset, indices).

    Returns:
        Tuple of (dataset, list of all indices)
    """
    ds = load_dataset("AI-MO/aimo-validation-aime")["train"]
    indices = list(range(len(ds)))
    return ds, indices


def get_aime_problem(ds: Any, idx: int) -> Dict[str, Any]:
    """Get a single AIME problem by index.

    Returns dict with keys:
        - problem: str (the question)
        - answer: str (integer answer 0-999)
        - solution: str (reference solution)
        - url: str (AoPS wiki link)
    """
    ex = ds[idx]
    return {
        'problem': ex['problem'],
        'answer': str(ex['answer']),
        'solution': ex.get('solution', ''),
        'url': ex.get('url', ''),
    }


def build_aime_prompt(problem: str) -> str:
    """Build prompt for AIME problem.

    Args:
        problem: The problem statement

    Returns:
        Formatted prompt string
    """
    return f"""Solve this AIME problem. AIME answers are always integers from 0 to 999.

Problem: {problem}

Show your work step by step. At the end, state your final answer on its own line in the format:
ANSWER: <integer>
where <integer> is a number from 0 to 999."""


def extract_aime_answer(response: str) -> Optional[int]:
    """Extract the integer answer from model response.

    Args:
        response: The model's full response text

    Returns:
        Extracted integer answer, or None if not found
    """
    response = response.strip()

    # Look for explicit ANSWER: pattern first
    answer_match = re.search(r"ANSWER:\s*(\d+)", response, re.IGNORECASE)
    if answer_match:
        try:
            return int(answer_match.group(1))
        except ValueError:
            pass

    # Look for boxed answer (common in math)
    boxed_match = re.search(r"\\boxed\{(\d+)\}", response)
    if boxed_match:
        try:
            return int(boxed_match.group(1))
        except ValueError:
            pass

    # Look for "answer is X" or "= X" at end
    patterns = [
        r"(?:final\s+)?answer(?:\s+is)?[:\s]+(\d+)",
        r"(?:therefore|thus|so)[,:\s]+(?:the\s+)?(?:answer\s+is\s+)?(\d+)",
        r"=\s*(\d+)\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass

    # Last resort: find all integers and take the last one that's 0-999
    numbers = re.findall(r'\b(\d{1,3})\b', response)
    for num in reversed(numbers):
        try:
            val = int(num)
            if 0 <= val <= 999:
                return val
        except ValueError:
            pass

    return None


def check_aime_answer(predicted: Optional[int], expected: str) -> bool:
    """Check if the predicted answer matches the expected answer.

    Args:
        predicted: The model's extracted integer answer (or None)
        expected: The ground truth answer as string

    Returns:
        True if answer is correct, False otherwise
    """
    if predicted is None:
        return False

    try:
        expected_int = int(expected)
        return predicted == expected_int
    except ValueError:
        return False


def get_aime_stats(ds: Any, indices: List[int]) -> Dict[str, Any]:
    """Get statistics about AIME dataset subset.

    Args:
        ds: The AIME dataset
        indices: List of problem indices to analyze

    Returns:
        Dict with statistics about the problems
    """
    answers = []
    for idx in indices:
        ex = ds[idx]
        try:
            answers.append(int(ex['answer']))
        except ValueError:
            pass

    return {
        'n_problems': len(indices),
        'answer_range': (min(answers), max(answers)) if answers else (0, 0),
        'avg_answer': sum(answers) / len(answers) if answers else 0,
    }
