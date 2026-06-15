"""HLE (Humanity's Last Exam) benchmark utilities.

This module provides utilities for loading, prompting, and evaluating
the HLE benchmark dataset.

Dataset: cais/hle on HuggingFace
- 2500 total problems (2158 text-only, 342 with images)
- Answer types: exactMatch (1909), multipleChoice (591)
- Categories: Math, Physics, CS/AI, Biology/Medicine, Chemistry,
              Humanities/Social Science, Engineering, Other
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from datasets import load_dataset


def load_hle_dataset(text_only: bool = True) -> Tuple[Any, List[int]]:
    """Load HLE dataset and return (dataset, indices).

    Args:
        text_only: If True, filter to problems without images (2158 problems).
                   If False, return all problems (2500 problems).

    Returns:
        Tuple of (dataset, list of valid indices)
    """
    ds = load_dataset("cais/hle")["test"]

    if text_only:
        indices = [
            i for i, ex in enumerate(ds)
            if not ex['image'] or not ex['image'].strip()
        ]
    else:
        indices = list(range(len(ds)))

    return ds, indices


def get_hle_problem(ds: Any, idx: int) -> Dict[str, Any]:
    """Get a single HLE problem by index.

    Returns dict with keys:
        - question: str
        - answer: str
        - answer_type: 'exactMatch' or 'multipleChoice'
        - category: str (e.g., 'Math', 'Physics')
        - subject: str (e.g., 'Mathematics', 'Chess')
        - has_image: bool
    """
    ex = ds[idx]
    return {
        'question': ex['question'],
        'answer': ex['answer'],
        'answer_type': ex['answer_type'],
        'category': ex['category'],
        'subject': ex['raw_subject'],
        'has_image': bool(ex['image'] and ex['image'].strip()),
    }


def build_hle_prompt(question: str, answer_type: str) -> str:
    """Build prompt for HLE problem.

    Args:
        question: The problem question text
        answer_type: 'exactMatch' or 'multipleChoice'

    Returns:
        Formatted prompt string
    """
    if answer_type == "multipleChoice":
        return f"""Answer the following question.

Question: {question}

Think through this step by step. At the end, state your final answer on its own line in the format:
ANSWER: X
where X is the letter (A, B, C, D, or E)."""
    else:
        return f"""Answer the following question.

Question: {question}

Think through this step by step. At the end, state your final answer on its own line in the format:
ANSWER: <your answer>
Keep the answer as concise as possible (just the value/expression, no explanation)."""


def extract_hle_answer(response: str, answer_type: str) -> str:
    """Extract the final answer from model response.

    Args:
        response: The model's full response text
        answer_type: 'exactMatch' or 'multipleChoice'

    Returns:
        Extracted answer string
    """
    response = response.strip()

    if answer_type == "multipleChoice":
        # Look for explicit ANSWER: pattern first
        answer_match = re.search(r"ANSWER:\s*\**([A-E])\**", response,
                                 re.IGNORECASE)
        if answer_match:
            return answer_match.group(1).upper()

        # Look for other patterns like "Answer: D" or "The answer is D"
        patterns = [
            r"(?:final\s+)?(?:answer|choice)(?:\s+is)?[:\s]+\**([A-E])\**\b",
            r"\b([A-E])\s*(?:is\s+(?:the\s+)?(?:correct|right|answer))",
            r"^([A-E])$",
            r"\b([A-E])\b\s*$",
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).upper()

        # Last resort: find any single letter in the last few lines
        last_lines = "\n".join(response.split("\n")[-5:])
        letters = re.findall(r'\b([A-E])\b', last_lines)
        if letters:
            return letters[-1].upper()

    # For exactMatch, try to extract final answer more carefully
    # Look for explicit "ANSWER:" pattern first (from our prompt)
    answer_match = re.search(r"ANSWER:\s*(.+?)(?:\s*$|\n)", response,
                             re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).strip()

    # Look for other explicit answer patterns
    final_patterns = [
        r"(?:final\s+)?answer[:\s]+\**(.+?)\**(?:\s*$|\n)",
        r"(?:therefore|thus|so)[,:\s]+(?:the\s+)?(?:answer\s+is\s+)?(.+?)(?:\.|$)",
        r"=\s*\**(.+?)\**\s*$",
    ]
    for pattern in final_patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
        if match:
            ans = match.group(1).strip().strip('*').strip()
            if ans:
                return ans

    # Return the last non-empty line as fallback
    lines = [l.strip() for l in response.split('\n') if l.strip()]
    if lines:
        return lines[-1]

    return response


def normalize_answer(text: str) -> str:
    """Normalize answer for comparison."""
    text = text.strip().lower()
    # Remove leading "as " or "the "
    text = re.sub(r'^(as|the)\s+', '', text)
    # Remove punctuation and extra whitespace
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def check_hle_answer(predicted: str, expected: str, answer_type: str) -> bool:
    """Check if the predicted answer matches the expected answer.

    Uses multiple matching strategies for exactMatch:
    1. Exact match (case-insensitive)
    2. Normalized match (remove punctuation, whitespace, leading articles)
    3. Substring containment for short answers
    4. LaTeX formatting normalization
    5. Yes/No/True/False equivalence

    Args:
        predicted: The model's extracted answer
        expected: The ground truth answer
        answer_type: 'exactMatch' or 'multipleChoice'

    Returns:
        True if answer is correct, False otherwise
    """
    predicted = predicted.strip()
    expected = expected.strip()

    if answer_type == "multipleChoice":
        return predicted.lower() == expected.lower()

    # For exactMatch, try multiple matching strategies

    # 1. Exact match (case-insensitive)
    if predicted.lower() == expected.lower():
        return True

    # 2. Normalized match (remove punctuation, whitespace, leading articles)
    pred_norm = normalize_answer(predicted)
    exp_norm = normalize_answer(expected)
    if pred_norm == exp_norm:
        return True

    # 3. Yes/No/True/False equivalence
    yes_words = {'yes', 'true', 'correct', 'right'}
    no_words = {'no', 'false', 'incorrect', 'wrong'}
    pred_lower = predicted.lower().strip()
    exp_lower = expected.lower().strip()
    if pred_lower in yes_words and exp_lower in yes_words:
        return True
    if pred_lower in no_words and exp_lower in no_words:
        return True

    # 4. Substring containment (for short expected answers)
    if len(expected) < 50 and expected.lower() in predicted.lower():
        return True

    # 5. For math answers, try removing LaTeX formatting
    pred_math = re.sub(r'[\$\\{}]', '', predicted.lower())
    exp_math = re.sub(r'[\$\\{}]', '', expected.lower())
    pred_math = re.sub(r'mathbb{(\w)}', r'\1', pred_math)
    exp_math = re.sub(r'mathbb{(\w)}', r'\1', exp_math)
    if pred_math == exp_math:
        return True

    return False


def get_hle_stats(ds: Any, indices: List[int]) -> Dict[str, Any]:
    """Get statistics about HLE dataset subset.

    Args:
        ds: The HLE dataset
        indices: List of problem indices to analyze

    Returns:
        Dict with statistics about the problems
    """
    from collections import Counter

    categories = Counter()
    answer_types = Counter()
    subjects = Counter()

    for idx in indices:
        ex = ds[idx]
        categories[ex['category']] += 1
        answer_types[ex['answer_type']] += 1
        subjects[ex['raw_subject']] += 1

    return {
        'n_problems': len(indices),
        'categories': dict(categories),
        'answer_types': dict(answer_types),
        'top_subjects': dict(subjects.most_common(10)),
    }
