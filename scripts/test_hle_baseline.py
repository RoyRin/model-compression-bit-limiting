#!/usr/bin/env python3
"""Quick test of HLE baseline."""

import argparse
import anthropic
import re
import sys
from pathlib import Path
from datasets import load_dataset

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.llm_api import get_anthropic_key

client = anthropic.Anthropic(api_key=get_anthropic_key())

MODEL_IDS = {
    'haiku': 'claude-haiku-4-5-20251001',
    'sonnet': 'claude-sonnet-4-5-20250929',
    'opus': 'claude-opus-4-5-20251101',
}


def extract_answer(response: str, answer_type: str) -> str:
    """Extract the final answer from model response."""
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


def check_answer(predicted: str, expected: str, answer_type: str) -> bool:
    """Check if the predicted answer matches the expected answer."""
    predicted = predicted.strip()
    expected = expected.strip()

    if answer_type == "multipleChoice":
        return predicted.lower() == expected.lower()
    else:
        # For exactMatch, try multiple matching strategies
        # 1. Exact match (case-insensitive)
        if predicted.lower() == expected.lower():
            return True

        # 2. Normalized match (remove punctuation, whitespace, leading articles)
        pred_norm = normalize_answer(predicted)
        exp_norm = normalize_answer(expected)
        if pred_norm == exp_norm:
            return True

        # 3. Substring containment (for short expected answers)
        if len(expected) < 50 and expected.lower() in predicted.lower():
            return True

        # 4. For math answers, try removing LaTeX formatting
        pred_math = re.sub(r'[\$\\{}]', '', predicted.lower())
        exp_math = re.sub(r'[\$\\{}]', '', expected.lower())
        pred_math = re.sub(r'mathbb{(\w)}', r'\1', pred_math)
        exp_math = re.sub(r'mathbb{(\w)}', r'\1', exp_math)
        if pred_math == exp_math:
            return True

        return False


def run_hle_baseline(num_problems: int = 20,
                     skip_images: bool = True,
                     model: str = 'opus'):
    """Run HLE baseline with specified model."""
    model_id = MODEL_IDS.get(model, model)
    print(f"Model: {model_id}")
    print("Loading HLE dataset...")
    ds = load_dataset("cais/hle")["test"]

    # Filter to text-only if requested
    if skip_images:
        indices = [
            i for i, ex in enumerate(ds)
            if not ex['image'] or not ex['image'].strip()
        ]
        print(f"Filtered to {len(indices)} text-only problems")
    else:
        indices = list(range(len(ds)))

    results = []
    correct = 0
    mc_correct = 0
    mc_total = 0
    exact_correct = 0
    exact_total = 0

    for i, idx in enumerate(indices[:num_problems]):
        ex = ds[idx]
        question = ex['question']
        answer = ex['answer']
        answer_type = ex['answer_type']
        category = ex['category']
        subject = ex['raw_subject']

        # Build prompt
        if answer_type == "multipleChoice":
            prompt = f"""Answer the following question.

Question: {question}

Think through this step by step. At the end, state your final answer on its own line in the format:
ANSWER: X
where X is the letter (A, B, C, D, or E)."""
        else:
            prompt = f"""Answer the following question.

Question: {question}

Think through this step by step. At the end, state your final answer on its own line in the format:
ANSWER: <your answer>
Keep the answer as concise as possible (just the value/expression, no explanation)."""

        try:
            response = client.messages.create(model=model_id,
                                              max_tokens=2048,
                                              messages=[{
                                                  "role": "user",
                                                  "content": prompt
                                              }])
            model_response = response.content[0].text
            predicted = extract_answer(model_response, answer_type)
            is_correct = check_answer(predicted, answer, answer_type)

            if is_correct:
                correct += 1

            if answer_type == "multipleChoice":
                mc_total += 1
                if is_correct:
                    mc_correct += 1
            else:
                exact_total += 1
                if is_correct:
                    exact_correct += 1

            status = "✓" if is_correct else "✗"
            print(
                f"[{i+1}/{num_problems}] {status} ({category}/{subject}) - {answer_type}"
            )
            if not is_correct:
                print(f"  Expected: {answer}")
                print(f"  Got: {predicted}")
                print(f"  Response: {model_response[:200]}...")

            results.append({
                "idx": idx,
                "question": question[:100],
                "answer": answer,
                "answer_type": answer_type,
                "category": category,
                "subject": subject,
                "predicted": predicted,
                "correct": is_correct,
            })

        except Exception as e:
            print(f"[{i+1}/{num_problems}] Error: {e}")
            results.append({
                "idx": idx,
                "error": str(e),
                "correct": False,
            })

    print("\n" + "=" * 50)
    print(
        f"RESULTS: {correct}/{num_problems} correct ({100*correct/num_problems:.1f}%)"
    )
    if mc_total > 0:
        print(
            f"  Multiple Choice: {mc_correct}/{mc_total} ({100*mc_correct/mc_total:.1f}%)"
        )
    if exact_total > 0:
        print(
            f"  Exact Match: {exact_correct}/{exact_total} ({100*exact_correct/exact_total:.1f}%)"
        )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test HLE baseline with Claude models")
    parser.add_argument("--num-problems",
                        "-n",
                        type=int,
                        default=20,
                        help="Number of problems to test")
    parser.add_argument("--model",
                        "-m",
                        type=str,
                        default="opus",
                        choices=["haiku", "sonnet", "opus"],
                        help="Model to use")
    parser.add_argument("--verbose",
                        "-v",
                        action="store_true",
                        help="Show full responses")
    args = parser.parse_args()

    run_hle_baseline(num_problems=args.num_problems, model=args.model)
