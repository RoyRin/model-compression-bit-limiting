"""LLM-as-judge evaluation for freeform answers."""

from lossy_compression import model_completion, MODEL_ALIAS_MAP, MODEL_ALIAS_MAP_old


def judge_freeform_answer(question: str,
                          gold_answer: str,
                          model_response: str,
                          judge_model: str = None,
                          model_map: dict = None) -> bool:
    """Use Opus as judge to evaluate if model's answer is semantically correct.

    Args:
        question: The original question
        gold_answer: The correct answer
        model_response: The model's response to evaluate
        judge_model: Optional override for judge model (defaults to Opus from model_map)
        model_map: Optional model alias map (defaults to MODEL_ALIAS_MAP)

    Returns:
        True if the answer is correct, False otherwise.
    """
    if not model_response:
        return False

    judge_prompt = f"""You are an EXTREMELY STRICT evaluator. Your job is to check if the model's answer matches the correct answer.

Correct Answer: {gold_answer}

Model's Response: {model_response}

STRICT RULES:
1. If the correct answer has MULTIPLE PARTS, ALL parts must be correct. Getting some parts right is still INCORRECT.
2. The model must state the SAME conclusion as the correct answer, not a different one.
3. For numerical answers: must match within 1% (e.g., 0.7 vs 0.693 is OK, but 0.7 vs 0.5 is INCORRECT)
4. Pay attention to WHICH entities are described - "G1 is epistatic to G3" is DIFFERENT from "G2 is epistatic to G1"
5. The model discussing the topic or showing work is NOT enough - the final answer must match.
6. When in doubt, mark INCORRECT. Only mark CORRECT if you are confident the answer fully matches.

EXAMPLES:
- Gold: "G1 is epistatic towards G3", Model says "G2 is epistatic to G1 and G3" → INCORRECT (wrong gene relationships)
- Gold: "G2 is a transcription factor, G1 is epistatic to G3", Model says "G2 is a transcription factor, G2 is epistatic" → INCORRECT (got epistasis wrong)
- Gold: "-0.7", Model calculates and gets "-0.5" → INCORRECT
- Gold: "-0.7", Model calculates and gets "-0.7" → CORRECT
- Gold: "compound X", Model discusses chemistry but concludes "compound Y" → INCORRECT
- Gold: "42", Model says "the answer is 42" → CORRECT

Respond with ONLY "CORRECT" or "INCORRECT"."""

    # Use provided model map or default
    if model_map is None:
        model_map = MODEL_ALIAS_MAP

    # Default to Opus as judge
    if judge_model is None:
        judge_model = model_map.get('opus', 'claude-opus-4-5-20251101')

    try:
        response = model_completion(
            model=judge_model,
            system=
            "You are a precise scientific answer evaluator. Respond only with CORRECT or INCORRECT.",
            prompt=judge_prompt,
            max_tokens=10,
            temperature=0.0,
        )

        if response:
            return 'CORRECT' in response.upper()
        return False
    except Exception as e:
        print(f"  Judge error: {e}")
        return False
