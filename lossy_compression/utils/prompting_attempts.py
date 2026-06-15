###########

import re

STRICT_CODE_SYSTEM = """You are a Python code completion assistant.

Hard rules (must follow exactly):
1) Output ONLY Python source code. No prose, no markdown fences, no explanations.
2) You MUST include the entire content between '# BEGIN-LOCKED' and '# END-LOCKED' UNCHANGED at the top of your output.
3) You may write or modify code ONLY between '# BEGIN-EDITABLE' and '# END-EDITABLE', or APPEND code AFTER the locked block if no editable region exists.
4) Do not delete, reorder, or reformat any line from the locked region. Preserve spacing/indentation exactly.
5) Make the smallest correct completion that satisfies the requirements/hints.
6) If unsure, keep the locked region unchanged and only append minimal code needed.
"""


def _format_guidance(qs, ans, extra=None):
    lines = []
    if qs and ans:
        for q, a in zip(qs, ans):
            yn = a if isinstance(
                a, bool) else str(a).strip().lower() in ("yes", "true", "1")
            lines.append(f"- {q}: {'YES' if yn else 'NO'}")
    if extra:
        lines.extend(f"- {g}" for g in extra)
    return "\n".join(lines) if lines else ""


def _wrap_prompt_with_markers(original_prompt: str,
                              small_model_answer: str | None,
                              guidance_text: str | None) -> str:
    """
    Build a single clear prompt the model can complete directly.
    The model is instructed to return raw Python code only.
    """
    parts = []
    parts.append("# BEGIN-LOCKED")
    parts.append(original_prompt.rstrip())
    parts.append("# END-LOCKED\n")

    parts.append("# BEGIN-EDITABLE")
    # If there is a partial attempt to improve/finish, place it here.
    if small_model_answer:
        parts.append(small_model_answer.rstrip())
    else:
        # Give a concrete completion cue if helpful
        parts.append(
            "# TODO: complete the implementation below without changing the locked code above."
        )
    parts.append("# END-EDITABLE")

    if guidance_text:
        parts.append(
            "\n# HINTS (for your reasoning only; DO NOT copy these lines to your output beyond comments here)"
        )
        parts.append("# " + "\n# ".join(guidance_text.splitlines()))
    return "\n".join(parts)


def small_model_improve_with_guidance(original_prompt,
                                      system_prompt,
                                      small_model_answer,
                                      guiding_questions=None,
                                      guiding_answers=None,
                                      open_ended_guidances=None,
                                      small_model_name=SLM,
                                      temperature=DEFAULT_SLM_TEMPERATURE,
                                      seed=None,
                                      use_local=False,
                                      local_model=None,
                                      verbose=False):
    """Generate improved code using binary Q&A pairs and open-ended guidance, via prompt-only constraints."""
    # Treat as code if it looks like code
    is_code_task = (system_prompt and ("code" in system_prompt.lower() or "python" in system_prompt.lower())) or \
                   ("def " in original_prompt or "```" in original_prompt or "import " in original_prompt)

    if is_code_task:
        guidance_text = _format_guidance(guiding_questions, guiding_answers,
                                         open_ended_guidances)
        prompt = _wrap_prompt_with_markers(original_prompt, small_model_answer,
                                           guidance_text)
        improvement_system_prompt = STRICT_CODE_SYSTEM
        # If your API supports stop sequences, consider:
        # stop = ["# HINTS"]  # prevents echoing hints into output
    else:
        # Fallback for non-code (keeps it simple, still “no prose” isn’t enforced)
        improvement_system_prompt = (
            system_prompt or
            "You are refining an answer based on Q&A guidance. Answer directly without preamble."
        )
        prompt_parts = [
            f"Original question: {original_prompt}",
            f"\nCurrent answer: {small_model_answer or ''}"
        ]
        if guiding_questions and guiding_answers:
            prompt_parts.append(
                "\n" + _format_guidance(guiding_questions, guiding_answers))
        if open_ended_guidances:
            prompt_parts.append("\n" +
                                "\n".join(f"- {g}"
                                          for g in open_ended_guidances))
        prompt_parts.append("\nProvide the improved answer directly.")
        prompt = "\n".join(prompt_parts)
    ####
    raw = generate_SLM_response(
        prompt,
        system_prompt=improvement_system_prompt,
        model_name=small_model_name,
        temperature=temperature,
        verbose=False,
        # stop=stop if supported
    )

    if verbose:
        print(f"DEBUG – system prompt:\n{improvement_system_prompt}\n---")
        print(f"DEBUG – user prompt:\n{prompt}\n---")
        print(f"DEBUG – model out (first 150): {raw[:150]!r}\n---")

    # Return the model output directly (raw Python expected in code path)
    return raw
