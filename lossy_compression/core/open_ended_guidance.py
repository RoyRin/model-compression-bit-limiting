from lossy_compression.utils import model_wrappers as model_messaging_wrappers
from lossy_compression import LLM, DEFAULT_LLM_TEMPERATURE, SLM, DEFAULT_SLM_TEMPERATURE


def should_trigger_guidance(guiding_questions, quality_scores, guidance_used,
                            max_iterations):
    """Check if guidance should be triggered at 2/3 of max iterations."""
    current_iteration = len(guiding_questions)
    two_thirds_point = int(max_iterations * 2 / 3)

    # Trigger at 2/3 point if not already used
    return (current_iteration >= two_thirds_point and not guidance_used)


def handle_open_ended_guidance(original_prompt,
                               large_model_answer,
                               current_answer,
                               large_model_name,
                               small_model_name,
                               llm_temperature,
                               slm_temperature,
                               seed,
                               guidances,
                               guiding_questions,
                               guiding_answers,
                               local_model=None,
                               verbose=False):
    """Handle open-ended guidance generation and incorporation."""
    if verbose:
        print("💡 Requesting open-ended guidance from LLM...")

    guidance = model_messaging_wrappers.open_ended_LLM_guidance(
        original_prompt,
        large_model_answer,
        current_answer,
        large_model_name=large_model_name,
        temperature=llm_temperature,
        seed=seed,
        verbose=verbose)
    guidances.append(guidance)

    updated_answer = model_messaging_wrappers.small_model_improve_with_guidance(
        original_prompt,
        small_model_answer=current_answer,
        guiding_questions=guiding_questions,
        guiding_answers=guiding_answers,
        open_ended_guidances=guidances,
        small_model_name=small_model_name,
        temperature=slm_temperature,
        seed=seed,
        use_local=local_model is not None,
        local_model=local_model,
        verbose=verbose)

    return updated_answer, guidances


def open_ended_LLM_guidance(original_prompt,
                            large_model_answer,
                            small_model_answer,
                            large_model_name=LLM,
                            temperature=DEFAULT_LLM_TEMPERATURE,
                            seed=None,
                            max_tokens=100,
                            verbose=False):
    """Get open-ended guidance from the large model to help improve the small model's answer."""
    if verbose:
        print("💡 LLM providing open-ended guidance...")
    prompt = f"""You are a large language model that is helping guide a small model. The original prompt is: {original_prompt}. Your answer is: {large_model_answer}. The small model answer's is: {small_model_answer}. 
        
Please provide general guidance principles for answering this type of question better. DO NOT reference specific details from the small model's current answer, as it may generate a different response. Instead, give guidance like:
- "You should avoid doing X"
- "You should emphasize Y"
- "Make sure to include Z"
- "Focus on explaining A before B"

Provide clear, actionable principles that would improve ANY answer to this question. Keep it under 100 words tops; ideally less than 40 words."""

    resp = generate_LLM_response(prompt,
                                 model_name=large_model_name,
                                 temperature=temperature,
                                 seed=seed,
                                 verbose=False,
                                 max_tokens=max_tokens)

    if verbose:
        print(f"💡 LLM guidance: {resp}")

    return resp


def _small_model_improve_with_guidance__1(
        original_prompt,
        system_prompt,  # NEW
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
    """Generate improved answer using both binary Q&A pairs and open-ended guidance.
    
    This unified function combines the functionality of:
    - small_model_generate_answers_given_binary_answers
    - small_model_incorporate_open_ended_guidance

    
    Returns:
        Improved answer incorporating all available guidance
    """

    # Detect if this is a code generation task
    is_code_task = (system_prompt and ("code" in system_prompt.lower() or "python" in system_prompt.lower() or "function" in system_prompt.lower())) or \
                   ("def " in original_prompt or "```" in original_prompt or "import " in original_prompt)

    # Create an improvement-specific system prompt
    if system_prompt:
        # Enhance existing system prompt with improvement context
        if is_code_task:
            improvement_system_prompt = f"""{system_prompt}

You are now improving your code based on additional Q&A guidance. Fix any errors, handle edge cases mentioned, and incorporate all corrections indicated by the Q&A pairs. Ensure the code is syntactically correct and properly formatted."""
        else:
            improvement_system_prompt = f"""{system_prompt}

You are now improving your previous answer based on additional Q&A guidance. Incorporate all provided information while maintaining accuracy and coherence. Focus on addressing any gaps or corrections indicated by the Q&A pairs."""
    else:
        # Default improvement system prompt
        if is_code_task:
            improvement_system_prompt = """You are refining code based on Q&A guidance. 
Fix any errors, handle edge cases, and incorporate all corrections indicated.
Return only working, properly formatted code without explanations."""
        else:
            improvement_system_prompt = """You are refining an answer based on Q&A guidance. 
Incorporate all additional information provided to create a more complete and accurate response.
Answer directly without meta-commentary or acknowledgment of the revision process."""

    # Build a cleaner prompt structure
    if system_prompt:
        # With system prompt, we can be more direct since instructions are in system
        prompt_parts = [original_prompt]

        # Add context about previous attempt
        if small_model_answer:
            prompt_parts.append(f"\n[Previous attempt: {small_model_answer}]")

        # Add Q&A guidance if available
        if guiding_questions and guiding_answers:
            guiding_qa = format_guiding_questions_and_answers(
                guiding_questions, guiding_answers)
            prompt_parts.append(f"\n[Guidance from Q&A:\n{guiding_qa}]")

        # Add open-ended guidance if available
        if open_ended_guidances:
            combined_guidance = "\n".join(
                [f"- {g}" for g in open_ended_guidances])
            prompt_parts.append(
                f"\n[Additional guidance:\n{combined_guidance}]")

        prompt = "\n".join(prompt_parts)
    else:
        # Without system prompt, keep instructions in user prompt (fallback)
        prompt_parts = [
            f"Original question: {original_prompt}",
            f"\nCurrent answer: {small_model_answer}"
        ]

        if guiding_questions and guiding_answers:
            guiding_qa = format_guiding_questions_and_answers(
                guiding_questions, guiding_answers)
            prompt_parts.append(
                f"\nAdditional information to incorporate: {guiding_qa}")

        prompt_parts.append(
            "\nProvide an improved answer to the original question that incorporates the additional information above. Answer the question directly without any preamble or acknowledgment."
        )

        prompt = "\n".join(prompt_parts)

    new_response = generate_SLM_response(
        prompt,
        system_prompt=
        improvement_system_prompt,  # Use the enhanced system prompt
        model_name=small_model_name,
        temperature=temperature,
        verbose=False)

    if verbose:
        print(f"🔄 Improved answer: {new_response[:100]}...")

    return new_response


def small_model_improve_with_guidance__decent(
        original_prompt,
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
    """Generate improved answer using both binary Q&A pairs and open-ended guidance."""

    # Detect if this is a code generation task
    is_code_task = (system_prompt and ("code" in system_prompt.lower() or "python" in system_prompt.lower())) or \
                   ("def " in original_prompt or "```" in original_prompt or "import " in original_prompt)

    if is_code_task:
        # Build the system prompt with ALL instructions and guidance
        system_parts = [
            "You are a Python code completion assistant.",
            "Complete the given Python function by providing the full implementation including the function signature.",
            "Return only valid Python code without any markdown formatting, explanations, or additional text."
        ]

        # Add Q&A guidance to the system prompt
        if guiding_questions and guiding_answers:
            system_parts.append("\nRequirements based on Q&A:")
            for q, a in zip(guiding_questions, guiding_answers):
                # Handle boolean answers properly
                if isinstance(a, bool):
                    answer_is_yes = a
                elif isinstance(a, str):
                    answer_is_yes = a.lower() in ['yes', 'true', '1']
                else:
                    answer_is_yes = bool(a)

                if answer_is_yes:
                    system_parts.append(f"- {q}: YES")
                else:
                    system_parts.append(f"- {q}: NO")

        # Add note about previous attempt if exists
        if small_model_answer:
            system_parts.append(
                f"\nNote: A previous attempt had issues. Ensure your implementation addresses all the requirements above."
            )

        improvement_system_prompt = "\n".join(system_parts)

        # User prompt is JUST the code to complete - super clean!
        prompt = original_prompt
        prompt = f"Please repeat the prompt in your answer, and answer the original prompt. If there are multiple functions in the prompt, include all of them in your answer:\n{original_prompt}"

    else:
        # Non-code improvement - keep original approach
        improvement_system_prompt = system_prompt or "You are refining an answer based on Q&A guidance."

        prompt_parts = [
            f"Original question: {original_prompt}",
            f"\nCurrent answer: {small_model_answer}"
        ]

        if guiding_questions and guiding_answers:
            guiding_qa = format_guiding_questions_and_answers(
                guiding_questions, guiding_answers)
            prompt_parts.append(
                f"\nAdditional information to incorporate: {guiding_qa}")

        if open_ended_guidances:
            combined_guidance = "\n".join(
                [f"- {g}" for g in open_ended_guidances])
            prompt_parts.append(f"\nAdditional guidance:\n{combined_guidance}")

        prompt_parts.append(
            "\nProvide an improved answer that incorporates all the above information, without any preamble or acknowledgment."
        )
        prompt_parts.append(
            f"Please repeat the prompt in your answer, and answer the original prompt. If there are multiple functions in the prompt, include all of them in your answer:\n{original_prompt}"
        )

        prompt = "\n".join(prompt_parts)

    # Generate response

    new_response = generate_SLM_response(
        prompt,
        system_prompt=improvement_system_prompt,
        model_name=small_model_name,
        temperature=temperature,
        verbose=False)

    if verbose:
        print(f"🔄 Improved answer: {new_response[:100]}...")

    return new_response
