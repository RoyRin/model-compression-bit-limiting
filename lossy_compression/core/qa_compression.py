from utils.llm_api import openai_completion, anthropic_completion
import random
import re
from tqdm import tqdm
from functools import partial
from lossy_compression.utils import model_wrappers

# For backwards compatibility with existing code
model_messaging_wrappers = model_wrappers
from lossy_compression.utils.parallel_utils import parallel_execute, parallel_map
# Configuration - Choose which provider to use
from lossy_compression import LLM, SLM, QUESTION_SLM, DEFAULT_LLM_TEMPERATURE, DEFAULT_SLM_TEMPERATURE, DEFAULT_SEED, LOCAL_SLM, USE_ANTHROPIC
from lossy_compression.core.open_ended_guidance import should_trigger_guidance, handle_open_ended_guidance

# Define evaluation modes
EVAL_MODE_DEFAULT = "default"
EVAL_MODE_CODE = "code"
EVAL_MODE_MATH = "math"
EVAL_MODE_SCIENCE = "science"

# TODO - we should assume that we have the LLM's token-ranks as well, in order to figure out where it diverges maximally. We can add that in later.


# Helper functions for refactored iterative loop
def setup_iteration_config(original_prompt, large_model_name, small_model_name,
                           question_model_name, llm_temperature,
                           slm_temperature, seed, device, verbose):
    """Setup initial configuration and models."""
    random.seed(seed)

    if verbose:
        print(f"Starting iterative SLM loop...")
        print(f"Small model: {small_model_name}")
        print(f"Question generation model: {question_model_name}")

    # Determine if we should use a local model based on the model name
    # If it's a HuggingFace model (contains '/'), we'll use local
    use_local_model = '/' in small_model_name

    local_model = None
    if use_local_model:
        # Lazy import - only load HFModel when actually using local models
        from utils.hf_models import HFModel

        # Auto-detect device if not specified
        if device is None:
            import torch
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'

        if verbose:
            print(f"🔧 Initializing local model: {small_model_name}")
            print(
                f"🖥️  Device: {device} {'(Apple Silicon GPU)' if device == 'mps' else '(NVIDIA GPU)' if device == 'cuda' else '(CPU)'}"
            )
        local_model = HFModel(small_model_name, device=device)
    elif verbose:
        print(f"🌐 Using remote API model: {small_model_name}")

    return local_model


def log_slm_answer(slm_answers,
                   iteration,
                   answer,
                   score,
                   answer_type,
                   verbose=False):
    """Log an SLM answer with its metadata."""
    slm_answers.append({
        'iteration': iteration,
        'answer': answer,
        'score': score,
        'type': answer_type
    })
    if verbose:
        print(
            f"📊 {answer_type.title()} SLM answer logged with score: {score}/10"
        )
    return slm_answers


JUDGE_MODE_COMPARISON = "comparison"  # Compare SLM's answer vs LLM's answer
JUDGE_MODE_OBJECTIVE = "objective"  # Standalone quality evaluation (no reference)


def evaluate_and_log_quality(original_prompt,
                             large_model_answer,
                             current_answer,
                             llm_temperature,
                             seed,
                             slm_answers,
                             iteration,
                             answer_type,
                             verbose=False,
                             evaluation_mode=EVAL_MODE_DEFAULT,
                             gold_answer=None,
                             judge_model=None,
                             judge_mode=JUDGE_MODE_COMPARISON):
    """Evaluate quality and log the answer.

    Args:
        judge_mode: "comparison" = compare SLM's answer vs LLM's own answer as reference.
                    "objective" = standalone quality evaluation (no reference needed).
                    Only applies when gold_answer is None and large_model_answer is available.

    Scale: 9-10=Excellent, 7-8=Good, 5-6=Adequate, 3-4=Poor, 1-2=Very Poor
    """
    use_comparison = (judge_mode == JUDGE_MODE_COMPARISON
                      and large_model_answer is not None
                      and gold_answer is None)

    # Use appropriate evaluation based on mode
    if evaluation_mode == EVAL_MODE_MATH:
        # Math-specific evaluation
        if gold_answer:
            # If we have gold answer, check exact match first
            is_correct, extracted = model_messaging_wrappers.evaluate_math_against_gold(
                current_answer, gold_answer, aime=True)
            quality_score = 10 if is_correct else model_messaging_wrappers.evaluate_math_solution_quality(
                problem=original_prompt,
                solution_text=current_answer,
                model_name=judge_model or LLM,
                temperature=llm_temperature,
                seed=seed,
                verbose=False)
        elif use_comparison:
            # Compare SLM's solution against LLM's solution
            quality_score = model_messaging_wrappers.evaluate_math_answer_vs_reference(
                proposed_answer=current_answer,
                correct_answer=large_model_answer,
                model_name=judge_model or LLM,
                temperature=llm_temperature,
                seed=seed,
                verbose=verbose)
        else:
            quality_score = model_messaging_wrappers.evaluate_math_solution_quality(
                problem=original_prompt,
                solution_text=current_answer,
                model_name=judge_model or LLM,
                temperature=llm_temperature,
                seed=seed,
                verbose=False)
    elif evaluation_mode == EVAL_MODE_CODE:
        if use_comparison:
            # Compare SLM's code against LLM's code
            quality_score = model_messaging_wrappers.evaluate_answer_quality_comparison(
                original_prompt=original_prompt,
                large_model_answer=large_model_answer,
                small_model_answer=current_answer,
                temperature=llm_temperature,
                seed=seed,
                verbose=verbose,
                model_name=judge_model or LLM)
        else:
            quality_score = model_messaging_wrappers.evaluate_code_answer(
                prompt=original_prompt,
                answer=current_answer,
                temperature=llm_temperature,
                seed=seed,
                verbose=False,
                model_name=judge_model or LLM)
    else:
        # Default / Science evaluation mode
        if use_comparison:
            # Compare SLM's answer against LLM's answer
            quality_score = model_messaging_wrappers.evaluate_answer_quality_comparison(
                original_prompt=original_prompt,
                large_model_answer=large_model_answer,
                small_model_answer=current_answer,
                temperature=llm_temperature,
                seed=seed,
                verbose=verbose,
                model_name=judge_model or LLM)
        else:
            quality_score = model_messaging_wrappers.evaluate_answer_standalone(
                prompt=original_prompt,
                answer=current_answer,
                temperature=llm_temperature,
                seed=seed,
                verbose=False,
                model_name=judge_model or LLM)
    log_slm_answer(slm_answers, iteration, current_answer, quality_score,
                   answer_type, verbose)
    return quality_score


def should_trigger_guidance(guiding_questions, quality_scores, guidance_used,
                            max_iterations):
    """Check if guidance should be triggered at 2/3 of max iterations."""
    current_iteration = len(guiding_questions)
    two_thirds_point = int(max_iterations * 2 / 3)

    # Trigger at 2/3 point if not already used
    return (current_iteration >= two_thirds_point and not guidance_used)


def process_single_iteration(
        prompt,
        system_prompt,  # NEW
        large_model_answer,
        current_answer,
        question_model_name,
        small_model_name,
        local_model,
        slm_temperature,
        llm_temperature,
        seed,
        guiding_questions,
        guiding_answers,
        guidances,
        evaluation_mode=EVAL_MODE_DEFAULT,
        verbose=False,
        predict_base_rate=False,
        predicted_probs=None):
    """Process a single iteration of the loop.

    Args:
        predict_base_rate: If True, also track P(yes) predictions for probabilistic compression.
        predicted_probs: List to append probability predictions to (mutated in place).
    """
    # Generate question based on evaluation mode
    p_yes = 0.5  # Default probability if not predicting
    if evaluation_mode == EVAL_MODE_MATH:
        result = model_messaging_wrappers.small_model_generate_helpful_binary_questions_math(
            prompt,
            system_prompt=system_prompt,
            original_answer=current_answer,
            existing_guiding_questions=guiding_questions,
            existing_guiding_answers=guiding_answers,
            question_model_name=question_model_name,
            temperature=slm_temperature,
            seed=seed,
            verbose=verbose,
            predict_base_rate=predict_base_rate)
        if predict_base_rate:
            question, p_yes = result
        else:
            question = result
    elif evaluation_mode == EVAL_MODE_CODE:
        # Use default for now, could add code-specific later
        result = model_messaging_wrappers.small_model_generate_helpful_binary_questions(
            prompt,
            system_prompt=None,
            original_answer=current_answer,
            existing_guiding_questions=guiding_questions,
            existing_guiding_answers=guiding_answers,
            question_model_name=question_model_name,
            temperature=slm_temperature,
            seed=seed,
            verbose=verbose,
            predict_base_rate=predict_base_rate)
        if predict_base_rate:
            question, p_yes = result
        else:
            question = result
    else:
        # Default mode
        result = model_messaging_wrappers.small_model_generate_helpful_binary_questions(
            prompt,
            system_prompt=None,
            original_answer=current_answer,
            existing_guiding_questions=guiding_questions,
            existing_guiding_answers=guiding_answers,
            question_model_name=question_model_name,
            temperature=slm_temperature,
            seed=seed,
            verbose=verbose,
            predict_base_rate=predict_base_rate)
        if predict_base_rate:
            question, p_yes = result
        else:
            question = result

    # Get binary answer based on evaluation mode
    if evaluation_mode == EVAL_MODE_MATH:
        binary_answer = model_messaging_wrappers.large_model_answer_binary_question_math(
            prompt,
            system_prompt=system_prompt,
            large_model_answer=large_model_answer,
            small_model_answer=current_answer,
            question=question,
            model_name=LLM,
            temperature=llm_temperature,
            seed=seed,
            verbose=verbose)
    elif evaluation_mode == EVAL_MODE_CODE:
        # Use default for now, could add code-specific later
        binary_answer = model_messaging_wrappers.large_model_answer_binary_question(
            prompt,
            system_prompt=None,
            large_model_answer=large_model_answer,
            small_model_answer=current_answer,
            question=question,
            temperature=llm_temperature,
            seed=seed,
            verbose=verbose)
    else:
        # Default mode
        binary_answer = model_messaging_wrappers.large_model_answer_binary_question(
            prompt,
            system_prompt=None,
            large_model_answer=large_model_answer,
            small_model_answer=current_answer,
            question=question,
            temperature=llm_temperature,
            seed=seed,
            verbose=verbose)

    # Update guiding questions/answers
    guiding_questions.append(question)
    guiding_answers.append(binary_answer)

    # Track predicted probability if using probabilistic compression
    if predict_base_rate and predicted_probs is not None:
        predicted_probs.append(p_yes)

    # Update answer based on evaluation mode
    if evaluation_mode == EVAL_MODE_MATH:
        updated_answer = model_messaging_wrappers.small_model_improve_with_guidance_math(
            prompt,
            system_prompt=system_prompt,
            small_model_answer=current_answer,
            guiding_questions=guiding_questions,
            guiding_answers=guiding_answers,
            open_ended_guidances=guidances if guidances else None,
            small_model_name=small_model_name,
            temperature=slm_temperature,
            seed=seed,
            use_local=local_model is not None,
            local_model=local_model,
            verbose=verbose,
            aime=True)  # Set AIME mode for proper formatting
    elif evaluation_mode == EVAL_MODE_SCIENCE:
        # Science-specific improvement
        updated_answer = model_messaging_wrappers.small_model_improve_with_guidance_science(
            prompt,
            system_prompt=system_prompt,
            small_model_answer=current_answer,
            guiding_questions=guiding_questions,
            guiding_answers=guiding_answers,
            open_ended_guidances=guidances if guidances else None,
            small_model_name=small_model_name,
            temperature=slm_temperature,
            seed=seed,
            use_local=local_model is not None,
            local_model=local_model,
            verbose=verbose)
    elif evaluation_mode == EVAL_MODE_CODE:
        # Use default for now, could add code-specific later
        updated_answer = model_messaging_wrappers.small_model_improve_with_guidance(
            prompt,
            system_prompt=system_prompt,
            small_model_answer=current_answer,
            guiding_questions=guiding_questions,
            guiding_answers=guiding_answers,
            open_ended_guidances=guidances if guidances else None,
            small_model_name=small_model_name,
            temperature=slm_temperature,
            seed=seed,
            use_local=local_model is not None,
            local_model=local_model,
            verbose=verbose)
    else:
        # Default mode
        updated_answer = model_messaging_wrappers.small_model_improve_with_guidance(
            prompt,
            system_prompt=system_prompt,
            small_model_answer=current_answer,
            guiding_questions=guiding_questions,
            guiding_answers=guiding_answers,
            open_ended_guidances=guidances if guidances else None,
            small_model_name=small_model_name,
            temperature=slm_temperature,
            seed=seed,
            use_local=local_model is not None,
            local_model=local_model,
            verbose=verbose)

    return updated_answer, guiding_questions, guiding_answers


def find_best_answer(slm_answers,
                     guiding_questions,
                     guiding_answers,
                     quality_scores,
                     open_ended_guidance_used,
                     original_prompt=None,
                     verbose=False):
    """Find and return the first answer with the highest score."""
    max_score = max(slm_answers, key=lambda x: x['score'])['score']
    best_entry = next(answer for answer in slm_answers
                      if answer['score'] == max_score)

    if verbose:
        print(f"\n🏆 First answer with highest score:")
        print(f"   Iteration: {best_entry['iteration']}")
        print(f"   Score: {best_entry['score']}/10")
        print(f"   Type: {best_entry['type']}")
        if original_prompt:
            print(f"\n   Full Prompt: {original_prompt}")
    print(f"\n   Full Answer:------\n{best_entry['answer']}\n-------")

    return best_entry['answer'], (guiding_questions, guiding_answers), {
        'iterations': len(quality_scores),
        'final_quality_score': quality_scores[-1] if quality_scores else 0,
        'best_quality_score': best_entry['score'],
        'best_iteration': best_entry['iteration'],
        'total_qa_pairs': len(guiding_questions),
        'quality_scores': quality_scores,
        'quality_progression': quality_scores,
        'open_ended_guidance_used': open_ended_guidance_used,
        'slm_answers': slm_answers
    }


def get_SLM_log_probs(prompt, model_name=SLM):
    """ """
    # TODO implement this in llm_api
    raise


def token_guiding_LLM_response(original_prompt,
                               large_model_answer,
                               small_model_answer,
                               large_model_name=LLM):
    """ """
    raise


def print_iteration_summary(iteration_count,
                            quality_scores,
                            guidances,
                            guiding_questions,
                            quality_score,
                            open_ended_guidance_used,
                            verbose=False):
    """Print a summary of the iterative process results."""
    if not verbose:
        return

    print(f"\n--- Final Results ---")
    print(f"Total iterations: {iteration_count}")
    print(f"Final quality score: {quality_score}/10")
    print(f"Total Q&A pairs: {len(guiding_questions)}")
    print(
        f"Open-ended guidance used: {'Yes' if open_ended_guidance_used else 'No'}"
    )

    print(f"\n{'='*60}")
    print("📊 ITERATIVE PROCESS SUMMARY")
    print(f"{'='*60}")
    print(f"Quality scores: {quality_scores}")
    print(f"Quality progression: {quality_scores}")

    if len(quality_scores) > 1:
        improvement = quality_scores[-1] - quality_scores[0]
        print(f"Overall improvement: {improvement:+.1f}")
        print(f"  Best score: {max(quality_scores)}/10")
        print(
            f"  Average score: {sum(quality_scores)/len(quality_scores):.1f}/10"
        )

    if guidances:
        print(f"Guidances provided:")
        for i, guidance in enumerate(guidances):
            print(f"  Iteration {i+1}: {guidance}")


def iterative_SLM_loop(
        prompt,  # The actual task/question (renamed from original_prompt)
        system_prompt=None,  # Instructions on HOW to respond (NEW)
        large_model_name=LLM,
        small_model_name=SLM,
        question_model_name=QUESTION_SLM,
        use_local_slm=False,
        max_iterations=5,
        quality_threshold=7,
        seed=DEFAULT_SEED,
        llm_temperature=DEFAULT_LLM_TEMPERATURE,
        slm_temperature=DEFAULT_SLM_TEMPERATURE,
        device=None,  # Device for local model (cuda/mps/cpu/None for auto)
        open_ended_guidance=False,  # Enable open-ended guidance (default: False)
        enable_parallel=False,  # Enable parallel API calls where possible
        use_code_evaluation=False,  # DEPRECATED: Use evaluation_mode="code" instead
        evaluation_mode=EVAL_MODE_DEFAULT,  # "default", "code", or "math"
        gold_answer=None,  # Gold answer for math problems (optional)
        skip_llm_initial=False,  # Skip initial LLM generation to save costs
        batch_mode=False,  # Enable batch Q&A generation
        batch_size=10,  # Number of questions to generate at once in batch mode
        oracle_solution=None,  # Oracle mode: use this as the reference solution instead of LLM
        predict_base_rate=False,  # Enable probabilistic compression (SLM predicts P(yes))
        judge_mode=JUDGE_MODE_COMPARISON,  # "comparison" or "objective"
        verbose=False):
    """Iterative SLM loop with Q&A compression.

    Args:
        predict_base_rate: If True, SLM predicts P(yes) for each question, enabling
                          probabilistic compression. Uses Shannon information theory:
                          bits = -log2(P(actual_answer)) instead of 1 bit per question.
        judge_mode: "comparison" = judge compares SLM answer vs LLM answer as reference.
                    "objective" = judge evaluates SLM answer quality standalone (no reference).
                          This can achieve < 1 bit per question if SLM is well-calibrated.
    """
    # Setup configuration and models
    local_model = setup_iteration_config(prompt, large_model_name,
                                         small_model_name, question_model_name,
                                         llm_temperature, slm_temperature,
                                         seed, device, verbose)

    # Generate initial answers
    if oracle_solution is not None:
        # Oracle mode: use provided solution as reference
        large_model_answer = oracle_solution
        if verbose:
            print("🔮 Oracle mode: using provided solution as reference")
    elif skip_llm_initial:
        # Skip LLM generation to save costs - use placeholder
        large_model_answer = None
        if verbose:
            print("⏩ Skipping initial LLM generation (skip_llm_initial=True)")
    else:
        # Generate LLM answer (needed for Q&A guidance)
        if enable_parallel and not use_local_slm:
            if verbose:
                print("⚡ Generating initial answers in parallel...")

            # Prepare parallel tasks
            tasks = [
                (model_messaging_wrappers.generate_LLM_response, (prompt, ), {
                    'system_prompt': system_prompt,
                    'model_name': large_model_name,
                    'temperature': llm_temperature,
                    'seed': seed,
                    'verbose': False
                }),
                (model_messaging_wrappers.generate_SLM_response, (prompt, ), {
                    'system_prompt': system_prompt,
                    'model_name': small_model_name,
                    'temperature': slm_temperature,
                    'verbose': False
                })
            ]

            # Execute in parallel
            results = parallel_execute(tasks, max_workers=2)
            large_model_answer = results[0]
            small_model_answer = results[1]
        else:
            # Sequential execution
            large_model_answer = model_messaging_wrappers.generate_LLM_response(
                prompt,
                system_prompt=system_prompt,
                model_name=large_model_name,
                temperature=llm_temperature,
                seed=seed,
                verbose=False)

            small_model_answer = model_messaging_wrappers.generate_SLM_response(
                prompt,
                system_prompt=system_prompt,
                model_name=small_model_name,
                temperature=slm_temperature,
                verbose=False)

    # Always generate initial SLM answer if not already done
    if skip_llm_initial or oracle_solution is not None:
        small_model_answer = model_messaging_wrappers.generate_SLM_response(
            prompt,
            system_prompt=system_prompt,
            model_name=small_model_name,
            temperature=slm_temperature,
            verbose=False)

    if verbose:
        if large_model_answer:
            print(f"Initial large model answer: {large_model_answer[:100]}...")
        else:
            print("Initial large model answer: [SKIPPED]")
        print(f"Initial small model answer: {small_model_answer[:100]}...")

    # Initialize state
    state = {
        'guiding_questions': [],
        'guiding_answers': [],
        'predicted_probs':
        [],  # P(yes) predictions for probabilistic compression
        'current_answer': small_model_answer,
        'quality_scores': [],
        'slm_answers': [],
        'guidances': [],
        'open_ended_guidance_used': False,
        'iteration_count': 0
    }

    # Handle deprecated use_code_evaluation parameter
    if use_code_evaluation and evaluation_mode == EVAL_MODE_DEFAULT:
        evaluation_mode = EVAL_MODE_CODE

    # Log initial answer - skip quality evaluation if no LLM answer for comparison
    if skip_llm_initial and evaluation_mode == EVAL_MODE_DEFAULT:
        # Can't evaluate without reference, use default low score
        initial_score = 3
        if verbose:
            print("📊 Initial score set to 3 (no LLM reference for comparison)")
    else:
        initial_score = evaluate_and_log_quality(prompt,
                                                 large_model_answer,
                                                 small_model_answer,
                                                 llm_temperature,
                                                 seed,
                                                 state['slm_answers'],
                                                 0,
                                                 'initial',
                                                 verbose,
                                                 evaluation_mode,
                                                 gold_answer,
                                                 large_model_name,
                                                 judge_mode=judge_mode)
    state['quality_scores'].append(initial_score)

    # Main iteration loop
    if batch_mode:
        # Batch mode: Process questions in batches
        if verbose:
            total_questions = max_iterations
            num_batches = (total_questions + batch_size -
                           1) // batch_size  # Ceiling division
            print(
                f"\n🚀 BATCH MODE: Processing {total_questions} questions in {num_batches} batch(es) of up to {batch_size} each"
            )

        # Check if we need to generate LLM answer for Q&A guidance
        if large_model_answer is None:
            if verbose:
                print("🔄 Generating LLM answer for Q&A guidance...")
            large_model_answer = model_messaging_wrappers.generate_LLM_response(
                prompt,
                system_prompt=system_prompt,
                model_name=large_model_name,
                temperature=llm_temperature,
                seed=seed,
                verbose=False)

        # Process in batches
        questions_processed = 0
        batch_num = 0

        while questions_processed < max_iterations:
            batch_num += 1
            # Calculate questions for this batch
            questions_remaining = max_iterations - questions_processed
            current_batch_size = min(batch_size, questions_remaining)

            if verbose:
                print(
                    f"\n📦 Batch {batch_num}: Generating {current_batch_size} questions..."
                )

            # Generate questions for this batch
            batch_result = model_messaging_wrappers.batch_generate_questions(
                prompt,
                system_prompt,
                state['current_answer'],
                state['guiding_questions'],
                state['guiding_answers'],
                num_questions=current_batch_size,
                question_model_name=question_model_name,
                temperature=slm_temperature,
                seed=seed,
                verbose=verbose,
                evaluation_mode=evaluation_mode.lower() if evaluation_mode
                in [EVAL_MODE_MATH, EVAL_MODE_SCIENCE] else "default",
                predict_base_rate=predict_base_rate)

            # Handle probability predictions
            if predict_base_rate:
                batch_questions, batch_probs = batch_result
            else:
                batch_questions = batch_result
                batch_probs = [0.5] * len(
                    batch_questions)  # Default probabilities

            # Answer questions for this batch
            batch_answers = model_messaging_wrappers.batch_answer_questions(
                prompt,
                system_prompt,
                large_model_answer,
                state['current_answer'],
                batch_questions,
                model_name=large_model_name,
                temperature=llm_temperature,
                seed=seed,
                verbose=verbose,
                evaluation_mode=evaluation_mode.lower() if evaluation_mode
                in [EVAL_MODE_MATH, EVAL_MODE_SCIENCE] else "default")

            # Add Q&A pairs and probabilities to state
            state['guiding_questions'].extend(batch_questions)
            state['guiding_answers'].extend(batch_answers)
            state['predicted_probs'].extend(batch_probs)
            questions_processed += len(batch_questions)

            if verbose:
                print(
                    f"✅ Batch {batch_num} complete: {len(batch_questions)} Q&A pairs generated"
                )
                print(
                    f"📊 Total Q&A pairs so far: {questions_processed}/{max_iterations}"
                )

                # Show sample Q&A pairs from this batch
                for i, (q, a) in enumerate(
                        zip(batch_questions[:3], batch_answers[:3]), 1):
                    print(f"  Q{i}: {q}")
                    print(f"  A{i}: {'Yes' if a else 'No'}")
                if len(batch_questions) > 3:
                    print(
                        f"  ... and {len(batch_questions) - 3} more Q&A pairs")

            # After each batch, improve the answer with all Q&A pairs collected so far
            if evaluation_mode == EVAL_MODE_MATH:
                state[
                    'current_answer'] = model_messaging_wrappers.small_model_improve_with_guidance_math(
                        prompt,
                        system_prompt=system_prompt,
                        small_model_answer=state['current_answer'],
                        guiding_questions=state['guiding_questions'],
                        guiding_answers=state['guiding_answers'],
                        small_model_name=small_model_name,
                        temperature=slm_temperature,
                        seed=seed,
                        use_local=local_model is not None,
                        local_model=local_model,
                        verbose=verbose)
            elif evaluation_mode == EVAL_MODE_SCIENCE:
                state[
                    'current_answer'] = model_messaging_wrappers.small_model_improve_with_guidance_science(
                        prompt,
                        system_prompt=system_prompt,
                        small_model_answer=state['current_answer'],
                        guiding_questions=state['guiding_questions'],
                        guiding_answers=state['guiding_answers'],
                        small_model_name=small_model_name,
                        temperature=slm_temperature,
                        seed=seed,
                        use_local=local_model is not None,
                        local_model=local_model,
                        verbose=verbose)
            else:
                state[
                    'current_answer'] = model_messaging_wrappers.small_model_improve_with_guidance(
                        prompt,
                        system_prompt,
                        state['current_answer'],
                        guiding_questions=state['guiding_questions'],
                        guiding_answers=state['guiding_answers'],
                        open_ended_guidances=state['guidances']
                        if state['guidances'] else None,
                        small_model_name=small_model_name,
                        temperature=slm_temperature,
                        seed=seed,
                        use_local=local_model is not None,
                        local_model=local_model,
                        verbose=verbose)

            # Evaluate after each batch
            batch_score = evaluate_and_log_quality(
                prompt,
                large_model_answer,
                state['current_answer'],
                llm_temperature,
                seed,
                state['slm_answers'],
                questions_processed,
                f'batch_{batch_num}_updated',
                verbose,
                evaluation_mode,
                gold_answer,
                large_model_name,
                judge_mode=judge_mode)
            state['quality_scores'].append(batch_score)

            if verbose:
                print(f"📈 Quality after batch {batch_num}: {batch_score}/10")
                print(f"📊 Quality progression: {state['quality_scores']}")

            # Check if quality threshold is met
            if batch_score >= quality_threshold:
                if verbose:
                    print(
                        f"✨ Quality threshold met ({batch_score} >= {quality_threshold}), stopping early"
                    )
                break

        state['iteration_count'] = questions_processed

        if verbose:
            print(f"\n📊 Batch processing complete:")
            print(f"  - Total Q&A pairs: {questions_processed}")
            print(f"  - Batches processed: {batch_num}")
            print(
                f"  - Final quality: {state['quality_scores'][-1] if state['quality_scores'] else 0}/10"
            )

    else:
        # Regular iteration mode (unchanged)
        while state['iteration_count'] < max_iterations:
            state['iteration_count'] += 1

            if verbose:
                print(f"\n--- Iteration {state['iteration_count']} ---")

            # Check quality of current answer before proceeding
            # (Skip evaluation on first iteration since we already have initial score)
            if state['iteration_count'] == 1:
                quality_score = state['quality_scores'][
                    -1]  # Use initial score
            else:
                # Only evaluate if answer has changed
                quality_score = evaluate_and_log_quality(
                    prompt,
                    large_model_answer,
                    state['current_answer'],
                    llm_temperature,
                    seed,
                    state['slm_answers'],
                    state['iteration_count'],
                    'updated',
                    verbose,
                    evaluation_mode,
                    gold_answer,
                    large_model_name,
                    judge_mode=judge_mode)
                state['quality_scores'].append(quality_score)

            if verbose:
                print(f"Quality score: {quality_score}/10")
                print(f"Quality progression: {state['quality_scores']}")

            if quality_score >= quality_threshold:
                if verbose:
                    print(f"Answer quality threshold met: {quality_score}")
                break

            # Check if we need to generate LLM answer for Q&A guidance
            if large_model_answer is None and state['iteration_count'] == 1:
                # Generate LLM answer now since we need it for Q&A
                if verbose:
                    print("🔄 Generating LLM answer for Q&A guidance...")
                large_model_answer = model_messaging_wrappers.generate_LLM_response(
                    prompt,
                    system_prompt=system_prompt,
                    model_name=large_model_name,
                    temperature=llm_temperature,
                    seed=seed,
                    verbose=False)

            # Process iteration
            state['current_answer'], state['guiding_questions'], state[
                'guiding_answers'] = process_single_iteration(
                    prompt,
                    system_prompt,
                    large_model_answer,
                    state['current_answer'],
                    question_model_name,
                    small_model_name,
                    local_model,
                    slm_temperature,
                    llm_temperature,
                    seed,
                    state['guiding_questions'],
                    state['guiding_answers'],
                    state['guidances'],
                    evaluation_mode,
                    verbose,
                    predict_base_rate=predict_base_rate,
                    predicted_probs=state['predicted_probs'])

        if verbose:
            print(f"Updated small model answer: {state['current_answer']}")

    # Print summary
    print_iteration_summary(
        state['iteration_count'], state['quality_scores'], state['guidances'],
        state['guiding_questions'],
        state['quality_scores'][-1] if state['quality_scores'] else 0,
        state['open_ended_guidance_used'], verbose)

    # Get best answer and metrics
    best_answer, qa_pairs, metrics = find_best_answer(
        state['slm_answers'], state['guiding_questions'],
        state['guiding_answers'], state['quality_scores'],
        state['open_ended_guidance_used'], prompt, verbose)

    # Add LLM response info for compression ratio calculation
    if large_model_answer is not None:
        metrics['llm_response'] = large_model_answer
        metrics['llm_response_length'] = len(large_model_answer)
        metrics['llm_response_tokens_approx'] = len(
            large_model_answer) // 4  # Rough estimate
    else:
        metrics['llm_response'] = None
        metrics['llm_response_length'] = 0
        metrics['llm_response_tokens_approx'] = 0

    # Add final SLM answer info
    metrics['slm_final_answer'] = best_answer
    metrics['slm_final_answer_length'] = len(best_answer) if best_answer else 0

    # Add Q&A compression info
    qa_text = "\n".join(
        [f"Q: {q}\nA: {a}" for q, a in zip(qa_pairs[0], qa_pairs[1])])
    metrics['qa_text'] = qa_text
    metrics['qa_text_length'] = len(qa_text)

    # Calculate compression ratio (Q&A length vs LLM response length)
    if large_model_answer and len(large_model_answer) > 0:
        metrics['compression_ratio'] = len(qa_text) / len(large_model_answer)

    # Probabilistic compression metrics (if predict_base_rate was enabled)
    guiding_questions = qa_pairs[0]
    guiding_answers = qa_pairs[1]
    predicted_probs = state.get('predicted_probs', [])

    # Default to 1 bit per question (uniform distribution assumption)
    metrics['total_bits_uniform'] = len(
        guiding_questions)  # 1 bit per question
    metrics['bits_per_question_uniform'] = 1.0

    if predict_base_rate and predicted_probs and len(predicted_probs) == len(
            guiding_answers):
        # Calculate actual bits using Shannon information
        import math
        bits_list = []
        for p_yes, answer in zip(predicted_probs, guiding_answers):
            # Clamp probability for numerical stability
            p_yes = max(0.001, min(0.999, p_yes))
            if answer:  # LLM said yes
                p_actual = p_yes
            else:  # LLM said no
                p_actual = 1.0 - p_yes
            bits = -math.log2(p_actual)
            bits_list.append(bits)

        total_bits_prob = sum(bits_list)
        avg_bits_per_q = total_bits_prob / len(bits_list) if bits_list else 1.0

        metrics['predict_base_rate_enabled'] = True
        metrics['predicted_probs'] = predicted_probs
        metrics['bits_per_question'] = bits_list
        metrics['total_bits_probabilistic'] = total_bits_prob
        metrics['avg_bits_per_question'] = avg_bits_per_q
        metrics['bits_saved_vs_uniform'] = metrics[
            'total_bits_uniform'] - total_bits_prob
        metrics['compression_improvement_pct'] = (
            1 - total_bits_prob / metrics['total_bits_uniform']
        ) * 100 if metrics['total_bits_uniform'] > 0 else 0

        if verbose:
            print(f"\n📊 Probabilistic Compression Metrics:")
            print(f"   Questions asked: {len(guiding_questions)}")
            print(
                f"   Uniform encoding: {metrics['total_bits_uniform']:.2f} bits (1 bit/question)"
            )
            print(
                f"   Probabilistic encoding: {total_bits_prob:.2f} bits ({avg_bits_per_q:.3f} bits/question)"
            )
            print(
                f"   Bits saved: {metrics['bits_saved_vs_uniform']:.2f} ({metrics['compression_improvement_pct']:.1f}% improvement)"
            )
    else:
        metrics['predict_base_rate_enabled'] = False
        metrics['total_bits_probabilistic'] = metrics['total_bits_uniform']
        metrics['avg_bits_per_question'] = 1.0

    if large_model_answer is None or len(large_model_answer) == 0:
        metrics['compression_ratio'] = None

    return best_answer, qa_pairs, metrics


# Example usage and testing
if __name__ == "__main__":
    import sys

    # Check for command line arguments
    use_local = "--local" in sys.argv
    use_mps = "--mps" in sys.argv
    use_parallel = "--parallel" in sys.argv
    use_batch = "--batch" in sys.argv

    # Parse batch size if provided
    batch_size = 10  # default
    for i, arg in enumerate(sys.argv):
        if arg == "--batch-size" and i + 1 < len(sys.argv):
            try:
                batch_size = int(sys.argv[i + 1])
            except ValueError:
                print(f"Error: Invalid batch size '{sys.argv[i + 1]}'")
                sys.exit(1)

    if "--help" in sys.argv:
        print("Usage: python SLM_question_answering_compression.py [options]")
        print("Options:")
        print("  --local         Use local model for SLM")
        print("  --mps           Force MPS device (Apple Silicon)")
        print("  --parallel      Enable parallel API calls")
        print(
            "  --batch         Use batch question generation (all questions upfront)"
        )
        print("  --batch-size N  Set batch size (default: 10)")
        print("  --help          Show this help message")
        sys.exit(0)

    test_prompt = "What is the capital of France and why is it significant?"
    test_prompt = "What is a quad tree? And how would you implement it in pseudocode?"
    LLM_TEMPERATURE = 1.0
    SLM_TEMPERATURE = 0.0

    max_iterations = 10
    max_iterations = 20  # Fewer iterations for local testing

    if use_local:
        # Example 2: Using local model with MPS/CUDA/CPU
        print("=== Example: Local Model (Meta-Llama-3-8B) ===")
        print(f"Large model (API): {LLM}")
        print(f"Small model (Local): {LOCAL_SLM}")
        print(f"Question model: {QUESTION_SLM}")
        if use_parallel:
            print("⚡ Parallel execution: ENABLED")

        device = 'mps' if use_mps else None  # Force MPS or auto-detect

        final_answer, qa_pairs, metrics = iterative_SLM_loop(
            test_prompt,
            large_model_name=LLM,
            small_model_name=LOCAL_SLM,
            question_model_name=QUESTION_SLM,
            use_local_slm=
            True,  # IMPORTANT: Tell the function to use local model
            max_iterations=max_iterations,
            quality_threshold=7,
            seed=DEFAULT_SEED,
            llm_temperature=LLM_TEMPERATURE,
            slm_temperature=SLM_TEMPERATURE,
            device=device,  # Pass device parameter
            enable_parallel=use_parallel,  # Enable parallel if flag set
            batch_mode=use_batch,  # Enable batch mode if flag set
            batch_size=batch_size,  # Pass batch size
            verbose=True)
    else:
        # Example 1: Using remote models (API) with verbose output
        print("=== Example: Remote Models (API) ===")
        print(f"Answer model: {SLM}")
        print(f"Question model: {QUESTION_SLM}")
        print(f"LLM model: {LLM}")
        if use_parallel:
            print("⚡ Parallel execution: ENABLED")
        if use_batch:
            print(f"📦 Batch mode: ENABLED (size={batch_size})")
        print("Tip: Use --local to test with local Meta-Llama-3-8B")
        print("     Use --local --mps to force MPS device on Mac")
        print("     Use --parallel to enable parallel API calls")
        print("     Use --batch to enable batch Q&A generation")

        final_answer, qa_pairs, metrics = iterative_SLM_loop(
            test_prompt,
            max_iterations=max_iterations,
            quality_threshold=8,
            seed=DEFAULT_SEED,
            llm_temperature=LLM_TEMPERATURE,
            slm_temperature=SLM_TEMPERATURE,
            enable_parallel=use_parallel,  # Enable parallel if flag set
            batch_mode=use_batch,  # Enable batch mode if flag set
            batch_size=batch_size,  # Pass batch size
            verbose=True)

    print(f"\nFinal answer: {final_answer}")
    print(f"Metrics: {metrics}")
    print(f"Quality scores: {metrics['quality_scores']}")
    print(f"Quality progression: {metrics['quality_progression']}")
