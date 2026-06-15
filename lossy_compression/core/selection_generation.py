#!/usr/bin/env python3
"""
SLM Selection Generation: 
Generate multiple candidate answers from SLM and have LLM select the best one iteratively.
"""

from typing import List, Tuple, Dict, Any
from lossy_compression.utils import model_wrappers as model_messaging_wrappers
from lossy_compression import LLM, SLM, DEFAULT_LLM_TEMPERATURE, DEFAULT_SLM_TEMPERATURE, model_completion
from utils.llm_api import anthropic_continue, get_anthropic_key

SEED = 42


def generate_multiple_slm_continuations(original_prompt: str,
                                        partial_answer: str,
                                        num_candidates: int = 16,
                                        slm_model: str = SLM,
                                        temperature: float = 0.7,
                                        max_tokens: int = 100,
                                        seed: int = SEED,
                                        verbose: bool = False,
                                        top_p: float = 0.9,
                                        presence_penalty: float = 0.0,
                                        frequency_penalty: float = 0.0,
                                        stop: List[str] = None) -> List[str]:
    """Generate multiple candidate continuations from SLM."""

    stop = stop or [
        "\n\nQuestion:", "\n\nOriginal question:",
        "\n\nCurrent partial answer:"
    ]

    # Get API key once
    api_key = get_anthropic_key()

    candidates = []
    for i in range(num_candidates):
        if partial_answer:
            # Use prefill_assistant for true continuation
            # Remove trailing whitespace to avoid API error, but remember if we had a space
            had_trailing_space = partial_answer and partial_answer[-1] == ' '
            cleaned_partial = partial_answer.rstrip()

            response = anthropic_continue(
                prompt=
                f"Question: {original_prompt}\n\nAnswer the question directly:",
                model=slm_model,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=(seed + i) if seed is not None else None,
                stop_sequences=stop,
                prefill_assistant=cleaned_partial,  # Remove trailing whitespace
                api_key=api_key).strip()

            # If we had a trailing space, add it back to the beginning of the response
            if had_trailing_space and response and not response[0].isspace():
                response = ' ' + response
        else:
            # For initial generation, use regular mode
            response = anthropic_continue(
                prompt=f"Question: {original_prompt}\n\nAnswer:",
                model=slm_model,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=(seed + i) if seed is not None else None,
                stop_sequences=stop,
                api_key=api_key).strip()
        candidates.append(response)

    if verbose:
        print(
            f"   Generated {len(candidates)} candidates ({max_tokens} tokens each)"
        )
        for i, candidate in enumerate(candidates, 1):
            print(f"     {i}: {candidate}")

    return candidates


def llm_select_best_continuation(
        original_prompt: str,
        partial_answer: str,
        candidates: List[str],
        llm_model: str = LLM,
        temperature: float = 0.0,
        seed: int = SEED,
        fallback_tokens: int = 10,
        verbose: bool = False) -> Tuple[int, str, bool]:
    """
    Have LLM select the best continuation from the list.
    
    Returns:
        Tuple of (selected_idx, selected_text, is_llm_generated)
        If is_llm_generated is True, the LLM provided its own text instead of selecting.
    """

    # Format candidates for selection
    candidates_text = "\n\n".join([
        f"Option {i+1}:\n{candidate}" for i, candidate in enumerate(candidates)
    ])

    selection_prompt = f"""Original question: {original_prompt}

Current partial answer: {partial_answer}

Here are {len(candidates)} possible continuations:

{candidates_text}

Which option best continues the answer? 
- Reply with ONLY the number (1-{len(candidates)}) if one of the options is acceptable.
- Reply with "NONE" if all options are poor, then provide exactly {fallback_tokens} tokens to CONTINUE from where the partial answer left off.
IMPORTANT: Do NOT restart or restate the problem. Continue exactly from where the partial answer ends."""

    response = model_completion(
        selection_prompt,
        model=llm_model,
        temperature=temperature,
        seed=seed,
        max_tokens=fallback_tokens +
        30  # Allow space for "NONE" + the tokens (20 tokens ~80 chars)
    )

    # Check if LLM rejected all options
    if "none" in response.lower()[:10]:  # Check first 10 chars for "NONE"
        if verbose:
            print(f"   ⚠️ LLM fallback ({fallback_tokens} tokens)")
        # Extract the LLM's continuation after "NONE"
        llm_continuation = response[response.lower().find("none") + 4:].strip()
        # Limit to approximately fallback_tokens (rough estimate: 4 chars per token)
        char_limit = fallback_tokens * 4
        llm_continuation = llm_continuation[:char_limit]
        return -1, llm_continuation, True

    # Parse the selection
    import re
    match = re.search(r'\b(\d+)\b', response)
    if match:
        selected_idx = int(match.group(1)) - 1
        if 0 <= selected_idx < len(candidates):
            if verbose:
                print(f"   ✅ Selected option {selected_idx + 1}")
            return selected_idx, candidates[selected_idx], False

    # Fallback to first candidate if parsing fails
    if verbose:
        print(f"⚠️  Could not parse selection, defaulting to first candidate")
    return 0, candidates[0], False


def llm_check_answer_sufficient(original_prompt: str,
                                current_answer: str,
                                llm_model: str = LLM,
                                temperature: float = 0.0,
                                seed: int = SEED,
                                verbose: bool = False) -> bool:
    """Check if LLM thinks the answer is sufficient."""

    check_prompt = f"""Original question: {original_prompt}

Current answer: {current_answer}

Is this answer sufficient and complete? Answer YES if done, NO if it needs more content.
Reply with only YES or NO."""

    response = model_completion(
        check_prompt,
        model=llm_model,
        temperature=temperature,
        seed=seed,
        max_tokens=10,
    )

    is_sufficient = "yes" in response.lower()

    if verbose:
        print(f"   {'✅ Sufficient!' if is_sufficient else '↻ Continuing...'}")

    return is_sufficient


def iterative_selection_generation(
        original_prompt: str,
        llm_model: str = LLM,
        slm_model: str = SLM,
        num_candidates: int = 16,
        max_iterations:
    int = 100,  # High limit, but will stop when LLM says sufficient
        max_tokens_per_iteration: int = 30,
        slm_temperature: float = 0.7,
        fallback_tokens: int = 20,
        llm_temperature: float = 0.0,
        seed: int = SEED,
        verbose: bool = True) -> Tuple[str, Dict[str, Any]]:
    """
    Main function: Iteratively generate candidates and select best until complete.
    
    Returns:
        Tuple of (final_answer, metrics)
    """

    # Store reference answers for later evaluation
    slm_baseline = None
    llm_reference = None

    if verbose:
        print(f"\n{'='*60}")
        print(f"📝 PROMPT: {original_prompt}")
        print(f"{'='*60}")

        # Generate baseline SLM answer for comparison
        print(f"\n📊 Generating baseline SLM answer (no iteration)...")
        slm_baseline = model_completion(
            prompt=f"Question: {original_prompt}\n\nAnswer:",
            model=slm_model,
            temperature=slm_temperature,
            seed=seed).strip()
        print(f"✅ SLM Baseline Answer ({len(slm_baseline)} chars):")
        print(f"   {slm_baseline}")

        # Generate LLM reference answer for comparison
        print(f"\n📊 Generating LLM reference answer for comparison...")
        llm_reference = model_completion(
            prompt=f"Question: {original_prompt}\n\nAnswer:",
            model=llm_model,
            temperature=0.0,
            seed=seed).strip()
        print(f"✅ LLM Reference Answer ({len(llm_reference)} chars):")
        print(f"   {llm_reference}")

        print(f"\n{'='*60}")
        print(f"📈 INITIAL EVALUATIONS")
        print(f"{'='*60}")

        # 1. Evaluate LLM standalone
        print(f"\n1️⃣ LLM Standalone Evaluation:")
        llm_standalone_score = model_messaging_wrappers.evaluate_answer_standalone(
            prompt=original_prompt,
            answer=llm_reference,
            temperature=0.0,
            seed=seed,
            verbose=False)
        print(f"   LLM Answer Quality: {llm_standalone_score}/10")

        # 2. Evaluate SLM baseline standalone
        print(f"\n2️⃣ SLM Baseline Standalone Evaluation:")
        slm_baseline_standalone = model_messaging_wrappers.evaluate_answer_standalone(
            prompt=original_prompt,
            answer=slm_baseline,
            temperature=0.0,
            seed=seed,
            verbose=False)
        print(f"   SLM Baseline Quality: {slm_baseline_standalone}/10")

        # 4. Compare SLM baseline against LLM
        print(f"\n4️⃣ SLM Baseline vs LLM Comparison:")
        slm_vs_llm_score = model_messaging_wrappers.evaluate_answer_quality_comparison(
            original_prompt=original_prompt,
            large_model_answer=llm_reference,
            small_model_answer=slm_baseline,
            temperature=0.0,
            seed=seed,
            verbose=False)
        print(f"   SLM vs LLM: {slm_vs_llm_score}/10")

        # Early stopping: if SLM baseline already matches LLM quality
        if slm_baseline_standalone >= llm_standalone_score:
            print(f"\n{'='*60}")
            print(
                f"✅ EARLY STOPPING: SLM baseline already matches/exceeds LLM quality!"
            )
            print(f"   SLM Score: {slm_baseline_standalone}/10")
            print(f"   LLM Score: {llm_standalone_score}/10")
            print(f"   No iteration needed - returning SLM baseline answer")
            print(f"{'='*60}")

            # Return the SLM baseline with minimal metrics
            metrics = {
                "iterations": 0,
                "total_candidates_generated": 0,
                "selection_history": [],
                "final_length": len(slm_baseline),
                "completed": True,
                "llm_fallback_count": 0,
                "llm_fallback_rate": 0,
                "llm_generated_chars": 0,
                "slm_selected_chars": len(slm_baseline),
                "llm_generated_tokens": 0,
                "slm_selected_tokens": len(slm_baseline) // 4,
                "early_stopped": True,
                "early_stop_reason": "SLM baseline matches LLM quality"
            }
            return slm_baseline, metrics

        print(f"{'='*60}")

        print(
            f"\n🚀 Starting iterative selection ({num_candidates} candidates/iter, {max_tokens_per_iteration} tokens each)"
        )
        print(f"{'='*60}\n")

    current_answer = ""
    iteration = 0
    selection_history = []

    while iteration < max_iterations:
        iteration += 1

        if verbose:
            print(
                f"\n⭕ Iteration {iteration}: {len(current_answer)} chars accumulated"
            )

            # Print stats every 10 iterations
            if iteration % 10 == 0:
                llm_fallback_count = sum(1 for s in selection_history
                                         if s.get('llm_generated', False))
                print(f"\n📊 Stats at iteration {iteration}:")
                print(f"   Total chars generated: {len(current_answer)}")
                print(
                    f"   Avg chars per iteration: {len(current_answer) / iteration:.1f}"
                )
                print(
                    f"   LLM fallbacks so far: {llm_fallback_count}/{iteration} ({llm_fallback_count/iteration*100:.1f}%)"
                )
                print(
                    f"   Total candidates generated: {iteration * num_candidates}"
                )
                print()

        # Generate candidates
        candidates = generate_multiple_slm_continuations(
            original_prompt=original_prompt,
            partial_answer=current_answer,
            num_candidates=num_candidates,
            slm_model=slm_model,
            temperature=slm_temperature,
            max_tokens=max_tokens_per_iteration,
            seed=seed + iteration * 100,  # Vary seed per iteration
            verbose=verbose)

        # Select best candidate
        selected_idx, selected_text, is_llm_generated = llm_select_best_continuation(
            original_prompt=original_prompt,
            partial_answer=current_answer,
            candidates=candidates,
            fallback_tokens=fallback_tokens,
            llm_model=llm_model,
            temperature=llm_temperature,
            seed=seed,
            verbose=verbose)

        # Append selected text to answer
        current_answer += selected_text

        if verbose:
            print(f"   --- Current answer ---")
            print(f"{current_answer}")
            print(f"   ---------------------")
        selection_history.append({
            "iteration": iteration,
            "selected_idx": selected_idx,
            "num_candidates": len(candidates),
            "llm_generated": is_llm_generated,
            "text_length": len(selected_text)
        })

        if verbose and is_llm_generated:
            print(f"     LLM text: {selected_text}")

        # Check if sufficient
        if llm_check_answer_sufficient(original_prompt=original_prompt,
                                       current_answer=current_answer,
                                       llm_model=llm_model,
                                       temperature=llm_temperature,
                                       seed=seed,
                                       verbose=verbose):
            if verbose:
                print(
                    f"\n✅ Complete after {iteration} iterations (answer sufficient)"
                )
            break

    # Compile metrics
    llm_fallbacks = sum(1 for s in selection_history
                        if s.get('llm_generated', False))
    llm_generated_chars = sum(
        s.get('text_length', 0) for s in selection_history
        if s.get('llm_generated', False))
    slm_selected_chars = sum(
        s.get('text_length', 0) for s in selection_history
        if not s.get('llm_generated', False))

    # Rough token estimation (1 token ≈ 4 chars)
    llm_generated_tokens = llm_generated_chars // 4
    slm_selected_tokens = slm_selected_chars // 4

    completed_naturally = iteration < max_iterations  # True if LLM said sufficient
    metrics = {
        "iterations": iteration,
        "total_candidates_generated": iteration * num_candidates,
        "selection_history": selection_history,
        "final_length": len(current_answer),
        "completed": completed_naturally,
        "llm_fallback_count": llm_fallbacks,
        "llm_fallback_rate": llm_fallbacks / iteration if iteration > 0 else 0,
        "llm_generated_chars": llm_generated_chars,
        "slm_selected_chars": slm_selected_chars,
        "llm_generated_tokens": llm_generated_tokens,
        "slm_selected_tokens": slm_selected_tokens
    }

    if verbose:
        # Check if we early stopped
        if metrics.get('early_stopped', False):
            print(f"\n📊 Final Statistics:")
            print(
                f"   Status: ✅ Early stopped - SLM baseline matched LLM quality"
            )
            print(f"   Iterations: 0 (no iteration needed)")
            print(f"   Final answer length: {len(current_answer)} chars")
        else:
            print(f"\n📊 Final Statistics:")
            print(f"   Iterations: {iteration}")
            print(
                f"   Total candidates generated: {metrics['total_candidates_generated']}"
            )
            print(
                f"   LLM fallbacks: {llm_fallbacks}/{iteration} ({metrics['llm_fallback_rate']:.1%})"
            )
            print(f"\n📝 Text Generation Breakdown:")
            print(
                f"   LLM-generated: {llm_generated_chars} chars (~{llm_generated_tokens} tokens)"
            )
            print(
                f"   SLM-selected:  {slm_selected_chars} chars (~{slm_selected_tokens} tokens)"
            )
            if llm_generated_chars + slm_selected_chars > 0:
                print(
                    f"   Ratio: {llm_generated_chars/(llm_generated_chars+slm_selected_chars)*100:.1f}% from LLM"
                )
            print(f"\n   Final answer length: {len(current_answer)} chars")
            print(
                f"   Status: {'✅ Completed (LLM satisfied)' if completed_naturally else '⚠️ Hit iteration limit'}"
            )

        # Final comprehensive evaluations
        print(f"\n{'='*60}")
        print(f"📊 FINAL EVALUATIONS")
        print(f"{'='*60}")

        # If early stopped, we already have the evaluations
        if metrics.get('early_stopped', False):
            print(
                f"\n✅ Using SLM baseline as final answer (no iteration performed)"
            )
            slm_guided_standalone = slm_baseline_standalone  # Use baseline score
        else:
            # 3. Evaluate SLM-guided result standalone
            print(f"\n3️⃣ SLM-Guided (Iterative) Standalone Evaluation:")
            slm_guided_standalone = model_messaging_wrappers.evaluate_answer_standalone(
                prompt=original_prompt,
                answer=current_answer,
                temperature=0.0,
                seed=seed,
                verbose=False)
            print(f"   SLM-Guided Quality: {slm_guided_standalone}/10")

        # 5. Compare SLM-guided against LLM
        if metrics.get('early_stopped', False):
            # If early stopped, use the baseline comparison score
            slm_guided_vs_llm = slm_vs_llm_score
            print(f"\n5️⃣ SLM-Guided vs LLM Comparison:")
            print(
                f"   SLM-Guided vs LLM: {slm_guided_vs_llm}/10 (using baseline)"
            )
        else:
            print(f"\n5️⃣ SLM-Guided vs LLM Comparison:")
            if llm_reference:  # Only if we have LLM reference from verbose mode
                slm_guided_vs_llm = model_messaging_wrappers.evaluate_answer_quality_comparison(
                    original_prompt=original_prompt,
                    large_model_answer=llm_reference,
                    small_model_answer=current_answer,
                    temperature=0.0,
                    seed=seed,
                    verbose=False)
                print(f"   SLM-Guided vs LLM: {slm_guided_vs_llm}/10")
            else:
                # Generate LLM reference if not in verbose mode
                llm_reference = model_completion(
                    prompt=f"Question: {original_prompt}\n\nAnswer:",
                    model=llm_model,
                    temperature=0.0,
                    seed=seed).strip()
                slm_guided_vs_llm = model_messaging_wrappers.evaluate_answer_quality_comparison(
                    original_prompt=original_prompt,
                    large_model_answer=llm_reference,
                    small_model_answer=current_answer,
                    temperature=0.0,
                    seed=seed,
                    verbose=False)
                print(f"   SLM-Guided vs LLM: {slm_guided_vs_llm}/10")

        # Summary comparison
        print(f"\n{'='*60}")
        print(f"📈 SUMMARY OF ALL EVALUATIONS")
        print(f"{'='*60}")

        if slm_baseline:  # If we have all the data from verbose mode
            print(
                f"\n📊 Standalone Quality Scores (How well each answers the question):"
            )
            print(f"   • LLM:        {llm_standalone_score}/10")
            print(f"   • SLM Base:   {slm_baseline_standalone}/10")
            print(f"   • SLM-Guided: {slm_guided_standalone}/10")

            print(f"\n📊 Comparison to LLM (How close to LLM quality):")
            print(f"   • SLM Base vs LLM:   {slm_vs_llm_score}/10")
            print(f"   • SLM-Guided vs LLM: {slm_guided_vs_llm}/10")

            # Only show improvement if we actually did iteration
            if not metrics.get('early_stopped', False):
                print(f"\n📊 Improvement from Guidance:")
                standalone_improvement = slm_guided_standalone - slm_baseline_standalone
                comparative_improvement = slm_guided_vs_llm - slm_vs_llm_score
                print(
                    f"   • Standalone: {'+' if standalone_improvement >= 0 else ''}{standalone_improvement:.1f} points"
                )
                print(
                    f"   • vs LLM:     {'+' if comparative_improvement >= 0 else ''}{comparative_improvement:.1f} points"
                )
            else:
                print(
                    f"\n📊 No iteration performed - SLM baseline already optimal"
                )
        else:
            print(f"\n📊 Final Scores:")
            print(f"   • SLM-Guided Standalone: {slm_guided_standalone}/10")
            print(f"   • SLM-Guided vs LLM:     {slm_guided_vs_llm}/10")

        print(f"{'='*60}")

    return current_answer, metrics


def self_evaluate_answer(original_prompt: str,
                         generated_answer: str,
                         llm_model: str = LLM,
                         verbose: bool = False) -> Dict[str, Any]:
    """Have LLM self-evaluate the generated answer."""

    # Get LLM's reference answer
    reference_prompt = f"Please answer this question concisely and accurately:\n{original_prompt}"
    reference_answer = model_completion(reference_prompt,
                                        model=llm_model,
                                        temperature=0.0,
                                        seed=SEED,
                                        max_tokens=500)

    # Evaluate quality using the existing function

    quality_score = model_messaging_wrappers.evaluate_answer_quality_comparison(
        original_prompt=original_prompt,
        large_model_answer=reference_answer,
        small_model_answer=generated_answer,
        model_name=llm_model,  # large model evaluates response
        temperature=0.0,
        seed=SEED,
        verbose=False)

    if verbose:
        print(f"\n📊 Self-Evaluation:")
        print(f"   Quality Score: {quality_score}/10")
        print(f"   Reference answer length: {len(reference_answer)} chars")
        print(f"   Generated answer length: {len(generated_answer)} chars")

    return {
        "quality_score": quality_score,
        "reference_answer": reference_answer,
        "reference_length": len(reference_answer),
        "generated_length": len(generated_answer)
    }


def run_experiment_questions(question_ids: List[int] = None,
                             llm_model: str = LLM,
                             slm_model: str = SLM,
                             num_candidates: int = 4,
                             max_iterations: int = 100,
                             max_tokens_per_iteration: int = 30,
                             slm_temperature: float = 0.7,
                             fallback_tokens: int = 20,
                             llm_temperature: float = 0.0,
                             seed: int = SEED,
                             verbose: bool = True) -> Dict[str, Any]:
    """Run selection generation on experiment questions and self-evaluate."""

    import json
    import time

    # Load questions
    with open("experiment_questions.json", "r") as f:
        data = json.load(f)
        questions = data["questions"]

    # Filter by IDs if specified
    if question_ids:
        questions = [q for q in questions if q["id"] in question_ids]
    else:
        questions = questions[:5]  # Default to first 5

    print(f"🚀 Running Selection Generation on Experiment Questions")
    print(f"   Questions: {len(questions)}")
    print(f"   LLM: {llm_model}")
    print(f"   SLM: {slm_model}")
    print(f"   Candidates per iteration: {num_candidates}")
    print("=" * 60)

    results = []
    total_start = time.time()

    for i, q in enumerate(questions, 1):
        print(f"\n📝 Question {i}/{len(questions)} (ID: {q['id']})")
        print(f"   Category: {q['category']}")
        print(f"   Question: {q['question']}...")

        start_time = time.time()

        #try:
        # Generate answer using selection
        answer, metrics = iterative_selection_generation(
            original_prompt=q["question"],
            llm_model=llm_model,
            slm_model=slm_model,
            num_candidates=num_candidates,
            max_iterations=max_iterations,
            max_tokens_per_iteration=max_tokens_per_iteration,
            slm_temperature=slm_temperature,
            fallback_tokens=fallback_tokens,
            llm_temperature=llm_temperature,
            seed=seed,
            verbose=verbose)

        # Self-evaluate
        eval_results = self_evaluate_answer(original_prompt=q["question"],
                                            generated_answer=answer,
                                            llm_model=llm_model,
                                            verbose=verbose)

        elapsed = time.time() - start_time

        result = {
            "question_id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "generated_answer": answer,
            "metrics": metrics,
            "evaluation": eval_results,
            "elapsed_time": elapsed,
            "success": True
        }

        print(f"   ✅ Completed in {elapsed:.2f}s")
        print(f"   📊 Quality: {eval_results['quality_score']}/10")
        print(f"   🔄 Iterations: {metrics['iterations']}")
        print(f"   ⚠️  LLM fallbacks: {metrics['llm_fallback_count']}")

        #except Exception as e:
        if False:
            print(f"   ❌ Error: {e}")
            result = {
                "question_id": q["id"],
                "category": q["category"],
                "question": q["question"],
                "error": str(e),
                "success": False
            }

        results.append(result)

    total_elapsed = time.time() - total_start

    # Calculate summary statistics
    successful = [r for r in results if r.get("success", False)]

    if successful:
        avg_quality = sum(r["evaluation"]["quality_score"]
                          for r in successful) / len(successful)
        avg_iterations = sum(r["metrics"]["iterations"]
                             for r in successful) / len(successful)
        avg_fallbacks = sum(r["metrics"]["llm_fallback_count"]
                            for r in successful) / len(successful)
        total_fallbacks = sum(r["metrics"]["llm_fallback_count"]
                              for r in successful)
        total_iterations = sum(r["metrics"]["iterations"] for r in successful)
        fallback_rate = total_fallbacks / total_iterations if total_iterations > 0 else 0
    else:
        avg_quality = avg_iterations = avg_fallbacks = fallback_rate = 0

    # Print final summary
    print("\n" + "=" * 60)
    print("📊 EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"Questions processed: {len(questions)}")
    print(f"Successful: {len(successful)}")
    if successful:
        print(f"\n📈 Performance Metrics:")
        print(f"   Average Quality Score: {avg_quality:.1f}/10")
        print(f"   Average Iterations: {avg_iterations:.1f}")
        print(f"   Average LLM Fallbacks: {avg_fallbacks:.1f}")
        print(f"   Overall Fallback Rate: {fallback_rate:.1%}")
        print(f"   Total Time: {total_elapsed:.2f}s")

        # Show quality distribution
        quality_scores = [r["evaluation"]["quality_score"] for r in successful]
        print(f"\n📊 Quality Distribution:")
        print(
            f"   Excellent (9-10): {sum(1 for s in quality_scores if s >= 9)}")
        print(f"   Good (7-8): {sum(1 for s in quality_scores if 7 <= s < 9)}")
        print(f"   Fair (5-6): {sum(1 for s in quality_scores if 5 <= s < 7)}")
        print(f"   Poor (1-4): {sum(1 for s in quality_scores if s < 5)}")

    return {
        "results": results,
        "summary": {
            "avg_quality": avg_quality,
            "avg_iterations": avg_iterations,
            "fallback_rate": fallback_rate,
            "total_time": total_elapsed
        }
    }


def main():
    """Main function with experiment questions support."""
    import argparse

    parser = argparse.ArgumentParser(description="SLM Selection Generation")
    parser.add_argument("--experiment",
                        action="store_true",
                        help="Run on experiment questions")
    parser.add_argument("--question-ids",
                        type=int,
                        nargs="+",
                        help="Specific question IDs for experiment")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain how neural networks work",
        help="Single question to answer (if not using --experiment)")
    parser.add_argument("--llm", type=str, default=LLM, help="LLM model")
    parser.add_argument("--slm", type=str, default=SLM, help="SLM model")
    parser.add_argument(
        "--candidates",
        type=int,
        default=16,
        help="Number of candidates per iteration (default: 16)")
    parser.add_argument(
        "--max-iter",
        type=int,
        default=100,
        help="Maximum iterations (default: 100, stops when sufficient)")
    parser.add_argument("--max-tokens",
                        type=int,
                        default=30,
                        help="Max tokens per iteration (default: 30)")
    parser.add_argument("--slm-temp",
                        type=float,
                        default=0.7,
                        help="SLM temperature (default: 0.7)")
    parser.add_argument(
        "--fallback-tokens",
        type=int,
        default=20,
        help="Fallback tokens when LLM rejects all candidates (default: 20)")
    parser.add_argument("--llm-temp",
                        type=float,
                        default=0.0,
                        help="LLM temperature (default: 0.0)")
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--verbose",
                        action="store_true",
                        default=True,
                        help="Verbose output")

    args = parser.parse_args()

    if args.experiment:
        # Run on experiment questions
        results = run_experiment_questions(
            question_ids=args.question_ids,
            llm_model=args.llm,
            slm_model=args.slm,
            num_candidates=args.candidates,
            max_iterations=args.max_iter,
            max_tokens_per_iteration=args.max_tokens,
            slm_temperature=args.slm_temp,
            fallback_tokens=args.fallback_tokens,
            llm_temperature=args.llm_temp,
            seed=args.seed,
            verbose=args.verbose)
        return results
    else:
        # Run on single prompt
        answer, metrics = iterative_selection_generation(
            args.prompt,
            llm_model=args.llm,
            slm_model=args.slm,
            num_candidates=args.candidates,
            max_iterations=args.max_iter,
            max_tokens_per_iteration=args.max_tokens,
            slm_temperature=args.slm_temp,
            fallback_tokens=args.fallback_tokens,
            llm_temperature=args.llm_temp,
            seed=args.seed,
            verbose=args.verbose)

        print(f"\n📄 Final Answer:")
        print("=" * 60)
        print(answer)
        print("=" * 60)

        # Self-evaluate if requested
        eval_results = self_evaluate_answer(original_prompt=args.prompt,
                                            generated_answer=answer,
                                            llm_model=args.llm,
                                            verbose=True)

        return answer, metrics


if __name__ == "__main__":
    main()
