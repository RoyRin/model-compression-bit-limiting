#!/usr/bin/env python3
"""AIME problem solver using different Claude models."""

import json
import time
import argparse
from datasets import load_dataset
from lossy_compression import model_completion, LLM, SLM, QUESTION_SLM, MODEL_ALIAS_MAP
from lossy_compression.core.qa_compression import iterative_SLM_loop, EVAL_MODE_MATH
from lossy_compression.utils.model_wrappers import MATH_SOLVER_SYSTEM
import re
from datetime import datetime
import os


def extract_numerical_answer(response_text):
    """Extract numerical answer from model response.

    Looks for patterns like:
    - "The answer is X"
    - "Answer: X"
    - Numbers in boxes like \\boxed{X}
    - Final numerical value at the end
    """
    # Look for boxed answers first (common in math solutions)
    boxed_pattern = r'\\boxed\{([^}]+)\}'
    boxed_match = re.search(boxed_pattern, response_text)
    if boxed_match:
        return boxed_match.group(1).strip()

    # Look for explicit answer patterns
    answer_patterns = [
        r'(?:the\s+)?answer\s+is:?\s*(\d+)',
        r'(?:final\s+)?answer:?\s*(\d+)',
        r'therefore:?\s*(\d+)',
        r'=\s*(\d+)\s*(?:$|\n)',  # Equals sign followed by number at end
    ]

    for pattern in answer_patterns:
        match = re.search(pattern, response_text.lower())
        if match:
            return match.group(1)

    # Look for the last standalone number in the text
    numbers = re.findall(r'\b(\d+)\b', response_text)
    if numbers:
        return numbers[-1]

    return None


def solve_with_model(problem_text, model, verbose=False):
    """Solve AIME problem using Claude Haiku."""

    system_prompt = """You are a skilled mathematician solving AIME (American Invitational Mathematics Examination) problems.
Provide a clear, step-by-step solution and end with the numerical answer.
Format your final answer clearly, preferably using \\boxed{answer} notation."""

    prompt = f"""Solve this AIME problem:

{problem_text}

Please provide a complete solution with clear mathematical reasoning, and state the final numerical answer."""

    if verbose:
        print(f"\n{'='*50}")
        print(f"Solving with {model}...")
        print(f"{'='*50}")

    start_time = time.time()
    response = model_completion(
        prompt,
        model=model,
        system=system_prompt,
        temperature=0.1,  # Low temperature for math problems
        max_tokens=2000)
    solve_time = time.time() - start_time

    if verbose:
        print(f"Response received in {solve_time:.2f}s")
        print(f"Response length: {len(response)} characters")

    return {
        "model": model,
        "response": response,
        "extracted_answer": extract_numerical_answer(response),
        "solve_time": solve_time
    }


def solve_with_haiku(problem_text, verbose=False):
    """Solve AIME problem using Claude Haiku."""
    return solve_with_model(problem_text,
                            "claude-3-haiku-20240307",
                            verbose=verbose)


def solve_with_sonnet(problem_text, verbose=False):
    """Solve AIME problem using Claude Sonnet."""
    return solve_with_model(problem_text,
                            "claude-3-7-sonnet-20250219",
                            verbose=verbose)


def solve_with_opus(problem_text, verbose=False):
    """Solve AIME problem using Claude Opus."""
    return solve_with_model(problem_text,
                            "claude-opus-4-1-20250805",
                            verbose=verbose)


def solve_with_qa_compression(problem_text,
                              llm_model=None,
                              slm_model=None,
                              question_model=None,
                              num_questions=25,
                              batch_mode=False,
                              batch_size=10,
                              gold_answer=None,
                              verbose=False):
    """Solve AIME problem using Q&A compression method with math-specific logic.

    Args:
        problem_text: AIME problem text
        llm_model: Large model for generating reference answer
        slm_model: Small model for answering questions
        question_model: Model for generating questions
        num_questions: Number of Q&A iterations
        gold_answer: Correct answer (optional, for evaluation)
        verbose: Print detailed output

    Returns:
        Dict with solution, Q&A pairs, and metrics
    """
    # Use defaults if not specified, and resolve aliases to full model names
    if llm_model:
        llm_model = MODEL_ALIAS_MAP.get(llm_model.lower(), llm_model)
    else:
        llm_model = LLM

    if slm_model:
        slm_model = MODEL_ALIAS_MAP.get(slm_model.lower(), slm_model)
    else:
        slm_model = SLM

    if question_model:
        question_model = MODEL_ALIAS_MAP.get(question_model.lower(),
                                             question_model)
    else:
        question_model = QUESTION_SLM

    # Use the math-specific system prompt from model_messaging_wrappers
    system_prompt = MATH_SOLVER_SYSTEM

    # Simple problem statement (the system prompt handles formatting rules)
    prompt = problem_text

    if verbose:
        print(f"\n{'='*50}")
        print(f"Solving with Q&A Compression (MATH MODE)")
        print(f"LLM: {llm_model}")
        print(f"SLM: {slm_model}")
        print(f"Question Model: {question_model}")
        print(f"Max questions: {num_questions}")
        print(f"{'='*50}")

    start_time = time.time()

    # Run the iterative SLM loop with math evaluation mode
    final_answer, qa_pairs, metrics = iterative_SLM_loop(
        prompt=prompt,
        system_prompt=system_prompt,
        large_model_name=llm_model,
        small_model_name=slm_model,
        question_model_name=question_model,
        use_local_slm=False,
        max_iterations=num_questions,
        quality_threshold=9,  # High threshold for math (9-10 means correct)
        open_ended_guidance=False,
        enable_parallel=False,
        evaluation_mode=EVAL_MODE_MATH,  # Use math evaluation mode
        gold_answer=gold_answer,  # Pass gold answer if available
        skip_llm_initial=
        True,  # Skip initial LLM generation to save costs (default)
        batch_mode=batch_mode,  # Enable batch mode if requested
        batch_size=batch_size,  # Number of questions to generate at once
        verbose=verbose)

    solve_time = time.time() - start_time

    if verbose:
        print(f"\nQ&A Compression completed in {solve_time:.2f}s")
        print(f"Iterations: {metrics.get('iterations', 0)}")
        print(f"Final quality: {metrics.get('final_quality', 0)}")
        print(f"Q&A pairs generated: {len(qa_pairs)}")

    return {
        "method": "qa_compression",
        "llm_model": llm_model,
        "slm_model": slm_model,
        "question_model": question_model,
        "response": final_answer,
        "extracted_answer": extract_numerical_answer(final_answer),
        "solve_time": solve_time,
        "qa_pairs": qa_pairs,
        "metrics": metrics
    }


def solve_problem(problem_data, models=None, verbose=False):
    """Solve a single AIME problem with specified models.

    Args:
        problem_data: Dictionary with 'problem', 'answer', etc.
        models: List of model names to use (default: all)
        verbose: Whether to print detailed output

    Returns:
        Dictionary with results from each model
    """
    if models is None:
        models = ['haiku', 'sonnet', 'opus']

    problem_text = problem_data['problem']
    correct_answer = problem_data['answer']

    results = {
        'problem_id': problem_data.get('id', 'unknown'),
        'problem': problem_text,
        'correct_answer': correct_answer,
        'model_solutions': {}
    }

    # Map model names to solver functions
    solvers = {
        'haiku': solve_with_haiku,
        'sonnet': solve_with_sonnet,
        'opus': solve_with_opus
    }

    for model_name in models:
        if model_name in solvers:
            if verbose:
                print(f"\n{'='*60}")
                print(
                    f"Solving problem {results['problem_id']} with {model_name.upper()}"
                )
                print(f"{'='*60}")

            solution = solvers[model_name](problem_text, verbose=verbose)
            results['model_solutions'][model_name] = solution

            # Check if answer is correct
            is_correct = solution['extracted_answer'] == correct_answer
            solution['is_correct'] = is_correct

            if verbose:
                print(f"\nExtracted answer: {solution['extracted_answer']}")
                print(f"Correct answer: {correct_answer}")
                print(
                    f"Result: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")

    return results


def main():
    """Main function to solve AIME problems."""
    parser = argparse.ArgumentParser(
        description='Solve AIME problems with Claude models')
    parser.add_argument('--problems',
                        type=int,
                        nargs='+',
                        help='Problem indices to solve (0-indexed)')
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        choices=['haiku', 'sonnet', 'opus', 'qa'],
        default=['haiku', 'sonnet', 'opus'],
        help='Models to use for solving (including "qa" for Q&A compression)')
    parser.add_argument('--verbose',
                        action='store_true',
                        help='Print detailed output')
    parser.add_argument('--output',
                        type=str,
                        help='Output file for results (JSON format)')
    parser.add_argument('--all',
                        action='store_true',
                        help='Solve all problems in dataset')
    parser.add_argument('--no-save',
                        action='store_true',
                        help='Do not save results to file')

    # Q&A Compression specific arguments
    parser.add_argument(
        '-q',
        '--num-questions',
        type=int,
        default=25,
        help='Number of Q&A iterations for QA method (default: 25)')
    parser.add_argument('--llm-model',
                        type=str,
                        default=None,
                        help='LLM model for QA method (default: opus)')
    parser.add_argument('--slm-model',
                        type=str,
                        default=None,
                        help='SLM model for QA method (default: haiku)')
    parser.add_argument(
        '--question-model',
        type=str,
        default=None,
        help='Question generation model for QA method (default: opus)')
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Enable batch Q&A generation (generate all questions at once)')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help=
        'Number of questions to generate at once in batch mode (default: 10)')

    args = parser.parse_args()

    # Set up default output path if not specified and not disabled
    if not args.output and not args.no_save:
        # Create results directory if it doesn't exist
        os.makedirs('lossy_compression/results', exist_ok=True)
        # Generate timestamp-based filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        models_str = '_'.join(args.models) if args.models else 'all_models'
        args.output = f'lossy_compression/results/aime_{models_str}_{timestamp}.json'

    # Load AIME dataset
    print("Loading AIME dataset...")
    ds = load_dataset("AI-MO/aimo-validation-aime")
    dataset = ds['train']
    print(f"Loaded {len(dataset)} problems")

    # Determine which problems to solve
    if args.all:
        problem_indices = list(range(len(dataset)))
    elif args.problems:
        problem_indices = args.problems
    else:
        # Default: solve first problem
        problem_indices = [0]

    all_results = []

    # Solve each problem
    for idx in problem_indices:
        if idx >= len(dataset):
            print(
                f"Warning: Problem index {idx} out of range (max: {len(dataset)-1})"
            )
            continue

        problem_data = dataset[idx]
        print(f"\n{'='*70}")
        print(f"Problem {idx} (ID: {problem_data['id']})")
        print(f"{'='*70}")
        print(f"{problem_data['problem'][:200]}...")
        print(f"Correct answer: {problem_data['answer']}")

        # Check if we need to use Q&A compression
        if 'qa' in args.models:
            qa_results = solve_with_qa_compression(
                problem_data['problem'],
                llm_model=args.llm_model,
                slm_model=args.slm_model,
                question_model=args.question_model,
                num_questions=args.num_questions,
                batch_mode=args.batch,
                batch_size=args.batch_size,
                gold_answer=problem_data[
                    'answer'],  # Pass gold answer for evaluation
                verbose=args.verbose)

            # Format results to match expected structure
            results = {
                'problem_id': problem_data.get('id', 'unknown'),
                'problem': problem_data['problem'],
                'correct_answer': problem_data['answer'],
                'model_solutions': {
                    'qa': qa_results
                }
            }

            # Check if answer is correct
            qa_results['is_correct'] = qa_results[
                'extracted_answer'] == problem_data['answer']

            # Also run other models if specified
            other_models = [m for m in args.models if m != 'qa']
            if other_models:
                other_results = solve_problem(problem_data,
                                              models=other_models,
                                              verbose=args.verbose)
                results['model_solutions'].update(
                    other_results['model_solutions'])
        else:
            results = solve_problem(problem_data,
                                    models=args.models,
                                    verbose=args.verbose)

        all_results.append(results)

        # Print summary
        print(f"\n{'='*40}")
        print(f"Summary for Problem {idx}:")
        print(f"{'='*40}")
        for model_name, solution in results['model_solutions'].items():
            status = "✓" if solution['is_correct'] else "✗"
            print(
                f"{model_name.upper():8} {status} Answer: {solution['extracted_answer']:10} "
                f"(Time: {solution['solve_time']:.2f}s)")

    # Save results if output file specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")

    # Overall statistics
    if len(all_results) > 1:
        print(f"\n{'='*50}")
        print("Overall Statistics:")
        print(f"{'='*50}")

        model_stats = {}
        for result in all_results:
            for model_name, solution in result['model_solutions'].items():
                if model_name not in model_stats:
                    model_stats[model_name] = {
                        'correct': 0,
                        'total': 0,
                        'time': 0
                    }
                model_stats[model_name]['total'] += 1
                model_stats[model_name]['time'] += solution['solve_time']
                if solution['is_correct']:
                    model_stats[model_name]['correct'] += 1

        for model_name, stats in model_stats.items():
            accuracy = stats['correct'] / stats['total'] * 100
            avg_time = stats['time'] / stats['total']
            print(f"{model_name.upper():8} Accuracy: {accuracy:5.1f}% "
                  f"({stats['correct']}/{stats['total']}) "
                  f"Avg time: {avg_time:.2f}s")


if __name__ == "__main__":
    main()
