from evalplus.evaluate import check_correctness, get_groundtruth
from evalplus.provider import DecoderBase
from utils.llm_api import anthropic_completion
import re
from typing import List
from lossy_compression.core.qa_compression import iterative_SLM_loop
from lossy_compression import MODEL_ALIAS_MAP
from lossy_compression.utils.formatting import rewrite_answer_to_be_syntactically_correct
import json
import os
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import random


def extract_code(response: str) -> str:
    """Extract code from Claude's response, removing markdown and explanations."""
    if not response or len(response.strip()) == 0:
        raise ValueError(
            "Empty response received from Claude. This may indicate an API error or model issue."
        )

    # Remove markdown code blocks if present
    code_pattern = r'```(?:python)?\s*\n(.*?)\n```'
    matches = re.findall(code_pattern, response, re.DOTALL)

    if matches:
        # Return the first code block found
        extracted_code = matches[0].strip()
        if len(extracted_code) == 0:
            raise ValueError(
                "Empty code block extracted from Claude response.")
        return extracted_code

    # Otherwise, Claude returned clean code, just return it
    clean_code = response.strip()
    if len(clean_code) == 0:
        raise ValueError("Empty code received from Claude response.")
    return clean_code


class ClaudeDecoder(DecoderBase):
    """A DecoderBase implementation that uses Claude API for code generation."""

    def __init__(self,
                 model_name: str = "claude-3-7-sonnet-20250219",
                 temperature: float = 0.0):
        """Initialize the Claude decoder.
        
        Args:
            model_name: The Claude model to use
            temperature: Sampling temperature
        """
        self.model_name = model_name
        self.temperature = temperature
        self.is_direct_completion_flag = True  # Claude returns complete code

    def codegen(self,
                prompt: str,
                do_sample: bool = True,
                num_samples: int = 1) -> List[str]:
        """Generate code using Claude API.
        
        Args:
            prompt: The prompt to send to Claude
            do_sample: Whether to use sampling (ignored, always uses temperature)
            num_samples: Number of samples to generate
            
        Returns:
            List of generated code completions
        """
        # For code generation, we want to use a specific system prompt
        system_prompt = """You are a Python code completion assistant. Complete the given Python function by providing the full implementation including the function signature. Return only valid Python code without any markdown formatting, explanations, or additional text."""

        completions = []
        for _ in range(num_samples):
            try:
                # Call Claude API
                response = anthropic_completion(
                    prompt=prompt,
                    model=self.model_name,
                    temperature=self.temperature,
                    max_tokens=2048,  # Increased for code generation
                    system=system_prompt)

                # Clean up the response - extract just the code
                cleaned_response = extract_code(response)
                completions.append(cleaned_response)  # beep boop baap

            except Exception as e:
                print(f"Error calling Claude API: {e}")
                # Return empty string as fallback
                completions.append("")

        return completions

    def is_direct_completion(self) -> bool:
        """Return True if the model returns direct completions."""
        return self.is_direct_completion_flag


def resolve_claude_model_name(name: str) -> str:
    """Resolve short aliases (haiku, sonnet, opus) to full Claude model IDs.

    Accepts full IDs and returns them unchanged. Case-insensitive for aliases.
    """
    if not name:
        return "claude-3-haiku-20240307"
    lower = name.lower()

    return MODEL_ALIAS_MAP.get(lower, name)


def create_claude_model(model_name: str = "claude-3-7-sonnet-20250219",
                        temperature: float = 0.0) -> DecoderBase:
    """Create a Claude-based DecoderBase model.
    
    Args:
        model_name: The Claude model to use
        temperature: Sampling temperature
        
    Returns:
        A DecoderBase instance that uses Claude API
    """
    print(f"Creating Claude model: {model_name}")
    return ClaudeDecoder(model_name=model_name, temperature=temperature)


def set_up_model_logic(model_name: str = "claude-3-7-sonnet-20250219",
                       temperature: float = 0.0,
                       backend: str = "claude") -> DecoderBase:
    """Set up a model for code generation.
    
    Args:
        model_name: The model to use
        temperature: Sampling temperature
        backend: Backend type (currently only supports "claude")
        
    Returns:
        A DecoderBase instance
    """
    if backend == "claude":
        return create_claude_model(model_name, temperature)
    else:
        raise ValueError(
            f"Unsupported backend: {backend}. Currently only supports 'claude'"
        )


def get_dataset(dataset: str = "humaneval"):
    # Load only what we need
    if dataset == "humaneval":
        from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash
        problems = get_human_eval_plus()
        dataset_hash = get_human_eval_plus_hash()
        tasks_only_output_not_none = [
        ]  # HumanEval doesn't have this special case
    else:  # mbpp
        from evalplus.data import get_mbpp_plus, get_mbpp_plus_hash
        from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
        problems = get_mbpp_plus()
        dataset_hash = get_mbpp_plus_hash()
        tasks_only_output_not_none = MBPP_OUTPUT_NOT_NONE_TASKS

    return problems, dataset_hash, tasks_only_output_not_none


"""
TODO - figure out how to call a single model solver (use codegen to start.)
"""


def get_single_problem(task_id: str, dataset: str = "humaneval"):
    """Get a single problem and its expected outputs.
    
    Args:
        task_id: The task ID to load
        dataset: Dataset name ("humaneval" or "mbpp")
        
    Returns:
        Tuple of (problem, expected_output)
    """
    import os
    from evalplus.data.utils import CACHE_DIR

    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Load the full dataset
    problems, dataset_hash, tasks_only_output_not_none = get_dataset(dataset)

    # Create a safe cache key by replacing forward slashes
    safe_task_id = task_id.replace("/", "_")
    cache_key = f"{dataset_hash}_{safe_task_id}"

    # Get ground truth for just this problem
    expected_output = get_groundtruth(
        {task_id: problems[task_id]},
        cache_key,  # Use safe cache key
        [] if dataset == "humaneval" else tasks_only_output_not_none)

    return problems[task_id], expected_output[task_id]


def evaluate_single_response(task_id: str,
                             solution_response: str,
                             dataset: str = "humaneval",
                             base_only: bool = False):
    """Evaluate a single response for a given task.
    
    Args:
        task_id: The task ID to evaluate (e.g., "HumanEval/1")
        response: The generated code response to evaluate
        dataset: Dataset name ("humaneval" or "mbpp")
        base_only: Whether to only run base tests (skip plus tests)
        
    Returns:
        Dict containing evaluation results
    """
    # Load the problem and expected outputs
    problem, expected_output = get_single_problem(task_id, dataset)

    # Prepare the solution - check if we need to add the prompt
    # If the response already contains the function definition, use as-is
    # Otherwise, combine with prompt
    if "def " in solution_response:
        # Response contains function definition, likely complete
        # But still need to add imports from prompt
        prompt_lines = problem["prompt"].strip().split('\n')
        import_lines = [
            line for line in prompt_lines
            if line.startswith(('import ', 'from '))
        ]
        if import_lines:
            solution = '\n'.join(
                import_lines) + '\n\n' + solution_response.strip()
        else:
            solution = solution_response
    else:
        # Response is just the function body, combine with prompt
        solution = problem["prompt"] + solution_response

    # Sanitize the solution
    # sanitized_solution = sanitize(solution, entrypoint=problem["entry_point"])

    # Check correctness
    result = check_correctness(
        dataset=dataset,
        completion_id=0,  # Single completion
        problem=problem,
        solution=solution,
        expected_output=expected_output,
        base_only=base_only,
        fast_check=False,  # Use thorough checking
        identifier=f"{task_id}_single",
        min_time_limit=1.0,  # Reduced time limit for macOS compatibility
        gt_time_limit_factor=4.0,  # Increased factor for more lenient timeout
    )

    # Extract the relevant information
    base_status, base_details = result["base"]
    base_fail_tests = []

    if base_status != "passed":
        # Get failed test cases
        if base_details:
            base_fail_tests = [
                problem["base_input"][i] for i in range(len(base_details))
                if not base_details[i]
            ]
        else:
            # If no details, just return the last test case
            base_fail_tests = [problem["base_input"][-1]
                               ] if problem["base_input"] else []

    # Initialize plus test results
    plus_status = None
    plus_fail_tests = []

    if not base_only:
        plus_status, plus_details = result["plus"]
        if plus_status != "passed":
            if plus_details:
                plus_fail_tests = [
                    problem["plus_input"][i] for i in range(len(plus_details))
                    if not plus_details[i]
                ]
            else:
                plus_fail_tests = [problem["plus_input"][-1]
                                   ] if problem["plus_input"] else []

    # Format results similar to the main evaluate function
    evaluation_result = {
        "task_id": task_id,
        "solution": solution,
        "base_status": base_status,
        "plus_status": plus_status,
        "base_fail_tests": base_fail_tests,
        "plus_fail_tests": plus_fail_tests,
        "passed": base_status == "passed",
        "passed_plus": plus_status == "passed" if plus_status else None,
    }

    return evaluation_result


def evaluate_single_solution(task_id: str,
                             model_code: str,
                             dataset: str = "humaneval",
                             base_only: bool = False):
    """Evaluate a single solution for one problem."""
    # Get just this problem
    problems, dataset_hash, tasks_only_output_not_none = get_dataset(dataset)
    problem, expected_output = get_single_problem(
        task_id,
        problems=problems,
        dataset_hash=dataset_hash,
        tasks_only_output_not_none=tasks_only_output_not_none,
        dataset=dataset)

    # Build full solution
    if model_code.startswith(problem["prompt"]):
        full_solution = model_code
    else:
        full_solution = problem["prompt"] + model_code

    # Evaluate
    result = check_correctness(dataset=dataset,
                               completion_id=0,
                               problem=problem,
                               solution=full_solution,
                               expected_output=expected_output[task_id],
                               base_only=base_only,
                               fast_check=False)

    # Format results
    base_status, base_details = result["base"]
    plus_status, plus_details = result.get("plus", (None, None))

    return {
        "task_id": task_id,
        "base_passed": base_status == "pass",
        "base_status": base_status,
        "base_tests_passed": sum(base_details) if base_details else 0,
        "base_tests_total": len(problem["base_input"]),
        "plus_passed": plus_status == "pass" if plus_status else None,
        "plus_status": plus_status,
        "plus_tests_passed": sum(plus_details) if plus_details else 0,
        "plus_tests_total": len(problem["plus_input"]) if not base_only else 0,
        "solution": full_solution
    }


def model_answer_single_question(
    model: DecoderBase,
    prompt: str,
    greedy=False,
    n_samples=1,
):
    """Answer a single question and return the response.
    
    Args:
        model: The model to use for generation
        prompt: The question/prompt to answer
        greedy: Whether to use greedy decoding
        n_samples: Number of samples to generate
        
    Returns:
        List of generated responses (strings)
    """
    print(f"Generating {n_samples} sample(s) for prompt: {prompt[:100]}...")

    # Clean up the prompt
    clean_prompt = prompt.strip() + "\n"

    # Generate responses
    outputs = model.codegen(
        clean_prompt,
        do_sample=not greedy,
        num_samples=n_samples,
    )

    assert outputs, "No outputs from model!"

    # Process outputs
    responses = []
    for impl in outputs:
        # For Claude, impl already contains the full function (but without imports)
        # We just need to add it to responses as-is
        responses.append(impl)

    print(f"Generated {len(responses)} response(s)")
    return responses


def evaluate_single_response_simple(task_id: str,
                                    response: str,
                                    dataset: str = "humaneval"):
    """Simplified evaluation that returns just pass/fail status.
    
    Args:
        task_id: The task ID to evaluate
        response: The generated code response
        dataset: Dataset name ("humaneval" or "mbpp")
        
    Returns:
        Dict with simple pass/fail information
    """
    result = evaluate_single_response(task_id,
                                      response,
                                      dataset,
                                      base_only=True)

    # Check if it actually passed
    passed = result["base_status"] == "pass"

    return {
        "task_id": task_id,
        "passed": passed,
        "base_status": result["base_status"],
        "failed_tests": result["base_fail_tests"],
        "solution": result["solution"]
    }


def print_evaluation_summary(result: dict, verbose: bool = True):
    """Print a clear summary of evaluation results.
    
    Args:
        result: The evaluation result dictionary
        verbose: Whether to print the full solution
    """
    print(f"\n{'='*60}")
    print("📊 EVALUATION SUMMARY")
    print(f"{'='*60}")

    print(f"Task: {result['task_id']}")
    print(f"Status: {result['base_status']}")

    if result['passed']:
        print("🎉 SUCCESS: All test cases passed!")
        print("✅✅✅ The generated code is correct!")
    else:
        print("❌❌❌ FAILURE: Some test cases failed")
        if result['failed_tests']:
            print(f"🔍 Failed test inputs: {result['failed_tests']}")

    if verbose:
        print(f"\n{'='*60}")
        print("📝 COMPLETE SOLUTION (COPY THIS):")
        print("=" * 60)
        print(result['solution'])
        print("=" * 60)

    return result['passed']


# Example usage
def example_evaluation():
    """Example of how to use the evaluation functions."""

    # Example task ID and response
    task_id = "HumanEval/1"
    response = """
def has_close_elements(numbers, threshold):
    \"\"\"
    Check if any two numbers in the list are within threshold of each other.
    \"\"\"
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if abs(numbers[i] - numbers[j]) <= threshold:
                return True
    return False
"""

    # Evaluate the response
    result = evaluate_single_response_simple(task_id, response)

    print(f"Task: {result['task_id']}")
    print(f"Passed: {result['passed']}")
    print(f"Status: {result['base_status']}")

    if not result['passed']:
        print(f"Failed tests: {result['failed_tests']}")

    print(f"Solution: {result['solution'][:200]}...")

    return result


# Example usage
def save_results(results_dict: dict,
                 model_name: str,
                 num_questions: int = None,
                 model_config: dict = None,
                 output_name: str = None):
    """Save evaluation results to organized directory structure.
    
    Args:
        results_dict: Dictionary containing results from evaluation
        model_name: Name of the model (e.g., 'QA', 'haiku', 'sonnet')
        num_questions: Number of Q&A iterations (for QA method)
    
    Returns:
        Path to the saved results directory
    """
    # Create timestamp (still used for metadata even if not directory name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Clean model name for directory
    clean_model_name = model_name.replace("/", "_").replace(" ", "_")
    if model_name.upper() == "QA" and num_questions is not None:
        # For QA, just use QA_q{num} - full config is in metadata
        clean_model_name = f"QA_q{num_questions}"

    # Use custom output name if provided, otherwise use timestamp
    dir_name = output_name if output_name else timestamp

    # Create results directory structure
    results_dir = Path("results") / clean_model_name / dir_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save summary statistics with model configuration
    summary = {
        "timestamp": timestamp,
        "date_str": datetime.now().strftime("%Y-%m-%d"),
        "model_name": model_name,
        "num_questions": num_questions if model_name.upper() == "QA" else None,
        "model_config": model_config or {},
        "total_tasks": results_dict["total_tasks"],
        "passed_count": results_dict["passed_count"],
        "failed_count": results_dict["failed_count"],
        "success_rate": results_dict["success_rate"],
        "task_ids": [r["task_id"] for r in results_dict["results"]]
    }

    summary_path = results_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Save detailed results
    detailed_results_path = results_dir / "detailed_results.json"
    with open(detailed_results_path, "w") as f:
        json.dump(results_dict, f, indent=2)

    # Save individual problem results with all details
    problems_dir = results_dir / "problems"
    problems_dir.mkdir(exist_ok=True)

    for result in results_dict["results"]:
        # Create filename from task_id (e.g., HumanEval_1.json)
        task_filename = result["task_id"].replace("/", "_") + ".json"
        task_path = problems_dir / task_filename

        # Enhanced problem data with QA metrics
        problem_data = {
            "task_id": result["task_id"],
            "passed": result["passed"],
            "status": result.get("status", "unknown"),
            "failed_tests": result.get("failed_tests", []),
            "solution": result.get("solution", ""),
            "error": result.get("error", None)
        }

        # Add QA-specific metrics if available
        if "qa_metrics" in result and result["qa_metrics"]:
            qa_metrics = result["qa_metrics"]
            problem_data["qa_analysis"] = {
                "num_qa_iterations":
                qa_metrics.get("num_qa_iterations", 0),
                "bits_of_information":
                qa_metrics.get("bits_of_information", 0),
                "qa_pairs":
                qa_metrics.get("qa_pairs", []),
                "quality_scores":
                qa_metrics.get("metrics", {}).get("quality_scores", []),
                "quality_progression":
                qa_metrics.get("metrics", {}).get("quality_progression", []),
                "final_quality_score":
                qa_metrics.get("metrics", {}).get("final_quality_score", 0),
                "iterations_to_solution":
                qa_metrics.get("metrics", {}).get("iterations", 0)
            }

        with open(task_path, "w") as f:
            json.dump(problem_data, f, indent=2)

    # Save a markdown report for easy reading
    report_path = results_dir / "report.md"
    with open(report_path, "w") as f:
        f.write(f"# HumanEval Results Report\n\n")
        f.write(f"**Model:** {model_name}\n")
        if num_questions is not None:
            f.write(f"**Max Q&A Iterations:** {num_questions}\n")
        f.write(f"**Timestamp:** {timestamp}\n")
        f.write(f"**Success Rate:** {results_dict['success_rate']:.1%} ")
        f.write(
            f"({results_dict['passed_count']}/{results_dict['total_tasks']})\n\n"
        )

        # Add QA statistics if this is a QA run
        qa_results = [
            r for r in results_dict["results"]
            if "qa_metrics" in r and r["qa_metrics"]
        ]
        if qa_results:
            f.write("## QA Method Statistics\n\n")
            total_bits = sum(r["qa_metrics"]["bits_of_information"]
                             for r in qa_results)
            avg_bits = total_bits / len(qa_results) if qa_results else 0
            f.write(f"- **Total bits of information used:** {total_bits}\n")
            f.write(f"- **Average bits per problem:** {avg_bits:.1f}\n")
            f.write(f"- **Problems using QA:** {len(qa_results)}\n\n")

        f.write("## Task Results\n\n")
        if qa_results:
            f.write("| Task ID | Status | Result | QA Iterations | Bits |\n")
            f.write("|---------|--------|--------|--------------|------|\n")
        else:
            f.write("| Task ID | Status | Result |\n")
            f.write("|---------|--------|--------|\n")

        for result in results_dict["results"]:
            status_icon = "✅" if result['passed'] else "❌"
            status_text = result.get('status', 'unknown')
            if "qa_metrics" in result and result["qa_metrics"]:
                qa_iters = result["qa_metrics"]["num_qa_iterations"]
                bits = result["qa_metrics"]["bits_of_information"]
                f.write(
                    f"| {result['task_id']} | {status_text} | {status_icon} | {qa_iters} | {bits} |\n"
                )
            else:
                f.write(
                    f"| {result['task_id']} | {status_text} | {status_icon} |\n"
                )

        f.write("\n## Failed Tasks\n\n")
        failed_tasks = [r for r in results_dict["results"] if not r['passed']]
        if failed_tasks:
            for task in failed_tasks:
                f.write(f"### {task['task_id']}\n")
                if 'error' in task:
                    f.write(f"Error: {task['error']}\n")
                if 'failed_tests' in task and task['failed_tests']:
                    f.write(f"Failed tests: {task['failed_tests']}\n")
                f.write("\n")
        else:
            f.write("No failed tasks! 🎉\n")

    print(f"\n💾 Results saved to: {results_dir}")
    return results_dir


def example_claude_codegen_multiple(
        task_numbers: List[int],
        model_name: str = "claude-3-7-sonnet-20250219",
        verbose: bool = True,
        prompt_answerer=None,
        parallel: bool = False,
        max_workers: int = 4,
        rate_limit_delay: float = 1.0,
        status_interval: int = 10) -> dict:
    """Run Claude code generation on multiple HumanEval problems.
    
    Args:
        task_numbers: List of HumanEval task numbers to test (e.g., [1, 2, 5, 10])
        model_name: Claude model to use
        verbose: Whether to print detailed output for each task
        
    Returns:
        Dict with overall statistics and per-task results
    """
    # Normalize model name from short aliases if needed
    resolved_model_name = resolve_claude_model_name(model_name)

    print(
        f"🚀 Testing {len(task_numbers)} HumanEval problems with {resolved_model_name}"
    )
    print(f"{'='*60}")

    if prompt_answerer is None:
        # Fallback to a default Claude-backed answerer
        model = create_claude_model(model_name=resolved_model_name,
                                    temperature=0.0)

        def default_prompt_answerer(prompt):
            return model_answer_single_question(
                model=model,
                prompt=prompt,
                greedy=True,
                n_samples=1,
            )[0]

        prompt_answerer = default_prompt_answerer

    results = []
    passed_count = 0

    # Check if parallel execution is requested
    if parallel:
        print(f"\n🚀 Running in PARALLEL mode with {max_workers} workers")
        print(f"   Rate limit delay: {rate_limit_delay}s")
        print(f"{'='*60}\n")

        # Prepare task arguments for parallel execution
        task_args = [(f"HumanEval/{task_num}", task_num, len(task_numbers),
                      prompt_answerer, verbose, rate_limit_delay, i)
                     for i, task_num in enumerate(task_numbers)]

        # Execute tasks in parallel
        start_time = datetime.now()
        last_status_time = time.time()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(process_single_task, args): args[0]
                for args in task_args
            }

            # Track running tasks
            running_tasks = set(future_to_task.keys())
            completed_tasks = []

            # Process completed tasks
            completed = 0

            # Create a separate thread for periodic status updates
            def print_status():
                while running_tasks:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    minutes = int(elapsed // 60)
                    seconds = int(elapsed % 60)

                    # Count tasks in different states
                    n_completed = len(completed_tasks)
                    n_running = len(running_tasks) - n_completed
                    n_passed = sum(1 for r in results
                                   if r.get('passed', False))

                    print(
                        f"\n⏰ Status Update - Elapsed: {minutes:02d}:{seconds:02d}"
                    )
                    print(
                        f"   Completed: {n_completed}/{len(task_numbers)} | Running: {n_running} | Passed: {n_passed}"
                    )
                    print(f"   {'-'*40}")

                    time.sleep(status_interval)

            # Start status thread
            import threading as thread_module
            status_thread = thread_module.Thread(target=print_status,
                                                 daemon=True)
            status_thread.start()

            for future in as_completed(future_to_task):
                task_id = future_to_task[future]
                completed += 1

                try:
                    result = future.result()
                    results.append(result)
                    completed_tasks.append(task_id)

                    if result['passed']:
                        passed_count += 1

                    # Progress indicator with elapsed time
                    elapsed = (datetime.now() - start_time).total_seconds()
                    status = "✅" if result['passed'] else "❌"
                    print(
                        f"[{completed}/{len(task_numbers)}] {task_id}: {status} (elapsed: {elapsed:.1f}s)"
                    )

                except Exception as e:
                    print(f"❌ Task {task_id} failed with exception: {e}")
                    results.append({
                        "task_id": task_id,
                        "passed": False,
                        "status": "exception",
                        "error": str(e)
                    })
                    completed_tasks.append(task_id)

                # Remove from running set
                running_tasks.discard(future)

        # Sort results by task_id to maintain order
        results.sort(key=lambda x: int(x['task_id'].split('/')[-1]))

        # Print final timing summary
        total_elapsed = (datetime.now() - start_time).total_seconds()
        minutes = int(total_elapsed // 60)
        seconds = int(total_elapsed % 60)
        print(f"\n{'='*60}")
        print(f"🏁 Parallel Execution Complete!")
        print(
            f"   Total time: {minutes:02d}:{seconds:02d} ({total_elapsed:.1f}s)"
        )
        print(f"   Tasks/minute: {len(task_numbers) / (total_elapsed/60):.1f}")
        print(f"   Average per task: {total_elapsed/len(task_numbers):.1f}s")
        print(f"{'='*60}")

    else:
        # Sequential execution (original code)
        for i, task_number in enumerate(task_numbers):
            task_id = f"HumanEval/{task_number}"

            if verbose:
                print(f"\n📝 Task {i+1}/{len(task_numbers)}: {task_id}")
                print(f"{'='*40}")

            try:
                # Get the problem
                problem, expected_output = get_single_problem(
                    task_id, "humaneval")
                prompt = problem["prompt"]

                if verbose:
                    print(f"Problem:\n{prompt}\n------")

                # Generate code using injected prompt_answerer
                task_start_time = datetime.now()
                response = prompt_answerer(prompt)
                task_elapsed_time = (datetime.now() -
                                     task_start_time).total_seconds()

                # Handle both QA method (returns dict) and regular methods (returns string)
                qa_metrics = None
                if isinstance(response, dict):
                    # QA method returns a dict with answer, qa_pairs, metrics
                    generated = response['answer']
                    qa_metrics = {
                        'qa_pairs':
                        response.get('qa_pairs', []),
                        'num_qa_iterations':
                        response.get('num_qa_iterations', 0),
                        'bits_of_information':
                        response.get('bits_of_information', 0),
                        'metrics':
                        response.get('metrics', {})
                    }
                    if verbose and qa_metrics['num_qa_iterations'] > 0:
                        print(
                            f"📊 QA iterations: {qa_metrics['num_qa_iterations']}"
                        )
                        print(
                            f"📊 Bits of information: {qa_metrics['bits_of_information']}"
                        )
                        # Log quality progression if available
                        if 'metrics' in qa_metrics and 'quality_progression' in qa_metrics[
                                'metrics']:
                            quality_progression = qa_metrics['metrics'][
                                'quality_progression']
                            print(
                                f"📊 Quality progression: {quality_progression}"
                            )
                        print(f"⏱️  Task time: {task_elapsed_time:.1f}s")
                else:
                    # Regular methods return just the string answer
                    generated = response
                    if verbose:
                        print(f"⏱️  Task time: {task_elapsed_time:.1f}s")

                # Print the solution explicitly if verbose
                if verbose and generated:
                    print("\n" + "=" * 60)
                    print("📝 GENERATED SOLUTION (COPY THIS):")
                    print("=" * 60)
                    print(generated)
                    print("=" * 60 + "\n")

                if not generated:
                    if verbose:
                        print("❌ No response generated")
                    results.append({
                        "task_id": task_id,
                        "passed": False,
                        "status": "no_response",
                        "error": "No response generated",
                        "qa_metrics": qa_metrics
                    })
                    continue

                # Evaluate the response
                result = evaluate_single_response_simple(task_id, generated)

                if result['passed']:
                    passed_count += 1
                    if verbose:
                        print("✅ PASSED")
                        # Also print the complete working solution for easy copying
                        print("\n" + "=" * 60)
                        print("✨ COMPLETE WORKING SOLUTION (COPY THIS):")
                        print("=" * 60)
                        print(result['solution'])
                        print("=" * 60)
                else:
                    if verbose:
                        print("❌ FAILED")
                        # Still print the solution for debugging
                        print("\n" + "=" * 60)
                        print("⚠️ FAILED SOLUTION (FOR DEBUGGING):")
                        print("=" * 60)
                        print(result['solution'])
                        print("=" * 60)

                # Store result
                task_result = {
                    "task_id": task_id,
                    "passed": result['passed'],
                    "status": result['base_status'],
                    "failed_tests": result['failed_tests'],
                    "solution": result['solution'],
                    "qa_metrics": qa_metrics  # Add QA metrics if available
                }
                results.append(task_result)

            except Exception as e:
                if verbose:
                    print(f"❌ ERROR: {e}")
                results.append({
                    "task_id": task_id,
                    "passed": False,
                    "status": "error",
                    "error": str(e)
                })

    # Calculate overall statistics
    total_tasks = len(task_numbers)
    success_rate = passed_count / total_tasks if total_tasks > 0 else 0

    # Print summary
    print(f"\n{'='*60}")
    print("📊 FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Model: {resolved_model_name}")
    print(f"Tasks tested: {total_tasks}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {total_tasks - passed_count}")
    print(f"Success rate: {success_rate:.1%}")

    # Show which tasks passed/failed
    print("\n📋 Task-by-task results:")
    for result in results:
        status_icon = "✅" if result['passed'] else "❌"
        print(f"  {status_icon} {result['task_id']}: {result['status']}")

    return {
        "model_name": resolved_model_name,
        "total_tasks": total_tasks,
        "passed_count": passed_count,
        "failed_count": total_tasks - passed_count,
        "success_rate": success_rate,
        "results": results
    }


def example_claude_codegen(task_number: int = 1,
                           model_name: str = "claude-3-7-sonnet-20250219"):
    """Example of how to use the Claude model for code generation."""

    # Create a Claude model
    model = create_claude_model(model_name=model_name, temperature=0.0)

    # Get an actual HumanEval problem
    task_id = f"HumanEval/{task_number}"  # Use the first problem
    problem, expected_output = get_single_problem(task_id, "humaneval")

    # Use the actual prompt from the dataset
    prompt = problem["prompt"]

    print(f"Using task: {task_id}")
    print(f"Prompt: {prompt[:200]}...")

    # Generate code
    responses = model_answer_single_question(model=model,
                                             prompt=prompt,
                                             greedy=True,
                                             n_samples=1)

    print("\n" + "=" * 60)
    print("📝 GENERATED SOLUTION (COPY THIS):")
    print("=" * 60)
    if responses:
        for i, response in enumerate(responses):
            if len(responses) > 1:
                print(f"\n--- Response {i+1} ---")
            print(response)
    print("=" * 60)
    print(f"\nGenerated {len(responses)} response(s)")

    # Evaluate the response
    if responses:
        prompted_response = responses[0]
        print(f"\n{'='*50}")
        print("EVALUATION RESULTS")
        print(f"{'='*50}")

        result = evaluate_single_response_simple(task_id, prompted_response)

        # Use the new summary function for clearer output
        success = print_evaluation_summary(result, verbose=True)

        # Return whether it passed or failed
        return success


def parse_cli_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run HumanEval with Claude models")
    parser.add_argument(
        "--model",
        type=str,
        default="claude-3-haiku-20240307",
        help=("Claude model to use. Accepts full ID or alias: "
              "haiku | sonnet | opus (default: claude-3-haiku-20240307)"))
    parser.add_argument("--num-tasks",
                        type=int,
                        default=5,
                        help="Number of tasks to evaluate (default: 5)")
    parser.add_argument(
        "--task-ids",
        type=int,
        nargs="+",
        help="Specific task IDs to run (overrides --num-tasks)")
    parser.add_argument(
        "-q",
        "--num-questions",
        type=int,
        default=25,
        help="Number of Q&A iterations for QA method (default: 25)")
    parser.add_argument("--llm-model",
                        type=str,
                        default=None,
                        help="LLM model for QA method (default: opus)")
    parser.add_argument("--slm-model",
                        type=str,
                        default=None,
                        help="SLM model for QA method (default: haiku)")
    parser.add_argument(
        "--question-model",
        type=str,
        default=None,
        help="Question generation model for QA method (default: opus)")
    parser.add_argument("--verbose",
                        action="store_true",
                        default=True,
                        help="Print detailed output (default: True)")
    parser.add_argument("--save-results",
                        action="store_true",
                        default=True,
                        help="Save results to disk (default: True)")
    parser.add_argument("--no-save",
                        action="store_true",
                        help="Skip saving results to disk")
    parser.add_argument("--parallel",
                        action="store_true",
                        help="Enable parallel execution of tasks")
    parser.add_argument("--max-workers",
                        type=int,
                        default=4,
                        help="Maximum number of parallel workers (default: 4)")
    parser.add_argument(
        "--rate-limit-delay",
        type=float,
        default=1.0,
        help="Base delay in seconds for rate limit backoff (default: 1.0)")
    parser.add_argument(
        "--status-interval",
        type=int,
        default=10,
        help="Seconds between status updates in parallel mode (default: 10)")
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Custom name for output directory (instead of timestamp)")
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Enable batch Q&A generation (generate all questions at once)")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help=
        "Number of questions to generate at once in batch mode (default: 10)")

    return parser.parse_args()


CACHE = {}

# Global rate limiting state for parallel execution
rate_limit_lock = threading.Lock()
rate_limit_backoff = {}  # task_id -> (retry_count, last_attempt_time)


def exponential_backoff(retry_count, base_delay=1.0):
    """Calculate exponential backoff with jitter."""
    delay = min(base_delay * (2**retry_count), 60)  # Cap at 60 seconds
    jitter = random.uniform(0, delay * 0.1)  # Add 10% jitter
    return delay + jitter


def process_single_task(args):
    """Process a single HumanEval task. Used for parallel execution."""
    (task_id, task_number, total_tasks, prompt_answerer, verbose,
     rate_limit_delay, task_index) = args

    max_retries = 5
    retry_count = 0

    while retry_count < max_retries:
        try:
            # Get the problem
            problem, expected_output = get_single_problem(task_id, "humaneval")
            prompt = problem["prompt"]

            if verbose:
                print(
                    f"\n📝 Task {task_index+1}/{total_tasks}: {task_id} [Thread-{threading.current_thread().name}]"
                )
                print(f"{'='*40}")
                print(f"Problem:\n{prompt}\n------")

            # Generate code using injected prompt_answerer
            task_start_time = datetime.now()
            response = prompt_answerer(prompt)
            task_elapsed_time = (datetime.now() -
                                 task_start_time).total_seconds()

            # Handle both QA method (returns dict) and regular methods (returns string)
            qa_metrics = None
            if isinstance(response, dict):
                # QA method returns a dict with answer, qa_pairs, metrics
                generated = response['answer']
                qa_metrics = {
                    'qa_pairs': response.get('qa_pairs', []),
                    'num_qa_iterations': response.get('num_qa_iterations', 0),
                    'bits_of_information':
                    response.get('bits_of_information', 0),
                    'metrics': response.get('metrics', {})
                }
                if verbose and qa_metrics['num_qa_iterations'] > 0:
                    print(
                        f"📊 QA iterations: {qa_metrics['num_qa_iterations']}")
                    print(
                        f"📊 Bits of information: {qa_metrics['bits_of_information']}"
                    )
                    if 'metrics' in qa_metrics and 'quality_progression' in qa_metrics[
                            'metrics']:
                        quality_progression = qa_metrics['metrics'][
                            'quality_progression']
                        print(f"📊 Quality progression: {quality_progression}")
                    print(f"⏱️  Task time: {task_elapsed_time:.1f}s")
            else:
                # Regular methods return just the string answer
                generated = response
                if verbose:
                    print(f"⏱️  Task time: {task_elapsed_time:.1f}s")

            if not generated:
                return {
                    "task_id": task_id,
                    "passed": False,
                    "status": "no_response",
                    "error": "No response generated",
                    "qa_metrics": qa_metrics
                }

            # Evaluate the response
            result = evaluate_single_response_simple(task_id, generated)

            if verbose:
                if result['passed']:
                    print(f"✅ Task {task_id} passed!")
                else:
                    print(f"❌ Task {task_id} failed")
                    if result['failed_tests']:
                        print(f"Failed tests: {result['failed_tests'][:3]}...")

            # Prepare task result
            task_result = {
                "task_id": task_id,
                "passed": result['passed'],
                "status": result['base_status'],
                "failed_tests": result['failed_tests'],
                "solution": result['solution'],
                "qa_metrics": qa_metrics  # Add QA metrics if available
            }

            return task_result

        except Exception as e:
            error_msg = str(e)
            if "RATE_LIMIT_ERROR" in error_msg:
                retry_count += 1
                if retry_count >= max_retries:
                    print(
                        f"❌ Task {task_id}: Max retries ({max_retries}) exceeded due to rate limits"
                    )
                    return {
                        "task_id": task_id,
                        "passed": False,
                        "status": "rate_limit_error",
                        "error":
                        f"Rate limit exceeded after {max_retries} retries"
                    }

                # Calculate backoff delay
                delay = exponential_backoff(retry_count, rate_limit_delay)
                print(
                    f"⚠️  Task {task_id}: Rate limit hit, retry {retry_count}/{max_retries} after {delay:.1f}s"
                )
                time.sleep(delay)
                continue
            else:
                # Non-rate-limit error
                print(f"❌ Task {task_id} error: {e}")
                return {
                    "task_id": task_id,
                    "passed": False,
                    "status": "error",
                    "error": str(e)
                }

    return {
        "task_id": task_id,
        "passed": False,
        "status": "max_retries",
        "error": "Maximum retries exceeded"
    }


def cache_prompt_answerer(prompt: str) -> str:
    if prompt in CACHE:
        return CACHE[prompt]
    # ...generate somehow...
    generated = """
def is_palindrome(string: str) -> bool:
    return string == string[::-1]
    
def make_palindrome(string: str) -> str:
    if not string:
        return ''
    
    for i in range(len(string), -1, -1): # this answer is wrong, because of the -1, -1
        if is_palindrome(string[i:]):
            return string + string[:i][::-1]
    
    return string[::-1] + string
                    """

    opus_generated = """
def is_palindrome(string: str) -> bool:
    return string == string[::-1]


def make_palindrome(string: str) -> str:
    if not string:
        return ''

    # Start from the beginning and find the longest suffix that is a palindrome
    for i in range(len(string)):
        # Check if the suffix starting from position i is a palindrome
        if is_palindrome(string[i:]):
            # Append the reverse of the prefix (before position i) to the original string
            return string + string[:i][::-1]

    # This line should never be reached since the last character is always a palindrome
    return string    
    """
    qa_generated = """
def is_palindrome(string: str) -> bool:
    return string == string[::-1]


def make_palindrome(string: str) -> str:
    if not string:
        return ''

    for i in range(len(string)):
        if is_palindrome(string[i:]):
            return string + string[:i][::-1]

    return string[::-1] + string
    """

    return qa_generated


def filter_prompt_from_answer(answer: str, prompt) -> str:
    # ask Claude to filter the prompt from the answer
    filter_prompt = f"""Sometimes Claude will return the prompt in the answer. Please filter it out.

Answer: {answer}

Prompt: {prompt}"""

    filtered_answer = anthropic_completion(prompt=filter_prompt,
                                           model=MODEL_ALIAS_MAP["sonnet"],
                                           temperature=0.0,
                                           max_tokens=2048)
    return filtered_answer


def add_tabs_to_answer(answer: str) -> str:
    ## deterministically add 4 spaces to the start of each line
    return "\n".join([f"    {line}" for line in answer.split("\n")])


def check_if_answer_has_tabs(answer: str) -> bool:
    # check if the answer has 4 spaces or a tab at the start of each line
    return all(
        line.startswith("    ") or line.startswith("\t")
        for line in answer.split("\n"))


def prompt_answerer_SLM_qa_method(
        prompt,
        llm_model: str = MODEL_ALIAS_MAP["opus"],
        slm_model: str = MODEL_ALIAS_MAP["haiku"],
        question_model: str = MODEL_ALIAS_MAP["opus"],
        num_questions: int = 25,
        use_guidance: bool = False,
        batch_mode: bool = False,
        batch_size: int = 10,
        verbose: bool = True,
        seed: int = None) -> dict:
    """Run HumanEval problems using LLM-SLM compression.
    
    Returns:
        Dictionary with:
        - 'answer': The generated code answer
        - 'qa_pairs': List of (question, answer) tuples
        - 'metrics': Dictionary of metrics including iterations, quality scores, etc.
    """

    system_prompt = """You are a Python code completion assistant. Complete the given Python function by providing the full implementation including the function signature. Return only valid Python code without any markdown formatting, explanations, or additional text."""
    ##SLM_prompt = f"""Complete this Python function. Return ONLY the complete function implementation excluding the signature, (starting exactly after the prompt) no explanations or markdown. Do not include any quotes or backticks. Please make sure the function is correctly tabbed and formatted with 4 spaces. :  \n{prompt}"""

    final_answer, qa_pairs, metrics = iterative_SLM_loop(
        prompt=prompt,
        system_prompt=system_prompt,
        large_model_name=llm_model,
        small_model_name=slm_model,
        question_model_name=question_model,
        use_local_slm=False,
        max_iterations=num_questions,  # Use the passed-in num_questions
        quality_threshold=10,  # Stop if quality reaches this
        open_ended_guidance=False,
        enable_parallel=False,
        use_code_evaluation=True,  # Use code evaluation for judging
        batch_mode=batch_mode,  # Enable batch mode if requested
        batch_size=batch_size,  # Number of questions to generate at once
        seed=seed,  # Pass seed for reproducibility
        verbose=verbose)
    #filtered_answer = filter_prompt_from_answer(final_answer, SLM_prompt)
    final_answer = rewrite_answer_to_be_syntactically_correct(
        final_answer, prompt)
    final_answer = extract_code(final_answer)
    #if not check_if_answer_has_tabs(final_answer):
    #    final_answer = add_tabs_to_answer(final_answer)

    return {
        'answer': final_answer,
        'qa_pairs': qa_pairs,
        'metrics': metrics,
        'num_qa_iterations': len(qa_pairs) if qa_pairs else 0,
        'bits_of_information':
        len(qa_pairs) if qa_pairs else 0  # Each yes/no question = 1 bit
    }


def main():
    """Main function to run HumanEval evaluation with command-line arguments."""
    args = parse_cli_args()

    # Determine which tasks to run
    if args.task_ids:
        task_numbers = args.task_ids
    else:
        # Default tasks based on num_tasks
        if args.num_tasks <= 5:
            task_numbers = [1, 2, 5, 10, 15][:args.num_tasks]
        else:
            task_numbers = list(range(1, args.num_tasks + 1))
    print(f"Using model: `{args.model}`")
    model_config = {}
    if args.model.lower() == "qa":
        # Extract model configuration for QA method (use args or defaults)
        default_llm = MODEL_ALIAS_MAP.get("opus", "claude-opus-4-1-20250805")
        default_slm = MODEL_ALIAS_MAP.get("haiku", "claude-3-haiku-20240307")
        default_question = MODEL_ALIAS_MAP.get("opus",
                                               "claude-opus-4-1-20250805")

        # Resolve model names from arguments or use defaults
        if args.llm_model:
            llm_model = MODEL_ALIAS_MAP.get(args.llm_model.lower(),
                                            args.llm_model)
        else:
            llm_model = default_llm

        if args.slm_model:
            slm_model = MODEL_ALIAS_MAP.get(args.slm_model.lower(),
                                            args.slm_model)
        else:
            slm_model = default_slm

        if args.question_model:
            question_model = MODEL_ALIAS_MAP.get(args.question_model.lower(),
                                                 args.question_model)
        else:
            question_model = default_question

        model_config = {
            "llm_model": llm_model,
            "slm_model": slm_model,
            "question_model": question_model,
            "max_iterations": args.num_questions
        }

        print(f"Using QA method with {args.num_questions} iterations")
        print(f"  LLM: {llm_model}")
        print(f"  SLM: {slm_model}")
        print(f"  Question Model: {question_model}")

        # Create a lambda wrapper to pass num_questions and batch parameters
        prompt_answerer = lambda prompt: prompt_answerer_SLM_qa_method(
            prompt,
            num_questions=args.num_questions,
            llm_model=llm_model,
            slm_model=slm_model,
            question_model=question_model,
            batch_mode=args.batch,
            batch_size=args.batch_size)
        resolved_model = "QA"
    elif args.model.lower() == "cache":
        print(f"Using cached method")
        prompt_answerer = cache_prompt_answerer
        resolved_model = "Cached"
    else:
        resolved_model = resolve_claude_model_name(args.model)

        print(
            f"Testing {len(task_numbers)} HumanEval problems with {resolved_model}..."
        )
        # Define a default Claude-backed prompt_answerer; callers can swap this out
        model = create_claude_model(model_name=resolved_model, temperature=0.0)

        def prompt_answerer(prompt):
            return model_answer_single_question(
                model=model,
                prompt=prompt,
                greedy=True,
                n_samples=1,
            )[0]

    results = example_claude_codegen_multiple(
        task_numbers,
        resolved_model,
        verbose=args.verbose,
        prompt_answerer=prompt_answerer,
        parallel=args.parallel,
        max_workers=args.max_workers,
        rate_limit_delay=args.rate_limit_delay,
        status_interval=args.status_interval)

    print(
        f"\n🎯 Final Score: {results['success_rate']:.1%} ({results['passed_count']}/{results['total_tasks']})"
    )

    # Save results to organized directory structure (unless --no-save is specified)
    if not args.no_save:
        num_questions = args.num_questions if args.model.lower(
        ) == "qa" else None
        save_results(results, resolved_model, num_questions, model_config,
                     args.output_name)

    return results


if __name__ == "__main__":
    main()
"""
Other things I can try to improve if the respones are poorly formatted:

1. use prefill
2. use examples 

"""
