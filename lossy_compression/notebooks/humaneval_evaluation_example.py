# Setup paths and imports
import sys
import os

# Import the evaluation functions
from lossy_compression.run_human_eval import (get_single_problem,
                                              evaluate_single_response_simple,
                                              create_claude_model,
                                              model_answer_single_question)

# Get a specific HumanEval problem
task_id = "HumanEval/1"  # You can change this to any valid task ID

# Extract the problem and expected output
problem, expected_output = get_single_problem(task_id)

# Get the prompt
prompt = problem["prompt"]

print(f"Task ID: {task_id}")
print(f"Entry Point: {problem['entry_point']}")
print("\nPrompt:")
print("=" * 60)
print(prompt)
print("=" * 60)

# Example: Manual solution for HumanEval/1
manual_solution = '''
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False
'''

# Evaluate the solution
result = evaluate_single_response_simple(task_id, manual_solution)

print(f"✅ Passed: {result['passed']}")
print(f"Status: {result['base_status']}")
if not result['passed']:
    print(f"Failed tests: {result['failed_tests']}")
