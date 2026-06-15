#!/usr/bin/env python3
"""
Per-Problem Code Evaluation Script

This script provides a modular approach to evaluate code generation models
on a per-problem basis, with separate functions for:
1. Getting a problem
2. Calling a model to solve the problem  
3. Evaluating that problem

This allows for more granular control and easier debugging compared to
the batch evaluation approach.
"""

import json
import os
import time
import torch
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import evaluation functions from the existing harness
from evalplus.data import (
    get_human_eval_plus,
    get_human_eval_plus_hash,
    get_mbpp_plus,
    get_mbpp_plus_hash,
)
from evalplus.eval import (
    PASS,
    untrusted_check,
)
from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
from evalplus.gen.util import trusted_exec
from evalplus.data.mbpp import mbpp_serialize_inputs
from evalplus.data.utils import CACHE_DIR

# Type definitions
Result = Tuple[str, List[bool]]  # (status, details)


class PerProblemEvaluator:
    """Main class for per-problem evaluation."""

    def __init__(self,
                 model_name: str,
                 dataset: str = "humaneval",
                 device: Optional[str] = None,
                 load_in_8bit: bool = False,
                 load_in_4bit: bool = False,
                 use_fp16: bool = True,
                 max_new_tokens: int = 512,
                 temperature: float = 0.1,
                 do_sample: bool = False,
                 base_only: bool = False,
                 fast_check: bool = True,
                 min_time_limit: float = 1.0,
                 gt_time_limit_factor: float = 2.0,
                 mini: bool = False,
                 noextreme: bool = False,
                 version: str = "default",
                 verbose: bool = False):
        """Initialize the per-problem evaluator.
        
        Args:
            model_name: Name of the model to evaluate
            dataset: Dataset to use ("humaneval" or "mbpp")
            device: Device to run on (None for auto-detect)
            load_in_8bit: Whether to load model in 8-bit precision
            load_in_4bit: Whether to load model in 4-bit precision
            use_fp16: Whether to use fp16 precision
            max_new_tokens: Maximum tokens to generate
            temperature: Generation temperature
            do_sample: Whether to use sampling
            base_only: Only test base cases (not plus cases)
            fast_check: Use fast checking mode
            min_time_limit: Minimum time limit for execution
            gt_time_limit_factor: Ground truth time limit factor
            mini: Use mini dataset
            noextreme: Exclude extreme cases
            version: Dataset version
        """
        self.model_name = model_name
        self.dataset = dataset
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.base_only = base_only
        self.fast_check = fast_check
        self.min_time_limit = min_time_limit
        self.gt_time_limit_factor = gt_time_limit_factor
        self.verbose = verbose

        # Load model
        self.model, self.tokenizer, self.device = (self._load_model(
            model_name, device, load_in_8bit, load_in_4bit, use_fp16))

        # Load dataset and ground truth
        self.problems = self._load_dataset(mini, noextreme, version)
        self.expected_outputs = self._load_ground_truth(
            mini, noextreme, version)

        # Results storage
        self.results = []

    def _load_model(self, model_name: str, device: Optional[str],
                    load_in_8bit: bool, load_in_4bit: bool,
                    use_fp16: bool) -> Tuple[Any, Any, str]:
        """Load the model and tokenizer."""
        if device is None:
            # Check for CUDA first, then MPS, then fall back to CPU
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends,
                         'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        print(f"Loading model: {model_name}")
        print(f"Using device: {device}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                  trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Model loading arguments
        model_args = {"trust_remote_code": True}

        if device == "cuda":
            if load_in_8bit:
                model_args["load_in_8bit"] = True
                model_args["device_map"] = "auto"
            elif load_in_4bit:
                model_args["load_in_4bit"] = True
                model_args["device_map"] = "auto"
            else:
                model_args[
                    "dtype"] = torch.float16 if use_fp16 else torch.float32
                model_args["device_map"] = "auto"
        elif device == "mps":
            # MPS doesn't support 8-bit/4-bit quantization
            if load_in_8bit or load_in_4bit:
                print(
                    "⚠️ MPS doesn't support 8-bit/4-bit quantization, using fp32"
                )
            model_args["dtype"] = torch.float32
            # Don't use device_map with MPS, we'll move manually
        else:  # CPU
            model_args["dtype"] = torch.float32
            if load_in_8bit or load_in_4bit:
                print(
                    "⚠️ CPU doesn't support 8-bit/4-bit quantization, using fp32"
                )

        # Load model
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_args)

        # Move to device if needed (for MPS/CPU)
        if device in ["mps", "cpu"]:
            model = model.to(device)

        model.eval()
        print(f"✓ Model loaded successfully on {device}")

        # Print device info for debugging
        if device == "mps":
            print("🚀 Using Apple Silicon MPS for acceleration")
        elif device == "cuda":
            print("🚀 Using CUDA for acceleration")
        else:
            print("⚠️ Using CPU (slow)")

        return model, tokenizer, device

    def _load_dataset(self, mini: bool, noextreme: bool, version: str) -> Dict:
        """Load the dataset problems."""
        print(f"Loading {self.dataset} dataset...")

        if self.dataset == "humaneval":
            problems = get_human_eval_plus(mini=mini,
                                           noextreme=noextreme,
                                           version=version)
        elif self.dataset == "mbpp":
            problems = get_mbpp_plus(mini=mini,
                                     noextreme=noextreme,
                                     version=version)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")

        print(f"✓ Loaded {len(problems)} problems")
        return problems

    def _load_ground_truth(self, mini: bool, noextreme: bool,
                           version: str) -> Dict:
        """Load or compute ground truth for the dataset."""
        if self.dataset == "humaneval":
            dataset_hash = get_human_eval_plus_hash(mini=mini,
                                                    noextreme=noextreme,
                                                    version=version)
            tasks_only_output_not_none = []
        elif self.dataset == "mbpp":
            dataset_hash = get_mbpp_plus_hash(mini=mini,
                                              noextreme=noextreme,
                                              version=version)
            tasks_only_output_not_none = MBPP_OUTPUT_NOT_NONE_TASKS
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")

        return self._get_ground_truth(self.problems, dataset_hash,
                                      tasks_only_output_not_none)

    def _get_ground_truth(self, problems: Dict, hashcode: str,
                          tasks_only_output_not_none: List[str]) -> Dict:
        """Get ground truth for problems (cached if available)."""
        cache_file = os.path.join(CACHE_DIR, f"{hashcode}.pkl")

        if os.path.exists(cache_file):
            print(f"Loading ground truth from {cache_file}")
            import pickle
            with open(cache_file, "rb") as f:
                return pickle.load(f)

        os.makedirs(CACHE_DIR, exist_ok=True)
        print("Computing expected output...")
        tbegin = time.time()
        expected_output = {}

        for task_id, problem in problems.items():
            oracle = {}
            oracle["base"], oracle["base_time"] = trusted_exec(
                problem["prompt"] + problem["canonical_solution"],
                problem["base_input"],
                problem["entry_point"],
                record_time=True,
                output_not_none=problem["entry_point"]
                in tasks_only_output_not_none,
            )

            oracle["plus"], oracle["plus_time"] = trusted_exec(
                problem["prompt"] + problem["canonical_solution"],
                problem["plus_input"],
                problem["entry_point"],
                record_time=True,
                output_not_none=problem["entry_point"]
                in tasks_only_output_not_none,
            )
            expected_output[task_id] = oracle

        print(f"Expected outputs computed in {time.time() - tbegin:.2f}s")

        import pickle
        with open(cache_file, "wb") as f:
            pickle.dump(expected_output, f)

        return expected_output

    def get_problem(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific problem by task_id.
        
        Args:
            task_id: The task ID to retrieve
            
        Returns:
            Problem dictionary or None if not found
        """
        if task_id not in self.problems:
            print(f"❌ Task {task_id} not found in dataset")
            return None

        problem = self.problems[task_id]
        print(f"✓ Retrieved problem: {task_id}")
        print(f"  Prompt length: {len(problem['prompt'])} characters")
        print(f"  Entry point: {problem['entry_point']}")

        if self.verbose:
            print(f"\n{'='*60}")
            print("PROBLEM PROMPT:")
            print(f"{'='*60}")
            print(problem['prompt'])
            print(f"{'='*60}")

        return problem

    def solve_problem(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a solution for a given problem.
        
        Args:
            problem: Problem dictionary from get_problem()
            
        Returns:
            Dictionary containing the solution and metadata
        """
        task_id = problem["task_id"]
        prompt = problem["prompt"]

        print(f"Generating solution for {task_id}...")

        try:
            # Encode prompt
            prompt_ids = self.tokenizer.encode(prompt,
                                               return_tensors="pt",
                                               add_special_tokens=False).to(
                                                   self.device)

            if prompt_ids.numel() == 0:
                prompt_ids = torch.tensor([[self.tokenizer.eos_token_id]],
                                          dtype=torch.long).to(self.device)

            attention_mask = torch.ones_like(prompt_ids)

            # Generate completion
            t0 = time.time()
            with torch.no_grad():
                gen_out = self.model.generate(
                    prompt_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature if self.do_sample else None,
                    do_sample=self.do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=None,
                )
            gen_time = time.time() - t0

            # Extract completion (excluding prompt)
            full_ids = gen_out[0].tolist()
            completion_ids = full_ids[len(prompt_ids[0]):]
            completion = self.tokenizer.decode(completion_ids,
                                               skip_special_tokens=True)

            # Construct full solution
            solution = prompt + completion

            result = {
                "task_id": task_id,
                "prompt": prompt,
                "completion": completion,
                "solution": solution,
                "generation_time": gen_time,
                "completion_length": len(completion_ids),
                "full_solution_length": len(solution)
            }

            print(f"✓ Generated solution in {gen_time:.2f}s")
            print(f"  Completion length: {len(completion_ids)} tokens")

            return result

        except Exception as e:
            print(f"❌ Error generating solution: {e}")
            return {
                "task_id": task_id,
                "prompt": prompt,
                "completion": "",
                "solution": prompt,
                "generation_time": 0.0,
                "completion_length": 0,
                "full_solution_length": len(prompt),
                "error": str(e)
            }

    def evaluate_solution(self, problem: Dict[str, Any],
                          solution_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a solution against the problem.
        
        Args:
            problem: Problem dictionary from get_problem()
            solution_data: Solution data from solve_problem()
            
        Returns:
            Dictionary containing evaluation results
        """
        task_id = problem["task_id"]
        solution = solution_data["solution"]
        expected_output = self.expected_outputs[task_id]

        print(f"Evaluating solution for {task_id}...")

        try:
            # Test base cases
            base_result = untrusted_check(
                self.dataset,
                solution,
                problem["base_input"],
                problem["entry_point"],
                expected=expected_output["base"],
                atol=problem["atol"],
                ref_time=expected_output["base_time"],
                fast_check=self.fast_check,
                min_time_limit=self.min_time_limit,
                gt_time_limit_factor=self.gt_time_limit_factor,
            )

            # Test plus cases (if not base_only)
            plus_result = None
            if not self.base_only:
                plus_result = untrusted_check(
                    self.dataset,
                    solution,
                    problem["plus_input"],
                    problem["entry_point"],
                    expected=expected_output["plus"],
                    atol=problem["atol"],
                    ref_time=expected_output["plus_time"],
                    fast_check=self.fast_check,
                    min_time_limit=self.min_time_limit,
                    gt_time_limit_factor=self.gt_time_limit_factor,
                )

            # Process results
            base_status, base_details = base_result
            base_passed = base_status == PASS
            base_fail_tests = []

            if base_status != PASS and base_details:
                if self.dataset == "mbpp":
                    base_fail_tests = mbpp_serialize_inputs(
                        task_id,
                        [problem["base_input"][len(base_details) - 1]])
                else:
                    base_fail_tests = [
                        problem["base_input"][len(base_details) - 1]
                    ]

            # Process plus results
            plus_status = None
            plus_passed = None
            plus_fail_tests = []

            if plus_result:
                plus_status, plus_details = plus_result
                plus_passed = plus_status == PASS

                if plus_status != PASS and plus_details:
                    if self.dataset == "mbpp":
                        plus_fail_tests = mbpp_serialize_inputs(
                            task_id,
                            [problem["plus_input"][len(plus_details) - 1]])
                    else:
                        plus_fail_tests = [
                            problem["plus_input"][len(plus_details) - 1]
                        ]

            evaluation_result = {
                "task_id":
                task_id,
                "base_passed":
                base_passed,
                "base_status":
                base_status,
                "base_fail_tests":
                base_fail_tests,
                "plus_passed":
                plus_passed,
                "plus_status":
                plus_status,
                "plus_fail_tests":
                plus_fail_tests,
                "overall_passed":
                base_passed
                and (plus_passed if plus_passed is not None else True)
            }

            # Print status
            status_symbol = "✅" if evaluation_result["overall_passed"] else "❌"
            print(
                f"{status_symbol} {task_id}: Base {'✓' if base_passed else '✗'}",
                end="")
            if plus_passed is not None:
                print(f", Plus {'✓' if plus_passed else '✗'}", end="")
            print()

            return evaluation_result

        except Exception as e:
            print(f"❌ Error evaluating solution: {e}")
            return {
                "task_id": task_id,
                "base_passed": False,
                "base_status": "error",
                "base_fail_tests": [],
                "plus_passed": False,
                "plus_status": "error",
                "plus_fail_tests": [],
                "overall_passed": False,
                "error": str(e)
            }

    def evaluate_problem(self, task_id: str) -> Dict[str, Any]:
        """Complete evaluation pipeline for a single problem.
        
        Args:
            task_id: The task ID to evaluate
            
        Returns:
            Dictionary containing all results for the problem
        """
        print(f"\n{'='*60}")
        print(f"Evaluating problem: {task_id}")
        print(f"{'='*60}")

        # Step 1: Get problem
        problem = self.get_problem(task_id)
        if problem is None:
            return {"error": f"Problem {task_id} not found"}

        # Step 2: Generate solution
        solution_data = self.solve_problem(problem)

        # Step 3: Evaluate solution
        evaluation_result = self.evaluate_solution(problem, solution_data)

        # Combine all results
        complete_result = {
            "task_id": task_id,
            "model_name": self.model_name,
            "dataset": self.dataset,
            "timestamp": datetime.now().isoformat(),
            **solution_data,
            **evaluation_result
        }

        # Store result
        self.results.append(complete_result)

        return complete_result

    def evaluate_multiple_problems(
            self, task_ids: List[str]) -> List[Dict[str, Any]]:
        """Evaluate multiple problems.
        
        Args:
            task_ids: List of task IDs to evaluate
            
        Returns:
            List of evaluation results
        """
        results = []

        for i, task_id in enumerate(task_ids, 1):
            print(f"\nProgress: {i}/{len(task_ids)}")
            result = self.evaluate_problem(task_id)
            results.append(result)

            # Small delay to avoid overwhelming the system
            time.sleep(0.1)

        return results

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics from all evaluated problems."""
        if not self.results:
            return {"error": "No results available"}

        total_problems = len(self.results)
        base_passed = sum(1 for r in self.results
                          if r.get("base_passed", False))
        plus_passed = sum(1 for r in self.results
                          if r.get("plus_passed", False))
        overall_passed = sum(1 for r in self.results
                             if r.get("overall_passed", False))

        avg_generation_time = np.mean(
            [r.get("generation_time", 0) for r in self.results])
        avg_completion_length = np.mean(
            [r.get("completion_length", 0) for r in self.results])

        stats = {
            "total_problems":
            total_problems,
            "base_pass_rate":
            base_passed / total_problems if total_problems > 0 else 0,
            "plus_pass_rate":
            plus_passed / total_problems if total_problems > 0 else 0,
            "overall_pass_rate":
            overall_passed / total_problems if total_problems > 0 else 0,
            "base_passed":
            base_passed,
            "plus_passed":
            plus_passed,
            "overall_passed":
            overall_passed,
            "avg_generation_time":
            avg_generation_time,
            "avg_completion_length":
            avg_completion_length,
            "model_name":
            self.model_name,
            "dataset":
            self.dataset
        }

        return stats

    def save_results(self, output_file: str):
        """Save all results to a JSON file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare data for saving
        save_data = {
            "config": {
                "model_name": self.model_name,
                "dataset": self.dataset,
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "do_sample": self.do_sample,
                "base_only": self.base_only,
                "timestamp": datetime.now().isoformat()
            },
            "summary": self.get_summary_stats(),
            "results": self.results
        }

        with open(output_path, 'w') as f:
            json.dump(save_data, f, indent=2)

        print(f"✓ Results saved to {output_path}")

    def print_summary(self):
        """Print a summary of all results."""
        stats = self.get_summary_stats()

        if "error" in stats:
            print(f"❌ {stats['error']}")
            return

        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Model: {stats['model_name']}")
        print(f"Dataset: {stats['dataset']}")
        print(f"Total problems: {stats['total_problems']}")
        print(
            f"Base pass rate: {stats['base_pass_rate']:.1%} ({stats['base_passed']}/{stats['total_problems']})"
        )

        if not self.base_only:
            print(
                f"Plus pass rate: {stats['plus_pass_rate']:.1%} ({stats['plus_passed']}/{stats['total_problems']})"
            )
            print(
                f"Overall pass rate: {stats['overall_pass_rate']:.1%} ({stats['overall_passed']}/{stats['total_problems']})"
            )

        print(f"Average generation time: {stats['avg_generation_time']:.2f}s")
        print(
            f"Average completion length: {stats['avg_completion_length']:.1f} tokens"
        )


def main():
    """Example usage of the PerProblemEvaluator."""
    import argparse

    parser = argparse.ArgumentParser(description="Per-problem code evaluation")
    parser.add_argument("model_name", help="Model to evaluate")
    parser.add_argument("--dataset",
                        default="humaneval",
                        choices=["humaneval", "mbpp"],
                        help="Dataset to use")
    parser.add_argument("--task-id", help="Specific task ID to evaluate")
    parser.add_argument("--task-ids",
                        nargs="+",
                        help="Multiple task IDs to evaluate")
    parser.add_argument("--max-tokens",
                        type=int,
                        default=512,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature",
                        type=float,
                        default=0.1,
                        help="Generation temperature")
    parser.add_argument("--do-sample",
                        action="store_true",
                        help="Use sampling")
    parser.add_argument("--base-only",
                        action="store_true",
                        help="Only test base cases")
    parser.add_argument("--output", help="Output file for results")
    parser.add_argument("--mini", action="store_true", help="Use mini dataset")
    parser.add_argument("--noextreme",
                        action="store_true",
                        help="Exclude extreme cases")
    parser.add_argument("--verbose",
                        action="store_true",
                        help="Print problem prompts")

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = PerProblemEvaluator(model_name=args.model_name,
                                    dataset=args.dataset,
                                    max_new_tokens=args.max_tokens,
                                    temperature=args.temperature,
                                    do_sample=args.do_sample,
                                    base_only=args.base_only,
                                    mini=args.mini,
                                    noextreme=args.noextreme,
                                    verbose=args.verbose)

    # Determine what to evaluate
    if args.task_id:
        # Single problem
        result = evaluator.evaluate_problem(args.task_id)
        print(f"\nSingle problem result: {result}")

    elif args.task_ids:
        # Multiple specific problems
        results = evaluator.evaluate_multiple_problems(args.task_ids)
        evaluator.print_summary()

    else:
        # First few problems from dataset
        task_ids = list(evaluator.problems.keys())[:5]  # First 5 problems
        print(f"Evaluating first {len(task_ids)} problems: {task_ids}")
        results = evaluator.evaluate_multiple_problems(task_ids)
        evaluator.print_summary()

    # Save results if output file specified
    if args.output:
        evaluator.save_results(args.output)


if __name__ == "__main__":
    main()
