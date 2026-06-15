#!/usr/bin/env python3
"""
Run EvalPlus evaluation with HuggingFace models, with automatic device detection (MPS/CUDA/CPU).
"""

import os
import sys
import json
import torch
from pathlib import Path
from typing import Optional, Dict, Any

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from evalplus.data import get_human_eval_plus, get_mbpp_plus, write_jsonl
from transformers import AutoTokenizer, AutoModelForCausalLM


def detect_device():
    """Automatically detect the best available device."""
    if torch.cuda.is_available():
        device = "cuda"
        device_name = torch.cuda.get_device_name(0)
        print(f"🚀 Using CUDA GPU: {device_name}")
        return device
    elif torch.backends.mps.is_available():
        device = "mps"
        print(f"🍎 Using Apple Silicon GPU (MPS)")
        return device
    else:
        device = "cpu"
        print(f"💻 Using CPU (no GPU detected)")
        return device


def get_dtype_for_device(device: str, use_fp16: bool = True):
    """Get appropriate dtype for the device."""
    if device == "cuda" and use_fp16:
        return torch.float16
    elif device == "mps":
        # MPS works better with float32 for some models
        return torch.float32
    else:
        return torch.float32


def load_model(model_name: str,
               device: Optional[str] = None,
               load_in_8bit: bool = False,
               load_in_4bit: bool = False,
               use_fp16: bool = True):
    """Load a HuggingFace model with appropriate settings for the device."""

    if device is None:
        device = detect_device()

    print(f"Loading model: {model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model loading arguments
    model_args = {
        "trust_remote_code": True,
    }

    # Configure for different devices
    if device == "cuda":
        if load_in_8bit:
            model_args["load_in_8bit"] = True
            model_args["device_map"] = "auto"
        elif load_in_4bit:
            model_args["load_in_4bit"] = True
            model_args["device_map"] = "auto"
        else:
            model_args["dtype"] = get_dtype_for_device(device, use_fp16)
            model_args["device_map"] = "auto"
    elif device == "mps":
        # MPS doesn't support 8-bit/4-bit quantization
        if load_in_8bit or load_in_4bit:
            print(
                "⚠️ MPS doesn't support 8-bit/4-bit quantization, using fp32")
        model_args["dtype"] = torch.float32
        # Don't use device_map with MPS, we'll move manually
    else:  # CPU
        model_args["dtype"] = torch.float32
        if load_in_8bit or load_in_4bit:
            print(
                "⚠️ CPU doesn't support 8-bit/4-bit quantization, using fp32")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_args)

    # Move to device if needed (for MPS/CPU)
    if device in ["mps", "cpu"]:
        model = model.to(device)

    return model, tokenizer, device


def generate_completion(model,
                        tokenizer,
                        prompt: str,
                        device: str,
                        max_new_tokens: int = 512,
                        temperature: float = 0.1,
                        do_sample: bool = False) -> str:
    """Generate completion for a given prompt."""

    # Tokenize
    inputs = tokenizer(prompt,
                       return_tensors="pt",
                       truncation=True,
                       max_length=1024)

    # Move inputs to device
    if device in ["cuda", "mps"]:
        inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        # Build generation kwargs
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id
        }

        # Only add temperature if do_sample is True
        if do_sample:
            gen_kwargs["temperature"] = temperature

        # Generate
        outputs = model.generate(**inputs, **gen_kwargs)

    # Decode
    completion = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove the prompt from completion
    if completion.startswith(prompt):
        completion = completion[len(prompt):]

    return completion


def _run_simple_evaluation(output_path: str, dataset: str, problems: Dict):
    """Run a simplified evaluation for a subset of problems."""
    import json
    from evalplus.gen.util import trusted_exec

    # Load solutions
    with open(output_path, 'r') as f:
        solutions = [json.loads(line) for line in f]

    # Evaluate each solution
    results = []
    for solution_data in solutions:
        task_id = solution_data["task_id"]
        completion = solution_data["completion"]
        problem = problems[task_id]

        # Construct full solution
        full_solution = problem["prompt"] + completion

        # Test with base inputs
        try:
            # Run the solution with test inputs
            exec_result = trusted_exec(full_solution, problem["base_input"],
                                       problem["entry_point"])

            # Check if it passed
            passed = exec_result[0] == "passed"
            results.append({
                "task_id": task_id,
                "passed": passed,
                "error": None if passed else exec_result[0]
            })

            status = "✅" if passed else "❌"
            error_msg = "" if passed else f" - {exec_result[0]}"
            print(
                f"  {status} {task_id}: {'Passed' if passed else 'Failed'}{error_msg}"
            )

        except Exception as e:
            results.append({
                "task_id": task_id,
                "passed": False,
                "error": str(e)
            })
            print(f"  ❌ {task_id}: Error - {e}")

    # Calculate pass rate
    passed_count = sum(1 for r in results if r["passed"])
    total = len(results)
    pass_rate = passed_count / total if total > 0 else 0

    print(f"\n📊 Results Summary:")
    print(f"  Pass rate: {pass_rate:.1%} ({passed_count}/{total})")

    # Save results
    results_path = output_path.replace(".jsonl", "_simple_results.json")
    with open(results_path, 'w') as f:
        json.dump(
            {
                "dataset": dataset,
                "total": total,
                "passed": passed_count,
                "pass_rate": pass_rate,
                "details": results
            },
            f,
            indent=2)
    print(f"  Results saved to: {results_path}")


def run_evaluation(model_name: str = "codellama/CodeLlama-7b-Python-hf",
                   dataset: str = "humaneval",
                   n_problems: Optional[int] = None,
                   device: Optional[str] = None,
                   load_in_8bit: bool = False,
                   load_in_4bit: bool = False,
                   use_fp16: bool = True,
                   output_dir: str = "./evalplus_results"):
    """Run EvalPlus evaluation with a HuggingFace model."""

    # Load model
    model, tokenizer, device = load_model(model_name,
                                          device=device,
                                          load_in_8bit=load_in_8bit,
                                          load_in_4bit=load_in_4bit,
                                          use_fp16=use_fp16)

    # Load dataset
    if dataset == "humaneval":
        problems = get_human_eval_plus()
    elif dataset == "mbpp":
        problems = get_mbpp_plus()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Limit problems if specified
    if n_problems:
        problems = dict(list(problems.items())[:n_problems])
        print(f"Evaluating on first {n_problems} problems")

    # Generate solutions
    solutions = []
    print(f"\nGenerating solutions for {len(problems)} problems...")

    for i, (task_id, problem) in enumerate(problems.items(), 1):
        print(f"[{i}/{len(problems)}] {task_id}")

        prompt = problem["prompt"]

        try:
            completion = generate_completion(model,
                                             tokenizer,
                                             prompt,
                                             device,
                                             max_new_tokens=512,
                                             temperature=0.1,
                                             do_sample=False)

            solutions.append({"task_id": task_id, "completion": completion})
        except Exception as e:
            print(f"  ❌ Error: {e}")
            solutions.append({"task_id": task_id, "completion": ""})

    # Save solutions
    os.makedirs(output_dir, exist_ok=True)
    model_safe_name = model_name.replace("/", "--")
    output_path = f"{output_dir}/{dataset}_{model_safe_name}_samples.jsonl"
    write_jsonl(output_path, solutions)
    print(f"\n✅ Solutions saved to {output_path}")

    # Run evaluation
    print("\nRunning evaluation...")

    # Set environment variable to avoid tokenizer warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # If we limited problems, we need to handle evaluation differently
    if n_problems:
        print(f"Running simplified evaluation for {n_problems} problems...")
        _run_simple_evaluation(output_path, dataset, problems)
    else:
        # Full evaluation using the existing harness
        from lossy_compression.evals.eval_plus_harness import evaluate

        evaluate(
            dataset=dataset,
            samples=output_path,
            parallel=1,  # Single worker to avoid resource issues on Mac
            base_only=False)

    print(f"\n✅ Evaluation complete! Check results in {output_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run EvalPlus with HuggingFace models")
    parser.add_argument("--model",
                        type=str,
                        default="codellama/CodeLlama-7b-Python-hf",
                        help="HuggingFace model name/path")
    parser.add_argument("--dataset",
                        type=str,
                        default="humaneval",
                        choices=["humaneval", "mbpp"],
                        help="Dataset to evaluate on")
    parser.add_argument("--n-problems",
                        type=int,
                        default=None,
                        help="Limit to first N problems (for testing)")
    parser.add_argument("--device",
                        type=str,
                        default=None,
                        choices=["cuda", "mps", "cpu", None],
                        help="Device to use (auto-detect if not specified)")
    parser.add_argument("--load-in-8bit",
                        action="store_true",
                        help="Load model in 8-bit (CUDA only)")
    parser.add_argument("--load-in-4bit",
                        action="store_true",
                        help="Load model in 4-bit (CUDA only)")
    parser.add_argument("--no-fp16",
                        action="store_true",
                        help="Don't use fp16 (use fp32 instead)")
    parser.add_argument("--output-dir",
                        type=str,
                        default="./evalplus_results",
                        help="Output directory for results")

    args = parser.parse_args()

    # Print system info
    print("=" * 60)
    print("System Information")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print("=" * 60)

    run_evaluation(model_name=args.model,
                   dataset=args.dataset,
                   n_problems=args.n_problems,
                   device=args.device,
                   load_in_8bit=args.load_in_8bit,
                   load_in_4bit=args.load_in_4bit,
                   use_fp16=not args.no_fp16,
                   output_dir=args.output_dir)


if __name__ == "__main__":
    main()
