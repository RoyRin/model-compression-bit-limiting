#!/usr/bin/env python3
"""
Script to evaluate OpenAI and HuggingFace models on EvalPlus benchmarks using the Python API.

Usage:
    # First install EvalPlus:
    pip install --upgrade "evalplus[vllm] @ git+https://github.com/evalplus/evalplus"
    
    # Set API keys if using OpenAI:
    export OPENAI_API_KEY="your-key-here"
    
    # Run the script:
    python run_evaluation.py
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional

# EvalPlus imports
from evalplus.data import get_human_eval_plus, get_mbpp_plus
from evalplus.provider import make_model
from evalplus.codegen import codegen
from evalplus.evaluate import evaluate as evalplus_evaluate


def run_evaluation(model: str,
                   dataset: str,
                   backend: str,
                   temperature: float = 0.0,
                   n_samples: int = 1,
                   output_dir: str = "evalplus_results",
                   **kwargs) -> str:
    """
    Run evaluation for a single model on a dataset.
    
    Args:
        model: Model name/path
        dataset: Dataset name ('humaneval' or 'mbpp')
        backend: Backend to use ('openai', 'vllm', 'hf', etc.)
        temperature: Sampling temperature (0.0 for greedy)
        n_samples: Number of samples to generate per problem
        output_dir: Directory to save results
        **kwargs: Additional arguments for the model
    
    Returns:
        Path to the generated samples file
    """

    print(f"\n{'='*60}")
    print(f"Evaluating {model} on {dataset}")
    print(
        f"Backend: {backend}, Temperature: {temperature}, Samples: {n_samples}"
    )
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    # Load dataset
    if dataset == "humaneval":
        problems = get_human_eval_plus()
    elif dataset == "mbpp":
        problems = get_mbpp_plus()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Create model
    model_obj = make_model(model=model,
                           backend=backend,
                           dataset=dataset,
                           temperature=temperature,
                           **kwargs)

    # Generate output path
    model_name = model.replace("/", "--")
    output_path = f"{output_dir}/{dataset}/{model_name}_{backend}_temp_{temperature}.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Run code generation
    print(f"Generating code solutions...")
    codegen(target_path=output_path,
            model=model_obj,
            dataset=problems,
            greedy=(temperature == 0.0),
            n_samples=n_samples,
            resume=True)

    print(f"Code generation complete. Results saved to: {output_path}")

    # Now run evaluation
    print(f"\nRunning correctness evaluation...")
    evalplus_evaluate(
        dataset=dataset,
        samples=output_path,
        parallel=None,  # Use default parallelism
        test_details=True,
        output_file=output_path.replace(".jsonl", "_eval.json"))

    # Load and display results
    eval_file = output_path.replace(".jsonl", "_eval.json")
    if os.path.exists(eval_file):
        with open(eval_file, 'r') as f:
            results = json.load(f)

        print(f"\n{'='*40}")
        print(f"Results for {model} on {dataset}:")
        print(f"{'='*40}")

        # Calculate pass@k metrics
        if "eval" in results:
            total = len(results["eval"])
            base_pass = sum(1 for task in results["eval"].values() if any(
                r.get("base_status") == "pass" for r in task))
            plus_pass = sum(1 for task in results["eval"].values() if any(
                r.get("plus_status") == "pass" for r in task))

            print(f"Total problems: {total}")
            print(
                f"Base tests pass@1: {base_pass}/{total} ({100*base_pass/total:.1f}%)"
            )
            print(
                f"Plus tests pass@1: {plus_pass}/{total} ({100*plus_pass/total:.1f}%)"
            )

    return output_path


def main():
    """Main function to run evaluations."""

    # Check environment
    openai_key = os.environ.get("OPENAI_API_KEY")

    # Configuration
    evaluations = []

    # Add OpenAI evaluation if API key is available
    if openai_key:
        evaluations.append({
            "model": "gpt-4o-mini",  # Cost-efficient model
            "backend": "openai",
            "datasets": ["humaneval", "mbpp"],
            "kwargs": {}
        })
    else:
        print("⚠️  OPENAI_API_KEY not set. Skipping OpenAI models.")
        print("   Set it with: export OPENAI_API_KEY='your-key-here'\n")

    # Add HuggingFace Llama 3 8B evaluation
    # Using vLLM backend for faster inference
    evaluations.append({
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "backend": "vllm",
        "datasets": ["humaneval", "mbpp"],
        "kwargs": {
            "tp": 1,  # Tensor parallel size
            "trust_remote_code": True,
        }
    })

    # Alternative: Use HF transformers backend (slower but more compatible)
    # Uncomment to use instead of vLLM:
    # evaluations.append({
    #     "model": "meta-llama/Meta-Llama-3-8B-Instruct",
    #     "backend": "hf",
    #     "datasets": ["humaneval", "mbpp"],
    #     "kwargs": {
    #         "trust_remote_code": True,
    #         "device_map": "auto",  # Automatic device placement
    #         "dtype": "bfloat16",
    #     }
    # })

    # Results summary
    all_results = []

    # Run evaluations
    for eval_config in evaluations:
        model = eval_config["model"]
        backend = eval_config["backend"]

        for dataset in eval_config["datasets"]:
            try:
                print(f"\n{'#'*60}")
                print(f"Starting: {model} on {dataset}")
                print(f"{'#'*60}")

                output_path = run_evaluation(
                    model=model,
                    dataset=dataset,
                    backend=backend,
                    temperature=0.0,  # Greedy decoding for deterministic results
                    n_samples=1,
                    **eval_config.get("kwargs", {}))

                all_results.append({
                    "model": model,
                    "dataset": dataset,
                    "backend": backend,
                    "status": "✓ Complete",
                    "output": output_path
                })

            except Exception as e:
                print(f"\n❌ Error evaluating {model} on {dataset}: {e}")
                all_results.append({
                    "model": model,
                    "dataset": dataset,
                    "backend": backend,
                    "status": "✗ Failed",
                    "error": str(e)
                })

    # Print final summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")

    for result in all_results:
        model_short = result["model"].split(
            "/")[-1] if "/" in result["model"] else result["model"]
        print(
            f"{result['status']}: {model_short} on {result['dataset']} ({result['backend']})"
        )
        if "output" in result:
            print(f"    Output: {result['output']}")
        if "error" in result:
            print(f"    Error: {result['error']}")

    print(f"\n{'='*60}")
    print("All evaluations complete!")
    print(f"Results are saved in: ./evalplus_results/")


if __name__ == "__main__":
    main()
