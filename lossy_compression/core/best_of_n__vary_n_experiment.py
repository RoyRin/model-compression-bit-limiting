#!/usr/bin/env python3
"""
Experiment: Vary N (number of generations) and measure best compression ratio.

This script runs the best-of-N compression experiment with different values of N
to understand how the best compression ratio improves with more samples.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict, Any
import json
import time
from pathlib import Path
import numpy as np

# Import the main compression function
from lossy_compression_tools import load_compression_model

# Import diverse sampling for better generation diversity
from diverse_sampling import diverse_sample_api_vary_iterative_sampling

# Default experiment parameters
DEFAULT_PROMPTS = [
    "Explain how neural networks work:",
    "Write a story about a robot learning to paint:",
    "Describe the process of photosynthesis:",
    "What are the main causes of climate change?",
    "How does a computer processor work?",
]

DEFAULT_N_VALUES = [1, 2, 5, 10, 15, 25]
DEFAULT_MODEL = "claude-opus-4-1-20250805"
DEFAULT_COMPRESSION_MODEL = "meta-llama/Llama-3.1-8B"


def run_experiment(prompts: List[str] = DEFAULT_PROMPTS,
                   n_values: List[int] = DEFAULT_N_VALUES,
                   generation_model: str = DEFAULT_MODEL,
                   compression_model_path: str = DEFAULT_COMPRESSION_MODEL,
                   compression_model=None,
                   compression_tokenizer=None,
                   max_tokens: int = 200,
                   temperature: float = 0.8,
                   seed: int = 42,
                   output_dir: str = "results/vary_n_experiment",
                   use_diverse_sampling: bool = True,
                   verbose: bool = False) -> Dict[str, Any]:
    """
    Run experiment varying N for multiple prompts.
    
    Args:
        prompts: List of prompts to test
        n_values: List of N values to test
        generation_model: Model for text generation
        compression_model: Model for compression
        max_tokens: Max tokens per generation
        temperature: Sampling temperature
        seed: Random seed
        output_dir: Directory for results
        verbose: Print detailed progress
        
    Returns:
        Dictionary with experiment results
    """
    print(f"\n{'='*60}")
    print(f"🧪 EXPERIMENT: Best-of-N Compression Ratio vs N")
    print(f"{'='*60}")
    print(f"📝 Testing {len(prompts)} prompts")
    print(f"🔢 N values: {n_values}")
    print(f"🤖 Generation model: {generation_model}")
    print(f"📦 Compression model: {compression_model_path}")
    print(
        f"🎯 Sampling method: {'Iterative diversity' if use_diverse_sampling else 'Standard'}"
    )
    print(f"{'='*60}\n")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Results storage
    all_results = []
    summary_stats = {}

    total_runs = len(prompts) * len(n_values)
    run_count = 0

    # Load compression model once if not provided
    if compression_model is None or compression_tokenizer is None:
        import torch
        device = "cuda" if torch.cuda.is_available(
        ) else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"\n📦 Loading compression model once for all experiments...")
        compression_model, compression_tokenizer = load_compression_model(
            model_path=compression_model_path, device=device)
        print(f"✅ Model loaded and will be reused across all runs")
    else:
        print(f"\n📦 Using provided compression model")
        device = next(compression_model.parameters()).device.type

    # Run experiments
    for prompt_idx, prompt in enumerate(prompts):
        print(f"\n{'='*50}")
        print(f"PROMPT {prompt_idx + 1}/{len(prompts)}: {prompt[:60]}...")
        print(f"{'='*50}")

        prompt_results = {'prompt': prompt, 'results_by_n': {}}

        for n in n_values:
            run_count += 1
            print(f"\n[Run {run_count}/{total_runs}] Testing N={n}...")

            try:
                # Run compression-guided generation with shared model
                from lossy_compression_tools import (
                    select_best_by_compression, log_stats)
                import time

                # Generate outputs
                generation_start = time.time()

                if use_diverse_sampling:
                    # Use iterative sampling for better diversity
                    if verbose:
                        print(
                            f"  🔄 Generating {n} diverse outputs using iterative sampling..."
                        )

                    diverse_samples = diverse_sample_api_vary_iterative_sampling(
                        prompt=prompt,
                        num_samples=n,
                        model=generation_model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        seed=seed + prompt_idx * 1000,
                        verbose=False,  # Don't print each sample
                        diversity_instruction=
                        "Generate a significantly different response from the previous ones. Vary the style, approach, and content."
                    )

                    # Extract just the text from the samples
                    generations = [
                        sample['text'] for sample in diverse_samples
                    ]
                else:
                    # Use standard generation (original method)
                    if verbose:
                        print(
                            f"  🔄 Generating {n} outputs using standard sampling..."
                        )

                    from lossy_compression_tools import generate_multiple_outputs
                    generations = generate_multiple_outputs(
                        prompt=prompt,
                        n_generations=n,
                        model=generation_model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        seed=seed + prompt_idx * 1000,
                        verbose=verbose)

                generation_time = time.time() - generation_start

                if verbose:
                    print(f"  ✅ Generated {len(generations)} outputs")

                if not generations:
                    raise ValueError("No generations produced")

                # Select best by compression using shared model
                selection_start = time.time()
                best_index, best_text, selection_metrics = select_best_by_compression(
                    prompt=prompt,
                    generations=generations,
                    compression_model=compression_model,
                    compression_tokenizer=compression_tokenizer,
                    bit_precision=64,
                    device=device,
                    verbose=verbose,
                    verify_correctness=False)
                selection_time = time.time() - selection_start

                # Compile results
                result = {
                    'prompt': prompt,
                    'parameters': {
                        'n_generations':
                        n,
                        'generation_model':
                        generation_model,
                        'compression_model':
                        compression_model_path,
                        'sampling_method':
                        'iterative_diverse'
                        if use_diverse_sampling else 'standard',
                        'max_tokens':
                        max_tokens,
                        'temperature':
                        temperature,
                        'seed':
                        seed + prompt_idx * 1000
                    },
                    'best_generation': {
                        'index': best_index,
                        'text': best_text,
                        'compression_pct': selection_metrics['best_pct']
                    },
                    'all_generations': generations,
                    'selection_metrics': selection_metrics,
                    'timing': {
                        'generation_time': generation_time,
                        'selection_time': selection_time,
                        'total_time': generation_time + selection_time
                    }
                }

                # Log stats
                log_stats(result, output_dir)

                # Extract key metrics and generation lengths
                generation_lengths = [
                    len(gen) for gen in result['all_generations']
                ]
                min_length = min(
                    generation_lengths) if generation_lengths else 0
                max_length = max(
                    generation_lengths) if generation_lengths else 0
                mean_length = sum(generation_lengths) / len(
                    generation_lengths) if generation_lengths else 0

                prompt_results['results_by_n'][n] = {
                    'best_pct': result['best_generation']['compression_pct'],
                    'best_index': result['best_generation']['index'],
                    'statistics': result['selection_metrics']['statistics'],
                    'timing': result['timing'],
                    'generation_lengths': {
                        'min': min_length,
                        'max': max_length,
                        'mean': mean_length
                    }
                }

                print(
                    f"  ✅ Best: {result['best_generation']['compression_pct']:.1f}% of original"
                )
                print(
                    f"  📏 Generation lengths: min={min_length}, max={max_length}, mean={mean_length:.0f}"
                )

            except Exception as e:
                print(f"  ❌ Failed: {e}")
                prompt_results['results_by_n'][n] = {
                    'error': str(e),
                    'best_pct': None
                }

        all_results.append(prompt_results)

    # Calculate summary statistics across prompts
    print(f"\n{'='*60}")
    print(f"📊 CALCULATING SUMMARY STATISTICS")
    print(f"{'='*60}")

    for n in n_values:
        pcts = []
        for prompt_result in all_results:
            if n in prompt_result['results_by_n']:
                pct = prompt_result['results_by_n'][n].get('best_pct')
                if pct is not None:
                    pcts.append(pct)

        if pcts:
            summary_stats[n] = {
                'mean': np.mean(pcts),
                'std': np.std(pcts),
                'min': np.min(pcts),
                'max': np.max(pcts),
                'count': len(pcts)
            }

            print(
                f"N={n:2d}: mean={summary_stats[n]['mean']:.1f}%±{summary_stats[n]['std']:.1f}%, "
                f"range=[{summary_stats[n]['min']:.1f}%, {summary_stats[n]['max']:.1f}%]"
            )

    # Save results
    experiment_data = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'parameters': {
            'prompts':
            prompts,
            'n_values':
            n_values,
            'generation_model':
            str(generation_model)
            if not isinstance(generation_model, str) else generation_model,
            'compression_model':
            compression_model_path,
            'sampling_method':
            'iterative_diverse' if use_diverse_sampling else 'standard',
            'max_tokens':
            max_tokens,
            'temperature':
            temperature,
            'seed':
            seed
        },
        'results': all_results,
        'summary_stats': summary_stats
    }

    # Save to JSON
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    json_file = output_path / f"vary_n_experiment_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(experiment_data, f, indent=2)

    print(f"\n💾 Results saved to: {json_file}")

    # Also save a simple CSV for easy plotting
    csv_file = output_path / f"vary_n_summary_{timestamp}.csv"
    sampling_method = 'iterative_diverse' if use_diverse_sampling else 'standard'
    with open(csv_file, 'w') as f:
        f.write("n,mean_pct,std_pct,min_pct,max_pct,count,sampling_method\n")
        for n in sorted(summary_stats.keys()):
            s = summary_stats[n]
            f.write(
                f"{n},{s['mean']},{s['std']},{s['min']},{s['max']},{s['count']},{sampling_method}\n"
            )

    print(f"📊 Summary CSV saved to: {csv_file}")

    print(f"\n{'='*60}")
    print(f"✅ EXPERIMENT COMPLETE")
    print(f"{'='*60}\n")

    return experiment_data


def parse_args():
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run best-of-N compression experiment with varying N")

    # Experiment parameters
    parser.add_argument(
        "--prompts",
        type=str,
        nargs='+',
        default=None,
        help="Custom prompts to test (default: use built-in prompts)")
    parser.add_argument("--n-values",
                        type=int,
                        nargs='+',
                        default=DEFAULT_N_VALUES,
                        help=f"N values to test (default: {DEFAULT_N_VALUES})")

    # Model parameters
    parser.add_argument("--generation-model",
                        type=str,
                        default=DEFAULT_MODEL,
                        help=f"Generation model (default: {DEFAULT_MODEL})")
    parser.add_argument(
        "--compression-model",
        type=str,
        default=DEFAULT_COMPRESSION_MODEL,
        help=f"Compression model (default: {DEFAULT_COMPRESSION_MODEL})")

    # Generation parameters
    parser.add_argument("--max-tokens",
                        type=int,
                        default=200,
                        help="Maximum tokens per generation (default: 200)")
    parser.add_argument("--temperature",
                        type=float,
                        default=0.8,
                        help="Sampling temperature (default: 0.8)")
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="Random seed (default: 42)")

    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/vary_n_experiment",
        help="Output directory (default: results/vary_n_experiment)")
    parser.add_argument("--verbose",
                        action="store_true",
                        help="Show detailed progress")
    parser.add_argument(
        "--no-diverse-sampling",
        action="store_true",
        help="Use standard sampling instead of iterative diverse sampling")

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Use custom prompts if provided, otherwise use defaults
    prompts = args.prompts if args.prompts else DEFAULT_PROMPTS

    # Run experiment with shared model
    results = run_experiment(prompts=prompts,
                             n_values=args.n_values,
                             generation_model=args.generation_model,
                             compression_model_path=args.compression_model,
                             max_tokens=args.max_tokens,
                             temperature=args.temperature,
                             seed=args.seed,
                             output_dir=args.output_dir,
                             use_diverse_sampling=not args.no_diverse_sampling,
                             verbose=args.verbose)

    return results


if __name__ == "__main__":
    main()
