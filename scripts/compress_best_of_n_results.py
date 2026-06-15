#!/usr/bin/env python3
"""
Compress Best-of-N Experiment Results.

Reads JSONL files with text data (verbose solutions and rewrites) and computes
actual compression ratios using a GPU-based LLM compression model.

Input: JSONL files with structure:
  {
    "problem_id": ...,
    "verbose_solution_original": "...",
    "rewrites": [{"response": "...", ...}, ...]
  }

Output: JSON with compression statistics for each N value.
"""

import sys
import os

# Add repo root to path for imports
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

# Import compression utilities
from lossy_compression.core.lossy_compression_tools import (
    load_compression_model, compress_text)


def compress_texts_batch(texts: List[str],
                         compression_model,
                         compression_tokenizer,
                         device: str = "cuda",
                         verbose: bool = False) -> List[Dict[str, Any]]:
    """Compress a batch of texts and return compression metrics."""
    results = []

    for i, text in enumerate(texts):
        if verbose and i % 10 == 0:
            print(f"  Compressing text {i+1}/{len(texts)}...", flush=True)

        try:
            _, compression_pct, metrics = compress_text(
                text=text,
                compression_model=compression_model,
                compression_tokenizer=compression_tokenizer,
                device=device,
                verbose=False)
            results.append({
                'compression_pct': compression_pct,
                'compressed_bytes': metrics['compressed_bytes'],
                'original_bytes': metrics['original_bytes'],
                'bits_per_token': metrics['bits_per_token'],
                'n_tokens': metrics['n_tokens'],
                'success': True
            })
        except Exception as e:
            print(f"  Warning: compression failed for text {i}: {e}")
            results.append({
                'compression_pct': 100.0,
                'compressed_bytes': len(text.encode('utf-8')),
                'original_bytes': len(text.encode('utf-8')),
                'bits_per_token': 8.0,
                'n_tokens': 0,
                'success': False
            })

    return results


def process_just_ask_data(input_path: str,
                          compression_model,
                          compression_tokenizer,
                          device: str = "cuda",
                          verbose: bool = False) -> Dict[str, Any]:
    """Process just-ask best-of-N data."""

    print(f"Loading data from: {input_path}", flush=True)

    with open(input_path, 'r') as f:
        records = [json.loads(line) for line in f]

    print(f"Loaded {len(records)} problems", flush=True)

    results = []

    for idx, record in enumerate(records):
        problem_id = record.get('problem_id', idx)
        print(f"\nProblem {idx+1}/{len(records)} (ID: {problem_id})",
              flush=True)

        # Compress verbose solution
        verbose_solution = record.get('verbose_solution_original', '')
        if verbose_solution:
            print(
                f"  Compressing verbose solution ({len(verbose_solution)} chars)...",
                flush=True)
            verbose_result = compress_texts_batch([verbose_solution],
                                                  compression_model,
                                                  compression_tokenizer,
                                                  device)[0]
        else:
            verbose_result = {
                'compression_pct': 0,
                'compressed_bytes': 0,
                'original_bytes': 0
            }

        # Compress all rewrites
        rewrites = record.get('rewrites', [])
        rewrite_texts = [r.get('response', '') for r in rewrites]

        print(f"  Compressing {len(rewrite_texts)} rewrites...", flush=True)
        rewrite_results = compress_texts_batch(rewrite_texts,
                                               compression_model,
                                               compression_tokenizer, device,
                                               verbose)

        # Compute compression_vs_verbose for each rewrite
        # This is: rewrite_compressed_bytes / verbose_original_bytes * 100
        verbose_original_bytes = verbose_result['original_bytes']
        for rr in rewrite_results:
            if verbose_original_bytes > 0:
                rr['compression_vs_verbose'] = (rr['compressed_bytes'] /
                                                verbose_original_bytes) * 100
            else:
                rr['compression_vs_verbose'] = 100.0

        results.append({
            'problem_id':
            problem_id,
            'correct_answer':
            record.get('correct_answer', ''),
            'verbose_is_correct':
            record.get('verbose_is_correct', False),
            'verbose_compression':
            verbose_result,
            'verbose_original_bytes':
            verbose_result['original_bytes'],  # For easy access
            'rewrite_compressions':
            rewrite_results,
            'rewrite_is_correct':
            [r.get('is_correct', False) for r in rewrites]
        })

    return {
        'approach': 'just_ask',
        'source_file': input_path,
        'n_problems': len(records),
        'results': results
    }


def compute_statistics_by_n(
        data: Dict[str, Any],
        n_values: List[int] = [1, 3, 5, 10]) -> Dict[str, Any]:
    """Compute compression statistics grouped by N."""

    stats = {}

    for n in n_values:
        # For each problem, get best compression among first N rewrites
        best_compressions = []
        best_compressions_vs_verbose = []
        verbose_compressions = []

        for problem in data['results']:
            rewrites = problem['rewrite_compressions'][:n]

            if rewrites:
                # Best compression_pct among first N (lower is better)
                valid_compressions = [
                    r['compression_pct'] for r in rewrites
                    if r.get('success', True)
                ]
                if valid_compressions:
                    best_compressions.append(min(valid_compressions))

                # Best compression_vs_verbose among first N
                valid_vs_verbose = [
                    r['compression_vs_verbose'] for r in rewrites
                    if r.get('success', True)
                ]
                if valid_vs_verbose:
                    best_compressions_vs_verbose.append(min(valid_vs_verbose))

            # Verbose solution compression
            verbose_compressions.append(
                problem['verbose_compression']['compression_pct'])

        stats[str(n)] = {
            'n':
            n,
            'n_problems':
            len(best_compressions),
            # Best compression of rewrites (rewrite_compressed / rewrite_original)
            'avg_best_compression_pct':
            float(np.mean(best_compressions)) if best_compressions else 0,
            'std_best_compression_pct':
            float(np.std(best_compressions)) if best_compressions else 0,
            # Best compression vs verbose (rewrite_compressed / verbose_original)
            'avg_compression_vs_verbose':
            float(np.mean(best_compressions_vs_verbose))
            if best_compressions_vs_verbose else 0,
            'std_compression_vs_verbose':
            float(np.std(best_compressions_vs_verbose))
            if best_compressions_vs_verbose else 0,
            # Verbose baseline
            'avg_verbose_compression_pct':
            float(np.mean(verbose_compressions))
            if verbose_compressions else 0,
            'std_verbose_compression_pct':
            float(np.std(verbose_compressions)) if verbose_compressions else 0,
        }

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Compress best-of-N experiment texts on GPU")

    parser.add_argument("input",
                        type=str,
                        help="Input JSONL file with text data")
    parser.add_argument("--output",
                        "-o",
                        type=str,
                        default=None,
                        help="Output JSON file (default: auto-generated)")
    parser.add_argument(
        "--compression-model",
        type=str,
        default="meta-llama/Llama-3.1-8B",
        help="Compression model path (default: meta-llama/Llama-3.1-8B)")
    parser.add_argument("--device",
                        type=str,
                        default="cuda",
                        help="Device to use (default: cuda)")
    parser.add_argument(
        "--n-values",
        type=str,
        default="1,3,5,10",
        help="Comma-separated N values for statistics (default: 1,3,5,10)")
    parser.add_argument("--verbose",
                        "-v",
                        action="store_true",
                        help="Verbose output")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of problems to process (for testing)")

    args = parser.parse_args()

    # Parse N values
    n_values = [int(x) for x in args.n_values.split(',')]

    print("=" * 60)
    print("Best-of-N Compression Analysis")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Compression model: {args.compression_model}")
    print(f"Device: {args.device}")
    print(f"N values: {n_values}")
    print("=" * 60)

    # Load compression model
    print("\nLoading compression model...", flush=True)
    compression_model, compression_tokenizer = load_compression_model(
        args.compression_model, device=args.device)

    # Process data
    print("\nProcessing data...", flush=True)
    start_time = time.time()

    # Optionally limit records for testing
    if args.limit:
        print(f"Limiting to {args.limit} problems", flush=True)
        # Read, limit, write to temp file
        import tempfile
        with open(args.input, 'r') as f:
            records = [json.loads(line) for line in f][:args.limit]
        with tempfile.NamedTemporaryFile(mode='w',
                                         suffix='.jsonl',
                                         delete=False) as tmp:
            for r in records:
                tmp.write(json.dumps(r) + '\n')
            temp_path = tmp.name
        data = process_just_ask_data(temp_path, compression_model,
                                     compression_tokenizer, args.device,
                                     args.verbose)
        os.unlink(temp_path)
    else:
        data = process_just_ask_data(args.input, compression_model,
                                     compression_tokenizer, args.device,
                                     args.verbose)

    elapsed = time.time() - start_time
    print(f"\nProcessing complete in {elapsed:.1f}s", flush=True)

    # Compute statistics
    print("\nComputing statistics...", flush=True)
    stats = compute_statistics_by_n(data, n_values)

    # Add stats to data
    data['summary'] = stats
    data['parameters'] = {
        'n_values': n_values,
        'compression_model': args.compression_model,
        'processing_time_seconds': elapsed
    }

    # Print summary
    print("\n" + "=" * 70)
    print("COMPRESSION STATISTICS BY N")
    print("=" * 70)
    print("\nMetrics explained:")
    print("  Self-Comp%  = rewrite_compressed / rewrite_original * 100")
    print("               (how compressible the rewrite text is)")
    print("  vs Orig%    = rewrite_compressed / verbose_original * 100")
    print("               (compression ratio relative to original response)")
    print()
    print(
        f"{'N':>5} {'Self-Comp%':>12} {'Std':>8} {'vs Orig%':>12} {'Std':>8}")
    print("-" * 50)
    for n_key, s in sorted(stats.items(), key=lambda x: int(x[0])):
        print(
            f"{s['n']:>5} {s['avg_best_compression_pct']:>12.2f} {s['std_best_compression_pct']:>8.2f} "
            f"{s['avg_compression_vs_verbose']:>12.2f} {s['std_compression_vs_verbose']:>8.2f}"
        )
    print("-" * 50)
    verbose_avg = stats['1']['avg_verbose_compression_pct']
    verbose_std = stats['1']['std_verbose_compression_pct']
    print(
        f"{'Verbose':>5} {verbose_avg:>12.2f} {verbose_std:>8.2f} {'(self-compression of verbose)':>30}"
    )

    # Save results
    if args.output:
        output_path = args.output
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        input_stem = Path(args.input).stem
        output_path = f"results/compression_{input_stem}_{timestamp}.json"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
