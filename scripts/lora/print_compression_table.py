#!/usr/bin/env python3
"""Print compression ratio table from RAG LoRA evaluation results.

Compression ratio is computed as:
    compression_ratio = total_bits / (num_tokens * log2(vocab_size))

Where:
    - total_bits = num_blocks * (bit_precision + bits_for_encoding_count)
                 = num_blocks * (64 + 7) = num_blocks * 71
    - vocab_size = 32000 for Mistral-7B
    - log2(32000) ≈ 14.97 bits per token (uniform encoding baseline)

This ratio compares the arithmetic coding bits to uniform token encoding.
A ratio of 0.07 means we use 7% of the bits needed for uniform encoding.
"""

import argparse
import json
import numpy as np
from pathlib import Path


def load_compression_ratios_from_jobs(base_dir: Path):
    """Load per-sample compression ratios from job directories.

    Returns dict mapping cluster_id -> {baseline_ratios, rag_ratios, correct_ratios, acc}
    """
    clusters = {}

    for job_dir in sorted(base_dir.glob("job_*")):
        result_files = list(job_dir.glob("rag_lora_results_*.json"))
        if not result_files:
            continue

        with open(result_files[0]) as f:
            d = json.load(f)

        comp = d.get('compression', {})
        acc_data = d.get('accuracy', {})

        for cluster_id, cluster_data in comp.get('per_cluster', {}).items():
            results = cluster_data.get('results', [])
            if not results:
                continue

            # Extract per-sample compression ratios
            baseline_ratios = [
                r['baseline_compression_ratio'] for r in results
            ]
            rag_ratios = [r['rag_compression_ratio'] for r in results]
            correct_ratios = [
                r.get('correct_compression_ratio', r['rag_compression_ratio'])
                for r in results
            ]

            # Get accuracy
            cluster_acc = acc_data.get('per_cluster', {}).get(cluster_id, {})
            acc_pct = cluster_acc.get('accuracy', 0) * 100

            clusters[int(cluster_id)] = {
                'baseline_ratios': baseline_ratios,
                'rag_ratios': rag_ratios,
                'correct_ratios': correct_ratios,
                'acc': acc_pct,
                'samples': len(results),
            }

    return clusters


def load_compression_ratios_from_single_file(results_path: Path):
    """Load per-sample compression ratios from a single results file.

    Handles two formats:
    1. Regular results: d['compression']['per_cluster'][cluster_id]['results']
    2. Merged results: d['per_cluster'][cluster_id]['results']
    """
    with open(results_path) as f:
        d = json.load(f)

    clusters = {}

    # Try regular format first, then merged format
    if 'compression' in d:
        per_cluster = d['compression'].get('per_cluster', {})
        acc_data = d.get('accuracy', {})
    elif 'per_cluster' in d:
        # Merged results format
        per_cluster = d['per_cluster']
        acc_data = None  # accuracy is in per_cluster data
    else:
        return clusters

    for cluster_id, cluster_data in per_cluster.items():
        results = cluster_data.get('results', [])
        if not results:
            continue

        baseline_ratios = [r['baseline_compression_ratio'] for r in results]
        rag_ratios = [r['rag_compression_ratio'] for r in results]
        correct_ratios = [
            r.get('correct_compression_ratio', r['rag_compression_ratio'])
            for r in results
        ]

        # Get accuracy from appropriate location
        if acc_data is not None:
            cluster_acc = acc_data.get('per_cluster', {}).get(cluster_id, {})
            acc_pct = cluster_acc.get('accuracy', 0) * 100
        else:
            # Merged format: accuracy is stored as correct_lora_pct in cluster_data
            acc_pct = cluster_data.get('correct_lora_pct', 0)

        clusters[int(cluster_id)] = {
            'baseline_ratios': baseline_ratios,
            'rag_ratios': rag_ratios,
            'correct_ratios': correct_ratios,
            'acc': acc_pct,
            'samples': len(results),
        }

    return clusters


def print_table(results_path: Path, sort_by: str = None):
    """Print compression ratio table.

    Args:
        results_path: Path to results directory (with job_* subdirs) or single JSON file
        sort_by: Column to sort by ('baseline', 'rag', 'correct', 'acc', 'samples', 'cluster')
    """
    # Determine if this is a directory with job subdirs or a single file
    if results_path.is_dir() and list(results_path.glob("job_*")):
        clusters = load_compression_ratios_from_jobs(results_path)
    elif results_path.is_file():
        clusters = load_compression_ratios_from_single_file(results_path)
    else:
        print(
            f"Error: {results_path} is neither a job directory nor a results file"
        )
        return

    if not clusters:
        print("No compression data found")
        return

    # Build rows with averaged compression ratios
    rows = []
    for cluster_id, data in clusters.items():
        rows.append({
            'cluster_id': cluster_id,
            'samples': data['samples'],
            'baseline': np.mean(data['baseline_ratios']),
            'rag': np.mean(data['rag_ratios']),
            'correct': np.mean(data['correct_ratios']),
            'acc': data['acc'],
            # Keep raw ratios for weighted averaging
            'baseline_ratios': data['baseline_ratios'],
            'rag_ratios': data['rag_ratios'],
            'correct_ratios': data['correct_ratios'],
        })

    # Sort rows
    sort_keys = {
        'cluster': lambda x: x['cluster_id'],
        'samples': lambda x: -x['samples'],  # descending
        'baseline': lambda x: x['baseline'],
        'rag': lambda x: x['rag'],
        'correct': lambda x: x['correct'],
        'acc': lambda x: -x['acc'],  # descending
    }
    sort_key = sort_keys.get(sort_by, sort_keys['cluster'])
    rows.sort(key=sort_key)

    print(
        f"{'Cluster':<8} {'Samples':<8} {'Baseline':<12} {'RAG LoRA':<12} {'Correct LoRA':<14} {'RAG Acc':<10}"
    )
    print("=" * 72)

    # Collect all ratios for weighted average
    all_baseline = []
    all_rag = []
    all_correct = []

    for r in rows:
        print(
            f"{r['cluster_id']:<8} {r['samples']:<8} {r['baseline']:<12.4f} {r['rag']:<12.4f} {r['correct']:<14.4f} {r['acc']:<10.1f}%"
        )
        all_baseline.extend(r['baseline_ratios'])
        all_rag.extend(r['rag_ratios'])
        all_correct.extend(r['correct_ratios'])

    print("=" * 72)

    # Weighted average (by sample count)
    total_samples = len(all_rag)
    avg_baseline = np.mean(all_baseline)
    avg_rag = np.mean(all_rag)
    avg_correct = np.mean(all_correct)

    print(
        f"{'TOTAL':<8} {total_samples:<8} {avg_baseline:<12.4f} {avg_rag:<12.4f} {avg_correct:<14.4f}"
    )

    print(f"\nCompression ratio = bits / (tokens × log2(vocab_size))")
    print(
        f"Comparing to uniform token encoding (~15 bits/token for Mistral-7B)")


def main():
    parser = argparse.ArgumentParser(
        description="Print compression ratio table from RAG LoRA results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Compression ratio formula:
    ratio = total_bits / (num_tokens × log2(vocab_size))

Where total_bits = num_blocks × 71 (64 bit_precision + 7 encoding count)
and log2(32000) ≈ 14.97 for Mistral-7B.

A ratio of 0.07 means using 7% of uniform encoding bits.
        """)
    parser.add_argument(
        "results_path",
        type=Path,
        help=
        "Path to results directory (with job_* subdirs) or single JSON file")
    parser.add_argument(
        "--sort",
        choices=['cluster', 'samples', 'baseline', 'rag', 'correct', 'acc'],
        default='cluster',
        help="Column to sort by (default: cluster)")
    args = parser.parse_args()

    if not args.results_path.exists():
        print(f"Error: {args.results_path} not found")
        return 1

    print_table(args.results_path, sort_by=args.sort)
    return 0


if __name__ == "__main__":
    exit(main())
