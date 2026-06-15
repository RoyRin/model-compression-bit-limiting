#!/usr/bin/env python3
"""Merge results from parallel RAG LoRA evaluation jobs."""
import json
import numpy as np
from pathlib import Path
from datetime import datetime


def merge_results(base_dir: Path, num_jobs: int = 6) -> dict:
    """Merge results from multiple job directories."""

    merged = {
        'per_cluster': {},
        'all_bpt': [],
        'all_baseline_bpt': [],
        'all_correct_bpt': [],
    }

    for job_id in range(num_jobs):
        job_dir = base_dir / f"job_{job_id}"

        # Try to find the results file
        result_files = list(job_dir.glob("rag_lora_results_*.json"))
        if not result_files:
            # Check intermediate
            intermediate = job_dir / "intermediate_compression.json"
            if intermediate.exists():
                print(f"Job {job_id}: Using intermediate results")
                with open(intermediate) as f:
                    data = json.load(f)
                for cluster_id, cluster_data in data.get('per_cluster',
                                                         {}).items():
                    merged['per_cluster'][int(cluster_id)] = cluster_data
            else:
                print(f"Job {job_id}: No results found in {job_dir}")
            continue

        result_file = sorted(result_files)[-1]  # Latest
        print(f"Job {job_id}: Loading {result_file.name}")

        with open(result_file) as f:
            data = json.load(f)

        if 'compression' in data:
            comp = data['compression']
            for cluster_id, cluster_data in comp.get('per_cluster',
                                                     {}).items():
                merged['per_cluster'][int(cluster_id)] = cluster_data

                # Aggregate sample-level data
                for sample in cluster_data.get('results', []):
                    merged['all_bpt'].append(sample['rag_bpt'])
                    merged['all_baseline_bpt'].append(sample['baseline_bpt'])
                    if 'correct_bpt' in sample:
                        merged['all_correct_bpt'].append(sample['correct_bpt'])

    # Compute overall averages
    merged['overall_avg_rag_bpt'] = float(np.mean(
        merged['all_bpt'])) if merged['all_bpt'] else 0
    merged['overall_avg_baseline_bpt'] = float(
        np.mean(
            merged['all_baseline_bpt'])) if merged['all_baseline_bpt'] else 0
    merged['overall_avg_correct_bpt'] = float(
        np.mean(merged['all_correct_bpt'])) if merged['all_correct_bpt'] else 0
    merged['total_samples'] = len(merged['all_bpt'])
    merged['num_clusters'] = len(merged['per_cluster'])

    return merged


def print_summary(results: dict):
    """Print a summary table."""
    print("\n" + "=" * 90)
    print("MERGED RESULTS SUMMARY")
    print("=" * 90)

    print(f"\nTotal samples: {results['total_samples']}")
    print(f"Total clusters: {results['num_clusters']}")

    print(f"\nOverall compression (bits per byte):")
    print(f"  Baseline:     {results['overall_avg_baseline_bpt']:.4f}")
    print(f"  RAG LoRA:     {results['overall_avg_rag_bpt']:.4f}")
    print(f"  Correct LoRA: {results['overall_avg_correct_bpt']:.4f}")

    # Compression ratios
    print(f"\nOverall compression ratios (8 / bpb):")
    print(f"  Baseline:     {8/results['overall_avg_baseline_bpt']:.2f}x")
    print(f"  RAG LoRA:     {8/results['overall_avg_rag_bpt']:.2f}x")
    print(f"  Correct LoRA: {8/results['overall_avg_correct_bpt']:.2f}x")

    # Per-cluster table
    print(
        f"\n{'Cluster':<8} {'Samples':<8} {'Baseline':<10} {'RAG':<10} {'Correct':<10} {'RAG Acc':<10}"
    )
    print("-" * 60)

    for cluster_id in sorted(results['per_cluster'].keys()):
        c = results['per_cluster'][cluster_id]
        baseline = c.get('avg_baseline_bpt', 0)
        rag = c.get('avg_rag_bpt', 0)
        correct = c.get('avg_correct_bpt', 0)
        samples = c.get('num_samples', 0)
        acc = c.get('correct_lora_pct', 0)

        print(
            f"{cluster_id:<8} {samples:<8} {8/baseline:<10.2f}x {8/rag:<10.2f}x {8/correct:<10.2f}x {acc:<8.1f}%"
        )

    print("-" * 60)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir",
                        type=str,
                        default="results/rag_lora_evaluation_enwik9-50-full")
    parser.add_argument("--num-jobs", type=int, default=6)
    parser.add_argument("--output",
                        type=str,
                        default=None,
                        help="Output file for merged results")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    results = merge_results(base_dir, args.num_jobs)

    print_summary(results)

    # Save merged results
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = base_dir / f"merged_results_{timestamp}.json"

    # Remove large lists before saving (keep per-cluster data)
    results_to_save = {
        k: v
        for k, v in results.items()
        if k not in ['all_bpt', 'all_baseline_bpt', 'all_correct_bpt']
    }

    with open(output_path, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    print(f"\nSaved merged results to {output_path}")


if __name__ == "__main__":
    main()
