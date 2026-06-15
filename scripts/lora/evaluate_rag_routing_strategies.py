#!/usr/bin/env python3
"""
Compare different RAG routing strategies for LoRA selection.

Strategies:
1. Prompt-RAG: Route based on user messages only (what we'd have before generation)
2. Response-RAG: Route based on assistant messages only (best signal for compression)
3. Full-RAG: Route based on full conversation (current approach)

The hypothesis is that Response-RAG will have better routing accuracy since
the response text is what we're compressing and has clearer domain signals.
"""

import argparse
import json
import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from collections import Counter
import random
import ast

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.lora.lora_router import LoRARouterRAG


def parse_conversation(conv_str: str) -> tuple[str, str]:
    """Parse conversation string into (prompt, response) tuple."""
    try:
        # Parse the conversation list
        conv = ast.literal_eval(conv_str)

        prompts = []
        responses = []

        for turn in conv:
            content = turn.get('content', '')
            role = turn.get('role', '')

            if role == 'user':
                prompts.append(content)
            elif role == 'assistant':
                responses.append(content)

        prompt_text = '\n'.join(prompts)
        response_text = '\n'.join(responses)

        return prompt_text, response_text
    except:
        # Fallback: treat first line as prompt, rest as response
        lines = conv_str.strip().split('\n', 1)
        if len(lines) == 2:
            return lines[0], lines[1]
        return conv_str, conv_str


def load_test_samples(cluster_dir: Path,
                      tokenizer,
                      max_samples: int = 50,
                      min_tokens: int = 200) -> list[dict]:
    """Load test samples with parsed prompt/response."""
    path = cluster_dir / "test.json"
    with open(path, 'r') as f:
        data = json.load(f)

    samples = []

    for sample in data.get('samples', []):
        conv_str = sample.get('conversation', '')
        prompt, response = parse_conversation(conv_str)
        full_text = f"{prompt}\n{response}"

        # Filter by minimum token count on the response (what we compress)
        num_tokens = len(tokenizer.encode(response, add_special_tokens=False))
        if num_tokens >= min_tokens:
            samples.append({
                'prompt': prompt,
                'response': response,
                'full': full_text,
                'num_tokens': num_tokens,
            })

    if max_samples and max_samples < len(samples):
        random.seed(42)
        samples = random.sample(samples, max_samples)

    return samples


def evaluate_routing_strategies(
    router: LoRARouterRAG,
    clusters_root: Path,
    tokenizer,
    cluster_ids: list[int],
    max_samples: int = 50,
    min_tokens: int = 200,
    k: int = 10,
) -> dict:
    """Evaluate routing accuracy for different text selection strategies."""

    strategies = ['prompt', 'response', 'full']

    results = {
        strategy: {
            'per_cluster': {},
            'total_correct': 0,
            'total_samples': 0,
        }
        for strategy in strategies
    }

    # Also track per-sample results for analysis
    all_samples = []

    for cluster_id in tqdm(cluster_ids, desc="Evaluating routing strategies"):
        cluster_dir = clusters_root / f"cluster_{cluster_id:03d}"
        samples = load_test_samples(cluster_dir, tokenizer, max_samples,
                                    min_tokens)

        if not samples:
            print(f"  Skipping cluster {cluster_id}: no samples")
            continue

        # Initialize per-cluster results
        for strategy in strategies:
            results[strategy]['per_cluster'][cluster_id] = {
                'correct': 0,
                'total': len(samples),
                'predictions': [],
            }

        for sample in samples:
            sample_result = {
                'cluster_id': cluster_id,
                'prompt_len': len(sample['prompt']),
                'response_len': len(sample['response']),
                'num_tokens': sample['num_tokens'],
            }

            for strategy in strategies:
                text = sample[strategy]

                # Skip if text is too short
                if len(text.strip()) < 10:
                    predicted_cluster = -1  # Invalid
                else:
                    _, details = router.route(text, k=k, return_details=True)
                    predicted_cluster = details['best_cluster']

                is_correct = predicted_cluster == cluster_id

                results[strategy]['per_cluster'][cluster_id][
                    'predictions'].append(predicted_cluster)
                if is_correct:
                    results[strategy]['per_cluster'][cluster_id][
                        'correct'] += 1
                    results[strategy]['total_correct'] += 1

                sample_result[f'{strategy}_prediction'] = predicted_cluster
                sample_result[f'{strategy}_correct'] = is_correct

            results['prompt']['total_samples'] += 1
            results['response']['total_samples'] += 1
            results['full']['total_samples'] += 1

            all_samples.append(sample_result)

        # Print per-cluster results
        print(f"  Cluster {cluster_id}:")
        for strategy in strategies:
            acc = results[strategy]['per_cluster'][cluster_id][
                'correct'] / len(samples) * 100
            print(f"    {strategy:>10}: {acc:5.1f}%")

    # Calculate overall accuracy
    for strategy in strategies:
        total = results[strategy]['total_samples']
        correct = results[strategy]['total_correct']
        results[strategy][
            'overall_accuracy'] = correct / total if total > 0 else 0

    return results, all_samples


def main():
    parser = argparse.ArgumentParser(
        description="Compare RAG routing strategies")
    parser.add_argument("--dataset",
                        type=str,
                        choices=['lmsys', 'wildchat'],
                        default='lmsys')
    parser.add_argument("--num-clusters", type=int, default=10)
    parser.add_argument("--cluster-ids", type=int, nargs='+', default=None)
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--min-tokens", type=int, default=200)
    parser.add_argument("--k",
                        type=int,
                        default=10,
                        help="Number of neighbors for RAG routing")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    # Set paths based on dataset
    if args.dataset == 'lmsys':
        clusters_root = Path(
            "/n/netscratch/sham_lab/Lab/rrinberg/compression/lmsys-clustered/clusters"
        )
        index_path = Path(
            "/n/netscratch/sham_lab/Lab/rrinberg/compression/lmsys-clustered/lora_rag_index"
        )
    else:
        clusters_root = Path(
            "/n/netscratch/sham_lab/Lab/rrinberg/compression/wildchat-clustered/clusters"
        )
        index_path = Path(
            "/n/netscratch/sham_lab/Lab/rrinberg/compression/wildchat-clustered/lora_rag_index"
        )

    output_dir = Path(args.output_dir) if args.output_dir else Path(
        f"results/rag_routing_strategies_{args.dataset}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine cluster IDs
    if args.cluster_ids:
        cluster_ids = args.cluster_ids
    else:
        cluster_ids = list(range(args.num_clusters))

    print("=" * 60)
    print(f"RAG Routing Strategy Comparison")
    print(f"Dataset: {args.dataset}")
    print(f"Clusters: {cluster_ids}")
    print(f"Max samples per cluster: {args.max_samples}")
    print(f"k (neighbors): {args.k}")
    print("=" * 60)

    # Initialize router
    print("\nLoading RAG router...")
    router = LoRARouterRAG(
        index_path=str(index_path),
        clusters_root=str(clusters_root),
    )
    router.load_index(str(index_path))

    # Get tokenizer for filtering
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2")

    # Run evaluation
    print("\n" + "=" * 60)
    print("Evaluating Routing Strategies")
    print("=" * 60)

    results, all_samples = evaluate_routing_strategies(
        router=router,
        clusters_root=clusters_root,
        tokenizer=tokenizer,
        cluster_ids=cluster_ids,
        max_samples=args.max_samples,
        min_tokens=args.min_tokens,
        k=args.k,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n{'Strategy':<15} {'Accuracy':>10} {'Correct':>10} {'Total':>10}")
    print("-" * 50)
    for strategy in ['prompt', 'response', 'full']:
        acc = results[strategy]['overall_accuracy'] * 100
        correct = results[strategy]['total_correct']
        total = results[strategy]['total_samples']
        print(f"{strategy:<15} {acc:>9.1f}% {correct:>10} {total:>10}")

    # Analyze disagreements
    print("\n" + "=" * 60)
    print("DISAGREEMENT ANALYSIS")
    print("=" * 60)

    prompt_response_agree = sum(
        1 for s in all_samples
        if s['prompt_prediction'] == s['response_prediction'])
    prompt_correct_response_wrong = sum(
        1 for s in all_samples
        if s['prompt_correct'] and not s['response_correct'])
    response_correct_prompt_wrong = sum(
        1 for s in all_samples
        if s['response_correct'] and not s['prompt_correct'])
    both_correct = sum(1 for s in all_samples
                       if s['prompt_correct'] and s['response_correct'])
    both_wrong = sum(1 for s in all_samples
                     if not s['prompt_correct'] and not s['response_correct'])

    total = len(all_samples)
    print(
        f"Prompt & Response agree: {prompt_response_agree}/{total} ({prompt_response_agree/total*100:.1f}%)"
    )
    print(
        f"Both correct: {both_correct}/{total} ({both_correct/total*100:.1f}%)"
    )
    print(f"Both wrong: {both_wrong}/{total} ({both_wrong/total*100:.1f}%)")
    print(
        f"Prompt correct, Response wrong: {prompt_correct_response_wrong}/{total} ({prompt_correct_response_wrong/total*100:.1f}%)"
    )
    print(
        f"Response correct, Prompt wrong: {response_correct_prompt_wrong}/{total} ({response_correct_prompt_wrong/total*100:.1f}%)"
    )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"routing_strategies_{timestamp}.json"

    output_data = {
        'timestamp': timestamp,
        'config': {
            'dataset': args.dataset,
            'cluster_ids': cluster_ids,
            'max_samples': args.max_samples,
            'min_tokens': args.min_tokens,
            'k': args.k,
        },
        'summary': {
            strategy: {
                'overall_accuracy': results[strategy]['overall_accuracy'],
                'total_correct': results[strategy]['total_correct'],
                'total_samples': results[strategy]['total_samples'],
            }
            for strategy in ['prompt', 'response', 'full']
        },
        'per_cluster': {
            strategy: {
                str(cid): {
                    'accuracy':
                    results[strategy]['per_cluster'][cid]['correct'] /
                    results[strategy]['per_cluster'][cid]['total'],
                    'correct':
                    results[strategy]['per_cluster'][cid]['correct'],
                    'total':
                    results[strategy]['per_cluster'][cid]['total'],
                }
                for cid in results[strategy]['per_cluster']
            }
            for strategy in ['prompt', 'response', 'full']
        },
        'disagreement_analysis': {
            'prompt_response_agree': prompt_response_agree,
            'both_correct': both_correct,
            'both_wrong': both_wrong,
            'prompt_correct_response_wrong': prompt_correct_response_wrong,
            'response_correct_prompt_wrong': response_correct_prompt_wrong,
        },
        'samples': all_samples,
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
