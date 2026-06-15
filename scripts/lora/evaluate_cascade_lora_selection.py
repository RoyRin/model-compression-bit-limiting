#!/usr/bin/env python3
"""
Cascade LoRA Selection: RAG → Perplexity Refinement

Process:
1. Use RAG to retrieve top-N candidate LoRAs
2. Compute perplexity with each candidate LoRA
3. Select the LoRA with lowest perplexity

Evaluation compares compression with:
- Baseline (no LoRA)
- RAG-selected LoRA
- Cascade-context-selected LoRA (perplexity on prompt)
- Cascade-response-selected LoRA (perplexity on response)
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
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from compression.block_coder import BlockEmissionArithmeticCoder


def parse_conversation(conv_str: str) -> tuple[str, str]:
    """Parse conversation string into (prompt, response) tuple."""
    try:
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
        return '\n'.join(prompts), '\n'.join(responses)
    except:
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

        num_tokens = len(tokenizer.encode(response, add_special_tokens=False))
        if num_tokens >= min_tokens:
            samples.append({
                'prompt': prompt,
                'response': response,
                'full': f"{prompt}\n{response}",
                'num_tokens': num_tokens,
            })

    if max_samples and max_samples < len(samples):
        random.seed(42)
        samples = random.sample(samples, max_samples)

    return samples


def compute_perplexity(model,
                       tokenizer,
                       text: str,
                       device: str = "cuda") -> float:
    """Compute perplexity of text under the model."""
    if not text.strip():
        return float('inf')

    encodings = tokenizer(text,
                          return_tensors="pt",
                          truncation=True,
                          max_length=2048)
    input_ids = encodings.input_ids.to(device)

    if input_ids.shape[1] < 2:
        return float('inf')

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        neg_log_likelihood = outputs.loss

    return torch.exp(neg_log_likelihood).item()


def compute_compression_bpt(encoder: BlockEmissionArithmeticCoder,
                            text: str) -> float:
    """Compute bits per token for compression."""
    try:
        tokens = encoder.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) == 0:
            return float('inf')
        encoded_values, _ = encoder.encode(tokens)
        total_bits = len(encoded_values) * encoder.bit_precision
        return total_bits / len(tokens)
    except:
        return float('inf')


def evaluate_cascade_selection(
    router: LoRARouterRAG,
    base_model,
    tokenizer,
    clusters_root: Path,
    loras_root: Path,
    cluster_ids: list[int],
    top_n: int = 10,
    max_samples: int = 50,
    min_tokens: int = 200,
    k_rag: int = 10,
    bit_precision: int = 64,
    device: str = "cuda",
) -> dict:
    """
    Evaluate cascade LoRA selection.

    For each sample:
    1. RAG retrieves top-N candidate LoRAs
    2. Compute perplexity on context and response with each candidate
    3. Select best by context-perplexity and by response-perplexity
    4. Compress with baseline and each selected LoRA
    """

    # Track routing accuracy
    results = {
        'rag_only': {
            'correct': 0,
            'total': 0
        },
        'cascade_context': {
            'correct': 0,
            'total': 0
        },
        'cascade_response': {
            'correct': 0,
            'total': 0
        },
        'correct_in_topn': {
            'count': 0,
            'total': 0
        },
    }

    # Track compression results
    compression_results = {
        'baseline': [],
        'rag_selected': [],
        'cascade_context_selected': [],
        'cascade_response_selected': [],
    }

    all_samples = []
    lora_cache = {}
    encoder_cache = {}

    # Create baseline encoder (no LoRA)
    baseline_encoder = BlockEmissionArithmeticCoder(
        model=base_model,
        tokenizer=tokenizer,
        bit_precision=bit_precision,
        device=device,
    )

    def get_lora_model(cluster_id: int):
        """Load or retrieve cached LoRA model."""
        if cluster_id not in lora_cache:
            lora_path = loras_root / f"cluster_{cluster_id:03d}"
            if not lora_path.exists():
                return None
            model_with_lora = PeftModel.from_pretrained(
                base_model,
                lora_path,
                is_trainable=False,
            )
            model_with_lora.eval()
            lora_cache[cluster_id] = model_with_lora
        return lora_cache[cluster_id]

    def get_encoder(cluster_id: int):
        """Get encoder for a specific LoRA."""
        if cluster_id not in encoder_cache:
            model = get_lora_model(cluster_id)
            if model is None:
                return None
            encoder_cache[cluster_id] = BlockEmissionArithmeticCoder(
                model=model,
                tokenizer=tokenizer,
                bit_precision=bit_precision,
                device=device,
            )
        return encoder_cache[cluster_id]

    for cluster_id in tqdm(cluster_ids, desc="Evaluating clusters"):
        cluster_dir = clusters_root / f"cluster_{cluster_id:03d}"
        samples = load_test_samples(cluster_dir, tokenizer, max_samples,
                                    min_tokens)

        if not samples:
            print(f"  Skipping cluster {cluster_id}: no samples")
            continue

        for sample in tqdm(samples,
                           desc=f"  Cluster {cluster_id}",
                           leave=False):
            sample_result = {
                'cluster_id': cluster_id,
                'prompt_len': len(sample['prompt']),
                'response_len': len(sample['response']),
            }

            # Step 1: RAG retrieval to get top-N candidates
            _, rag_details = router.route(sample['full'],
                                          k=k_rag,
                                          return_details=True)

            # Get top-N unique clusters from RAG results
            cluster_votes = rag_details.get('cluster_votes', {})
            top_n_clusters = sorted(cluster_votes.keys(),
                                    key=lambda x: cluster_votes[x],
                                    reverse=True)[:top_n]

            rag_prediction = rag_details['best_cluster']
            sample_result['rag_prediction'] = rag_prediction
            sample_result['rag_correct'] = rag_prediction == cluster_id
            sample_result['top_n_clusters'] = top_n_clusters
            sample_result['correct_in_topn'] = cluster_id in top_n_clusters

            results['rag_only']['total'] += 1
            if rag_prediction == cluster_id:
                results['rag_only']['correct'] += 1

            results['correct_in_topn']['total'] += 1
            if cluster_id in top_n_clusters:
                results['correct_in_topn']['count'] += 1

            # Step 2: Compute perplexity with each top-N LoRA
            perplexities_context = {}
            perplexities_response = {}

            for cand_cluster in top_n_clusters:
                model = get_lora_model(cand_cluster)
                if model is None:
                    continue

                # Perplexity on context (prompt)
                ppl_context = compute_perplexity(model, tokenizer,
                                                 sample['prompt'], device)
                perplexities_context[cand_cluster] = ppl_context

                # Perplexity on response
                ppl_response = compute_perplexity(model, tokenizer,
                                                  sample['response'], device)
                perplexities_response[cand_cluster] = ppl_response

            # Step 3: Select best by each criterion
            best_by_context = None
            best_by_response = None

            if perplexities_context:
                best_by_context = min(perplexities_context.keys(),
                                      key=lambda x: perplexities_context[x])
                sample_result['cascade_context_prediction'] = best_by_context
                sample_result[
                    'cascade_context_correct'] = best_by_context == cluster_id
                results['cascade_context']['total'] += 1
                if best_by_context == cluster_id:
                    results['cascade_context']['correct'] += 1

            if perplexities_response:
                best_by_response = min(perplexities_response.keys(),
                                       key=lambda x: perplexities_response[x])
                sample_result['cascade_response_prediction'] = best_by_response
                sample_result[
                    'cascade_response_correct'] = best_by_response == cluster_id
                results['cascade_response']['total'] += 1
                if best_by_response == cluster_id:
                    results['cascade_response']['correct'] += 1

            # Step 4: Compress with baseline and each selected LoRA
            response_text = sample['response']

            # Baseline compression (no LoRA)
            baseline_bpt = compute_compression_bpt(baseline_encoder,
                                                   response_text)
            sample_result['baseline_bpt'] = baseline_bpt
            compression_results['baseline'].append(baseline_bpt)

            # RAG-selected LoRA compression
            rag_encoder = get_encoder(rag_prediction)
            if rag_encoder:
                rag_bpt = compute_compression_bpt(rag_encoder, response_text)
                sample_result['rag_selected_bpt'] = rag_bpt
                compression_results['rag_selected'].append(rag_bpt)

            # Cascade-context selected LoRA compression
            if best_by_context is not None:
                context_encoder = get_encoder(best_by_context)
                if context_encoder:
                    context_bpt = compute_compression_bpt(
                        context_encoder, response_text)
                    sample_result['cascade_context_bpt'] = context_bpt
                    compression_results['cascade_context_selected'].append(
                        context_bpt)

            # Cascade-response selected LoRA compression
            if best_by_response is not None:
                response_encoder = get_encoder(best_by_response)
                if response_encoder:
                    response_bpt = compute_compression_bpt(
                        response_encoder, response_text)
                    sample_result['cascade_response_bpt'] = response_bpt
                    compression_results['cascade_response_selected'].append(
                        response_bpt)

            sample_result['perplexities_context'] = {
                str(k): v
                for k, v in perplexities_context.items()
            }
            sample_result['perplexities_response'] = {
                str(k): v
                for k, v in perplexities_response.items()
            }

            all_samples.append(sample_result)

        # Print cluster summary
        cluster_samples = [
            s for s in all_samples if s['cluster_id'] == cluster_id
        ]
        if cluster_samples:
            avg_baseline = np.mean([
                s['baseline_bpt'] for s in cluster_samples
                if 'baseline_bpt' in s
            ])
            avg_rag = np.mean([
                s['rag_selected_bpt'] for s in cluster_samples
                if 'rag_selected_bpt' in s
            ])
            avg_cascade_resp = np.mean([
                s['cascade_response_bpt'] for s in cluster_samples
                if 'cascade_response_bpt' in s
            ])
            print(
                f"  Cluster {cluster_id}: Baseline={avg_baseline:.2f}, RAG={avg_rag:.2f}, Cascade-Resp={avg_cascade_resp:.2f} bpt"
            )

        # Clear some cache to manage memory
        if len(lora_cache) > 15:
            keys_to_remove = list(lora_cache.keys())[:-8]
            for k in keys_to_remove:
                del lora_cache[k]
                if k in encoder_cache:
                    del encoder_cache[k]
            torch.cuda.empty_cache()

    return results, compression_results, all_samples


def main():
    parser = argparse.ArgumentParser(
        description="Cascade LoRA Selection: RAG → Perplexity")
    parser.add_argument("--dataset",
                        type=str,
                        choices=['lmsys', 'wildchat'],
                        default='lmsys')
    parser.add_argument("--num-clusters", type=int, default=10)
    parser.add_argument("--cluster-ids", type=int, nargs='+', default=None)
    parser.add_argument("--max-samples", type=int, default=30)
    parser.add_argument("--min-tokens", type=int, default=200)
    parser.add_argument("--top-n",
                        type=int,
                        default=10,
                        help="Number of top candidates from RAG")
    parser.add_argument("--k-rag",
                        type=int,
                        default=10,
                        help="k for RAG nearest neighbors")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    # Set paths
    if args.dataset == 'lmsys':
        clusters_root = Path(
            "/n/netscratch/sham_lab/Lab/rrinberg/compression/lmsys-clustered/clusters"
        )
        loras_root = Path(
            "/n/netscratch/sham_lab/Lab/rrinberg/compression/lmsys-loras")
        index_path = Path(
            "/n/netscratch/sham_lab/Lab/rrinberg/compression/lmsys-clustered/lora_rag_index"
        )
    else:
        clusters_root = Path(
            "/n/netscratch/sham_lab/Lab/rrinberg/compression/wildchat-clustered/clusters"
        )
        loras_root = Path(
            "/n/netscratch/sham_lab/Lab/rrinberg/compression/wildchat-loras")
        index_path = Path(
            "/n/netscratch/sham_lab/Lab/rrinberg/compression/wildchat-clustered/lora_rag_index"
        )

    output_dir = Path(args.output_dir) if args.output_dir else Path(
        f"results/cascade_lora_selection_{args.dataset}")
    output_dir.mkdir(parents=True, exist_ok=True)

    cluster_ids = args.cluster_ids if args.cluster_ids else list(
        range(args.num_clusters))

    print("=" * 70)
    print("Cascade LoRA Selection: RAG → Perplexity Refinement")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Clusters: {cluster_ids}")
    print(f"Top-N from RAG: {args.top_n}")
    print(f"Max samples per cluster: {args.max_samples}")
    print("=" * 70)

    # Load models
    print("\nLoading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2")

    print("Loading RAG router...")
    router = LoRARouterRAG(
        index_path=str(index_path),
        clusters_root=str(clusters_root),
    )
    router.load_index(str(index_path))

    # Run evaluation
    print("\n" + "=" * 70)
    print("Running Evaluation")
    print("=" * 70)

    results, compression_results, all_samples = evaluate_cascade_selection(
        router=router,
        base_model=base_model,
        tokenizer=tokenizer,
        clusters_root=clusters_root,
        loras_root=loras_root,
        cluster_ids=cluster_ids,
        top_n=args.top_n,
        max_samples=args.max_samples,
        min_tokens=args.min_tokens,
        k_rag=args.k_rag,
    )

    # Print routing accuracy summary
    print("\n" + "=" * 70)
    print("ROUTING ACCURACY")
    print("=" * 70)

    print(f"\n{'Method':<25} {'Accuracy':>10} {'Correct':>10} {'Total':>10}")
    print("-" * 60)

    for method in [
            'rag_only', 'cascade_context', 'cascade_response',
            'correct_in_topn'
    ]:
        if method == 'correct_in_topn':
            total = results[method]['total']
            correct = results[method]['count']
        else:
            total = results[method]['total']
            correct = results[method]['correct']
        if total > 0:
            acc = correct / total * 100
            print(f"{method:<25} {acc:>9.1f}% {correct:>10} {total:>10}")

    # Print compression summary
    print("\n" + "=" * 70)
    print("COMPRESSION RESULTS (bits per token)")
    print("=" * 70)

    print(f"\n{'Method':<30} {'Avg BPT':>10} {'Std':>10} {'Count':>10}")
    print("-" * 65)

    for method in [
            'baseline', 'rag_selected', 'cascade_context_selected',
            'cascade_response_selected'
    ]:
        bpts = compression_results[method]
        if bpts:
            avg = np.mean(bpts)
            std = np.std(bpts)
            print(f"{method:<30} {avg:>10.3f} {std:>10.3f} {len(bpts):>10}")

    # Improvement over baseline
    print("\n" + "=" * 70)
    print("IMPROVEMENT OVER BASELINE")
    print("=" * 70)

    baseline_avg = np.mean(compression_results['baseline']
                           ) if compression_results['baseline'] else 0
    for method in [
            'rag_selected', 'cascade_context_selected',
            'cascade_response_selected'
    ]:
        bpts = compression_results[method]
        if bpts and baseline_avg > 0:
            avg = np.mean(bpts)
            improvement = (baseline_avg - avg) / baseline_avg * 100
            print(f"{method:<30}: {improvement:>6.1f}% improvement")

    # Analysis: When does cascade help?
    print("\n" + "=" * 70)
    print("ANALYSIS: When Does Cascade Help?")
    print("=" * 70)

    rag_wrong_cascade_context_right = sum(
        1 for s in all_samples if not s.get('rag_correct', False)
        and s.get('cascade_context_correct', False))
    rag_wrong_cascade_response_right = sum(
        1 for s in all_samples if not s.get('rag_correct', False)
        and s.get('cascade_response_correct', False))
    rag_right_cascade_context_wrong = sum(
        1 for s in all_samples if s.get('rag_correct', False)
        and not s.get('cascade_context_correct', False))
    rag_right_cascade_response_wrong = sum(
        1 for s in all_samples if s.get('rag_correct', False)
        and not s.get('cascade_response_correct', False))

    total = len(all_samples)
    print(
        f"RAG wrong, Cascade-Context right: {rag_wrong_cascade_context_right}/{total}"
    )
    print(
        f"RAG wrong, Cascade-Response right: {rag_wrong_cascade_response_right}/{total}"
    )
    print(
        f"RAG right, Cascade-Context wrong: {rag_right_cascade_context_wrong}/{total}"
    )
    print(
        f"RAG right, Cascade-Response wrong: {rag_right_cascade_response_wrong}/{total}"
    )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"cascade_selection_{timestamp}.json"

    # Build routing summary
    routing_summary = {}
    for method in results:
        if method == 'correct_in_topn':
            routing_summary[method] = {
                'rate':
                results[method]['count'] / results[method]['total']
                if results[method]['total'] > 0 else 0,
                'count':
                results[method]['count'],
                'total':
                results[method]['total'],
            }
        else:
            routing_summary[method] = {
                'accuracy':
                results[method]['correct'] / results[method]['total']
                if results[method]['total'] > 0 else 0,
                'correct':
                results[method]['correct'],
                'total':
                results[method]['total'],
            }

    # Build compression summary
    compression_summary = {}
    for method in compression_results:
        bpts = compression_results[method]
        if bpts:
            compression_summary[method] = {
                'avg_bpt': float(np.mean(bpts)),
                'std_bpt': float(np.std(bpts)),
                'count': len(bpts),
            }

    output_data = {
        'timestamp': timestamp,
        'config': {
            'dataset': args.dataset,
            'cluster_ids': cluster_ids,
            'top_n': args.top_n,
            'max_samples': args.max_samples,
            'min_tokens': args.min_tokens,
            'k_rag': args.k_rag,
        },
        'routing_summary': routing_summary,
        'compression_summary': compression_summary,
        'samples': all_samples,
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
