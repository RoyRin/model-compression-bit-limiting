#!/usr/bin/env python3
"""
Evaluate LoRA RAG routing and compression.

Two experiments:
1. RAG Accuracy: What % of the time does RAG pick the correct LoRA?
2. RAG Compression: Compress with RAG-selected LoRA and measure bits/token.
"""
import math
import argparse
import json
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
from datetime import datetime
from collections import Counter
import random

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from compression.block_coder import BlockEmissionArithmeticCoder
from scripts.lora.lora_router import LoRARouterRAG


def load_test_texts(cluster_dir: Path,
                    tokenizer,
                    max_samples: int = 50,
                    min_tokens: int = 200) -> list[str]:
    """Load test texts from a cluster, filtering by minimum token count."""
    path = cluster_dir / "test.json"
    with open(path, 'r') as f:
        data = json.load(f)

    texts = data['texts']
    total_before = len(texts)

    # Filter by minimum token count
    filtered_texts = []
    for t in texts:
        if len(t.strip()) > 0:
            num_tokens = len(tokenizer.encode(t, add_special_tokens=False))
            if num_tokens >= min_tokens:
                filtered_texts.append(t)

    if max_samples and max_samples < len(filtered_texts):
        random.seed(42)
        filtered_texts = random.sample(filtered_texts, max_samples)

    return filtered_texts


def evaluate_rag_accuracy(
    router: LoRARouterRAG,
    clusters_root: Path,
    tokenizer,
    cluster_ids: list[int],
    max_samples: int = 50,
    min_tokens: int = 200,
    k: int = 10,
) -> dict:
    """Evaluate RAG routing accuracy - what % picks the correct LoRA."""

    results = {
        'per_cluster': {},
        'total_correct': 0,
        'total_samples': 0,
    }

    for cluster_id in tqdm(cluster_ids, desc="Evaluating RAG accuracy"):
        cluster_dir = clusters_root / f"cluster_{cluster_id:03d}"
        texts = load_test_texts(cluster_dir, tokenizer, max_samples,
                                min_tokens)

        if not texts:
            print(f"  Skipping cluster {cluster_id}: no texts")
            continue

        correct = 0
        cluster_predictions = []

        for text in texts:
            lora_path, details = router.route(text, k=k, return_details=True)
            predicted_cluster = details['best_cluster']
            cluster_predictions.append(predicted_cluster)

            if predicted_cluster == cluster_id:
                correct += 1

        accuracy = correct / len(texts) if texts else 0
        # Convert numpy int keys to Python int for JSON serialization
        predictions_dict = {
            int(k): v
            for k, v in Counter(cluster_predictions).items()
        }
        results['per_cluster'][int(cluster_id)] = {
            'accuracy': accuracy,
            'correct': correct,
            'total': len(texts),
            'predictions': predictions_dict,
        }
        results['total_correct'] += correct
        results['total_samples'] += len(texts)

        print(
            f"  Cluster {cluster_id}: {accuracy*100:.1f}% ({correct}/{len(texts)})"
        )

    results['overall_accuracy'] = results['total_correct'] / results[
        'total_samples'] if results['total_samples'] > 0 else 0
    return results


def compress_text(encoder: BlockEmissionArithmeticCoder,
                  text: str,
                  use_prefill: bool = False) -> tuple[float, int, int, float]:
    """Compress a text and return bits per token and compression ratio."""
    try:
        tokens = encoder.tokenizer.encode(text, add_special_tokens=False)
        num_tokens = len(tokens)
        vocab_size = encoder.tokenizer.vocab_size
        default_bits_in_a_single_token = math.log2(vocab_size)
        if num_tokens == 0:
            return 0.0, 0, 0, 0.0

        encoded_values, encoding_info = encoder.encode(tokens,
                                                       use_prefill=use_prefill)
        total_bits = len(encoded_values) * (encoder.bit_precision +
                                            encoder.bits_for_encoding_count)
        bits_per_token = total_bits / num_tokens
        compression_ratio = total_bits / (num_tokens *
                                          default_bits_in_a_single_token)
        return bits_per_token, num_tokens, total_bits, compression_ratio

    except Exception as e:
        print(f"    Error compressing: {e}")
        return None, None, None, None


def evaluate_rag_compression(
    router: LoRARouterRAG,
    clusters_root: Path,
    loras_root: Path,
    base_model,
    tokenizer,
    cluster_ids: list[int],
    max_samples: int = 50,
    min_tokens: int = 200,
    bit_precision: int = 64,
    bits_for_encoding_count=7,
    k: int = 10,
    device: str = "cuda",
    with_correct_lora: bool = False,
    output_dir: Path = None,
    use_prefill: bool = False,
) -> dict:
    """Evaluate compression using RAG-selected LoRA, with baseline comparison.

    If with_correct_lora=True, also compresses with the oracle/correct LoRA for each cluster.

    OPTIMIZED: Batches samples by predicted cluster to avoid repeated LoRA loading.
    """

    results = {
        'per_cluster': {},
        'all_bpt': [],  # RAG bits per token across all samples
        'all_baseline_bpt': [],  # Baseline (no LoRA) bits per token
    }
    if with_correct_lora:
        results['all_correct_bpt'] = []  # Correct LoRA bits per token

    for cluster_id in tqdm(cluster_ids, desc="Evaluating RAG compression"):
        cluster_dir = clusters_root / f"cluster_{cluster_id:03d}"
        texts = load_test_texts(cluster_dir, tokenizer, max_samples,
                                min_tokens)

        if not texts:
            print(f"  Skipping cluster {cluster_id}: no texts")
            continue

        # --- Baseline compression (no LoRA) ---
        print(f"  Cluster {cluster_id}: computing baseline compression...")
        baseline_encoder = BlockEmissionArithmeticCoder(
            model=base_model,
            tokenizer=tokenizer,
            bit_precision=bit_precision,
            device=device,
            verbose=False,
        )
        baseline_results_list = []
        for text in tqdm(texts, desc=f"C{cluster_id} baseline", leave=False):
            bpt, n_tokens, total_bits, comp_ratio = compress_text(
                baseline_encoder, text, use_prefill=use_prefill)
            baseline_results_list.append(
                (bpt, n_tokens, total_bits, comp_ratio))
        del baseline_encoder
        torch.cuda.empty_cache()
        valid_baseline = [
            b for b, _, _, _ in baseline_results_list if b is not None
        ]
        if valid_baseline:
            print(
                f"  Cluster {cluster_id} baseline done: {np.mean(valid_baseline):.4f} bpt (n={len(valid_baseline)})"
            )
            sys.stdout.flush()

        # --- Correct LoRA compression (oracle) ---
        correct_results_list = [None] * len(texts)
        if with_correct_lora:
            correct_lora_dir = loras_root / f"cluster_{cluster_id:03d}"
            if correct_lora_dir.exists():
                print(
                    f"  Cluster {cluster_id}: computing correct LoRA compression..."
                )
                correct_model = PeftModel.from_pretrained(
                    base_model, correct_lora_dir)
                correct_model.eval()
                correct_encoder = BlockEmissionArithmeticCoder(
                    model=correct_model,
                    tokenizer=tokenizer,
                    bit_precision=bit_precision,
                    bits_for_encoding_count=bits_for_encoding_count,
                    device=device,
                    verbose=False,
                )
                for idx, text in enumerate(
                        tqdm(texts,
                             desc=f"C{cluster_id} correct LoRA",
                             leave=False)):
                    bpt, n_tokens, total_bits, comp_ratio = compress_text(
                        correct_encoder, text, use_prefill=use_prefill)
                    correct_results_list[idx] = (bpt, n_tokens, total_bits,
                                                 comp_ratio)
                del correct_encoder
                base_model = correct_model.unload()
                del correct_model
                torch.cuda.empty_cache()
                valid_correct = [
                    b for b, _, _, _ in correct_results_list if b is not None
                ]
                if valid_correct:
                    print(
                        f"  Cluster {cluster_id} correct LoRA done: {np.mean(valid_correct):.4f} bpt (n={len(valid_correct)})"
                    )
                    sys.stdout.flush()
            else:
                print(
                    f"  Cluster {cluster_id}: correct LoRA not found at {correct_lora_dir}"
                )

        # --- RAG-selected LoRA compression (BATCHED by predicted cluster) ---
        print(f"  Cluster {cluster_id}: routing all samples...")

        # Step 1: Route ALL samples first to get predictions
        predictions = []  # (idx, text, predicted_cluster)
        for idx, text in enumerate(
                tqdm(texts, desc=f"C{cluster_id} routing", leave=False)):
            lora_path, details = router.route(text, k=k, return_details=True)
            predicted_cluster = details['best_cluster']
            predictions.append((idx, text, predicted_cluster))

        # Step 2: Group by predicted cluster
        from collections import defaultdict
        samples_by_pred_cluster = defaultdict(list)
        for idx, text, pred_cluster in predictions:
            samples_by_pred_cluster[pred_cluster].append((idx, text))

        unique_preds = sorted(samples_by_pred_cluster.keys())
        print(
            f"  Cluster {cluster_id}: {len(texts)} samples routed to {len(unique_preds)} unique LoRAs: {unique_preds}"
        )
        sys.stdout.flush()

        # Step 3: Process each predicted cluster batch (load LoRA once per batch)
        rag_results_list = [None] * len(texts)  # Will fill in by idx
        correct_lora_count = sum(1 for _, _, pc in predictions
                                 if pc == cluster_id)

        for pred_cluster in tqdm(unique_preds,
                                 desc=f"C{cluster_id} RAG batches",
                                 leave=False):
            samples = samples_by_pred_cluster[pred_cluster]

            # Load this LoRA once
            lora_dir = loras_root / f"cluster_{pred_cluster:03d}"
            if lora_dir.exists():
                lora_model = PeftModel.from_pretrained(base_model, lora_dir)
                lora_model.eval()
                model_to_use = lora_model
            else:
                print(f"    LoRA not found: {lora_dir}")
                model_to_use = base_model
                lora_model = None

            # Create encoder once for this batch
            encoder = BlockEmissionArithmeticCoder(
                model=model_to_use,
                tokenizer=tokenizer,
                bit_precision=bit_precision,
                bits_for_encoding_count=bits_for_encoding_count,
                device=device,
                verbose=False,
            )

            # Compress all samples for this predicted cluster
            for idx, text in samples:
                bpt, n_tokens, total_bits, comp_ratio = compress_text(
                    encoder, text, use_prefill=use_prefill)
                rag_results_list[idx] = (bpt, n_tokens, total_bits, comp_ratio,
                                         pred_cluster)

            del encoder

            # Unload LoRA
            if lora_model is not None:
                base_model = lora_model.unload()
                del lora_model
            torch.cuda.empty_cache()

        # Step 4: Aggregate results
        cluster_results = []
        for idx in range(len(texts)):
            baseline_bpt, baseline_tokens, baseline_bits, baseline_ratio = baseline_results_list[
                idx]
            rag_bpt, num_tokens, total_bits, rag_ratio, predicted_cluster = rag_results_list[
                idx]

            if rag_bpt is not None and baseline_bpt is not None:
                sample_result = {
                    'rag_bpt':
                    float(rag_bpt) if hasattr(rag_bpt, 'item') else rag_bpt,
                    'rag_compression_ratio':
                    float(rag_ratio)
                    if rag_ratio and hasattr(rag_ratio, 'item') else rag_ratio,
                    'baseline_bpt':
                    float(baseline_bpt)
                    if hasattr(baseline_bpt, 'item') else baseline_bpt,
                    'baseline_compression_ratio':
                    float(baseline_ratio) if baseline_ratio
                    and hasattr(baseline_ratio, 'item') else baseline_ratio,
                    'tokens':
                    int(num_tokens) if num_tokens else num_tokens,
                    'rag_bits':
                    int(total_bits) if total_bits else total_bits,
                    'baseline_bits':
                    int(baseline_bits) if baseline_bits else baseline_bits,
                    'predicted_cluster':
                    int(predicted_cluster),
                    'correct_prediction':
                    bool(predicted_cluster == cluster_id),
                }
                if with_correct_lora and correct_results_list[idx] is not None:
                    c_bpt, c_tokens, c_bits, c_ratio = correct_results_list[
                        idx]
                    if c_bpt is not None:
                        sample_result['correct_bpt'] = float(c_bpt) if hasattr(
                            c_bpt, 'item') else c_bpt
                        sample_result['correct_compression_ratio'] = float(
                            c_ratio) if c_ratio and hasattr(
                                c_ratio, 'item') else c_ratio
                        sample_result['correct_bits'] = int(
                            c_bits) if c_bits else c_bits
                        results['all_correct_bpt'].append(c_bpt)

                cluster_results.append(sample_result)
                results['all_bpt'].append(rag_bpt)
                results['all_baseline_bpt'].append(baseline_bpt)

        if cluster_results:
            avg_rag_bpt = np.mean([r['rag_bpt'] for r in cluster_results])
            avg_baseline_bpt = np.mean(
                [r['baseline_bpt'] for r in cluster_results])
            cluster_summary = {
                'avg_rag_bpt': float(avg_rag_bpt),
                'avg_baseline_bpt': float(avg_baseline_bpt),
                'num_samples': len(cluster_results),
                'correct_lora_pct': correct_lora_count / len(texts) * 100,
                'results': cluster_results,
            }
            if with_correct_lora:
                correct_bpts = [
                    r['correct_bpt'] for r in cluster_results
                    if 'correct_bpt' in r
                ]
                if correct_bpts:
                    cluster_summary['avg_correct_bpt'] = float(
                        np.mean(correct_bpts))

            results['per_cluster'][int(cluster_id)] = cluster_summary

            msg = f"  Cluster {cluster_id}: RAG={avg_rag_bpt:.4f} bpt, Baseline={avg_baseline_bpt:.4f} bpt"
            if with_correct_lora and 'avg_correct_bpt' in cluster_summary:
                msg += f", Correct={cluster_summary['avg_correct_bpt']:.4f} bpt"
            msg += f", {correct_lora_count}/{len(texts)} correct routing"
            print(msg)
            sys.stdout.flush()

            # Save intermediate results after each cluster
            if output_dir is not None:
                intermediate = {
                    'status': 'in_progress',
                    'completed_clusters': list(results['per_cluster'].keys()),
                    'per_cluster': {
                        int(k): {
                            'avg_rag_bpt':
                            float(v['avg_rag_bpt']),
                            'avg_baseline_bpt':
                            float(v['avg_baseline_bpt']),
                            'num_samples':
                            v['num_samples'],
                            'correct_lora_pct':
                            float(v['correct_lora_pct']),
                            **(({
                                'avg_correct_bpt': float(v['avg_correct_bpt'])
                            } if 'avg_correct_bpt' in v else {})),
                        }
                        for k, v in results['per_cluster'].items()
                    },
                }
                intermediate_path = output_dir / "intermediate_compression.json"
                with open(intermediate_path, 'w') as f:
                    json.dump(intermediate, f, indent=2)
                print(f"  Saved intermediate results to {intermediate_path}")
                sys.stdout.flush()

    results['overall_avg_bpt'] = np.mean(
        results['all_bpt']) if results['all_bpt'] else 0
    results['overall_avg_baseline_bpt'] = np.mean(
        results['all_baseline_bpt']) if results['all_baseline_bpt'] else 0
    if with_correct_lora:
        results['overall_avg_correct_bpt'] = np.mean(
            results['all_correct_bpt']) if results['all_correct_bpt'] else 0
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LoRA RAG routing and compression")

    parser.add_argument(
        "--clusters-root",
        type=str,
        default=
        "/n/netscratch/sham_lab/Lab/rrinberg/compression/lmsys-clustered/clusters"
    )
    parser.add_argument(
        "--loras-root",
        type=str,
        default="/n/netscratch/sham_lab/Lab/rrinberg/compression/lmsys-loras")
    parser.add_argument(
        "--index-path",
        type=str,
        default=
        "/n/netscratch/sham_lab/Lab/rrinberg/compression/lmsys-clustered/lora_rag_index"
    )
    parser.add_argument("--base-model",
                        type=str,
                        default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--num-clusters", type=int, default=10)
    parser.add_argument("--cluster-ids",
                        type=str,
                        default=None,
                        help="Comma-separated cluster IDs")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--min-tokens", type=int, default=200)
    parser.add_argument("--k",
                        type=int,
                        default=10,
                        help="Number of neighbors for RAG voting")
    parser.add_argument("--bit-precision", type=int, default=64)
    parser.add_argument("--output-dir",
                        type=str,
                        default="results/rag_lora_evaluation")
    parser.add_argument("--plot-dir",
                        type=str,
                        default="writing/695fe28d3a9ed52bd3824bba/assets/plts",
                        help="Output directory for plots")
    parser.add_argument("--plot-format",
                        type=str,
                        choices=["png", "pdf"],
                        default="pdf",
                        help="Output format for plots")

    # Experiment selection
    parser.add_argument("--accuracy-only",
                        action="store_true",
                        help="Only run accuracy experiment (no compression)")
    parser.add_argument("--compression-only",
                        action="store_true",
                        help="Only run compression experiment")
    parser.add_argument(
        "--with-correct-lora",
        action="store_true",
        help="Also compute correct/oracle LoRA compression for comparison")
    parser.add_argument(
        "--use-prefill",
        action="store_true",
        help=
        "Use prefill mode for faster compression (batch processes all tokens)")

    args = parser.parse_args()

    clusters_root = Path(args.clusters_root)
    loras_root = Path(args.loras_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Determine cluster IDs
    if args.cluster_ids:
        cluster_ids = [int(x.strip()) for x in args.cluster_ids.split(",")]
    else:
        cluster_ids = list(range(args.num_clusters))

    print(f"Cluster IDs: {cluster_ids}")
    print(f"Index path: {args.index_path}")
    print(f"Device: {device}")
    print("=" * 60)

    # Load tokenizer for filtering
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model,
                                              trust_remote_code=True)

    # Load RAG router
    print("Loading RAG router...")
    router = LoRARouterRAG(
        loras_root=str(loras_root),
        clusters_root=str(clusters_root),
    )
    router.load_index(args.index_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = {'timestamp': timestamp, 'config': vars(args)}

    # Experiment 1: RAG Accuracy
    if not args.compression_only:
        print("\n" + "=" * 60)
        print("EXPERIMENT 1: RAG Routing Accuracy")
        print("=" * 60)

        accuracy_results = evaluate_rag_accuracy(
            router=router,
            clusters_root=clusters_root,
            tokenizer=tokenizer,
            cluster_ids=cluster_ids,
            max_samples=args.max_samples,
            min_tokens=args.min_tokens,
            k=args.k,
        )

        print(
            f"\nOverall RAG Accuracy: {accuracy_results['overall_accuracy']*100:.2f}%"
        )
        print(
            f"Total: {accuracy_results['total_correct']}/{accuracy_results['total_samples']} correct"
        )

        all_results['accuracy'] = accuracy_results

    # Experiment 2: RAG Compression
    if not args.accuracy_only:
        print("\n" + "=" * 60)
        print("EXPERIMENT 2: RAG-Selected LoRA Compression")
        print("=" * 60)

        # Load base model
        print(f"Loading base model: {args.base_model}")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        base_model.eval()

        compression_results = evaluate_rag_compression(
            router=router,
            clusters_root=clusters_root,
            loras_root=loras_root,
            base_model=base_model,
            tokenizer=tokenizer,
            cluster_ids=cluster_ids,
            max_samples=args.max_samples,
            min_tokens=args.min_tokens,
            bit_precision=args.bit_precision,
            k=args.k,
            device=device,
            with_correct_lora=args.with_correct_lora,
            output_dir=output_dir,
            use_prefill=args.use_prefill,
        )

        print(
            f"\nOverall RAG Compression: {compression_results['overall_avg_bpt']:.4f} bits/token"
        )
        print(
            f"Overall Baseline Compression: {compression_results['overall_avg_baseline_bpt']:.4f} bits/token"
        )
        print(
            f"RAG vs Baseline: {compression_results['overall_avg_bpt'] - compression_results['overall_avg_baseline_bpt']:.4f} bpt difference"
        )
        if args.with_correct_lora and 'overall_avg_correct_bpt' in compression_results:
            print(
                f"Overall Correct LoRA Compression: {compression_results['overall_avg_correct_bpt']:.4f} bits/token"
            )
            print(
                f"Correct vs Baseline: {compression_results['overall_avg_correct_bpt'] - compression_results['overall_avg_baseline_bpt']:.4f} bpt difference"
            )

        # Save full results including per-sample data for plotting
        compression_summary = {
            'overall_avg_rag_bpt':
            float(compression_results['overall_avg_bpt']),
            'overall_avg_baseline_bpt':
            float(compression_results['overall_avg_baseline_bpt']),
            'per_cluster': {
                int(k): {
                    'avg_rag_bpt':
                    float(v['avg_rag_bpt']),
                    'avg_baseline_bpt':
                    float(v['avg_baseline_bpt']),
                    'num_samples':
                    v['num_samples'],
                    'correct_lora_pct':
                    float(v['correct_lora_pct']),
                    'results':
                    v['results'],  # Keep per-sample data for plotting
                    **(({
                        'avg_correct_bpt': float(v['avg_correct_bpt'])
                    } if 'avg_correct_bpt' in v else {})),
                }
                for k, v in compression_results['per_cluster'].items()
            }
        }
        if 'overall_avg_correct_bpt' in compression_results:
            compression_summary['overall_avg_correct_bpt'] = float(
                compression_results['overall_avg_correct_bpt'])
        all_results['compression'] = compression_summary

    # Save results
    json_path = output_dir / f"rag_lora_results_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved results to {json_path}")

    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if 'accuracy' in all_results:
        print(
            f"\nRAG Routing Accuracy: {all_results['accuracy']['overall_accuracy']*100:.2f}%"
        )
        print(f"\nPer-cluster accuracy:")
        for cid in cluster_ids:
            if cid in all_results['accuracy']['per_cluster']:
                acc = all_results['accuracy']['per_cluster'][cid][
                    'accuracy'] * 100
                print(f"  Cluster {cid}: {acc:.1f}%")

    if 'compression' in all_results:
        comp = all_results['compression']
        print(f"\nBaseline: {comp['overall_avg_baseline_bpt']:.4f} bits/token")
        print(f"RAG LoRA: {comp['overall_avg_rag_bpt']:.4f} bits/token")
        if 'overall_avg_correct_bpt' in comp:
            print(
                f"Correct LoRA: {comp['overall_avg_correct_bpt']:.4f} bits/token"
            )
        print(f"\nPer-cluster compression:")
        header = f"  {'Cluster':>8} {'Baseline':>10} {'RAG':>10}"
        if 'overall_avg_correct_bpt' in comp:
            header += f" {'Correct':>10}"
        header += f" {'RAG Routing':>12}"
        print(header)
        print(f"  {'-'*52}")
        for cid in cluster_ids:
            cid_key = cid if cid in comp['per_cluster'] else str(cid)
            if cid_key in comp['per_cluster']:
                pc = comp['per_cluster'][cid_key]
                line = f"  {cid:>8} {pc['avg_baseline_bpt']:>10.4f} {pc['avg_rag_bpt']:>10.4f}"
                if 'avg_correct_bpt' in pc:
                    line += f" {pc['avg_correct_bpt']:>10.4f}"
                line += f" {pc['correct_lora_pct']:>10.1f}%"
                print(line)


if __name__ == "__main__":
    main()
