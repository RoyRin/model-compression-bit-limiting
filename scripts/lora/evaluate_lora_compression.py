#!/usr/bin/env python3
"""
Evaluate LoRA compression across multiple clusters using actual arithmetic coding.

Compares compression ratio for:
1. No LoRA (baseline)
2. Correct LoRA (trained on same cluster)
3. Wrong LoRA (trained on different cluster)

Generates:
- Grouped bar chart showing bits/token for each condition per cluster
- Summary table with averages across all clusters
"""

import argparse
import gzip
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
import random

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from compression.block_coder import BlockEmissionArithmeticCoder


def load_cluster_texts(cluster_dir: Path,
                       tokenizer,
                       split: str = "test",
                       max_samples: int = None,
                       min_tokens: int = 200) -> list[str]:
    """Load texts from a cluster, filtering by minimum token count."""
    path = cluster_dir / f"{split}.json"
    with open(path, 'r') as f:
        data = json.load(f)

    texts = data['texts']
    total_before = len(texts)

    # Filter by minimum token count
    filtered_texts = []
    for t in texts:
        if len(t.strip()) > 0:
            num_tokens = len(tokenizer.encode(t, add_special_tokens=False))
            if num_tokens > min_tokens:
                filtered_texts.append(t)

    print(
        f"    Found {len(filtered_texts)}/{total_before} texts with >{min_tokens} tokens"
    )

    texts = filtered_texts
    if max_samples and max_samples < len(texts):
        random.seed(42)
        texts = random.sample(texts, max_samples)
        print(f"    Sampled {max_samples} texts")

    return texts


def create_encoder(model,
                   tokenizer,
                   device: str,
                   bit_precision: int = 64) -> BlockEmissionArithmeticCoder:
    """Create an arithmetic coder for a model."""
    return BlockEmissionArithmeticCoder(
        model=model,
        tokenizer=tokenizer,
        bit_precision=bit_precision,
        bits_for_encoding_count=8,
        device=device,
        verbose=False,
    )


def compute_ce_bits_per_token(model, tokenizer, text: str,
                              device: str) -> tuple[float, int, float]:
    """Compute bits per token using cross-entropy loss (fast, theoretical limit).

    Returns:
        Tuple of (bits_per_token, num_tokens, total_bits)
    """
    try:
        tokens = tokenizer.encode(text,
                                  add_special_tokens=False,
                                  return_tensors="pt").to(device)
        num_tokens = tokens.shape[1]

        if num_tokens == 0:
            return 0.0, 0, 0.0

        with torch.no_grad():
            outputs = model(tokens, labels=tokens)
            ce_loss = outputs.loss.item()  # nats per token

        # Convert nats to bits: bits = nats / ln(2)
        bits_per_token = ce_loss / np.log(2)
        total_bits = bits_per_token * num_tokens

        return bits_per_token, num_tokens, total_bits

    except Exception as e:
        print(f"    Error computing CE: {e}")
        return None, None, None


def compress_text(encoder: BlockEmissionArithmeticCoder,
                  text: str) -> tuple[float, int, int]:
    """Compress a text using actual arithmetic coding.

    Returns:
        Tuple of (bits_per_token, num_tokens, total_bits)
    """
    try:
        # Tokenize the text first
        tokens = encoder.tokenizer.encode(text, add_special_tokens=False)
        num_tokens = len(tokens)

        if num_tokens == 0:
            return 0.0, 0, 0

        # Encode the tokens
        encoded_values, encoding_info = encoder.encode(tokens)
        # Each encoded value is bit_precision bits
        total_bits = len(encoded_values) * encoder.bit_precision

        bits_per_token = total_bits / num_tokens
        return bits_per_token, num_tokens, total_bits

    except Exception as e:
        print(f"    Error compressing text: {e}")
        return None, None, None


def compute_gzip_bits_per_token(text: str,
                                tokenizer) -> tuple[float, int, int]:
    """Compute bits per token using gzip compression.

    Returns:
        Tuple of (bits_per_token, num_tokens, total_bits)
    """
    try:
        # Get number of tokens
        tokens = tokenizer.encode(text, add_special_tokens=False)
        num_tokens = len(tokens)

        if num_tokens == 0:
            return 0.0, 0, 0

        # Compress the text with gzip
        text_bytes = text.encode('utf-8')
        compressed = gzip.compress(text_bytes, compresslevel=9)
        total_bits = len(compressed) * 8

        bits_per_token = total_bits / num_tokens
        return bits_per_token, num_tokens, total_bits

    except Exception as e:
        print(f"    Error computing gzip: {e}")
        return None, None, None


def evaluate_cluster_pairwise(
    cluster_id: int,
    clusters_root: Path,
    loras_root: Path,
    base_model_obj,
    tokenizer,
    all_cluster_ids: list[int],
    max_samples: int = 50,
    min_tokens: int = 200,
    split: str = "test",
    bit_precision: int = 64,
    device: str = "cuda",
    use_compression: bool = False,
) -> tuple[dict, object]:
    """Evaluate compression for a cluster against ALL other LoRAs (pairwise)."""
    cluster_dir = clusters_root / f"cluster_{cluster_id:03d}"
    correct_lora_dir = loras_root / f"cluster_{cluster_id:03d}"

    if not cluster_dir.exists():
        raise FileNotFoundError(f"Cluster dir not found: {cluster_dir}")
    if not correct_lora_dir.exists():
        raise FileNotFoundError(f"Correct LoRA not found: {correct_lora_dir}")

    texts = load_cluster_texts(cluster_dir,
                               tokenizer,
                               split=split,
                               max_samples=max_samples,
                               min_tokens=min_tokens)
    if len(texts) == 0:
        raise ValueError(
            f"No texts found in cluster {cluster_id} with >{min_tokens} tokens"
        )

    results = {
        'cluster_id': cluster_id,
        'num_samples': len(texts),
        'gzip': [],
        'baseline': [],
        'correct_lora': [],
        'wrong_loras': {},  # Maps wrong_cluster_id -> list of results
    }

    method = "arithmetic coding" if use_compression else "CE loss"

    # Gzip baseline (fast, no model needed)
    print(f"    Evaluating gzip baseline...")
    for text in tqdm(texts, desc="Gzip", leave=False):
        bpt, n_tokens, total_bits = compute_gzip_bits_per_token(
            text, tokenizer)
        if bpt is not None:
            results['gzip'].append({
                'bpt': bpt,
                'tokens': n_tokens,
                'bits': total_bits
            })
    if results['gzip']:
        print(
            f"    Gzip done: {np.mean([r['bpt'] for r in results['gzip']]):.4f} bpt (n={len(results['gzip'])})"
        )
        sys.stdout.flush()

    # Baseline (no LoRA)
    print(f"    Evaluating baseline ({method})...")
    if use_compression:
        encoder = create_encoder(base_model_obj, tokenizer, device,
                                 bit_precision)
        for text in tqdm(texts, desc="Baseline", leave=False):
            bpt, n_tokens, total_bits = compress_text(encoder, text)
            if bpt is not None:
                results['baseline'].append({
                    'bpt': bpt,
                    'tokens': n_tokens,
                    'bits': total_bits
                })
        del encoder
    else:
        for text in tqdm(texts, desc="Baseline", leave=False):
            bpt, n_tokens, total_bits = compute_ce_bits_per_token(
                base_model_obj, tokenizer, text, device)
            if bpt is not None:
                results['baseline'].append({
                    'bpt': bpt,
                    'tokens': n_tokens,
                    'bits': total_bits
                })
    torch.cuda.empty_cache()
    if results['baseline']:
        print(
            f"    Baseline done: {np.mean([r['bpt'] for r in results['baseline']]):.4f} bpt (n={len(results['baseline'])})"
        )
        sys.stdout.flush()

    # Correct LoRA
    print(f"    Evaluating correct LoRA ({method})...")
    correct_model = PeftModel.from_pretrained(base_model_obj, correct_lora_dir)
    correct_model.eval()
    if use_compression:
        encoder = create_encoder(correct_model, tokenizer, device,
                                 bit_precision)
        for text in tqdm(texts, desc="Correct LoRA", leave=False):
            bpt, n_tokens, total_bits = compress_text(encoder, text)
            if bpt is not None:
                results['correct_lora'].append({
                    'bpt': bpt,
                    'tokens': n_tokens,
                    'bits': total_bits
                })
        del encoder
    else:
        for text in tqdm(texts, desc="Correct LoRA", leave=False):
            bpt, n_tokens, total_bits = compute_ce_bits_per_token(
                correct_model, tokenizer, text, device)
            if bpt is not None:
                results['correct_lora'].append({
                    'bpt': bpt,
                    'tokens': n_tokens,
                    'bits': total_bits
                })
    base_model_obj = correct_model.unload()
    del correct_model
    torch.cuda.empty_cache()
    if results['correct_lora']:
        print(
            f"    Correct LoRA done: {np.mean([r['bpt'] for r in results['correct_lora']]):.4f} bpt (n={len(results['correct_lora'])})"
        )
        sys.stdout.flush()

    # All wrong LoRAs (pairwise)
    for wrong_id in all_cluster_ids:
        if wrong_id == cluster_id:
            continue

        wrong_lora_dir = loras_root / f"cluster_{wrong_id:03d}"
        if not wrong_lora_dir.exists():
            print(f"    Skipping wrong LoRA {wrong_id}: not found")
            continue

        print(f"    Evaluating wrong LoRA {wrong_id} ({method})...")
        wrong_model = PeftModel.from_pretrained(base_model_obj, wrong_lora_dir)
        wrong_model.eval()

        wrong_results = []
        if use_compression:
            encoder = create_encoder(wrong_model, tokenizer, device,
                                     bit_precision)
            for text in tqdm(texts, desc=f"Wrong {wrong_id}", leave=False):
                bpt, n_tokens, total_bits = compress_text(encoder, text)
                if bpt is not None:
                    wrong_results.append({
                        'bpt': bpt,
                        'tokens': n_tokens,
                        'bits': total_bits
                    })
            del encoder
        else:
            for text in tqdm(texts, desc=f"Wrong {wrong_id}", leave=False):
                bpt, n_tokens, total_bits = compute_ce_bits_per_token(
                    wrong_model, tokenizer, text, device)
                if bpt is not None:
                    wrong_results.append({
                        'bpt': bpt,
                        'tokens': n_tokens,
                        'bits': total_bits
                    })

        results['wrong_loras'][wrong_id] = wrong_results
        base_model_obj = wrong_model.unload()
        del wrong_model
        torch.cuda.empty_cache()
        if wrong_results:
            print(
                f"    Wrong LoRA {wrong_id} done: {np.mean([r['bpt'] for r in wrong_results]):.4f} bpt (n={len(wrong_results)})"
            )
            sys.stdout.flush()

    # Compute averages
    if results['gzip']:
        results['gzip_avg'] = np.mean([r['bpt'] for r in results['gzip']])
    else:
        results['gzip_avg'] = 0

    if results['baseline']:
        results['baseline_avg'] = np.mean(
            [r['bpt'] for r in results['baseline']])
    else:
        results['baseline_avg'] = 0

    if results['correct_lora']:
        results['correct_lora_avg'] = np.mean(
            [r['bpt'] for r in results['correct_lora']])
    else:
        results['correct_lora_avg'] = 0

    results['wrong_lora_avgs'] = {}
    for wrong_id, wrong_results in results['wrong_loras'].items():
        if wrong_results:
            results['wrong_lora_avgs'][wrong_id] = np.mean(
                [r['bpt'] for r in wrong_results])

    # Average across all wrong LoRAs
    if results['wrong_lora_avgs']:
        results['wrong_lora_avg'] = np.mean(
            list(results['wrong_lora_avgs'].values()))
    else:
        results['wrong_lora_avg'] = 0

    return results, base_model_obj


def evaluate_cluster(
    cluster_id: int,
    clusters_root: Path,
    loras_root: Path,
    base_model_obj,
    tokenizer,
    wrong_cluster_id: int,
    max_samples: int = 50,
    min_tokens: int = 200,
    split: str = "test",
    bit_precision: int = 64,
    device: str = "cuda",
    use_compression: bool = False,
) -> tuple[dict, object]:
    """Evaluate compression for a single cluster."""
    cluster_dir = clusters_root / f"cluster_{cluster_id:03d}"
    correct_lora_dir = loras_root / f"cluster_{cluster_id:03d}"
    wrong_lora_dir = loras_root / f"cluster_{wrong_cluster_id:03d}"

    # Check paths exist
    if not cluster_dir.exists():
        raise FileNotFoundError(f"Cluster dir not found: {cluster_dir}")
    if not correct_lora_dir.exists():
        raise FileNotFoundError(f"Correct LoRA not found: {correct_lora_dir}")
    if not wrong_lora_dir.exists():
        raise FileNotFoundError(f"Wrong LoRA not found: {wrong_lora_dir}")

    # Load texts
    texts = load_cluster_texts(cluster_dir,
                               tokenizer,
                               split=split,
                               max_samples=max_samples,
                               min_tokens=min_tokens)
    if len(texts) == 0:
        raise ValueError(
            f"No texts found in cluster {cluster_id} with >{min_tokens} tokens"
        )

    results = {
        'cluster_id': cluster_id,
        'wrong_cluster_id': wrong_cluster_id,
        'num_samples': len(texts),
        'gzip': [],
        'baseline': [],
        'correct_lora': [],
        'wrong_lora': [],
    }

    method = "arithmetic coding" if use_compression else "CE loss"

    # Gzip baseline (fast, no model needed)
    print(f"    Evaluating gzip baseline...")
    for text in tqdm(texts, desc="Gzip", leave=False):
        bpt, n_tokens, total_bits = compute_gzip_bits_per_token(
            text, tokenizer)
        if bpt is not None:
            results['gzip'].append({
                'bpt': bpt,
                'tokens': n_tokens,
                'bits': total_bits
            })
    if results['gzip']:
        print(
            f"    Gzip done: {np.mean([r['bpt'] for r in results['gzip']]):.4f} bpt (n={len(results['gzip'])})"
        )
        sys.stdout.flush()

    # Baseline (no LoRA)
    print(f"    Evaluating with baseline model ({method})...")
    if use_compression:
        encoder = create_encoder(base_model_obj, tokenizer, device,
                                 bit_precision)
        for text in tqdm(texts, desc="Baseline", leave=False):
            bpt, n_tokens, total_bits = compress_text(encoder, text)
            if bpt is not None:
                results['baseline'].append({
                    'bpt': bpt,
                    'tokens': n_tokens,
                    'bits': total_bits
                })
        del encoder
    else:
        for text in tqdm(texts, desc="Baseline", leave=False):
            bpt, n_tokens, total_bits = compute_ce_bits_per_token(
                base_model_obj, tokenizer, text, device)
            if bpt is not None:
                results['baseline'].append({
                    'bpt': bpt,
                    'tokens': n_tokens,
                    'bits': total_bits
                })
    torch.cuda.empty_cache()
    if results['baseline']:
        print(
            f"    Baseline done: {np.mean([r['bpt'] for r in results['baseline']]):.4f} bpt (n={len(results['baseline'])})"
        )
        sys.stdout.flush()

    # Correct LoRA
    print(f"    Evaluating with correct LoRA ({method})...")
    correct_model = PeftModel.from_pretrained(base_model_obj, correct_lora_dir)
    correct_model.eval()
    if use_compression:
        encoder = create_encoder(correct_model, tokenizer, device,
                                 bit_precision)
        for text in tqdm(texts, desc="Correct LoRA", leave=False):
            bpt, n_tokens, total_bits = compress_text(encoder, text)
            if bpt is not None:
                results['correct_lora'].append({
                    'bpt': bpt,
                    'tokens': n_tokens,
                    'bits': total_bits
                })
        del encoder
    else:
        for text in tqdm(texts, desc="Correct LoRA", leave=False):
            bpt, n_tokens, total_bits = compute_ce_bits_per_token(
                correct_model, tokenizer, text, device)
            if bpt is not None:
                results['correct_lora'].append({
                    'bpt': bpt,
                    'tokens': n_tokens,
                    'bits': total_bits
                })
    # Properly unload the LoRA to restore base model
    base_model_obj = correct_model.unload()
    del correct_model
    torch.cuda.empty_cache()
    if results['correct_lora']:
        print(
            f"    Correct LoRA done: {np.mean([r['bpt'] for r in results['correct_lora']]):.4f} bpt (n={len(results['correct_lora'])})"
        )
        sys.stdout.flush()

    # Wrong LoRA
    print(
        f"    Evaluating with wrong LoRA (cluster {wrong_cluster_id}, {method})..."
    )
    wrong_model = PeftModel.from_pretrained(base_model_obj, wrong_lora_dir)
    wrong_model.eval()
    if use_compression:
        encoder = create_encoder(wrong_model, tokenizer, device, bit_precision)
        for text in tqdm(texts, desc="Wrong LoRA", leave=False):
            bpt, n_tokens, total_bits = compress_text(encoder, text)
            if bpt is not None:
                results['wrong_lora'].append({
                    'bpt': bpt,
                    'tokens': n_tokens,
                    'bits': total_bits
                })
        del encoder
    else:
        for text in tqdm(texts, desc="Wrong LoRA", leave=False):
            bpt, n_tokens, total_bits = compute_ce_bits_per_token(
                wrong_model, tokenizer, text, device)
            if bpt is not None:
                results['wrong_lora'].append({
                    'bpt': bpt,
                    'tokens': n_tokens,
                    'bits': total_bits
                })
    # Properly unload the LoRA to restore base model
    base_model_obj = wrong_model.unload()
    del wrong_model
    torch.cuda.empty_cache()
    if results['wrong_lora']:
        print(
            f"    Wrong LoRA done: {np.mean([r['bpt'] for r in results['wrong_lora']]):.4f} bpt (n={len(results['wrong_lora'])})"
        )
        sys.stdout.flush()

    # Compute averages
    if results['gzip']:
        results['gzip_avg'] = np.mean([r['bpt'] for r in results['gzip']])
        results['gzip_total_bits'] = sum(r['bits'] for r in results['gzip'])
    else:
        results['gzip_avg'] = 0

    if results['baseline']:
        results['baseline_avg'] = np.mean(
            [r['bpt'] for r in results['baseline']])
        results['baseline_total_bits'] = sum(r['bits']
                                             for r in results['baseline'])
        results['baseline_total_tokens'] = sum(r['tokens']
                                               for r in results['baseline'])
    else:
        results['baseline_avg'] = 0

    if results['correct_lora']:
        results['correct_lora_avg'] = np.mean(
            [r['bpt'] for r in results['correct_lora']])
        results['correct_lora_total_bits'] = sum(
            r['bits'] for r in results['correct_lora'])
    else:
        results['correct_lora_avg'] = 0

    if results['wrong_lora']:
        results['wrong_lora_avg'] = np.mean(
            [r['bpt'] for r in results['wrong_lora']])
        results['wrong_lora_total_bits'] = sum(r['bits']
                                               for r in results['wrong_lora'])
    else:
        results['wrong_lora_avg'] = 0

    return results, base_model_obj


def plot_results(all_results: list[dict], output_path: Path):
    """Create grouped bar chart of results."""
    n_clusters = len(all_results)
    x = np.arange(n_clusters)
    width = 0.2

    gzip_vals = [r.get('gzip_avg', 0) for r in all_results]
    baseline_vals = [r['baseline_avg'] for r in all_results]
    correct_vals = [r['correct_lora_avg'] for r in all_results]
    wrong_vals = [r['wrong_lora_avg'] for r in all_results]

    fig, ax = plt.subplots(figsize=(14, 6))

    bars0 = ax.bar(x - 1.5 * width,
                   gzip_vals,
                   width,
                   label='Gzip',
                   color='#9b59b6')
    bars1 = ax.bar(x - 0.5 * width,
                   baseline_vals,
                   width,
                   label='No LoRA (Baseline)',
                   color='#2ecc71')
    bars2 = ax.bar(x + 0.5 * width,
                   correct_vals,
                   width,
                   label='Correct LoRA',
                   color='#3498db')
    bars3 = ax.bar(x + 1.5 * width,
                   wrong_vals,
                   width,
                   label='Wrong LoRA',
                   color='#e74c3c')

    ax.set_xlabel('Cluster ID', fontsize=12)
    ax.set_ylabel('Bits per Token (Actual Compression)', fontsize=12)
    ax.set_title('LoRA Compression: Actual Bits per Token by Cluster',
                 fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{r['cluster_id']}" for r in all_results])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center',
                        va='bottom',
                        fontsize=6)

    add_labels(bars0)
    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved plot to {output_path}")


def print_summary_table(all_results: list[dict], use_compression: bool,
                        bit_precision: int):
    """Print summary table with averages."""
    gzip_avg = np.mean([r.get('gzip_avg', 0) for r in all_results])
    baseline_avg = np.mean([r['baseline_avg'] for r in all_results])
    correct_avg = np.mean([r['correct_lora_avg'] for r in all_results])
    wrong_avg = np.mean([r['wrong_lora_avg'] for r in all_results])

    correct_improvement = (baseline_avg - correct_avg
                           ) / baseline_avg * 100 if baseline_avg > 0 else 0
    wrong_improvement = (baseline_avg - wrong_avg
                         ) / baseline_avg * 100 if baseline_avg > 0 else 0
    gzip_vs_baseline = (gzip_avg - baseline_avg
                        ) / baseline_avg * 100 if baseline_avg > 0 else 0

    print("\n" + "=" * 70)
    print("SUMMARY TABLE (Averaged across all clusters)")
    if use_compression:
        print(f"Method: Arithmetic coding (bit_precision={bit_precision})")
    else:
        print("Method: Cross-entropy loss (theoretical limit)")
    print("=" * 70)
    print(f"\n{'Condition':<25} {'Bits/Token':>15} {'Δ from Baseline':>20}")
    print("-" * 60)
    print(f"{'Gzip':<25} {gzip_avg:>15.4f} {gzip_vs_baseline:>+19.2f}%")
    print(f"{'No LoRA (Baseline)':<25} {baseline_avg:>15.4f} {'--':>20}")
    print(
        f"{'Correct LoRA':<25} {correct_avg:>15.4f} {correct_improvement:>+19.2f}%"
    )
    print(
        f"{'Wrong LoRA':<25} {wrong_avg:>15.4f} {wrong_improvement:>+19.2f}%")
    print("-" * 60)

    # Per-cluster breakdown
    print(
        f"\n{'Cluster':<10} {'Gzip':>10} {'Baseline':>10} {'Correct':>10} {'Wrong':>10} {'Δ Correct':>12} {'Δ Wrong':>12}"
    )
    print("-" * 80)
    for r in all_results:
        c_impr = (r['baseline_avg'] - r['correct_lora_avg']
                  ) / r['baseline_avg'] * 100 if r['baseline_avg'] > 0 else 0
        w_impr = (r['baseline_avg'] - r['wrong_lora_avg']
                  ) / r['baseline_avg'] * 100 if r['baseline_avg'] > 0 else 0
        gzip_val = r.get('gzip_avg', 0)
        print(
            f"{r['cluster_id']:<10} {gzip_val:>10.4f} {r['baseline_avg']:>10.4f} {r['correct_lora_avg']:>10.4f} {r['wrong_lora_avg']:>10.4f} {c_impr:>+11.2f}% {w_impr:>+11.2f}%"
        )

    return {
        'gzip_avg': gzip_avg,
        'baseline_avg': baseline_avg,
        'correct_lora_avg': correct_avg,
        'wrong_lora_avg': wrong_avg,
        'correct_improvement_pct': correct_improvement,
        'wrong_improvement_pct': wrong_improvement,
    }


def main():
    parser = argparse.ArgumentParser(
        description=
        "Evaluate LoRA compression across clusters using arithmetic coding")

    parser.add_argument(
        "--clusters-root",
        type=str,
        default=
        "/n/netscratch/sham_lab/Lab/rrinberg/compression/lmsys-clustered/clusters",
        help="Root directory containing cluster folders")
    parser.add_argument(
        "--loras-root",
        type=str,
        default="/n/netscratch/sham_lab/Lab/rrinberg/compression/lmsys-loras",
        help="Root directory containing LoRA folders")
    parser.add_argument("--base-model",
                        type=str,
                        default="mistralai/Mistral-7B-Instruct-v0.2",
                        help="Base model name")
    parser.add_argument("--num-clusters",
                        type=int,
                        default=10,
                        help="Number of clusters to evaluate")
    parser.add_argument("--max-samples",
                        type=int,
                        default=50,
                        help="Max samples per cluster")
    parser.add_argument("--split",
                        type=str,
                        default="test",
                        help="Which split to use (train/test)")
    parser.add_argument("--output-dir",
                        type=str,
                        default="results/lora_evaluation",
                        help="Output directory for results")
    parser.add_argument(
        "--cluster-ids",
        type=str,
        default=None,
        help=
        "Comma-separated list of cluster IDs to evaluate (default: 0 to num-clusters-1)"
    )

    # Compression hyperparameters
    parser.add_argument(
        "--bit-precision",
        type=int,
        default=64,
        help="Bit precision for arithmetic coding (default: 64)")
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=200,
        help="Minimum number of tokens for a text to be included (default: 200)"
    )
    parser.add_argument(
        "--compression",
        action="store_true",
        help=
        "Use actual arithmetic coding compression (slow). Default: use CE loss (fast)"
    )
    parser.add_argument(
        "--no-pairwise",
        action="store_true",
        help=
        "Disable pairwise evaluation (only test one wrong LoRA per cluster)")
    parser.add_argument("--plot-dir",
                        type=str,
                        default="writing/695fe28d3a9ed52bd3824bba/assets/plts",
                        help="Output directory for plots")
    parser.add_argument("--plot-format",
                        type=str,
                        choices=["png", "pdf"],
                        default="pdf",
                        help="Output format for plots")

    args = parser.parse_args()
    args.pairwise = not args.no_pairwise  # Pairwise is default

    clusters_root = Path(args.clusters_root)
    loras_root = Path(args.loras_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Determine which clusters to evaluate
    if args.cluster_ids:
        cluster_ids = [int(x.strip()) for x in args.cluster_ids.split(",")]
    else:
        cluster_ids = list(range(args.num_clusters))

    method = "arithmetic coding" if args.compression else "CE loss"
    print(f"Evaluating clusters: {cluster_ids}")
    print(f"Clusters root: {clusters_root}")
    print(f"LoRAs root: {loras_root}")
    print(f"Base model: {args.base_model}")
    print(f"Max samples per cluster: {args.max_samples}")
    print(f"Min tokens per text: {args.min_tokens}")
    print(f"Pairwise evaluation: {args.pairwise}")
    print(f"Split: {args.split}")
    print(f"Method: {method}")
    if args.compression:
        print(f"Bit precision: {args.bit_precision}")
    print(f"Device: {device}")
    print("=" * 60)

    # Load base model
    print(f"\nLoading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model,
                                              trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )
    base_model.eval()

    # Evaluate each cluster
    all_results = []
    for i, cluster_id in enumerate(cluster_ids):
        print(f"\n{'='*60}")
        if args.pairwise:
            print(
                f"Cluster {cluster_id} (pairwise: testing against all other LoRAs)"
            )
        else:
            wrong_cluster_id = cluster_ids[(i + 1) % len(cluster_ids)]
            print(
                f"Cluster {cluster_id} (wrong LoRA from cluster {wrong_cluster_id})"
            )
        print(f"{'='*60}")

        try:
            if args.pairwise:
                result, base_model = evaluate_cluster_pairwise(
                    cluster_id=cluster_id,
                    clusters_root=clusters_root,
                    loras_root=loras_root,
                    base_model_obj=base_model,
                    tokenizer=tokenizer,
                    all_cluster_ids=cluster_ids,
                    max_samples=args.max_samples,
                    min_tokens=args.min_tokens,
                    split=args.split,
                    bit_precision=args.bit_precision,
                    device=device,
                    use_compression=args.compression,
                )
            else:
                wrong_cluster_id = cluster_ids[(i + 1) % len(cluster_ids)]
                result, base_model = evaluate_cluster(
                    cluster_id=cluster_id,
                    clusters_root=clusters_root,
                    loras_root=loras_root,
                    base_model_obj=base_model,
                    tokenizer=tokenizer,
                    wrong_cluster_id=wrong_cluster_id,
                    max_samples=args.max_samples,
                    min_tokens=args.min_tokens,
                    split=args.split,
                    bit_precision=args.bit_precision,
                    device=device,
                    use_compression=args.compression,
                )
            all_results.append(result)
            gzip_val = result.get('gzip_avg', 0)
            print(
                f"  Results: Gzip={gzip_val:.4f}, Baseline={result['baseline_avg']:.4f}, Correct={result['correct_lora_avg']:.4f}, Wrong(avg)={result['wrong_lora_avg']:.4f}"
            )
            sys.stdout.flush()

            # Save intermediate results after each cluster
            intermediate_path = output_dir / "intermediate_results.json"
            intermediate = {
                'status':
                'in_progress',
                'completed_clusters': [r['cluster_id'] for r in all_results],
                'per_cluster': [{
                    'cluster_id':
                    r['cluster_id'],
                    'gzip_avg':
                    float(r.get('gzip_avg', 0)),
                    'baseline_avg':
                    float(r['baseline_avg']),
                    'correct_lora_avg':
                    float(r['correct_lora_avg']),
                    'wrong_lora_avg':
                    float(r['wrong_lora_avg']),
                    'num_samples':
                    r['num_samples'],
                } for r in all_results],
            }
            with open(intermediate_path, 'w') as f:
                json.dump(intermediate, f, indent=2)
            print(f"  Saved intermediate results to {intermediate_path}")
            sys.stdout.flush()
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not all_results:
        print("No results to report!")
        return

    # Generate outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Plot - save to plot directory
    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / f"lora_pairwise_comparison.{args.plot_format}"
    plot_results(all_results, plot_path)

    # Summary table
    summary = print_summary_table(all_results, args.compression,
                                  args.bit_precision)

    # Save JSON results
    json_path = output_dir / f"lora_compression_results_{timestamp}.json"
    output_data = {
        'timestamp':
        timestamp,
        'config': {
            'clusters_root': str(clusters_root),
            'loras_root': str(loras_root),
            'base_model': args.base_model,
            'num_clusters': len(cluster_ids),
            'cluster_ids': cluster_ids,
            'max_samples': args.max_samples,
            'min_tokens': args.min_tokens,
            'split': args.split,
            'method': 'arithmetic_coding' if args.compression else 'ce_loss',
            'bit_precision': args.bit_precision if args.compression else None,
        },
        'summary':
        summary,
        'per_cluster_results': [
            {
                'cluster_id': r['cluster_id'],
                'wrong_cluster_id': r.get('wrong_cluster_id'),
                'num_samples': r['num_samples'],
                'gzip_avg': r.get('gzip_avg', 0),
                'baseline_avg': r['baseline_avg'],
                'correct_lora_avg': r['correct_lora_avg'],
                'wrong_lora_avg': r['wrong_lora_avg'],
                # Per-sample data for plotting
                'samples': {
                    'gzip': r.get('gzip', []),
                    'baseline': r.get('baseline', []),
                    'correct_lora': r.get('correct_lora', []),
                    'wrong_lora': r.get('wrong_lora', []),
                    'wrong_loras': r.get('wrong_loras',
                                         {}),  # For pairwise evaluation
                },
            } for r in all_results
        ],
    }
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved JSON results to {json_path}")


if __name__ == "__main__":
    main()
