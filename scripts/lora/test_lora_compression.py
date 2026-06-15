#!/usr/bin/env python3
"""
Test LoRA compression improvement on training data.

Compares compression ratio with and without LoRA on the cluster's training set.
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm


def load_cluster_texts(cluster_dir: Path,
                       split: str = "train",
                       max_samples: int = None) -> list[str]:
    """Load texts from a cluster."""
    path = cluster_dir / f"{split}.json"
    with open(path, 'r') as f:
        data = json.load(f)

    texts = data['texts']
    if max_samples and max_samples < len(texts):
        import random
        random.seed(42)
        texts = random.sample(texts, max_samples)

    return texts


def compute_bits_per_token(model,
                           tokenizer,
                           text: str,
                           device: str = "cuda") -> tuple[float, int]:
    """Compute bits per token for a text.

    Returns:
        Tuple of (bits_per_token, num_tokens)
    """
    inputs = tokenizer(text,
                       return_tensors="pt",
                       truncation=True,
                       max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss.item()  # Cross-entropy loss in nats

    # Convert nats to bits: bits = nats * log2(e) = nats / ln(2)
    bits_per_token = loss / np.log(2)
    num_tokens = inputs["input_ids"].shape[1]

    return bits_per_token, num_tokens


def test_compression(
    cluster_dir: str,
    lora_dir: str,
    wrong_lora_dir: str = None,
    base_model: str = "mistralai/Mistral-7B-Instruct-v0.2",
    max_samples: int = 100,
    split: str = "train",
):
    """Test compression with and without LoRA."""
    cluster_dir = Path(cluster_dir)
    lora_dir = Path(lora_dir)
    if wrong_lora_dir:
        wrong_lora_dir = Path(wrong_lora_dir)

    print(f"Cluster: {cluster_dir.name}")
    print(f"Correct LoRA: {lora_dir}")
    if wrong_lora_dir:
        print(f"Wrong LoRA: {wrong_lora_dir}")
    print(f"Base model: {base_model}")
    print(f"Split: {split}")
    print(f"Max samples: {max_samples}")
    print("=" * 60)

    # Load texts
    print("Loading texts...")
    texts = load_cluster_texts(cluster_dir,
                               split=split,
                               max_samples=max_samples)
    print(f"Loaded {len(texts)} texts")

    # Filter short texts upfront
    texts = [t for t in texts if len(t.strip()) >= 50]
    print(f"After filtering short texts: {len(texts)}")

    # Load tokenizer
    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model,
                                              trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    print(f"Loading base model: {base_model}")
    base_model_obj = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )
    base_model_obj.eval()

    # Compute baseline compression (no LoRA)
    print("\n[Phase 1] Computing baseline compression (no LoRA)...")
    baseline_results = []
    for text in tqdm(texts, desc="Baseline"):
        bpt, n_tokens = compute_bits_per_token(base_model_obj, tokenizer, text)
        baseline_results.append({
            'bits_per_token': bpt,
            'num_tokens': n_tokens,
            'total_bits': bpt * n_tokens,
        })

    # Load correct LoRA
    print(f"\n[Phase 2] Loading correct LoRA from {lora_dir}...")
    lora_model = PeftModel.from_pretrained(base_model_obj, lora_dir)
    lora_model.eval()

    # Compute correct LoRA compression
    print("Computing correct LoRA compression...")
    correct_lora_results = []
    for text in tqdm(texts, desc="Correct LoRA"):
        bpt, n_tokens = compute_bits_per_token(lora_model, tokenizer, text)
        correct_lora_results.append({
            'bits_per_token': bpt,
            'num_tokens': n_tokens,
            'total_bits': bpt * n_tokens,
        })

    # Unload correct LoRA
    del lora_model
    torch.cuda.empty_cache()

    # Test wrong LoRA if provided
    wrong_lora_results = None
    if wrong_lora_dir and wrong_lora_dir.exists():
        print(f"\n[Phase 3] Loading wrong LoRA from {wrong_lora_dir}...")
        wrong_lora_model = PeftModel.from_pretrained(base_model_obj,
                                                     wrong_lora_dir)
        wrong_lora_model.eval()

        print("Computing wrong LoRA compression...")
        wrong_lora_results = []
        for text in tqdm(texts, desc="Wrong LoRA"):
            bpt, n_tokens = compute_bits_per_token(wrong_lora_model, tokenizer,
                                                   text)
            wrong_lora_results.append({
                'bits_per_token': bpt,
                'num_tokens': n_tokens,
                'total_bits': bpt * n_tokens,
            })

        del wrong_lora_model
        torch.cuda.empty_cache()

    # Compute statistics
    baseline_bpt = np.mean([r['bits_per_token'] for r in baseline_results])
    baseline_total_bits = sum(r['total_bits'] for r in baseline_results)
    baseline_total_tokens = sum(r['num_tokens'] for r in baseline_results)

    correct_bpt = np.mean([r['bits_per_token'] for r in correct_lora_results])
    correct_total_bits = sum(r['total_bits'] for r in correct_lora_results)

    correct_improvement_bpt = (baseline_bpt - correct_bpt) / baseline_bpt * 100
    correct_improvement_bits = (baseline_total_bits -
                                correct_total_bits) / baseline_total_bits * 100

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nSamples evaluated: {len(baseline_results)}")
    print(f"Total tokens: {baseline_total_tokens}")

    if wrong_lora_results:
        wrong_bpt = np.mean([r['bits_per_token'] for r in wrong_lora_results])
        wrong_total_bits = sum(r['total_bits'] for r in wrong_lora_results)
        wrong_improvement_bpt = (baseline_bpt - wrong_bpt) / baseline_bpt * 100
        wrong_improvement_bits = (baseline_total_bits -
                                  wrong_total_bits) / baseline_total_bits * 100

        print(
            f"\n{'Metric':<20} {'Baseline':>10} {'Correct':>10} {'Wrong':>10} {'Δ Correct':>10} {'Δ Wrong':>10}"
        )
        print("-" * 72)
        print(
            f"{'Bits/token':<20} {baseline_bpt:>10.4f} {correct_bpt:>10.4f} {wrong_bpt:>10.4f} {correct_improvement_bpt:>+9.2f}% {wrong_improvement_bpt:>+9.2f}%"
        )
        print(
            f"{'Total bits':<20} {baseline_total_bits:>10.0f} {correct_total_bits:>10.0f} {wrong_total_bits:>10.0f} {correct_improvement_bits:>+9.2f}% {wrong_improvement_bits:>+9.2f}%"
        )

        # Per-sample breakdown
        print(f"\n{'Per-sample breakdown (first 10):'}")
        print(
            f"{'#':<4} {'Tokens':>6} {'Base':>8} {'Correct':>8} {'Wrong':>8} {'Δ Corr':>8} {'Δ Wrong':>8}"
        )
        print("-" * 58)
        for i in range(min(10, len(baseline_results))):
            b = baseline_results[i]
            c = correct_lora_results[i]
            w = wrong_lora_results[i]
            delta_c = (b['bits_per_token'] -
                       c['bits_per_token']) / b['bits_per_token'] * 100
            delta_w = (b['bits_per_token'] -
                       w['bits_per_token']) / b['bits_per_token'] * 100
            print(
                f"{i:<4} {b['num_tokens']:>6} {b['bits_per_token']:>8.3f} {c['bits_per_token']:>8.3f} {w['bits_per_token']:>8.3f} {delta_c:>+7.2f}% {delta_w:>+7.2f}%"
            )
    else:
        print(
            f"\n{'Metric':<25} {'Baseline':>12} {'With LoRA':>12} {'Improvement':>12}"
        )
        print("-" * 60)
        print(
            f"{'Bits per token':<25} {baseline_bpt:>12.4f} {correct_bpt:>12.4f} {correct_improvement_bpt:>+11.2f}%"
        )
        print(
            f"{'Total bits':<25} {baseline_total_bits:>12.0f} {correct_total_bits:>12.0f} {correct_improvement_bits:>+11.2f}%"
        )

        # Per-sample breakdown
        print(f"\n{'Per-sample breakdown (first 10):'}")
        print(
            f"{'Sample':<8} {'Tokens':>8} {'Base BPT':>10} {'LoRA BPT':>10} {'Δ':>8}"
        )
        print("-" * 50)
        for i in range(min(10, len(baseline_results))):
            b = baseline_results[i]
            c = correct_lora_results[i]
            delta = (b['bits_per_token'] -
                     c['bits_per_token']) / b['bits_per_token'] * 100
            print(
                f"{i:<8} {b['num_tokens']:>8} {b['bits_per_token']:>10.4f} {c['bits_per_token']:>10.4f} {delta:>+7.2f}%"
            )

    # Save results
    results = {
        'cluster': cluster_dir.name,
        'correct_lora_dir': str(lora_dir),
        'wrong_lora_dir': str(wrong_lora_dir) if wrong_lora_dir else None,
        'base_model': base_model,
        'split': split,
        'num_samples': len(baseline_results),
        'total_tokens': baseline_total_tokens,
        'baseline': {
            'bits_per_token': baseline_bpt,
            'total_bits': baseline_total_bits,
        },
        'correct_lora': {
            'bits_per_token': correct_bpt,
            'total_bits': correct_total_bits,
            'improvement_bpt_pct': correct_improvement_bpt,
            'improvement_bits_pct': correct_improvement_bits,
        },
    }

    if wrong_lora_results:
        results['wrong_lora'] = {
            'bits_per_token': wrong_bpt,
            'total_bits': wrong_total_bits,
            'improvement_bpt_pct': wrong_improvement_bpt,
            'improvement_bits_pct': wrong_improvement_bits,
        }

    output_file = lora_dir / f"compression_test_{split}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Test LoRA compression on training data")

    parser.add_argument("--cluster-dir",
                        type=str,
                        required=True,
                        help="Cluster directory containing train.json")
    parser.add_argument(
        "--lora-dir",
        type=str,
        required=True,
        help="Directory containing trained LoRA adapter (correct LoRA)")
    parser.add_argument(
        "--wrong-lora-dir",
        type=str,
        default=None,
        help="Directory containing wrong LoRA adapter for comparison")
    parser.add_argument("--base-model",
                        type=str,
                        default="mistralai/Mistral-7B-Instruct-v0.2",
                        help="Base model")
    parser.add_argument("--max-samples",
                        type=int,
                        default=100,
                        help="Max samples to evaluate (default: 100)")
    parser.add_argument("--split",
                        type=str,
                        default="train",
                        choices=["train", "test"],
                        help="Which split to evaluate (default: train)")

    args = parser.parse_args()

    test_compression(
        cluster_dir=args.cluster_dir,
        lora_dir=args.lora_dir,
        wrong_lora_dir=args.wrong_lora_dir,
        base_model=args.base_model,
        max_samples=args.max_samples,
        split=args.split,
    )


if __name__ == "__main__":
    main()
