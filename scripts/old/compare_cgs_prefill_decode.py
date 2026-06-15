#!/usr/bin/env python3
"""
Compare CGS (Convolved Gaussian Score) robustness between decode and prefill.

This experiment generates tokens autoregressively while computing CGS scores,
then runs prefill to recompute CGS scores, and compares the top-K tokens
selected by CGS in both modes.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from datetime import datetime
import sys

# Add compression module to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from compression.scoring_functions.convolved_gaussian_score import (
    get_seed, draw_u, compute_convolved_gaussian_score)

# Diverse prompts to test across different domains
DEFAULT_PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "In a groundbreaking scientific discovery, researchers have found",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
    "The capital of France is Paris, which is known for",
    "Question: What is the meaning of life?\nAnswer:",
]


def logits_to_cdf(logits: torch.Tensor) -> torch.Tensor:
    """Convert logits to CDF values.

    Args:
        logits: [V] logits tensor

    Returns:
        cdf: [V] cumulative distribution function values
    """
    probs = torch.softmax(logits, dim=-1)
    cdf = torch.cumsum(probs, dim=-1)
    return cdf


def generate_and_compute_cgs_decode(model,
                                    tokenizer,
                                    prompt: str,
                                    num_tokens: int,
                                    base_seed: int,
                                    sigma: float,
                                    top_k: int = 20):
    """Generate tokens autoregressively and compute CGS top-K at each step.

    Returns:
        generated_ids: List of generated token IDs
        decode_top_k_tokens: List of top-K token lists (one per position)
        decode_cgs_scores: List of CGS score tensors (one per position)
    """
    print(f"Generating {num_tokens} tokens with CGS scoring (decode)...")

    # Encode initial prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    prompt_len = input_ids.shape[1]

    generated_ids = []
    decode_top_k_tokens = []
    decode_cgs_scores = []

    # Use CPU generator for deterministic u generation
    cpu_generator = torch.Generator(device='cpu')

    with torch.no_grad():
        for i in range(num_tokens):
            # Forward pass
            outputs = model(input_ids)
            logits = outputs.logits[0, -1, :].cpu()  # [V]

            # Convert to CDF
            cdf = logits_to_cdf(logits)

            # Generate deterministic u based on seed and past tokens
            past_tokens = generated_ids.copy()
            seed = get_seed(base_seed, past_tokens)
            u = draw_u(seed, cpu_generator)

            # Compute CGS scores for all tokens
            cgs_log_scores = compute_convolved_gaussian_score(cdf, u, sigma)

            # Get top-K tokens by CGS score
            top_k_scores, top_k_indices = torch.topk(cgs_log_scores, k=top_k)
            decode_top_k_tokens.append(top_k_indices.cpu().tolist())
            decode_cgs_scores.append(cgs_log_scores.cpu())

            # Sample next token (greedy for reproducibility)
            next_token_id = torch.argmax(logits).item()
            generated_ids.append(next_token_id)

            # Append to input for next iteration
            input_ids = torch.cat(
                [input_ids,
                 torch.tensor([[next_token_id]]).to(model.device)],
                dim=1)

            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1}/{num_tokens} tokens")

    print(f"✓ Generated {len(generated_ids)} tokens with CGS scoring")
    return generated_ids, decode_top_k_tokens, decode_cgs_scores


def compute_cgs_prefill(model,
                        tokenizer,
                        prompt: str,
                        generated_ids: list,
                        base_seed: int,
                        sigma: float,
                        top_k: int = 20):
    """Run prefill and compute CGS top-K for each position.

    Returns:
        prefill_top_k_tokens: List of top-K token lists (one per position)
        prefill_cgs_scores: List of CGS score tensors (one per position)
    """
    print(
        f"Computing CGS scores via prefill on {len(generated_ids)} tokens...")

    # Encode prompt + generated tokens
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    full_sequence = torch.cat(
        [prompt_ids,
         torch.tensor([generated_ids]).to(model.device)], dim=1)

    with torch.no_grad():
        outputs = model(full_sequence)
        logits = outputs.logits[0].cpu()  # [seq_len, V]

    # Extract logits for positions corresponding to generated tokens
    prompt_len = prompt_ids.shape[1]

    prefill_top_k_tokens = []
    prefill_cgs_scores = []

    generator = torch.Generator(device='cpu')

    for i in range(len(generated_ids)):
        # Get logits at position that predicts token i
        position_logits = logits[prompt_len - 1 + i]  # [V]

        # Convert to CDF
        cdf = logits_to_cdf(position_logits)

        # Generate deterministic u (same as decode)
        past_tokens = generated_ids[:i]
        seed = get_seed(base_seed, past_tokens)
        u = draw_u(seed, generator)

        # Compute CGS scores
        cgs_log_scores = compute_convolved_gaussian_score(cdf, u, sigma)

        # Get top-K tokens
        top_k_scores, top_k_indices = torch.topk(cgs_log_scores, k=top_k)
        prefill_top_k_tokens.append(top_k_indices.cpu().tolist())
        prefill_cgs_scores.append(cgs_log_scores.cpu())

    print(f"✓ Computed CGS scores for {len(prefill_top_k_tokens)} positions")
    return prefill_top_k_tokens, prefill_cgs_scores


def compare_top_k_lists(decode_top_k, prefill_top_k, decode_scores,
                        prefill_scores):
    """Compare top-K token lists between decode and prefill.

    Args:
        decode_top_k: List of top-K token lists from decode
        prefill_top_k: List of top-K token lists from prefill
        decode_scores: List of full CGS score tensors from decode
        prefill_scores: List of full CGS score tensors from prefill

    Returns:
        metrics: Dict with overlap statistics and non-overlap rank information
    """
    assert len(decode_top_k) == len(
        prefill_top_k), "Lists must have same length"

    intersection_sizes = []
    jaccard_similarities = []

    # Track ranks of non-overlapping tokens
    decode_only_ranks_in_prefill = [
    ]  # Tokens in decode top-K but not prefill top-K, ranked by prefill
    prefill_only_ranks_in_decode = [
    ]  # Tokens in prefill top-K but not decode top-K, ranked by decode

    for dec_tokens, pre_tokens, dec_scores, pre_scores in zip(
            decode_top_k, prefill_top_k, decode_scores, prefill_scores):
        dec_set = set(dec_tokens)
        pre_set = set(pre_tokens)

        intersection = dec_set & pre_set
        union = dec_set | pre_set

        intersection_sizes.append(len(intersection))
        jaccard = len(intersection) / len(union) if len(union) > 0 else 0.0
        jaccard_similarities.append(jaccard)

        # Find tokens in decode top-K but not in prefill top-K
        decode_only = dec_set - pre_set
        for token_id in decode_only:
            # Find this token's rank in prefill scores
            # Rank = number of tokens with higher score + 1
            rank = (pre_scores > pre_scores[token_id]).sum().item() + 1
            decode_only_ranks_in_prefill.append(rank)

        # Find tokens in prefill top-K but not in decode top-K
        prefill_only = pre_set - dec_set
        for token_id in prefill_only:
            # Find this token's rank in decode scores
            rank = (dec_scores > dec_scores[token_id]).sum().item() + 1
            prefill_only_ranks_in_decode.append(rank)

    metrics = {
        'intersection_sizes': intersection_sizes,
        'jaccard_similarities': jaccard_similarities,
        'mean_intersection': np.mean(intersection_sizes),
        'mean_jaccard': np.mean(jaccard_similarities),
        'std_intersection': np.std(intersection_sizes),
        'std_jaccard': np.std(jaccard_similarities),
        'decode_only_ranks_in_prefill': decode_only_ranks_in_prefill,
        'prefill_only_ranks_in_decode': prefill_only_ranks_in_decode,
    }

    return metrics


def plot_comparison_results(all_metrics, output_dir: Path, model_name: str,
                            num_prompts: int, top_k: int, sigma: float):
    """Plot comparison results."""
    print(f"\nPlotting results...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate across all prompts
    all_intersections = []
    all_jaccards = []
    all_decode_only_ranks = []
    all_prefill_only_ranks = []

    for metrics in all_metrics:
        all_intersections.extend(metrics['intersection_sizes'])
        all_jaccards.extend(metrics['jaccard_similarities'])
        all_decode_only_ranks.extend(metrics['decode_only_ranks_in_prefill'])
        all_prefill_only_ranks.extend(metrics['prefill_only_ranks_in_decode'])

    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f'CGS Top-{top_k} Overlap: Decode vs Prefill ({num_prompts} prompts, σ={sigma})\n{model_name}',
        fontsize=12)

    # Plot 1: Intersection sizes
    ax1 = axes[0, 0]
    ax1.hist(all_intersections,
             bins=min(top_k + 1, 20),
             alpha=0.7,
             edgecolor='black',
             range=(0, top_k))
    ax1.axvline(np.mean(all_intersections),
                color='red',
                linestyle='--',
                linewidth=2,
                label=f'Mean: {np.mean(all_intersections):.2f}')
    ax1.set_xlabel('Intersection Size (# overlapping tokens)')
    ax1.set_ylabel('Frequency (log scale)')
    ax1.set_yscale('log')
    ax1.set_title(f'Overlap in Top-{top_k} Tokens\n(max={top_k})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Jaccard similarity
    ax2 = axes[0, 1]
    ax2.hist(all_jaccards, bins=20, alpha=0.7, edgecolor='black', range=(0, 1))
    ax2.axvline(np.mean(all_jaccards),
                color='red',
                linestyle='--',
                linewidth=2,
                label=f'Mean: {np.mean(all_jaccards):.3f}')
    ax2.set_xlabel('Jaccard Similarity')
    ax2.set_ylabel('Frequency (log scale)')
    ax2.set_yscale('log')
    ax2.set_title(f'Jaccard Similarity\n(intersection / union)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Ranks of decode-only tokens in prefill
    ax3 = axes[1, 0]
    if all_decode_only_ranks:
        # Create bins: 1-50 individual, then 50+ aggregated
        rank_cutoff = 50
        bins = list(range(1,
                          rank_cutoff + 1)) + [max(all_decode_only_ranks) + 1]
        counts, edges = np.histogram(all_decode_only_ranks, bins=bins)

        # Plot bars
        bar_positions = list(range(1, rank_cutoff)) + [rank_cutoff]
        ax3.bar(bar_positions,
                counts,
                alpha=0.7,
                edgecolor='black',
                color='blue',
                width=1.0)

        ax3.axvline(np.mean(all_decode_only_ranks),
                    color='red',
                    linestyle='--',
                    linewidth=2,
                    label=f'Mean: {np.mean(all_decode_only_ranks):.1f}')
        ax3.axvline(top_k,
                    color='green',
                    linestyle=':',
                    linewidth=2,
                    label=f'Top-K cutoff: {top_k}')
        ax3.set_xlabel('Rank in Prefill CGS')
        ax3.set_ylabel('Frequency (log scale)')
        ax3.set_yscale('log')
        ax3.set_xlim(0, rank_cutoff + 1)

        # Custom x-tick labels: show 1, 10, 20, 30, 40, 50, 50+
        tick_positions = [1, 10, 20, 30, 40, rank_cutoff]
        tick_labels = ['1', '10', '20', '30', '40', f'{rank_cutoff}+']
        ax3.set_xticks(tick_positions)
        ax3.set_xticklabels(tick_labels)

        ax3.set_title(
            f'Decode Top-{top_k} tokens NOT in Prefill Top-{top_k}\n(n={len(all_decode_only_ranks)} tokens)'
        )
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
    else:
        ax3.text(0.5,
                 0.5,
                 'Perfect overlap!\nNo decode-only tokens',
                 ha='center',
                 va='center',
                 transform=ax3.transAxes)
        ax3.set_title(f'Decode Top-{top_k} tokens NOT in Prefill Top-{top_k}')

    # Plot 4: Ranks of prefill-only tokens in decode
    ax4 = axes[1, 1]
    if all_prefill_only_ranks:
        # Create bins: 1-50 individual, then 50+ aggregated
        rank_cutoff = 50
        bins = list(range(
            1, rank_cutoff + 1)) + [max(all_prefill_only_ranks) + 1]
        counts, edges = np.histogram(all_prefill_only_ranks, bins=bins)

        # Plot bars
        bar_positions = list(range(1, rank_cutoff)) + [rank_cutoff]
        ax4.bar(bar_positions,
                counts,
                alpha=0.7,
                edgecolor='black',
                color='orange',
                width=1.0)

        ax4.axvline(np.mean(all_prefill_only_ranks),
                    color='red',
                    linestyle='--',
                    linewidth=2,
                    label=f'Mean: {np.mean(all_prefill_only_ranks):.1f}')
        ax4.axvline(top_k,
                    color='green',
                    linestyle=':',
                    linewidth=2,
                    label=f'Top-K cutoff: {top_k}')
        ax4.set_xlabel('Rank in Decode CGS')
        ax4.set_ylabel('Frequency (log scale)')
        ax4.set_yscale('log')
        ax4.set_xlim(0, rank_cutoff + 1)

        # Custom x-tick labels
        tick_positions = [1, 10, 20, 30, 40, rank_cutoff]
        tick_labels = ['1', '10', '20', '30', '40', f'{rank_cutoff}+']
        ax4.set_xticks(tick_positions)
        ax4.set_xticklabels(tick_labels)

        ax4.set_title(
            f'Prefill Top-{top_k} tokens NOT in Decode Top-{top_k}\n(n={len(all_prefill_only_ranks)} tokens)'
        )
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
    else:
        ax4.text(0.5,
                 0.5,
                 'Perfect overlap!\nNo prefill-only tokens',
                 ha='center',
                 va='center',
                 transform=ax4.transAxes)
        ax4.set_title(f'Prefill Top-{top_k} tokens NOT in Decode Top-{top_k}')

    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"cgs_decode_prefill_comparison_{timestamp}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to {output_path}")

    plt.close()

    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics:")
    print("=" * 60)
    print(
        f"Intersection size: {np.mean(all_intersections):.2f} ± {np.std(all_intersections):.2f} (out of {top_k})"
    )
    print(
        f"  Min: {np.min(all_intersections)}, Max: {np.max(all_intersections)}"
    )
    print(
        f"Jaccard similarity: {np.mean(all_jaccards):.3f} ± {np.std(all_jaccards):.3f}"
    )
    print(
        f"  Min: {np.min(all_jaccards):.3f}, Max: {np.max(all_jaccards):.3f}")
    print("")
    if all_decode_only_ranks:
        print(
            f"Decode-only tokens rank in prefill: {np.mean(all_decode_only_ranks):.1f} ± {np.std(all_decode_only_ranks):.1f}"
        )
        print(
            f"  Min: {np.min(all_decode_only_ranks)}, Max: {np.max(all_decode_only_ranks)}"
        )
        print(f"  Median: {np.median(all_decode_only_ranks):.1f}")
    else:
        print(f"Decode-only tokens: None (perfect overlap)")
    print("")
    if all_prefill_only_ranks:
        print(
            f"Prefill-only tokens rank in decode: {np.mean(all_prefill_only_ranks):.1f} ± {np.std(all_prefill_only_ranks):.1f}"
        )
        print(
            f"  Min: {np.min(all_prefill_only_ranks)}, Max: {np.max(all_prefill_only_ranks)}"
        )
        print(f"  Median: {np.median(all_prefill_only_ranks):.1f}")
    else:
        print(f"Prefill-only tokens: None (perfect overlap)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Compare CGS robustness: prefill vs decode")
    parser.add_argument("--model",
                        default="mistralai/Mistral-7B-Instruct-v0.2",
                        help="Model to use")
    parser.add_argument("--num-tokens",
                        type=int,
                        default=200,
                        help="Number of tokens to generate per prompt")
    parser.add_argument("--prompts",
                        nargs="+",
                        default=None,
                        help="Custom prompts to test")
    parser.add_argument("--output-dir",
                        type=Path,
                        default=Path("data/results/cgs_prefill_decode"),
                        help="Directory to save results")
    parser.add_argument("--top-k",
                        type=int,
                        default=20,
                        help="Number of top CGS tokens to compare")
    parser.add_argument("--sigma",
                        type=float,
                        default=0.1,
                        help="CGS Gaussian noise parameter")
    parser.add_argument("--base-seed",
                        type=int,
                        default=42,
                        help="Base random seed for deterministic u generation")

    args = parser.parse_args()

    # Use default prompts if none specified
    prompts = args.prompts if args.prompts else DEFAULT_PROMPTS

    print("=" * 60)
    print("CGS Robustness: Decode vs Prefill Comparison")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Num prompts: {len(prompts)}")
    print(f"Num tokens per prompt: {args.num_tokens}")
    print(f"Top-K: {args.top_k}")
    print(f"Sigma: {args.sigma}")
    print(f"Base seed: {args.base_seed}")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.eval()
    print(f"✓ Loaded model on {device}")

    # Run experiment for each prompt
    all_metrics = []

    for i, prompt in enumerate(prompts):
        print(f"\n{'='*60}")
        print(f"Prompt {i+1}/{len(prompts)}")
        print(f"{'='*60}")
        print(f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        print(f"{'='*60}")

        # Generate with decode + CGS
        generated_ids, decode_top_k, decode_scores = generate_and_compute_cgs_decode(
            model, tokenizer, prompt, args.num_tokens, args.base_seed,
            args.sigma, args.top_k)

        # Compute CGS via prefill
        prefill_top_k, prefill_scores = compute_cgs_prefill(
            model, tokenizer, prompt, generated_ids, args.base_seed,
            args.sigma, args.top_k)

        # Compare top-K lists
        metrics = compare_top_k_lists(decode_top_k, prefill_top_k,
                                      decode_scores, prefill_scores)
        all_metrics.append(metrics)

        print(f"\nResults for this prompt:")
        print(
            f"  Mean intersection: {metrics['mean_intersection']:.2f}/{args.top_k}"
        )
        print(f"  Mean Jaccard: {metrics['mean_jaccard']:.3f}")

        # Free memory
        del decode_top_k, decode_scores, prefill_top_k, prefill_scores, generated_ids
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Plot aggregated results
    plot_comparison_results(all_metrics, args.output_dir,
                            args.model.split('/')[-1], len(prompts),
                            args.top_k, args.sigma)

    print("\n" + "=" * 60)
    print("✓ Experiment complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
