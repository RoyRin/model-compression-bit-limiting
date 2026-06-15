#!/usr/bin/env python3
"""
Compare logits from decode (autoregressive) vs prefill (batch processing).

This experiment generates tokens autoregressively, storing the logits at each step,
then runs prefill on the entire sequence to compare if there are numerical differences.

Analyzes how divergence changes over context length, reporting statistics (min, mean,
median, 90th, 100th percentiles) for L-inf, L1, L2 norms of PDF and CDF differences.
"""

import gc
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from datetime import datetime
import sys

# Add notebooks directory to path for importing analysis functions
_notebooks_dir = Path(__file__).parent.parent / "notebooks"
if str(_notebooks_dir) not in sys.path:
    sys.path.insert(0, str(_notebooks_dir))

try:
    from analyze_bucketed_probs import (
        load_prob_file, analyze_bucket_sizes, plot_norms_vs_bucket_size,
        plot_norms_over_position, analyze_spearman_correlations,
        plot_spearman_correlations, analyze_contiguous_ranges,
        plot_contiguous_ranges)
    BUCKETED_ANALYSIS_AVAILABLE = True
except ImportError:
    BUCKETED_ANALYSIS_AVAILABLE = False
    print("Warning: Could not import bucketed analysis functions")

# Diverse prompts to test across different domains
DEFAULT_PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "In a groundbreaking scientific discovery, researchers have found",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
    "The capital of France is Paris, which is known for",
    "Question: What is the meaning of life?\nAnswer:",
    "Once upon a time in a faraway kingdom, there lived",
    "The following is a technical explanation of how neural networks",
]


def generate_and_store_decode_logits(model,
                                     tokenizer,
                                     prompt: str,
                                     num_tokens: int = 1000):
    """Generate tokens autoregressively and store logits at each decode step.

    Uses KV cache for proper decode behavior (single-token forward passes after initial prefill).

    Returns:
        generated_ids: List of generated token IDs (length = num_tokens)
        decode_logits: List of logits tensors, one per generation step
    """
    print(
        f"Generating {num_tokens} tokens autoregressively (decode phase with KV cache)..."
    )

    # Encode initial prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    generated_ids = []
    decode_logits = []
    past_key_values = None

    with torch.no_grad():
        # Initial prefill to get KV cache for the prompt
        outputs = model(input_ids, use_cache=True)
        past_key_values = outputs.past_key_values

        # Get logits for the first generated token (from prefill)
        next_token_logits = outputs.logits[0, -1, :].cpu()
        decode_logits.append(next_token_logits)

        next_token_id = torch.argmax(next_token_logits).item()
        generated_ids.append(next_token_id)

        # Now decode remaining tokens one at a time using KV cache
        for i in range(1, num_tokens):
            # Forward pass with only the new token, using cached KV
            new_token = torch.tensor([[next_token_id]]).to(model.device)
            outputs = model(new_token,
                            past_key_values=past_key_values,
                            use_cache=True)
            past_key_values = outputs.past_key_values

            # Get logits for next token prediction
            next_token_logits = outputs.logits[0, -1, :].cpu()
            decode_logits.append(next_token_logits)

            # Sample next token (greedy for reproducibility)
            next_token_id = torch.argmax(next_token_logits).item()
            generated_ids.append(next_token_id)

            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1}/{num_tokens} tokens")

    print(
        f"✓ Generated {len(generated_ids)} tokens (first from prefill, rest from decode)"
    )
    return generated_ids, decode_logits


def get_prefill_logits(model, tokenizer, prompt: str, generated_ids: list):
    """Run prefill on the entire sequence (prompt + generated tokens).

    Returns:
        prefill_logits: List of logits tensors, one per position in generated sequence
    """
    print(f"Running prefill on {len(generated_ids)} generated tokens...")

    # Encode prompt + generated tokens
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    full_sequence = torch.cat(
        [prompt_ids,
         torch.tensor([generated_ids]).to(model.device)], dim=1)

    with torch.no_grad():
        outputs = model(full_sequence)
        logits = outputs.logits[0]  # Shape: (seq_len, vocab_size)

    # Extract logits for positions corresponding to generated tokens
    # The logit at position i predicts token i+1, so we want logits at positions [prompt_len-1 : prompt_len+num_generated-1]
    prompt_len = prompt_ids.shape[1]
    prefill_logits = [
        logits[prompt_len - 1 + i].cpu() for i in range(len(generated_ids))
    ]

    print(f"✓ Extracted {len(prefill_logits)} prefill logits")
    return prefill_logits


def compute_logit_differences(decode_logits, prefill_logits):
    """Compute L1, L2, and L-infinity norms of differences between decode and prefill probabilities.

    Computes norms for PDF and CDF across 4 slices:
    - top-100: top 100 probability tokens
    - top-1000: top 1000 probability tokens
    - full: full vocabulary
    - bottom-1000: bottom 1000 probability tokens

    Args:
        decode_logits: List of logits from decode phase
        prefill_logits: List of logits from prefill phase

    Returns:
        results: Dict with 'pdf' and 'cdf' sub-dicts, each containing norms for each slice
                 Each norm list is indexed by position (context length)
    """
    print(f"\nComputing probability differences...")

    results = {'pdf': {}, 'cdf': {}}
    vocab_size = decode_logits[0].shape[0]

    # Define slices: (name, slice_func)
    # slice_func takes (probs, diff) and returns the relevant slice of diff
    slices = [
        ('top100', lambda probs, diff: diff[torch.topk(probs, k=100).indices]),
        ('top1000',
         lambda probs, diff: diff[torch.topk(probs, k=1000).indices]),
        ('full', lambda probs, diff: diff),
        ('bottom1000', lambda probs, diff: diff[torch.topk(
            probs, k=1000, largest=False).indices]),
    ]

    # Initialize storage
    for dist_type in ['pdf', 'cdf']:
        for slice_name, _ in slices:
            results[dist_type][slice_name] = {
                'l1_norms': [],
                'l2_norms': [],
                'linf_norms': [],
            }

    for i, (dec_logits,
            pre_logits) in enumerate(zip(decode_logits, prefill_logits)):
        # Convert logits to probabilities (PDF)
        dec_probs = torch.softmax(dec_logits, dim=-1)
        pre_probs = torch.softmax(pre_logits, dim=-1)

        # PDF difference
        pdf_diff = dec_probs - pre_probs

        # CDF (cumulative)
        dec_cdf = torch.cumsum(dec_probs, dim=0)
        pre_cdf = torch.cumsum(pre_probs, dim=0)
        cdf_diff = dec_cdf - pre_cdf

        # Compute norms for each slice
        for slice_name, slice_func in slices:
            # PDF norms (use decode probs for selecting indices)
            pdf_slice = slice_func(dec_probs, pdf_diff)
            results['pdf'][slice_name]['l1_norms'].append(
                torch.norm(pdf_slice, p=1).item())
            results['pdf'][slice_name]['l2_norms'].append(
                torch.norm(pdf_slice, p=2).item())
            results['pdf'][slice_name]['linf_norms'].append(
                torch.norm(pdf_slice, p=float('inf')).item())

            # CDF norms (use decode probs for selecting indices)
            cdf_slice = slice_func(dec_probs, cdf_diff)
            results['cdf'][slice_name]['l1_norms'].append(
                torch.norm(cdf_slice, p=1).item())
            results['cdf'][slice_name]['l2_norms'].append(
                torch.norm(cdf_slice, p=2).item())
            results['cdf'][slice_name]['linf_norms'].append(
                torch.norm(cdf_slice, p=float('inf')).item())

    # Print statistics with percentiles
    print_divergence_stats(results, slices)

    return results


def print_divergence_stats(results, slices):
    """Print min, mean, median, 90th, and 100th percentile statistics for all norms."""
    norm_types = ['linf_norms', 'l1_norms', 'l2_norms']
    norm_labels = {'linf_norms': 'L-inf', 'l1_norms': 'L1', 'l2_norms': 'L2'}

    for dist_type in ['pdf', 'cdf']:
        print(f"\n  {dist_type.upper()} Statistics:")
        for slice_name, _ in slices:
            print(f"    {slice_name}:")
            for norm_type in norm_types:
                data = np.array(results[dist_type][slice_name][norm_type])
                stats = {
                    'min': np.min(data),
                    'mean': np.mean(data),
                    'median': np.median(data),
                    'p90': np.percentile(data, 90),
                    'p100': np.max(data),  # 100th percentile = max
                }
                print(
                    f"      {norm_labels[norm_type]:5s}: min={stats['min']:.2e}, mean={stats['mean']:.2e}, "
                    f"median={stats['median']:.2e}, p90={stats['p90']:.2e}, p100={stats['p100']:.2e}"
                )


def plot_divergence_over_context(all_results, output_dir: Path,
                                 model_name: str, num_prompts: int,
                                 num_tokens: int):
    """Plot how divergence changes over context length.

    Creates plots showing min, mean, median, 90th, and 100th percentiles of norms
    as a function of position (context length) for both PDF and CDF.
    """
    print(f"\nPlotting divergence over context length...")

    # Create timestamped subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    subdir = output_dir / f"context_divergence_{timestamp}"
    subdir.mkdir(parents=True, exist_ok=True)

    # Aggregate results across all prompts by position
    # Structure: aggregated[dist_type][slice_name][norm_type][position] = list of values across prompts
    slice_names = ['top100', 'top1000', 'full', 'bottom1000']
    norm_types = ['linf_norms', 'l1_norms', 'l2_norms']
    norm_labels = {'linf_norms': 'L-∞', 'l1_norms': 'L1', 'l2_norms': 'L2'}
    slice_labels = {
        'top100': 'Top-100',
        'top1000': 'Top-1000',
        'full': 'Full Vocab',
        'bottom1000': 'Bottom-1000'
    }

    # First, find the minimum length across all results
    min_len = min(
        len(results['pdf']['full']['l1_norms']) for results in all_results)

    # Aggregate by position
    aggregated = {'pdf': {}, 'cdf': {}}
    for dist_type in ['pdf', 'cdf']:
        for slice_name in slice_names:
            aggregated[dist_type][slice_name] = {}
            for norm_type in norm_types:
                # Collect values at each position across prompts
                by_position = []
                for pos in range(min_len):
                    values_at_pos = [
                        results[dist_type][slice_name][norm_type][pos]
                        for results in all_results
                    ]
                    by_position.append(values_at_pos)
                aggregated[dist_type][slice_name][norm_type] = by_position

    # Define window size for rolling statistics
    window_size = max(100, min_len // 100)  # Adaptive window size

    # Plot for each distribution type (PDF, CDF)
    for dist_type in ['pdf', 'cdf']:
        # Create figure: 3 rows (L-inf, L1, L2) x 4 cols (slices)
        fig, axes = plt.subplots(3, 4, figsize=(20, 12))
        fig.suptitle(
            f'{dist_type.upper()} Divergence vs Context Length ({num_prompts} prompts, {num_tokens} tokens)\n{model_name}',
            fontsize=14)

        for row, norm_type in enumerate(norm_types):
            for col, slice_name in enumerate(slice_names):
                ax = axes[row, col]
                data_by_pos = aggregated[dist_type][slice_name][norm_type]

                # Compute rolling statistics
                positions = []
                mins, means, medians, p90s, p100s = [], [], [], [], []

                for start in range(0, min_len - window_size + 1,
                                   window_size // 2):
                    end = min(start + window_size, min_len)
                    window_data = []
                    for pos in range(start, end):
                        window_data.extend(data_by_pos[pos])
                    window_data = np.array(window_data)

                    positions.append((start + end) / 2)
                    mins.append(np.min(window_data))
                    means.append(np.mean(window_data))
                    medians.append(np.median(window_data))
                    p90s.append(np.percentile(window_data, 90))
                    p100s.append(np.max(window_data))

                # Plot lines
                ax.plot(positions, mins, label='Min', alpha=0.8, linewidth=1)
                ax.plot(positions,
                        means,
                        label='Mean',
                        alpha=0.8,
                        linewidth=1.5)
                ax.plot(positions,
                        medians,
                        label='Median',
                        alpha=0.8,
                        linewidth=1.5)
                ax.plot(positions, p90s, label='P90', alpha=0.8, linewidth=1)
                ax.plot(positions,
                        p100s,
                        label='P100 (Max)',
                        alpha=0.8,
                        linewidth=1)

                ax.set_xlabel('Context Position')
                ax.set_ylabel(f'{norm_labels[norm_type]} Norm')
                ax.set_title(f'{slice_labels[slice_name]}')
                ax.set_yscale('log')
                ax.grid(True, alpha=0.3)
                if row == 0 and col == 3:
                    ax.legend(loc='upper right', fontsize=8)

        plt.tight_layout()
        output_path = subdir / f"{dist_type}_divergence_over_context.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved {dist_type.upper()} context plot to {output_path}")
        plt.close()

    # Also create a summary statistics table
    print_final_summary(aggregated, slice_names, norm_types, norm_labels,
                        subdir)

    print(f"\n✓ All context divergence plots saved to {subdir}")
    return subdir


def print_final_summary(aggregated, slice_names, norm_types, norm_labels,
                        output_dir):
    """Print and save a summary table of divergence statistics."""
    summary_lines = []
    summary_lines.append("=" * 100)
    summary_lines.append(
        "FINAL DIVERGENCE SUMMARY (across all positions and prompts)")
    summary_lines.append("=" * 100)

    for dist_type in ['pdf', 'cdf']:
        summary_lines.append(f"\n{dist_type.upper()}:")
        summary_lines.append("-" * 90)
        header = f"{'Slice':<15} {'Norm':<8} {'Min':>12} {'Mean':>12} {'Median':>12} {'P90':>12} {'P100':>12}"
        summary_lines.append(header)
        summary_lines.append("-" * 90)

        for slice_name in slice_names:
            for norm_type in norm_types:
                # Flatten all data for this slice/norm combo
                all_data = []
                for pos_data in aggregated[dist_type][slice_name][norm_type]:
                    all_data.extend(pos_data)
                all_data = np.array(all_data)

                row = (
                    f"{slice_name:<15} {norm_labels[norm_type]:<8} "
                    f"{np.min(all_data):>12.2e} {np.mean(all_data):>12.2e} "
                    f"{np.median(all_data):>12.2e} {np.percentile(all_data, 90):>12.2e} "
                    f"{np.max(all_data):>12.2e}")
                summary_lines.append(row)

    summary_lines.append("=" * 100)

    # Print to console
    for line in summary_lines:
        print(line)

    # Save to file
    summary_path = output_dir / "summary_stats.txt"
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    print(f"\n✓ Summary saved to {summary_path}")


def plot_histograms(all_results, output_dir: Path, model_name: str,
                    num_prompts: int):
    """Plot histograms of L-inf, L1, L2 norms for PDF and CDF.

    Creates 2 figures:
    - PDF: 3 rows (L-inf, L1, L2) x 4 cols (top-100, top-1000, full, bottom-1000)
    - CDF: 3 rows (L-inf, L1, L2) x 4 cols (top-100, top-1000, full, bottom-1000)
    """
    print(f"\nPlotting histograms...")

    # Create timestamped subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    subdir = output_dir / f"run_{timestamp}"
    subdir.mkdir(parents=True, exist_ok=True)

    # Aggregate results across all prompts
    # Structure: aggregated[dist_type][slice_name][norm_type] = list of values
    aggregated = {'pdf': {}, 'cdf': {}}
    slice_names = ['top100', 'top1000', 'full', 'bottom1000']

    for dist_type in ['pdf', 'cdf']:
        for slice_name in slice_names:
            aggregated[dist_type][slice_name] = {
                'l1_norms': [],
                'l2_norms': [],
                'linf_norms': [],
            }
            for results in all_results:
                aggregated[dist_type][slice_name]['l1_norms'].extend(
                    results[dist_type][slice_name]['l1_norms'])
                aggregated[dist_type][slice_name]['l2_norms'].extend(
                    results[dist_type][slice_name]['l2_norms'])
                aggregated[dist_type][slice_name]['linf_norms'].extend(
                    results[dist_type][slice_name]['linf_norms'])

    # Norm types in order: L-inf, L1, L2
    norm_types = ['linf_norms', 'l1_norms', 'l2_norms']
    norm_labels = ['L-∞ Norm', 'L1 Norm', 'L2 Norm']
    slice_labels = ['Top-100', 'Top-1000', 'Full Vocab', 'Bottom-1000']

    # Plot PDF figure: 3 rows x 4 cols
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    fig.suptitle(
        f'PDF Difference: Decode vs Prefill ({num_prompts} prompts)\n{model_name}',
        fontsize=14)

    for row, (norm_type, norm_label) in enumerate(zip(norm_types,
                                                      norm_labels)):
        for col, (slice_name,
                  slice_label) in enumerate(zip(slice_names, slice_labels)):
            ax = axes[row, col]
            norms = aggregated['pdf'][slice_name][norm_type]

            # Plot histogram with percentage weights
            weights = np.ones_like(norms) * 100 / len(norms)
            ax.hist(norms,
                    bins=50,
                    weights=weights,
                    alpha=0.7,
                    edgecolor='black',
                    color='blue')
            ax.set_yscale('log')
            ax.set_ylim(0.01, 100)  # 0.01% to 100%
            ax.set_xlabel(f'{norm_label}')
            ax.set_ylabel('Percentage (log scale)')
            ax.set_title(
                f'{slice_label}\n(mean={np.mean(norms):.2e}, max={np.max(norms):.2e})'
            )
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = subdir / "pdf_norms.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to {output_path}")
    plt.close()

    # Plot CDF figure: 3 rows x 4 cols
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    fig.suptitle(
        f'CDF Difference: Decode vs Prefill ({num_prompts} prompts)\n{model_name}',
        fontsize=14)

    for row, (norm_type, norm_label) in enumerate(zip(norm_types,
                                                      norm_labels)):
        for col, (slice_name,
                  slice_label) in enumerate(zip(slice_names, slice_labels)):
            ax = axes[row, col]
            norms = aggregated['cdf'][slice_name][norm_type]

            # Plot histogram with percentage weights
            weights = np.ones_like(norms) * 100 / len(norms)
            ax.hist(norms,
                    bins=50,
                    weights=weights,
                    alpha=0.7,
                    edgecolor='black',
                    color='green')
            ax.set_yscale('log')
            ax.set_ylim(0.01, 100)  # 0.01% to 100%
            ax.set_xlabel(f'{norm_label}')
            ax.set_ylabel('Percentage (log scale)')
            ax.set_title(
                f'{slice_label}\n(mean={np.mean(norms):.2e}, max={np.max(norms):.2e})'
            )
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = subdir / "cdf_norms.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to {output_path}")
    plt.close()

    print(f"\n✓ All plots saved to {subdir}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare prefill vs decode logits")
    parser.add_argument("--model",
                        default="mistralai/Mistral-7B-Instruct-v0.2",
                        help="Model to use")
    parser.add_argument("--num-tokens",
                        type=int,
                        default=10000,
                        help="Number of tokens to generate per prompt")
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=None,
        help="Custom prompts to test (uses DEFAULT_PROMPTS if not specified)")
    parser.add_argument("--output-dir",
                        type=Path,
                        default=Path("data/results/prefill_decode_comparison"),
                        help="Directory to save results")
    parser.add_argument("--skip-bucketed-analysis",
                        action="store_true",
                        help="Skip automatic bucketed probability analysis")
    parser.add_argument(
        "--quantization",
        type=str,
        default="fp32",
        nargs="+",
        choices=["4bit", "8bit", "bf16", "fp32"],
        help=
        "Quantization/precision level(s). Can specify multiple to run as experiment."
    )

    args = parser.parse_args()

    # Normalize quantization to list
    quant_levels = args.quantization if isinstance(
        args.quantization, list) else [args.quantization]

    # Use default prompts if none specified
    prompts = args.prompts if args.prompts else DEFAULT_PROMPTS
    print(f"prompts: {prompts}")
    prompts = prompts[:3]

    print("=" * 60)
    print("Prefill vs Decode Logit Comparison")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Quantization levels: {quant_levels}")
    print(f"Num prompts: {len(prompts)}")
    print(f"Num tokens per prompt: {args.num_tokens}")
    print(f"Slices: top-100, top-1000, full vocab, bottom-1000")
    print("=" * 60)

    # Run experiment for each quantization level
    for quant_idx, quantization in enumerate(quant_levels):
        print(f"\n{'#' * 70}")
        print(
            f"# QUANTIZATION {quant_idx + 1}/{len(quant_levels)}: {quantization}"
        )
        print(f"{'#' * 70}")

        # Load model with appropriate quantization
        print(f"\nLoading model with quantization={quantization}...")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        load_kwargs = {
            "device_map": "auto" if device == "cuda" else None,
        }

        if quantization == "fp32":
            load_kwargs["torch_dtype"] = torch.float32
        elif quantization == "bf16":
            load_kwargs["torch_dtype"] = torch.bfloat16
        elif quantization == "8bit":
            load_kwargs["load_in_8bit"] = True
        elif quantization == "4bit":
            load_kwargs["load_in_4bit"] = True

        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model.eval()
        print(f"✓ Loaded model on {device} (quantization={quantization})")

        # Run experiment for each prompt
        all_results = []

        # Create directory for saved probabilities (include quantization in path)
        quant_suffix = f"_{quantization}"
        probs_dir = args.output_dir / f"saved_probabilities{quant_suffix}"
        probs_dir.mkdir(parents=True, exist_ok=True)

        for i, prompt in enumerate(prompts):
            print(f"\n{'='*60}")
            print(f"[{quantization}] Prompt {i+1}/{len(prompts)}")
            print(f"{'='*60}")
            print(
                f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
            print(f"{'='*60}")

            # Generate tokens and get decode logits (using KV cache)
            generated_ids, decode_logits = generate_and_store_decode_logits(
                model, tokenizer, prompt, args.num_tokens)

            # Get prefill logits
            prefill_logits = get_prefill_logits(model, tokenizer, prompt,
                                                generated_ids)

            # Convert logits to probabilities (convert to float32 for numpy compatibility)
            decode_probs = [
                torch.softmax(logits, dim=-1).float().numpy()
                for logits in decode_logits
            ]
            prefill_probs = [
                torch.softmax(logits, dim=-1).float().numpy()
                for logits in prefill_logits
            ]

            # Save probabilities to npz file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            npz_path = probs_dir / f"probs_prompt{i}_{timestamp}.npz"

            np.savez_compressed(
                npz_path,
                decode_probs=np.array(
                    decode_probs),  # Shape: (num_tokens, vocab_size)
                prefill_probs=np.array(
                    prefill_probs),  # Shape: (num_tokens, vocab_size)
                generated_ids=np.array(generated_ids),  # Shape: (num_tokens,)
                prompt=prompt,
                model_name=args.model,
                quantization=quantization,
            )
            print(f"✓ Saved probabilities to {npz_path}")

            # Run bucketed analysis on the saved probabilities
            if BUCKETED_ANALYSIS_AVAILABLE and not args.skip_bucketed_analysis:
                print(f"\nRunning bucketed analysis for prompt {i}...")
                try:
                    # Load the data we just saved
                    data = load_prob_file(npz_path)
                    vocab_size = data['decode_probs'].shape[1]

                    # Default bucket sizes (powers of 2)
                    bucket_sizes = [
                        1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048,
                        4096, 8192, 16384, 32768, 65536
                    ]
                    valid_bucket_sizes = [
                        L for L in bucket_sizes if L <= vocab_size
                    ]

                    # Analyze bucket norms
                    bucketed_results = analyze_bucket_sizes(
                        data, valid_bucket_sizes)

                    # Analyze Spearman correlations
                    spearman_top_n = [10, 100, 1000, vocab_size]
                    spearman_results = analyze_spearman_correlations(
                        data, spearman_top_n)

                    # Analyze contiguous ranges (for multiple token positions)
                    token_positions = [0, 10, 100, 500]
                    valid_positions = [
                        pos for pos in token_positions if pos < args.num_tokens
                    ]
                    if len(valid_positions) < len(token_positions):
                        print(
                            f"  Note: Some token positions exceed num_tokens {args.num_tokens}, using {valid_positions}"
                        )

                    epsilon_values = np.logspace(-10, 0,
                                                 15)  # Use 15 points for speed
                    all_decode_ranges = []
                    all_prefill_ranges = []

                    for token_pos in valid_positions:
                        print(
                            f"  Analyzing contiguous ranges for token position {token_pos}..."
                        )
                        decode_ranges = analyze_contiguous_ranges(
                            data['decode_probs'][token_pos], epsilon_values)
                        prefill_ranges = analyze_contiguous_ranges(
                            data['prefill_probs'][token_pos], epsilon_values)
                        all_decode_ranges.append(decode_ranges)
                        all_prefill_ranges.append(prefill_ranges)

                    # Create bucketed analysis output directory
                    bucketed_dir = probs_dir.parent / f"bucketed_analysis{quant_suffix}"
                    bucketed_dir.mkdir(exist_ok=True, parents=True)

                    # Plot norms vs bucket size
                    model_short = args.model.split('/')[-1]
                    plot_path = bucketed_dir / f"norms_vs_bucket_size_prompt{i}_{timestamp}.png"
                    plot_norms_vs_bucket_size(bucketed_results, plot_path,
                                              model_short, args.num_tokens)

                    # Plot Spearman correlations
                    spearman_plot_path = bucketed_dir / f"spearman_correlations_prompt{i}_{timestamp}.png"
                    plot_spearman_correlations(spearman_results,
                                               spearman_plot_path, model_short,
                                               args.num_tokens)

                    # Plot contiguous ranges
                    contiguous_plot_path = bucketed_dir / f"contiguous_ranges_prompt{i}_{timestamp}.png"
                    plot_contiguous_ranges(all_decode_ranges,
                                           all_prefill_ranges, epsilon_values,
                                           contiguous_plot_path, model_short,
                                           valid_positions)

                    # Save bucketed results
                    results_npz_path = bucketed_dir / f"bucketed_results_prompt{i}_{timestamp}.npz"
                    save_dict = {
                        'bucket_sizes': np.array(valid_bucket_sizes),
                        'vocab_size': vocab_size,
                        'num_tokens': args.num_tokens,
                        'model_name': args.model,
                        'prompt': prompt,
                        'quantization': quantization,
                    }
                    for L in valid_bucket_sizes:
                        save_dict[f'L{L}_linf'] = np.array(
                            bucketed_results[L]['linf'])
                        save_dict[f'L{L}_l1'] = np.array(
                            bucketed_results[L]['l1'])
                        save_dict[f'L{L}_l2'] = np.array(
                            bucketed_results[L]['l2'])
                    # Add Spearman correlations
                    save_dict['spearman_top_n_values'] = np.array(
                        spearman_top_n)
                    for N in spearman_top_n:
                        save_dict[f'spearman_top{N}'] = np.array(
                            spearman_results[N])
                    # Add contiguous range results
                    save_dict['contiguous_epsilon_values'] = epsilon_values
                    save_dict['contiguous_token_positions'] = np.array(
                        valid_positions)
                    for pos_idx, pos in enumerate(valid_positions):
                        save_dict[
                            f'contiguous_decode_ranges_pos{pos}'] = all_decode_ranges[
                                pos_idx]
                        save_dict[
                            f'contiguous_prefill_ranges_pos{pos}'] = all_prefill_ranges[
                                pos_idx]
                    np.savez_compressed(results_npz_path, **save_dict)
                    print(f"✓ Saved bucketed analysis to {bucketed_dir}/")

                except Exception as e:
                    print(f"⚠️  Bucketed analysis failed: {e}")
            else:
                print("⚠️  Skipping bucketed analysis (import failed)")

            # Compute differences
            results = compute_logit_differences(decode_logits, prefill_logits)
            all_results.append(results)

            # Free up memory
            del decode_logits, prefill_logits, decode_probs, prefill_probs, generated_ids
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Plot divergence over context length (new primary analysis)
        plot_divergence_over_context(
            all_results, args.output_dir / quantization,
            args.model.split('/')[-1] + f" ({quantization})", len(prompts),
            args.num_tokens)

        # Also plot histograms for reference
        plot_histograms(all_results, args.output_dir / quantization,
                        args.model.split('/')[-1] + f" ({quantization})",
                        len(prompts))

        print(f"\n✓ Completed experiment for quantization={quantization}")

        # Clean up model to free memory before loading next quantization
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

    print("\n" + "=" * 60)
    print("✓ All experiments complete!")
    print(f"  Quantization levels tested: {quant_levels}")
    print("=" * 60)


if __name__ == "__main__":
    main()
