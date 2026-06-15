"""Plotting utilities for compression analysis."""

from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np


def compare_encoder_decoder_probs(enc_probs,
                                  dec_probs,
                                  tokens,
                                  tokenizer,
                                  save_plot: str = None):
    """Compare probability distributions from encoder and decoder."""
    import torch
    import matplotlib.pyplot as plt

    min_len = min(len(enc_probs), len(dec_probs))
    l2_diffs = [
        torch.linalg.vector_norm(enc_probs[i] - dec_probs[i], ord=2).item()
        for i in range(min_len)
    ]

    print(
        f"Prob comparison: L2 mean={np.mean(l2_diffs):.6f}, max={np.max(l2_diffs):.6f}, min={np.min(l2_diffs):.6f}"
    )

    if save_plot:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(l2_diffs, linewidth=2, color='red')
        ax.set_xlabel('Token Position')
        ax.set_ylabel('L2 Difference')
        ax.set_title('Encoder vs Decoder Probability Differences')
        ax.grid(True, alpha=0.3)
        plt.savefig(save_plot, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved plot to: {save_plot}")

    return l2_diffs


def plot_lora_comparison(all_lora_results: List[Dict], output_dir: Path,
                         dataset_name: str):
    """Generate comparison plots for multiple LoRAs on same dataset."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    lora_names = []
    avg_compressions = []
    avg_bpts = []
    colors = []

    for lora_result in all_lora_results:
        lora_name = lora_result['lora_name']
        is_matching = lora_result['is_matching']
        results = lora_result['results']

        successful = [r for r in results if r['success']]
        if not successful:
            continue

        avg_compression = sum(1.0 / r['compression_ratio']
                              for r in successful) / len(successful)
        avg_bpt = sum(r['bits_per_token']
                      for r in successful) / len(successful)

        lora_names.append(lora_name)
        avg_compressions.append(avg_compression)
        avg_bpts.append(avg_bpt)

        if lora_name == 'baseline':
            colors.append('gray')
        elif is_matching:
            colors.append('green')
        else:
            colors.append('blue')

    if not lora_names:
        print("No successful compressions to plot")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    x = np.arange(len(lora_names))
    bars1 = ax1.bar(x,
                    avg_compressions,
                    color=colors,
                    alpha=0.7,
                    edgecolor='black')
    ax1.set_xlabel('LoRA / Model')
    ax1.set_ylabel('Compression Ratio (x)')
    ax1.set_title(f'Average Compression Ratio\n{dataset_name}')
    ax1.set_xticks(x)
    ax1.set_xticklabels(lora_names, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars1, avg_compressions):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.02,
                 f'{val:.2f}x',
                 ha='center',
                 va='bottom',
                 fontsize=10)

    bars2 = ax2.bar(x, avg_bpts, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('LoRA / Model')
    ax2.set_ylabel('Bits per Token')
    ax2.set_title(f'Average Bits per Token\n{dataset_name}')
    ax2.set_xticks(x)
    ax2.set_xticklabels(lora_names, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars2, avg_bpts):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.1,
                 f'{val:.2f}',
                 ha='center',
                 va='bottom',
                 fontsize=10)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='gray', label='Baseline (No LoRA)'),
        Patch(facecolor='green', label='Matching Task LoRA'),
        Patch(facecolor='blue', label='Different Task LoRA')
    ]
    fig.legend(handles=legend_elements,
               loc='upper center',
               bbox_to_anchor=(0.5, 0.98),
               ncol=3)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = output_dir / f"lora_comparison_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved LoRA comparison plot to {plot_path}")
