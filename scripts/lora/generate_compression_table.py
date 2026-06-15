#!/usr/bin/env python3
"""
Generate LaTeX table with compression ratios from LoRA evaluation results.
"""

import argparse
import json
import math
from pathlib import Path
from transformers import AutoTokenizer

# Topic descriptions for each cluster (from previous analysis)
CLUSTER_TOPICS = {
    0: "General Chat",
    1: "Creative Writing",
    2: "Code/Technical",
    3: "Academic/Education",
    4: "Roleplay/Fiction",
    5: "Business/Professional",
    6: "Philosophy/Ethics",
    7: "Science/Math",
    8: "Translation/Language",
    9: "Casual Q\\&A",
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate LaTeX table from LoRA results")
    parser.add_argument("--results-json",
                        type=str,
                        required=True,
                        help="Path to the results JSON file")
    parser.add_argument("--base-model",
                        type=str,
                        default="mistralai/Mistral-7B-Instruct-v0.2",
                        help="Base model to get vocab size")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for LaTeX table (prints to stdout if not specified)")
    args = parser.parse_args()

    # Load results
    with open(args.results_json, 'r') as f:
        results = json.load(f)

    # Get vocab size from tokenizer
    print(f"Loading tokenizer from {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model,
                                              trust_remote_code=True)
    vocab_size = tokenizer.vocab_size
    bits_per_token_baseline = math.log2(vocab_size)

    print(f"Vocab size: {vocab_size}")
    print(f"Bits per token (uniform): {bits_per_token_baseline:.4f}")
    print()

    # Extract per-cluster results
    per_cluster = results.get('per_cluster_results', [])

    # Compute compression ratios
    table_rows = []

    for cluster_data in per_cluster:
        cluster_id = cluster_data['cluster_id']
        topic = CLUSTER_TOPICS.get(cluster_id, f"Cluster {cluster_id}")

        gzip_bpt = cluster_data.get('gzip_avg', 0)
        baseline_bpt = cluster_data['baseline_avg']
        correct_bpt = cluster_data['correct_lora_avg']
        wrong_bpt = cluster_data['wrong_lora_avg']

        # Compression ratio = BPT / log2(vocab_size)
        gzip_cr = gzip_bpt / bits_per_token_baseline * 100 if gzip_bpt > 0 else 0
        baseline_cr = baseline_bpt / bits_per_token_baseline * 100
        correct_cr = correct_bpt / bits_per_token_baseline * 100
        wrong_cr = wrong_bpt / bits_per_token_baseline * 100

        # Improvement multiplier = baseline_cr / lora_cr (vs LLM baseline)
        correct_improvement = baseline_cr / correct_cr if correct_cr > 0 else 0
        wrong_improvement = baseline_cr / wrong_cr if wrong_cr > 0 else 0
        # Improvement vs gzip
        gzip_vs_correct = gzip_cr / correct_cr if correct_cr > 0 else 0

        table_rows.append({
            'cluster_id': cluster_id,
            'topic': topic,
            'gzip_cr': gzip_cr,
            'baseline_cr': baseline_cr,
            'correct_cr': correct_cr,
            'wrong_cr': wrong_cr,
            'correct_improvement': correct_improvement,
            'wrong_improvement': wrong_improvement,
            'gzip_vs_correct': gzip_vs_correct,
        })

    # Compute averages
    avg_gzip_cr = sum(
        r['gzip_cr']
        for r in table_rows) / len(table_rows) if table_rows else 0
    avg_baseline_cr = sum(r['baseline_cr']
                          for r in table_rows) / len(table_rows)
    avg_correct_cr = sum(r['correct_cr'] for r in table_rows) / len(table_rows)
    avg_wrong_cr = sum(r['wrong_cr'] for r in table_rows) / len(table_rows)
    avg_correct_improvement = avg_baseline_cr / avg_correct_cr if avg_correct_cr > 0 else 0
    avg_wrong_improvement = avg_baseline_cr / avg_wrong_cr if avg_wrong_cr > 0 else 0
    avg_gzip_vs_correct = avg_gzip_cr / avg_correct_cr if avg_correct_cr > 0 else 0

    # Generate LaTeX table
    latex = []
    latex.append(r"\begin{table}[h]")
    latex.append(r"\centering")
    latex.append(
        r"\caption{Compression Ratio by Cluster (lower is better). Vocab size: "
        + f"{vocab_size:,}, " + r"$\log_2(\text{vocab}) = " +
        f"{bits_per_token_baseline:.2f}$ bits" + r"}")
    latex.append(r"\begin{tabular}{llcccc}")
    latex.append(r"\toprule")
    latex.append(
        r"Cluster & Topic & Gzip & Baseline & Correct LoRA & Wrong LoRA \\")
    latex.append(r"\midrule")

    for row in table_rows:
        gzip_str = f"{row['gzip_cr']:.1f}\\%" if row['gzip_cr'] > 0 else "--"
        latex.append(
            f"{row['cluster_id']} & {row['topic']} & "
            f"{gzip_str} & "
            f"{row['baseline_cr']:.1f}\\% & "
            f"{row['correct_cr']:.1f}\\% ({row['correct_improvement']:.2f}x) & "
            f"{row['wrong_cr']:.1f}\\% ({row['wrong_improvement']:.2f}x) \\\\")

    latex.append(r"\midrule")
    gzip_avg_str = f"{avg_gzip_cr:.1f}\\%" if avg_gzip_cr > 0 else "--"
    latex.append(
        f"\\textbf{{Average}} & & "
        f"\\textbf{{{gzip_avg_str}}} & "
        f"\\textbf{{{avg_baseline_cr:.1f}\\%}} & "
        f"\\textbf{{{avg_correct_cr:.1f}\\% ({avg_correct_improvement:.2f}x)}} & "
        f"\\textbf{{{avg_wrong_cr:.1f}\\% ({avg_wrong_improvement:.2f}x)}} \\\\"
    )
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\label{tab:compression_ratio}")
    latex.append(r"\end{table}")

    latex_str = "\n".join(latex)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(latex_str)
        print(f"Saved LaTeX table to {args.output}")
    else:
        print("\n" + "=" * 60)
        print("LaTeX Table:")
        print("=" * 60)
        print(latex_str)

    # Also print a plain text summary
    print("\n" + "=" * 60)
    print("Summary (plain text):")
    print("=" * 60)
    print(
        f"{'Cluster':<8} {'Topic':<22} {'Gzip':<10} {'Baseline':<10} {'Correct':<18} {'Wrong':<18}"
    )
    print("-" * 90)
    for row in table_rows:
        gzip_str = f"{row['gzip_cr']:>6.1f}%" if row[
            'gzip_cr'] > 0 else "    --"
        print(
            f"{row['cluster_id']:<8} {row['topic']:<22} "
            f"{gzip_str}    "
            f"{row['baseline_cr']:>6.1f}%    "
            f"{row['correct_cr']:>5.1f}% ({row['correct_improvement']:.2f}x)    "
            f"{row['wrong_cr']:>5.1f}% ({row['wrong_improvement']:.2f}x)")
    print("-" * 90)
    gzip_avg_str = f"{avg_gzip_cr:>6.1f}%" if avg_gzip_cr > 0 else "    --"
    print(f"{'Average':<8} {'':<22} "
          f"{gzip_avg_str}    "
          f"{avg_baseline_cr:>6.1f}%    "
          f"{avg_correct_cr:>5.1f}% ({avg_correct_improvement:.2f}x)    "
          f"{avg_wrong_cr:>5.1f}% ({avg_wrong_improvement:.2f}x)")


if __name__ == "__main__":
    main()
