#!/usr/bin/env python3
"""
Compare probability distributions from teacher-forcing vs prefill modes.

Uses ModelProbabilityGenerator from probability_generator.py to compute
probabilities on pre-written text, then compares L2/L-inf norms.
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from compression.probability_generator import (
    ModelProbabilityGenerator,
    get_token_probabilities_prefill,
)

DEFAULT_TEXTS = [
    "The quick brown fox jumps over the lazy dog. This is a simple test sentence to verify that the probability distributions match between different inference modes.",
    "In machine learning, neural networks are computational models inspired by biological neural networks. They consist of layers of interconnected nodes that process information.",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n\n# Calculate first 10 numbers\nfor i in range(10):\n    print(fibonacci(i))",
]


def get_teacher_forcing_probs(model, tokenizer, tokens, temperature=1.0):
    """
    Get probability distributions using teacher-forcing (incremental) mode.

    For each position i, returns P(tokens[i] | tokens[:i])
    """
    gen = ModelProbabilityGenerator(
        model,
        tokenizer=tokenizer,
        temperature=temperature,
        use_cache=True,
        keep_on_device=True,
    )

    # Start with BOS token as context
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokens[
        0]
    gen.reset_teacher_forcing([bos_id])
    gen.compute_token_prob()

    probs = []
    # First distribution: P(tokens[0] | BOS)
    probs.append(gen.get_token_probability().cpu().clone())

    # For each subsequent token, add it and get distribution for next
    for i, token in enumerate(tokens[:-1]):
        gen.add_next_token_teacher_forcing(token)
        probs.append(gen.get_token_probability().cpu().clone())

    return probs


def get_prefill_probs_model_probability_generator(model,
                                                  tokenizer,
                                                  tokens,
                                                  temperature=1.0):
    """
    Get probability distributions using ModelProbabilityGenerator's prefill mode.

    This mimics exactly what block_coder.py does with prefill, so we can compare
    it against the direct get_token_probabilities_prefill() call.
    """
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokens[
        0]
    initial_context = [bos_id]

    # Create ModelProbabilityGenerator (same as block_coder does)
    model_generator = ModelProbabilityGenerator(
        model,
        tokenizer=tokenizer,
        temperature=temperature,
        use_cache=True,
        keep_on_device=True,
    )

    # Prefill: [BOS] + tokens[:-1] to get distributions for all positions
    # This is exactly what block_coder.py does in encode with use_prefill=True
    tokens_with_context = initial_context + list(tokens[:-1])
    model_generator.prefill(tokens_with_context)

    # Set start index (same as block_coder.py)
    start_idx = len(initial_context) - 1
    model_generator.current_index = start_idx
    model_generator._last_probs = model_generator.distributions[start_idx]

    # Collect probabilities the same way block_coder's encode loop does
    probs = []
    for i in range(len(tokens)):
        # Get the probability FIRST (before advancing)
        probs.append(model_generator.get_token_probability().cpu().clone())
        # Then advance to next distribution
        model_generator.add_token_prefill()

    return probs


def get_prefill_probs(model, tokenizer, tokens, temperature=1.0):
    """
    Get probability distributions using prefill (batch) mode.

    For each position i, returns P(tokens[i] | tokens[:i])
    """
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokens[
        0]

    # Prefill: [BOS] + tokens[:-1] to get distributions for all positions
    # distributions[i] = P(next | tokens[:i+1])
    # So distributions[0] = P(tokens[0] | BOS)
    #    distributions[1] = P(tokens[1] | BOS, tokens[0])
    #    etc.
    tokens_with_bos = [bos_id] + list(tokens[:-1])

    probs = get_token_probabilities_prefill(
        model,
        tokens_with_bos,
        temperature=temperature,
        use_cache=False,
        keep_on_device=False,
    )

    return probs


def compare_distributions(teacher_probs, prefill_probs):
    """
    Compare two lists of probability distributions.

    Returns dict with L2 and L-inf norms for each position.
    """
    assert len(teacher_probs) == len(prefill_probs), \
        f"Length mismatch: {len(teacher_probs)} vs {len(prefill_probs)}"

    l2_norms = []
    linf_norms = []

    for i, (tp, pp) in enumerate(zip(teacher_probs, prefill_probs)):
        diff = (tp - pp).abs()
        l2 = torch.sqrt((diff**2).sum()).item()
        linf = diff.max().item()
        l2_norms.append(l2)
        linf_norms.append(linf)

    return {
        'l2_norms': l2_norms,
        'linf_norms': linf_norms,
        'l2_max': max(l2_norms),
        'l2_mean': np.mean(l2_norms),
        'linf_max': max(linf_norms),
        'linf_mean': np.mean(linf_norms),
        'num_positions': len(l2_norms),
    }


def main():
    parser = argparse.ArgumentParser(
        description=
        "Compare teacher-forcing vs prefill probability distributions")
    parser.add_argument("--model",
                        default="mistralai/Mistral-7B-Instruct-v0.2",
                        help="Model to use")
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Text to process (uses default texts if not specified)")
    parser.add_argument("--dtype",
                        choices=["fp32", "bf16", "fp16"],
                        default="fp32",
                        help="Data type for model")
    parser.add_argument("--temperature",
                        type=float,
                        default=1.0,
                        help="Temperature for softmax")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum number of tokens to process (None = all)")
    parser.add_argument("--show-per-position",
                        action="store_true",
                        help="Show norms for each position")

    args = parser.parse_args()

    # Select texts
    texts = [args.text] if args.text else DEFAULT_TEXTS

    print("=" * 70)
    print("Teacher-Forcing vs Prefill Comparison")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Dtype: {args.dtype}")
    print(f"Temperature: {args.temperature}")
    print(f"Num texts: {len(texts)}")
    print("=" * 70)

    # Load model
    dtype_map = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }

    print(f"\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype_map[args.dtype],
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model,
                                              trust_remote_code=True)
    model.eval()
    print(f"Model loaded on {next(model.parameters()).device}")

    # Process each text
    all_results = []

    for i, text in enumerate(texts):
        print(f"\n{'='*70}")
        print(f"Text {i+1}/{len(texts)}")
        print(f"{'='*70}")
        print(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}")

        # Tokenize
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if args.max_tokens:
            tokens = tokens[:args.max_tokens]
        print(f"Tokens: {len(tokens)}")

        # Get distributions from all three modes
        print("Computing teacher-forcing distributions...")
        with torch.no_grad():
            teacher_probs = get_teacher_forcing_probs(
                model, tokenizer, tokens, temperature=args.temperature)
        print(f"  Got {len(teacher_probs)} distributions")

        print("Computing direct prefill distributions...")
        with torch.no_grad():
            prefill_probs = get_prefill_probs(model,
                                              tokenizer,
                                              tokens,
                                              temperature=args.temperature)
        print(f"  Got {len(prefill_probs)} distributions")

        print("Computing ModelProbabilityGenerator prefill distributions...")
        with torch.no_grad():
            prefill_mpg_probs = get_prefill_probs_model_probability_generator(
                model, tokenizer, tokens, temperature=args.temperature)
        print(f"  Got {len(prefill_mpg_probs)} distributions")

        # Compare all pairs
        results_tf_vs_prefill = compare_distributions(teacher_probs,
                                                      prefill_probs)
        results_tf_vs_mpg = compare_distributions(teacher_probs,
                                                  prefill_mpg_probs)
        results_prefill_vs_mpg = compare_distributions(prefill_probs,
                                                       prefill_mpg_probs)

        all_results.append({
            'tf_vs_prefill': results_tf_vs_prefill,
            'tf_vs_mpg': results_tf_vs_mpg,
            'prefill_vs_mpg': results_prefill_vs_mpg,
        })

        print(f"\nResults for text {i+1}:")
        print(f"  Teacher-Forcing vs Direct Prefill:")
        print(
            f"    L2:    max={results_tf_vs_prefill['l2_max']:.6e}, mean={results_tf_vs_prefill['l2_mean']:.6e}"
        )
        print(
            f"    L-inf: max={results_tf_vs_prefill['linf_max']:.6e}, mean={results_tf_vs_prefill['linf_mean']:.6e}"
        )
        print(f"  Teacher-Forcing vs MPG Prefill:")
        print(
            f"    L2:    max={results_tf_vs_mpg['l2_max']:.6e}, mean={results_tf_vs_mpg['l2_mean']:.6e}"
        )
        print(
            f"    L-inf: max={results_tf_vs_mpg['linf_max']:.6e}, mean={results_tf_vs_mpg['linf_mean']:.6e}"
        )
        print(f"  Direct Prefill vs MPG Prefill:")
        print(
            f"    L2:    max={results_prefill_vs_mpg['l2_max']:.6e}, mean={results_prefill_vs_mpg['l2_mean']:.6e}"
        )
        print(
            f"    L-inf: max={results_prefill_vs_mpg['linf_max']:.6e}, mean={results_prefill_vs_mpg['linf_mean']:.6e}"
        )

        if args.show_per_position:
            print(
                f"\n  Per-position L2 norms (Direct Prefill vs MPG Prefill):")
            for pos in range(min(20, len(results_prefill_vs_mpg['l2_norms']))):
                token_id = tokens[pos] if pos < len(tokens) else -1
                token_str = tokenizer.decode([token_id
                                              ]) if token_id >= 0 else "?"
                print(
                    f"    pos={pos:3d} token={token_id:6d} '{token_str:15s}' "
                    f"L2={results_prefill_vs_mpg['l2_norms'][pos]:.6e} L-inf={results_prefill_vs_mpg['linf_norms'][pos]:.6e}"
                )
            if len(results_prefill_vs_mpg['l2_norms']) > 20:
                print(
                    f"    ... ({len(results_prefill_vs_mpg['l2_norms']) - 20} more positions)"
                )

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    # Extract max L2/L-inf for each comparison type
    tf_vs_prefill_l2 = [r['tf_vs_prefill']['l2_max'] for r in all_results]
    tf_vs_mpg_l2 = [r['tf_vs_mpg']['l2_max'] for r in all_results]
    prefill_vs_mpg_l2 = [r['prefill_vs_mpg']['l2_max'] for r in all_results]

    tf_vs_prefill_linf = [r['tf_vs_prefill']['linf_max'] for r in all_results]
    tf_vs_mpg_linf = [r['tf_vs_mpg']['linf_max'] for r in all_results]
    prefill_vs_mpg_linf = [
        r['prefill_vs_mpg']['linf_max'] for r in all_results
    ]

    print(f"Across {len(texts)} texts:")
    print(f"\n  Teacher-Forcing vs Direct Prefill:")
    print(
        f"    L2 max:    min={min(tf_vs_prefill_l2):.6e}, max={max(tf_vs_prefill_l2):.6e}, mean={np.mean(tf_vs_prefill_l2):.6e}"
    )
    print(
        f"    L-inf max: min={min(tf_vs_prefill_linf):.6e}, max={max(tf_vs_prefill_linf):.6e}, mean={np.mean(tf_vs_prefill_linf):.6e}"
    )

    print(f"\n  Teacher-Forcing vs MPG Prefill:")
    print(
        f"    L2 max:    min={min(tf_vs_mpg_l2):.6e}, max={max(tf_vs_mpg_l2):.6e}, mean={np.mean(tf_vs_mpg_l2):.6e}"
    )
    print(
        f"    L-inf max: min={min(tf_vs_mpg_linf):.6e}, max={max(tf_vs_mpg_linf):.6e}, mean={np.mean(tf_vs_mpg_linf):.6e}"
    )

    print(
        f"\n  Direct Prefill vs MPG Prefill (KEY - should be 0 if MPG is correct):"
    )
    print(
        f"    L2 max:    min={min(prefill_vs_mpg_l2):.6e}, max={max(prefill_vs_mpg_l2):.6e}, mean={np.mean(prefill_vs_mpg_l2):.6e}"
    )
    print(
        f"    L-inf max: min={min(prefill_vs_mpg_linf):.6e}, max={max(prefill_vs_mpg_linf):.6e}, mean={np.mean(prefill_vs_mpg_linf):.6e}"
    )

    # Interpretation
    print(f"\nInterpretation:")
    max_linf_prefill_mpg = max(prefill_vs_mpg_linf)
    if max_linf_prefill_mpg < 1e-10:
        print(
            "  ✓ Direct Prefill and MPG Prefill match exactly (L-inf < 1e-10)")
        print("    This means ModelProbabilityGenerator indexing is correct.")
    elif max_linf_prefill_mpg < 1e-6:
        print(
            "  ~ Direct Prefill and MPG Prefill have tiny differences (L-inf < 1e-6)"
        )
        print("    Likely numerical precision, not indexing issue.")
    else:
        print(
            f"  ✗ Direct Prefill and MPG Prefill differ significantly (L-inf = {max_linf_prefill_mpg:.2e})"
        )
        print(
            "    This indicates an INDEXING BUG in ModelProbabilityGenerator!")


if __name__ == "__main__":
    main()
