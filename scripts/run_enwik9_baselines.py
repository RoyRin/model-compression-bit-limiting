#!/usr/bin/env python3
"""
Run compression baselines on enwik9 Wikipedia text.

Compares compression ratios across different models:
1. Mistral-7B-Instruct-v0.2
2. Llama-3.1-8B (base)
3. Llama-3.1-8B-Instruct
4. Qwen3-30B-A3B (MoE model)
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
import numpy as np

# Add repo root to path
_repo_root = Path(__file__).parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from transformers import AutoTokenizer, AutoModelForCausalLM

from compression.block_coder import (
    BlockEmissionArithmeticCoder,
    BlockEmissionArithmeticDecoder,
)
from compression.utils.config import load_compression_config

# Model configurations
MODELS = {
    "mistral-7b": {
        "hf_id": "mistralai/Mistral-7B-Instruct-v0.2",
        "dtype": torch.bfloat16,
        "description": "Mistral 7B Instruct v0.2",
    },
    "llama-3.1-8b": {
        "hf_id": "meta-llama/Llama-3.1-8B",
        "dtype": torch.bfloat16,
        "description": "Llama 3.1 8B (base)",
    },
    "llama-3.1-8b-instruct": {
        "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
        "dtype": torch.bfloat16,
        "description": "Llama 3.1 8B Instruct",
    },
    "qwen3-30b-a3b": {
        "hf_id": "Qwen/Qwen3-30B-A3B",
        "dtype": torch.bfloat16,
        "description": "Qwen3 30B-A3B (MoE, 3B active)",
    },
}


def extract_wikipedia_text(enwik9_path: Path,
                           num_chars: int = 100000,
                           skip_chars: int = 10000000) -> str:
    """Extract clean text from enwik9 XML dump.

    Args:
        enwik9_path: Path to enwik9 file
        num_chars: Number of characters to extract
        skip_chars: Number of characters to skip from start (to avoid XML header)

    Returns:
        Cleaned Wikipedia text
    """
    with open(enwik9_path, 'r', encoding='utf-8', errors='ignore') as f:
        # Skip initial XML/header content
        f.seek(skip_chars)
        raw = f.read(num_chars * 3)  # Read extra to have enough after cleaning

    # Basic XML cleanup
    # Remove XML tags
    text = re.sub(r'<[^>]+>', ' ', raw)
    # Remove wiki markup
    text = re.sub(r'\[\[(?:[^\]|]*\|)?([^\]]*)\]\]', r'\1',
                  text)  # [[link|text]] -> text
    text = re.sub(r'\{\{[^}]*\}\}', '', text)  # Remove templates
    text = re.sub(r"'''?", '', text)  # Remove bold/italic
    text = re.sub(r'&[a-z]+;', ' ', text)  # Remove HTML entities
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    return text[:num_chars]


def load_model_and_tokenizer(model_name: str, device: str = "cuda"):
    """Load model and tokenizer."""
    config = MODELS[model_name]
    hf_id = config["hf_id"]
    dtype = config["dtype"]

    print(f"Loading {config['description']} ({hf_id})...")

    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    return model, tokenizer


def compute_ce_bits_per_token(model,
                              tokenizer,
                              text: str,
                              max_tokens: Optional[int] = None) -> tuple:
    """Compute bits per token using cross-entropy loss (theoretical limit).

    Returns:
        Tuple of (bits_per_token, num_tokens, total_bits)
    """
    device = next(model.parameters()).device
    tokens = tokenizer.encode(text,
                              add_special_tokens=False,
                              return_tensors="pt").to(device)
    if max_tokens:
        tokens = tokens[:, :max_tokens]
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


def compute_compression_ratio(
    text: str,
    model,
    tokenizer,
    config: dict,
    max_tokens: Optional[int] = None,
    verbose: bool = False,
) -> Dict:
    """Compute compression ratio for given text using arithmetic coding.

    Returns dict with compression stats.
    """
    # Tokenize
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if max_tokens:
        tokens = tokens[:max_tokens]
    num_tokens = len(tokens)

    vocab_size = tokenizer.vocab_size
    orig_bits = num_tokens * np.log2(vocab_size)

    print(f"  Tokenized: {num_tokens} tokens (vocab: {vocab_size})")
    print(
        f"  Original bits (uniform): {orig_bits:.0f} ({np.log2(vocab_size):.2f} bits/token)"
    )

    # Compute CE bits/token as sanity check (theoretical lower bound)
    ce_bpt, _, ce_total = compute_ce_bits_per_token(model, tokenizer, text,
                                                    max_tokens)
    print(
        f"  Cross-entropy (theoretical): {ce_total:.0f} bits ({ce_bpt:.2f} bits/token)"
    )

    # Encode with higher precision for better compression
    encoder = BlockEmissionArithmeticCoder(
        model=model,
        tokenizer=tokenizer,
        bit_precision=config.get("bit_precision",
                                 120),  # Higher precision = more tokens/block
        bits_for_encoding_count=config.get("bits_for_encoding_count", 7),
        min_prob=config.get("min_prob", 1e-8),
        temperature=config.get("temperature", 1.0),
    )

    print(f"  Encoding {num_tokens} tokens... (this may take a few minutes)")
    sys.stdout.flush()

    # Handle tokenizers without BOS token (e.g., Qwen)
    if tokenizer.bos_token_id is not None:
        initial_context = None  # Let encoder use BOS default
    elif tokenizer.eos_token_id is not None:
        # Use EOS as fallback start token
        initial_context = [tokenizer.eos_token_id]
        print(
            f"  Note: No BOS token, using EOS ({tokenizer.eos_token_id}) as initial context"
        )
    else:
        # Last resort: use first token of the sequence as context
        initial_context = [tokens[0]]
        tokens = tokens[1:]  # Remove first token from encoding sequence
        print(f"  Note: No BOS/EOS, using first token as initial context")

    start_time = time.time()
    enc_buffer, _ = encoder.encode(tokens, initial_context=initial_context)
    encode_time = time.time() - start_time

    # Calculate compressed bits
    bits_per_block = encoder.bit_precision + encoder.bits_for_encoding_count
    compressed_bits = len(enc_buffer) * bits_per_block

    # Compression ratio
    compression_ratio = compressed_bits / orig_bits
    bits_per_token = compressed_bits / num_tokens

    print(
        f"  Encoding complete in {encode_time:.1f}s ({num_tokens/encode_time:.1f} tok/s)"
    )
    print(f"  Compressed bits: {compressed_bits:.0f}")
    print(f"  ====> COMPRESSION RATIO: {compression_ratio:.4f}")
    print(f"  ====> BITS PER TOKEN: {bits_per_token:.2f}")
    sys.stdout.flush()

    return {
        "num_tokens": num_tokens,
        "vocab_size": vocab_size,
        "original_bits": orig_bits,
        "compressed_bits": compressed_bits,
        "compression_ratio": compression_ratio,
        "bits_per_token": bits_per_token,
        "ce_bits_per_token": ce_bpt,  # Cross-entropy (theoretical lower bound)
        "encode_time_sec": encode_time,
        "tokens_per_sec": num_tokens / encode_time if encode_time > 0 else 0,
    }


def run_baseline(
    model_name: str,
    text: str,
    max_tokens: int = 1000,
    verbose: bool = False,
) -> Dict:
    """Run compression baseline for a single model."""
    print(f"\n{'='*60}")
    print(f"Model: {MODELS[model_name]['description']}")
    print(f"{'='*60}")

    # Load model
    model, tokenizer = load_model_and_tokenizer(model_name)

    # Load compression config
    config = load_compression_config()

    # Compute compression
    stats = compute_compression_ratio(
        text=text,
        model=model,
        tokenizer=tokenizer,
        config=config,
        max_tokens=max_tokens,
        verbose=verbose,
    )

    # Add model info
    stats["model_name"] = model_name
    stats["model_hf_id"] = MODELS[model_name]["hf_id"]
    stats["model_description"] = MODELS[model_name]["description"]

    # Print results
    print(f"\nResults:")
    print(f"  Tokens: {stats['num_tokens']}")
    print(f"  Original bits: {stats['original_bits']:.0f}")
    print(f"  Compressed bits: {stats['compressed_bits']:.0f}")
    print(f"  Compression ratio: {stats['compression_ratio']:.4f}")
    print(f"  Bits per token (actual): {stats['bits_per_token']:.2f}")
    print(f"  Bits per token (CE theory): {stats['ce_bits_per_token']:.2f}")
    print(
        f"  Encode time: {stats['encode_time_sec']:.1f}s ({stats['tokens_per_sec']:.1f} tok/s)"
    )

    # Free memory
    del model
    torch.cuda.empty_cache()

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Run enwik9 compression baselines")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS.keys()) + ["all"],
        default=["all"],
        help="Models to benchmark (default: all)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1000,
        help="Maximum tokens to compress (default: 1000)",
    )
    parser.add_argument(
        "--num-chars",
        type=int,
        default=50000,
        help="Number of characters to extract from enwik9 (default: 50000)",
    )
    parser.add_argument(
        "--enwik9-path",
        type=Path,
        default=Path("data/enwiki9/enwik9"),
        help="Path to enwik9 file",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    args = parser.parse_args()

    # Determine models to run
    if "all" in args.models:
        models_to_run = list(MODELS.keys())
    else:
        models_to_run = args.models

    # Extract text from enwik9
    print(f"Extracting {args.num_chars} chars from {args.enwik9_path}...")
    text = extract_wikipedia_text(args.enwik9_path, num_chars=args.num_chars)
    print(f"Extracted {len(text)} characters of clean text")
    print(f"Sample: {text[:200]}...")

    # Run baselines
    results = []
    for model_name in models_to_run:
        try:
            stats = run_baseline(
                model_name=model_name,
                text=text,
                max_tokens=args.max_tokens,
                verbose=args.verbose,
            )
            results.append(stats)
        except Exception as e:
            print(f"\nError running {model_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "model_name": model_name,
                "error": str(e),
            })

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(
        f"{'Model':<30} {'Comp Ratio':<12} {'Bits/Tok':<12} {'CE Bits/Tok':<12}"
    )
    print("-" * 80)
    for r in results:
        if "error" in r:
            print(f"{r['model_name']:<30} ERROR: {r['error'][:40]}")
        else:
            print(
                f"{r['model_description']:<30} {r['compression_ratio']:<12.4f} {r['bits_per_token']:<12.2f} {r['ce_bits_per_token']:<12.2f}"
            )

    # Save results
    if args.output_json:
        output = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "max_tokens": args.max_tokens,
                "num_chars": args.num_chars,
                "enwik9_path": str(args.enwik9_path),
            },
            "text_sample": text[:500],
            "results": results,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
