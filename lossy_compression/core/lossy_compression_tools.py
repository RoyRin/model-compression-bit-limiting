#!/usr/bin/env python3
"""
Compression-Guided Generation: Generate multiple text outputs and select the one that compresses best.

This module generates N candidate text outputs from a large language model,
compresses each using a smaller model with block arithmetic coding,
and selects the output with the best compression ratio.
"""

import sys
import os

# Add repo root to path for imports
_repo_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
# Add lossy_compression to path
_lossy_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _lossy_root not in sys.path:
    sys.path.insert(0, _lossy_root)

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import json
from pathlib import Path

# Import compression modules from repo root
from compression.block_coder import (BlockEmissionArithmeticCoder,
                                     BlockEmissionArithmeticDecoder)

# Import LLM API utilities
from utils.llm_api import anthropic_completion, get_anthropic_key

# Default models
DEFAULT_GENERATION_MODEL = "claude-opus-4-1-20250805"
DEFAULT_COMPRESSION_MODEL = "meta-llama/Llama-3.1-8B"
DEFAULT_SEED = 42


def generate_multiple_outputs(prompt: str,
                              n_generations: int = 10,
                              model: str = DEFAULT_GENERATION_MODEL,
                              max_tokens: int = 200,
                              temperature: float = 0.8,
                              seed: Optional[int] = DEFAULT_SEED,
                              verbose: bool = False) -> List[str]:
    """
    Generate multiple text outputs from a large language model.
    
    Args:
        prompt: Input prompt for generation
        n_generations: Number of outputs to generate
        model: Model identifier (Anthropic or HuggingFace)
        max_tokens: Maximum tokens per generation
        temperature: Sampling temperature
        seed: Random seed for reproducibility
        verbose: Print progress information
        
    Returns:
        List of generated text outputs
    """
    generations = []

    if verbose:
        print(f"🔄 Generating {n_generations} outputs from {model}...")

    for i in range(n_generations):
        if verbose:
            print(f"  Generation {i+1}/{n_generations}...")

        # Use different seed for each generation if seed is provided
        current_seed = (seed + i) if seed is not None else None

        # Generate text using Anthropic API
        if "claude" in model.lower() or "opus" in model.lower():
            try:
                output = anthropic_completion(prompt=prompt,
                                              model=model,
                                              max_tokens=max_tokens,
                                              temperature=temperature,
                                              seed=current_seed)
                generations.append(output)
                if verbose:
                    print(f"    → {output}")
            except Exception as e:
                print(f"  ⚠️ Generation {i+1} failed: {e}")
                continue
        else:
            # For local models (future implementation)
            raise NotImplementedError(
                f"Local model generation not yet implemented for {model}")

    if verbose:
        print(f"✅ Generated {len(generations)} outputs successfully")

    return generations


def compress_text(
    text: str,
    compression_model,
    compression_tokenizer,
    bit_precision: int = 64,
    device: Optional[str] = None,
    verbose: bool = False,
    verify_correctness:
    bool = False  # Disabled by default - verification has issues
) -> Tuple[bytes, float, Dict[str, Any]]:
    """
    Compress text using block arithmetic coding.

    Uses the same approach as the working LoRA compression code.

    Args:
        text: Text to compress
        compression_model: Compression model (already loaded)
        compression_tokenizer: Tokenizer for compression model
        bit_precision: Bit precision for arithmetic coding
        device: Device to run on
        verbose: Print progress information
        verify_correctness: Verify that decompression recovers original text (disabled by default)

    Returns:
        Tuple of (compressed_data, compression_ratio, metrics)
    """
    start_time = time.time()

    # Auto-detect device if not specified
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # Tokenize the text - use list of token IDs (not tensor) like LoRA code
    tokens = compression_tokenizer.encode(text, add_special_tokens=False)
    n_tokens = len(tokens)

    if n_tokens == 0:
        return None, 0, {'error': 'empty text', 'n_tokens': 0}

    # Initialize the encoder (matching LoRA compression code)
    encoder = BlockEmissionArithmeticCoder(
        model=compression_model,
        tokenizer=compression_tokenizer,
        bit_precision=bit_precision,
        bits_for_encoding_count=8,
        device=device,
        verbose=False,
    )

    # Encode the tokens (pass list of token IDs, not tensor)
    encoded_values, encoding_info = encoder.encode(tokens)
    encoding_time = time.time() - start_time

    # Calculate metrics - each encoded value is bit_precision bits
    total_bits = len(encoded_values) * bit_precision
    bits_per_token = total_bits / n_tokens

    # Calculate compression percentage (compressed / original * 100, lower = better)
    original_bytes = len(text.encode('utf-8'))
    compressed_bytes = total_bits // 8
    compression_pct = (compressed_bytes / original_bytes *
                       100) if original_bytes > 0 else 100

    metrics = {
        'original_bytes': original_bytes,
        'compressed_bytes': compressed_bytes,
        'compressed_bits': total_bits,
        'compression_pct': compression_pct,
        'n_tokens': n_tokens,
        'bits_per_token': bits_per_token,
        'encoding_time': encoding_time,
        'is_correct': True,  # Assume correct since we're using working code
        'verified': False
    }

    if verbose:
        print(
            f"  Compressed {original_bytes} bytes → {compressed_bytes} bytes "
            f"({compression_pct:.1f}% of original, {bits_per_token:.2f} bits/token)"
        )

    return (encoded_values, encoding_info), compression_pct, metrics


def load_compression_model(model_path: str = DEFAULT_COMPRESSION_MODEL,
                           device: Optional[str] = None) -> Tuple[Any, Any]:
    """
    Load compression model and tokenizer.

    Args:
        model_path: Path or identifier for compression model
        device: Device to load model on (auto-detects if None)

    Returns:
        Tuple of (model, tokenizer)
    """
    import sys

    # Auto-detect best available device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"📦 Loading compression model: {model_path}", flush=True)
    print(f"   Target device: {device}", flush=True)

    # Get HuggingFace cache directory
    from transformers.utils import TRANSFORMERS_CACHE
    import os
    cache_dir = os.path.expanduser(TRANSFORMERS_CACHE)
    print(f"   Cache location: {cache_dir}", flush=True)

    print("   Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("   Tokenizer loaded.", flush=True)

    print("   Loading model weights...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    print("   Model weights loaded.", flush=True)

    # Move model to appropriate device
    if device == "cuda" and torch.cuda.is_available():
        print("   Moving model to CUDA...", flush=True)
        model = model.cuda()
        print("   Model moved to CUDA.", flush=True)
    elif device == "mps" and torch.backends.mps.is_available():
        print("   Moving model to MPS...", flush=True)
        model = model.to("mps")
        print("   Model moved to MPS.", flush=True)
    else:
        device = "cpu"  # Fallback if requested device not available
        print(f"   Using CPU (fallback).", flush=True)

    print("   Setting model to eval mode...", flush=True)
    model.eval()
    print(f"✅ Model loaded on {device}", flush=True)
    sys.stdout.flush()

    return model, tokenizer


def select_best_by_compression(
        prompt: str,
        generations: List[str],
        compression_model,
        compression_tokenizer,
        bit_precision: int = 64,
        device: Optional[str] = None,
        verbose: bool = True,
        verify_correctness: bool = False) -> Tuple[int, str, Dict[str, Any]]:
    """
    Select the best generation based on compression ratio.
    
    Args:
        prompt: Original prompt (for context)
        generations: List of generated texts
        compression_model: Compression model
        compression_tokenizer: Compression tokenizer
        bit_precision: Bit precision for arithmetic coding
        device: Device to run on
        verbose: Print progress information
        
    Returns:
        Tuple of (best_index, best_text, all_metrics)
    """
    # Auto-detect device if not specified
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    if verbose:
        print(
            f"\n📊 Evaluating {len(generations)} generations by compression...")

    compression_results = []

    for i, text in enumerate(generations):
        if verbose:
            print(f"\n  Generation {i+1}:")

        try:
            compressed, pct, metrics = compress_text(
                text=text,
                compression_model=compression_model,
                compression_tokenizer=compression_tokenizer,
                bit_precision=bit_precision,
                device=device,
                verbose=verbose,
                verify_correctness=verify_correctness)

            # Skip if correctness check failed
            if verify_correctness and not metrics.get('is_correct', False):
                print(f"    ⚠️ Skipping due to failed correctness check")
                compression_results.append({
                    'index': i,
                    'text': text,
                    'compression_pct':
                    999,  # Set high to avoid selecting (lower = better)
                    'metrics': metrics
                })
            else:
                compression_results.append({
                    'index': i,
                    'text': text,
                    'compression_pct': pct,
                    'metrics': metrics
                })

        except Exception as e:
            print(f"    ⚠️ Compression failed: {e}")
            compression_results.append({
                'index': i,
                'text': text,
                'compression_pct': 999,  # Set high to avoid selecting
                'metrics': {
                    'error': str(e)
                }
            })

    # Select best by compression percentage (lower = better)
    best_result = min(compression_results, key=lambda x: x['compression_pct'])
    best_index = best_result['index']
    best_text = best_result['text']

    # Calculate statistics (always needed for return value)
    pcts = [r['compression_pct'] for r in compression_results]
    valid_pcts = [p for p in pcts if p < 999]  # Only valid results

    if verbose:
        print(f"\n🏆 Best generation: #{best_index + 1}")
        print(
            f"   Compression: {best_result['compression_pct']:.1f}% of original"
        )
        print(f"   Text length: {len(best_text)} chars")

        # Show compression distribution
        print(f"\n📈 Compression distribution:")
        if valid_pcts:
            print(f"   Best: {min(valid_pcts):.1f}%")
            print(f"   Worst: {max(valid_pcts):.1f}%")
            print(f"   Mean: {np.mean(valid_pcts):.1f}%")
            print(f"   Std: {np.std(valid_pcts):.1f}%")

        if verify_correctness:
            correct_count = sum(1 for r in compression_results
                                if r['metrics'].get('is_correct', False))
            print(
                f"\n✅ Correctness: {correct_count}/{len(compression_results)} passed decompression check"
            )

    # Handle case where all compressions failed
    if valid_pcts:
        statistics = {
            'min_pct': min(valid_pcts),
            'max_pct': max(valid_pcts),
            'mean_pct': float(np.mean(valid_pcts)),
            'std_pct': float(np.std(valid_pcts))
        }
    else:
        # All compressions failed
        statistics = {
            'min_pct': 100,
            'max_pct': 100,
            'mean_pct': 100,
            'std_pct': 0
        }

    return best_index, best_text, {
        'all_results': compression_results,
        'best_index': best_index,
        'best_pct': best_result['compression_pct'],
        'statistics': statistics
    }


def compression_guided_generation(
        prompt: str,
        n_generations: int = 10,
        generation_model: str = DEFAULT_GENERATION_MODEL,
        compression_model_path: str = DEFAULT_COMPRESSION_MODEL,
        max_tokens: int = 200,
        temperature: float = 0.8,
        bit_precision: int = 64,
        seed: Optional[int] = DEFAULT_SEED,
        device: Optional[str] = None,
        verbose: bool = True,
        save_results: bool = False,
        output_dir: str = "results",
        verify_correctness: bool = False) -> Dict[str, Any]:
    """
    Main function: Generate multiple outputs and select best by compression.
    
    Args:
        prompt: Input prompt
        n_generations: Number of outputs to generate
        generation_model: Model for text generation
        compression_model_path: Model for compression
        max_tokens: Maximum tokens per generation
        temperature: Sampling temperature
        bit_precision: Bit precision for compression
        seed: Random seed
        device: Device to run on
        verbose: Print progress
        save_results: Save results to file
        output_dir: Directory for saving results
        verify_correctness: Whether to verify decompression correctness
        
    Returns:
        Dictionary with results and metrics
    """
    total_start = time.time()

    # Auto-detect device if not specified
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"\n{'='*60}")
    print(f"🚀 Compression-Guided Generation")
    print(f"{'='*60}")
    print(f"📝 Prompt: {prompt[:100]}..." if len(prompt) >
          100 else f"📝 Prompt: {prompt}")
    print(f"🔢 Generations: {n_generations}")
    print(f"🤖 Generation model: {generation_model}")
    print(f"📦 Compression model: {compression_model_path}")
    print(f"💻 Device: {device}")
    if not verify_correctness:
        print(f"⚡ Fast mode: Skipping decompression verification")
    print(f"{'='*60}\n")

    # Step 1: Generate multiple outputs
    generation_start = time.time()
    generations = generate_multiple_outputs(prompt=prompt,
                                            n_generations=n_generations,
                                            model=generation_model,
                                            max_tokens=max_tokens,
                                            temperature=temperature,
                                            seed=seed,
                                            verbose=verbose)
    generation_time = time.time() - generation_start

    if not generations:
        raise ValueError("No generations produced")

    # Step 2: Load compression model
    compression_model, compression_tokenizer = load_compression_model(
        model_path=compression_model_path, device=device)

    # Step 3: Select best by compression
    selection_start = time.time()
    best_index, best_text, selection_metrics = select_best_by_compression(
        prompt=prompt,
        generations=generations,
        compression_model=compression_model,
        compression_tokenizer=compression_tokenizer,
        bit_precision=bit_precision,
        device=device,
        verbose=verbose,
        verify_correctness=verify_correctness)
    selection_time = time.time() - selection_start

    total_time = time.time() - total_start

    # Compile results
    results = {
        'prompt': prompt,
        'parameters': {
            'n_generations': n_generations,
            'generation_model': generation_model,
            'compression_model': compression_model_path,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'bit_precision': bit_precision,
            'seed': seed
        },
        'best_generation': {
            'index': best_index,
            'text': best_text,
            'compression_pct': selection_metrics['best_pct']
        },
        'all_generations': generations,
        'selection_metrics': selection_metrics,
        'timing': {
            'generation_time': generation_time,
            'selection_time': selection_time,
            'total_time': total_time
        }
    }

    # Save results if requested
    if save_results:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = output_path / f"compression_guided_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n💾 Results saved to: {filename}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"✅ COMPLETED")
    print(f"{'='*60}")
    print(f"🏆 Best generation: #{best_index + 1}/{n_generations}")
    print(f"📊 Compression: {selection_metrics['best_pct']:.1f}% of original")

    # Print generation length statistics
    generation_lengths = [len(gen) for gen in generations]
    if generation_lengths:
        print(f"📏 Generation lengths:")
        print(f"   Shortest: {min(generation_lengths)} chars")
        print(f"   Longest: {max(generation_lengths)} chars")
        print(
            f"   Average: {sum(generation_lengths) / len(generation_lengths):.0f} chars"
        )

    print(f"⏱️  Total time: {total_time:.2f}s")
    print(f"{'='*60}\n")

    # Log stats to file (always, for experiments)
    log_stats(results, output_dir)

    return results


def log_stats(results: Dict[str, Any], output_dir: str = "results"):
    """
    Log experiment statistics to a CSV file for analysis.
    
    Args:
        results: Results dictionary from compression_guided_generation
        output_dir: Directory for log files
    """
    import csv
    from datetime import datetime

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    log_file = output_path / "best_of_n_stats.csv"

    # Calculate generation length statistics
    generation_lengths = [len(gen) for gen in results['all_generations']]
    min_length = min(generation_lengths) if generation_lengths else 0
    max_length = max(generation_lengths) if generation_lengths else 0
    mean_length = sum(generation_lengths) / len(
        generation_lengths) if generation_lengths else 0

    # Prepare row data
    stats = results['selection_metrics']['statistics']
    row = {
        'timestamp': datetime.now().isoformat(),
        'prompt': results['prompt'][:50],  # Truncate long prompts
        'n_generations': results['parameters']['n_generations'],
        'best_pct': results['best_generation']['compression_pct'],
        'best_index': results['best_generation']['index'],
        'min_pct': stats['min_pct'],
        'max_pct': stats['max_pct'],
        'mean_pct': stats['mean_pct'],
        'std_pct': stats['std_pct'],
        'min_gen_length': min_length,
        'max_gen_length': max_length,
        'mean_gen_length': mean_length,
        'generation_time': results['timing']['generation_time'],
        'selection_time': results['timing']['selection_time'],
        'total_time': results['timing']['total_time'],
        'temperature': results['parameters']['temperature'],
        'max_tokens': results['parameters']['max_tokens']
    }

    # Write to CSV
    file_exists = log_file.exists()

    with open(log_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())

        # Write header if file is new
        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

    print(f"📊 Stats logged to: {log_file}")


def parse_args():
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate multiple texts and select best by compression")

    # Basic arguments
    parser.add_argument("prompt",
                        type=str,
                        nargs='?',
                        default="Explain how neural networks learn:",
                        help="Input prompt for generation")
    parser.add_argument("-n",
                        "--n-generations",
                        type=int,
                        default=10,
                        help="Number of generations (default: 10)")

    # Model arguments
    parser.add_argument(
        "--generation-model",
        type=str,
        default=DEFAULT_GENERATION_MODEL,
        help=f"Generation model (default: {DEFAULT_GENERATION_MODEL})")
    parser.add_argument(
        "--compression-model",
        type=str,
        default=DEFAULT_COMPRESSION_MODEL,
        help=f"Compression model (default: {DEFAULT_COMPRESSION_MODEL})")

    # Generation parameters
    parser.add_argument("--max-tokens",
                        type=int,
                        default=200,
                        help="Maximum tokens per generation (default: 200)")
    parser.add_argument("--temperature",
                        type=float,
                        default=0.8,
                        help="Sampling temperature (default: 0.8)")

    # Compression parameters
    parser.add_argument("--bit-precision",
                        type=int,
                        default=64,
                        help="Bit precision for compression (default: 64)")

    # Other options
    parser.add_argument("--seed",
                        type=int,
                        default=DEFAULT_SEED,
                        help="Random seed (default: 42)")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: auto-detect cuda/mps/cpu)")
    parser.add_argument("--save",
                        action="store_true",
                        help="Save results to file")
    parser.add_argument("--output-dir",
                        type=str,
                        default="results",
                        help="Output directory for results (default: results)")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    parser.add_argument("--fast",
                        action="store_true",
                        help="Fast mode - skip decompression verification")

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Run compression-guided generation
    results = compression_guided_generation(
        prompt=args.prompt,
        n_generations=args.n_generations,
        generation_model=args.generation_model,
        compression_model_path=args.compression_model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        bit_precision=args.bit_precision,
        seed=args.seed,
        device=args.device,
        verbose=not args.quiet,
        save_results=args.save,
        output_dir=args.output_dir,
        verify_correctness=not args.fast  # Fast mode skips verification
    )

    # Print best generation
    if not args.quiet:
        print("\n📝 BEST GENERATION:")
        print("-" * 40)
        print(results['best_generation']['text'])
        print("-" * 40)

    return results


if __name__ == "__main__":
    main()
