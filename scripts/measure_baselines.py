#!/usr/bin/env python3
"""Measure baseline compression methods (gzip, zstd) without requiring a language model.

This script provides a lightweight way to measure compression baselines on text datasets
without the overhead of loading large language models. Useful for quick comparisons.

Usage:
    # Measure gzip and zstd on a dataset
    python scripts/measure_baselines.py data/compression_dataset_*.yaml

    # Measure on a text file
    python scripts/measure_baselines.py data/sample.txt

    # Measure specific methods only
    python scripts/measure_baselines.py data/sample.txt --methods gzip

    # Save results to JSON
    python scripts/measure_baselines.py data/sample.txt --output-json results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import yaml
from tqdm import tqdm

try:
    from transformers import AutoTokenizer
except ImportError:
    print("❌ transformers not found. Install with: pip install transformers")
    sys.exit(1)

try:
    from compression.baseline_coders import create_baseline_compressor
except ImportError as e:
    print(
        "❌ Unable to import baseline coders. Ensure your PYTHONPATH is correct."
    )
    raise


def load_text_from_file(filepath: Path) -> str:
    """Load text from .txt or .json file."""
    if filepath.suffix.lower() == '.json':
        with open(filepath, 'r') as f:
            data = json.load(f)
            # Try common keys for text content
            for key in ['text', 'content', 'data', 'document']:
                if key in data:
                    return data[key]
            # If no known key, stringify the whole thing
            return json.dumps(data)
    else:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()


def load_dataset(dataset_path: Path) -> List[Dict]:
    """Load compression dataset YAML and extract samples.

    Note: We compress only the generated text (not the prompt), matching
    the behavior of measure_compression.py where the prompt is used as context.
    """
    with open(dataset_path, 'r') as f:
        data = yaml.safe_load(f)

    samples = []
    for sample in data.get('samples', []):
        prompt = sample.get('prompt', '')
        generated = sample.get('generated_text', '')
        # Compress only the generated text (not prompt)
        # This matches measure_compression.py where prompt is context
        samples.append({
            'text': generated,  # Only compress generated text
            'prompt': prompt,
            'generated': generated,
            'prompt_id': sample.get('prompt_id'),
            'max_new_tokens': sample.get('max_new_tokens'),
        })

    return samples


def compress_with_baseline(text: str,
                           tokenizer,
                           method: str,
                           verbose: bool = False) -> Dict:
    """Compress text with a baseline method.

    This compresses the actual UTF-8 text bytes (not token IDs) to provide
    realistic baseline compression metrics on natural language text.

    Args:
        text: Text to compress (UTF-8 string)
        tokenizer: Tokenizer to use (for token count only)
        method: Compression method ('gzip', 'zstd', etc.)
        verbose: Print detailed information

    Returns:
        Dictionary with compression metrics
    """
    # Tokenize to get token count (for bits-per-token metric)
    tokens = tokenizer.encode(text)
    num_tokens = len(tokens)

    # Convert text to UTF-8 bytes for compression
    text_bytes = text.encode('utf-8')
    original_bytes = len(text_bytes)

    if verbose:
        print(f"  Tokens: {num_tokens}")
        print(f"  Text bytes: {original_bytes}")

    # Get the compression library
    try:
        if method == 'gzip':
            import gzip
            # Encode
            t0 = time.time()
            compressed = gzip.compress(text_bytes, compresslevel=9)
            encode_time = time.time() - t0
            # Decode
            t0 = time.time()
            decompressed = gzip.decompress(compressed)
            decode_time = time.time() - t0

        elif method == 'zstd' or method == 'zstd-fast' or method == 'zstd-high':
            import zstandard as zstd
            level = 3 if method == 'zstd' else (
                1 if method == 'zstd-fast' else 19)
            compressor = zstd.ZstdCompressor(level=level)
            decompressor = zstd.ZstdDecompressor()
            # Encode
            t0 = time.time()
            compressed = compressor.compress(text_bytes)
            encode_time = time.time() - t0
            # Decode
            t0 = time.time()
            decompressed = decompressor.decompress(compressed)
            decode_time = time.time() - t0

        elif method == 'lz4':
            import lz4.frame
            # Encode
            t0 = time.time()
            compressed = lz4.frame.compress(text_bytes, compression_level=0)
            encode_time = time.time() - t0
            # Decode
            t0 = time.time()
            decompressed = lz4.frame.decompress(compressed)
            decode_time = time.time() - t0

        elif method == 'brotli':
            import brotli
            # Encode
            t0 = time.time()
            compressed = brotli.compress(text_bytes, quality=11)
            encode_time = time.time() - t0
            # Decode
            t0 = time.time()
            decompressed = brotli.decompress(compressed)
            decode_time = time.time() - t0

        elif method == 'lzma' or method == 'xz':
            import lzma
            # Encode
            t0 = time.time()
            compressed = lzma.compress(text_bytes,
                                       format=lzma.FORMAT_XZ,
                                       preset=6)
            encode_time = time.time() - t0
            # Decode
            t0 = time.time()
            decompressed = lzma.decompress(compressed)
            decode_time = time.time() - t0

        elif method == 'bzip2':
            import bz2
            # Encode
            t0 = time.time()
            compressed = bz2.compress(text_bytes, compresslevel=9)
            encode_time = time.time() - t0
            # Decode
            t0 = time.time()
            decompressed = bz2.decompress(compressed)
            decode_time = time.time() - t0

        else:
            return {
                'success': False,
                'error': f'Unknown method: {method}',
                'num_tokens': num_tokens
            }

    except ImportError as e:
        return {
            'success': False,
            'error': f'{method} package not installed: {e}',
            'num_tokens': num_tokens
        }
    except Exception as e:
        return {'success': False, 'error': str(e), 'num_tokens': num_tokens}

    # Verify
    success = decompressed == text_bytes

    # Calculate metrics
    compressed_bytes = len(compressed)
    compressed_bits = compressed_bytes * 8

    return {
        'success':
        success,
        'compression_ratio':
        compressed_bytes / original_bytes if original_bytes > 0 else 0,
        'bits_per_token':
        compressed_bits / num_tokens if num_tokens > 0 else 0,
        'encode_time':
        encode_time,
        'decode_time':
        decode_time,
        'encode_time_per_token':
        encode_time / num_tokens if num_tokens > 0 else 0,
        'decode_time_per_token':
        decode_time / num_tokens if num_tokens > 0 else 0,
        'encode_tokens_per_sec':
        num_tokens / encode_time if encode_time > 0 else 0,
        'decode_tokens_per_sec':
        num_tokens / decode_time if decode_time > 0 else 0,
        'num_tokens':
        num_tokens,
        'compressed_bytes':
        compressed_bytes,
        'original_bytes':
        original_bytes,
    }


def process_dataset(samples: List[Dict], tokenizer, methods: List[str],
                    args) -> List[Dict]:
    """Process a dataset with baseline compression methods.

    Args:
        samples: List of sample dicts with 'text' key
        tokenizer: Tokenizer to use
        methods: List of compression methods to test
        args: Command-line arguments

    Returns:
        List of result dicts
    """
    all_results = []

    print(f"\n{'='*60}")
    print(
        f"Processing {len(samples)} samples with {len(methods)} method(s): {', '.join(methods)}"
    )
    print(f"{'='*60}\n")

    for i, sample in enumerate(tqdm(samples, desc="Compressing samples")):
        result = {
            'sample_id': i,
            'prompt_id': sample.get('prompt_id'),
            'max_new_tokens': sample.get('max_new_tokens'),
            'methods': {}
        }

        for method in methods:
            if args.verbose:
                print(f"\nSample {i+1}/{len(samples)} - {method.upper()}")

            method_result = compress_with_baseline(sample['text'], tokenizer,
                                                   method, args.verbose)
            result['methods'][method] = method_result

        all_results.append(result)

    return all_results


def print_summary(all_results: List[Dict], methods: List[str]):
    """Print summary statistics for all methods."""
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}\n")

    for method in methods:
        # Gather stats
        successful = []
        failed = []
        for result in all_results:
            method_result = result['methods'][method]
            if method_result.get('success'):
                successful.append(method_result)
            else:
                failed.append(method_result)

        total = len(all_results)
        success_rate = len(successful) / total * 100 if total > 0 else 0

        print(f"{method.upper()}")
        print(f"  Success: {len(successful)}/{total} ({success_rate:.1f}%)")

        if failed:
            print(f"  Failed: {len(failed)}")
            for f in failed[:3]:  # Show first 3 errors
                print(f"    - {f.get('error', 'Unknown error')}")

        if successful:
            # Calculate average metrics
            avg_compression_ratio = sum(r['compression_ratio']
                                        for r in successful) / len(successful)
            avg_bits_per_token = sum(r['bits_per_token']
                                     for r in successful) / len(successful)
            avg_encode_time = sum(r['encode_time_per_token']
                                  for r in successful) / len(successful)
            avg_decode_time = sum(r['decode_time_per_token']
                                  for r in successful) / len(successful)

            compression_factor = 1.0 / avg_compression_ratio if avg_compression_ratio > 0 else 0

            print(
                f"  Avg compression: {compression_factor:.2f}x ({avg_compression_ratio:.3f} ratio)"
            )
            print(f"  Avg bits/token: {avg_bits_per_token:.2f}")
            print(f"  Avg encode: {avg_encode_time*1000:.2f}ms/tok")
            print(f"  Avg decode: {avg_decode_time*1000:.2f}ms/tok")

        print()


def main(argv: List[str] | None = None):
    parser = argparse.ArgumentParser(
        description=
        "Measure baseline compression (gzip, zstd) on text or datasets")
    parser.add_argument(
        "file",
        type=Path,
        help="Path to .txt, .json, or YAML dataset (compression_dataset_*.yaml)"
    )
    parser.add_argument(
        "--methods",
        nargs='+',
        choices=[
            'gzip', 'zstd', 'zstd-fast', 'zstd-high', 'lz4', 'brotli', 'lzma',
            'xz', 'bzip2'
        ],
        default=['gzip', 'zstd', 'lz4', 'brotli', 'lzma', 'bzip2'],
        help=
        "Which compression methods to test (default: gzip, zstd, lz4, brotli, lzma, bzip2)"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="HuggingFace tokenizer to use (default: Llama-3.2-1B)")
    parser.add_argument("--limit-samples",
                        type=int,
                        default=None,
                        metavar="N",
                        help="Limit to first N samples from dataset")
    parser.add_argument("--output-json",
                        metavar="PATH",
                        type=Path,
                        help="Write detailed results to JSON file")
    parser.add_argument("--output-yaml",
                        metavar="PATH",
                        type=Path,
                        help="Write detailed results to YAML file")
    parser.add_argument("--verbose",
                        action="store_true",
                        help="Print detailed information for each sample")

    args = parser.parse_args(argv)

    # Check file exists
    if not args.file.exists():
        print(f"❌ File not found: {args.file}")
        sys.exit(1)

    # Load tokenizer
    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"  Vocab size: {tokenizer.vocab_size:,}\n")

    # Determine if dataset or single file
    is_dataset = args.file.suffix.lower() in ['.yaml', '.yml']

    if is_dataset:
        # Load dataset
        print(f"Loading dataset: {args.file}")
        samples = load_dataset(args.file)

        if args.limit_samples:
            samples = samples[:args.limit_samples]
            print(f"Limited to {len(samples)} samples")

        # Process dataset
        all_results = process_dataset(samples, tokenizer, args.methods, args)

        # Print summary
        print_summary(all_results, args.methods)

        # Save results if requested
        if args.output_json:
            output = {
                'metadata': {
                    'created_at': datetime.now().isoformat(),
                    'dataset': str(args.file),
                    'tokenizer': args.tokenizer,
                    'methods': args.methods,
                    'total_samples': len(all_results),
                },
                'results': all_results
            }
            with open(args.output_json, 'w') as f:
                json.dump(output, f, indent=2)
            print(f"✓ Saved results to {args.output_json}")

        if args.output_yaml:
            output = {
                'metadata': {
                    'created_at': datetime.now().isoformat(),
                    'dataset': str(args.file),
                    'tokenizer': args.tokenizer,
                    'methods': args.methods,
                    'total_samples': len(all_results),
                },
                'results': all_results
            }
            with open(args.output_yaml, 'w') as f:
                yaml.dump(output, f, default_flow_style=False)
            print(f"✓ Saved results to {args.output_yaml}")

    else:
        # Single file
        print(f"Loading file: {args.file}")
        text = load_text_from_file(args.file)
        print(f"  Text length: {len(text):,} chars\n")

        # Compress with each method
        results = {}
        for method in args.methods:
            print(f"→ {method.upper()}")
            result = compress_with_baseline(text,
                                            tokenizer,
                                            method,
                                            verbose=True)
            results[method] = result

            if result['success']:
                compression_factor = 1.0 / result[
                    'compression_ratio'] if result[
                        'compression_ratio'] > 0 else 0
                print(
                    f"  ✓ {compression_factor:.2f}x compression | {result['bits_per_token']:.2f} bpt"
                )
                print(
                    f"    Encode: {result['encode_time']:.3f}s ({result['encode_time_per_token']*1000:.2f}ms/tok)"
                )
                print(
                    f"    Decode: {result['decode_time']:.3f}s ({result['decode_time_per_token']*1000:.2f}ms/tok)"
                )
            else:
                print(f"  ❌ {result.get('error', 'Failed')}")
            print()

        # Save single file results if requested
        if args.output_json:
            output = {
                'file': str(args.file),
                'tokenizer': args.tokenizer,
                'text_length': len(text),
                'results': results
            }
            with open(args.output_json, 'w') as f:
                json.dump(output, f, indent=2)
            print(f"✓ Saved results to {args.output_json}")


if __name__ == "__main__":
    main()
