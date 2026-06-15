#!/usr/bin/env python3
"""
Script to compress a text file using a language model with arithmetic coding.

Usage:
    python compress_text_file.py input.txt [--model MODEL] [--output OUTPUT]
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Default compression model
DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B"
DEFAULT_TEMPERATURE = 0.5

# Import compression modules
from compression.block_encoding_arithmetic_coder import (
    BlockEmissionArithmeticCoder, BlockEmissionArithmeticDecoder)
from compression.incremental_scaling_arithmetic_coder import (
    IncrementalScalingArithmeticCoder, IncrementalScalingArithmeticDecoder)


def load_model(model_path: str = DEFAULT_MODEL):
    """Load model and tokenizer."""
    print(f"📦 Loading model: {model_path}")

    # Auto-detect device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"💻 Using device: {device}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # Move model to device
    if device == "cuda":
        model = model.cuda()
    elif device == "mps":
        model = model.to("mps")

    model.eval()
    print(f"✅ Model loaded successfully")

    return model, tokenizer, device


def compress_file(input_file: str,
                  model_path: str = DEFAULT_MODEL,
                  output_file: str = None,
                  bit_precision: int = 64,
                  temperature: float = DEFAULT_TEMPERATURE,
                  verify: bool = True,
                  verbose: bool = True,
                  encoding_method: str = "block",
                  prefix: str = ""):
    """Compress a text file using arithmetic coding."""

    # Read input file
    print(f"\n📄 Reading file: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    original_size = len(text.encode('utf-8'))
    print(f"📊 Original size: {original_size:,} bytes")
    print(f"📝 Text length: {len(text):,} characters")

    # Load model
    model, tokenizer, device = load_model(model_path)

    # Tokenize
    print(f"\n🔤 Tokenizing text...")
    tokens = tokenizer.encode(text, return_tensors='pt').to(device)
    n_tokens = tokens.shape[1]
    print(f"🔢 Number of tokens: {n_tokens}")

    # Compress
    print(
        f"\n🗜️ Compressing with {bit_precision}-bit precision, temperature={temperature}, method={encoding_method}..."
    )
    if prefix:
        print(
            f"📝 Using prefix: '{prefix[:50]}{'...' if len(prefix) > 50 else ''}'"
        )
    start_time = time.time()

    if encoding_method == "block":
        encoder = BlockEmissionArithmeticCoder(model=model,
                                               tokenizer=tokenizer,
                                               bit_precision=bit_precision,
                                               device=device,
                                               verbose=verbose,
                                               use_fast=True,
                                               use_optimization=True,
                                               temperature=temperature)
    elif encoding_method == "incremental":
        encoder = IncrementalScalingArithmeticCoder(model=model,
                                                    tokenizer=tokenizer,
                                                    precision=bit_precision,
                                                    device=device,
                                                    temperature=temperature)
    else:
        raise ValueError(f"Unknown encoding method: {encoding_method}")

    encoded_buffer, encoding_info = encoder.encode(tokens[0],
                                                   initial_context=prefix)
    encoding_time = time.time() - start_time

    # Calculate compressed size
    compressed_bits = len(encoded_buffer) * bit_precision
    compressed_bytes = compressed_bits // 8
    compression_ratio = original_size / compressed_bytes if compressed_bytes > 0 else 0

    print(f"✅ Compression complete!")
    print(f"⏱️  Encoding time: {encoding_time:.2f} seconds")
    print(f"📦 Compressed size: {compressed_bytes:,} bytes")
    print(f"📈 Compression ratio: {compression_ratio:.2f}x")
    print(f"📊 Bits per token: {compressed_bits / n_tokens:.2f}")

    # Save compressed data
    if output_file is None:
        output_file = input_file.replace('.txt', '_compressed.bin')

    print(f"\n💾 Saving compressed data to: {output_file}")
    import pickle
    with open(output_file, 'wb') as f:
        pickle.dump(
            {
                'encoded_buffer': encoded_buffer,
                'encoding_info': encoding_info,
                'n_tokens': n_tokens,
                'bit_precision': bit_precision,
                'model_path': model_path,
                'original_size': original_size,
                'compressed_bytes': compressed_bytes,
                'compression_ratio': compression_ratio,
                'encoding_method': encoding_method,
                'prefix': prefix
            }, f)

    # Verify by decompressing
    if verify:
        print(f"\n🔍 Verifying compression by decompressing...")
        verify_start = time.time()

        if encoding_method == "block":
            decoder = BlockEmissionArithmeticDecoder(
                model=model,
                tokenizer=tokenizer,
                bit_precision=bit_precision,
                device=device,
                verbose=False,
                use_fast=True,
                use_optimization=True,
                temperature=temperature)
        elif encoding_method == "incremental":
            decoder = IncrementalScalingArithmeticDecoder(
                model=model,
                tokenizer=tokenizer,
                precision=bit_precision,
                device=device,
                temperature=temperature)

        decoded_tokens = decoder.decode((encoded_buffer, encoding_info),
                                        n_tokens,
                                        initial_context=prefix)
        verify_time = time.time() - verify_start

        # Check if tokens match
        original_tokens = tokens[0].tolist()
        if decoded_tokens == original_tokens:
            print(f"✅ Verification successful! Decompression is lossless.")

            # Decode to text
            decoded_text = tokenizer.decode(decoded_tokens,
                                            skip_special_tokens=True)

            # Check text match
            if decoded_text == text:
                print(f"✅ Text perfectly reconstructed!")
            else:
                print(
                    f"⚠️  Text slightly different (likely tokenizer artifacts)"
                )
                print(f"   Original length: {len(text)}")
                print(f"   Decoded length: {len(decoded_text)}")
        else:
            print(f"❌ Verification failed! Tokens don't match.")
            print(f"   Original tokens: {original_tokens[:10]}...")
            print(f"   Decoded tokens: {decoded_tokens[:10]}...")

        print(f"⏱️  Decoding time: {verify_time:.2f} seconds")

    # Print summary
    print(f"\n{'='*60}")
    print(f"📊 COMPRESSION SUMMARY")
    print(f"{'='*60}")
    print(f"File: {input_file}")
    print(f"Model: {model_path}")
    print(f"Original: {original_size:,} bytes")
    print(f"Compressed: {compressed_bytes:,} bytes")
    print(f"Ratio: {compression_ratio:.2f}x")
    print(f"Tokens: {n_tokens}")
    print(f"Bits/token: {compressed_bits / n_tokens:.2f}")
    print(f"Encoding time: {encoding_time:.2f}s")
    if verify:
        print(f"Decoding time: {verify_time:.2f}s")
    print(f"Output: {output_file}")
    print(f"{'='*60}")

    return {
        'input_file': input_file,
        'output_file': output_file,
        'original_size': original_size,
        'compressed_size': compressed_bytes,
        'compression_ratio': compression_ratio,
        'n_tokens': n_tokens,
        'bits_per_token': compressed_bits / n_tokens,
        'encoding_time': encoding_time,
        'verification_passed': verify
    }


def main():
    parser = argparse.ArgumentParser(
        description=
        "Compress a text file using language model arithmetic coding")

    parser.add_argument("input_file", help="Input text file to compress")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model to use for compression (default: {DEFAULT_MODEL})")
    parser.add_argument("--output",
                        "-o",
                        help="Output file (default: input_compressed.bin)")
    parser.add_argument(
        "--bit-precision",
        type=int,
        default=64,
        help="Bit precision for arithmetic coding (default: 64)")
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=
        f"Temperature for encoding/decoding (default: {DEFAULT_TEMPERATURE})")
    parser.add_argument(
        "--encoding-method",
        choices=["block", "incremental"],
        default="block",
        help=
        "Encoding method to use: 'block' (block emission) or 'incremental' (incremental scaling) (default: block)"
    )
    parser.add_argument(
        "--prefix-file",
        type=str,
        default=None,
        help=
        "Text file containing prefix/context to load into model before encoding (improves compression)"
    )
    parser.add_argument("--no-verify",
                        action="store_true",
                        help="Skip decompression verification")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")

    args = parser.parse_args()

    # Check input file exists
    if not os.path.exists(args.input_file):
        print(f"❌ Error: Input file '{args.input_file}' not found")
        sys.exit(1)

    # Load prefix from file if provided
    prefix = ""
    if args.prefix_file:
        if not os.path.exists(args.prefix_file):
            print(f"❌ Error: Prefix file '{args.prefix_file}' not found")
            sys.exit(1)
        with open(args.prefix_file, 'r', encoding='utf-8') as f:
            prefix = f.read()
        print(
            f"📝 Loaded prefix from {args.prefix_file} ({len(prefix)} characters)"
        )

    # Run compression
    compress_file(input_file=args.input_file,
                  model_path=args.model,
                  output_file=args.output,
                  bit_precision=args.bit_precision,
                  temperature=args.temperature,
                  verify=not args.no_verify,
                  verbose=not args.quiet,
                  encoding_method=args.encoding_method,
                  prefix=prefix)


if __name__ == "__main__":
    main()
