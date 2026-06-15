#!/usr/bin/env python3
"""
Time decode (autoregressive generation) vs prefill across different quantization levels.
"""

import gc
import time
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse


def time_decode(model, tokenizer, prompt: str,
                num_tokens: int) -> tuple[float, list[int]]:
    """Time autoregressive generation with KV cache.

    Returns:
        elapsed_time: Time in seconds
        generated_ids: List of generated token IDs
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    generated_ids = []
    past_key_values = None

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.perf_counter()

    with torch.no_grad():
        # Initial prefill
        outputs = model(input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token_id = torch.argmax(outputs.logits[0, -1, :]).item()
        generated_ids.append(next_token_id)

        # Decode remaining tokens
        for _ in range(1, num_tokens):
            new_token = torch.tensor([[next_token_id]]).to(model.device)
            outputs = model(new_token,
                            past_key_values=past_key_values,
                            use_cache=True)
            past_key_values = outputs.past_key_values
            next_token_id = torch.argmax(outputs.logits[0, -1, :]).item()
            generated_ids.append(next_token_id)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed_time = time.perf_counter() - start_time

    return elapsed_time, generated_ids


def time_prefill(model, tokenizer, prompt: str,
                 generated_ids: list[int]) -> float:
    """Time prefill (batch forward pass) on the full sequence.

    Returns:
        elapsed_time: Time in seconds
    """
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    full_sequence = torch.cat(
        [prompt_ids,
         torch.tensor([generated_ids]).to(model.device)], dim=1)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.perf_counter()

    with torch.no_grad():
        _ = model(full_sequence)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed_time = time.perf_counter() - start_time

    return elapsed_time


def load_model(model_name: str, quantization: str, device: str):
    """Load model with specified quantization."""
    load_kwargs = {
        "device_map": "auto" if device == "cuda" else None,
    }

    if quantization == "fp32":
        load_kwargs["torch_dtype"] = torch.float32
    elif quantization == "bf16":
        load_kwargs["torch_dtype"] = torch.bfloat16
    elif quantization == "fp16":
        load_kwargs["torch_dtype"] = torch.float16
    elif quantization == "8bit":
        load_kwargs["load_in_8bit"] = True
    elif quantization == "4bit":
        load_kwargs["load_in_4bit"] = True

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Time decode vs prefill across quantization levels")
    parser.add_argument("--model",
                        default="mistralai/Mistral-7B-Instruct-v0.2",
                        help="Model to use")
    parser.add_argument("--num-tokens",
                        type=int,
                        default=500,
                        help="Number of tokens to generate")
    parser.add_argument("--quantization",
                        type=str,
                        nargs="+",
                        default=["4bit", "bf16", "fp32"],
                        choices=["4bit", "8bit", "fp16", "bf16", "fp32"],
                        help="Quantization levels to test")
    parser.add_argument("--prompt",
                        type=str,
                        default="The quick brown fox jumps over the lazy dog.",
                        help="Prompt to use for generation")
    parser.add_argument("--num-runs",
                        type=int,
                        default=3,
                        help="Number of runs to average timing")
    parser.add_argument("--warmup-runs",
                        type=int,
                        default=1,
                        help="Number of warmup runs before timing")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("Decode vs Prefill Timing Experiment")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Num tokens: {args.num_tokens}")
    print(f"Quantization levels: {args.quantization}")
    print(f"Num runs: {args.num_runs} (+ {args.warmup_runs} warmup)")
    print("=" * 70)

    results = {}

    for quantization in args.quantization:
        print(f"\n{'#' * 70}")
        print(f"# Quantization: {quantization}")
        print(f"{'#' * 70}")

        # Load model
        print(f"\nLoading model with {quantization}...")
        model, tokenizer = load_model(args.model, quantization, device)
        print(f"✓ Model loaded")

        # Warmup runs
        print(f"\nWarmup ({args.warmup_runs} runs)...")
        for i in range(args.warmup_runs):
            _, generated_ids = time_decode(model, tokenizer, args.prompt,
                                           args.num_tokens)
            _ = time_prefill(model, tokenizer, args.prompt, generated_ids)
            print(f"  Warmup {i+1}/{args.warmup_runs} complete")

        # Timed runs
        print(f"\nTiming ({args.num_runs} runs)...")
        decode_times = []
        prefill_times = []

        for i in range(args.num_runs):
            decode_time, generated_ids = time_decode(model, tokenizer,
                                                     args.prompt,
                                                     args.num_tokens)
            prefill_time = time_prefill(model, tokenizer, args.prompt,
                                        generated_ids)

            decode_times.append(decode_time)
            prefill_times.append(prefill_time)

            print(
                f"  Run {i+1}/{args.num_runs}: decode={decode_time:.3f}s, prefill={prefill_time:.3f}s"
            )

        # Compute statistics
        decode_mean = np.mean(decode_times)
        decode_std = np.std(decode_times)
        prefill_mean = np.mean(prefill_times)
        prefill_std = np.std(prefill_times)

        results[quantization] = {
            'decode_mean': decode_mean,
            'decode_std': decode_std,
            'prefill_mean': prefill_mean,
            'prefill_std': prefill_std,
            'decode_times': decode_times,
            'prefill_times': prefill_times,
        }

        print(f"\n  Results for {quantization}:")
        print(
            f"    Decode:  {decode_mean:.3f}s ± {decode_std:.3f}s ({args.num_tokens/decode_mean:.1f} tok/s)"
        )
        print(
            f"    Prefill: {prefill_mean:.3f}s ± {prefill_std:.3f}s ({args.num_tokens/prefill_mean:.1f} tok/s)"
        )
        print(
            f"    Ratio (decode/prefill): {decode_mean/prefill_mean:.2f}x slower"
        )

        # Cleanup
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(
        f"\n{'Quantization':<12} {'Decode (s)':<15} {'Prefill (s)':<15} {'Decode tok/s':<15} {'Prefill tok/s':<15} {'Ratio':<10}"
    )
    print("-" * 82)

    for quant in args.quantization:
        r = results[quant]
        decode_tps = args.num_tokens / r['decode_mean']
        prefill_tps = args.num_tokens / r['prefill_mean']
        ratio = r['decode_mean'] / r['prefill_mean']
        print(f"{quant:<12} {r['decode_mean']:.3f} ± {r['decode_std']:.3f}   "
              f"{r['prefill_mean']:.3f} ± {r['prefill_std']:.3f}   "
              f"{decode_tps:<15.1f} {prefill_tps:<15.1f} {ratio:.2f}x")

    print("\n" + "=" * 70)
    print("✓ Timing experiment complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
