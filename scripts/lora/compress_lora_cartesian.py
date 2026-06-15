#!/usr/bin/env python3
"""
Compression Cartesian product: LoRA × Dataset × Temperature.

Tests how well different LoRAs compress text from different datasets at various temperatures.
"""

import argparse
from pathlib import Path
from typing import List, Dict, Optional
import yaml
import json
from datetime import datetime
from datasets import load_dataset as hf_load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# PEFT for LoRA support
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

# Local imports
from compression.block_coder import (
    BlockEmissionArithmeticCoder,
    BlockEmissionArithmeticDecoder,
)

# Configuration: 3 diverse LoRAs
LORAS = [
    "task561",  # Translation (en -> bg)
    "task581",  # Social IQA question generation
    "task1431",  # Medical QA (head_qa)
]

LORA_DATASETS = {
    "task561": "Lots-of-LoRAs/task561_alt_translation_en_bg",
    "task581": "Lots-of-LoRAs/task581_socialiqa_question_generation",
    "task1431": "Lots-of-LoRAs/task1431_head_qa_answer_generation",
}

TEMPERATURES = [0.5, 1.0, 2.0]


def load_dataset_samples(dataset_name: str,
                         split: str = "valid",
                         limit: Optional[int] = None) -> List[Dict]:
    """Load samples from a Lots-of-LoRAs dataset.

    Returns list of dicts with 'id', 'input', 'output', 'text' (combined).
    """
    print(f"Loading dataset {dataset_name} ({split})...")
    dataset = hf_load_dataset(dataset_name, split=split)

    samples = []
    for i, item in enumerate(dataset):
        if limit and i >= limit:
            break

        input_text = item.get('input', '')
        output_text = item.get('output', '')
        # Combine input + output as the text to compress
        combined_text = f"{input_text}\n{output_text}" if output_text else input_text

        samples.append({
            'id': item.get('id', f'{dataset_name}-{i}'),
            'input': input_text,
            'output': output_text,
            'text': combined_text,
        })

    print(f"  Loaded {len(samples)} samples")
    return samples


def load_model_with_lora(base_model: str, lora_adapter: Optional[str] = None):
    """Load model with optional LoRA adapter."""
    print(f"Loading model: {base_model}")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    if lora_adapter:
        if not PEFT_AVAILABLE:
            raise ImportError(
                "PEFT is required for LoRA support. Install with: pip install peft"
            )
        print(f"  Applying LoRA: {lora_adapter}")
        model = PeftModel.from_pretrained(model, lora_adapter)

    model.eval()
    return model, tokenizer


def compress_text(
    model,
    tokenizer,
    text: str,
    temperature: float,
    device: str = "cuda",
) -> Dict:
    """Compress text and return compression stats."""

    # Tokenize
    tokens = tokenizer.encode(text, add_special_tokens=False)
    num_tokens = len(tokens)

    if num_tokens == 0:
        return {'error': 'empty text', 'num_tokens': 0}

    # Create coder with empty prefix (no prompt)
    coder = BlockEmissionArithmeticCoder(
        model=model,
        tokenizer=tokenizer,
        device=device,
        temperature=temperature,
    )

    # Encode
    start_time = datetime.now()
    compressed_bits = coder.encode(tokens, prefix_tokens=[])
    encode_time = (datetime.now() - start_time).total_seconds()

    # Calculate compression stats
    num_bits = len(compressed_bits)
    bits_per_token = num_bits / num_tokens if num_tokens > 0 else 0

    # Theoretical minimum (entropy)
    raw_bits = num_tokens * 16  # ~16 bits per token for vocab size ~32k
    compression_ratio = raw_bits / num_bits if num_bits > 0 else 0

    return {
        'num_tokens': num_tokens,
        'num_bits': num_bits,
        'bits_per_token': bits_per_token,
        'compression_ratio': compression_ratio,
        'encode_time_sec': encode_time,
    }


def main():
    parser = argparse.ArgumentParser(
        description=
        "Compression Cartesian product: LoRA × Dataset × Temperature")
    parser.add_argument("--base-model",
                        default="mistralai/Mistral-7B-Instruct-v0.2",
                        help="Base model to use")
    parser.add_argument("--lora-rank",
                        type=int,
                        default=16,
                        help="LoRA rank (default: 16)")
    parser.add_argument("--lora-bits",
                        type=int,
                        default=4,
                        help="LoRA bits (default: 4)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/n/netscratch/sham_lab/Lab/rrinberg/compression/lora_compression_cartesian"
        ),
        help="Output directory for results")
    parser.add_argument(
        "--limit-samples",
        type=int,
        default=10,
        help="Number of samples to compress per combination (default: 10)")
    parser.add_argument("--split",
                        default="valid",
                        choices=["train", "test", "valid"],
                        help="Dataset split to use")
    parser.add_argument("--skip-baseline",
                        action="store_true",
                        help="Skip baseline (no LoRA) compression")
    parser.add_argument(
        "--loras-only",
        nargs="+",
        metavar="TASK",
        help="Only test specific LoRAs (e.g., task561 task581)")
    parser.add_argument(
        "--datasets-only",
        nargs="+",
        metavar="TASK",
        help="Only use specific datasets (e.g., task561 task581)")
    parser.add_argument("--temps-only",
                        nargs="+",
                        type=float,
                        help="Only test specific temperatures (e.g., 0.5 1.0)")

    args = parser.parse_args()

    # Determine which configurations to test
    loras_to_test = args.loras_only if args.loras_only else LORAS
    datasets_to_test = args.datasets_only if args.datasets_only else LORAS
    temps_to_test = args.temps_only if args.temps_only else TEMPERATURES

    # Add baseline if not skipped
    models_to_test = ["baseline"] if not args.skip_baseline else []
    models_to_test.extend(loras_to_test)

    total_combinations = len(models_to_test) * len(temps_to_test) * len(
        datasets_to_test)

    print(f"\n{'='*80}")
    print("LoRA Compression Cartesian Product")
    print(f"{'='*80}")
    print(f"Base model: {args.base_model}")
    print(
        f"Models: {len(models_to_test)} (baseline + {len(loras_to_test)} LoRAs)"
    )
    print(f"Temperatures: {len(temps_to_test)} {temps_to_test}")
    print(f"Datasets: {len(datasets_to_test)} {datasets_to_test}")
    print(f"Total combinations: {total_combinations}")
    print(f"Samples per combination: {args.limit_samples}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*80}\n")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {args.output_dir}\n")

    # Pre-load all datasets
    dataset_cache = {}
    for dataset_key in datasets_to_test:
        dataset_name = LORA_DATASETS[dataset_key]
        dataset_cache[dataset_key] = load_dataset_samples(
            dataset_name, split=args.split, limit=args.limit_samples)

    # Save configuration
    config = {
        'created_at': datetime.now().isoformat(),
        'base_model': args.base_model,
        'lora_rank': args.lora_rank,
        'lora_bits': args.lora_bits,
        'split': args.split,
        'limit_samples': args.limit_samples,
        'models': models_to_test,
        'temperatures': temps_to_test,
        'datasets': datasets_to_test,
        'total_combinations': total_combinations,
    }

    with open(args.output_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Results accumulator
    all_results = []

    # Main loop: iterate over models (load once per model)
    current = 0
    for model_name in models_to_test:
        # Construct LoRA adapter path
        if model_name == "baseline":
            lora_adapter = None
            model_label = "baseline"
        else:
            lora_adapter = f"Lots-of-LoRAs/Mistral-7B-Instruct-v0.2-{args.lora_bits}b-r{args.lora_rank}-{model_name}"
            model_label = model_name

        # Load model once per LoRA
        model, tokenizer = load_model_with_lora(args.base_model, lora_adapter)

        for temperature in temps_to_test:
            for dataset_key in datasets_to_test:
                current += 1

                # Check if already done
                output_file = args.output_dir / f"model_{model_label}_temp_{temperature}_dataset_{dataset_key}.json"
                if output_file.exists():
                    print(
                        f"[{current}/{total_combinations}] Already exists, skipping: {output_file.name}"
                    )
                    continue

                print(f"\n{'='*80}")
                print(f"Combination {current}/{total_combinations}")
                print(f"{'='*80}")
                print(f"Model: {model_label}")
                print(f"Temperature: {temperature}")
                print(f"Dataset: {dataset_key}")
                print(f"{'='*80}\n")

                samples = dataset_cache[dataset_key]

                # Compress each sample
                sample_results = []
                for i, sample in enumerate(samples):
                    print(f"  Compressing sample {i+1}/{len(samples)}...",
                          end=" ")

                    try:
                        result = compress_text(
                            model=model,
                            tokenizer=tokenizer,
                            text=sample['text'],
                            temperature=temperature,
                        )
                        result['sample_id'] = sample['id']
                        sample_results.append(result)
                        print(f"{result['bits_per_token']:.2f} bits/token")

                    except Exception as e:
                        print(f"ERROR: {e}")
                        sample_results.append({
                            'sample_id': sample['id'],
                            'error': str(e),
                        })

                # Compute aggregates
                valid_results = [r for r in sample_results if 'error' not in r]
                if valid_results:
                    avg_bits_per_token = sum(
                        r['bits_per_token']
                        for r in valid_results) / len(valid_results)
                    avg_compression_ratio = sum(
                        r['compression_ratio']
                        for r in valid_results) / len(valid_results)
                    total_tokens = sum(r['num_tokens'] for r in valid_results)
                    total_bits = sum(r['num_bits'] for r in valid_results)
                else:
                    avg_bits_per_token = 0
                    avg_compression_ratio = 0
                    total_tokens = 0
                    total_bits = 0

                # Save results
                output_data = {
                    'metadata': {
                        'created_at': datetime.now().isoformat(),
                        'model': model_label,
                        'lora_adapter': lora_adapter,
                        'temperature': temperature,
                        'dataset': dataset_key,
                        'dataset_full_name': LORA_DATASETS[dataset_key],
                        'split': args.split,
                        'num_samples': len(samples),
                        'num_valid': len(valid_results),
                    },
                    'aggregate': {
                        'avg_bits_per_token': avg_bits_per_token,
                        'avg_compression_ratio': avg_compression_ratio,
                        'total_tokens': total_tokens,
                        'total_bits': total_bits,
                    },
                    'samples': sample_results,
                }

                with open(output_file, 'w') as f:
                    json.dump(output_data, f, indent=2)

                print(f"\nSaved to {output_file}")
                print(f"  Avg bits/token: {avg_bits_per_token:.3f}")
                print(f"  Avg compression ratio: {avg_compression_ratio:.2f}x")

                # Add to all results
                all_results.append({
                    'model': model_label,
                    'temperature': temperature,
                    'dataset': dataset_key,
                    'avg_bits_per_token': avg_bits_per_token,
                    'avg_compression_ratio': avg_compression_ratio,
                    'is_matching':
                    model_label == dataset_key,  # LoRA matches dataset
                })

        # Unload model to free memory before loading next
        del model
        torch.cuda.empty_cache()

    # Save summary
    print(f"\n{'='*80}")
    print("Compression Complete!")
    print(f"{'='*80}")

    summary = {
        'completed_at': datetime.now().isoformat(),
        'config': config,
        'results': all_results,
    }

    with open(args.output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary table
    print(
        f"\n{'Model':<12} {'Temp':<6} {'Dataset':<12} {'Bits/Tok':<10} {'Match':<6}"
    )
    print("-" * 50)
    for r in all_results:
        match_str = "YES" if r['is_matching'] else ""
        print(
            f"{r['model']:<12} {r['temperature']:<6} {r['dataset']:<12} {r['avg_bits_per_token']:<10.3f} {match_str:<6}"
        )

    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
