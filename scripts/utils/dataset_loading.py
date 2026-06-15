"""
Dataset loading and processing utilities for compression experiments.

This module handles loading datasets from various sources (YAML, JSON, HuggingFace)
and processing them through the compression pipeline.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

# Datasets library for loading HuggingFace datasets
try:
    from datasets import load_dataset as hf_load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

# Local imports
try:
    from compression.utils.test_texts import load_text_from_file
except ImportError:
    load_text_from_file = None


def _load_from_json(fp: Path) -> Tuple[str, List[int]]:
    """Load *fp* assuming it's a JSON file produced by ``load_text_from_file``."""
    if load_text_from_file is None:
        raise ImportError("Could not import load_text_from_file")
    text, tokens, _meta = load_text_from_file(str(fp))
    return text, tokens


def _load_plain_text(fp: Path, tokenizer) -> Tuple[str, List[int]]:
    text = fp.read_text(encoding="utf-8")
    tokens = tokenizer.encode(text)
    return text, tokens


def load_text_and_tokens(fp: Path, tokenizer) -> Tuple[str, List[int]]:
    """Load text and tokens from a file (JSON or plain text)."""
    if fp.suffix.lower() == ".json":
        return _load_from_json(fp)
    return _load_plain_text(fp, tokenizer)


def load_hf_dataset(dataset_name: str,
                    tokenizer,
                    split: str = "test",
                    limit: Optional[int] = None) -> Tuple[List[Dict], Dict]:
    """Load dataset from HuggingFace (e.g., Lots-of-LoRAs datasets).

    Args:
        dataset_name: HuggingFace dataset name (e.g., "Lots-of-LoRAs/task561_alt_translation_en_bg")
        tokenizer: Tokenizer
        split: Dataset split to use
        limit: Optional limit on samples

    Returns:
        (samples, metadata) tuple in compression dataset format
    """
    if not DATASETS_AVAILABLE:
        raise ImportError(
            "datasets library required for HuggingFace datasets. Install with: pip install datasets"
        )

    print(f"Loading HuggingFace dataset: {dataset_name}")
    print(f"  Split: {split}")
    dataset = hf_load_dataset(dataset_name, split=split)
    print(f"  Total samples in split: {len(dataset)}")

    samples = []
    for i, item in enumerate(dataset):
        if limit and i >= limit:
            break

        # Extract fields (Lots-of-LoRAs format: input, output, id)
        input_text = item.get('input', '')
        output_text = item.get('output', '')

        # Handle output being a list/sequence
        if isinstance(output_text, (list, tuple)):
            output_text = output_text[0] if output_text else ''

        # Use input as prompt (context), output as text to compress
        samples.append({
            'prompt_id': item.get('id', f'sample_{i}'),
            'max_new_tokens': None,  # Not applicable for pre-generated data
            'prompt_text': input_text,
            'generated_text': output_text,
        })

    # Extract task ID from dataset name
    task_id = dataset_name.split('/')[-1].split(
        '_')[0] if '/' in dataset_name else 'unknown'

    metadata = {
        'dataset_name': dataset_name,
        'split': split,
        'task_id': task_id,
        'total_samples': len(dataset),
        'loaded_samples': len(samples),
    }

    print(f"Loaded {len(samples)} samples from {dataset_name}")
    return samples, metadata


def load_compression_dataset(
        dataset_path: Path,
        tokenizer,
        limit: Optional[int] = None) -> Tuple[List[Dict], Dict]:
    """Load compression dataset YAML and extract samples.

    Args:
        dataset_path: Path to YAML dataset file
        tokenizer: Tokenizer to encode prompts
        limit: Optional limit on number of samples to load

    Returns:
        (samples, metadata) tuple
    """
    import yaml

    with open(dataset_path, 'r') as f:
        dataset = yaml.safe_load(f)

    samples = []
    for sample in dataset['samples']:
        samples.append({
            'prompt_id': sample['prompt_id'],
            'max_new_tokens': sample['max_new_tokens'],
            'prompt_text': sample['prompt_text'],
            'generated_text': sample['generated_text'],
        })

        # Stop if we've reached the limit
        if limit is not None and len(samples) >= limit:
            break

    metadata = dataset.get('metadata', {})
    print(f"Loaded dataset: {metadata.get('model', 'unknown')}")
    print(f"  Created: {metadata.get('created_at', 'unknown')}")
    print(f"  Total samples in dataset: {len(dataset['samples'])}")
    if limit is not None and len(samples) < len(dataset['samples']):
        print(f"  Limited to: {len(samples)} samples")
    else:
        print(f"  Samples loaded: {len(samples)}")
    print(f"  Temperature: {metadata.get('temperature', 'unknown')}")

    return samples, metadata


def _to_python_type(val):
    """Convert numpy/torch scalars to native Python types for YAML serialization."""
    if val is None:
        return None
    if isinstance(val, (np.integer, np.floating)):
        return val.item()
    if isinstance(val, np.ndarray):
        return val.tolist()
    if torch.is_tensor(val):
        return val.item() if val.numel() == 1 else val.tolist()
    return val


def process_dataset(samples: List[Dict],
                    metadata: Dict,
                    encoder,
                    decoder,
                    tokenizer,
                    args,
                    compress_fn=None,
                    run_dir: Optional[Path] = None,
                    run_name: str = "compression") -> List[Dict]:
    """Process compression dataset: compress each sample with prompt as context.

    Args:
        samples: List of sample dicts from dataset
        metadata: Dataset metadata
        encoder: Encoder instance
        decoder: Decoder instance
        tokenizer: Tokenizer
        args: Namespace with options (use_prefill, compare_probs, output_json, file)
        compress_fn: The compress function to use (if None, must be imported)
        run_dir: Optional directory for results
        run_name: Name for the run

    Returns:
        List of compression results for each sample
    """
    # Extract options from args
    use_prefill = getattr(args, 'use_prefill', False)
    compare_probs = getattr(args, 'compare_probs', False)
    output_json = getattr(args, 'output_json', None)
    file_path = getattr(args, 'file', None)
    print(f"\n{'='*80}")
    print(f"DATASET COMPRESSION MODE")
    print(f"{'='*80}")
    print(f"Using prompts as initial context for compression")
    print(f"Total samples: {len(samples)}")
    print(f"{'='*80}\n")

    # Group samples by max_new_tokens for organized reporting
    samples_by_max_tokens = defaultdict(list)
    for sample in samples:
        samples_by_max_tokens[sample['max_new_tokens']].append(sample)

    all_results = []
    total_skipped = 0

    # Set up plots directory if run_dir is provided
    plots_dir = run_dir / "plots" if run_dir else None

    # Process each max_tokens group
    for max_tokens in sorted(samples_by_max_tokens.keys()):
        group_samples = samples_by_max_tokens[max_tokens]
        print(f"\n{'='*80}")
        print(
            f"Processing {len(group_samples)} samples with max_tokens={max_tokens}"
        )
        print(f"{'='*80}")

        group_results = []
        skipped_count = 0
        for idx, sample in enumerate(
                tqdm(group_samples,
                     desc=f"Compressing (max_tokens={max_tokens})")):
            print(f"idx - {idx}")
            # Use prompt as context, compress generated tokens
            generated_text = sample['generated_text']
            prompt_text = sample['prompt_text']

            # CRITICAL: Tokenize prompt+generated together to ensure correct tokenization!
            full_text = prompt_text + generated_text
            full_tokens = tokenizer.encode(full_text, add_special_tokens=True)

            # Also tokenize prompt alone to find the split point
            prompt_tokens_only = tokenizer.encode(prompt_text,
                                                  add_special_tokens=True)

            # Split: everything after prompt is the generated tokens
            prompt_tokens = prompt_tokens_only
            generated_tokens = full_tokens[len(prompt_tokens_only):]

            print(f"  Tokenization check:")
            print(f"    Prompt tokens: {len(prompt_tokens)}")
            print(f"    Generated tokens: {len(generated_tokens)}")
            print(
                f"    First 3 generated tokens: {generated_tokens[:3] if len(generated_tokens) >= 3 else generated_tokens}"
            )

            # Skip samples with no meaningful content
            if len(generated_tokens) < 2:
                skipped_count += 1
                continue

            print(f"compress - generated text - {generated_text[:100]}")

            # Compress with prompt as initial_context
            result = compress_fn(generated_tokens,
                                 encoder,
                                 decoder,
                                 "block",
                                 tokenizer=tokenizer,
                                 save_plot=None,
                                 use_prefill=use_prefill,
                                 initial_context=prompt_tokens,
                                 plots_dir=plots_dir,
                                 compare_probs=compare_probs)

            # Add sample metadata to result
            result['prompt_id'] = sample['prompt_id']
            result['max_new_tokens'] = sample['max_new_tokens']
            result['prompt_length'] = len(prompt_tokens)
            result['generated_length'] = len(generated_tokens)

            # Report stats immediately
            if result['success']:
                comp_ratio = 1.0 / result['compression_ratio']
                enc_ms_per_tok = result['encode_time_per_token'] * 1000
                dec_ms_per_tok = result['decode_time_per_token'] * 1000
                print(
                    f"\n  [Sample {sample['prompt_id']}] ✓ {comp_ratio:.2f}x compression | {result['bits_per_token']:.2f} bpt | "
                    f"enc: {enc_ms_per_tok:.2f}ms/tok | dec: {dec_ms_per_tok:.2f}ms/tok"
                )
            else:
                print(f"\n  [Sample {sample['prompt_id']}] ❌ Decode failed")

            group_results.append(result)
            all_results.append(result)

        # Print summary for this max_tokens group
        _print_group_summary(group_results, max_tokens, skipped_count)
        total_skipped += skipped_count

    # Overall summary
    successful_all = [r for r in all_results if r['success']]
    _print_overall_summary(all_results, successful_all, total_skipped,
                           samples_by_max_tokens)

    # Save results
    if run_dir is None:
        results_dir = Path(__file__).parent.parent.parent / "data" / "results"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = results_dir / f"run_{timestamp}"

    run_dir.mkdir(exist_ok=True, parents=True)

    _save_results_yaml(all_results, successful_all, total_skipped, metadata,
                       encoder, run_dir, run_name, file_path)
    _generate_plots(all_results, run_dir)

    # Save to JSON if requested
    if output_json:
        _save_results_json(all_results, successful_all, metadata, output_json)

    return all_results


def _print_group_summary(group_results: List[Dict], max_tokens: int,
                         skipped_count: int):
    """Print summary for a max_tokens group."""
    print(f"\n----- Summary for max_tokens={max_tokens} -----")
    if skipped_count > 0:
        print(f"  Skipped: {skipped_count} samples (empty/too short)")

    successful = [r for r in group_results if r['success']]
    if successful:
        avg_compression = sum(1.0 / r['compression_ratio']
                              for r in successful) / len(successful)
        avg_bpt = sum(r['bits_per_token']
                      for r in successful) / len(successful)
        avg_enc_time = sum(r['encode_time_per_token']
                           for r in successful) / len(successful)
        avg_dec_time = sum(r['decode_time_per_token']
                           for r in successful) / len(successful)

        print(f"  Successful: {len(successful)}/{len(group_results)}")
        print(f"  Avg compression: {avg_compression:.2f}x")
        print(f"  Avg bits/token: {avg_bpt:.2f}")
        print(f"  Avg encode time: {avg_enc_time*1000:.2f}ms/tok")
        print(f"  Avg decode time: {avg_dec_time*1000:.2f}ms/tok")
    else:
        print(f"  ❌ No successful compressions")

    if len(successful) < len(group_results):
        failed = len(group_results) - len(successful)
        print(f"  ⚠️  Failed: {failed} samples")


def _print_overall_summary(all_results: List[Dict], successful_all: List[Dict],
                           total_skipped: int, samples_by_max_tokens: Dict):
    """Print overall dataset summary."""
    print(f"\n{'='*80}")
    print(f"OVERALL DATASET SUMMARY")
    print(f"{'='*80}")
    print(f"Total samples processed: {len(all_results)}")
    if total_skipped > 0:
        print(f"Skipped (empty/too short): {total_skipped}")
    if all_results:
        print(
            f"Successful: {len(successful_all)} ({len(successful_all)/len(all_results)*100:.1f}%)"
        )
    else:
        print(f"Successful: 0")

    if successful_all:
        overall_avg_compression = sum(
            1.0 / r['compression_ratio']
            for r in successful_all) / len(successful_all)
        overall_avg_bpt = sum(r['bits_per_token']
                              for r in successful_all) / len(successful_all)

        print(f"\nOverall averages (successful only):")
        print(f"  Compression: {overall_avg_compression:.2f}x")
        print(f"  Bits/token: {overall_avg_bpt:.2f}")

        # Breakdown by max_tokens
        print(f"\nBreakdown by max_tokens:")
        for max_tokens in sorted(samples_by_max_tokens.keys()):
            group_successful = [
                r for r in successful_all if r['max_new_tokens'] == max_tokens
            ]
            if group_successful:
                avg_comp = sum(
                    1.0 / r['compression_ratio']
                    for r in group_successful) / len(group_successful)
                avg_bpt = sum(
                    r['bits_per_token']
                    for r in group_successful) / len(group_successful)
                print(
                    f"  max_tokens={max_tokens}: {avg_comp:.2f}x compression, {avg_bpt:.2f} bpt ({len(group_successful)} samples)"
                )

    print(f"{'='*80}")


def _save_results_yaml(all_results: List[Dict], successful_all: List[Dict],
                       total_skipped: int, metadata: Dict, encoder,
                       run_dir: Path, run_name: str, file_path: Optional[str]):
    """Save results to YAML file."""
    import yaml

    results_yaml_path = run_dir / f"{run_name}_results.yaml"

    # Prepare compact results
    compact_results = []
    for r in all_results:
        compact_results.append({
            'prompt_id':
            str(r['prompt_id']),
            'max_new_tokens':
            int(r['max_new_tokens'])
            if r['max_new_tokens'] is not None else None,
            'encoding_time_per_token':
            float(_to_python_type(r['encode_time_per_token'])),
            'decoding_time_per_token':
            float(_to_python_type(r['decode_time_per_token'])),
            'num_prompt_tokens':
            int(r['prompt_length']),
            'num_generated_tokens':
            int(r['generated_length']),
            'compression_ratio':
            float(1.0 / r['compression_ratio'])
            if r['success'] and r['compression_ratio'] else None,
            'bits_per_token':
            float(_to_python_type(r['bits_per_token']))
            if r['success'] else None,
            'success':
            bool(r['success'])
        })

    yaml_output = {
        'metadata': {
            'created_at':
            datetime.now().isoformat(),
            'dataset':
            str(file_path) if file_path else 'unknown',
            'model':
            str(metadata.get('model', 'unknown')),
            'compression_config': {
                'bit_precision': int(encoder.bit_precision),
                'bits_for_encoding_count':
                int(encoder.bits_for_encoding_count),
                'min_prob': float(encoder.min_prob),
                'temperature': float(encoder.temperature),
            },
            'total_samples_processed':
            int(len(all_results)),
            'total_skipped':
            int(total_skipped),
            'successful':
            int(len(successful_all)),
            'success_rate':
            float(len(successful_all) /
                  len(all_results)) if all_results else 0.0,
        },
        'results': compact_results
    }

    with open(results_yaml_path, 'w') as f:
        yaml.dump(yaml_output, f, default_flow_style=False, sort_keys=False)
    print(f"\n✓ Saved compression results to {results_yaml_path}")


def _generate_plots(all_results: List[Dict], run_dir: Path):
    """Generate plots for the compression results."""
    print(f"\nGenerating plots...")
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)

    plot_timestamp = run_dir.name

    # Prepare compact results for plotting
    compact_results = []
    for r in all_results:
        compact_results.append({
            'prompt_id':
            str(r['prompt_id']),
            'max_new_tokens':
            r['max_new_tokens'],
            'encoding_time_per_token':
            r['encode_time_per_token'],
            'decoding_time_per_token':
            r['decode_time_per_token'],
            'num_prompt_tokens':
            r['prompt_length'],
            'num_generated_tokens':
            r['generated_length'],
            'compression_ratio':
            1.0 / r['compression_ratio']
            if r['success'] and r['compression_ratio'] else None,
            'bits_per_token':
            r['bits_per_token'] if r['success'] else None,
            'success':
            r['success']
        })

    try:
        import matplotlib
        matplotlib.use('Agg')

        import sys
        scripts_dir = Path(__file__).parent.parent
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        from plot_compression_results import (plot_time_vs_tokens,
                                              plot_compression_ratio_histogram,
                                              plot_bits_per_token_histogram)

        plot_time_vs_tokens(compact_results, plots_dir, plot_timestamp)
        plot_compression_ratio_histogram(compact_results, plots_dir,
                                         plot_timestamp)
        plot_bits_per_token_histogram(compact_results, plots_dir,
                                      plot_timestamp)

        print(f"✓ Saved plots to {plots_dir}/")
    except ImportError as e:
        print(f"⚠️  Could not import plotting functions: {e}")
    except Exception as e:
        print(f"⚠️  Error generating plots: {e}")


def _save_results_json(all_results: List[Dict], successful_all: List[Dict],
                       metadata: Dict, output_path: Path):
    """Save detailed results to JSON."""
    output_data = {
        'dataset_metadata': metadata,
        'compression_results': all_results,
        'summary': {
            'total_samples':
            len(all_results),
            'successful':
            len(successful_all),
            'success_rate':
            len(successful_all) / len(all_results) if all_results else 0,
        }
    }
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    print(f"\n✓ Saved detailed results to {output_path}")
