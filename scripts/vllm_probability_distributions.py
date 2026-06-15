"""
Get logits for token sequences using prefill.

Simple utility to run a forward pass on a sequence and extract logits for each position.
"""

import os
import sys
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from typing import List, Optional, Dict, Any, Tuple
import gc
import pickle
import yaml
from dataclasses import dataclass
import time


@dataclass
class LogitsConfig:
    """Configuration for logits extraction."""

    # Model settings
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"

    # Input/output settings
    pickle_file: Optional[str] = None
    output_file: Optional[str] = None

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "LogitsConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Extract model section
        model_config = config_dict.get("model", {})

        # Get model_name
        return cls(model_name=model_config.get("model_name", cls.model_name))


def set_tokenizer_pad_token(tokenizer, model, model_name):
    """Set pad token if not already set."""
    if not tokenizer.pad_token and "llama" in model_name.lower():
        tokenizer.pad_token_id = (model.config.eos_token_id[0] if isinstance(
            model.config.eos_token_id, list) else model.config.eos_token_id)
    elif not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def get_logits_for_sequence(
    model_name: str,
    token_ids: List[int],
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    Run prefill on a sequence and return logits for each position.

    Args:
        model_name: HuggingFace model name or path
        token_ids: List of token IDs to run prefill on
        device: Device to run on (defaults to auto)

    Returns:
        torch.Tensor: Logits of shape [num_tokens, vocab_size]
                     logits[i] = logits for predicting token at position i+1
    """
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto" if device is None else device,
        low_cpu_mem_usage=True,
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer = set_tokenizer_pad_token(tokenizer, model, model_name)

    device = model.device
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    print(f"Running prefill on {len(token_ids)} tokens...")
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits

    # Return logits as [seq_len, vocab_size]
    logits = logits.squeeze(0).float()

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return logits


def process_generated_outputs(
    model_name: str,
    pickle_file: str,
    output_file: Optional[str] = None,
    device: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Process vllm_generate.py outputs and get logits for each sequence.

    Args:
        model_name: HuggingFace model name or path
        pickle_file: Path to generated_outputs.pkl from vllm_generate.py
        output_file: Optional path to save results as pickle
        device: Device to run on (defaults to auto)

    Returns:
        Tuple of (results, timing_stats)
        - results: List of dicts with prompt_token_ids, generated_token_ids, and logits
        - timing_stats: Dict with timing information
    """
    # Load generated outputs
    print(f"Loading generated outputs from {pickle_file}...")
    with open(pickle_file, 'rb') as f:
        outputs = pickle.load(f)

    print(f"Loaded {len(outputs)} generated sequences")

    # Load model once
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto" if device is None else device,
        low_cpu_mem_usage=True,
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer = set_tokenizer_pad_token(tokenizer, model, model_name)
    device = model.device

    results = []
    total_tokens = 0

    # Start timing
    start_time = time.time()

    # Process each sequence
    for i, output in enumerate(tqdm(outputs, desc="Processing sequences")):
        # Extract token IDs (handle both vLLM RequestOutput and dict formats)
        if hasattr(output, 'prompt_token_ids'):
            # vLLM RequestOutput format
            prompt_ids = list(output.prompt_token_ids)
            gen_ids = list(output.outputs[0].token_ids)
        else:
            # Dict format
            prompt_ids = output['prompt_token_ids']
            gen_ids = output['generated_token_ids']

        # Combine prompt + generated tokens
        full_sequence = prompt_ids + gen_ids
        total_tokens += len(full_sequence)

        # Get logits
        input_ids = torch.tensor([full_sequence],
                                 dtype=torch.long,
                                 device=device)
        with torch.no_grad():
            logits = model(input_ids=input_ids).logits

        logits = logits.squeeze(
            0).float().cpu()  # Move to CPU to save GPU memory

        results.append({
            'prompt_token_ids': prompt_ids,
            'generated_token_ids': gen_ids,
            'logits': logits,  # Shape: [len(full_sequence), vocab_size]
        })

    # End timing
    end_time = time.time()
    total_time = end_time - start_time

    # Calculate timing statistics
    timing_stats = {
        'total_time_seconds':
        total_time,
        'total_tokens':
        total_tokens,
        'tokens_per_second':
        total_tokens / total_time if total_time > 0 else 0,
        'time_per_1000_tokens':
        (total_time / total_tokens * 1000) if total_tokens > 0 else 0,
        'num_sequences':
        len(outputs),
        'avg_tokens_per_sequence':
        total_tokens / len(outputs) if len(outputs) > 0 else 0,
    }

    # Cleanup
    del model
    torch.cuda.empty_cache()
    gc.collect()

    # Save results if requested
    if output_file:
        print(f"\nSaving results to {output_file}...")
        with open(output_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"Saved {len(results)} results")

    return results, timing_stats


def main():
    """Example usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Get logits for token sequences")
    parser.add_argument("--config", type=str, help="Path to YAML config file")

    # Optional overrides
    parser.add_argument("--model", type=str, help="Model name or path")
    parser.add_argument("--text",
                        type=str,
                        help="Text to tokenize and get logits for")
    parser.add_argument("--tokens", type=str, help="Comma-separated token IDs")
    parser.add_argument(
        "--pickle",
        type=str,
        help="Path to generated_outputs.pkl from vllm_generate.py")
    parser.add_argument("--output",
                        type=str,
                        help="Output pickle file for batch processing")
    parser.add_argument("--device",
                        type=str,
                        default=None,
                        help="Device (cuda/cpu/auto)")

    args = parser.parse_args()

    # Load config from YAML or use defaults
    if args.config is not None:
        print(f"Loading configuration from {args.config}")
        cfg = LogitsConfig.from_yaml(args.config)
    else:
        cfg = LogitsConfig()

    # Override config with command-line arguments
    if args.model is not None:
        cfg.model_name = args.model
    if args.pickle is not None:
        cfg.pickle_file = args.pickle
    if args.output is not None:
        cfg.output_file = args.output

    print("=" * 80)
    print("LOGITS EXTRACTION")
    print("=" * 80)
    print(f"Model: {cfg.model_name}")
    if cfg.pickle_file:
        print(f"Input: {cfg.pickle_file}")
    if cfg.output_file:
        print(f"Output: {cfg.output_file}")
    print("=" * 80)

    # Batch processing mode
    if cfg.pickle_file or args.pickle:
        pickle_file = cfg.pickle_file or args.pickle
        results, timing_stats = process_generated_outputs(
            cfg.model_name, pickle_file, cfg.output_file, args.device)

        print(f"\nProcessed {len(results)} sequences")
        print(f"Each result contains:")
        print(f"  - prompt_token_ids: list of prompt tokens")
        print(f"  - generated_token_ids: list of generated tokens")
        print(f"  - logits: tensor of shape [seq_len, vocab_size]")

        # Display timing statistics
        print("\n" + "=" * 80)
        print("TIMING STATISTICS")
        print("=" * 80)
        print(
            f"Total time:               {timing_stats['total_time_seconds']:.2f} seconds"
        )
        print(f"Total tokens processed:   {timing_stats['total_tokens']:,}")
        print(f"Number of sequences:      {timing_stats['num_sequences']}")
        print(
            f"Avg tokens per sequence:  {timing_stats['avg_tokens_per_sequence']:.1f}"
        )
        print(f"\nThroughput:")
        print(
            f"  Tokens per second:      {timing_stats['tokens_per_second']:.2f}"
        )
        print(
            f"  Time per 1000 tokens:   {timing_stats['time_per_1000_tokens']:.2f} seconds"
        )
        print("=" * 80)
        return

    # Single sequence mode
    if args.text:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        token_ids = tokenizer.encode(args.text)
        print(f"Tokenized text into {len(token_ids)} tokens")
    elif args.tokens:
        token_ids = [int(t.strip()) for t in args.tokens.split(",")]
    else:
        print("Error: Must provide either --text, --tokens, or --pickle")
        return

    # Get logits
    logits = get_logits_for_sequence(cfg.model_name, token_ids, args.device)

    print(f"\nLogits shape: {logits.shape}")
    print(f"  - Position 0 logits predict token at position 1")
    print(
        f"  - Position {len(token_ids)-1} logits predict next token after sequence"
    )

    # Show some stats
    print(f"\nLogit statistics:")
    print(f"  - Min: {logits.min().item():.2f}")
    print(f"  - Max: {logits.max().item():.2f}")
    print(f"  - Mean: {logits.mean().item():.2f}")

    # Show top-5 predictions for first position
    probs = torch.softmax(logits[0], dim=-1)
    top_probs, top_indices = torch.topk(probs, k=5)
    print(f"\nTop-5 predictions for position 0:")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
        token_str = tokenizer.decode([idx.item()])
        print(
            f"  {i+1}. Token {idx.item()} ('{token_str}'): {prob.item():.4f}")


if __name__ == "__main__":
    main()
