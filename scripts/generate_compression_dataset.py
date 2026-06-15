#!/usr/bin/env python3
"""
Generate a compression test dataset by sampling prompts from lmsys-chat-1m
and generating continuations with Llama 3.1 8B at temperature 1.0.

This creates a YAML dataset with prompts and generated text for testing compression.
Uses vLLM for fast batch generation.
"""

import argparse
import yaml
import gc
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm

import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset


def load_ultrachat_prompts(num_prompts: int, target_prompt_length: int,
                           tokenizer) -> List[Dict[str, Any]]:
    """
    Load prompts from UltraChat dataset and concatenate conversation history to reach target length.

    Args:
        num_prompts: Number of prompts to sample
        target_prompt_length: Target token length for prompts (~500)
        tokenizer: Tokenizer to measure length

    Returns:
        List of dicts with 'text' and 'token_length' keys
    """
    print(f"Loading UltraChat dataset...")
    dataset = load_dataset("stingning/ultrachat",
                           split="train",
                           streaming=True)

    prompts = []
    seen = 0
    length_stats = []  # Track lengths for debugging

    print(f"Filtering prompts to ~{target_prompt_length} tokens...")
    print(f"  (Concatenating conversation turns to reach target length)")

    for example in dataset:
        if len(prompts) >= num_prompts:
            break

        seen += 1
        if seen % 1000 == 0:
            if len(length_stats) > 0:
                avg_len = sum(length_stats) / len(length_stats)
                print(
                    f"  Scanned {seen} examples, found {len(prompts)} matching prompts (avg first msg length: {avg_len:.0f} tokens)..."
                )
            else:
                print(
                    f"  Scanned {seen} examples, found {len(prompts)} matching prompts..."
                )

        # Extract messages
        messages = example.get('data', [])
        if not messages or len(messages) == 0:
            continue

        # Build prompt by concatenating conversation turns until we reach target length
        # UltraChat first messages are typically short, so we concatenate turns
        prompt_parts = []
        total_tokens = 0

        for i, msg in enumerate(messages):
            # Handle both string and dict formats
            if isinstance(msg, str):
                content = msg
            elif isinstance(msg, dict):
                content = msg.get('content', '')
            else:
                continue

            # Stop after first user message for very first example (to track stats)
            if i == 0 and len(length_stats) < 100:
                first_msg_tokens = tokenizer.encode(content,
                                                    add_special_tokens=False)
                length_stats.append(len(first_msg_tokens))

            # Add message to prompt
            prompt_parts.append(content)
            combined_text = "\n\n".join(prompt_parts)
            tokens = tokenizer.encode(combined_text, add_special_tokens=False)
            total_tokens = len(tokens)

            # Check if we've reached target length (with 20% tolerance)
            min_length = int(target_prompt_length * 0.8)
            max_length = int(target_prompt_length * 1.2)

            if min_length <= total_tokens <= max_length:
                # Found good prompt
                prompts.append({
                    'text': combined_text,
                    'token_length': total_tokens
                })
                break
            elif total_tokens > max_length:
                # Exceeded target, truncate to target length
                tokens = tokens[:target_prompt_length]
                text = tokenizer.decode(tokens)
                prompts.append({
                    'text':
                    text,
                    'token_length':
                    len(tokenizer.encode(text, add_special_tokens=False))
                })
                break

    print(f"Found {len(prompts)} prompts with ~{target_prompt_length} tokens")
    if len(length_stats) > 0:
        print(
            f"  (Note: UltraChat first messages average {sum(length_stats)/len(length_stats):.0f} tokens, so we concatenate conversation turns)"
        )
    return prompts


def load_lmsys_prompts(num_prompts: int,
                       target_prompt_length: int,
                       tokenizer,
                       english_only: bool = True) -> List[Dict[str, Any]]:
    """
    Load prompts from lmsys-chat-1m dataset and filter/truncate to target length.

    Args:
        num_prompts: Number of prompts to sample
        target_prompt_length: Target token length for prompts (~500)
        tokenizer: Tokenizer to measure length
        english_only: Only accept English prompts (default: True)

    Returns:
        List of dicts with 'text' and 'token_length' keys
    """
    print(f"Loading lmsys-chat-1m dataset...")
    dataset = load_dataset("lmsys/lmsys-chat-1m",
                           split="train",
                           streaming=True)

    prompts = []
    seen = 0

    # Import langdetect if needed
    if english_only:
        try:
            from langdetect import detect, LangDetectException
        except ImportError:
            print(
                "Warning: langdetect not installed. Install with: pip install langdetect"
            )
            print("Proceeding without language filtering...")
            english_only = False

    filter_msg = f"~{target_prompt_length} tokens" + (" (English only)"
                                                      if english_only else "")
    print(f"Filtering prompts to {filter_msg}...")

    for example in dataset:
        if len(prompts) >= num_prompts:
            break

        seen += 1
        if seen % 1000 == 0:
            print(
                f"  Scanned {seen} examples, found {len(prompts)} matching prompts..."
            )

        # Extract first user message from conversation
        conversation = example.get('conversation', [])
        if not conversation or len(conversation) == 0:
            continue

        # Get first user message
        first_message = None
        for msg in conversation:
            if msg.get('role') == 'user':
                first_message = msg.get('content', '')
                break

        if not first_message:
            continue

        # Check if English (if filtering enabled)
        if english_only:
            try:
                lang = detect(first_message)
                if lang != 'en':
                    continue
            except LangDetectException:
                # Skip if language detection fails
                continue

        # Tokenize and check length
        tokens = tokenizer.encode(first_message, add_special_tokens=False)
        token_length = len(tokens)

        # Accept prompts within 20% of target length
        min_length = int(target_prompt_length * 0.8)
        max_length = int(target_prompt_length * 1.2)

        if min_length <= token_length <= max_length:
            # Truncate to exact target if longer
            if token_length > target_prompt_length:
                tokens = tokens[:target_prompt_length]
                text = tokenizer.decode(tokens)
            else:
                text = first_message

            prompts.append({
                'text':
                text,
                'token_length':
                len(tokenizer.encode(text, add_special_tokens=False))
            })

    print(f"Found {len(prompts)} prompts with ~{target_prompt_length} tokens")
    return prompts


def generate_continuations_vllm(
        prompts: List[Dict[str, Any]],
        model_name: str,
        tokenizer,
        max_tokens_list: List[int],
        temperature: float = 1.0,
        gpu_memory_utilization: float = 0.7,
        seed: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Generate continuations for each prompt with different max_tokens settings using vLLM.

    Args:
        prompts: List of prompt dicts with 'text' and 'token_length'
        model_name: Model name/path
        tokenizer: Tokenizer
        max_tokens_list: List of max_tokens to generate (e.g., [500, 1000, 2000])
        temperature: Sampling temperature
        gpu_memory_utilization: GPU memory fraction to use
        seed: Random seed for reproducibility

    Returns:
        List of generation results with metadata
    """
    results = []

    # Generate for each max_tokens setting
    for max_new_tokens in max_tokens_list:
        print(f"\n{'='*80}")
        print(f"Generating with max_tokens={max_new_tokens}")
        print(f"{'='*80}")

        # Load vLLM model
        print(f"Loading vLLM model: {model_name}...")
        llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            enforce_eager=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )

        # Prepare sampling params
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
            seed=seed,
        )

        # Tokenize all prompts
        prompt_token_ids = [
            tokenizer.encode(prompt_data['text'], add_special_tokens=True)
            for prompt_data in prompts
        ]

        # Generate in batch
        print(
            f"Generating {len(prompts)} sequences with max_tokens={max_new_tokens}..."
        )
        start_time = time.time()
        outputs = llm.generate(prompt_token_ids=prompt_token_ids,
                               sampling_params=sampling_params)
        generation_time = time.time() - start_time

        # Extract results
        for prompt_idx, (prompt_data,
                         output) in enumerate(zip(prompts, outputs)):
            generated_text = output.outputs[0].text

            results.append({
                'prompt_id': prompt_idx,
                'prompt_text': prompt_data['text'],
                'prompt_token_length': prompt_data['token_length'],
                'max_new_tokens': max_new_tokens,
                'generated_text': generated_text,
                'temperature': temperature,
            })

        # Calculate statistics
        total_generated = sum(
            len(output.outputs[0].token_ids) for output in outputs)
        tokens_per_sec = total_generated / generation_time if generation_time > 0 else 0

        print(
            f"✓ Generated {len(outputs)} sequences in {generation_time:.2f}s")
        print(f"  Total tokens: {total_generated:,}")
        print(f"  Throughput: {tokens_per_sec:.1f} tokens/sec")

        # Clean up
        del llm
        torch.cuda.empty_cache()
        gc.collect()

    return results


def generate_continuations_anthropic(
        prompts: List[Dict[str, Any]],
        model_name: str,
        tokenizer,
        max_tokens_list: List[int],
        temperature: float = 1.0,
        seed: Optional[int] = None,
        concurrency: int = 10) -> List[Dict[str, Any]]:
    """
    Generate continuations for each prompt with different max_tokens settings using Anthropic API.

    Args:
        prompts: List of prompt dicts with 'text' and 'token_length'
        model_name: Anthropic model name (e.g., 'claude-sonnet-4-20250514')
        tokenizer: Tokenizer for measuring token lengths
        max_tokens_list: List of max_tokens to generate (e.g., [500, 1000, 2000])
        temperature: Sampling temperature
        seed: Random seed (note: Anthropic doesn't support seed, kept for compatibility)
        concurrency: Number of concurrent API calls (default: 10)

    Returns:
        List of generation results with metadata
    """
    import asyncio
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.llm_api import get_anthropic_key
    import anthropic

    async def generate_single(prompt_idx: int, prompt_data: Dict[str, Any],
                              max_new_tokens: int,
                              semaphore) -> Dict[str, Any]:
        """Generate a single completion with concurrency control."""
        async with semaphore:
            try:
                # Get API key
                api_key = get_anthropic_key()
                client = anthropic.AsyncAnthropic(api_key=api_key)

                # Call Anthropic API asynchronously
                response = await client.messages.create(
                    model=model_name,
                    messages=[{
                        "role": "user",
                        "content": prompt_data['text']
                    }],
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                )

                generated_text = response.content[0].text

                # Tokenize the generated text to measure length
                generated_ids = tokenizer.encode(generated_text,
                                                 add_special_tokens=False)

                return {
                    'prompt_id': prompt_idx,
                    'prompt_text': prompt_data['text'],
                    'prompt_token_length': prompt_data['token_length'],
                    'max_new_tokens': max_new_tokens,
                    'generated_text': generated_text,
                    'temperature': temperature,
                    'num_tokens': len(generated_ids),
                    'success': True,
                }

            except Exception as e:
                print(f"\n  Error generating for prompt {prompt_idx}: {e}")
                # Return empty result for failed generation
                return {
                    'prompt_id': prompt_idx,
                    'prompt_text': prompt_data['text'],
                    'prompt_token_length': prompt_data['token_length'],
                    'max_new_tokens': max_new_tokens,
                    'generated_text': '',
                    'temperature': temperature,
                    'num_tokens': 0,
                    'success': False,
                }

    async def generate_batch(prompts: List[Dict[str, Any]],
                             max_new_tokens: int) -> List[Dict[str, Any]]:
        """Generate completions for a batch of prompts with concurrency control."""
        semaphore = asyncio.Semaphore(concurrency)

        # Create tasks for all prompts
        tasks = [
            generate_single(prompt_idx, prompt_data, max_new_tokens, semaphore)
            for prompt_idx, prompt_data in enumerate(prompts)
        ]

        # Run with progress tracking
        batch_results = []
        completed = 0
        start_time = time.time()

        for coro in asyncio.as_completed(tasks):
            result = await coro
            batch_results.append(result)
            completed += 1

            # Progress indicator every 10 completions
            if completed % 10 == 0 or completed == len(tasks):
                elapsed = time.time() - start_time
                avg_time = elapsed / completed
                remaining = avg_time * (len(tasks) - completed)
                print(f"  Progress: {completed}/{len(tasks)} | "
                      f"Avg: {avg_time:.2f}s/prompt | "
                      f"ETA: {remaining/60:.1f}min")

        return batch_results

    results = []

    # Generate for each max_tokens setting
    for max_new_tokens in max_tokens_list:
        print(f"\n{'='*80}")
        print(
            f"Generating with max_tokens={max_new_tokens} using {model_name}")
        print(f"Concurrency: {concurrency} parallel requests")
        print(f"{'='*80}")

        print(
            f"Generating {len(prompts)} sequences with max_tokens={max_new_tokens}..."
        )
        start_time = time.time()

        # Run async batch generation
        batch_results = asyncio.run(generate_batch(prompts, max_new_tokens))

        # Sort results by prompt_id to maintain order
        batch_results.sort(key=lambda x: x['prompt_id'])

        # Add to results (removing the 'num_tokens' field we used internally)
        total_generated_tokens = 0
        for result in batch_results:
            total_generated_tokens += result.pop('num_tokens', 0)
            result.pop('success', None)
            results.append(result)

        generation_time = time.time() - start_time
        tokens_per_sec = total_generated_tokens / generation_time if generation_time > 0 else 0

        print(
            f"✓ Generated {len(prompts)} sequences in {generation_time:.2f}s")
        print(f"  Total tokens: {total_generated_tokens:,}")
        print(f"  Throughput: {tokens_per_sec:.1f} tokens/sec")

    return results


def save_dataset(results: List[Dict[str, Any]], output_path: Path,
                 metadata: Dict[str, Any]):
    """Save results to YAML with metadata."""
    dataset = {'metadata': metadata, 'samples': results}

    with open(output_path, 'w') as f:
        yaml.dump(dataset, f, default_flow_style=False, sort_keys=False)

    print(f"\n✓ Saved dataset to {output_path}")
    print(f"  Total samples: {len(results)}")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description=
        "Generate compression test dataset from conversation prompts")
    parser.add_argument(
        "--prompt-source",
        choices=["ultrachat", "lmsys"],
        default="ultrachat",
        help=
        "Source dataset for prompts: ultrachat or lmsys (default: ultrachat)")
    parser.add_argument(
        "--backend",
        choices=["vllm", "anthropic"],
        default="vllm",
        help=
        "Generation backend: vllm (local) or anthropic (API) (default: vllm)")
    parser.add_argument(
        "--model",
        default=None,
        help=
        "Model name. For vllm: HF model path (default: meta-llama/Llama-3.1-8B). "
        "For anthropic: claude model (default: claude-sonnet-4-20250514)")
    parser.add_argument("--num-prompts",
                        type=int,
                        default=40,
                        help="Number of prompts to sample (default: 100)")
    parser.add_argument("--prompt-length",
                        type=int,
                        default=500,
                        help="Target prompt length in tokens (default: 500)")
    parser.add_argument(
        "--max-tokens",
        type=int,
        nargs="+",
        default=[500, 1000],
        help="Max tokens to generate for each prompt (default: 500 1000 2000)")
    parser.add_argument("--temperature",
                        type=float,
                        default=1.0,
                        help="Sampling temperature (default: 1.0)")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=
        "Output YAML path (default: compression_dataset_YYYYMMDD_HHMMSS.yaml)")
    parser.add_argument("--gpu-memory-utilization",
                        type=float,
                        default=0.7,
                        help="GPU memory utilization for vLLM (default: 0.7)")
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help=
        "Number of concurrent API calls for Anthropic backend (default: 10)")
    parser.add_argument(
        "--no-english-filter",
        action="store_true",
        help=
        "Disable English-only filtering for prompts (default: English only)")

    args = parser.parse_args()

    # Set default model based on backend
    if args.model is None:
        if args.backend == "vllm":
            args.model = "meta-llama/Llama-3.1-8B"
        else:  # anthropic
            args.model = "claude-sonnet-4-20250514"

    # Set output path with timestamp if not specified
    if args.output is None:
        # Create data directory if it doesn't exist
        data_dir = Path(__file__).parent.parent / "data"
        data_dir.mkdir(exist_ok=True, parents=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backend_suffix = args.backend
        args.output = data_dir / f"compression_dataset_{backend_suffix}_{timestamp}.yaml"

    print("=" * 80)
    print(f"COMPRESSION DATASET GENERATION ({args.backend.upper()})")
    print("=" * 80)
    print(f"Prompt source: {args.prompt_source}")
    print(f"Backend: {args.backend}")
    print(f"Model: {args.model}")
    print(f"Number of prompts: {args.num_prompts}")
    print(f"Target prompt length: {args.prompt_length} tokens")
    print(f"Max tokens per generation: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Seed: {args.seed}")
    if args.backend == "vllm":
        print(f"GPU memory utilization: {args.gpu_memory_utilization}")
    else:  # anthropic
        print(f"Concurrency (parallel requests): {args.concurrency}")
    print(f"Output: {args.output}")
    print("=" * 80)

    # Load tokenizer (for filtering prompts and tokenizing generated text)
    # For Anthropic, we still need a tokenizer to measure prompt lengths
    if args.backend == "vllm":
        tokenizer_model = args.model
    else:
        # For Anthropic, use a standard tokenizer for measuring lengths
        tokenizer_model = "meta-llama/Llama-3.1-8B"

    print(f"\nLoading tokenizer: {tokenizer_model}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    print("✓ Tokenizer loaded")

    # Load and filter prompts based on source
    if args.prompt_source == "ultrachat":
        prompts = load_ultrachat_prompts(args.num_prompts, args.prompt_length,
                                         tokenizer)
    else:  # lmsys
        english_only = not args.no_english_filter
        prompts = load_lmsys_prompts(args.num_prompts,
                                     args.prompt_length,
                                     tokenizer,
                                     english_only=english_only)

    if len(prompts) < args.num_prompts:
        print(
            f"⚠️  Warning: Only found {len(prompts)} prompts, requested {args.num_prompts}"
        )

    # Generate continuations using selected backend
    if args.backend == "vllm":
        print(
            f"\nGenerating {len(args.max_tokens)} continuation(s) per prompt using vLLM..."
        )
        results = generate_continuations_vllm(
            prompts,
            args.model,
            tokenizer,
            args.max_tokens,
            temperature=args.temperature,
            gpu_memory_utilization=args.gpu_memory_utilization,
            seed=args.seed)
    else:  # anthropic
        print(
            f"\nGenerating {len(args.max_tokens)} continuation(s) per prompt using Anthropic API..."
        )
        results = generate_continuations_anthropic(
            prompts,
            args.model,
            tokenizer,
            args.max_tokens,
            temperature=args.temperature,
            seed=args.seed,
            concurrency=args.concurrency)

    # Prepare metadata
    metadata = {
        'created_at': datetime.now().isoformat(),
        'prompt_source': args.prompt_source,
        'backend': args.backend,
        'model': args.model,
        'num_prompts': len(prompts),
        'target_prompt_length': args.prompt_length,
        'max_tokens_list': args.max_tokens,
        'temperature': args.temperature,
        'seed': args.seed,
        'total_samples': len(results),
        'generation_method': args.backend,
    }

    # Add backend-specific metadata
    if args.backend == 'vllm':
        metadata['gpu_memory_utilization'] = args.gpu_memory_utilization

    # Save dataset
    save_dataset(results, args.output, metadata)

    # Print statistics
    print("\n" + "=" * 80)
    print("DATASET STATISTICS")
    print("=" * 80)
    for max_tokens in args.max_tokens:
        samples = [r for r in results if r['max_new_tokens'] == max_tokens]
        # Calculate average length by tokenizing (samples don't store token IDs anymore)
        lengths = [
            len(tokenizer.encode(r['generated_text'],
                                 add_special_tokens=False)) for r in samples
        ]
        avg_length = sum(lengths) / len(lengths) if lengths else 0
        print(
            f"max_tokens={max_tokens}: {len(samples)} samples, avg length={avg_length:.1f} tokens"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
