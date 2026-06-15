#!/usr/bin/env python3
"""
Generate topic-specific text data for distillation and compression evaluation.
Creates train/test splits (80/20) for multiple topics.
Supports both vLLM and transformers backends.
"""

import argparse
from tqdm import tqdm
import yaml
import os
from typing import List, Dict, Optional
import random
import torch

# Topic-specific prompts to guide generation
TOPIC_PROMPTS = {
    "music": [
        "The history of jazz music began in",
        "When composing a symphony, the most important elements are",
        "The Beatles revolutionized modern music by",
        "Classical music theory teaches us that",
        "The evolution of hip-hop from the 1980s to today shows",
        "Playing the piano requires understanding",
        "Music production techniques have evolved with",
        "The relationship between melody and harmony in",
        "Famous composers like Mozart and Beethoven",
        "The impact of electronic music on contemporary",
    ],
    "math": [
        "To prove this theorem, we first observe that",
        "The fundamental theorem of calculus states that",
        "In linear algebra, matrix multiplication is defined as",
        "Let x be a real number such that",
        "The derivative of f(x) with respect to x is",
        "Solving the differential equation requires",
        "The proof by induction proceeds as follows:",
        "In probability theory, the expected value of",
        "Consider a function f: R → R defined by",
        "The integral from 0 to infinity of",
    ],
    "coding": [
        "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n",
        "class BinaryTree:\n    def __init__(self, value):\n        self.value = value\n        self.left = None\n        self.right = None\n    \n    def insert(self, value):\n",
        "import numpy as np\nimport tensorflow as tf\n\ndef build_neural_network(input_dim, hidden_dim, output_dim):\n",
        "// Implementing a hash map in C++\n#include <vector>\n#include <list>\n\ntemplate<typename K, typename V>\nclass HashMap {\nprivate:\n",
        "fn fibonacci(n: u64) -> u64 {\n    match n {\n        0 => 0,\n        1 => 1,\n        _ => fibonacci(n - 1) + fibonacci(n - 2),\n    }\n}\n\n",
        "SELECT users.name, COUNT(orders.id) as order_count\nFROM users\nLEFT JOIN orders ON users.id = orders.user_id\n",
        "async function fetchUserData(userId) {\n    try {\n        const response = await fetch(`/api/users/${userId}`);\n",
        "package main\n\nimport (\n    \"fmt\"\n    \"sync\"\n)\n\nfunc worker(id int, jobs <-chan int, results chan<- int) {\n",
    ],
    "financial": [
        "Investment strategies for retirement planning should consider",
        "The Federal Reserve's monetary policy affects",
        "When evaluating a stock's valuation, key metrics include",
        "Portfolio diversification is important because",
        "The difference between stocks and bonds lies in",
        "Compound interest over time demonstrates that",
        "Risk management in financial markets requires",
        "The relationship between inflation and interest rates",
        "Tax-efficient investing strategies involve",
        "Modern portfolio theory suggests that",
    ],
}


def generate_sequences_vllm(
    llm,
    topic: str,
    num_sequences: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[List[str], List[str]]:
    """Generate sequences for a specific topic using vLLM.

    Returns:
        Tuple of (prompts, completions) where each completion is the generated continuation
    """
    from vllm import SamplingParams

    topic_prompts = TOPIC_PROMPTS[topic]

    # Create prompts by cycling through topic prompts
    prompts = [
        topic_prompts[i % len(topic_prompts)] for i in range(num_sequences)
    ]

    print(f"\nGenerating {num_sequences} sequences for topic: {topic}")

    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    # Generate all sequences in batches (vLLM handles batching internally)
    outputs = llm.generate(prompts, sampling_params)

    # Extract generated completions (vLLM returns just the completion, not the prompt)
    completions = [output.outputs[0].text for output in outputs]

    return prompts, completions


def generate_sequences_transformers(
    model,
    tokenizer,
    topic: str,
    num_sequences: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    device: str,
) -> tuple[List[str], List[str]]:
    """Generate sequences for a specific topic using transformers.

    Returns:
        Tuple of (prompts, completions) where each completion is the generated continuation
    """
    topic_prompts = TOPIC_PROMPTS[topic]

    # Create prompts by cycling through topic prompts
    prompts = [
        topic_prompts[i % len(topic_prompts)] for i in range(num_sequences)
    ]

    print(f"\nGenerating {num_sequences} sequences for topic: {topic}")

    completions = []

    for prompt in tqdm(prompts, desc=f"Generating {topic}"):
        # Tokenize prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode only the generated part (excluding prompt)
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
        completions.append(completion)

    return prompts, completions


def split_train_test(
    prompts: List[str],
    completions: List[str],
    train_ratio: float = 0.8
) -> tuple[tuple[List[str], List[str]], tuple[List[str], List[str]]]:
    """Split sequences into train and test sets.

    Returns:
        ((train_prompts, train_completions), (test_prompts, test_completions))
    """
    # Create paired list and shuffle
    paired = list(zip(prompts, completions))
    random.shuffle(paired)

    split_idx = int(len(paired) * train_ratio)
    train_pairs = paired[:split_idx]
    test_pairs = paired[split_idx:]

    # Unzip back into separate lists
    train_prompts, train_completions = zip(
        *train_pairs) if train_pairs else ([], [])
    test_prompts, test_completions = zip(*test_pairs) if test_pairs else ([],
                                                                          [])

    return (list(train_prompts),
            list(train_completions)), (list(test_prompts),
                                       list(test_completions))


def save_text_format(prompts: List[str], completions: List[str],
                     filepath: str):
    """Save sequences as text file (one per line) for distillation.

    Combines prompts and completions into full sequences.
    """
    with open(filepath, 'w') as f:
        for prompt, completion in zip(prompts, completions):
            # Combine prompt and completion
            full_seq = prompt + completion
            # Replace newlines with special token to keep one sequence per line
            seq_clean = full_seq.replace('\n', ' \\n ')
            f.write(seq_clean + '\n')
    print(f"Saved {len(prompts)} sequences to {filepath}")


def save_yaml_format(prompts: List[str], completions: List[str], filepath: str,
                     topic: str):
    """Save sequences as YAML for compression evaluation."""
    samples = []
    for prompt, completion in zip(prompts, completions):
        samples.append({
            'prompt': prompt,
            'generated_text': completion,
            'topic': topic
        })

    with open(filepath, 'w') as f:
        yaml.dump(samples, f, default_flow_style=False, sort_keys=False)
    print(f"Saved {len(samples)} samples to {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate topic-specific text data")
    parser.add_argument("model_path",
                        type=str,
                        help="Path or name of model to use for generation")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/n/netscratch/sham_lab/Lab/rrinberg/compression",
        help="Output directory for generated data")
    parser.add_argument("--sequences-per-topic",
                        type=int,
                        default=250,
                        help="Number of sequences to generate per topic")
    parser.add_argument("--max-length",
                        type=int,
                        default=512,
                        help="Maximum length of generated sequences")
    parser.add_argument("--temperature",
                        type=float,
                        default=0.9,
                        help="Sampling temperature")
    parser.add_argument("--top-p",
                        type=float,
                        default=0.95,
                        help="Nucleus sampling top-p")
    parser.add_argument("--train-ratio",
                        type=float,
                        default=0.8,
                        help="Ratio of data for training (rest is test)")
    parser.add_argument(
        "--topics",
        nargs="+",
        choices=["music", "math", "coding", "financial", "all"],
        default=["all"],
        help="Topics to generate data for")
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="Random seed for reproducibility")
    parser.add_argument(
        "--backend",
        type=str,
        choices=["vllm", "transformers"],
        default="vllm",
        help="Backend to use for generation (vllm or transformers)")
    parser.add_argument(
        "--quantization",
        type=str,
        choices=["awq", "gptq", "squeezellm", "fp8", "4bit", "8bit"],
        help="Quantization method (vLLM: awq/gptq/fp8, transformers: 4bit/8bit)"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs to use for tensor parallelism (vLLM only)")
    parser.add_argument("--gpu-memory-utilization",
                        type=float,
                        default=0.9,
                        help="GPU memory utilization (0.0-1.0, vLLM only)")
    parser.add_argument("--max-model-len",
                        type=int,
                        default=None,
                        help="Maximum model sequence length (vLLM only)")

    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Determine topics to generate
    if "all" in args.topics:
        topics = ["music", "math", "coding", "financial"]
    else:
        topics = args.topics

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model based on backend
    model = None
    tokenizer = None
    llm = None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.backend == "vllm":
        from vllm import LLM

        print(f"Loading model with vLLM: {args.model_path}")
        llm_kwargs = {
            "model": args.model_path,
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
        }

        if args.max_model_len:
            llm_kwargs["max_model_len"] = args.max_model_len
            print(f"Using max model length: {args.max_model_len}")

        if args.quantization and args.quantization in [
                "awq", "gptq", "squeezellm", "fp8"
        ]:
            llm_kwargs["quantization"] = args.quantization
            print(f"Using quantization: {args.quantization}")

        if args.tensor_parallel_size > 1:
            print(
                f"Using tensor parallelism across {args.tensor_parallel_size} GPUs"
            )

        llm = LLM(**llm_kwargs)

    elif args.backend == "transformers":
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        print(f"Loading model with transformers: {args.model_path}")
        print(f"Device: {device}")

        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs = {}

        if args.quantization:
            if args.quantization == "4bit":
                print("Using 4-bit quantization (NF4)")
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4")
                model_kwargs['quantization_config'] = bnb_config
                model_kwargs['device_map'] = 'auto'
            elif args.quantization == "8bit":
                print("Using 8-bit quantization")
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                model_kwargs['quantization_config'] = bnb_config
                model_kwargs['device_map'] = 'auto'
        else:
            model_kwargs[
                'torch_dtype'] = torch.float16 if device == "cuda" else torch.float32

        model = AutoModelForCausalLM.from_pretrained(args.model_path,
                                                     **model_kwargs)
        if 'device_map' not in model_kwargs:
            model = model.to(device)
        model.eval()

    print(f"\nConfiguration:")
    print(f"  Topics: {topics}")
    print(f"  Sequences per topic: {args.sequences_per_topic}")
    print(f"  Max length: {args.max_length}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Top-p: {args.top_p}")
    print(
        f"  Train/Test split: {args.train_ratio:.0%}/{1-args.train_ratio:.0%}")
    print(f"  Output directory: {args.output_dir}")
    print("-" * 80)

    # Generate data for each topic
    all_train_prompts = []
    all_train_completions = []
    all_test_prompts = []
    all_test_completions = []
    all_test_topics = []

    for topic in topics:
        # Generate sequences based on backend
        if args.backend == "vllm":
            prompts, completions = generate_sequences_vllm(
                llm,
                topic,
                args.sequences_per_topic,
                args.max_length,
                args.temperature,
                args.top_p,
            )
        else:  # transformers
            prompts, completions = generate_sequences_transformers(
                model,
                tokenizer,
                topic,
                args.sequences_per_topic,
                args.max_length,
                args.temperature,
                args.top_p,
                device,
            )

        # Split train/test
        (train_prompts,
         train_completions), (test_prompts,
                              test_completions) = split_train_test(
                                  prompts, completions, args.train_ratio)

        print(f"  Train: {len(train_prompts)} sequences")
        print(f"  Test: {len(test_prompts)} sequences")

        # Accumulate for combined files
        all_train_prompts.extend(train_prompts)
        all_train_completions.extend(train_completions)
        all_test_prompts.extend(test_prompts)
        all_test_completions.extend(test_completions)
        all_test_topics.extend([topic] * len(test_prompts))

        # Save topic-specific files
        topic_dir = os.path.join(args.output_dir, topic)
        os.makedirs(topic_dir, exist_ok=True)

        # Save train as text for distillation
        save_text_format(train_prompts, train_completions,
                         os.path.join(topic_dir, f"{topic}_train.txt"))

        # Save test as YAML for compression evaluation
        save_yaml_format(test_prompts, test_completions,
                         os.path.join(topic_dir, f"{topic}_test.yaml"), topic)

    # Save combined train file (all topics mixed)
    print("\nSaving combined files...")
    # Shuffle train data together
    train_paired = list(zip(all_train_prompts, all_train_completions))
    random.shuffle(train_paired)
    shuffled_train_prompts, shuffled_train_completions = zip(
        *train_paired) if train_paired else ([], [])

    save_text_format(list(shuffled_train_prompts),
                     list(shuffled_train_completions),
                     os.path.join(args.output_dir, "all_topics_train.txt"))

    # Save combined test YAML (shuffle test data)
    test_paired = list(zip(all_test_prompts, all_test_completions))
    random.shuffle(test_paired)
    shuffled_test_prompts, shuffled_test_completions = zip(
        *test_paired) if test_paired else ([], [])

    save_yaml_format(list(shuffled_test_prompts),
                     list(shuffled_test_completions),
                     os.path.join(args.output_dir, "all_topics_test.yaml"),
                     "mixed")

    print("\n" + "=" * 80)
    print("Data generation complete!")
    print(f"Total train sequences: {len(all_train_prompts)}")
    print(f"Total test sequences: {len(all_test_prompts)}")
    print(f"\nUse for distillation:")
    print(f"  python compression/knowledge_distillation.py \\")
    print(f"    --mode supervised \\")
    print(f"    --data-file {args.output_dir}/all_topics_train.txt \\")
    print(f"    --teacher-model <teacher> --student-model <student>")
    print(f"\nUse for compression evaluation:")
    print(f"  python scripts/measure_compression.py <model> \\")
    print(f"    {args.output_dir}/all_topics_test.yaml")
    print("=" * 80)


if __name__ == "__main__":
    main()
