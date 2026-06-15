#!/usr/bin/env python3
"""
Generate text using Cartesian product of: model-LoRA × temperature × topic.

Creates a comprehensive dataset for evaluating LoRA effects on generation and compression.
"""

import argparse
from pathlib import Path
from typing import List, Dict, Optional
import yaml
from datetime import datetime
from datasets import load_dataset as hf_load_dataset

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

TEMPERATURES = [0.0, 0.5, 1.0, 2.0]


def load_prompts(dataset_name: str,
                 split: str = "valid",
                 limit: Optional[int] = None) -> List[Dict]:
    """Load prompts from a Lots-of-LoRAs dataset.

    Args:
        dataset_name: HuggingFace dataset name
        split: Dataset split to use
        limit: Optional limit on number of prompts

    Returns:
        List of prompt dicts with 'id' and 'prompt' keys
    """
    print(f"Loading prompts from {dataset_name} ({split})...")
    dataset = hf_load_dataset(dataset_name, split=split)

    prompts = []
    for i, item in enumerate(dataset):
        if limit and i >= limit:
            break

        # Use 'input' field as prompt
        prompt_text = item.get('input', '')
        prompt_id = item.get('id', f'{dataset_name}-{i}')

        prompts.append({
            'id': prompt_id,
            'prompt': prompt_text,
        })

    print(f"  Loaded {len(prompts)} prompts")
    return prompts


def load_vllm_model(base_model: str, enable_lora: bool = True):
    """Load vLLM model once for reuse across generations.

    Args:
        base_model: Base model name
        enable_lora: Whether to enable LoRA support

    Returns:
        vLLM LLM instance
    """
    from vllm import LLM

    print(f"Loading model: {base_model} (enable_lora={enable_lora})")

    llm = LLM(
        model=base_model,
        enable_lora=enable_lora,
        max_loras=4,  # Support multiple LoRAs
        max_model_len=2048,
        gpu_memory_utilization=0.9,
    )

    return llm


def generate_with_vllm(
    llm,
    prompts: List[Dict],
    lora_adapter: Optional[str],
    temperature: float,
    max_tokens: int = 200,
    top_p: float = 0.95,
) -> List[Dict]:
    """Generate completions using vLLM.

    Args:
        llm: Pre-loaded vLLM model instance
        prompts: List of prompt dicts
        lora_adapter: Optional LoRA adapter name
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        top_p: Nucleus sampling parameter

    Returns:
        List of results with 'prompt_id', 'prompt', 'generated_text'
    """
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest

    # Configure sampling
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    # Prepare prompts
    prompt_texts = [p['prompt'] for p in prompts]

    # Generate
    print(
        f"Generating {len(prompt_texts)} completions (temp={temperature}, max_tokens={max_tokens})..."
    )

    if lora_adapter:
        print(f"  Using LoRA: {lora_adapter}")
        # Generate with LoRA
        lora_request = LoRARequest(
            lora_name=lora_adapter.split('/')[-1],  # Use task ID as name
            lora_int_id=1,
            lora_path=lora_adapter,
        )
        outputs = llm.generate(prompt_texts,
                               sampling_params,
                               lora_request=lora_request)
    else:
        # Generate without LoRA (baseline)
        outputs = llm.generate(prompt_texts, sampling_params)

    # Collect results
    results = []
    for prompt, output in zip(prompts, outputs):
        generated_text = output.outputs[0].text
        results.append({
            'prompt_id': prompt['id'],
            'prompt': prompt['prompt'],
            'generated_text': generated_text,
        })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate text using LoRA Cartesian product")
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
            "/n/netscratch/sham_lab/Lab/rrinberg/compression/lora_cartesian"),
        help="Output directory for generated data")
    parser.add_argument("--max-tokens",
                        type=int,
                        default=200,
                        help="Maximum tokens to generate per prompt")
    parser.add_argument("--limit-prompts",
                        type=int,
                        default=None,
                        help="Limit number of prompts per topic (for testing)")
    parser.add_argument("--split",
                        default="valid",
                        choices=["train", "test", "valid"],
                        help="Dataset split to use for prompts")
    parser.add_argument("--skip-baseline",
                        action="store_true",
                        help="Skip baseline (no LoRA) generation")
    parser.add_argument(
        "--loras-only",
        nargs="+",
        metavar="TASK",
        help="Only test specific LoRAs (e.g., task561 task581)")
    parser.add_argument(
        "--topics-only",
        nargs="+",
        metavar="TASK",
        help="Only use specific topics (e.g., task561 task581)")
    parser.add_argument("--temps-only",
                        nargs="+",
                        type=float,
                        help="Only test specific temperatures (e.g., 0.0 1.0)")

    args = parser.parse_args()

    # Determine which configurations to test
    loras_to_test = args.loras_only if args.loras_only else LORAS
    topics_to_test = args.topics_only if args.topics_only else LORAS
    temps_to_test = args.temps_only if args.temps_only else TEMPERATURES

    # Add baseline if not skipped
    models_to_test = ["baseline"] if not args.skip_baseline else []
    models_to_test.extend(loras_to_test)

    total_combinations = len(models_to_test) * len(temps_to_test) * len(
        topics_to_test)

    print(f"\n{'='*80}")
    print("LoRA Cartesian Product Text Generation")
    print(f"{'='*80}")
    print(f"Base model: {args.base_model}")
    print(
        f"Models: {len(models_to_test)} (baseline + {len(loras_to_test)} LoRAs)"
    )
    print(f"Temperatures: {len(temps_to_test)} {temps_to_test}")
    print(f"Topics: {len(topics_to_test)} {topics_to_test}")
    print(f"Total combinations: {total_combinations}")
    print(f"Max tokens per prompt: {args.max_tokens}")
    if args.limit_prompts:
        print(f"Limiting to {args.limit_prompts} prompts per topic")
    print(f"Output: {args.output_dir}")
    print(f"{'='*80}\n")

    # Create output directory (no timestamp - reuse existing)
    run_dir = args.output_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"✓ Output directory: {run_dir}\n")

    # Save configuration
    config = {
        'created_at': datetime.now().isoformat(),
        'base_model': args.base_model,
        'lora_rank': args.lora_rank,
        'lora_bits': args.lora_bits,
        'max_tokens': args.max_tokens,
        'split': args.split,
        'models': models_to_test,
        'temperatures': temps_to_test,
        'topics': topics_to_test,
        'total_combinations': total_combinations,
    }

    with open(run_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Load model ONCE (with LoRA support enabled if we have any LoRAs to test)
    has_loras = any(m != "baseline" for m in models_to_test)
    llm = load_vllm_model(args.base_model, enable_lora=has_loras)

    # Pre-load prompts for all topics (avoid reloading same dataset)
    prompts_cache = {}
    for topic in topics_to_test:
        dataset_name = LORA_DATASETS[topic]
        prompts_cache[topic] = load_prompts(dataset_name,
                                            split=args.split,
                                            limit=args.limit_prompts)

    # Generate for each combination
    current = 0
    for model_name in models_to_test:
        # Construct LoRA adapter name once per model
        if model_name == "baseline":
            lora_adapter = None
            model_label = "baseline"
        else:
            lora_adapter = f"Lots-of-LoRAs/Mistral-7B-Instruct-v0.2-{args.lora_bits}b-r{args.lora_rank}-{model_name}"
            model_label = model_name

        for temperature in temps_to_test:
            for topic in topics_to_test:
                current += 1

                print(f"\n{'='*80}")
                print(f"Combination {current}/{total_combinations}")
                print(f"{'='*80}")
                print(f"Model: {model_name}")
                print(f"Temperature: {temperature}")
                print(f"Topic: {topic}")
                print(f"{'='*80}\n")

                # Check if already generated
                output_file = run_dir / f"model_{model_label}_temp_{temperature}_topic_{topic}.yaml"
                if output_file.exists():
                    print(f"✓ Already exists, skipping: {output_file.name}")
                    continue

                # Get cached prompts
                prompts = prompts_cache[topic]

                # Generate
                try:
                    results = generate_with_vllm(
                        llm=llm,
                        prompts=prompts,
                        lora_adapter=lora_adapter,
                        temperature=temperature,
                        max_tokens=args.max_tokens,
                    )

                    output_data = {
                        'metadata': {
                            'created_at': datetime.now().isoformat(),
                            'model': model_label,
                            'lora_adapter': lora_adapter,
                            'temperature': temperature,
                            'topic': topic,
                            'dataset': dataset_name,
                            'split': args.split,
                            'num_samples': len(results),
                            'max_tokens': args.max_tokens,
                        },
                        'samples': results,
                    }

                    with open(output_file, 'w') as f:
                        yaml.dump(output_data,
                                  f,
                                  default_flow_style=False,
                                  sort_keys=False)

                    print(
                        f"\n✓ Saved {len(results)} generated samples to {output_file}"
                    )

                except Exception as e:
                    print(
                        f"\n✗ Error generating for {model_label} / temp={temperature} / topic={topic}"
                    )
                    print(f"  Error: {e}")
                    continue

    print(f"\n{'='*80}")
    print("Generation Complete!")
    print(f"{'='*80}")
    print(f"Total combinations: {total_combinations}")
    print(f"Results saved to: {run_dir}")
    print(f"{'='*80}\n")

    # Create summary
    summary = {
        'completed_at': datetime.now().isoformat(),
        'config': config,
        'total_combinations': total_combinations,
    }

    with open(run_dir / "summary.yaml", 'w') as f:
        yaml.dump(summary, f, default_flow_style=False)


if __name__ == "__main__":
    main()
