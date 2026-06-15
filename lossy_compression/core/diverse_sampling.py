#!/usr/bin/env python3
"""
Diverse Sampling: Various techniques for generating diverse outputs from LLMs.

This module provides different strategies for generating diverse text samples
from language models, useful for exploration, creativity, and finding optimal outputs.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# Import LLM API utilities
from utils.llm_api import anthropic_completion

# Default model
DEFAULT_MODEL = "claude-3-5-haiku-20241022"
DEFAULT_SEED = 42


def diverse_sample_api_vary_seed(
        prompt: str,
        num_samples: int,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.8,
        max_tokens: int = 200,
        base_seed: int = DEFAULT_SEED,
        verbose: bool = False) -> List[Dict[str, Any]]:
    """
    Generate diverse samples by varying the random seed.
    
    Each sample uses a different seed to ensure different random paths
    through the model's probability distributions.
    
    Args:
        prompt: Input prompt for generation
        num_samples: Number of samples to generate
        model: Model identifier
        temperature: Sampling temperature (constant across samples)
        max_tokens: Maximum tokens per generation
        base_seed: Base seed (incremented for each sample)
        verbose: Print progress information
        
    Returns:
        List of dicts with 'text', 'seed', and metadata
    """
    samples = []

    if verbose:
        print(f"Generating {num_samples} samples with varying seeds...")

    for i in range(num_samples):
        seed = base_seed + i

        try:
            output = anthropic_completion(prompt=prompt,
                                          model=model,
                                          max_tokens=max_tokens,
                                          temperature=temperature,
                                          seed=seed)

            samples.append({
                'text': output,
                'seed': seed,
                'temperature': temperature,
                'method': 'vary_seed',
                'index': i
            })

        except Exception as e:
            if verbose:
                print(f"  Sample {i+1} failed: {e}")
            continue

    return samples


def diverse_sample_api_vary_temperature(
        prompt: str,
        num_samples: int,
        model: str = DEFAULT_MODEL,
        temperature_range: Tuple[float, float] = (0.3, 1.5),
        max_tokens: int = 200,
        seed: Optional[int] = None,
        verbose: bool = False) -> List[Dict[str, Any]]:
    """
    Generate diverse samples by varying the temperature parameter.
    
    Temperature controls randomness: lower values (0.3) are more focused,
    higher values (1.5) are more creative/random.
    
    Args:
        prompt: Input prompt for generation
        num_samples: Number of samples to generate
        model: Model identifier
        temperature_range: (min, max) temperature values
        max_tokens: Maximum tokens per generation
        seed: Random seed (if None, uses different seeds)
        verbose: Print progress information
        
    Returns:
        List of dicts with 'text', 'temperature', and metadata
    """
    samples = []

    # Generate temperature values
    if num_samples == 1:
        temperatures = [np.mean(temperature_range)]
    else:
        temperatures = np.linspace(temperature_range[0], temperature_range[1],
                                   num_samples)

    if verbose:
        print(
            f"Generating {num_samples} samples with temperatures from {temperature_range[0]:.2f} to {temperature_range[1]:.2f}"
        )

    for i, temp in enumerate(temperatures):
        # Use different seeds if not specified
        current_seed = (seed + i) if seed is not None else None

        try:
            output = anthropic_completion(prompt=prompt,
                                          model=model,
                                          max_tokens=max_tokens,
                                          temperature=temp,
                                          seed=current_seed)

            samples.append({
                'text': output,
                'temperature': float(temp),
                'seed': current_seed,
                'method': 'vary_temperature',
                'index': i
            })

        except Exception as e:
            if verbose:
                print(f"  Sample {i+1} failed: {e}")
            continue

    return samples


def diverse_sample_api_vary_iterative_sampling(
    prompt: str,
    num_samples: int,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.8,
    max_tokens: int = 200,
    seed: Optional[int] = DEFAULT_SEED,
    verbose: bool = False,
    diversity_instruction:
    str = "Generate a different response from the previous ones. Be creative and explore different angles."
) -> List[Dict[str, Any]]:
    """
    Generate diverse samples using iterative sampling with context.
    
    Each new sample is generated with knowledge of previous samples,
    explicitly instructed to be different.
    
    Args:
        prompt: Input prompt for generation
        num_samples: Number of samples to generate
        model: Model identifier
        temperature: Sampling temperature
        max_tokens: Maximum tokens per generation
        seed: Random seed
        verbose: Print progress information
        diversity_instruction: Instruction for ensuring diversity
        
    Returns:
        List of dicts with 'text' and metadata
    """
    samples = []

    if verbose:
        print(f"Generating {num_samples} samples with iterative sampling...")

    for i in range(num_samples):
        # Build context with previous samples
        if i == 0:
            # First sample - just use the original prompt
            current_prompt = prompt
        else:
            # Include previous samples as context
            previous_samples_text = "\n\n".join([
                f"Previous response {j+1}:\n{s['text']}"
                for j, s in enumerate(samples)
            ])

            current_prompt = f"""{prompt}

{previous_samples_text}

{diversity_instruction}

New response:"""

        current_seed = (seed + i) if seed is not None else None

        try:
            output = anthropic_completion(prompt=current_prompt,
                                          model=model,
                                          max_tokens=max_tokens,
                                          temperature=temperature,
                                          seed=current_seed)

            samples.append({
                'text': output,
                'temperature': temperature,
                'seed': current_seed,
                'method': 'iterative_sampling',
                'iteration': i,
                'context_samples': i
            })

        except Exception as e:
            if verbose:
                print(f"  Sample {i+1} failed: {e}")
            continue

    return samples


def diverse_sample_api_with_llm(
        prompt: str,
        num_samples: int,
        model: str = DEFAULT_MODEL,
        orchestrator_model: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0.8,
        top_p: float = 0.9,
        seed: int = DEFAULT_SEED,
        max_tokens: int = 200,
        verbose: bool = False) -> List[Dict[str, Any]]:
    """
    Generate diverse samples using an LLM to orchestrate diversity.
    
    A more powerful LLM generates diverse variations of the prompt,
    then each variation is used to generate a sample.
    
    Args:
        prompt: Input prompt for generation
        num_samples: Number of samples to generate
        model: Model for generating final samples
        orchestrator_model: Model for generating prompt variations
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        seed: Random seed
        max_tokens: Maximum tokens per generation
        verbose: Print progress information
        
    Returns:
        List of dicts with 'text', 'variation_prompt', and metadata
    """
    samples = []

    if verbose:
        print(f"Generating {num_samples} samples with LLM orchestration...")

    # First, use orchestrator to generate diverse prompt variations
    variation_prompt = f"""Given this prompt: "{prompt}"

Generate {num_samples} diverse variations or perspectives on this prompt. Each variation should:
1. Maintain the core intent but approach it differently
2. Use different styles, tones, or angles
3. Be substantively different from each other

Format your response as a numbered list with ONLY the prompt variations, nothing else:
1. [first variation]
2. [second variation]
...
"""

    try:
        variations_text = anthropic_completion(
            prompt=variation_prompt,
            model=orchestrator_model,
            max_tokens=max_tokens * 2,  # Give more space for variations
            temperature=0.7,  # Moderate temperature for creativity
            seed=seed)

        # Parse variations
        variations = []
        for line in variations_text.strip().split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                # Remove numbering and clean up
                import re
                cleaned = re.sub(r'^[\d\-\.\)]+\s*', '', line).strip()
                if cleaned:
                    variations.append(cleaned)

    except Exception as e:
        if verbose:
            print(f"  Failed to generate variations: {e}")
        # Fallback to simple variation
        variations = [prompt] * num_samples

    # Generate samples using each variation
    for i, variation in enumerate(variations[:num_samples]):
        current_seed = (seed + i + 100) if seed is not None else None

        try:
            output = anthropic_completion(prompt=variation,
                                          model=model,
                                          max_tokens=max_tokens,
                                          temperature=temperature,
                                          seed=current_seed,
                                          top_p=top_p)

            samples.append({
                'text': output,
                'variation_prompt': variation,
                'original_prompt': prompt,
                'temperature': temperature,
                'top_p': top_p,
                'seed': current_seed,
                'method': 'llm_orchestrated',
                'index': i
            })

        except Exception as e:
            if verbose:
                print(f"  Sample {i+1} failed: {e}")
            continue

    return samples
