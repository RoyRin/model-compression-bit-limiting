#!/usr/bin/env python3
"""Extra prompt selection experiment.

Finds optimal token insertions between context and response that minimize
cross-entropy of the response predictions using beam search, then measures
actual compression improvement with arithmetic coding.

Given: [context] [response]
Find: [context] [inserted-tokens] [response] that minimizes CE(response)

Compression ratio = (compressed_response_bytes + inserted_token_bytes) / original_response_bytes
"""

import argparse
import json
import sys
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from compression.block_encoding_arithmetic_coder import BlockEmissionArithmeticCoder
from compression.activation_steering import ActivationSteering, run_pca_steering_experiment


@dataclass
class BeamCandidate:
    """A candidate in beam search."""
    tokens: List[int]  # inserted tokens
    ce_score: float  # cross-entropy of response with this insertion


def load_lmsys_data(num_examples: int = 10,
                    max_response_tokens: int = 50,
                    tokenizer=None) -> List[Dict]:
    """Load context-response pairs from lmsys-chat-1m dataset.

    Args:
        num_examples: Number of examples to load
        max_response_tokens: Maximum tokens in response (will truncate)
        tokenizer: Tokenizer for truncation

    Returns:
        List of dicts with 'context' and 'response' keys
    """
    print(f"Loading lmsys-chat-1m dataset (first {num_examples} examples)...")
    dataset = load_dataset("lmsys/lmsys-chat-1m",
                           split="train",
                           streaming=True)

    examples = []
    for i, item in enumerate(dataset):
        if i >= num_examples:
            break

        # lmsys format: conversation is a list of {"role": ..., "content": ...}
        conversation = item.get("conversation", [])
        if len(conversation) < 2:
            continue

        # Get first user message as context, first assistant message as response
        context = None
        response = None
        for turn in conversation:
            if turn["role"] == "user" and context is None:
                context = turn["content"]
            elif turn[
                    "role"] == "assistant" and context is not None and response is None:
                response = turn["content"]
                break

        if context and response:
            # Truncate response to max_response_tokens
            if tokenizer:
                response_tokens = tokenizer.encode(response,
                                                   add_special_tokens=False)
                if len(response_tokens) > max_response_tokens:
                    response_tokens = response_tokens[:max_response_tokens]
                    response = tokenizer.decode(response_tokens)

            examples.append({"context": context, "response": response})

    print(f"Loaded {len(examples)} context-response pairs")
    return examples


def load_hard_lmsys_examples(
    model,
    tokenizer,
    model_name: str,
    num_examples: int = 10,
    num_candidates: int = 200,
    max_response_tokens: int = 50,
    min_response_tokens: int = 20,
    device: str = "cuda",
    cache_dir: Path = Path("data/hard_examples_cache")
) -> List[Dict]:
    """Load context-response pairs that are hardest for the model to predict.

    Caches results to avoid recomputing. Cache key is based on model name and parameters.

    Args:
        model: Language model to evaluate
        tokenizer: Tokenizer
        model_name: Model name (for cache key)
        num_examples: Number of hard examples to return
        num_candidates: Number of candidates to scan through
        max_response_tokens: Maximum tokens in response
        min_response_tokens: Minimum tokens in response (skip short ones)
        device: Device
        cache_dir: Directory to cache results

    Returns:
        List of dicts with 'context', 'response', and 'avg_ce' keys
    """
    # Create cache key from parameters
    model_short = model_name.split("/")[-1]
    cache_file = cache_dir / f"hard_examples_{model_short}_n{num_candidates}_min{min_response_tokens}_max{max_response_tokens}.json"

    # Check if cache exists
    if cache_file.exists():
        print(f"Loading cached hard examples from {cache_file}")
        with open(cache_file, "r") as f:
            all_cached = json.load(f)
        # Return requested number of examples
        hard_examples = all_cached[:num_examples]
        print(
            f"Loaded {len(hard_examples)} hard examples from cache (avg_ce range: {hard_examples[0]['avg_ce']:.3f} - {hard_examples[-1]['avg_ce']:.3f})"
        )
        return hard_examples

    print(f"Scanning {num_candidates} examples to find hardest ones...")
    dataset = load_dataset("lmsys/lmsys-chat-1m",
                           split="train",
                           streaming=True)

    candidates = []
    scanned = 0

    for item in dataset:
        if scanned >= num_candidates:
            break

        conversation = item.get("conversation", [])
        if len(conversation) < 2:
            continue

        context = None
        response = None
        for turn in conversation:
            if turn["role"] == "user" and context is None:
                context = turn["content"]
            elif turn[
                    "role"] == "assistant" and context is not None and response is None:
                response = turn["content"]
                break

        if not context or not response:
            continue

        # Tokenize and check length
        response_tokens = tokenizer.encode(response, add_special_tokens=False)
        if len(response_tokens) < min_response_tokens:
            continue
        if len(response_tokens) > max_response_tokens:
            response_tokens = response_tokens[:max_response_tokens]
            response = tokenizer.decode(response_tokens)

        # Compute CE for this example
        context_ids = tokenizer.encode(context, add_special_tokens=True)
        full_ids = context_ids + response_tokens
        input_ids = torch.tensor([full_ids], device=device)

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits

        response_start = len(context_ids)
        total_ce = 0.0
        num_tokens = 0

        for i, target_token in enumerate(response_tokens):
            pred_pos = response_start + i - 1
            if pred_pos < 0:
                continue
            probs = F.softmax(logits[0, pred_pos, :].float(), dim=-1)
            ce = -torch.log(probs[target_token] + 1e-10).item()
            total_ce += ce
            num_tokens += 1

        avg_ce = total_ce / num_tokens if num_tokens > 0 else 0

        candidates.append({
            "context": context,
            "response": response,
            "avg_ce": avg_ce,
            "num_tokens": num_tokens
        })

        scanned += 1
        if scanned % 50 == 0:
            print(f"  Scanned {scanned}/{num_candidates}...")

    # Sort by avg_ce (highest first)
    candidates.sort(key=lambda x: x["avg_ce"], reverse=True)

    # Cache all candidates (sorted by difficulty)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(candidates, f, indent=2)
    print(f"Cached {len(candidates)} examples to {cache_file}")

    # Return requested number
    hard_examples = candidates[:num_examples]

    print(f"\nSelected {len(hard_examples)} hardest examples:")
    for i, ex in enumerate(hard_examples):
        print(
            f"  {i+1}. avg_ce={ex['avg_ce']:.3f}, tokens={ex['num_tokens']}, context={ex['context'][:50]}..."
        )

    return hard_examples


def compute_response_cross_entropy(
        model, tokenizer, context: str, inserted_tokens: List[int],
        response: str, device: str) -> Tuple[float, List[float], List[float]]:
    """Compute cross-entropy of response given context and inserted tokens.

    Args:
        model: Language model
        tokenizer: Tokenizer
        context: Context string
        inserted_tokens: List of token IDs to insert between context and response
        response: Response string
        device: Device to run on

    Returns:
        Tuple of (total_ce, per_token_ce_list, per_token_prob_list)
    """
    # Tokenize context and response
    context_ids = tokenizer.encode(context, add_special_tokens=True)
    response_ids = tokenizer.encode(response, add_special_tokens=False)

    # Build full sequence: context + inserted + response
    full_ids = context_ids + inserted_tokens + response_ids
    input_ids = torch.tensor([full_ids], device=device)

    # Get model predictions
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  # [1, seq_len, vocab_size]

    # Compute CE for response tokens only
    # Response starts at position len(context_ids) + len(inserted_tokens)
    response_start = len(context_ids) + len(inserted_tokens)

    per_token_ce = []
    per_token_prob = []
    total_ce = 0.0

    for i, target_token in enumerate(response_ids):
        # Prediction for position response_start + i comes from logits at position response_start + i - 1
        pred_pos = response_start + i - 1
        if pred_pos < 0:
            continue

        logits_at_pos = logits[0, pred_pos, :]
        probs = F.softmax(logits_at_pos, dim=-1)
        log_probs = torch.log(probs + 1e-10)

        prob = probs[target_token].item()
        ce = -log_probs[target_token].item()

        per_token_ce.append(ce)
        per_token_prob.append(prob)
        total_ce += ce

    return total_ce, per_token_ce, per_token_prob


def compress_response(
    encoder: BlockEmissionArithmeticCoder,
    tokenizer,
    context: str,
    inserted_tokens: List[int],
    response: str,
) -> Dict:
    """Compress response using arithmetic coding and compute compression ratio.

    Args:
        encoder: Arithmetic coder
        tokenizer: Tokenizer
        context: Context string
        inserted_tokens: List of token IDs inserted between context and response
        response: Response string

    Returns:
        Dict with compression stats including the adjusted ratio
    """
    # Get response tokens
    response_ids = tokenizer.encode(response, add_special_tokens=False)

    # Build the context prefix (context + inserted tokens)
    context_with_insertion = context + tokenizer.decode(inserted_tokens)

    # Encode the response with the context as prefix
    encoded_buffer, encoding_info = encoder.encode(
        tokens=response_ids, initial_context=context_with_insertion)

    # Calculate compressed size
    encoded_bits = len(encoded_buffer) * encoder.bit_precision
    compressed_bytes = encoded_bits / 8

    # Original response size (2 bytes per token, standard convention)
    original_response_bytes = len(response_ids) * 2

    # Inserted tokens cost (2 bytes per token)
    inserted_bytes = len(inserted_tokens) * 2

    # Compression ratio: (compressed + inserted) / original
    # Lower is better (< 1 means compression)
    total_cost_bytes = compressed_bytes + inserted_bytes
    compression_ratio = total_cost_bytes / original_response_bytes if original_response_bytes > 0 else float(
        'inf')

    # Compression factor: original / (compressed + inserted)
    # Higher is better (> 1 means compression)
    compression_factor = original_response_bytes / total_cost_bytes if total_cost_bytes > 0 else 0

    # Also compute bits per token for reference
    bits_per_token = encoded_bits / len(response_ids) if response_ids else 0

    return {
        "compressed_bytes": compressed_bytes,
        "inserted_bytes": inserted_bytes,
        "total_cost_bytes": total_cost_bytes,
        "original_response_bytes": original_response_bytes,
        "compression_ratio": compression_ratio,
        "compression_factor": compression_factor,
        "bits_per_token": bits_per_token,
        "num_response_tokens": len(response_ids),
        "num_inserted_tokens": len(inserted_tokens),
        "num_encoded_blocks": len(encoded_buffer),
    }


def plot_token_probabilities(baseline_probs: List[float],
                             insertion_probs: List[float],
                             insertion_text: str,
                             insertion_length: int,
                             example_idx: int,
                             output_dir: Path,
                             tokenizer=None,
                             response_tokens: List[int] = None):
    """Plot token probabilities comparing baseline vs with insertion.

    Args:
        baseline_probs: Per-token probabilities without insertion
        insertion_probs: Per-token probabilities with insertion
        insertion_text: The inserted text (for title)
        insertion_length: L value
        example_idx: Example index (for filename)
        output_dir: Directory to save plot
        tokenizer: Optional tokenizer for x-axis labels
        response_tokens: Optional response token IDs for labels
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1])

    positions = np.arange(len(baseline_probs))
    baseline_arr = np.array(baseline_probs)
    insertion_arr = np.array(insertion_probs)

    # Top plot: both lines
    ax1.plot(positions,
             baseline_probs,
             'b-',
             label='Baseline (no insertion)',
             alpha=0.8,
             linewidth=1.5)
    ax1.plot(positions,
             insertion_probs,
             'r-',
             label=f'With insertion L={insertion_length}',
             alpha=0.8,
             linewidth=1.5)

    # Fill between to show improvement
    ax1.fill_between(positions,
                     baseline_arr,
                     insertion_arr,
                     where=(insertion_arr > baseline_arr),
                     color='green',
                     alpha=0.2,
                     label='Improvement')
    ax1.fill_between(positions,
                     baseline_arr,
                     insertion_arr,
                     where=(insertion_arr < baseline_arr),
                     color='red',
                     alpha=0.2,
                     label='Degradation')

    ax1.set_ylabel('P(actual token)')
    ax1.set_title(
        f'Example {example_idx + 1}: Token Probabilities\nInsertion: {repr(insertion_text[:50])}'
    )
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    # Add average probability annotations
    avg_baseline = np.mean(baseline_probs)
    avg_insertion = np.mean(insertion_probs)
    improvement = (avg_insertion - avg_baseline
                   ) / avg_baseline * 100 if avg_baseline > 0 else 0

    ax1.axhline(y=avg_baseline, color='blue', linestyle='--', alpha=0.5)
    ax1.axhline(y=avg_insertion, color='red', linestyle='--', alpha=0.5)
    ax1.text(len(positions) * 0.02,
             avg_baseline + 0.02,
             f'avg={avg_baseline:.3f}',
             color='blue',
             fontsize=9)
    ax1.text(len(positions) * 0.02,
             avg_insertion + 0.02,
             f'avg={avg_insertion:.3f} ({improvement:+.1f}%)',
             color='red',
             fontsize=9)

    # Bottom plot: delta (insertion - baseline)
    delta = insertion_arr - baseline_arr
    colors = ['green' if d >= 0 else 'red' for d in delta]
    ax2.bar(positions, delta, color=colors, alpha=0.7, width=1.0)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.axhline(y=np.mean(delta),
                color='purple',
                linestyle='--',
                alpha=0.7,
                label=f'avg delta={np.mean(delta):.4f}')
    ax2.set_xlabel('Token Position in Response')
    ax2.set_ylabel('ΔP (insertion - baseline)')
    ax2.set_title('Probability Change per Token')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = output_dir / f"prefix_L{insertion_length}_ex{example_idx + 1}_{timestamp}.png"
    plt.savefig(plot_file, dpi=150)
    plt.close()

    print(f"    Saved plot: {plot_file}")
    return plot_file


def plot_regular_insertion_probabilities(baseline_probs: List[float],
                                         insertion_probs: List[float],
                                         insertions: List[List[int]],
                                         chunks: List[List[int]],
                                         example_idx: int,
                                         output_dir: Path,
                                         tokenizer=None,
                                         chunk_by_sentence: bool = False):
    """Plot token probabilities for regular insertion experiment.

    Shows vertical lines at chunk boundaries and marks where insertions occurred.

    Args:
        insertions: List of token lists (empty list means no insertion for that chunk)
        chunks: List of token chunks (for computing variable boundaries)
        chunk_by_sentence: Whether chunks are sentences (for title)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), height_ratios=[2, 1])

    positions = np.arange(len(baseline_probs))
    baseline_arr = np.array(baseline_probs)
    insertion_arr = np.array(insertion_probs)

    # Top plot: both lines
    ax1.plot(positions,
             baseline_probs,
             'b-',
             label='Baseline (no insertion)',
             alpha=0.8,
             linewidth=1.5)
    ax1.plot(positions,
             insertion_probs,
             'r-',
             label='With insertions',
             alpha=0.8,
             linewidth=1.5)

    # Fill between
    ax1.fill_between(positions,
                     baseline_arr,
                     insertion_arr,
                     where=(insertion_arr > baseline_arr),
                     color='green',
                     alpha=0.2)
    ax1.fill_between(positions,
                     baseline_arr,
                     insertion_arr,
                     where=(insertion_arr < baseline_arr),
                     color='red',
                     alpha=0.2)

    # Compute chunk boundaries from actual chunks
    chunk_boundaries = [0]
    pos = 0
    for chunk in chunks:
        pos += len(chunk)
        chunk_boundaries.append(pos)

    # Add dashed gray vertical lines at every chunk boundary
    num_chunks = len(insertions)
    for boundary in chunk_boundaries:
        if boundary <= len(baseline_probs):
            ax1.axvline(x=boundary,
                        color='gray',
                        linestyle='--',
                        alpha=0.5,
                        linewidth=1)
            ax2.axvline(x=boundary,
                        color='gray',
                        linestyle='--',
                        alpha=0.5,
                        linewidth=1)

    # Mark where insertions occurred with green lines and labels
    for chunk_idx in range(num_chunks):
        chunk_start = chunk_boundaries[chunk_idx]
        if chunk_start < len(baseline_probs) and len(
                insertions[chunk_idx]) > 0:
            ax1.axvline(x=chunk_start,
                        color='green',
                        linestyle='-',
                        alpha=0.8,
                        linewidth=2)
            ax2.axvline(x=chunk_start,
                        color='green',
                        linestyle='-',
                        alpha=0.8,
                        linewidth=2)
            if tokenizer:
                insert_text = tokenizer.decode(insertions[chunk_idx])
                num_toks = len(insertions[chunk_idx])
                label = f"{repr(insert_text)[:12]}({num_toks})"
                ax1.annotate(label, (chunk_start + 0.5, 0.95),
                             fontsize=7,
                             rotation=45,
                             ha='left')

    ax1.set_ylabel('P(actual token)')

    total_tokens = sum(len(ins) for ins in insertions)
    num_chunks_with_ins = len([ins for ins in insertions if len(ins) > 0])
    chunk_type = "sentences" if chunk_by_sentence else "chunks"
    ax1.set_title(
        f'Example {example_idx + 1}: Insertion ({num_chunks} {chunk_type}, {total_tokens} tokens inserted in {num_chunks_with_ins})'
    )
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)

    # Averages
    avg_baseline = np.mean(baseline_probs)
    avg_insertion = np.mean(insertion_probs)
    improvement = (avg_insertion - avg_baseline
                   ) / avg_baseline * 100 if avg_baseline > 0 else 0

    ax1.axhline(y=avg_baseline, color='blue', linestyle='--', alpha=0.5)
    ax1.axhline(y=avg_insertion, color='red', linestyle='--', alpha=0.5)
    ax1.text(len(positions) * 0.02,
             avg_baseline + 0.02,
             f'avg={avg_baseline:.3f}',
             color='blue',
             fontsize=9)
    ax1.text(len(positions) * 0.02,
             avg_insertion + 0.02,
             f'avg={avg_insertion:.3f} ({improvement:+.1f}%)',
             color='red',
             fontsize=9)

    # Bottom plot: delta (insertion - baseline)
    delta = insertion_arr - baseline_arr
    colors = ['green' if d >= 0 else 'red' for d in delta]
    ax2.bar(positions, delta, color=colors, alpha=0.7, width=1.0)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.axhline(y=np.mean(delta),
                color='purple',
                linestyle='--',
                alpha=0.7,
                label=f'avg delta={np.mean(delta):.4f}')
    ax2.set_xlabel('Token Position in Response')
    ax2.set_ylabel('ΔP (insertion - baseline)')
    ax2.set_title('Probability Change per Token')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    chunk_type_str = "sentence" if chunk_by_sentence else f"chunk{len(chunks[0]) if chunks else 0}"
    plot_file = output_dir / f"insertion_{chunk_type_str}_ex{example_idx + 1}_{timestamp}.png"
    plt.savefig(plot_file, dpi=150)
    plt.close()

    print(f"    Saved plot: {plot_file}")
    return plot_file


def get_gradient_scores(model, prefix_ids: List[int], chunk_ids: List[int],
                        device: str) -> torch.Tensor:
    """Get gradient-based scores for all tokens at the insertion position.

    Uses first-order Taylor approximation: tokens whose embeddings align with
    the negative gradient will reduce loss the most.

    Args:
        model: Language model
        prefix_ids: Token IDs before insertion point
        chunk_ids: Token IDs after insertion point (the target to predict well)
        device: Device

    Returns:
        Tensor of shape [vocab_size] with scores (higher = better)
    """
    embedding_layer = model.get_input_embeddings()
    all_embeddings = embedding_layer.weight  # [vocab_size, embed_dim]
    vocab_size, embed_dim = all_embeddings.shape
    dtype = all_embeddings.dtype  # Match model's dtype (e.g., float16)

    # Create input: prefix + [learnable embedding] + chunk
    prefix_tensor = torch.tensor([prefix_ids], device=device)
    chunk_tensor = torch.tensor([chunk_ids], device=device)

    # Get embeddings for prefix and chunk (detach to avoid unnecessary gradients)
    with torch.no_grad():
        prefix_embeds = embedding_layer(
            prefix_tensor).clone()  # [1, prefix_len, embed_dim]
        chunk_embeds = embedding_layer(
            chunk_tensor).clone()  # [1, chunk_len, embed_dim]

    # Create a learnable embedding for the insertion position
    # Use same dtype as model for forward pass
    insert_embed = torch.zeros(1,
                               1,
                               embed_dim,
                               device=device,
                               dtype=dtype,
                               requires_grad=True)

    # Concatenate: [prefix, insert, chunk]
    full_embeds = torch.cat([prefix_embeds, insert_embed, chunk_embeds], dim=1)

    # Forward pass
    outputs = model(inputs_embeds=full_embeds)
    logits = outputs.logits.float(
    )  # [1, seq_len, vocab_size] - cast to float32 for stable loss

    # Compute loss on chunk tokens only
    # Predictions for chunk[i] come from position (prefix_len + 1 + i - 1) = (prefix_len + i)
    loss = torch.tensor(0.0, device=device)
    prefix_len = len(prefix_ids)
    for i, target_token in enumerate(chunk_ids):
        pred_pos = prefix_len + i  # Position that predicts chunk[i]
        log_probs = F.log_softmax(logits[0, pred_pos, :], dim=-1)
        loss = loss + (-log_probs[target_token])

    # Backward to get gradient w.r.t. insert_embed
    loss.backward()
    grad = insert_embed.grad[0, 0, :].float()  # [embed_dim] - cast to float32

    # Score all tokens: higher score = more loss reduction
    # score = -grad · embedding (tokens aligned with negative gradient reduce loss)
    with torch.no_grad():
        scores = -torch.matmul(all_embeddings.float(), grad)  # [vocab_size]

    return scores


def split_by_sentences(response_ids: List[int],
                       tokenizer,
                       min_chunk_size: int = 8) -> List[List[int]]:
    """Split response tokens into chunks at sentence boundaries (periods).

    Args:
        response_ids: List of token IDs
        tokenizer: Tokenizer for decoding
        min_chunk_size: Minimum tokens per chunk; shorter sentences are merged with next

    Returns:
        List of token chunks, split after periods
    """
    # First pass: split at sentence boundaries
    raw_chunks = []
    current_chunk = []

    for token_id in response_ids:
        current_chunk.append(token_id)
        # Decode just this token to check if it ends with a period
        token_text = tokenizer.decode([token_id])
        # Check if token ends a sentence (period, exclamation, question mark)
        if any(token_text.rstrip().endswith(p) for p in ['.', '!', '?']):
            if current_chunk:
                raw_chunks.append(current_chunk)
                current_chunk = []

    # Don't forget the last chunk if it doesn't end with punctuation
    if current_chunk:
        raw_chunks.append(current_chunk)

    # Second pass: merge short chunks with next chunk
    chunks = []
    pending_chunk = []

    for chunk in raw_chunks:
        pending_chunk.extend(chunk)
        # Only emit if we have enough tokens
        if len(pending_chunk) >= min_chunk_size:
            chunks.append(pending_chunk)
            pending_chunk = []

    # Don't forget any remaining tokens
    if pending_chunk:
        if chunks:
            # Merge with last chunk if we have one
            chunks[-1].extend(pending_chunk)
        else:
            chunks.append(pending_chunk)

    return chunks


def regular_insertion_search(
    model,
    tokenizer,
    context: str,
    response: str,
    chunk_size: int = 10,
    chunk_by_sentence: bool = False,
    max_insert_tokens: int = 2,
    top_k_candidates: Optional[int] = None,
    use_gradient_filter: bool = True,
    gradient_top_k: int = 512,
    min_ce_improvement: float = 4.0,
    device: str = "cuda"
) -> Tuple[List[List[int]], float, List[float], List[float], List[List[int]]]:
    """Find optimal token insertions (0, 1, or 2 tokens) at regular intervals.

    Splits response into chunks (by size or by sentence) and for each chunk decides
    whether to insert 0, 1, or 2 tokens before it to maximize probability of that chunk.

    Args:
        model: Language model
        tokenizer: Tokenizer
        context: Context string
        response: Response string
        chunk_size: Size of each chunk (T) - ignored if chunk_by_sentence=True
        chunk_by_sentence: If True, split at sentence boundaries instead of fixed size
        max_insert_tokens: Maximum tokens to insert per chunk (1 or 2)
        top_k_candidates: Number of top tokens to consider (None = use gradient filtering)
        use_gradient_filter: Use GCG-style gradient filtering (default: True)
        gradient_top_k: Number of top gradient-scoring tokens to evaluate (default: 512)
        min_ce_improvement: Minimum CE improvement required to keep an insertion (default: 4.0)
        device: Device to run on

    Returns:
        Tuple of:
            - List of inserted token lists (empty list means no insertion for that chunk)
            - Total CE with insertions
            - Per-token probabilities with insertions
            - Per-token probabilities baseline (no insertions)
            - List of chunks (for plotting with variable-size chunks)
    """
    # Tokenize
    context_ids = tokenizer.encode(context, add_special_tokens=True)
    response_ids = tokenizer.encode(response, add_special_tokens=False)

    # Split response into chunks
    if chunk_by_sentence:
        chunks = split_by_sentences(response_ids, tokenizer)
        print(f"    Split by sentence: {len(chunks)} sentences")
        for i, chunk in enumerate(chunks):
            chunk_text = tokenizer.decode(chunk)
            print(
                f"      Sentence {i+1}: {len(chunk)} tokens - {repr(chunk_text[:50])}..."
            )
    else:
        num_chunks = (len(response_ids) + chunk_size - 1) // chunk_size
        chunks = []
        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, len(response_ids))
            chunks.append(response_ids[start:end])

    vocab_size = model.config.vocab_size
    num_chunks = len(chunks)

    # Determine search strategy
    if top_k_candidates is not None:
        search_mode = "top_k_prob"
        print(f"    Search: top-{top_k_candidates} by probability")
    elif use_gradient_filter:
        search_mode = "gradient"
        print(
            f"    Search: gradient filtering (top-{gradient_top_k} by gradient score)"
        )
    else:
        search_mode = "exhaustive"
        print(f"    Search: exhaustive (all {vocab_size} tokens)")

    if chunk_by_sentence:
        avg_chunk_size = len(
            response_ids) / num_chunks if num_chunks > 0 else 0
        print(
            f"    Response has {len(response_ids)} tokens, split into {num_chunks} sentences (avg {avg_chunk_size:.1f} tokens)"
        )
    else:
        print(
            f"    Response has {len(response_ids)} tokens, split into {num_chunks} chunks of size {chunk_size}"
        )
    print(f"    Max tokens per insertion: {max_insert_tokens}")

    # First compute baseline (no insertions)
    baseline_ce, baseline_ce_per_token, baseline_probs = compute_response_cross_entropy(
        model, tokenizer, context, [], response, device)

    # Now optimize each chunk independently
    best_insertions = []  # List of token lists for each chunk

    # Build sequence incrementally
    current_prefix = context_ids.copy()

    def compute_chunk_ce(prefix: List[int], inserted: List[int],
                         chunk: List[int]) -> float:
        """Compute CE for a chunk given prefix and inserted tokens."""
        test_seq = prefix + inserted + chunk
        input_ids = torch.tensor([test_seq], device=device)

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits

        chunk_ce = 0.0
        for i, token in enumerate(chunk):
            pred_pos = len(prefix) + len(inserted) + i - 1
            if pred_pos >= 0:
                probs = F.softmax(logits[0, pred_pos, :], dim=-1)
                chunk_ce += -torch.log(probs[token] + 1e-10).item()
        return chunk_ce

    for chunk_idx, chunk in enumerate(chunks):
        print(f"    Optimizing chunk {chunk_idx + 1}/{num_chunks}...",
              end=" ",
              flush=True)

        # Option 0: No insertion
        best_insert_tokens = []
        best_insert_ce = compute_chunk_ce(current_prefix, [], chunk)

        # Get candidate tokens based on search mode
        if search_mode == "gradient":
            # Use gradient-based filtering
            grad_scores = get_gradient_scores(model, current_prefix, chunk,
                                              device)
            _, top_grad_indices = torch.topk(grad_scores, gradient_top_k)
            candidate_tokens = top_grad_indices.tolist()
        elif search_mode == "top_k_prob":
            # Use model's probability distribution
            prefix_tensor = torch.tensor([current_prefix], device=device)
            with torch.no_grad():
                outputs = model(prefix_tensor)
                next_logits = outputs.logits[0, -1, :]
                next_probs = F.softmax(next_logits, dim=-1)
            _, top_indices = torch.topk(next_probs, top_k_candidates)
            candidate_tokens = top_indices.tolist()
        else:
            # Exhaustive search
            candidate_tokens = list(range(vocab_size))

        # Option 1: Try inserting 1 token
        for candidate_token in candidate_tokens:
            chunk_ce = compute_chunk_ce(current_prefix, [candidate_token],
                                        chunk)
            if chunk_ce < best_insert_ce:
                best_insert_ce = chunk_ce
                best_insert_tokens = [candidate_token]

        # Option 2: Try inserting 2 tokens (if enabled)
        if max_insert_tokens >= 2 and len(best_insert_tokens) > 0:
            # Start from best 1-token insertion and try adding another
            first_token = best_insert_tokens[0]
            prefix_with_first = current_prefix + [first_token]

            # Get candidate tokens for second position
            if search_mode == "gradient":
                grad_scores_2 = get_gradient_scores(model, prefix_with_first,
                                                    chunk, device)
                _, top_grad_indices_2 = torch.topk(grad_scores_2,
                                                   gradient_top_k)
                candidate_tokens_2 = top_grad_indices_2.tolist()
            elif search_mode == "top_k_prob":
                prefix_tensor = torch.tensor([prefix_with_first],
                                             device=device)
                with torch.no_grad():
                    outputs = model(prefix_tensor)
                    next_logits = outputs.logits[0, -1, :]
                    next_probs = F.softmax(next_logits, dim=-1)
                _, top_indices2 = torch.topk(next_probs, top_k_candidates)
                candidate_tokens_2 = top_indices2.tolist()
            else:
                candidate_tokens_2 = list(range(vocab_size))

            for second_token in candidate_tokens_2:
                chunk_ce = compute_chunk_ce(current_prefix,
                                            [first_token, second_token], chunk)
                if chunk_ce < best_insert_ce:
                    best_insert_ce = chunk_ce
                    best_insert_tokens = [first_token, second_token]

        # Report result
        no_insert_ce = compute_chunk_ce(current_prefix, [], chunk)
        if len(best_insert_tokens) > 0:
            improvement = no_insert_ce - best_insert_ce
            # Only keep insertion if improvement exceeds threshold
            if improvement >= min_ce_improvement:
                insert_text = tokenizer.decode(best_insert_tokens)
                print(
                    f"insert {repr(insert_text)} ({len(best_insert_tokens)} tok, CE: {no_insert_ce:.2f} -> {best_insert_ce:.2f}, -{improvement:.2f})"
                )
            else:
                # Improvement below threshold, discard insertion
                print(
                    f"no insertion (CE: {no_insert_ce:.2f}, best improvement {improvement:.2f} < {min_ce_improvement} threshold)"
                )
                best_insert_tokens = []
                best_insert_ce = no_insert_ce
        else:
            print(f"no insertion (CE: {no_insert_ce:.2f})")

        best_insertions.append(best_insert_tokens)
        current_prefix = current_prefix + best_insert_tokens + chunk

    # Compute final CE and probs with all insertions
    total_inserted = sum(len(ins) for ins in best_insertions)

    # Build final sequence with insertions at proper positions
    final_response_with_insertions = []
    for chunk_idx, chunk in enumerate(chunks):
        final_response_with_insertions.extend(best_insertions[chunk_idx])
        final_response_with_insertions.extend(chunk)

    # Compute CE for this augmented response
    final_input = context_ids + final_response_with_insertions
    input_ids = torch.tensor([final_input], device=device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    # Extract probs for original response tokens
    final_probs = []
    final_ce = 0.0

    # Map original response positions to final sequence positions
    pos_in_final = len(context_ids)

    for chunk_idx, chunk in enumerate(chunks):
        pos_in_final += len(best_insertions[chunk_idx])  # Skip inserted tokens

        for i, token in enumerate(chunk):
            pred_pos = pos_in_final + i - 1
            if pred_pos >= 0 and pred_pos < logits.shape[1]:
                probs = F.softmax(logits[0, pred_pos, :], dim=-1)
                prob = probs[token].item()
                ce = -np.log(prob + 1e-10)
                final_probs.append(prob)
                final_ce += ce

        pos_in_final += len(chunk)

    num_chunks_with_insertion = len(
        [ins for ins in best_insertions if len(ins) > 0])
    print(
        f"    Total: {total_inserted} tokens inserted across {num_chunks_with_insertion} chunks, CE: {baseline_ce:.2f} -> {final_ce:.2f}"
    )

    return best_insertions, final_ce, final_probs, baseline_probs, chunks


def beam_search_insertion(model,
                          tokenizer,
                          context: str,
                          response: str,
                          insertion_length: int,
                          beam_width: int = 10,
                          top_k_expand: Optional[int] = None,
                          use_gradient_filter: bool = True,
                          gradient_top_k: int = 512,
                          device: str = "cuda") -> Tuple[List[int], float]:
    """Use beam search to find optimal token insertion of given length.

    Args:
        model: Language model
        tokenizer: Tokenizer
        context: Context string
        response: Response string
        insertion_length: Number of tokens to insert (L)
        beam_width: Number of candidates to keep at each step
        top_k_expand: Number of top tokens to consider (None = use gradient filtering)
        use_gradient_filter: Use GCG-style gradient filtering (default: True)
        gradient_top_k: Number of top gradient-scoring tokens to evaluate (default: 512)
        device: Device to run on

    Returns:
        Tuple of (best_insertion_tokens, best_ce_score)
    """
    if insertion_length == 0:
        ce, _, _ = compute_response_cross_entropy(model, tokenizer, context,
                                                  [], response, device)
        return [], ce

    # Initialize beam with empty sequence
    # We'll expand one token at a time
    beam = [BeamCandidate(tokens=[], ce_score=float('inf'))]

    # Get vocabulary size and response tokens
    vocab_size = model.config.vocab_size
    context_ids = tokenizer.encode(context, add_special_tokens=True)
    response_ids = tokenizer.encode(response, add_special_tokens=False)

    # Determine search mode
    if top_k_expand is not None:
        search_mode = "top_k_prob"
    elif use_gradient_filter:
        search_mode = "gradient"
    else:
        search_mode = "exhaustive"

    for step in range(insertion_length):
        all_candidates = []

        for candidate in beam:
            # For each candidate, try adding tokens
            prefix_ids = context_ids + candidate.tokens

            # Get candidate tokens based on search mode
            if search_mode == "gradient":
                grad_scores = get_gradient_scores(model, prefix_ids,
                                                  response_ids, device)
                _, top_grad_indices = torch.topk(grad_scores, gradient_top_k)
                tokens_to_try = top_grad_indices.tolist()
            elif search_mode == "top_k_prob":
                prefix_tensor = torch.tensor([prefix_ids], device=device)
                with torch.no_grad():
                    outputs = model(prefix_tensor)
                    next_token_logits = outputs.logits[0, -1, :]
                    next_token_probs = F.softmax(next_token_logits, dim=-1)
                _, top_indices = torch.topk(next_token_probs, top_k_expand)
                tokens_to_try = top_indices.tolist()
            else:
                tokens_to_try = list(range(vocab_size))

            for token_id in tokens_to_try:
                new_tokens = candidate.tokens + [token_id]

                # If this is the final step, compute the actual response CE
                if step == insertion_length - 1:
                    ce, _, _ = compute_response_cross_entropy(
                        model, tokenizer, context, new_tokens, response,
                        device)
                    all_candidates.append(
                        BeamCandidate(tokens=new_tokens, ce_score=ce))
                else:
                    # For intermediate steps, use a heuristic score
                    # We'll use the probability of the token as a proxy
                    # (lower is better for CE, but we want high prob tokens)
                    # Actually, let's just compute partial CE or use uniform score
                    # For simplicity, compute full CE even at intermediate steps
                    ce, _, _ = compute_response_cross_entropy(
                        model, tokenizer, context, new_tokens, response,
                        device)
                    all_candidates.append(
                        BeamCandidate(tokens=new_tokens, ce_score=ce))

        # Keep top beam_width candidates (lowest CE)
        all_candidates.sort(key=lambda x: x.ce_score)
        beam = all_candidates[:beam_width]

        if step < insertion_length - 1:
            print(
                f"    Step {step+1}/{insertion_length}: best CE so far = {beam[0].ce_score:.4f}"
            )

    return beam[0].tokens, beam[0].ce_score


def run_experiment(model_name: str = "meta-llama/Llama-3.1-8B",
                   num_examples: int = 10,
                   max_response_tokens: int = 50,
                   max_insertion_length: int = 10,
                   beam_width: int = 10,
                   top_k: Optional[int] = None,
                   use_gradient_filter: bool = True,
                   gradient_top_k: int = 512,
                   output_dir: Path = Path("results"),
                   device: str = None,
                   do_compression: bool = False,
                   regular_insertion: bool = False,
                   chunk_size: int = 10,
                   chunk_by_sentence: bool = False,
                   max_insert_tokens: int = 1,
                   pca_steering: bool = False,
                   steering_layer: Optional[int] = None,
                   num_pcs: int = 1,
                   hard_examples: bool = False,
                   num_candidates: int = 200,
                   min_response_tokens: int = 50):
    """Run the extra prompt selection experiment.

    Args:
        model_name: HuggingFace model name
        num_examples: Number of examples to process
        max_response_tokens: Maximum response length in tokens
        max_insertion_length: Maximum insertion length L to try (prefix mode)
        beam_width: Beam search width (prefix mode)
        top_k: Limit search to top-k probable tokens (None = use gradient filtering)
        use_gradient_filter: Use GCG-style gradient filtering (default: True)
        gradient_top_k: Number of top gradient-scoring tokens to evaluate (default: 512)
        output_dir: Directory for output files
        device: Device to use (auto-detect if None)
        do_compression: Whether to run arithmetic coding compression
        regular_insertion: Use regular insertion mode instead of prefix insertion
        chunk_size: Chunk size T for regular insertion mode
        chunk_by_sentence: Split by sentence boundaries instead of fixed size
        max_insert_tokens: Maximum tokens to insert per chunk (default: 1)
        pca_steering: Run PCA activation steering experiment
        steering_layer: Layer for PCA steering (default: middle)
        num_pcs: Number of principal components to test
        hard_examples: Find hardest examples instead of using first N
        num_candidates: Number of candidates to scan for hard examples
        min_response_tokens: Minimum response tokens (default: 50)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model and tokenizer
    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None)
    model.eval()

    # Create arithmetic coder for compression (only if needed)
    encoder = None
    if do_compression:
        print("Initializing arithmetic coder...")
        encoder = BlockEmissionArithmeticCoder(
            model=model,
            tokenizer=tokenizer,
            bit_precision=110,
            bits_for_encoding_count=8,
            device=device,
            verbose=False,
            use_fast=True,
        )

    # Load data
    if hard_examples:
        examples = load_hard_lmsys_examples(
            model=model,
            tokenizer=tokenizer,
            model_name=model_name,
            num_examples=num_examples,
            num_candidates=num_candidates,
            max_response_tokens=max_response_tokens,
            min_response_tokens=min_response_tokens,
            device=device)
    else:
        examples = load_lmsys_data(num_examples=num_examples,
                                   max_response_tokens=max_response_tokens,
                                   tokenizer=tokenizer)

    # Run experiment
    all_results = []

    print(f"\n{'='*80}")
    if pca_steering:
        print("PCA ACTIVATION STEERING EXPERIMENT")
    elif regular_insertion:
        print("REGULAR INSERTION EXPERIMENT")
    else:
        print("PREFIX INSERTION EXPERIMENT")
    print(f"{'='*80}")
    print(f"Model: {model_name}")
    print(f"Examples: {len(examples)}")
    print(f"Max response tokens: {max_response_tokens}")
    vocab_size = model.config.vocab_size
    # Determine search description
    if top_k is not None:
        search_desc = f"top-{top_k} by probability"
    elif use_gradient_filter:
        search_desc = f"gradient filtering (top-{gradient_top_k})"
    else:
        search_desc = f"exhaustive ({vocab_size} tokens)"

    if pca_steering:
        print(f"Mode: PCA activation steering")
        print(
            f"Steering layer: {steering_layer if steering_layer is not None else 'middle'}"
        )
        print(f"Num PCs: {num_pcs}")
    elif regular_insertion:
        if chunk_by_sentence:
            print(f"Mode: Regular insertion (by sentence)")
        else:
            print(f"Mode: Regular insertion (chunk_size={chunk_size})")
        print(f"Search: {search_desc}")
    else:
        print(f"Mode: Prefix insertion (L=1 to {max_insertion_length})")
        print(f"Beam width: {beam_width}")
        print(f"Search: {search_desc}")
    print(f"Compression: {'enabled' if do_compression else 'disabled'}")
    print(f"{'='*80}\n")

    for ex_idx, example in enumerate(examples):
        context = example["context"]
        response = example["response"]

        print(f"\n{'='*80}")
        print(f"EXAMPLE {ex_idx + 1}/{len(examples)}")
        print(f"{'='*80}")
        print(f"Context: {context[:100]}{'...' if len(context) > 100 else ''}")
        print(
            f"Response: {response[:100]}{'...' if len(response) > 100 else ''}"
        )

        # Compute baseline (L=0)
        baseline_ce, baseline_per_token_ce, baseline_per_token_prob = compute_response_cross_entropy(
            model, tokenizer, context, [], response, device)
        avg_baseline_ce = baseline_ce / len(
            baseline_per_token_ce) if baseline_per_token_ce else 0

        print(f"\nBaseline (L=0):")
        print(f"  Total CE: {baseline_ce:.4f}")
        print(f"  Avg CE per token: {avg_baseline_ce:.4f}")
        print(f"  Response tokens: {len(baseline_per_token_ce)}")
        print(f"  Avg token prob: {np.mean(baseline_per_token_prob):.4f}")

        result = {
            "example_idx": ex_idx,
            "context": context,
            "response": response,
            "baseline_ce": baseline_ce,
            "baseline_avg_ce": avg_baseline_ce,
            "baseline_avg_prob": float(np.mean(baseline_per_token_prob)),
            "num_response_tokens": len(baseline_per_token_ce),
        }

        if pca_steering:
            # PCA activation steering mode
            print(f"\n  Running PCA steering experiment...")
            steering_results, _, steerer = run_pca_steering_experiment(
                model=model,
                tokenizer=tokenizer,
                context=context,
                response=response,
                layer_idx=steering_layer,
                num_pcs=num_pcs,
                device=device)

            result["mode"] = "pca_steering"
            result["steering_layer"] = steering_results["layer_idx"]
            result["num_layers"] = steering_results["num_layers"]
            result["variance_explained"] = steering_results[
                "variance_explained"]
            result["steering_results"] = steering_results["steering_results"]

            # Find best result across all PCs
            best_ce = steering_results["baseline_ce"]
            best_alpha = 0.0
            best_pc = None
            for pc_name, pc_results in steering_results[
                    "steering_results"].items():
                for r in pc_results:
                    if r["ce"] < best_ce:
                        best_ce = r["ce"]
                        best_alpha = r["alpha"]
                        best_pc = pc_name

            ce_improvement = steering_results["baseline_ce"] - best_ce
            ce_pct_improvement = (
                ce_improvement / steering_results["baseline_ce"] *
                100) if steering_results["baseline_ce"] > 0 else 0

            result["best_pc"] = best_pc
            result["best_alpha"] = best_alpha
            result["best_ce"] = best_ce
            result["ce_improvement"] = ce_improvement
            result["ce_pct_improvement"] = ce_pct_improvement

            print(f"\n  Summary:")
            print(f"    Baseline CE: {steering_results['baseline_ce']:.4f}")
            if best_pc:
                print(
                    f"    Best: {best_pc} alpha={best_alpha:+.1f}, CE={best_ce:.4f} ({ce_pct_improvement:+.2f}%)"
                )
            else:
                print(f"    No improvement found from steering")

        elif regular_insertion:
            # Regular insertion mode: insert 0, 1, or 2 tokens before each chunk
            mode_desc = "by sentence" if chunk_by_sentence else f"chunk_size={chunk_size}"
            print(f"\n  Running regular insertion search ({mode_desc}):")
            insertions, final_ce, final_probs, _, chunks = regular_insertion_search(
                model,
                tokenizer,
                context,
                response,
                chunk_size=chunk_size,
                chunk_by_sentence=chunk_by_sentence,
                max_insert_tokens=max_insert_tokens,
                top_k_candidates=top_k,
                use_gradient_filter=use_gradient_filter,
                gradient_top_k=gradient_top_k,
                device=device)

            total_tokens_inserted = sum(len(ins) for ins in insertions)
            num_chunks_with_insertion = len(
                [ins for ins in insertions if len(ins) > 0])
            ce_improvement = baseline_ce - final_ce
            ce_pct_improvement = (ce_improvement / baseline_ce *
                                  100) if baseline_ce > 0 else 0

            result["mode"] = "regular_insertion"
            result["chunk_size"] = chunk_size
            result["insertions"] = insertions  # List of lists
            result["insertion_texts"] = [
                tokenizer.decode(ins) if len(ins) > 0 else ""
                for ins in insertions
            ]
            result["total_tokens_inserted"] = total_tokens_inserted
            result["num_chunks_with_insertion"] = num_chunks_with_insertion
            result["final_ce"] = final_ce
            result["ce_improvement"] = ce_improvement
            result["ce_pct_improvement"] = ce_pct_improvement
            result["final_avg_prob"] = float(
                np.mean(final_probs)) if final_probs else 0

            # Plot
            print(f"\n  Generating plot...")
            plot_dir = output_dir / "plots"
            plot_regular_insertion_probabilities(
                baseline_probs=baseline_per_token_prob,
                insertion_probs=final_probs,
                insertions=insertions,
                chunks=chunks,
                example_idx=ex_idx,
                output_dir=plot_dir,
                tokenizer=tokenizer,
                chunk_by_sentence=chunk_by_sentence)

            # Compression comparison (if enabled)
            if do_compression:
                print(f"\n  Computing compression...")

                # Compress original response (baseline)
                response_ids = tokenizer.encode(response,
                                                add_special_tokens=False)
                baseline_encoded, baseline_info = encoder.encode(
                    tokens=response_ids, initial_context=context)
                baseline_bits = len(baseline_encoded) * encoder.bit_precision
                baseline_bpt = baseline_bits / len(
                    response_ids) if response_ids else 0

                # Print baseline block info
                # encoding_info is ([], encoded_token_counts) where encoded_token_counts is tokens per block
                baseline_block_sizes = baseline_info[1] if isinstance(
                    baseline_info, tuple) and len(baseline_info) > 1 else []
                print(
                    f"    Baseline blocks: {len(baseline_encoded)} blocks, tokens per block: {baseline_block_sizes}"
                )

                # Build augmented sequence: [insertions interleaved with response chunks]
                augmented_tokens = []
                for chunk_idx, chunk in enumerate(chunks):
                    augmented_tokens.extend(
                        insertions[chunk_idx])  # Insert tokens (may be empty)
                    augmented_tokens.extend(chunk)

                # Compress augmented sequence
                augmented_encoded, augmented_info = encoder.encode(
                    tokens=augmented_tokens, initial_context=context)
                augmented_bits = len(augmented_encoded) * encoder.bit_precision
                augmented_bpt = augmented_bits / len(
                    augmented_tokens) if augmented_tokens else 0

                # Print augmented block info
                # encoding_info is ([], encoded_token_counts) tuple
                augmented_block_sizes = augmented_info[1] if isinstance(
                    augmented_info, tuple) and len(augmented_info) > 1 else []
                print(
                    f"    Augmented blocks: {len(augmented_encoded)} blocks, tokens per block: {augmented_block_sizes}"
                )

                # Also compute bits per original response token (for fair comparison)
                augmented_bpt_original = augmented_bits / len(
                    response_ids) if response_ids else 0

                print(
                    f"    Baseline: {baseline_bits:.1f} bits, {baseline_bpt:.3f} bits/token ({len(response_ids)} tokens)"
                )
                print(
                    f"    Augmented: {augmented_bits:.1f} bits, {augmented_bpt:.3f} bits/token ({len(augmented_tokens)} tokens)"
                )
                print(
                    f"    Augmented bits per original token: {augmented_bpt_original:.3f}"
                )

                bpt_change = augmented_bpt_original - baseline_bpt
                bpt_pct_change = (bpt_change / baseline_bpt *
                                  100) if baseline_bpt > 0 else 0
                print(
                    f"    Change: {bpt_change:+.3f} bits/token ({bpt_pct_change:+.2f}%)"
                )

                result["compression"] = {
                    "baseline_bits": baseline_bits,
                    "baseline_bpt": baseline_bpt,
                    "baseline_tokens": len(response_ids),
                    "augmented_bits": augmented_bits,
                    "augmented_bpt": augmented_bpt,
                    "augmented_tokens": len(augmented_tokens),
                    "augmented_bpt_original": augmented_bpt_original,
                    "bpt_change": bpt_change,
                    "bpt_pct_change": bpt_pct_change,
                }

            # Summary
            print(f"\n  Summary:")
            print(
                f"    Chunks: {len(insertions)}, Chunks with insertion: {num_chunks_with_insertion}, Total tokens inserted: {total_tokens_inserted}"
            )
            print(
                f"    CE: {baseline_ce:.4f} -> {final_ce:.4f} ({ce_pct_improvement:+.2f}%)"
            )
            print(
                f"    Avg prob: {np.mean(baseline_per_token_prob):.4f} -> {np.mean(final_probs):.4f}"
            )

        else:
            # Prefix insertion mode: try L=1 to max_insertion_length
            result["mode"] = "prefix_insertion"
            result["insertions"] = {}

            # Try different insertion lengths (beam search only, no compression yet)
            print(f"\nSearching for optimal insertions:")
            best_insertions = {}
            for L in range(1, max_insertion_length + 1):
                print(f"\n  L={L}:")
                best_tokens, best_ce = beam_search_insertion(
                    model,
                    tokenizer,
                    context,
                    response,
                    insertion_length=L,
                    beam_width=beam_width,
                    top_k_expand=top_k,
                    use_gradient_filter=use_gradient_filter,
                    gradient_top_k=gradient_top_k,
                    device=device)

                improvement = baseline_ce - best_ce
                pct_improvement = (improvement / baseline_ce *
                                   100) if baseline_ce > 0 else 0

                # Decode inserted tokens
                inserted_text = tokenizer.decode(best_tokens)

                print(f"    Best insertion: {repr(inserted_text)}")
                print(f"    Token IDs: {best_tokens}")
                print(
                    f"    CE: {best_ce:.4f} (baseline: {baseline_ce:.4f}, improvement: {pct_improvement:.2f}%)"
                )

                best_insertions[L] = {
                    "tokens": best_tokens,
                    "text": inserted_text,
                    "ce": best_ce,
                    "ce_improvement": improvement,
                    "ce_pct_improvement": pct_improvement,
                }

            # Generate probability plots for the best insertion at each L
            print(f"\n  Generating probability plots...")
            plot_dir = output_dir / "plots"
            for L in range(1, max_insertion_length + 1):
                ins = best_insertions[L]
                # Recompute to get probabilities for the best insertion
                _, _, insertion_probs = compute_response_cross_entropy(
                    model, tokenizer, context, ins["tokens"], response, device)
                # Store avg prob (not full list - too large for JSON)
                best_insertions[L]["avg_prob"] = float(
                    np.mean(insertion_probs))

                # Plot comparison
                plot_token_probabilities(
                    baseline_probs=baseline_per_token_prob,
                    insertion_probs=insertion_probs,
                    insertion_text=ins["text"],
                    insertion_length=L,
                    example_idx=ex_idx,
                    output_dir=plot_dir)

        # Compression (only if --compression flag is set and not regular insertion mode)
        baseline_compression = None
        if do_compression and not regular_insertion:
            print(f"\n  Compressing baseline (L=0)...", end=" ", flush=True)
            baseline_compression = compress_response(encoder, tokenizer,
                                                     context, [], response)
            baseline_bits = baseline_compression['compressed_bytes'] * 8
            print(
                f"done! ratio={baseline_compression['compression_ratio']:.4f}, factor={baseline_compression['compression_factor']:.2f}x, bits={baseline_bits:.1f}"
            )

            result["baseline_compression"] = baseline_compression

            for L in range(1, max_insertion_length + 1):
                ins = best_insertions[L]
                print(f"  Compressing L={L} ({repr(ins['text'][:30])})...",
                      end=" ",
                      flush=True)

                insertion_compression = compress_response(
                    encoder, tokenizer, context, ins["tokens"], response)
                compressed_bits = insertion_compression['compressed_bytes'] * 8
                inserted_bits = insertion_compression['inserted_bytes'] * 8
                total_bits = compressed_bits + inserted_bits

                # Compare compression ratios
                baseline_ratio = baseline_compression['compression_ratio']
                baseline_factor = baseline_compression['compression_factor']
                new_ratio = insertion_compression['compression_ratio']
                new_factor = insertion_compression['compression_factor']
                ratio_improvement = baseline_ratio - new_ratio
                ratio_pct_improvement = (ratio_improvement / baseline_ratio *
                                         100) if baseline_ratio > 0 else 0
                factor_improvement = new_factor - baseline_factor
                factor_pct_improvement = (factor_improvement /
                                          baseline_factor *
                                          100) if baseline_factor > 0 else 0

                num_blocks = insertion_compression['num_encoded_blocks']
                tokens_per_block = insertion_compression[
                    'num_response_tokens'] / num_blocks if num_blocks > 0 else 0
                print(
                    f"done! ratio={new_ratio:.4f} ({ratio_pct_improvement:+.1f}%), factor={new_factor:.2f}x ({factor_pct_improvement:+.1f}%), bits={total_bits:.1f} (resp={compressed_bits:.1f} + ins={inserted_bits:.1f}), blocks={num_blocks} ({tokens_per_block:.1f} tok/block)"
                )

                result["insertions"][L] = {
                    **ins,
                    "compression": insertion_compression,
                    "compression_ratio_improvement": ratio_improvement,
                    "compression_ratio_pct_improvement": ratio_pct_improvement,
                    "compression_factor_improvement": factor_improvement,
                    "compression_factor_pct_improvement":
                    factor_pct_improvement,
                }
        elif not regular_insertion:
            # Without compression, just store the CE results
            for L in range(1, max_insertion_length + 1):
                result["insertions"][L] = best_insertions[L]

        # Summary for this example (prefix mode only)
        if not regular_insertion:
            print(f"\n  Summary for Example {ex_idx + 1}:")
            if do_compression:
                print(
                    f"  {'L':>3} | {'CE':>10} | {'CE Impr':>10} | {'Factor':>8} | {'Fact Impr':>10} | {'Blocks':>6} | {'Inserted Text'}"
                )
                print(
                    f"  {'-'*3}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*10}-+-{'-'*6}-+-{'-'*20}"
                )
                print(
                    f"  {'0':>3} | {baseline_ce:>10.4f} | {'baseline':>10} | {baseline_compression['compression_factor']:>7.2f}x | {'baseline':>10} | {baseline_compression['num_encoded_blocks']:>6} | "
                )
                for L in range(1, max_insertion_length + 1):
                    ins = result["insertions"][L]
                    print(
                        f"  {L:>3} | {ins['ce']:>10.4f} | {ins['ce_pct_improvement']:>9.2f}% | {ins['compression']['compression_factor']:>7.2f}x | {ins['compression_factor_pct_improvement']:>9.2f}% | {ins['compression']['num_encoded_blocks']:>6} | {repr(ins['text'][:20])}"
                    )
            else:
                print(
                    f"  {'L':>3} | {'CE':>10} | {'CE Impr':>10} | {'Inserted Text'}"
                )
                print(f"  {'-'*3}-+-{'-'*10}-+-{'-'*10}-+-{'-'*20}")
                print(
                    f"  {'0':>3} | {baseline_ce:>10.4f} | {'baseline':>10} | ")
                for L in range(1, max_insertion_length + 1):
                    ins = result["insertions"][L]
                    print(
                        f"  {L:>3} | {ins['ce']:>10.4f} | {ins['ce_pct_improvement']:>9.2f}% | {repr(ins['text'][:20])}"
                    )

        all_results.append(result)

    # Overall summary
    print(f"\n\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")

    if pca_steering:
        # PCA steering summary
        avg_ce_improvement = np.mean(
            [r.get("ce_pct_improvement", 0) for r in all_results])
        num_improved = sum(1 for r in all_results
                           if r.get("best_pc") is not None)

        print(f"\nPCA Steering Results:")
        print(
            f"  Examples with improvement: {num_improved}/{len(all_results)}")
        print(f"  Average CE improvement: {avg_ce_improvement:.2f}%")

        # Show per-example summary
        for r in all_results:
            ex_idx = r["example_idx"]
            if r.get("best_pc"):
                print(
                    f"  Ex {ex_idx+1}: {r['best_pc']} alpha={r['best_alpha']:+.1f}, CE improvement={r['ce_pct_improvement']:+.2f}%"
                )
            else:
                print(f"  Ex {ex_idx+1}: No improvement")

    elif regular_insertion:
        # Regular insertion summary
        avg_ce_improvement = np.mean(
            [r["ce_pct_improvement"] for r in all_results])
        avg_total_tokens_inserted = np.mean(
            [r["total_tokens_inserted"] for r in all_results])
        avg_chunks_with_insertion = np.mean(
            [r["num_chunks_with_insertion"] for r in all_results])
        avg_baseline_prob = np.mean(
            [r["baseline_avg_prob"] for r in all_results])
        avg_final_prob = np.mean([r["final_avg_prob"] for r in all_results])

        print(f"\nRegular Insertion Results (chunk_size={chunk_size}):")
        print(
            f"  Average tokens inserted per example: {avg_total_tokens_inserted:.1f}"
        )
        print(
            f"  Average chunks with insertion: {avg_chunks_with_insertion:.1f}"
        )
        print(f"  Average CE improvement: {avg_ce_improvement:.2f}%")
        print(
            f"  Average prob: {avg_baseline_prob:.4f} -> {avg_final_prob:.4f}")

    else:
        # Prefix insertion summary
        avg_ce_improvements = {}
        avg_factor_improvements = {}
        for L in range(1, max_insertion_length + 1):
            ce_improvements = [
                r["insertions"][L]["ce_pct_improvement"] for r in all_results
            ]
            avg_ce_improvements[L] = sum(ce_improvements) / len(
                ce_improvements)
            if do_compression and not regular_insertion:
                factor_improvements = [
                    r["insertions"][L]["compression_factor_pct_improvement"]
                    for r in all_results
                ]
                avg_factor_improvements[L] = sum(factor_improvements) / len(
                    factor_improvements)

        print(f"\nAverage % CE improvement by insertion length:")
        for L, avg_imp in avg_ce_improvements.items():
            bar = "#" * max(0, int(avg_imp / 2))
            print(f"  L={L:>2}: {avg_imp:>7.2f}% {bar}")

        best_L = max(avg_ce_improvements.keys(),
                     key=lambda k: avg_ce_improvements[k])
        print(
            f"\nBest insertion length for CE: L={best_L} ({avg_ce_improvements[best_L]:.2f}% avg CE improvement)"
        )

        if do_compression and not regular_insertion:
            print(
                f"\nAverage % compression factor improvement by insertion length:"
            )
            print(
                f"  (Positive = better compression, accounting for inserted token cost)"
            )
            for L, avg_imp in avg_factor_improvements.items():
                if avg_imp >= 0:
                    bar = "+" * max(0, int(avg_imp / 2))
                else:
                    bar = "-" * max(0, int(-avg_imp / 2))
                print(f"  L={L:>2}: {avg_imp:>7.2f}% {bar}")

            # Find best L for compression
            best_L = max(avg_factor_improvements.keys(),
                         key=lambda k: avg_factor_improvements[k])
            print(
                f"\nBest insertion length for compression: L={best_L} ({avg_factor_improvements[best_L]:.2f}% avg factor improvement)"
            )

    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = model_name.split("/")[-1]
    output_file = output_dir / f"extra_prompt_{model_short}_{timestamp}.json"

    output_data = {
        "config": {
            "model": model_name,
            "num_examples": num_examples,
            "max_response_tokens": max_response_tokens,
            "top_k": top_k,
            "use_gradient_filter": use_gradient_filter,
            "gradient_top_k": gradient_top_k,
            "do_compression": do_compression,
            "regular_insertion": regular_insertion,
        },
        "results": all_results
    }

    if pca_steering:
        output_data["config"]["steering_layer"] = steering_layer
        output_data["config"]["num_pcs"] = num_pcs
        output_data["avg_ce_improvement"] = float(avg_ce_improvement)
        output_data["num_improved"] = num_improved
    elif regular_insertion:
        output_data["config"]["chunk_size"] = chunk_size
        output_data["config"]["chunk_by_sentence"] = chunk_by_sentence
        output_data["avg_ce_improvement"] = float(avg_ce_improvement)
        output_data["avg_total_tokens_inserted"] = float(
            avg_total_tokens_inserted)
        output_data["avg_chunks_with_insertion"] = float(
            avg_chunks_with_insertion)
    else:
        output_data["config"]["max_insertion_length"] = max_insertion_length
        output_data["config"]["beam_width"] = beam_width
        output_data["avg_ce_improvements_by_length"] = avg_ce_improvements
        output_data["best_ce_insertion_length"] = best_L
        if do_compression:
            output_data[
                "avg_compression_factor_improvements_by_length"] = avg_factor_improvements
            output_data["best_compression_insertion_length"] = best_L

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Find optimal token insertions between context and response"
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-8B",
        help="HuggingFace model name (default: meta-llama/Llama-3.1-8B)")
    parser.add_argument("--num-examples",
                        type=int,
                        default=10,
                        help="Number of examples to process (default: 10)")
    parser.add_argument("--max-response-tokens",
                        type=int,
                        default=50,
                        help="Maximum response length in tokens (default: 50)")
    parser.add_argument("--max-insertion-length",
                        type=int,
                        default=10,
                        help="Maximum insertion length L to try (default: 10)")
    parser.add_argument("--beam-width",
                        type=int,
                        default=10,
                        help="Beam search width (default: 10)")
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help=
        "Limit search to top-k probable tokens (default: None = use gradient filtering)"
    )
    parser.add_argument(
        "--no-gradient-filter",
        action="store_true",
        help=
        "Disable gradient filtering (use exhaustive search if --top-k not set)"
    )
    parser.add_argument(
        "--gradient-top-k",
        type=int,
        default=512,
        help="Number of top gradient-scoring tokens to evaluate (default: 512)"
    )
    parser.add_argument("--output-dir",
                        type=Path,
                        default=Path("results"),
                        help="Output directory (default: results)")
    parser.add_argument("--device",
                        default=None,
                        help="Device to use (default: auto-detect)")
    parser.add_argument(
        "--compression",
        action="store_true",
        help=
        "Run arithmetic coding compression (slower, but measures actual compression)"
    )
    parser.add_argument(
        "--regular-insertion",
        action="store_true",
        help=
        "Use regular insertion mode (insert 0-1 token every T tokens) instead of prefix insertion"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10,
        help="Chunk size T for regular insertion mode (default: 10)")
    parser.add_argument(
        "--chunk-by-sentence",
        action="store_true",
        help="Split by sentence boundaries instead of fixed chunk size")
    parser.add_argument("--max-insert-tokens",
                        type=int,
                        default=1,
                        help="Maximum tokens to insert per chunk (default: 1)")
    parser.add_argument(
        "--min-ce-improvement",
        type=float,
        default=4.0,
        help=
        "Minimum CE improvement required to keep an insertion (default: 4.0)")
    parser.add_argument(
        "--pca-steering",
        action="store_true",
        help="Run PCA activation steering experiment instead of token insertion"
    )
    parser.add_argument(
        "--steering-layer",
        type=int,
        default=None,
        help="Layer to use for PCA steering (default: middle layer)")
    parser.add_argument(
        "--num-pcs",
        type=int,
        default=1,
        help="Number of principal components to test (default: 1)")
    parser.add_argument(
        "--hard-examples",
        action="store_true",
        help="Find hardest examples (highest CE) instead of using first N")
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=200,
        help=
        "Number of candidates to scan when using --hard-examples (default: 200)"
    )
    parser.add_argument("--min-response-tokens",
                        type=int,
                        default=50,
                        help="Minimum response tokens (default: 50)")

    args = parser.parse_args()

    run_experiment(model_name=args.model,
                   num_examples=args.num_examples,
                   max_response_tokens=args.max_response_tokens,
                   max_insertion_length=args.max_insertion_length,
                   beam_width=args.beam_width,
                   top_k=args.top_k,
                   use_gradient_filter=not args.no_gradient_filter,
                   gradient_top_k=args.gradient_top_k,
                   output_dir=args.output_dir,
                   device=args.device,
                   do_compression=args.compression,
                   regular_insertion=args.regular_insertion,
                   chunk_size=args.chunk_size,
                   chunk_by_sentence=args.chunk_by_sentence,
                   max_insert_tokens=args.max_insert_tokens,
                   pca_steering=args.pca_steering,
                   steering_layer=args.steering_layer,
                   num_pcs=args.num_pcs,
                   hard_examples=args.hard_examples,
                   num_candidates=args.num_candidates,
                   min_response_tokens=args.min_response_tokens)


if __name__ == "__main__":
    main()
