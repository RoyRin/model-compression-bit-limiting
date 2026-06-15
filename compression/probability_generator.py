#!/usr/bin/env python3
"""
Model Probability Generator for vLLM-style inference.

Provides both batch (prefill) and incremental (teacher-forcing) modes
for extracting probability distributions from language models.
"""

import os
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

# Uncomment to enable deterministic algorithms (slower):
# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import torch
import torch.nn.functional as F

# ----------------------------
# Utility / config helpers
# ----------------------------


@contextmanager
def sdp_backend(which: Optional[str] = None):
    """
    Context manager to force a specific SDPA backend.

    Args:
        which: "flash" | "mem" | "math" | None (use default)

    Usage:
        with sdp_backend("flash"):
            outputs = model(...)
    """
    if which is None or not torch.cuda.is_available():
        # No backend specified or CPU mode - use default
        yield
        return

    enable_flash = (which == "flash")
    enable_mem = (which == "mem")
    enable_math = (which == "math")

    with torch.backends.cuda.sdp_kernel(
            enable_flash=enable_flash,
            enable_mem_efficient=enable_mem,
            enable_math=enable_math,
    ):
        yield


def set_repro(seed: Optional[int] = 0, deterministic: bool = True):
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def force_eager_attention(model):
    # Not all models expose this field; make best effort.
    try:
        model.config.attn_implementation = "eager"
    except Exception:
        pass


def device_of(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


# Module-level debug flag
DEBUG = False


def set_debug(enabled: bool = True):
    """Enable or disable debug output for probability generator."""
    global DEBUG
    DEBUG = enabled


import math
from decimal import Decimal
from typing import List, Tuple

import torch


def _bits_to_int(bits: List[int]) -> int:
    x = 0
    for b in bits:
        x = (x << 1) | int(b)
    return x


def _int_to_bits(x: int, num_bits: int) -> List[int]:
    return [int(b) for b in bin(x)[2:].zfill(num_bits)]


def _prefix_interval(prefix: List[int], num_bits: int,
                     vocab_size: int) -> Tuple[int, int]:
    """
    Half-open interval [start, end) of token IDs whose num_bits-wide binary starts with prefix.
    Assumes token_id order corresponds to standard binary counting.
    """
    if len(prefix) > num_bits:
        return (0, 0)

    p = _bits_to_int(prefix)
    shift = num_bits - len(prefix)
    start = p << shift
    end = (p + 1) << shift

    # clamp to [0, vocab_size)
    start = max(0, min(start, vocab_size))
    end = max(0, min(end, vocab_size))
    return start, end


def _range_sum(prefix_cumsum: torch.Tensor, start: int, end: int) -> float:
    """Sum token_probs[start:end] using inclusive cumsum."""
    if end <= start:
        return 0.0
    if start == 0:
        return float(prefix_cumsum[end - 1].item())
    return float((prefix_cumsum[end - 1] - prefix_cumsum[start - 1]).item())


def _get_p0_p1_from_prefix(
    token_probs: torch.Tensor,
    prefix_cumsum: torch.Tensor,
    token_prefix: List[int],
    num_bits: int,
) -> Tuple[float, float]:
    """
    Returns (p0, p1) where:
      p0 = sum_{ids with bits prefix+[0]+*} token_probs[id]
      p1 = sum_{ids with bits prefix+[1]+*} token_probs[id]
    Uses contiguous intervals + cumsum => O(1) per query after building cumsum.
    """
    vocab_size = int(token_probs.shape[0])
    s0, e0 = _prefix_interval(token_prefix + [0], num_bits, vocab_size)
    s1, e1 = _prefix_interval(token_prefix + [1], num_bits, vocab_size)
    p0 = _range_sum(prefix_cumsum, s0, e0)
    p1 = _range_sum(prefix_cumsum, s1, e1)
    return p0, p1


def get_token_probabilities_prefill(
    model,
    tokens: List[int],
    temperature: float = 1.0,
    use_cache: bool = False,
    keep_on_device: bool = False,
    sdpa_backend: Optional[str] = None,
    return_past_key_values: bool = False
) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], Any]]:
    """
    Runs a single forward pass over the full token sequence and returns
    probability distributions per position:
      result[i] corresponds to predicting tokens[i] given tokens[:i]

    Args:
        model: Language model
        tokens: List of token IDs
        temperature: Temperature for softmax scaling (default: 1.0)
        use_cache: Enable KV caching (faster but uses more memory)
        keep_on_device: Keep tensors on GPU instead of moving to CPU (faster but uses more GPU memory)
        sdpa_backend: Force specific SDPA backend ("flash", "mem", "math", or None for default)
        return_past_key_values: If True, return tuple of (distributions, past_key_values)

    Returns:
        List of probability distributions (one per position), or if return_past_key_values=True,
        a tuple of (distributions, past_key_values)
    """
    dev = device_of(model)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=dev)
    attn = torch.ones_like(input_ids)

    if DEBUG:
        print(f"[GET_TOKEN_PROBABILITIES_PREFILL DEBUG]")
        print(f"  Input length: {len(tokens)}")
        print(f"  First 5 tokens: {tokens[:5]}")
        print(f"  Last 5 tokens: {tokens[-5:]}")
        print(f"  Model training mode: {model.training}")
        print(f"  use_cache: {use_cache}")
        print(f"  sdpa_backend: {sdpa_backend}")
        print(f"  Model dtype: {next(model.parameters()).dtype}")

    with torch.inference_mode(), sdp_backend(sdpa_backend):
        outputs = model(input_ids=input_ids,
                        attention_mask=attn,
                        use_cache=use_cache)
        logits = outputs.logits  # [1, seq_len, vocab]
        past_key_values = outputs.past_key_values if use_cache else None
        if DEBUG:
            print(f"  Output logits shape: {logits.shape}")
            print(f"  Will return {logits.shape[1]} distributions")

            # Debug: Check raw logits at position 510
            if logits.shape[1] > 510:
                logits_510 = logits[0, 510, :].float()
                top5_vals, top5_ids = logits_510.topk(5)
                print(
                    f"  RAW logits[510] before softmax - top-5 tokens: {top5_ids.tolist()}"
                )
                print(
                    f"  RAW logits[510] before softmax - top-5 values: {[f'{v:.4f}' for v in top5_vals.tolist()]}"
                )

    logits = logits.squeeze(0).float()  # [seq_len, vocab]

    # Apply temperature and softmax PER-TOKEN to exactly match teacher forcing
    # This ensures identical numerical behavior
    result_list = []
    for i in range(logits.shape[0]):
        token_logits = logits[i]  # [vocab]

        # Apply temperature
        if temperature != 1.0 and temperature > 0:
            token_logits = token_logits / temperature

        # Per-token softmax (matches teacher forcing exactly)
        token_probs = torch.softmax(token_logits, dim=-1).detach()
        result_list.append(token_probs)

    # Single CPU transfer at the end if needed
    if not keep_on_device:
        result_list = [t.cpu() for t in result_list]

    if return_past_key_values:
        return result_list, past_key_values
    return result_list


# ----------------------------
# Core routines
# ----------------------------


class ModelProbabilityGenerator:
    """
    Unified probability generator with two modes:

    Prefill mode (batch):
      - prefill(tokens): precompute all distributions
      - Loop: get_token_probability() + add_token_prefill()

    Teacher-forcing mode (incremental):
      - reset_teacher_forcing([BOS])
      - Loop: add_token(token) + get_token_probability()

    Both modes use get_token_probability() to read (idempotent).
    """

    def __init__(self,
                 model,
                 tokenizer=None,
                 device=None,
                 temperature: float = 1.0,
                 use_cache: bool = True,
                 keep_on_device: bool = True,
                 sdpa_backend: Optional[str] = None,
                 debug: bool = False,
                 **kwargs):
        """
        Initialize the probability generator.

        Args:
            model: HuggingFace model to use for generation
            tokenizer: Optional tokenizer (for compatibility)
            device: Optional device (for compatibility, ignored)
            temperature: Softmax temperature for probability distributions
            use_cache: Enable KV caching for faster computation
            keep_on_device: Keep tensors on GPU
            sdpa_backend: Force specific SDPA backend ("flash", "mem", "math", or None)
            debug: Enable debug output (default: False)
        """
        self.model = model
        self.tokenizer = tokenizer  # Store for compatibility
        self.temperature = temperature
        self.use_cache = use_cache
        self.keep_on_device = keep_on_device
        self.sdpa_backend = sdpa_backend
        self.debug = debug

        # Prefill storage (unchanged)
        self.distributions: Optional[List[torch.Tensor]] = None
        self.current_index: int = 0

        # Teacher-forcing state
        self.context_tokens: List[int] = []
        self.past_key_values = None
        self.staged_token: Optional[
            int] = None  # token to feed on next forward
        self._last_probs: Optional[torch.Tensor] = None  # idempotent buffer

    # ---------- Prefill path ----------
    def prefill(self, tokens: List[int]) -> None:
        """
        Precompute probability distributions for an entire sequence of tokens.
        Sets current_index to 0 and updates _last_probs with the first distribution.

        Args:
            tokens: List of token IDs to prefill
        """
        if self.debug:
            print(f"\n[ENCODER PREFILL CONFIG]")
            print(f"  use_cache: {self.use_cache}")
            print(f"  sdpa_backend: {self.sdpa_backend}")
            print(f"  keep_on_device: {self.keep_on_device}")
            print(f"  temperature: {self.temperature}")
        self.distributions = get_token_probabilities_prefill(
            self.model,
            tokens,
            temperature=self.temperature,
            use_cache=self.use_cache,
            keep_on_device=self.keep_on_device,
            sdpa_backend=self.sdpa_backend,
        )
        self.current_index = 0
        # Initialize _last_probs with the first distribution
        if self.distributions:
            self._last_probs = self.distributions[0]

    def add_token_prefill(self) -> None:
        """
        Increment the current index by 1 and update _last_probs with the distribution
        at the new index (prefill mode).
        """
        if self.distributions is None:
            raise RuntimeError("prefill() first")
        self.current_index += 1
        # Update _last_probs with the distribution at the new index
        if self.current_index < len(self.distributions):
            self._last_probs = self.distributions[
                self.current_index]  # TODO - it may be an off-by-one error

    # ---------- Teacher-forcing API (one forward per token) ----------
    def reset_teacher_forcing(self,
                              initial_tokens: Optional[List[int]] = None
                              ) -> None:
        """
        Call once before the loop. Processes context tokens through the model
        and caches the KV values for efficient incremental generation.

        Args:
            initial_tokens: Optional list of initial context tokens (e.g., [BOS])
        """
        self.context_tokens = initial_tokens.copy() if initial_tokens else []
        self._last_probs = None

        if not self.context_tokens:
            self.staged_token = None
            self.past_key_values = None
            return

        # If we have context tokens, prefill them to populate the KV cache
        if len(self.context_tokens) > 1:
            # Use get_token_probabilities_prefill to process context and get KV cache
            if self.debug:
                print(f"\n[DECODER RESET_TEACHER_FORCING - PREFILL CONFIG]")
                print(f"  use_cache: True (for KV cache)")
                print(f"  sdpa_backend: {self.sdpa_backend}")
                print(f"  keep_on_device: {self.keep_on_device}")
                print(f"  temperature: {self.temperature}")
                print(
                    f"decode context tokens - first 5: {self.context_tokens[:5]}"
                )
                print(
                    f"decode context tokens - last 5: {self.context_tokens[-5:]}"
                )
                print(
                    f"decode context tokens - length: {len(self.context_tokens)}"
                )
            # We also get the probability distributions - save the last one
            distributions, self.past_key_values = get_token_probabilities_prefill(
                self.model,
                self.context_tokens,
                temperature=self.temperature,
                use_cache=True,
                keep_on_device=self.keep_on_device,
                sdpa_backend=self.sdpa_backend,
                return_past_key_values=True)
            if self.debug:
                # just for debugging: print distributions [-2] - top 5 tokens, their probs
                top5_vals_second_last, top5_ids_second_last = distributions[
                    -2].topk(5)
                print(
                    f"decode context tokens - distribution [-2] - top 5 tokens: {top5_ids_second_last.tolist()}"
                )
                print(
                    f"decode context tokens - distribution [-2] - top 5 probs: {[f'{p:.6f}' for p in top5_vals_second_last.tolist()]}"
                )

            # Store decoder prefill distributions for debugging/comparison only
            self.decoder_prefill_distributions = [
                d.cpu().clone() for d in distributions
            ]

            # Use the last prefill distribution directly as P(next | full context)
            # HF semantics: distributions[-1] (at position len(context)-1) is P(next | full_context)
            self._last_probs = distributions[-1]

            # No staged token needed for this initial distribution
            self.staged_token = None

            if self.debug:
                # Debug logging
                print(f"[RESET_TEACHER_FORCING DEBUG]")
                print(f"  Prefilled {len(self.context_tokens)} context tokens")
                print(f"  Got {len(distributions)} distributions from prefill")
                print(f"  Using distributions[-1] as P(next | full context)")
                top5_vals, top5_ids = self._last_probs.topk(5)
                print(f"  distributions[-1] top-5 tokens: {top5_ids.tolist()}")
                print(
                    f"  distributions[-1] top-5 probs: {[f'{p:.6f}' for p in top5_vals.tolist()]}"
                )
                print(
                    f"  Stored {len(self.decoder_prefill_distributions)} decoder prefill distributions for comparison"
                )
        else:
            # Single token context - just stage it, no KV cache yet
            self.staged_token = self.context_tokens[-1]
            self.past_key_values = None

    def compute_token_prob(self) -> None:
        """
        Compute probability distribution for next token given current context.
        Does NOT modify context or staged_token - just computes and stores the distribution.
        Can be called multiple times (idempotent until context changes).

        This is useful for getting the initial distribution after reset_teacher_forcing().
        """
        # If we already have a distribution from prefill, no need to compute again
        if self._last_probs is not None:
            return

        if self.staged_token is None:
            raise RuntimeError(
                "Call reset_teacher_forcing([BOS]) before compute_token_prob()."
            )

        dev = device_of(self.model)

        with torch.inference_mode(), sdp_backend(self.sdpa_backend):
            # Build inputs for a single-step forward on the staged token
            input_ids = torch.tensor([[self.staged_token]],
                                     dtype=torch.long,
                                     device=dev)

            out = self.model(input_ids=input_ids,
                             past_key_values=self.past_key_values,
                             use_cache=self.use_cache)
            logits_next = out.logits[0, -1, :].float()

            # Update KV cache for efficiency on next call
            if self.use_cache:
                self.past_key_values = out.past_key_values

        # Convert to probabilities
        if self.temperature != 1.0 and self.temperature > 0:
            logits_next = logits_next / self.temperature
        probs = torch.softmax(logits_next, dim=-1).detach()
        if not self.keep_on_device:
            probs = probs.cpu()

        # Store in idempotent buffer
        self._last_probs = probs

    def add_next_token_teacher_forcing(self, token: int) -> None:
        """
        Add token to context, then compute P(next | new_context).

        Args:
            token: Token ID to add to context
        """
        # If staged_token is None, it means we just did prefill and _last_probs contains
        # the first distribution. Just stage the token and proceed.
        if self.staged_token is None and self._last_probs is None:
            raise RuntimeError(
                "Call reset_teacher_forcing([BOS]) before add_next_token_teacher_forcing()."
            )

        # Add token to context first
        self.context_tokens.append(token)
        self.staged_token = token

        # Now compute next distribution
        dev = device_of(self.model)

        with torch.inference_mode(), sdp_backend(self.sdpa_backend):
            input_ids = torch.tensor([[self.staged_token]],
                                     dtype=torch.long,
                                     device=dev)

            out = self.model(input_ids=input_ids,
                             past_key_values=self.past_key_values,
                             use_cache=self.use_cache)
            logits_next = out.logits[0, -1, :].float()

            if self.use_cache:
                self.past_key_values = out.past_key_values

        # Convert to probabilities
        if self.temperature != 1.0 and self.temperature > 0:
            logits_next = logits_next / self.temperature
        probs = torch.softmax(logits_next, dim=-1).detach()
        if not self.keep_on_device:
            probs = probs.cpu()

        self._last_probs = probs

    # ---------- Compatibility methods for original API ----------
    def reset(self, initial_text: str = "") -> None:
        """
        Compatibility method for original API.
        Tokenizes initial_text and calls reset_teacher_forcing().

        Args:
            initial_text: Text to tokenize as initial context (or empty for BOS)
        """
        if initial_text:
            if self.tokenizer is None:
                raise RuntimeError("tokenizer required for reset() with text")
            tokens = self.tokenizer.encode(initial_text)
            self.reset_teacher_forcing(tokens)
        else:
            # Start with BOS token if available
            bos_id = None
            if self.tokenizer is not None:
                bos_id = self.tokenizer.bos_token_id
                if bos_id is None:
                    bos_id = self.tokenizer.eos_token_id
            if bos_id is None:
                bos_id = 0
            self.reset_teacher_forcing([bos_id])

    def get_token_probability(self) -> torch.Tensor:
        """
        Returns the distribution stored by the most recent operation.
        Calling this multiple times returns the exact same tensor.

        Returns:
            Tensor of shape [vocab_size] with probability distribution
        """
        if self._last_probs is None:
            raise RuntimeError(
                "Call add_token(token) or prefill() before get_token_probability()."
            )
        return self._last_probs

    def get_next_token_probs(self) -> torch.Tensor:
        """
        Alias for get_token_probability() for compatibility with original API.
        """
        return self.get_token_probability()
