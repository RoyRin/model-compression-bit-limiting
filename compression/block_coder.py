import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, GPTNeoXForCausalLM
from typing import Generator, Optional, Tuple, List, Any, Union
import json
import math
from decimal import Decimal, getcontext
from compression import debug_print
from compression.probability_generator import ModelProbabilityGenerator
import logging

ZERO = Decimal(0)
ONE = Decimal(1)


def decimal_to_uint(d, bits):
    """Convert Decimal to uint"""
    return int(d * (2**bits - 1))


def uint_to_decimal(n, bits):
    """Convert uint to Decimal"""
    n_decimal = Decimal(n)
    denominator = Decimal(2**bits - 1)
    return n_decimal / denominator


def float_to_decimal(f):
    """Convert float to Decimal, handling precision issues"""
    return Decimal(str(f))


def tensor_to_cumsum_float32(tensor_probs, min_prob=1e-8):
    """Fast version using float32 for cumsum, only converting final values to Decimal."""
    # Work in float32/64
    tensor_probs = torch.clamp(tensor_probs, min=min_prob)
    tensor_probs = tensor_probs / tensor_probs.sum()

    # Get cumsum in float
    probs_np = tensor_probs.cpu().numpy().astype(np.float64)
    cumsum = np.cumsum(probs_np)
    cumsum = np.insert(cumsum, 0, 0)  # Add 0 at the beginning

    # Ensure last value is exactly 1.0
    cumsum[-1] = 1.0

    return cumsum[:-1], cumsum[1:]


def get_index_binary_search_float(cumsum_lows, cumsum_highs,
                                  value: float) -> int:
    """Binary search on float arrays."""
    left, right = 0, len(cumsum_lows) - 1

    while left <= right:
        mid = (left + right) // 2
        if value < cumsum_lows[mid]:
            right = mid - 1
        elif value >= cumsum_highs[mid]:
            left = mid + 1
        else:
            return mid

    # Edge case handling
    if value >= cumsum_highs[-1] - 1e-9:
        return len(cumsum_lows) - 1

    return left


def token_id_to_binary(token_id: int, vocab_size: int):
    # convert the token id to a binary string
    num_bits = math.ceil(math.log2(vocab_size))
    # return a list
    binary_list = [int(bit) for bit in bin(token_id)[2:].zfill(num_bits)]
    return binary_list


import math
import torch
from typing import List, Tuple


def _bits_to_int(bits: List[int]) -> int:
    x = 0
    for b in bits:
        x = (x << 1) | int(b)
    return x


def _prefix_interval(prefix: List[int], num_bits: int,
                     vocab_size: int) -> Tuple[int, int]:
    """Half-open interval [start, end) of token IDs whose num_bits-wide binary starts with prefix."""
    if len(prefix) > num_bits:
        return (0, 0)

    p = _bits_to_int(prefix)
    shift = num_bits - len(prefix)
    start = p << shift
    end = (p + 1) << shift

    # clamp to vocab range
    start = max(0, min(start, vocab_size))
    end = max(0, min(end, vocab_size))
    return start, end


def _range_sum(prefix_cumsum: torch.Tensor, start: int,
               end: int) -> torch.Tensor:
    """Sum token_prob[start:end] using inclusive cumsum."""
    if end <= start:
        return prefix_cumsum.new_zeros(())
    if start == 0:
        return prefix_cumsum[end - 1]
    return prefix_cumsum[end - 1] - prefix_cumsum[start - 1]


def get_binary_probability_distribution_by_prefix_fast(
    token_prob: torch.Tensor,
    prefix: List[int],
) -> Tuple[float, float]:
    """
    Returns (P(prefix+[0]), P(prefix+[1])).
    Uses contiguous ranges + cumsum for O(1) queries after O(V) preprocessing.
    """
    vocab_size = int(token_prob.shape[0])
    num_bits = math.ceil(math.log2(vocab_size))

    # O(V) preprocessing for this token_prob
    c = token_prob.cumsum(dim=0)

    s0, e0 = _prefix_interval(prefix + [0], num_bits, vocab_size)
    s1, e1 = _prefix_interval(prefix + [1], num_bits, vocab_size)

    p0 = _range_sum(c, s0, e0).item()
    p1 = _range_sum(c, s1, e1).item()
    return p0, p1


######
# Bit-level encoding/decoding functions
######


def _int_to_bits(x: int, num_bits: int) -> List[int]:
    return [int(b) for b in bin(x)[2:].zfill(num_bits)]


def _get_p0_p1_from_prefix(
    token_probs: torch.Tensor,
    prefix_cumsum: torch.Tensor,
    prefix: List[int],
    num_bits: int,
) -> Tuple[float, float]:
    """
    Returns (p0, p1) where:
      p0 = sum_{ids with bits prefix+[0]+*} token_probs[id]
      p1 = sum_{ids with bits prefix+[1]+*} token_probs[id]
    Uses contiguous intervals + cumsum => O(1) per query after building cumsum.
    """
    vocab_size = int(token_probs.shape[0])
    s0, e0 = _prefix_interval(prefix + [0], num_bits, vocab_size)
    s1, e1 = _prefix_interval(prefix + [1], num_bits, vocab_size)
    p0 = float(_range_sum(prefix_cumsum, s0, e0).item()) if isinstance(
        _range_sum(prefix_cumsum, s0, e0), torch.Tensor) else float(
            _range_sum(prefix_cumsum, s0, e0))
    p1 = float(_range_sum(prefix_cumsum, s1, e1).item()) if isinstance(
        _range_sum(prefix_cumsum, s1, e1), torch.Tensor) else float(
            _range_sum(prefix_cumsum, s1, e1))
    return p0, p1


def encode_single_token_binary(
    token_id: int,
    token_probs: torch.Tensor,
    low: Decimal,
    high: Decimal,
    min_prob: float = 1e-12,
) -> Tuple[Decimal, Decimal]:
    """
    Encode a single token via arithmetic coding at the *bit level*.

    - token_id -> num_bits bits (num_bits = ceil(log2(vocab_size))).
    - For each prefix, compute induced Bernoulli split:
        split = P(next_bit=0 | prefix) = p0/(p0+p1)
      then AC-update (low, high) based on the actual next bit.

    Returns: (new_low, new_high)
    """
    vocab_size = int(token_probs.shape[0])
    num_bits = math.ceil(math.log2(vocab_size))
    bits = _int_to_bits(token_id, num_bits)

    if token_id < 0 or token_id >= vocab_size:
        raise ValueError(
            f"token_id={token_id} out of range for vocab_size={vocab_size}")
    if token_probs.dim() != 1 or int(token_probs.shape[0]) != vocab_size:
        raise ValueError(
            "token_probs must be a 1D tensor of shape (vocab_size,)")

    prefix_cumsum = token_probs.cumsum(dim=0)

    prefix: List[int] = []
    cur_low, cur_high = low, high

    for bit in bits:
        p0, p1 = _get_p0_p1_from_prefix(token_probs, prefix_cumsum, prefix,
                                        num_bits)
        denom = p0 + p1

        if denom < float(min_prob):
            p0_norm = 0.5
        else:
            p0_norm = p0 / denom
            eps = float(min_prob)
            p0_norm = min(max(p0_norm, eps), 1.0 - eps)

        split = Decimal(str(p0_norm))
        rng = cur_high - cur_low
        if rng <= 0:
            raise ValueError(
                f"Invalid range during encoding: low={cur_low}, high={cur_high}"
            )

        if bit == 0:
            new_low = cur_low
            new_high = cur_low + rng * split
        else:
            new_low = cur_low + rng * split
            new_high = cur_high

        cur_low, cur_high = new_low, new_high
        prefix.append(int(bit))

    return cur_low, cur_high


def decode_single_token_binary(
    encoded_value: Decimal,
    token_probs: torch.Tensor,
    low: Decimal,
    high: Decimal,
    min_prob: float = 1e-12,
) -> Tuple[int, Decimal, Decimal]:
    """
    Decode a single token via arithmetic coding at the *bit level* under the same induced
    binary conditionals used by encode_single_token_binary.

    Returns: (token_id, new_low, new_high)
    """
    vocab_size = int(token_probs.shape[0])
    num_bits = math.ceil(math.log2(vocab_size))

    if token_probs.dim() != 1 or int(token_probs.shape[0]) != vocab_size:
        raise ValueError(
            "token_probs must be a 1D tensor of shape (vocab_size,)")

    cur_low, cur_high = low, high
    if cur_high - cur_low <= 0:
        raise ValueError(f"Invalid range: low={low}, high={high}")

    prefix_cumsum = token_probs.cumsum(dim=0)

    bits: List[int] = []
    prefix: List[int] = []

    for _ in range(num_bits):
        p0, p1 = _get_p0_p1_from_prefix(token_probs, prefix_cumsum, prefix,
                                        num_bits)
        denom = p0 + p1

        if denom < float(min_prob):
            p0_norm = 0.5
        else:
            p0_norm = p0 / denom
            eps = float(min_prob)
            p0_norm = min(max(p0_norm, eps), 1.0 - eps)

        split = Decimal(str(p0_norm))
        rng = cur_high - cur_low
        if rng <= 0:
            raise ValueError(
                f"Invalid range during decoding: low={cur_low}, high={cur_high}"
            )

        scaled = (encoded_value - cur_low) / rng
        if scaled < 0:
            scaled = Decimal(0)
        elif scaled > 1:
            scaled = Decimal(1)

        if scaled < split:
            bit = 0
            new_low = cur_low
            new_high = cur_low + rng * split
        else:
            bit = 1
            new_low = cur_low + rng * split
            new_high = cur_high

        bits.append(bit)
        prefix.append(bit)
        cur_low, cur_high = new_low, new_high

    token_id = _bits_to_int(bits)

    # vocab_size may not be a power of two
    if token_id >= vocab_size:
        raise ValueError(
            f"Decoded token_id={token_id} out of range for vocab_size={vocab_size}. "
            "If vocab_size is not a power of two, ensure token_probs assigns effectively "
            "zero mass to invalid IDs (or handle invalid leaf IDs explicitly)."
        )

    return token_id, cur_low, cur_high


######
# PMATIC Helper Bits + Quantized Probabilities
######


def in_delta_interior(p: float, k: int, m: int, r: float,
                      delta: float) -> bool:
    """
    Check if probability p is in the δ-interior of bin k.

    Args:
        p: probability value in [0, 1]
        k: bin index (1-indexed, from 1 to m)
        m: number of bins (m = 1/(2r))
        r: bin half-width parameter
        delta: tolerance parameter

    Returns:
        True if p is in the δ-interior of bin k
    """
    if k == 1:
        # Edge bin: I_1^δ = [0, 2r - delta]
        return p <= 2 * r - delta
    elif k == m:
        # Edge bin: I_m^δ = [2r(m-1) + delta, 1]
        return p >= 2 * r * (m - 1) + delta
    else:
        # Interior bin: I_k^δ = [2r(k-1) + delta, 2rk - delta]
        bin_low = 2 * r * (k - 1) + delta
        bin_high = 2 * r * k - delta
        return bin_low <= p <= bin_high


def pmatic_quantize(p: float, r: float, delta: float) -> Tuple[int, float]:
    """
    PMATIC quantization: returns (helper_bit, p_hat).

    Args:
        p: raw probability in [0, 1]
        r: bin half-width (must satisfy r > 2*delta, and 1/(2r) must be integer)
        delta: tolerance parameter

    Returns:
        (helper_bit, p_hat) where:
        - helper_bit = 0 if p is in δ-interior, 1 if near boundary
        - p_hat = quantized probability (bin center or boundary)
    """
    # Clamp p to [0, 1]
    p = max(0.0, min(1.0, p))

    # Number of bins
    m = int(round(1.0 / (2 * r)))

    # Find bin index k (1-indexed)
    # Bin k covers [2r(k-1), 2rk]
    k = min(m, int(p / (2 * r)) + 1)

    # Check if p is in the δ-interior of bin k
    if in_delta_interior(p, k, m, r, delta):
        # Case 1: helper = 0, quantize to bin center
        center = 2 * r * (k - 1) + r
        return 0, center
    else:
        # Case 2: helper = 1, quantize to nearest internal boundary
        # Internal boundaries are at 2r*k for k in {1, ..., m-1}
        best_k_star = 1
        best_dist = abs(p - 2 * r * 1)

        for k_star in range(1, m):
            boundary = 2 * r * k_star
            dist = abs(p - boundary)
            if dist < best_dist:
                best_dist = dist
                best_k_star = k_star

        return 1, 2 * r * best_k_star


def _encode_bit_arith(bit: int,
                      p_one: float,
                      low: Decimal,
                      high: Decimal,
                      min_prob: float = 1e-12) -> Tuple[Decimal, Decimal]:
    """
    Arithmetic-encode a single bit with P(bit=1) = p_one.

    Args:
        bit: the bit value (0 or 1)
        p_one: probability that bit = 1
        low, high: current arithmetic coding interval
        min_prob: minimum probability to clamp to

    Returns:
        (new_low, new_high)
    """
    # Clamp probability
    p_one = max(min_prob, min(1.0 - min_prob, p_one))
    p_zero = 1.0 - p_one

    rng = high - low
    split = low + rng * Decimal(str(p_zero))

    if bit == 0:
        return low, split
    else:
        return split, high


def _decode_bit_arith(encoded_value: Decimal,
                      p_one: float,
                      low: Decimal,
                      high: Decimal,
                      min_prob: float = 1e-12) -> Tuple[int, Decimal, Decimal]:
    """
    Arithmetic-decode a single bit with P(bit=1) = p_one.

    Args:
        encoded_value: the encoded value
        p_one: probability that bit = 1
        low, high: current arithmetic coding interval
        min_prob: minimum probability to clamp to

    Returns:
        (decoded_bit, new_low, new_high)
    """
    # Clamp probability
    p_one = max(min_prob, min(1.0 - min_prob, p_one))
    p_zero = 1.0 - p_one

    rng = high - low
    split = low + rng * Decimal(str(p_zero))

    if encoded_value < split:
        return 0, low, split
    else:
        return 1, split, high


def encode_single_token_pmatic(
    token_id: int,
    token_probs: torch.Tensor,
    low: Decimal,
    high: Decimal,
    r: float,
    delta: float,
    min_prob: float = 1e-12,
) -> Tuple[Decimal, Decimal]:
    """
    Encode a single token using PMATIC (helper bits + quantized probabilities).

    For each bit in the token:
    1. Compute raw probability p = p1 / (p0 + p1)
    2. Quantize: (helper, p_hat) = pmatic_quantize(p, r, delta)
    3. Arithmetic-code helper bit using P(helper=1) = delta/r
    4. Arithmetic-code actual token bit using P(bit=1) = p_hat

    Args:
        token_id: token to encode
        token_probs: probability distribution over vocab
        low, high: current arithmetic coding interval
        r: PMATIC bin half-width parameter
        delta: PMATIC tolerance parameter
        min_prob: minimum probability for clamping

    Returns:
        (new_low, new_high)
    """
    vocab_size = int(token_probs.shape[0])
    num_bits = math.ceil(math.log2(vocab_size))
    bits = _int_to_bits(token_id, num_bits)

    if token_id < 0 or token_id >= vocab_size:
        raise ValueError(
            f"token_id={token_id} out of range for vocab_size={vocab_size}")

    prefix_cumsum = token_probs.cumsum(dim=0)

    prefix: List[int] = []
    cur_low, cur_high = low, high

    # Probability of helper=1
    p_helper_one = delta / r

    for bit in bits:
        # Get p0, p1 for current prefix
        p0, p1 = _get_p0_p1_from_prefix(token_probs, prefix_cumsum, prefix,
                                        num_bits)
        denom = p0 + p1

        if denom < min_prob:
            p_raw = 0.5
        else:
            p_raw = p1 / denom  # P(next_bit = 1 | prefix)

        # PMATIC quantization
        helper, p_hat = pmatic_quantize(p_raw, r, delta)

        # Arithmetic-code the helper bit
        cur_low, cur_high = _encode_bit_arith(helper, p_helper_one, cur_low,
                                              cur_high, min_prob)

        # Arithmetic-code the actual token bit using p_hat
        cur_low, cur_high = _encode_bit_arith(bit, p_hat, cur_low, cur_high,
                                              min_prob)

        prefix.append(int(bit))

    return cur_low, cur_high


def _pmatic_quantize_with_helper(q_raw: float, helper: int, r: float) -> float:
    """
    Compute quantized probability using the decoded helper bit.

    This is the key to PMATIC robustness: the helper bit tells the decoder
    whether the encoder was in the δ-interior (helper=0) or near a boundary
    (helper=1), allowing correct quantization even when encoder/decoder
    probabilities differ slightly.

    Args:
        q_raw: decoder's raw probability in [0, 1]
        helper: decoded helper bit (0 = interior, 1 = boundary)
        r: bin half-width parameter

    Returns:
        q_hat: quantized probability
    """
    q_raw = max(0.0, min(1.0, q_raw))  # Clamp to [0, 1]
    m = int(round(1.0 / (2 * r)))  # Number of bins

    if helper == 0:
        # Encoder was in δ-interior → quantize to bin center
        k = min(m, int(q_raw / (2 * r)) + 1)
        q_hat = 2 * r * (k - 1) + r  # bin center
    else:
        # Encoder was near boundary → quantize to nearest internal boundary
        best_k_star = 1
        best_dist = abs(q_raw - 2 * r)
        for k_star in range(2, m):
            boundary = 2 * r * k_star
            dist = abs(q_raw - boundary)
            if dist < best_dist:
                best_dist = dist
                best_k_star = k_star
        q_hat = 2 * r * best_k_star

    return q_hat


def decode_single_token_pmatic(
    encoded_value: Decimal,
    token_probs: torch.Tensor,
    low: Decimal,
    high: Decimal,
    r: float,
    delta: float,
    min_prob: float = 1e-12,
) -> Tuple[int, Decimal, Decimal]:
    """
    Decode a single token using PMATIC (helper bits + quantized probabilities).

    For each bit position:
    1. Decode helper bit using P(helper=1) = delta/r
    2. Compute decoder's raw probability q = q1 / (q0 + q1)
    3. Use helper bit to determine q_hat (not just re-quantize blindly)
    4. Decode actual token bit using P(bit=1) = q_hat

    PMATIC guarantees p_hat == q_hat when |p_raw - q_raw| < δ, because the
    helper bit disambiguates boundary cases.

    Args:
        encoded_value: the encoded value to decode
        token_probs: probability distribution over vocab
        low, high: current arithmetic coding interval
        r: PMATIC bin half-width parameter
        delta: PMATIC tolerance parameter
        min_prob: minimum probability for clamping

    Returns:
        (token_id, new_low, new_high)
    """
    vocab_size = int(token_probs.shape[0])
    num_bits = math.ceil(math.log2(vocab_size))

    prefix_cumsum = token_probs.cumsum(dim=0)

    prefix: List[int] = []
    cur_low, cur_high = low, high

    # Probability of helper=1
    p_helper_one = delta / r

    for _ in range(num_bits):
        # Decode helper bit
        helper, cur_low, cur_high = _decode_bit_arith(encoded_value,
                                                      p_helper_one, cur_low,
                                                      cur_high, min_prob)

        # Get decoder's p0, p1 for current prefix
        q0, q1 = _get_p0_p1_from_prefix(token_probs, prefix_cumsum, prefix,
                                        num_bits)
        denom = q0 + q1

        if denom < min_prob:
            q_raw = 0.5
        else:
            q_raw = q1 / denom  # P(next_bit = 1 | prefix)

        # Use decoded helper bit to determine q_hat
        q_hat = _pmatic_quantize_with_helper(q_raw, helper, r)

        # Decode actual token bit using q_hat
        bit, cur_low, cur_high = _decode_bit_arith(encoded_value, q_hat,
                                                   cur_low, cur_high, min_prob)

        prefix.append(bit)

    token_id = _bits_to_int(prefix)

    if token_id >= vocab_size:
        raise ValueError(
            f"Decoded token_id={token_id} out of range for vocab_size={vocab_size}. "
            "If vocab_size is not a power of two, ensure token_probs assigns effectively "
            "zero mass to invalid IDs.")

    return token_id, cur_low, cur_high


######
# Legacy helper functions (kept for compatibility)
######


def tensor_to_decimal_cumsum(tensor_probs, min_prob=1e-8):
    """Convert tensor probabilities to Decimal cumulative sum arrays."""
    cumsum_lows, cumsum_highs = tensor_to_cumsum_float32(
        tensor_probs, min_prob)

    # Convert to Decimal only if needed
    decimal_lows = [Decimal(str(float(x))) for x in cumsum_lows]
    decimal_highs = [Decimal(str(float(x))) for x in cumsum_highs]

    return decimal_lows, decimal_highs


def get_index_of_token_in_decimal_cumsum(cumsum_lows,
                                         cumsum_highs,
                                         value: Decimal,
                                         min_prob=1e-8) -> int:
    """Find token index using binary search."""
    # Convert to float for search
    if isinstance(cumsum_lows[0], Decimal):
        cumsum_lows_float = [float(x) for x in cumsum_lows]
        cumsum_highs_float = [float(x) for x in cumsum_highs]
    else:
        cumsum_lows_float = cumsum_lows
        cumsum_highs_float = cumsum_highs

    value_float = float(value)

    return get_index_binary_search_float(cumsum_lows_float, cumsum_highs_float,
                                         value_float)


from typing import List, Sequence
from decimal import Decimal

# Set up logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Set high precision for Decimal operations
getcontext().prec = 100

# Decimal constants
ZERO = Decimal(0)
ONE = Decimal(1)
# Debug flag
DEBUG = False
DEFAULT_BIT_PRECISION = 64
BITS_FOR_ENCODING_COUNT = 7


###
# Enhanced encoder with Decimal arithmetic
###
class BlockEmissionArithmeticCoder:

    def __init__(self,
                 model,
                 tokenizer,
                 bit_precision: int = DEFAULT_BIT_PRECISION,
                 bits_for_encoding_count: int = BITS_FOR_ENCODING_COUNT,
                 min_prob: float = 1e-9,
                 device=None,
                 verbose: bool = False,
                 track_token_ranks: bool = False,
                 temperature: float = 1.0,
                 debug: bool = False,
                 use_pmatic: bool = False,
                 pmatic_r: float = 0.1,
                 pmatic_delta: float = 0.02):
        """Initialize the block emission arithmetic encoder.

        Args:
            use_pmatic: If True, use PMATIC helper bits + quantized probabilities
            pmatic_r: PMATIC bin half-width (must satisfy r > 2*delta, 1/(2r) must be int)
            pmatic_delta: PMATIC tolerance parameter
        """
        self.tokenizer = tokenizer
        self.min_prob = min_prob

        self.model_gen = ModelProbabilityGenerator(model,
                                                   tokenizer=tokenizer,
                                                   device=device,
                                                   temperature=temperature,
                                                   use_cache=True,
                                                   keep_on_device=True,
                                                   debug=debug)
        self.bit_precision = bit_precision
        self.bits_for_encoding_count = bits_for_encoding_count
        self.min_prob_decimal = Decimal(str(min_prob))
        self.verbose = verbose
        self.track_token_ranks = track_token_ranks
        self.temperature = temperature

        # PMATIC parameters
        self.use_pmatic = use_pmatic
        self.pmatic_r = pmatic_r
        self.pmatic_delta = pmatic_delta

        # Debug flag
        self.debug = debug

        # Storage for probability distributions (for debugging/analysis)
        self.stored_probs = []

        # Calculate derived constants
        self.max_val_decimal = Decimal(2**bit_precision - 1)
        self.near_max_val_decimal = Decimal(2**(bit_precision - 3) - 1)
        self.emission_threshold_decimal = Decimal(
            1) / self.near_max_val_decimal

    def arithmetic_encode_single_token(
            self, token_id: int, token_probs: torch.Tensor, low: Decimal,
            high: Decimal) -> Tuple[Decimal, Decimal]:
        """Arithmetic encoding with Decimal precision."""
        if low >= high:
            raise ValueError(
                f"Low ({low}) is greater than or equal to high ({high})!")
        if (high - low) < self.min_prob_decimal:
            debug_print(
                f"Warning: Range is potentially too small! {high - low}")

        # Convert probabilities to Decimal cumulative sums
        cumsum_lows, cumsum_highs = tensor_to_decimal_cumsum(
            token_probs, self.min_prob)

        # Get the probability range for this token
        token_low = cumsum_lows[token_id]
        token_high = cumsum_highs[token_id]

        # Update range using exact Decimal arithmetic
        range_size = high - low
        new_low = low + (range_size * token_low)
        new_high = low + (range_size * token_high)

        debug_print(
            f"Encoding token {token_id}: "
            f"cum_range=[{float(token_low):.6f}, {float(token_high):.6f}], "
            f"new_range=[{float(new_low):.10f}, {float(new_high):.10f}]")

        return new_low, new_high

    def arithmetic_encode_tokens(self,
                                 tokens: List[int],
                                 model_generator: ModelProbabilityGenerator,
                                 initial_context: Optional[List[int]] = None,
                                 store_probs: bool = False,
                                 use_prefill: bool = False):
        """Encode tokens using block emission arithmetic coding with Decimal precision.

        Args:
            tokens: List of token IDs to encode.
            model_generator: ModelProbabilityGenerator instance.
            initial_context: Optional context tokens (defaults to BOS).
            store_probs: If True, store probability distributions in self.stored_probs.
            use_prefill: If True, use batch prefill mode; otherwise use teacher forcing.
        """
        # Use provided context or default to BOS
        if initial_context is None:
            bos_id = model_generator.tokenizer.bos_token_id
            initial_context = [bos_id]

        # Initialize model in appropriate mode
        if use_prefill:
            # Prefill mode: batch compute all probabilities at once (faster)
            # HF semantics: logits[i] predicts next token after seeing tokens[:i+1]
            # So for context of length C, distributions[C-1] is P(next | full_context)
            # We prefill context + tokens[:-1] to get all distributions we need
            tokens_with_context = initial_context + tokens[:-1]
            if self.debug:
                print(f"[PREFILL DEBUG] initial_context={initial_context}")
                print(
                    f"[PREFILL DEBUG] tokens[:3]={tokens[:3]}, len(tokens)={len(tokens)}"
                )
                print(
                    f"[PREFILL DEBUG] tokens_with_context[:5]={tokens_with_context[:5]}, len={len(tokens_with_context)}"
                )
            model_generator.prefill(tokens_with_context)

            # Start at the last context position
            # distributions[C-1] is P(tokens[0] | full_context)
            start_idx = len(initial_context) - 1
            if self.debug:
                print(
                    f"[PREFILL DEBUG] start_idx={start_idx}, num_distributions={len(model_generator.distributions)}"
                )
            model_generator.current_index = start_idx
            model_generator._last_probs = model_generator.distributions[
                start_idx]
            # Debug: show what the first distribution looks like
            if self.debug:
                top5 = model_generator._last_probs.topk(5)
                print(
                    f"[PREFILL DEBUG] distributions[{start_idx}] top-5: {top5.indices.tolist()}"
                )
        else:
            # Teacher forcing mode: incremental generation (matches decoder)
            model_generator.reset_teacher_forcing(initial_context)
            model_generator.compute_token_prob()  # Compute first distribution

        encoding_buffer = []
        range_sizes = []
        low = ZERO
        high = ONE
        # Use instance variable for storing probs if store_probs is True
        if store_probs:
            self.stored_probs.clear()
            stored_probs = self.stored_probs
        else:
            stored_probs = None
        encoding_count = 0
        encoded_token_counts = []
        token_ranks = [] if self.track_token_ranks else None

        for i, token in enumerate(tokens):
            prob = model_generator.get_token_probability()

            # Debug: show what distribution we're using for this token
            if self.debug and i < 3:
                top5 = prob.topk(5)
                print(f"[ENCODE DEBUG] Position {i}, token={token}")
                print(
                    f"  current_index={model_generator.current_index if hasattr(model_generator, 'current_index') else 'N/A'}"
                )
                print(
                    f"  top-5 tokens: {top5.indices.tolist()}, probs: {[f'{p:.4f}' for p in top5.values.tolist()]}"
                )
                print(f"  P(actual_token={token}): {prob[token].item():.6f}")

            if store_probs:
                stored_probs.append(prob.cpu().clone())

            # Choose encoding function based on PMATIC setting
            if self.use_pmatic:
                encode_fn = lambda tid, p, lo, hi: encode_single_token_pmatic(
                    tid, p, lo, hi, self.pmatic_r, self.pmatic_delta, self.
                    min_prob)
            else:
                encode_fn = lambda tid, p, lo, hi: encode_single_token_binary(
                    tid, p, lo, hi, self.min_prob)

            # Check if encoding this token would make range too small
            new_low, new_high = encode_fn(token, prob, low, high)

            if (new_high - new_low < self.emission_threshold_decimal) or (
                    encoding_count >= 2**self.bits_for_encoding_count - 1):
                # Emit current range before encoding this token
                average = (low + high) / Decimal(2)
                uint_value = decimal_to_uint(average, self.bit_precision)
                encoding_buffer.append(uint_value)
                encoded_token_counts.append(encoding_count)

                # Encode this token with fresh range
                low, high = encode_fn(token, prob, ZERO, ONE)
                encoding_count = 1
            else:
                low, high = new_low, new_high
                encoding_count += 1
                if encoding_count > 2**self.bits_for_encoding_count:
                    raise ValueError(
                        f"Encoding count {encoding_count} exceeds max")

            # Update model state
            if use_prefill:
                model_generator.add_token_prefill()
            else:
                model_generator.add_next_token_teacher_forcing(token)
            range_sizes.append(high - low)

        # Don't forget the final value if range is not reset
        if low != ZERO or high != ONE:
            final_value = (low + high) / Decimal(2)

            uint_value = decimal_to_uint(final_value, self.bit_precision)
            encoding_buffer.append(uint_value)
            debug_print(f"Emitting final value: {float(final_value)}")
            range_sizes.append(high - low)

        encoded_token_counts.append(encoding_count)

        # Return results based on tracking options
        results = [range_sizes]
        if store_probs:
            results.append(stored_probs)
        results.append(encoded_token_counts)
        if self.track_token_ranks:
            results.append(token_ranks)

        return encoding_buffer, tuple(results)

    def encode(self,
               tokens: List[int],
               initial_context: Optional[List[int]] = None,
               store_probs: bool = False,
               use_prefill: bool = False) -> Tuple[List[int], Tuple]:
        """Encode a sequence of tokens into compressed representation.

        Returns:
            Tuple of (encoded_values, encoding_info).
            encoding_info contains: (range_sizes, [stored_probs], encoded_token_counts, ...).
            When store_probs=True, stored_probs is included in the tuple.
        """
        return self.arithmetic_encode_tokens(tokens,
                                             self.model_gen,
                                             initial_context,
                                             store_probs=store_probs,
                                             use_prefill=use_prefill)


###
# Enhanced decoder with Decimal arithmetic
###


class BlockEmissionArithmeticDecoder:

    def __init__(self,
                 model,
                 tokenizer,
                 bit_precision: int = 60,
                 bits_for_encoding_count: int = BITS_FOR_ENCODING_COUNT,
                 min_prob: float = 1e-9,
                 device=None,
                 verbose: bool = False,
                 temperature: float = 1.0,
                 debug: bool = False,
                 use_pmatic: bool = False,
                 pmatic_r: float = 0.1,
                 pmatic_delta: float = 0.02):
        """Initialize the block emission arithmetic decoder.

        Args:
            use_pmatic: If True, use PMATIC helper bits + quantized probabilities
            pmatic_r: PMATIC bin half-width (must satisfy r > 2*delta, 1/(2r) must be int)
            pmatic_delta: PMATIC tolerance parameter
        """
        self.tokenizer = tokenizer
        self.min_prob = min_prob

        self.model_gen = ModelProbabilityGenerator(model,
                                                   tokenizer=tokenizer,
                                                   device=device,
                                                   temperature=temperature,
                                                   use_cache=True,
                                                   keep_on_device=True,
                                                   debug=debug)
        self.bit_precision = bit_precision
        self.bits_for_encoding_count = bits_for_encoding_count
        self.min_prob_decimal = Decimal(str(min_prob))
        self.verbose = verbose
        self.temperature = temperature

        # PMATIC parameters
        self.use_pmatic = use_pmatic
        self.pmatic_r = pmatic_r
        self.pmatic_delta = pmatic_delta

        # Debug flag
        self.debug = debug

        # Storage for probability distributions (for debugging/analysis)
        self.stored_probs = []

        # Calculate derived constants
        self.max_val_decimal = Decimal(2**bit_precision - 1)
        self.near_max_val_decimal = Decimal(2**(bit_precision - 1) - 1)
        self.emission_threshold_decimal = Decimal(
            1) / self.near_max_val_decimal

        if self.verbose:
            print(
                f"Emission threshold decimal: {self.emission_threshold_decimal}"
            )
            print(f"Near max val decimal: {self.near_max_val_decimal}")
            print(f"Max val decimal: {self.max_val_decimal}")
            print(f"Bit precision: {self.bit_precision}")
            print(f"Bits for encoding count: {self.bits_for_encoding_count}")
            print(f"Min prob decimal: {self.min_prob_decimal}")
            print(f"Verbose: {self.verbose}")

    def arithmetic_decode_single_token(
            self, encoded_value: Decimal, token_probs: torch.Tensor,
            low: Decimal, high: Decimal) -> Tuple[int, Decimal, Decimal]:
        """Decode with Decimal arithmetic."""
        # Convert probabilities to Decimal cumulative sums
        cumsum_lows, cumsum_highs = tensor_to_decimal_cumsum(
            token_probs, self.min_prob)  # not that important

        # Scale the encoded value to unit interval [0,1) to find which bin
        dynamic_range = high - low

        # Check for degenerate range
        if dynamic_range <= 0:
            raise ValueError(
                f"Invalid range: low={low}, high={high}, range={dynamic_range}"
            )

        scaled_value = (encoded_value - low) / dynamic_range

        # Ensure scaled_value is in [0, 1]
        if scaled_value < 0 or scaled_value > 1:
            debug_print(f"WARNING: scaled_value {scaled_value} outside [0,1]!")
            scaled_value = max(ZERO, min(ONE, scaled_value))

        debug_print(f"Decoding: encoded_value={float(encoded_value):.10f}, "
                    f"range=[{float(low):.10f}, {float(high):.10f}], "
                    f"scaled_value={float(scaled_value):.10f}")

        # Find token
        token_id = get_index_of_token_in_decimal_cumsum(
            cumsum_lows, cumsum_highs, scaled_value)

        # Get the probability range for this token
        token_low = cumsum_lows[token_id]
        token_high = cumsum_highs[token_id]

        # Calculate new range using exact Decimal arithmetic
        new_low = low + (dynamic_range * token_low)
        new_high = low + (dynamic_range * token_high)

        debug_print(
            f"Decoded token {token_id}: "
            f"cum_range=[{float(token_low):.6f}, {float(token_high):.6f}], "
            f"new_range=[{float(new_low):.10f}, {float(new_high):.10f}]")

        return token_id, new_low, new_high

    def decode_block_emission(
        self,
        encoded_values: List[int],
        encoded_token_counts: List[int],
        model_generator: ModelProbabilityGenerator,
        initial_context: Optional[List[int]] = None,
        store_probs: bool = False,
    ) -> Tuple[List[int], List[Decimal], Optional[List[torch.Tensor]]]:
        """Decode encoded values back to tokens using block emission arithmetic coding."""
        # Use provided context or default to BOS
        if initial_context is None:
            bos_id = model_generator.tokenizer.bos_token_id
            initial_context = [bos_id]

        model_generator.reset_teacher_forcing(initial_context)
        model_generator.compute_token_prob()
        # Debug: show what the first decode distribution looks like
        if self.debug:
            first_probs = model_generator.get_token_probability()
            top5 = first_probs.topk(5)
            print(f"[DECODE INIT DEBUG] initial_context={initial_context}")
            print(
                f"[DECODE INIT DEBUG] First distribution top-5: {top5.indices.tolist()}"
            )

        # 1. Convert the payload back to Decimals in [0,1]
        decimal_values = [
            uint_to_decimal(v, self.bit_precision) for v in encoded_values
        ]

        decoded_tokens: List[int] = []
        range_sizes: List[Decimal] = []
        # Use instance variable for storing probs if store_probs is True
        if store_probs:
            self.stored_probs.clear()
            stored_probs = self.stored_probs
        else:
            stored_probs = None

        # Track global token index across all blocks
        global_token_idx = 0

        for value_idx in range(len(decimal_values)):
            encoded_value = decimal_values[value_idx]
            low, high = ZERO, ONE
            encoding_count = encoded_token_counts[value_idx]

            for token_i in range(encoding_count):
                token_probs = model_generator.get_token_probability()

                # Debug: show what distribution we're using for decoding
                if self.debug and global_token_idx < 3:
                    top5 = token_probs.topk(5)
                    print(f"[DECODE DEBUG] Position {global_token_idx}")
                    print(
                        f"  top-5 tokens: {top5.indices.tolist()}, probs: {[f'{p:.4f}' for p in top5.values.tolist()]}"
                    )

                if store_probs:
                    stored_probs.append(token_probs.cpu().clone())

                # Decode token using appropriate method
                if self.use_pmatic:
                    token_id, new_low, new_high = decode_single_token_pmatic(
                        encoded_value, token_probs, low, high, self.pmatic_r,
                        self.pmatic_delta, self.min_prob)
                else:
                    token_id, new_low, new_high = decode_single_token_binary(
                        encoded_value, token_probs, low, high, self.min_prob)

                decoded_tokens.append(token_id)

                # Debug: show what token was decoded
                if self.debug and global_token_idx < 3:
                    print(
                        f"  Decoded token: {token_id}, P(token): {token_probs[token_id].item():.6f}"
                    )

                model_generator.add_next_token_teacher_forcing(token_id)

                low, high = new_low, new_high
                range_sizes.append(new_high - new_low)
                global_token_idx += 1

                # safety valve to avoid infinite loops on malformed data
                if len(decoded_tokens) > 100_000:
                    raise RuntimeError(
                        "Decode exceeded 100 k tokens – aborting.")

        if store_probs:
            return decoded_tokens, range_sizes, stored_probs
        return decoded_tokens, range_sizes

    def decode(
        self,
        encoded_data: Any,
        num_tokens: int,
        initial_context: Optional[List[int]] = None,
        store_probs: bool = False
    ) -> Union[List[int], Tuple[List[int], List[torch.Tensor]]]:
        """Decode encoded data back to a sequence of tokens.

        Returns:
            If store_probs=False: List of decoded token IDs.
            If store_probs=True: Tuple of (decoded_tokens, stored_probs).
        """
        if isinstance(encoded_data, tuple) and len(encoded_data) == 2:
            encoded_values, encoding_info = encoded_data
            # Extract encoded_token_counts from encoding_info tuple
            # The structure is: (range_sizes, [stored_probs], encoded_token_counts, [token_ranks])
            # encoded_token_counts is always present but its position varies
            if isinstance(encoding_info, tuple):
                encoded_token_counts = None

                # Find the first list of integers that's not num_tokens long (that would be token_ranks)
                for item in encoding_info:
                    if isinstance(item, list) and len(item) > 0 and isinstance(
                            item[0], int):
                        if all(isinstance(x, int) for x in item):
                            # This is likely encoded_token_counts (shorter list) or token_ranks
                            if encoded_token_counts is None:
                                encoded_token_counts = item
                                break

                if encoded_token_counts is None:
                    raise ValueError(
                        "Could not find encoded_token_counts in encoding_info")
            else:
                encoded_token_counts = encoding_info
        else:
            raise ValueError(
                "encoded_data must be a tuple of (encoded_values, encoding_info)"
            )

        result = self.decode_block_emission(encoded_values,
                                            encoded_token_counts,
                                            self.model_gen,
                                            initial_context,
                                            store_probs=store_probs)
        if store_probs:
            # result is (decoded_tokens, range_sizes, stored_probs)
            return result[0], result[2]
        return result[0]  # Return just the decoded tokens


def compare_encoder_decoder_probs(
    encoder_probs: List[torch.Tensor],
    decoder_probs: List[torch.Tensor],
    prompt_length: int = 0,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
) -> dict:
    """
    Compare probability distributions from encoder and decoder.

    Args:
        encoder_probs: List of probability tensors from encoding.
        decoder_probs: List of probability tensors from decoding.
        prompt_length: Number of tokens in the prompt (for plotting boundary).
        save_path: Optional path to save the plot.
        title: Optional title for the plot.

    Returns:
        Dict with L2 and L-inf norms at each position.
    """
    import matplotlib.pyplot as plt

    assert len(encoder_probs) == len(decoder_probs), \
        f"Length mismatch: encoder={len(encoder_probs)}, decoder={len(decoder_probs)}"

    l2_norms = []
    linf_norms = []

    for enc_p, dec_p in zip(encoder_probs, decoder_probs):
        diff = (enc_p - dec_p).abs()
        l2 = torch.sqrt((diff**2).sum()).item()
        linf = diff.max().item()
        l2_norms.append(l2)
        linf_norms.append(linf)

    print(f"[DEBUG] L2 norms (first 20): {l2_norms[:20]}")
    print(f"[DEBUG] L-inf norms (first 20): {linf_norms[:20]}")
    print(
        f"[DEBUG] L2 max: {max(l2_norms)}, L2 mean: {sum(l2_norms)/len(l2_norms)}"
    )

    # Debug: show top-5 tokens for first 3 positions where there's a difference
    for i in range(min(3, len(encoder_probs))):
        if l2_norms[i] > 1e-6:
            enc_top5 = encoder_probs[i].topk(5)
            dec_top5 = decoder_probs[i].topk(5)
            print(f"[DEBUG] Position {i} - L2={l2_norms[i]:.6f}")
            print(
                f"  Encoder top-5 tokens: {enc_top5.indices.tolist()}, probs: {[f'{p:.6f}' for p in enc_top5.values.tolist()]}"
            )
            print(
                f"  Decoder top-5 tokens: {dec_top5.indices.tolist()}, probs: {[f'{p:.6f}' for p in dec_top5.values.tolist()]}"
            )

    # Create plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    positions = list(range(len(l2_norms)))

    # L2 norm plot
    axes[0].plot(positions, l2_norms, 'b-', linewidth=0.8, alpha=0.8)
    axes[0].set_ylabel('L2 Norm')
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(
        'L2 Norm of Probability Differences (Encoder vs Decoder)')

    # L-inf norm plot
    axes[1].plot(positions, linf_norms, 'r-', linewidth=0.8, alpha=0.8)
    axes[1].set_ylabel('L-inf Norm')
    axes[1].set_yscale('log')
    axes[1].set_xlabel('Token Position')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title(
        'L-inf Norm of Probability Differences (Encoder vs Decoder)')

    # Add vertical dashed line at prompt boundary
    if prompt_length > 0:
        for ax in axes:
            ax.axvline(x=prompt_length,
                       color='green',
                       linestyle='--',
                       linewidth=2,
                       label=f'Prompt boundary (pos {prompt_length})')
            ax.legend(loc='upper right')

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    plt.close()

    return {
        'l2_norms': l2_norms,
        'linf_norms': linf_norms,
        'l2_max': max(l2_norms),
        'l2_mean': sum(l2_norms) / len(l2_norms),
        'linf_max': max(linf_norms),
        'linf_mean': sum(linf_norms) / len(linf_norms),
        'num_positions': len(l2_norms),
        'prompt_length': prompt_length,
    }
