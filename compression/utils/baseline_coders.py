"""Baseline compression methods for comparison with arithmetic coding.

These baselines compress token sequences by encoding them as bytes and applying
standard compression algorithms. They do not use model probabilities.

Supported methods:
- gzip (DEFLATE, classic general-purpose)
- zstd (modern, fast with good ratios)
- lz4 (very fast, modest ratio)
- brotli (excellent for text/web content)
- lzma/xz (high compression ratio, slower)
- bzip2 (legacy, good text compression)
"""

import gzip
import bz2
import lzma
import time
from typing import List, Tuple, Dict, Optional
import numpy as np


class GzipCompressor:
    """Baseline compressor using gzip compression."""

    def __init__(self, level: int = 9):
        """Initialize gzip compressor.

        Args:
            level: Compression level (0-9, where 9 is maximum compression)
        """
        self.level = level
        # For compatibility with measure_compression.py interface
        self.bit_precision = 8  # Each compressed byte is 8 bits
        self.bits_for_encoding_count = 0  # No additional overhead per block

    def encode(self,
               tokens: List[int],
               initial_context: Optional[List[int]] = None,
               store_probs: bool = False,
               use_prefill: bool = False) -> Tuple[bytes, Dict]:
        """Encode tokens by converting to bytes and compressing with gzip.

        Args:
            tokens: List of token IDs to compress
            initial_context: Optional initial context (ignored for gzip)
            store_probs: Whether to store probabilities (ignored for gzip)
            use_prefill: Whether to use prefill mode (ignored for gzip)

        Returns:
            Tuple of (compressed_bytes, metadata_dict)
        """
        # Convert token IDs to bytes (using 4 bytes per int32)
        token_array = np.array(tokens, dtype=np.int32)
        token_bytes = token_array.tobytes()

        # Compress with gzip
        compressed = gzip.compress(token_bytes, compresslevel=self.level)

        # Metadata for decoder
        metadata = {
            'num_tokens': len(tokens),
            'original_bytes': len(token_bytes),
            'compressed_bytes': len(compressed)
        }

        return compressed, metadata

    def decode(self,
               encoded_data: Tuple[bytes, Dict],
               num_tokens: int,
               initial_context: Optional[List[int]] = None,
               store_probs: bool = False) -> List[int]:
        """Decode gzip-compressed tokens.

        Args:
            encoded_data: Tuple of (compressed_bytes, metadata_dict)
            num_tokens: Expected number of tokens (for verification)
            initial_context: Optional initial context (ignored for gzip)
            store_probs: Whether to store probabilities (ignored for gzip)

        Returns:
            List of decoded token IDs
        """
        compressed, metadata = encoded_data

        # Decompress
        decompressed_bytes = gzip.decompress(compressed)

        # Convert back to token IDs
        token_array = np.frombuffer(decompressed_bytes, dtype=np.int32)
        tokens = token_array.tolist()

        # Verify length
        assert len(
            tokens
        ) == num_tokens, f"Expected {num_tokens} tokens, got {len(tokens)}"

        return tokens


class ZstdCompressor:
    """Baseline compressor using Zstandard (zstd) compression."""

    def __init__(self, level: int = 3):
        """Initialize zstd compressor.

        Args:
            level: Compression level (1-22, where 22 is maximum compression)
                  Default 3 is the zstd default and provides good speed/ratio balance
        """
        try:
            import zstandard as zstd
            self.zstd = zstd
        except ImportError:
            raise ImportError(
                "zstandard package not found. Install with: pip install zstandard"
            )

        self.level = level
        # For compatibility with measure_compression.py interface
        self.bit_precision = 8  # Each compressed byte is 8 bits
        self.bits_for_encoding_count = 0  # No additional overhead per block

        # Create compressor and decompressor contexts
        self.compressor = zstd.ZstdCompressor(level=level)
        self.decompressor = zstd.ZstdDecompressor()

    def encode(self,
               tokens: List[int],
               initial_context: Optional[List[int]] = None,
               store_probs: bool = False,
               use_prefill: bool = False) -> Tuple[bytes, Dict]:
        """Encode tokens by converting to bytes and compressing with zstd.

        Args:
            tokens: List of token IDs to compress
            initial_context: Optional initial context (ignored for zstd)
            store_probs: Whether to store probabilities (ignored for zstd)
            use_prefill: Whether to use prefill mode (ignored for zstd)

        Returns:
            Tuple of (compressed_bytes, metadata_dict)
        """
        # Convert token IDs to bytes (using 4 bytes per int32)
        token_array = np.array(tokens, dtype=np.int32)
        token_bytes = token_array.tobytes()

        # Compress with zstd
        compressed = self.compressor.compress(token_bytes)

        # Metadata for decoder
        metadata = {
            'num_tokens': len(tokens),
            'original_bytes': len(token_bytes),
            'compressed_bytes': len(compressed)
        }

        return compressed, metadata

    def decode(self,
               encoded_data: Tuple[bytes, Dict],
               num_tokens: int,
               initial_context: Optional[List[int]] = None,
               store_probs: bool = False) -> List[int]:
        """Decode zstd-compressed tokens.

        Args:
            encoded_data: Tuple of (compressed_bytes, metadata_dict)
            num_tokens: Expected number of tokens (for verification)
            initial_context: Optional initial context (ignored for zstd)
            store_probs: Whether to store probabilities (ignored for zstd)

        Returns:
            List of decoded token IDs
        """
        compressed, metadata = encoded_data

        # Decompress
        decompressed_bytes = self.decompressor.decompress(compressed)

        # Convert back to token IDs
        token_array = np.frombuffer(decompressed_bytes, dtype=np.int32)
        tokens = token_array.tolist()

        # Verify length
        assert len(
            tokens
        ) == num_tokens, f"Expected {num_tokens} tokens, got {len(tokens)}"

        return tokens


class Lz4Compressor:
    """Baseline compressor using LZ4 compression (very fast, modest compression ratio)."""

    def __init__(self, level: int = 0):
        """Initialize LZ4 compressor.

        Args:
            level: Compression level (0-16, where 0 is fast and 16 is high compression)
                  Default 0 uses fast mode
        """
        try:
            import lz4.frame
            self.lz4_frame = lz4.frame
        except ImportError:
            raise ImportError(
                "lz4 package not found. Install with: pip install lz4")

        self.level = level
        # For compatibility with measure_compression.py interface
        self.bit_precision = 8
        self.bits_for_encoding_count = 0

    def encode(self,
               tokens: List[int],
               initial_context: Optional[List[int]] = None,
               store_probs: bool = False,
               use_prefill: bool = False) -> Tuple[bytes, Dict]:
        """Encode tokens by converting to bytes and compressing with LZ4."""
        token_array = np.array(tokens, dtype=np.int32)
        token_bytes = token_array.tobytes()

        # Compress with LZ4
        compressed = self.lz4_frame.compress(token_bytes,
                                             compression_level=self.level)

        metadata = {
            'num_tokens': len(tokens),
            'original_bytes': len(token_bytes),
            'compressed_bytes': len(compressed)
        }

        return compressed, metadata

    def decode(self,
               encoded_data: Tuple[bytes, Dict],
               num_tokens: int,
               initial_context: Optional[List[int]] = None,
               store_probs: bool = False) -> List[int]:
        """Decode LZ4-compressed tokens."""
        compressed, metadata = encoded_data
        decompressed_bytes = self.lz4_frame.decompress(compressed)
        token_array = np.frombuffer(decompressed_bytes, dtype=np.int32)
        tokens = token_array.tolist()
        assert len(
            tokens
        ) == num_tokens, f"Expected {num_tokens} tokens, got {len(tokens)}"
        return tokens


class BrotliCompressor:
    """Baseline compressor using Brotli (excellent for text/web content)."""

    def __init__(self, level: int = 11):
        """Initialize Brotli compressor.

        Args:
            level: Compression level (0-11, where 11 is maximum compression)
                  Default 11 is maximum for best text compression
        """
        try:
            import brotli
            self.brotli = brotli
        except ImportError:
            raise ImportError(
                "brotli package not found. Install with: pip install brotli")

        self.level = level
        # For compatibility with measure_compression.py interface
        self.bit_precision = 8
        self.bits_for_encoding_count = 0

    def encode(self,
               tokens: List[int],
               initial_context: Optional[List[int]] = None,
               store_probs: bool = False,
               use_prefill: bool = False) -> Tuple[bytes, Dict]:
        """Encode tokens by converting to bytes and compressing with Brotli."""
        token_array = np.array(tokens, dtype=np.int32)
        token_bytes = token_array.tobytes()

        # Compress with Brotli
        compressed = self.brotli.compress(token_bytes, quality=self.level)

        metadata = {
            'num_tokens': len(tokens),
            'original_bytes': len(token_bytes),
            'compressed_bytes': len(compressed)
        }

        return compressed, metadata

    def decode(self,
               encoded_data: Tuple[bytes, Dict],
               num_tokens: int,
               initial_context: Optional[List[int]] = None,
               store_probs: bool = False) -> List[int]:
        """Decode Brotli-compressed tokens."""
        compressed, metadata = encoded_data
        decompressed_bytes = self.brotli.decompress(compressed)
        token_array = np.frombuffer(decompressed_bytes, dtype=np.int32)
        tokens = token_array.tolist()
        assert len(
            tokens
        ) == num_tokens, f"Expected {num_tokens} tokens, got {len(tokens)}"
        return tokens


class LzmaCompressor:
    """Baseline compressor using LZMA/XZ (high compression ratio, slower)."""

    def __init__(self, preset: int = 6):
        """Initialize LZMA compressor.

        Args:
            preset: Compression preset (0-9, where 9 is maximum compression)
                   Default 6 is the standard xz default
        """
        self.preset = preset
        # For compatibility with measure_compression.py interface
        self.bit_precision = 8
        self.bits_for_encoding_count = 0

    def encode(self,
               tokens: List[int],
               initial_context: Optional[List[int]] = None,
               store_probs: bool = False,
               use_prefill: bool = False) -> Tuple[bytes, Dict]:
        """Encode tokens by converting to bytes and compressing with LZMA."""
        token_array = np.array(tokens, dtype=np.int32)
        token_bytes = token_array.tobytes()

        # Compress with LZMA
        compressed = lzma.compress(token_bytes,
                                   format=lzma.FORMAT_XZ,
                                   preset=self.preset)

        metadata = {
            'num_tokens': len(tokens),
            'original_bytes': len(token_bytes),
            'compressed_bytes': len(compressed)
        }

        return compressed, metadata

    def decode(self,
               encoded_data: Tuple[bytes, Dict],
               num_tokens: int,
               initial_context: Optional[List[int]] = None,
               store_probs: bool = False) -> List[int]:
        """Decode LZMA-compressed tokens."""
        compressed, metadata = encoded_data
        decompressed_bytes = lzma.decompress(compressed)
        token_array = np.frombuffer(decompressed_bytes, dtype=np.int32)
        tokens = token_array.tolist()
        assert len(
            tokens
        ) == num_tokens, f"Expected {num_tokens} tokens, got {len(tokens)}"
        return tokens


class Bzip2Compressor:
    """Baseline compressor using bzip2 (legacy but good text compression)."""

    def __init__(self, level: int = 9):
        """Initialize bzip2 compressor.

        Args:
            level: Compression level (1-9, where 9 is maximum compression)
                  Default 9 is maximum
        """
        self.level = level
        # For compatibility with measure_compression.py interface
        self.bit_precision = 8
        self.bits_for_encoding_count = 0

    def encode(self,
               tokens: List[int],
               initial_context: Optional[List[int]] = None,
               store_probs: bool = False,
               use_prefill: bool = False) -> Tuple[bytes, Dict]:
        """Encode tokens by converting to bytes and compressing with bzip2."""
        token_array = np.array(tokens, dtype=np.int32)
        token_bytes = token_array.tobytes()

        # Compress with bzip2
        compressed = bz2.compress(token_bytes, compresslevel=self.level)

        metadata = {
            'num_tokens': len(tokens),
            'original_bytes': len(token_bytes),
            'compressed_bytes': len(compressed)
        }

        return compressed, metadata

    def decode(self,
               encoded_data: Tuple[bytes, Dict],
               num_tokens: int,
               initial_context: Optional[List[int]] = None,
               store_probs: bool = False) -> List[int]:
        """Decode bzip2-compressed tokens."""
        compressed, metadata = encoded_data
        decompressed_bytes = bz2.decompress(compressed)
        token_array = np.frombuffer(decompressed_bytes, dtype=np.int32)
        tokens = token_array.tolist()
        assert len(
            tokens
        ) == num_tokens, f"Expected {num_tokens} tokens, got {len(tokens)}"
        return tokens


def create_baseline_compressor(method: str, level: Optional[int] = None):
    """Factory function to create a baseline compressor.

    Args:
        method: Compression method name
        level: Optional compression level (uses defaults if not specified)

    Returns:
        Compressor instance
    """
    if method == 'gzip':
        return GzipCompressor(level=level if level is not None else 9)
    elif method == 'zstd':
        return ZstdCompressor(level=level if level is not None else 3)
    elif method == 'zstd-fast':
        return ZstdCompressor(level=1)
    elif method == 'zstd-high':
        return ZstdCompressor(level=19)
    elif method == 'lz4':
        return Lz4Compressor(level=level if level is not None else 0)
    elif method == 'brotli':
        return BrotliCompressor(level=level if level is not None else 11)
    elif method == 'lzma' or method == 'xz':
        return LzmaCompressor(preset=level if level is not None else 6)
    elif method == 'bzip2':
        return Bzip2Compressor(level=level if level is not None else 9)
    else:
        raise ValueError(f"Unknown compression method: {method}")


def measure_baseline_compression(
        text: str,
        tokenizer,
        methods: List[str] = ['gzip', 'zstd']) -> Dict[str, Dict]:
    """Measure compression performance of baseline methods on text.

    This is a standalone function for quick baseline measurements without needing
    the full measure_compression.py infrastructure.

    Args:
        text: Text to compress
        tokenizer: Tokenizer to encode text
        methods: List of compression methods to test (gzip, zstd, lz4, brotli, lzma, bzip2, etc.)

    Returns:
        Dictionary mapping method name to results dict with keys:
            - compression_ratio: compressed_size / original_size
            - bits_per_token: compressed_bits / num_tokens
            - encode_time: encoding time in seconds
            - decode_time: decoding time in seconds
            - encode_time_per_token: encoding time per token
            - decode_time_per_token: decoding time per token
            - num_tokens: number of tokens
            - compressed_bytes: size of compressed data
            - original_bytes: size of original data
    """
    # Tokenize
    tokens = tokenizer.encode(text)

    results = {}

    for method in methods:
        try:
            compressor = create_baseline_compressor(method)
        except (ImportError, ValueError) as e:
            results[method] = {
                'success': False,
                'error': str(e),
                'num_tokens': len(tokens)
            }
            continue

        # Encode
        t0 = time.time()
        try:
            compressed, metadata = compressor.encode(tokens)
            encode_time = time.time() - t0
        except Exception as e:
            results[method] = {
                'success': False,
                'error': f"Encode failed: {e}",
                'num_tokens': len(tokens)
            }
            continue

        # Decode
        t0 = time.time()
        try:
            decoded_tokens = compressor.decode((compressed, metadata),
                                               len(tokens))
            decode_time = time.time() - t0
        except Exception as e:
            results[method] = {
                'success': False,
                'error': f"Decode failed: {e}",
                'num_tokens': len(tokens)
            }
            continue

        # Verify
        if decoded_tokens != tokens:
            results[method] = {
                'success': False,
                'error': f"{method} decode mismatch",
                'num_tokens': len(tokens)
            }
            continue

        # Calculate metrics
        num_tokens = len(tokens)
        compressed_bytes = len(compressed)
        original_bytes = len(tokens) * 4  # 4 bytes per int32
        compressed_bits = compressed_bytes * 8

        results[method] = {
            'compression_ratio':
            compressed_bytes / original_bytes if original_bytes > 0 else 0,
            'bits_per_token':
            compressed_bits / num_tokens if num_tokens > 0 else 0,
            'encode_time':
            encode_time,
            'decode_time':
            decode_time,
            'encode_time_per_token':
            encode_time / num_tokens if num_tokens > 0 else 0,
            'decode_time_per_token':
            decode_time / num_tokens if num_tokens > 0 else 0,
            'encode_tokens_per_sec':
            num_tokens / encode_time if encode_time > 0 else 0,
            'decode_tokens_per_sec':
            num_tokens / decode_time if decode_time > 0 else 0,
            'num_tokens':
            num_tokens,
            'compressed_bytes':
            compressed_bytes,
            'original_bytes':
            original_bytes,
            'success':
            True
        }

    return results
