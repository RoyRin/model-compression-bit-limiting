from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# PEFT for LoRA support
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

# Add repo root to path for local imports
_repo_root = Path(__file__).parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Local imports
try:
    from compression.block_coder import (
        BlockEmissionArithmeticCoder,
        BlockEmissionArithmeticDecoder,
        compare_encoder_decoder_probs,
    )
    from compression.utils.config import load_compression_config
    from scripts.utils.plotting import plot_lora_comparison
    from scripts.utils.dataset_loading import (
        load_text_and_tokens,
        load_hf_dataset,
        load_compression_dataset,
        process_dataset,
    )
    from scripts.utils.lora_rag import (
        query_lora_rag,
        compress_with_rag_lora,
        process_dataset_with_rag_lora,
        DEFAULT_RAG_INDEX_DIR,
        DEFAULT_RAG_EMBEDDING_MODEL,
        LORA_RAG_AVAILABLE,
        PEFT_AVAILABLE as LORA_PEFT_AVAILABLE,
    )
except ImportError as e:  # pragma: no cover
    print(f"❌ Unable to import local compression modules: {e}")
    raise

# ---------------------------------------------------------------------------
# Alias map
# ---------------------------------------------------------------------------

LLAMA_ALIASES: Dict[str, str] = {
    # Base models (for compression - better probability estimates on raw text)
    "1b": "meta-llama/Llama-3.2-1B",
    "3b": "meta-llama/Llama-3.2-3B",
    "8b": "meta-llama/Meta-Llama-3-8B",
    "70b": "meta-llama/Meta-Llama-3-70B",
    # Instruct models (if needed for chat-formatted data)
    "1b-instruct": "meta-llama/Llama-3.2-1B-Instruct",
    "3b-instruct": "meta-llama/Llama-3.2-3B-Instruct",
    "8b-instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "70b-instruct": "meta-llama/Meta-Llama-3-70B-Instruct",
    # Legacy Pythia aliases (kept for backwards compatibility)
    "pythia-70m": "EleutherAI/pythia-70m",
    "pythia-160m": "EleutherAI/pythia-160m",
    "pythia-410m": "EleutherAI/pythia-410m",
    "pythia-1b": "EleutherAI/pythia-1b",
    "pythia-1.4b": "EleutherAI/pythia-1.4b",
    # Mistral base (for Lots-of-LoRAs)
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
}

ModelPath = str

# ---------------------------------------------------------------------------
# Model‑path resolution helpers
# ---------------------------------------------------------------------------


def _locate_checkpoint_dir(root: Path,
                           preferred: Optional[str] = None) -> Path:
    """Return the actual model dir that holds *config.json* & weights.

    *root* may itself be the checkpoint or the *parent* directory containing
    ``checkpoint‑best``/``checkpoint‑final`` sub‑dirs.
    """
    if preferred:  # user asked for a specific name
        candidate = root / preferred
        if candidate.is_dir():
            return candidate
        raise FileNotFoundError(
            f"{candidate} not found for --checkpoint {preferred!r}")

    # Auto‑detect best → final → root fallback
    for sub in ("checkpoint-best", "checkpoint-final"):
        cand = root / sub
        if cand.is_dir():
            return cand
    return root  # assume root already contains the model files


def resolve_model_path(arg: str,
                       checkpoint_name: Optional[str] = None) -> ModelPath:
    """Translate *arg* into a path/identifier accepted by *from_pretrained*.

    • If *arg* matches an alias → mapped HF repo name.
    • If *arg* points to an existing dir/file → returns a path to the proper
      checkpoint directory (using *checkpoint_name* or auto‑detect).
    • Otherwise returns *arg* unchanged (assumed HF repo).
    """
    # Alias?
    if arg.lower() in LLAMA_ALIASES:
        return LLAMA_ALIASES[arg.lower()]

    p = Path(arg).expanduser()
    if p.exists():
        if p.is_file():
            return str(p)  # rare – single‑file checkpoint
        # Directory – locate correct checkpoint inside
        return str(_locate_checkpoint_dir(p, checkpoint_name))

    # Fallback: treat as repo name (e.g. meta-llama/Meta-Llama-3-8B-Instruct)
    return arg


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------


def load_model(model_path: str,
               lora_adapter: Optional[str] = None,
               quantization: Optional[str] = None,
               dtype: str = "fp32"):
    """Load model with optional LoRA adapter support.

    Args:
        model_path: Path to base model or full model
        lora_adapter: Optional path to LoRA adapter (e.g., "Lots-of-LoRAs/Mistral-7B-Instruct-v0.2-4b-r63-task1431")
        quantization: Optional quantization method ("4bit" or "8bit")
        dtype: Data type for model weights ("fp32", "bf16", "fp16")

    Returns:
        model, tokenizer tuple
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if lora_adapter:
        if not PEFT_AVAILABLE:
            raise ImportError(
                "PEFT is required for LoRA support. Install with: pip install peft"
            )

        print(f"[{ts}] Loading base model ▶ {model_path}")
        print(f"[{ts}] Will apply LoRA adapter ▶ {lora_adapter}")
    else:
        print(f"[{ts}] Loading model ▶ {model_path}")

    # Handle quantization
    model_kwargs = {}
    if quantization:
        from transformers import BitsAndBytesConfig
        if quantization == "4bit":
            print(f"[{ts}] Using 4-bit quantization")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4")
            model_kwargs['quantization_config'] = bnb_config
            model_kwargs['device_map'] = 'auto'
        elif quantization == "8bit":
            print(f"[{ts}] Using 8-bit quantization")
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            model_kwargs['quantization_config'] = bnb_config
            model_kwargs['device_map'] = 'auto'
    else:
        # Set dtype when not using quantization
        dtype_map = {
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }
        if dtype in dtype_map:
            model_kwargs['torch_dtype'] = dtype_map[dtype]
            print(f"[{ts}] Using dtype: {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

    # Apply LoRA adapter if specified
    if lora_adapter:
        print(f"[{ts}] Applying LoRA adapter...")
        model = PeftModel.from_pretrained(model, lora_adapter)
        print(f"[{ts}] LoRA adapter applied successfully")

    # Move to GPU if not using quantization (which handles device placement)
    if torch.cuda.is_available() and 'device_map' not in model_kwargs:
        model = model.cuda()

    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[{ts}] Model ready — params: {n_params:,} — architecture: {model.config.model_type}"
    )

    # Print precision and attention configuration for debugging
    print(f"\n[MODEL CONFIGURATION DEBUG]")
    print(f"  Model dtype: {next(model.parameters()).dtype}")
    if hasattr(model.config, '_attn_implementation'):
        print(
            f"  Attention implementation: {model.config._attn_implementation}")
    elif hasattr(model.config, 'attn_implementation'):
        print(
            f"  Attention implementation: {model.config.attn_implementation}")
    else:
        print(
            f"  Attention implementation: (not explicitly set - HF auto-select)"
        )
    if hasattr(model.config, 'use_cache'):
        print(f"  Config use_cache: {model.config.use_cache}")
    # Check for flash attention
    try:
        print(
            f"  CUDA Flash SDP enabled: {torch.backends.cuda.flash_sdp_enabled()}"
        )
        print(
            f"  CUDA Mem-efficient SDP enabled: {torch.backends.cuda.mem_efficient_sdp_enabled()}"
        )
        print(
            f"  CUDA Math SDP enabled: {torch.backends.cuda.math_sdp_enabled()}"
        )
    except:
        pass
    # Check if flash_attn package is available
    try:
        import flash_attn
        print(f"  flash_attn package version: {flash_attn.__version__}")
    except ImportError:
        print(f"  flash_attn package: NOT INSTALLED")

    return model, tokenizer


# ---------------------------------------------------------------------------
# Compression core
# ---------------------------------------------------------------------------


def compress(tokens: List[int],
             encoder,
             decoder,
             coder_type: str,
             tokenizer=None,
             save_plot: str = None,
             use_prefill: bool = False,
             initial_context: Optional[List[int]] = None,
             plots_dir: Optional[Path] = None,
             compare_probs: bool = False) -> Dict:
    """Compress and round-trip tokens using the provided encoder/decoder."""
    store_probs = save_plot is not None or compare_probs
    if store_probs and hasattr(encoder, 'stored_probs'):
        encoder.stored_probs = []
    if store_probs and hasattr(decoder, 'stored_probs'):
        decoder.stored_probs = []

    # Encode
    t0 = time.time()
    enc_buf, enc_info = encoder.encode(tokens,
                                       initial_context=initial_context,
                                       store_probs=store_probs,
                                       use_prefill=use_prefill)
    enc_t = time.time() - t0

    # Decode
    t0 = time.time()
    decode_result = decoder.decode((enc_buf, enc_info),
                                   len(tokens),
                                   initial_context=initial_context,
                                   store_probs=store_probs)
    dec_t = time.time() - t0

    # Handle new return format: (decoded_tokens, stored_probs) when store_probs=True
    if store_probs and isinstance(decode_result, tuple):
        dec_tokens, _ = decode_result  # Probs also stored in decoder.stored_probs
    else:
        dec_tokens = decode_result

    ok = dec_tokens == tokens

    # Optional probability comparison with plotting
    if compare_probs and hasattr(encoder, 'stored_probs') and hasattr(
            decoder, 'stored_probs'):
        enc_probs = encoder.stored_probs
        dec_probs = decoder.stored_probs
        print(
            f"[DEBUG] compare_probs={compare_probs}, len(enc_probs)={len(enc_probs)}, len(dec_probs)={len(dec_probs)}"
        )
        if enc_probs and dec_probs:
            # Compute prompt_length for vertical line at prompt boundary
            prompt_length = len(initial_context) if initial_context else 0

            # Create plot directory if needed
            if plots_dir is not None:
                plots_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = str(plots_dir /
                                f"encoder_decoder_probs_{timestamp}.png")
            else:
                save_path = None

            print(
                f"[DEBUG] About to call compare_encoder_decoder_probs, save_path={save_path}"
            )
            print(
                f"[DEBUG] len(enc_probs)={len(enc_probs)}, len(dec_probs)={len(dec_probs)}"
            )
            # Check first few distributions
            for i in range(min(3, len(enc_probs))):
                enc_top5 = enc_probs[i].topk(5)
                dec_top5 = dec_probs[i].topk(5)
                l2_diff = torch.sqrt(
                    ((enc_probs[i] - dec_probs[i])**2).sum()).item()
                print(f"[DEBUG] Position {i}: L2={l2_diff:.6f}")
                print(
                    f"  Enc top-5: tokens={enc_top5.indices.tolist()}, probs={[f'{p:.4f}' for p in enc_top5.values.tolist()]}"
                )
                print(
                    f"  Dec top-5: tokens={dec_top5.indices.tolist()}, probs={[f'{p:.4f}' for p in dec_top5.values.tolist()]}"
                )
            try:
                # Use the plotting function
                comparison_results = compare_encoder_decoder_probs(
                    enc_probs,
                    dec_probs,
                    prompt_length=prompt_length,
                    save_path=save_path,
                    title=
                    f"Encoder vs Decoder Probs (prompt_len={prompt_length})")
                print(
                    f"Prob comparison: L2 mean={comparison_results['l2_mean']:.6e}, "
                    f"max={comparison_results['l2_max']:.6e}")
            except Exception as e:
                print(f"[DEBUG] compare_encoder_decoder_probs failed: {e}")
                import traceback
                traceback.print_exc()

    # Log mismatch details
    if not ok:
        print(
            f"DECODE MISMATCH: orig_len={len(tokens)}, dec_len={len(dec_tokens)}"
        )
        for i in range(min(len(tokens), len(dec_tokens))):
            if tokens[i] != dec_tokens[i]:
                print(
                    f"  First mismatch at {i}: expected {tokens[i]}, got {dec_tokens[i]}"
                )
                break

    # Compute stats
    vocab_size = tokenizer.vocab_size
    orig_bits = len(tokens) * np.log2(vocab_size)
    bits_per_enc = 1 if coder_type == "incremental" else (
        encoder.bit_precision + encoder.bits_for_encoding_count)
    enc_bits = len(enc_buf) * bits_per_enc
    num_tokens = len(tokens)

    return {
        "success": ok,
        "compression_ratio": enc_bits / orig_bits if ok else None,
        "bits_per_token": enc_bits / num_tokens if ok else None,
        "encode_time": enc_t,
        "decode_time": dec_t,
        "encode_time_per_token": enc_t / num_tokens if num_tokens else 0,
        "decode_time_per_token": dec_t / num_tokens if num_tokens else 0,
        "encode_tokens_per_sec": num_tokens / enc_t if enc_t else 0,
        "decode_tokens_per_sec": num_tokens / dec_t if dec_t else 0,
        "num_tokens": num_tokens,
        "encoded_bits": enc_bits,
        "original_bits": orig_bits,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_PMATIC_DELTA = 1e-3
DEFAULT_PMATIC_R = 0.005


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compress a file with a language model arithmetic coder.")
    p.add_argument(
        "model",
        help=
        "Alias (1b, 3b, 8b use BASE models; mistral for Mistral-7B), HF repo, or local model directory"
    )
    p.add_argument(
        "file",
        help=
        "Path to .txt, JSON, YAML dataset, or HuggingFace dataset name (e.g., 'Lots-of-LoRAs/task561_alt_translation_en_bg')"
    )
    p.add_argument(
        "--method",
        choices=["block"],
        default="block",
        help="Which arithmetic coder to run (only 'block' currently supported)"
    )
    p.add_argument(
        "--checkpoint",
        metavar="NAME",
        default=None,
        help=
        "If *model* is a dir, choose a specific sub‑directory (e.g. checkpoint-best)"
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help=
        "Path to YAML config file for compression parameters (default: use built-in defaults)"
    )
    p.add_argument(
        "--lora-adapter",
        metavar="PATH",
        default=None,
        help=
        "LoRA adapter to apply (e.g., 'Lots-of-LoRAs/Mistral-7B-Instruct-v0.2-4b-r16-task1431')"
    )
    p.add_argument(
        "--compare-loras",
        nargs="+",
        metavar="TASK_ID",
        help=
        "Compare multiple LoRAs on same dataset (provide task IDs, e.g., 'task561 task1431')"
    )
    p.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="LoRA rank for auto-generated adapter names (default: 16)")
    p.add_argument(
        "--lora-bits",
        type=int,
        default=4,
        choices=[4],
        help="LoRA bits for auto-generated adapter names (default: 4)")
    p.add_argument("--quantization",
                   choices=["4bit", "8bit"],
                   default=None,
                   help="Quantization method for model loading")
    p.add_argument("--dtype",
                   choices=["fp32", "bf16", "fp16"],
                   default="fp32",
                   help="Data type for model weights (default: fp32)")
    p.add_argument("--fast",
                   action="store_true",
                   help="Skip rank/top‑k tracking for speed")
    p.add_argument("--verbose",
                   action="store_true",
                   help="Enable verbose output for debugging")
    p.add_argument("--debug",
                   action="store_true",
                   help="Enable debug output from probability generator")
    p.add_argument(
        "--use-pmatic",
        action="store_true",
        help="Use PMATIC helper bits + quantized probabilities for robustness")
    p.add_argument("--pmatic-r",
                   type=float,
                   default=DEFAULT_PMATIC_R,
                   help="PMATIC bin half-width parameter (default: 0.1)")
    p.add_argument(
        "--pmatic-delta",
        type=float,
        default=DEFAULT_PMATIC_DELTA,
        help=
        "PMATIC tolerance parameter (default: 0.02, must satisfy r > 2*delta)")
    p.add_argument("--compare-probs",
                   action="store_true",
                   help="Compare encoder/decoder probabilities and plot")
    p.add_argument(
        "--use-prefill",
        action="store_true",
        help=
        "Use prefill mode for encoding (faster but may differ from decoder)")
    p.add_argument(
        "--max-tokens",
        "-N",
        type=int,
        default=None,
        help="Maximum number of tokens to encode (default: encode all)")
    p.add_argument(
        "--dataset-mode",
        action="store_true",
        help="Treat input as compression dataset YAML (uses prompts as context)"
    )
    p.add_argument(
        "--hf-dataset",
        action="store_true",
        help="Load dataset from HuggingFace (file arg is dataset name)")
    p.add_argument(
        "--dataset-split",
        default="test",
        choices=["train", "test", "valid"],
        help="Split to use for HuggingFace datasets (default: test)")
    p.add_argument(
        "--limit-samples",
        type=int,
        default=None,
        metavar="N",
        help="Limit to first N samples from dataset (useful for testing)")
    p.add_argument("--output-json",
                   metavar="PATH",
                   type=Path,
                   help="Write stats to this JSON file")

    # RAG-based LoRA selection arguments
    p.add_argument(
        "--use-rag-lora",
        action="store_true",
        help="Use RAG to automatically select the best LoRA for the input text"
    )
    p.add_argument(
        "--rag-index-dir",
        type=str,
        default=DEFAULT_RAG_INDEX_DIR,
        help=
        f"Directory containing RAG index (default: {DEFAULT_RAG_INDEX_DIR})")
    p.add_argument("--rag-embedding-model",
                   type=str,
                   default=DEFAULT_RAG_EMBEDDING_MODEL,
                   choices=["qwen", "gemma", "nemotron"],
                   help="Embedding model for RAG queries (default: qwen)")
    p.add_argument(
        "--rag-top-k",
        type=int,
        default=25,
        help=
        "Number of RAG candidates to evaluate with perplexity (default: 25)")

    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main(argv: List[str] | None = None):
    args = parse_args(argv)
    model_path = resolve_model_path(args.model, args.checkpoint)

    # Load compression configuration
    config = load_compression_config(args.config)
    comp_cfg = config["compression"]

    if args.config:
        print(
            f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Loaded config from {args.config}"
        )

    # Auto-detect if file argument is a HuggingFace dataset
    is_hf_dataset = args.hf_dataset or (isinstance(args.file, str) and
                                        ('/' in args.file
                                         and not Path(args.file).exists()))

    # Handle --use-rag-lora mode
    if args.use_rag_lora:
        if not LORA_RAG_AVAILABLE:
            print(
                "Error: --use-rag-lora requires langchain-community and pyyaml."
            )
            print("Install with: pip install langchain-community pyyaml")
            return

        file_path = Path(args.file)
        is_yaml_dataset = file_path.suffix.lower() in ['.yaml', '.yml']

        # YAML dataset mode: per-sample RAG queries
        if is_yaml_dataset:
            print(f"\n{'='*80}")
            print("RAG-BASED LoRA COMPRESSION (YAML DATASET MODE)")
            print(f"{'='*80}")
            print(f"Dataset: {args.file}")
            print(f"RAG Index: {args.rag_index_dir}")
            print(f"Embedding Model: {args.rag_embedding_model}")
            print(f"{'='*80}\n")

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path)

            # Load YAML dataset
            samples, metadata = load_compression_dataset(
                file_path, tokenizer, limit=args.limit_samples)

            # Create run directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_base_dir = Path(
                __file__).parent.parent / "data" / "results"
            run_dir = results_base_dir / f"rag_lora_run_{timestamp}"

            # Process with two-stage RAG + perplexity selection
            results = process_dataset_with_rag_lora(
                samples=samples,
                metadata=metadata,
                base_model_path=model_path,
                compression_config=comp_cfg,
                rag_index_dir=args.rag_index_dir,
                rag_embedding_model=args.rag_embedding_model,
                lora_bits=args.lora_bits,
                lora_rank=args.lora_rank,
                quantization=args.quantization,
                verbose=args.verbose,
                use_prefill=args.use_prefill,
                run_dir=run_dir,
                rag_top_k=args.rag_top_k,
            )

            # Save to JSON if requested
            if args.output_json:
                import json
                with open(args.output_json, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                print(f"✓ Saved results to {args.output_json}")

            return

        # Single file mode: use text to query RAG once
        print(f"\n{'='*80}")
        print("RAG-BASED LoRA COMPRESSION MODE")
        print(f"{'='*80}")
        print(f"RAG Index: {args.rag_index_dir}")
        print(f"Embedding Model: {args.rag_embedding_model}")
        print(f"{'='*80}\n")

        # Load tokenizer first (needed for compress_with_rag_lora)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Load text from file
        if file_path.suffix.lower() == ".json":
            text, tokens = _load_from_json(file_path)
        else:
            text = file_path.read_text(encoding="utf-8")

        # Truncate if needed
        if args.max_tokens:
            tokens = tokenizer.encode(text, add_special_tokens=True)
            if len(tokens) > args.max_tokens:
                tokens = tokens[:args.max_tokens]
                text = tokenizer.decode(tokens)
                print(f"Truncated to {len(tokens)} tokens")

        # Run RAG-based compression
        result = compress_with_rag_lora(
            text=text,
            base_model_path=model_path,
            tokenizer=tokenizer,
            compression_config=comp_cfg,
            rag_index_dir=args.rag_index_dir,
            rag_embedding_model=args.rag_embedding_model,
            lora_bits=args.lora_bits,
            lora_rank=args.lora_rank,
            quantization=args.quantization,
            verbose=args.verbose,
            use_prefill=args.use_prefill,
        )

        # Save results if requested
        if args.output_json:
            import json
            with open(args.output_json, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"\n✓ Saved results to {args.output_json}")

        return

    # Handle --compare-loras mode
    if args.compare_loras:
        if not is_hf_dataset:
            print(
                "Error: --compare-loras requires a HuggingFace dataset (use --hf-dataset)"
            )
            return

        # Create run directory for comparison
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_base_dir = Path(__file__).parent.parent / "data" / "results"
        run_dir = results_base_dir / f"run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n✓ Created run directory: {run_dir}\n")

        # Extract dataset task ID
        dataset_task_id = args.file.split('/')[-1].split(
            '_')[0] if '/' in args.file else None
        print(f"\n{'='*80}")
        print(f"Comparing multiple LoRAs on dataset: {args.file}")
        print(f"Dataset task: {dataset_task_id}")
        print(
            f"Testing baseline + LoRAs: baseline, {', '.join(args.compare_loras)}"
        )
        print(f"{'='*80}\n")

        all_results = []

        # First, run baseline (no LoRA)
        print(f"\n{'='*80}")
        print(f"Testing BASELINE (No LoRA)")
        print(f"{'='*80}\n")

        load_start = time.time()
        model, tokenizer = load_model(model_path,
                                      lora_adapter=None,
                                      quantization=args.quantization,
                                      dtype=args.dtype)
        load_time = time.time() - load_start
        print(f"Model loading took: {load_time:.2f}s\n")

        block_enc = BlockEmissionArithmeticCoder(
            model,
            tokenizer,
            bit_precision=comp_cfg["bit_precision"],
            bits_for_encoding_count=comp_cfg["bits_for_encoding_count"],
            min_prob=comp_cfg["min_prob"],
            temperature=comp_cfg["temperature"],
            verbose=args.verbose,
            track_token_ranks=not args.fast,
            debug=args.debug,
            use_pmatic=args.use_pmatic,
            pmatic_r=args.pmatic_r,
            pmatic_delta=args.pmatic_delta,
        )
        block_dec = BlockEmissionArithmeticDecoder(
            model,
            tokenizer,
            bit_precision=comp_cfg["bit_precision"],
            bits_for_encoding_count=comp_cfg["bits_for_encoding_count"],
            min_prob=comp_cfg["min_prob"],
            temperature=comp_cfg["temperature"],
            verbose=args.verbose,
            debug=args.debug,
            use_pmatic=args.use_pmatic,
            pmatic_r=args.pmatic_r,
            pmatic_delta=args.pmatic_delta,
        )

        samples, dataset_metadata = load_hf_dataset(args.file,
                                                    tokenizer,
                                                    split=args.dataset_split,
                                                    limit=args.limit_samples)
        baseline_results = process_dataset(samples,
                                           dataset_metadata,
                                           block_enc,
                                           block_dec,
                                           tokenizer,
                                           args,
                                           compress_fn=compress,
                                           run_dir=run_dir,
                                           run_name="baseline")

        del model, block_enc, block_dec
        import torch
        torch.cuda.empty_cache()

        all_results.append({
            'lora_name': 'baseline',
            'lora_task': 'baseline',
            'is_matching': False,
            'results': baseline_results,
        })

        # Now test each LoRA
        for task_id in args.compare_loras:
            lora_adapter = f"Lots-of-LoRAs/Mistral-7B-Instruct-v0.2-{args.lora_bits}b-r{args.lora_rank}-{task_id}"
            is_matching = (task_id == dataset_task_id)

            print(f"\n{'='*80}")
            print(
                f"Testing LoRA: {task_id} {'(MATCHING TASK)' if is_matching else '(DIFFERENT TASK)'}"
            )
            print(f"{'='*80}\n")

            # Load model with this LoRA
            load_start = time.time()
            model, tokenizer = load_model(model_path,
                                          lora_adapter=lora_adapter,
                                          quantization=args.quantization,
                                          dtype=args.dtype)
            load_time = time.time() - load_start
            print(f"Model loading took: {load_time:.2f}s\n")

            # Build encoder/decoder
            block_enc = BlockEmissionArithmeticCoder(
                model,
                tokenizer,
                bit_precision=comp_cfg["bit_precision"],
                bits_for_encoding_count=comp_cfg["bits_for_encoding_count"],
                min_prob=comp_cfg["min_prob"],
                temperature=comp_cfg["temperature"],
                verbose=args.verbose,
                track_token_ranks=not args.fast,
                debug=args.debug,
            )
            block_dec = BlockEmissionArithmeticDecoder(
                model,
                tokenizer,
                bit_precision=comp_cfg["bit_precision"],
                bits_for_encoding_count=comp_cfg["bits_for_encoding_count"],
                min_prob=comp_cfg["min_prob"],
                temperature=comp_cfg["temperature"],
                verbose=args.verbose,
                debug=args.debug,
                use_pmatic=args.use_pmatic,
                pmatic_r=args.pmatic_r,
                pmatic_delta=args.pmatic_delta,
            )

            # Load dataset
            samples, dataset_metadata = load_hf_dataset(
                args.file,
                tokenizer,
                split=args.dataset_split,
                limit=args.limit_samples)

            # Run compression
            lora_results = process_dataset(samples,
                                           dataset_metadata,
                                           block_enc,
                                           block_dec,
                                           tokenizer,
                                           args,
                                           compress_fn=compress,
                                           run_dir=run_dir,
                                           run_name=f"lora_{task_id}")

            # Clean up model to free GPU memory
            del model, block_enc, block_dec
            import torch
            torch.cuda.empty_cache()

            all_results.append({
                'lora_name': task_id,
                'lora_task': task_id,
                'is_matching': is_matching,
                'results': lora_results,
            })

        # Print summary
        print(f"\n{'='*80}")
        print("LoRA Comparison Summary")
        print(f"{'='*80}")
        print(f"Dataset: {args.file}")
        print(f"Split: {args.dataset_split}")
        print(f"Dataset task: {dataset_task_id}")
        print(f"\nResults:")

        for lora_result in all_results:
            lora_name = lora_result['lora_name']
            is_matching = lora_result['is_matching']
            results = lora_result['results']
            successful = [r for r in results if r['success']]

            if successful:
                avg_compression = sum(1.0 / r['compression_ratio']
                                      for r in successful) / len(successful)
                avg_bpt = sum(r['bits_per_token']
                              for r in successful) / len(successful)
                match_str = " (MATCHING)" if is_matching else ""
                print(f"  {lora_name}{match_str}:")
                print(f"    - Compression: {avg_compression:.2f}x")
                print(f"    - Bits/token: {avg_bpt:.2f}")
                print(f"    - Success: {len(successful)}/{len(results)}")
            else:
                print(f"  {lora_name}: No successful compressions")

        print(f"{'='*80}\n")

        # Save comparison summary YAML
        import yaml
        comparison_yaml_path = run_dir / "comparison_summary.yaml"

        comparison_summary = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'dataset': args.file,
                'split': args.dataset_split,
                'dataset_task': dataset_task_id,
                'base_model': model_path,
                'quantization': args.quantization,
                'num_samples': len(samples),
                'loras_tested': [r['lora_name'] for r in all_results],
            },
            'results': []
        }

        for lora_result in all_results:
            successful = [r for r in lora_result['results'] if r['success']]
            if successful:
                comparison_summary['results'].append({
                    'lora_name':
                    lora_result['lora_name'],
                    'is_matching':
                    lora_result['is_matching'],
                    'avg_compression_ratio':
                    float(
                        sum(1.0 / r['compression_ratio']
                            for r in successful) / len(successful)),
                    'avg_bits_per_token':
                    float(
                        sum(r['bits_per_token']
                            for r in successful) / len(successful)),
                    'num_successful':
                    len(successful),
                    'num_total':
                    len(lora_result['results']),
                    'success_rate':
                    float(len(successful) / len(lora_result['results'])),
                })

        with open(comparison_yaml_path, 'w') as f:
            yaml.dump(comparison_summary,
                      f,
                      default_flow_style=False,
                      sort_keys=False)

        print(f"✓ Saved comparison summary to {comparison_yaml_path}\n")

        # Generate comparison plot
        plots_dir = run_dir / "plots"
        plot_lora_comparison(all_results, plots_dir, args.file)

        print(f"\n{'='*80}")
        print("Comparison Complete!")
        print(f"Tested {len(all_results)} models on {args.file}")
        print(f"Results saved to: {run_dir}")
        print(f"{'='*80}\n")
        return

    # Regular mode: single LoRA or no LoRA
    # Time model loading separately
    load_start = time.time()
    model, tokenizer = load_model(model_path,
                                  lora_adapter=args.lora_adapter,
                                  quantization=args.quantization,
                                  dtype=args.dtype)
    load_time = time.time() - load_start
    print(f"Model loading took: {load_time:.2f}s\n")

    # Build coder(s) using config values
    block_enc = BlockEmissionArithmeticCoder(
        model,
        tokenizer,
        bit_precision=comp_cfg["bit_precision"],
        bits_for_encoding_count=comp_cfg["bits_for_encoding_count"],
        min_prob=comp_cfg["min_prob"],
        temperature=comp_cfg["temperature"],
        verbose=args.verbose,
        track_token_ranks=not args.fast,
        use_pmatic=args.use_pmatic,
        pmatic_r=args.pmatic_r,
        pmatic_delta=args.pmatic_delta,
    )
    block_dec = BlockEmissionArithmeticDecoder(
        model,
        tokenizer,
        bit_precision=comp_cfg["bit_precision"],
        bits_for_encoding_count=comp_cfg["bits_for_encoding_count"],
        min_prob=comp_cfg["min_prob"],
        temperature=comp_cfg["temperature"],
        verbose=args.verbose,
        use_pmatic=args.use_pmatic,
        pmatic_r=args.pmatic_r,
        pmatic_delta=args.pmatic_delta,
    )

    # Check if HuggingFace dataset mode
    if is_hf_dataset:
        samples, dataset_metadata = load_hf_dataset(args.file,
                                                    tokenizer,
                                                    split=args.dataset_split,
                                                    limit=args.limit_samples)

        # Create run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_base_dir = Path(__file__).parent.parent / "data" / "results"
        run_dir = results_base_dir / f"run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n✓ Created run directory: {run_dir}\n")

        # If using LoRA, also run baseline for comparison
        if args.lora_adapter:
            print(f"\n{'='*80}")
            print("Running baseline for comparison...")
            print(f"{'='*80}\n")

            # Load baseline model (no LoRA)
            baseline_model, baseline_tokenizer = load_model(
                model_path,
                lora_adapter=None,
                quantization=args.quantization,
                dtype=args.dtype)
            baseline_enc = BlockEmissionArithmeticCoder(
                baseline_model,
                baseline_tokenizer,
                bit_precision=comp_cfg["bit_precision"],
                bits_for_encoding_count=comp_cfg["bits_for_encoding_count"],
                min_prob=comp_cfg["min_prob"],
                temperature=comp_cfg["temperature"],
                verbose=args.verbose,
                track_token_ranks=not args.fast,
                use_pmatic=args.use_pmatic,
                pmatic_r=args.pmatic_r,
                pmatic_delta=args.pmatic_delta,
            )
            baseline_dec = BlockEmissionArithmeticDecoder(
                baseline_model,
                baseline_tokenizer,
                bit_precision=comp_cfg["bit_precision"],
                bits_for_encoding_count=comp_cfg["bits_for_encoding_count"],
                min_prob=comp_cfg["min_prob"],
                temperature=comp_cfg["temperature"],
                verbose=args.verbose,
                use_pmatic=args.use_pmatic,
                pmatic_r=args.pmatic_r,
                pmatic_delta=args.pmatic_delta,
            )

            baseline_results = process_dataset(samples,
                                               dataset_metadata,
                                               baseline_enc,
                                               baseline_dec,
                                               baseline_tokenizer,
                                               args,
                                               compress_fn=compress,
                                               run_dir=run_dir,
                                               run_name="baseline")

            del baseline_model, baseline_enc, baseline_dec
            import torch
            torch.cuda.empty_cache()

            print(f"\n{'='*80}")
            print("Running with LoRA...")
            print(f"{'='*80}\n")

        run_name = "lora" if args.lora_adapter else "compression"
        lora_results = process_dataset(samples,
                                       dataset_metadata,
                                       block_enc,
                                       block_dec,
                                       tokenizer,
                                       args,
                                       compress_fn=compress,
                                       run_dir=run_dir,
                                       run_name=run_name)

        # Print comparison if we ran baseline
        if args.lora_adapter:
            print(f"\n{'='*80}")
            print("Baseline vs LoRA Comparison")
            print(f"{'='*80}")
            print(f"Dataset: {args.file}")
            print(f"Split: {args.dataset_split} (evaluation on test set)")

            baseline_successful = [r for r in baseline_results if r['success']]
            lora_successful = [r for r in lora_results if r['success']]

            if baseline_successful:
                baseline_avg_compression = sum(
                    1.0 / r['compression_ratio']
                    for r in baseline_successful) / len(baseline_successful)
                baseline_avg_bpt = sum(
                    r['bits_per_token']
                    for r in baseline_successful) / len(baseline_successful)
                print(f"\nBaseline (No LoRA):")
                print(f"  - Compression: {baseline_avg_compression:.2f}x")
                print(f"  - Bits/token: {baseline_avg_bpt:.2f}")
                print(
                    f"  - Success: {len(baseline_successful)}/{len(baseline_results)}"
                )

            if lora_successful:
                lora_avg_compression = sum(
                    1.0 / r['compression_ratio']
                    for r in lora_successful) / len(lora_successful)
                lora_avg_bpt = sum(
                    r['bits_per_token']
                    for r in lora_successful) / len(lora_successful)

                # Extract LoRA task ID
                lora_task = args.lora_adapter.split(
                    '-')[-1] if '-' in args.lora_adapter else 'unknown'
                dataset_task = args.file.split('/')[-1].split(
                    '_')[0] if '/' in args.file else 'unknown'
                is_matching = (lora_task == dataset_task)

                print(
                    f"\nLoRA ({lora_task}){'  [MATCHING TASK]' if is_matching else ''}:"
                )
                print(f"  - Compression: {lora_avg_compression:.2f}x")
                print(f"  - Bits/token: {lora_avg_bpt:.2f}")
                print(
                    f"  - Success: {len(lora_successful)}/{len(lora_results)}")

                if baseline_successful:
                    improvement_compression = (
                        (lora_avg_compression - baseline_avg_compression) /
                        baseline_avg_compression) * 100
                    improvement_bpt = ((baseline_avg_bpt - lora_avg_bpt) /
                                       baseline_avg_bpt) * 100
                    print(f"\nImprovement over baseline:")
                    print(f"  - Compression: {improvement_compression:+.1f}%")
                    print(f"  - Bits/token: {improvement_bpt:+.1f}%")

            print(f"{'='*80}\n")

            # Save comparison summary YAML
            import yaml
            comparison_yaml_path = run_dir / "comparison_summary.yaml"

            comparison_summary = {
                'metadata': {
                    'created_at': datetime.now().isoformat(),
                    'dataset': args.file,
                    'split': args.dataset_split,
                    'dataset_task': dataset_task,
                    'base_model': model_path,
                    'lora_adapter': args.lora_adapter,
                    'quantization': args.quantization,
                    'num_samples': len(samples),
                },
                'results': {
                    'baseline': {
                        'avg_compression_ratio':
                        baseline_avg_compression
                        if baseline_successful else None,
                        'avg_bits_per_token':
                        baseline_avg_bpt if baseline_successful else None,
                        'num_successful':
                        len(baseline_successful),
                        'num_total':
                        len(baseline_results),
                        'success_rate':
                        len(baseline_successful) /
                        len(baseline_results) if baseline_results else 0,
                    },
                    'lora': {
                        'lora_task':
                        lora_task,
                        'is_matching':
                        is_matching,
                        'avg_compression_ratio':
                        lora_avg_compression if lora_successful else None,
                        'avg_bits_per_token':
                        lora_avg_bpt if lora_successful else None,
                        'num_successful':
                        len(lora_successful),
                        'num_total':
                        len(lora_results),
                        'success_rate':
                        len(lora_successful) /
                        len(lora_results) if lora_results else 0,
                    }
                }
            }

            if baseline_successful and lora_successful:
                improvement_compression = (
                    (lora_avg_compression - baseline_avg_compression) /
                    baseline_avg_compression) * 100
                improvement_bpt = (
                    (baseline_avg_bpt - lora_avg_bpt) / baseline_avg_bpt) * 100
                comparison_summary['improvement'] = {
                    'compression_ratio_percent':
                    float(improvement_compression),
                    'bits_per_token_percent': float(improvement_bpt),
                }

            with open(comparison_yaml_path, 'w') as f:
                yaml.dump(comparison_summary,
                          f,
                          default_flow_style=False,
                          sort_keys=False)

            print(f"✓ Saved comparison summary to {comparison_yaml_path}")

            # Generate comparison plot
            plots_dir = run_dir / "plots"
            comparison_data = [{
                'lora_name': 'baseline',
                'is_matching': False,
                'results': baseline_results,
            }, {
                'lora_name': lora_task,
                'is_matching': is_matching,
                'results': lora_results,
            }]
            plot_lora_comparison(comparison_data, plots_dir, args.file)

        return

    # Check if local YAML dataset mode
    file_path = Path(args.file)
    is_yaml_dataset = args.dataset_mode or (file_path.suffix.lower()
                                            in ['.yaml', '.yml'])

    if is_yaml_dataset:
        # Dataset mode: process each sample with prompt as context
        samples, dataset_metadata = load_compression_dataset(
            file_path, tokenizer, limit=args.limit_samples)

        # Create run directory for results and plots
        results_base_dir = Path("data/results")
        results_base_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = results_base_dir / f"run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        process_dataset(samples,
                        dataset_metadata,
                        block_enc,
                        block_dec,
                        tokenizer,
                        args,
                        compress_fn=compress,
                        run_dir=run_dir)
        return

    # Regular mode: Load text & tokens
    text, tokens = load_text_and_tokens(args.file, tokenizer)
    print(
        f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Loaded {len(tokens):,} tokens from {args.file}"
    )

    # Truncate to max_tokens if specified
    if args.max_tokens is not None and len(tokens) > args.max_tokens:
        tokens = tokens[:args.max_tokens]
        print(
            f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Truncated to {len(tokens):,} tokens (--max-tokens={args.max_tokens})"
        )

    # Create plots directory and datestring if comparing probabilities
    datestring = None
    if args.compare_probs:
        Path("plots").mkdir(exist_ok=True)
        datestring = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Run compression
    mode_str = "prefill" if args.use_prefill else "teacher-forcing"
    print(f"→ Running Block‑Emission coder (encoder mode: {mode_str}) …")
    save_plot = f"plots/block_probs_comparison_{datestring}.png" if args.compare_probs else None
    plots_dir = Path("plots") if args.compare_probs else None
    results = {}
    results["block"] = compress(tokens,
                                block_enc,
                                block_dec,
                                "block",
                                tokenizer=tokenizer,
                                save_plot=save_plot,
                                use_prefill=args.use_prefill,
                                plots_dir=plots_dir,
                                compare_probs=args.compare_probs)

    # TODO: Re-add incremental scaling coder when implemented
    # if inc_enc:
    #     mode_str = "prefill" if args.use_prefill else "teacher-forcing"
    #     print(f"→ Running Incremental‑Scaling coder (encoder mode: {mode_str}) …")
    #     save_plot = f"plots/incremental_probs_comparison_{datestring}.png" if args.compare_probs else None
    #     results["incremental"] = compress(tokens, inc_enc, inc_dec, "incremental",
    #                                      tokenizer=tokenizer, save_plot=save_plot,
    #                                      use_prefill=args.use_prefill)

    # Pretty summary
    print("\n===== Compression Stats =====")
    for method, r in results.items():
        if not r["success"]:
            print(
                f"{method.title():12} » ❌ decode mismatch | {r['num_tokens']} tokens"
            )
        else:
            compression_factor = 1.0 / r['compression_ratio'] if r[
                'compression_ratio'] > 0 else 0
            print(
                f"{method.title():12} » {compression_factor:.1f}x compression | "
                f"{r['bits_per_token']:.2f} bpt | "
                f"{r['num_tokens']} tokens")
        # Always print timing info
        print(f"{'':12}   Encode: {r['encode_time']:.2f}s total | "
              f"{r['encode_time_per_token']*1000:.2f}ms/tok | "
              f"{r['encode_tokens_per_sec']:.1f} tok/s")
        print(f"{'':12}   Decode: {r['decode_time']:.2f}s total | "
              f"{r['decode_time_per_token']*1000:.2f}ms/tok | "
              f"{r['decode_tokens_per_sec']:.1f} tok/s")

    # JSON dump
    if args.output_json:
        payload = {
            m: {
                **r,
                "model": model_path,
                "file": str(args.file),
                "datetime": datetime.now().isoformat(),
            }
            for m, r in results.items()
        }
        args.output_json.write_text(json.dumps(payload, indent=2))
        print(f"\nSaved detailed stats → {args.output_json}")


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
