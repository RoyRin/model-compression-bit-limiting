"""
LoRA RAG (Retrieval Augmented Generation) utilities for compression experiments.

This module handles:
- Querying a FAISS index to find the best LoRA adapter for a given text
- Computing perplexity to select the best LoRA from candidates
- Two-stage LoRA selection (RAG + perplexity)
- Compression with RAG-selected LoRA adapters
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from tqdm import tqdm

# PEFT for LoRA support
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

# LoRA RAG support
try:
    import yaml as yaml_lib
    from langchain_community.vectorstores import FAISS as LangchainFAISS
    from langchain_core.embeddings import Embeddings
    LORA_RAG_AVAILABLE = True
except ImportError:
    LORA_RAG_AVAILABLE = False

# Default RAG index paths
DEFAULT_RAG_INDEX_DIR = "/n/netscratch/sham_lab/Lab/rrinberg/compression/LORAS/rag_index"
DEFAULT_RAG_EMBEDDING_MODEL = "qwen"

# Embedding model presets (must match build_rag.py)
RAG_EMBEDDING_MODELS = {
    'nemotron': 'nvidia/NV-Embed-v2',
    'gemma': 'google/embeddinggemma-300m',
    'qwen': 'Qwen/Qwen3-Embedding-0.6B',
}


class HuggingFaceEmbeddings(Embeddings):
    """Custom LangChain Embeddings wrapper for HuggingFace models.

    This is duplicated from LORA/build_rag.py for standalone usage.
    """

    def __init__(self,
                 model_path: str,
                 device: str = None,
                 batch_size: int = 8):
        if not LORA_RAG_AVAILABLE:
            raise ImportError(
                "LoRA RAG requires langchain-community. Install with: pip install langchain-community"
            )

        self.model_path = model_path
        self.device = device or ("cuda"
                                 if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        print(f"Loading RAG embedding model: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                       trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.float16
            if self.device == "cuda" else torch.float32,
            trust_remote_code=True,
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"✓ RAG embedding model loaded on {self.device}")

    def _embed(self, texts: list) -> list:
        """Embed texts using mean pooling."""
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            inputs = self.tokenizer(batch,
                                    padding=True,
                                    truncation=True,
                                    max_length=512,
                                    return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                attention_mask = inputs['attention_mask']
                embeddings = outputs.last_hidden_state
                mask_expanded = attention_mask.unsqueeze(-1).expand(
                    embeddings.size()).float()
                sum_embeddings = torch.sum(embeddings * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                mean_embeddings = sum_embeddings / sum_mask
                mean_embeddings = torch.nn.functional.normalize(
                    mean_embeddings, p=2, dim=1)
                all_embeddings.extend(mean_embeddings.cpu().numpy().tolist())

        return all_embeddings

    def embed_documents(self, texts: list) -> list:
        return self._embed(texts)

    def embed_query(self, text: str) -> list:
        return self._embed([text])[0]


def query_lora_rag(
    query_text: str,
    index_dir: str = DEFAULT_RAG_INDEX_DIR,
    embedding_model: str = DEFAULT_RAG_EMBEDDING_MODEL,
    top_k: int = 5,
) -> List[Dict]:
    """Query the LoRA RAG index to find the best LoRA for a given text.

    Args:
        query_text: Text to find a matching LoRA for (can be prompt, sample, or description)
        index_dir: Directory containing the RAG index (built by LORA/build_rag.py)
        embedding_model: Embedding model preset ('qwen', 'gemma', 'nemotron') or HF path
        top_k: Number of results to return

    Returns:
        List of dicts with keys: task_id, score, description, hf_lora, text (preview)
    """
    if not LORA_RAG_AVAILABLE:
        raise ImportError(
            "LoRA RAG requires langchain-community and pyyaml. "
            "Install with: pip install langchain-community pyyaml")

    index_dir = Path(index_dir)

    # Resolve embedding model name
    if embedding_model in RAG_EMBEDDING_MODELS:
        model_path = RAG_EMBEDDING_MODELS[embedding_model]
    else:
        model_path = embedding_model

    # Load metadata to verify model matches
    metadata_path = index_dir / "metadata.yaml"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = yaml_lib.safe_load(f)
        index_model = metadata.get('model_name', '')
        if index_model and index_model != model_path:
            print(
                f"⚠️  Warning: Index was built with {index_model}, but using {model_path}"
            )

    # Create embeddings wrapper
    embeddings = HuggingFaceEmbeddings(model_path)

    # Load FAISS index
    print(f"Loading RAG index from {index_dir}...")
    vectorstore = LangchainFAISS.load_local(
        str(index_dir), embeddings, allow_dangerous_deserialization=True)

    # Search
    docs_with_scores = vectorstore.similarity_search_with_score(query_text,
                                                                k=top_k)

    # Format results
    results = []
    for doc, score in docs_with_scores:
        text_preview = doc.page_content[:200] + "..." if len(
            doc.page_content) > 200 else doc.page_content
        results.append({
            'task_id': doc.metadata.get('task_id', 'unknown'),
            'score': float(score),
            'description': doc.metadata.get('description', ''),
            'hf_lora': doc.metadata.get('hf_lora', ''),
            'hf_dataset': doc.metadata.get('hf_dataset', ''),
            'text': text_preview,
        })

    return results


def compute_perplexity(
    model,
    tokenizer,
    text: str,
    max_length: int = 512,
) -> float:
    """Compute perplexity of text under a model.

    Args:
        model: The model (can be base or PEFT-wrapped)
        tokenizer: Tokenizer
        text: Text to compute perplexity on
        max_length: Maximum sequence length

    Returns:
        Perplexity (lower = model assigns higher probability to text)
    """
    tokens = tokenizer.encode(text,
                              add_special_tokens=True,
                              truncation=True,
                              max_length=max_length)
    input_ids = torch.tensor([tokens]).to(model.device)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

    return torch.exp(loss).item()


def compute_perplexity_batch(
    model,
    tokenizer,
    texts: List[str],
    max_length: int = 512,
) -> List[float]:
    """Compute perplexity for multiple texts efficiently.

    Args:
        model: The model
        tokenizer: Tokenizer
        texts: List of texts
        max_length: Maximum sequence length

    Returns:
        List of perplexities
    """
    perplexities = []
    for text in texts:
        ppl = compute_perplexity(model, tokenizer, text, max_length)
        perplexities.append(ppl)
    return perplexities


def select_best_lora_by_perplexity(
    text: str,
    candidate_task_ids: List[str],
    base_model,
    tokenizer,
    lora_bits: int = 4,
    lora_rank: int = 16,
) -> Tuple[str, float, List[Dict]]:
    """Select the best LoRA from candidates by computing perplexity.

    Args:
        text: Text to evaluate
        candidate_task_ids: List of candidate task IDs from RAG
        base_model: Base model (already loaded, will wrap with LoRAs)
        tokenizer: Tokenizer
        lora_bits: LoRA bits
        lora_rank: LoRA rank

    Returns:
        (best_task_id, best_perplexity, all_perplexities)
    """
    if not PEFT_AVAILABLE:
        raise ImportError(
            "PEFT is required for LoRA support. Install with: pip install peft"
        )

    all_perplexities = []

    for task_id in candidate_task_ids:
        lora_adapter = get_lora_adapter_name(
            task_id=task_id,
            bits=lora_bits,
            rank=lora_rank,
        )

        try:
            # Wrap base model with this LoRA
            peft_model = PeftModel.from_pretrained(base_model, lora_adapter)
            peft_model.eval()

            # Compute perplexity
            ppl = compute_perplexity(peft_model, tokenizer, text)

            all_perplexities.append({
                'task_id': task_id,
                'perplexity': ppl,
                'lora_adapter': lora_adapter,
            })

            # Clean up PEFT wrapper
            del peft_model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"    Error loading LoRA {task_id}: {e}")
            all_perplexities.append({
                'task_id': task_id,
                'perplexity': float('inf'),
                'error': str(e),
            })

    # Sort by perplexity (lower is better)
    all_perplexities.sort(key=lambda x: x['perplexity'])

    best = all_perplexities[0]
    return best['task_id'], best['perplexity'], all_perplexities


def get_lora_adapter_name(
    task_id: str,
    base_model: str = "Mistral-7B-Instruct-v0.2",
    bits: int = 4,
    rank: int = 16,
) -> str:
    """Generate the HuggingFace LoRA adapter name for a task.

    Args:
        task_id: Task ID (e.g., "task561" or just "561")
        base_model: Base model name
        bits: Quantization bits
        rank: LoRA rank

    Returns:
        Full HuggingFace adapter path (e.g., "Lots-of-LoRAs/Mistral-7B-Instruct-v0.2-4b-r16-task561")
    """
    # Ensure task_id has "task" prefix
    if not task_id.startswith("task"):
        task_id = f"task{task_id}"

    return f"Lots-of-LoRAs/{base_model}-{bits}b-r{rank}-{task_id}"


def compress_with_rag_lora(
    text: str,
    base_model_path: str,
    tokenizer,
    compression_config: Dict,
    load_model_fn,
    compress_fn,
    encoder_cls,
    decoder_cls,
    rag_index_dir: str = DEFAULT_RAG_INDEX_DIR,
    rag_embedding_model: str = DEFAULT_RAG_EMBEDDING_MODEL,
    lora_bits: int = 4,
    lora_rank: int = 16,
    quantization: Optional[str] = None,
    verbose: bool = False,
    use_prefill: bool = False,
) -> Dict:
    """Compress text using the best LoRA selected by RAG.

    This function:
    1. Queries the RAG index to find the best matching LoRA for the text
    2. Loads the base model with the selected LoRA adapter
    3. Compresses the text using the LoRA-enhanced model

    Args:
        text: Text to compress
        base_model_path: Path to base model (e.g., "mistralai/Mistral-7B-Instruct-v0.2")
        tokenizer: Tokenizer instance
        compression_config: Dict with compression params (bit_precision, min_prob, etc.)
        load_model_fn: Function to load model (from measure_compression)
        compress_fn: Function to compress tokens (from measure_compression)
        encoder_cls: Encoder class (BlockEmissionArithmeticCoder)
        decoder_cls: Decoder class (BlockEmissionArithmeticDecoder)
        rag_index_dir: Directory containing RAG index
        rag_embedding_model: Embedding model for RAG queries
        lora_bits: LoRA quantization bits
        lora_rank: LoRA rank
        quantization: Model quantization ("4bit", "8bit", or None)
        verbose: Enable verbose output
        use_prefill: Use prefill mode for encoding

    Returns:
        Dict with compression results and selected LoRA info
    """
    if not PEFT_AVAILABLE:
        raise ImportError(
            "PEFT is required for LoRA support. Install with: pip install peft"
        )

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Step 1: Query RAG to find best LoRA
    print(f"\n[{ts}] Querying LoRA RAG for best matching adapter...")
    print(f"  Query text preview: {text[:100]}...")

    rag_results = query_lora_rag(
        query_text=text,
        index_dir=rag_index_dir,
        embedding_model=rag_embedding_model,
        top_k=3,
    )

    if not rag_results:
        raise ValueError("RAG query returned no results")

    # Select top result
    best_match = rag_results[0]
    selected_task_id = best_match['task_id']
    selected_score = best_match['score']
    selected_description = best_match['description']

    print(f"\n[{ts}] RAG Results (top 3):")
    for i, r in enumerate(rag_results[:3]):
        marker = "→" if i == 0 else " "
        print(
            f"  {marker} {i+1}. Task {r['task_id']} (score: {r['score']:.4f})")
        print(f"       {r['description'][:60]}...")

    # Step 2: Generate LoRA adapter name
    lora_adapter = get_lora_adapter_name(
        task_id=selected_task_id,
        bits=lora_bits,
        rank=lora_rank,
    )
    print(f"\n[{ts}] Selected LoRA: {lora_adapter}")

    # Step 3: Load model with LoRA
    print(f"[{ts}] Loading model with selected LoRA adapter...")
    model, _ = load_model_fn(
        base_model_path,
        lora_adapter=lora_adapter,
        quantization=quantization,
    )

    # Step 4: Create encoder/decoder
    encoder = encoder_cls(
        model,
        tokenizer,
        bit_precision=compression_config["bit_precision"],
        bits_for_encoding_count=compression_config["bits_for_encoding_count"],
        min_prob=compression_config["min_prob"],
        temperature=compression_config["temperature"],
        verbose=verbose,
    )
    decoder = decoder_cls(
        model,
        tokenizer,
        bit_precision=compression_config["bit_precision"],
        bits_for_encoding_count=compression_config["bits_for_encoding_count"],
        min_prob=compression_config["min_prob"],
        temperature=compression_config["temperature"],
        verbose=verbose,
    )

    # Step 5: Tokenize and compress
    tokens = tokenizer.encode(text, add_special_tokens=True)
    print(f"[{ts}] Compressing {len(tokens)} tokens...")

    compression_result = compress_fn(
        tokens,
        encoder,
        decoder,
        "block",
        tokenizer=tokenizer,
        use_prefill=use_prefill,
    )

    # Clean up
    del model, encoder, decoder
    torch.cuda.empty_cache()

    # Add RAG selection info to result
    compression_result['rag_selection'] = {
        'selected_task_id':
        selected_task_id,
        'selected_lora':
        lora_adapter,
        'rag_score':
        selected_score,
        'description':
        selected_description,
        'top_candidates': [{
            'task_id': r['task_id'],
            'score': r['score'],
            'description': r['description']
        } for r in rag_results[:3]],
    }

    # Print summary
    if compression_result['success']:
        comp_ratio = 1.0 / compression_result['compression_ratio']
        print(f"\n[{ts}] ✓ RAG-LoRA Compression Complete")
        print(
            f"  Selected LoRA: {selected_task_id} ({selected_description[:40]}...)"
        )
        print(
            f"  Compression: {comp_ratio:.2f}x | {compression_result['bits_per_token']:.2f} bpt"
        )
    else:
        print(f"\n[{ts}] ❌ Compression failed")

    return compression_result


def process_dataset_with_rag_lora(
    samples: List[Dict],
    metadata: Dict,
    base_model_path: str,
    compression_config: Dict,
    compress_fn,
    encoder_cls,
    decoder_cls,
    rag_index_dir: str = DEFAULT_RAG_INDEX_DIR,
    rag_embedding_model: str = DEFAULT_RAG_EMBEDDING_MODEL,
    lora_bits: int = 4,
    lora_rank: int = 16,
    quantization: Optional[str] = None,
    verbose: bool = False,
    use_prefill: bool = False,
    run_dir: Optional[Path] = None,
    rag_top_k: int = 25,
) -> List[Dict]:
    """Process a dataset with per-sample two-stage LoRA selection.

    Two-stage approach:
    1. RAG retrieval: Get top-k candidate LoRAs based on text similarity
    2. Perplexity selection: Compute perplexity for each candidate, pick lowest

    For each sample:
    1. Compress with baseline (no LoRA) first
    2. Query RAG using prompt_text to get top-k candidate LoRAs
    3. Compute perplexity on candidates to find best LoRA
    4. Compress generated_text with prompt_text as context using best LoRA
    5. Report both baseline and LoRA compression results

    Args:
        samples: List of sample dicts with 'prompt_text', 'generated_text', etc.
        metadata: Dataset metadata
        base_model_path: Path to base model
        compression_config: Compression parameters
        compress_fn: The compress function
        encoder_cls: Encoder class
        decoder_cls: Decoder class
        rag_index_dir: Directory containing RAG index
        rag_embedding_model: Embedding model for RAG
        lora_bits: LoRA quantization bits
        lora_rank: LoRA rank
        quantization: Model quantization
        verbose: Enable verbose output
        use_prefill: Use prefill mode for encoding
        run_dir: Directory to save results
        rag_top_k: Number of RAG candidates to evaluate with perplexity (default: 25)

    Returns:
        List of compression results with both baseline and LoRA results
    """
    if not PEFT_AVAILABLE:
        raise ImportError(
            "PEFT is required for LoRA support. Install with: pip install peft"
        )
    if not LORA_RAG_AVAILABLE:
        raise ImportError(
            "LoRA RAG requires langchain-community. Install with: pip install langchain-community"
        )

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n{'='*80}")
    print("TWO-STAGE LoRA SELECTION (RAG + PERPLEXITY)")
    print(f"{'='*80}")
    print(f"Total samples: {len(samples)}")
    print(f"RAG Index: {rag_index_dir}")
    print(f"Embedding Model: {rag_embedding_model}")
    print(f"RAG top-k candidates: {rag_top_k}")
    print(f"Stage 1: RAG retrieval → top {rag_top_k} candidates")
    print(f"Stage 2: Perplexity evaluation → pick lowest")
    print(f"{'='*80}\n")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    # =========================================================================
    # PHASE 1: Baseline compression (no LoRA)
    # =========================================================================
    print(f"\n[{ts}] PHASE 1: Baseline Compression (No LoRA)")
    print(f"{'='*80}")

    baseline_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16
        if torch.cuda.is_available() else torch.float32,
        device_map="auto" if quantization else None,
    )
    if torch.cuda.is_available() and not quantization:
        baseline_model = baseline_model.cuda()
    baseline_model.eval()

    baseline_encoder = encoder_cls(
        baseline_model,
        tokenizer,
        bit_precision=compression_config["bit_precision"],
        bits_for_encoding_count=compression_config["bits_for_encoding_count"],
        min_prob=compression_config["min_prob"],
        temperature=compression_config["temperature"],
        verbose=verbose,
    )
    baseline_decoder = decoder_cls(
        baseline_model,
        tokenizer,
        bit_precision=compression_config["bit_precision"],
        bits_for_encoding_count=compression_config["bits_for_encoding_count"],
        min_prob=compression_config["min_prob"],
        temperature=compression_config["temperature"],
        verbose=verbose,
    )

    # Store baseline results keyed by sample index
    baseline_results: Dict[int, Dict] = {}

    for idx, sample in enumerate(tqdm(samples, desc="Baseline compression")):
        prompt_text = sample['prompt_text']
        generated_text = sample['generated_text']

        # Tokenize with prompt as context
        full_text = prompt_text + generated_text
        full_tokens = tokenizer.encode(full_text, add_special_tokens=True)
        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=True)
        generated_tokens = full_tokens[len(prompt_tokens):]

        if len(generated_tokens) < 2:
            continue

        result = compress_fn(
            generated_tokens,
            baseline_encoder,
            baseline_decoder,
            "block",
            tokenizer=tokenizer,
            use_prefill=use_prefill,
            initial_context=prompt_tokens,
        )

        baseline_results[idx] = {
            'success': result['success'],
            'compression_ratio': result['compression_ratio'],
            'bits_per_token': result['bits_per_token'],
            'prompt_length': len(prompt_tokens),
            'generated_length': len(generated_tokens),
        }

    # Clean up baseline model
    del baseline_model, baseline_encoder, baseline_decoder
    torch.cuda.empty_cache()

    baseline_successful = [
        r for r in baseline_results.values() if r['success']
    ]
    if baseline_successful:
        baseline_avg_bpt = sum(
            r['bits_per_token']
            for r in baseline_successful) / len(baseline_successful)
        baseline_avg_comp = sum(
            1.0 / r['compression_ratio']
            for r in baseline_successful) / len(baseline_successful)
        print(
            f"\nBaseline: {baseline_avg_comp:.2f}x compression, {baseline_avg_bpt:.2f} bpt"
        )

    # =========================================================================
    # PHASE 2: Two-stage LoRA selection + compression
    # =========================================================================
    print(f"\n[{ts}] PHASE 2: Two-Stage LoRA Selection + Compression")
    print(f"{'='*80}")

    # Pre-load RAG embeddings model
    print(f"[{ts}] Pre-loading RAG embedding model...")
    if rag_embedding_model in RAG_EMBEDDING_MODELS:
        embedding_model_path = RAG_EMBEDDING_MODELS[rag_embedding_model]
    else:
        embedding_model_path = rag_embedding_model

    rag_embeddings = HuggingFaceEmbeddings(embedding_model_path)

    # Load FAISS index
    print(f"[{ts}] Loading RAG index from {rag_index_dir}...")
    vectorstore = LangchainFAISS.load_local(
        str(rag_index_dir),
        rag_embeddings,
        allow_dangerous_deserialization=True)

    # Load base model ONCE - we'll swap LoRA adapters on top of it
    print(f"\n[{ts}] Loading base model (will swap LoRA adapters)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16
        if torch.cuda.is_available() else torch.float32,
        device_map="auto" if quantization else None,
    )
    if torch.cuda.is_available() and not quantization:
        base_model = base_model.cuda()
    base_model.eval()
    print(
        f"[{ts}] Base model loaded - will swap LoRA adapters without reloading"
    )

    # Track current LoRA state
    current_lora_task: Optional[str] = None
    current_peft_model = None
    encoder = None
    decoder = None

    all_results = []
    lora_selection_counts: Dict[str, int] = {}

    for idx, sample in enumerate(
            tqdm(samples, desc="Two-stage LoRA compression")):
        prompt_text = sample['prompt_text']
        generated_text = sample['generated_text']

        # Skip if baseline skipped this sample
        if idx not in baseline_results:
            continue

        # =====================================================================
        # Stage 1: RAG retrieval - get top-k candidates
        # =====================================================================
        docs_with_scores = vectorstore.similarity_search_with_score(
            prompt_text, k=rag_top_k)

        if not docs_with_scores:
            print(f"  [Sample {idx}] No RAG results, skipping")
            continue

        candidate_task_ids = [
            doc.metadata.get('task_id', 'unknown')
            for doc, score in docs_with_scores
        ]

        print(
            f"\n  [Sample {idx}] Stage 1: RAG retrieved {len(candidate_task_ids)} candidates"
        )

        # =====================================================================
        # Stage 2: Perplexity evaluation - find best LoRA
        # =====================================================================
        eval_text = prompt_text + generated_text

        # First compute baseline perplexity (no LoRA)
        baseline_ppl = compute_perplexity(base_model, tokenizer, eval_text)

        print(
            f"  [Sample {idx}] Stage 2: Computing perplexity on {len(candidate_task_ids)} candidates..."
        )
        print(f"    Baseline (no LoRA) perplexity: {baseline_ppl:.2f}")

        best_task_id, best_ppl, all_ppls = select_best_lora_by_perplexity(
            text=eval_text,
            candidate_task_ids=candidate_task_ids,
            base_model=base_model,
            tokenizer=tokenizer,
            lora_bits=lora_bits,
            lora_rank=lora_rank,
        )

        # Find the RAG rank of the selected LoRA
        rag_rank = candidate_task_ids.index(
            best_task_id) + 1 if best_task_id in candidate_task_ids else -1

        # Get description from RAG results
        selected_description = ''
        for doc, score in docs_with_scores:
            if doc.metadata.get('task_id') == best_task_id:
                selected_description = doc.metadata.get('description', '')
                break

        print(
            f"  [Sample {idx}] Selected: {best_task_id} (ppl={best_ppl:.2f}, RAG rank={rag_rank}/{len(candidate_task_ids)})"
        )

        # Track selection counts
        lora_selection_counts[best_task_id] = lora_selection_counts.get(
            best_task_id, 0) + 1

        # =====================================================================
        # Load selected LoRA for compression (if different from current)
        # =====================================================================
        if best_task_id != current_lora_task:
            lora_adapter = get_lora_adapter_name(
                task_id=best_task_id,
                bits=lora_bits,
                rank=lora_rank,
            )

            # Clean up previous PEFT wrapper (but keep base_model)
            if current_peft_model is not None:
                del current_peft_model, encoder, decoder
                torch.cuda.empty_cache()

            # Wrap base model with new LoRA adapter
            current_peft_model = PeftModel.from_pretrained(
                base_model, lora_adapter)
            current_peft_model.eval()

            encoder = encoder_cls(
                current_peft_model,
                tokenizer,
                bit_precision=compression_config["bit_precision"],
                bits_for_encoding_count=compression_config[
                    "bits_for_encoding_count"],
                min_prob=compression_config["min_prob"],
                temperature=compression_config["temperature"],
                verbose=verbose,
            )
            decoder = decoder_cls(
                current_peft_model,
                tokenizer,
                bit_precision=compression_config["bit_precision"],
                bits_for_encoding_count=compression_config[
                    "bits_for_encoding_count"],
                min_prob=compression_config["min_prob"],
                temperature=compression_config["temperature"],
                verbose=verbose,
            )

            current_lora_task = best_task_id

        # Step 3: Tokenize with prompt as context
        full_text = prompt_text + generated_text
        full_tokens = tokenizer.encode(full_text, add_special_tokens=True)
        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=True)
        generated_tokens = full_tokens[len(prompt_tokens):]

        # Step 4: Compress with LoRA
        lora_result = compress_fn(
            generated_tokens,
            encoder,
            decoder,
            "block",
            tokenizer=tokenizer,
            use_prefill=use_prefill,
            initial_context=prompt_tokens,
        )

        # Get baseline result for this sample
        baseline_r = baseline_results[idx]

        # Build combined result
        result = {
            'prompt_id': sample.get('prompt_id', idx),
            'max_new_tokens': sample.get('max_new_tokens'),
            'prompt_length': len(prompt_tokens),
            'generated_length': len(generated_tokens),
            # Baseline results
            'baseline': {
                'success': baseline_r['success'],
                'compression_ratio': baseline_r['compression_ratio'],
                'bits_per_token': baseline_r['bits_per_token'],
            },
            # LoRA results
            'lora': {
                'success': lora_result['success'],
                'compression_ratio': lora_result['compression_ratio'],
                'bits_per_token': lora_result['bits_per_token'],
            },
            # Two-stage selection info
            'two_stage_selection': {
                'selected_task_id': best_task_id,
                'selected_perplexity': best_ppl,
                'rag_rank': rag_rank,
                'num_candidates': len(candidate_task_ids),
                'description': selected_description,
            },
        }

        # Calculate improvement
        if baseline_r['success'] and lora_result['success']:
            baseline_bpt = baseline_r['bits_per_token']
            lora_bpt = lora_result['bits_per_token']
            improvement_bpt = ((baseline_bpt - lora_bpt) / baseline_bpt) * 100
            result['improvement_bpt_pct'] = improvement_bpt
        else:
            result['improvement_bpt_pct'] = None

        all_results.append(result)

    # Clean up
    if current_peft_model is not None:
        del current_peft_model
    if encoder is not None:
        del encoder
    if decoder is not None:
        del decoder
    del base_model
    torch.cuda.empty_cache()

    # Also clean up RAG embedding model
    del rag_embeddings, vectorstore
    torch.cuda.empty_cache()

    # Print summary
    _print_rag_lora_summary(all_results, lora_selection_counts)

    # Save results if run_dir provided
    if run_dir:
        _save_rag_lora_results(all_results, lora_selection_counts,
                               base_model_path, rag_index_dir,
                               rag_embedding_model, rag_top_k, run_dir)

    return all_results


def _print_rag_lora_summary(all_results: List[Dict],
                            lora_selection_counts: Dict[str, int]):
    """Print summary of RAG-LoRA compression results."""
    print(f"\n{'='*80}")
    print("RAG-LoRA vs BASELINE COMPARISON")
    print(f"{'='*80}")

    successful_both = [
        r for r in all_results
        if r['baseline']['success'] and r['lora']['success']
    ]

    print(f"Total samples: {len(all_results)}")
    print(f"Both successful: {len(successful_both)}")

    if successful_both:
        avg_baseline_bpt = sum(r['baseline']['bits_per_token']
                               for r in successful_both) / len(successful_both)
        avg_lora_bpt = sum(r['lora']['bits_per_token']
                           for r in successful_both) / len(successful_both)
        avg_improvement = sum(
            r['improvement_bpt_pct'] for r in successful_both
            if r['improvement_bpt_pct']) / len(successful_both)

        print(
            f"\n  {'Metric':<25} {'Baseline':<15} {'RAG-LoRA':<15} {'Improvement':<15}"
        )
        print(f"  {'-'*70}")
        print(
            f"  {'Avg bits/token':<25} {avg_baseline_bpt:<15.3f} {avg_lora_bpt:<15.3f} {avg_improvement:+.2f}%"
        )

        improved = sum(
            1 for r in successful_both
            if r['improvement_bpt_pct'] and r['improvement_bpt_pct'] > 0)
        regressed = sum(
            1 for r in successful_both
            if r['improvement_bpt_pct'] and r['improvement_bpt_pct'] < 0)

        print(
            f"\n  Samples improved:  {improved} ({100*improved/len(successful_both):.1f}%)"
        )
        print(
            f"  Samples regressed: {regressed} ({100*regressed/len(successful_both):.1f}%)"
        )

    print(f"\nLoRA Selection Distribution:")
    for task_id, count in sorted(lora_selection_counts.items(),
                                 key=lambda x: -x[1]):
        pct = 100 * count / len(all_results) if all_results else 0
        print(f"  {task_id:<12} {count:<10} ({pct:.1f}%)")

    print(f"{'='*80}\n")


def _save_rag_lora_results(all_results: List[Dict],
                           lora_selection_counts: Dict[str, int],
                           base_model_path: str, rag_index_dir: str,
                           rag_embedding_model: str, rag_top_k: int,
                           run_dir: Path):
    """Save RAG-LoRA results to YAML."""
    import yaml

    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "rag_lora_results.yaml"

    successful_both = [
        r for r in all_results
        if r['baseline']['success'] and r['lora']['success']
    ]

    # Convert results for YAML
    yaml_results = []
    for r in all_results:
        yaml_results.append({
            'prompt_id':
            str(r['prompt_id']),
            'prompt_length':
            int(r['prompt_length']),
            'generated_length':
            int(r['generated_length']),
            'baseline_success':
            bool(r['baseline']['success']),
            'baseline_bits_per_token':
            float(r['baseline']['bits_per_token'])
            if r['baseline']['success'] else None,
            'lora_success':
            bool(r['lora']['success']),
            'lora_bits_per_token':
            float(r['lora']['bits_per_token'])
            if r['lora']['success'] else None,
            'selected_lora':
            r['two_stage_selection']['selected_task_id'],
            'selected_perplexity':
            float(r['two_stage_selection']['selected_perplexity']),
            'rag_rank':
            int(r['two_stage_selection']['rag_rank']),
            'improvement_bpt_pct':
            float(r['improvement_bpt_pct'])
            if r['improvement_bpt_pct'] is not None else None,
        })

    # Compute summary stats
    avg_baseline_bpt = sum(r['baseline']['bits_per_token']
                           for r in successful_both) / len(
                               successful_both) if successful_both else None
    avg_lora_bpt = sum(r['lora']['bits_per_token']
                       for r in successful_both) / len(
                           successful_both) if successful_both else None
    avg_improvement = sum(r['improvement_bpt_pct']
                          for r in successful_both if r['improvement_bpt_pct']
                          ) / len(successful_both) if successful_both else None

    output_data = {
        'metadata': {
            'created_at': datetime.now().isoformat(),
            'base_model': base_model_path,
            'rag_index': rag_index_dir,
            'embedding_model': rag_embedding_model,
            'rag_top_k': rag_top_k,
            'selection_method': 'two_stage_rag_perplexity',
            'total_samples': len(all_results),
            'successful_both': len(successful_both),
            'lora_selection_counts': lora_selection_counts,
        },
        'summary': {
            'baseline_avg_bits_per_token':
            float(avg_baseline_bpt) if avg_baseline_bpt else None,
            'lora_avg_bits_per_token':
            float(avg_lora_bpt) if avg_lora_bpt else None,
            'avg_improvement_bpt_pct':
            float(avg_improvement) if avg_improvement else None,
        },
        'results': yaml_results,
    }

    with open(results_path, 'w') as f:
        yaml.dump(output_data, f, default_flow_style=False, sort_keys=False)
    print(f"\n✓ Saved results to {results_path}")
