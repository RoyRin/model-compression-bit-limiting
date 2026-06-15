#!/usr/bin/env python3
"""
Build a RAG index from downloaded datasets, linking embeddings to task IDs.

Supports multiple embedding models:
- nemotron: nvidia/NV-Embed-v2
- gemma: google/embeddinggemma-300m
- qwen: Qwen/Qwen3-Embedding-0.6B

Uses langchain-community for vector store.
"""

import yaml
import argparse
import numpy as np
import torch
from pathlib import Path
from datasets import load_from_disk
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from langchain_community.vectorstores import FAISS as LangchainFAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# Preset embedding models
EMBEDDING_MODELS = {
    'nemotron': 'nvidia/NV-Embed-v2',
    'gemma': 'google/embeddinggemma-300m',
    'qwen': 'Qwen/Qwen3-Embedding-0.6B',
}


def get_text_from_example(example: dict) -> str:
    """Extract text from a dataset example.

    Tries common field names and concatenates all text fields.
    """
    text_fields = [
        'text', 'input', 'question', 'premise', 'sentence', 'sentence1',
        'sentence2', 'context', 'passage', 'source', 'target', 'output',
        'answer', 'label_text'
    ]

    texts = []
    for field in text_fields:
        if field in example and example[field]:
            val = example[field]
            if isinstance(val, str):
                texts.append(val)
            elif isinstance(val, list):
                texts.extend([v for v in val if isinstance(v, str)])

    # If no known fields, try all string fields
    if not texts:
        for key, val in example.items():
            if isinstance(val, str) and len(val) > 10:
                texts.append(val)

    return " ".join(texts)


class HuggingFaceEmbeddings(Embeddings):
    """Custom LangChain Embeddings wrapper for HuggingFace models."""

    def __init__(self,
                 model_path: str,
                 device: str = None,
                 batch_size: int = 8):
        self.model_path = model_path
        self.device = device or ("cuda"
                                 if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        print(f"Loading embedding model: {model_path}")
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
        print(f"✓ Model loaded on {self.device}")

    def _embed(self, texts: list[str]) -> list[list[float]]:
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

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]


def resolve_model_name(model_arg: str) -> str:
    """Resolve model argument to full HuggingFace model path.

    Args:
        model_arg: Either a preset name (nemotron, gemma, qwen) or full HF path

    Returns:
        Full HuggingFace model path
    """
    if model_arg in EMBEDDING_MODELS:
        return EMBEDDING_MODELS[model_arg]
    return model_arg


def build_rag_index(
    data_dir: str,
    output_dir: str,
    model_name: str = "qwen",
    samples_per_task: int = 50,
    batch_size: int = 8,
):
    """Build RAG index from downloaded datasets.

    Creates ONE embedding per dataset/task by aggregating sampled examples.

    Args:
        data_dir: Directory containing downloaded datasets
        output_dir: Directory to save RAG index
        model_name: Embedding model preset (nemotron, gemma, qwen) or full HF path
        samples_per_task: Number of samples to aggregate per task for embedding
        batch_size: Batch size for embedding
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = data_dir / "datasets"
    manifest_path = data_dir / "manifest.yaml"

    # Resolve model name
    model_path = resolve_model_name(model_name)

    # Load manifest
    if manifest_path.exists():
        with open(manifest_path, 'r') as f:
            manifest = yaml.safe_load(f)
        tasks = manifest.get('tasks', {})
    else:
        # Build from directory
        tasks = {}
        for task_path in dataset_dir.glob("task*"):
            task_id = task_path.name.replace("task", "")
            tasks[task_id] = {
                'dataset_path': str(task_path),
                'description': f"Task {task_id}"
            }

    print(f"Found {len(tasks)} tasks")
    print(f"Embedding model: {model_path}")
    if model_name in EMBEDDING_MODELS:
        print(f"  (preset: {model_name})")
    print(f"Samples per task (for aggregation): {samples_per_task}")
    print(f"One embedding per task (not per sample)")
    print("=" * 60)

    # Create embeddings wrapper for LangChain
    embeddings = HuggingFaceEmbeddings(model_path, batch_size=batch_size)

    # Collect all documents - ONE per task
    all_documents = []
    metadata = {}

    for task_id, info in tqdm(tasks.items(), desc="Processing tasks"):
        dataset_path = info.get('dataset_path',
                                str(dataset_dir / f"task{task_id}"))

        if not Path(dataset_path).exists():
            print(f"  Skipping task {task_id}: dataset not found")
            continue

        try:
            # Load dataset
            dataset = load_from_disk(dataset_path)

            # Get train split (or first available split)
            if 'train' in dataset:
                split = dataset['train']
            else:
                split_name = list(dataset.keys())[0]
                split = dataset[split_name]

            # Sample examples to aggregate
            num_samples = min(samples_per_task, len(split))
            indices = np.random.choice(len(split), num_samples, replace=False)

            texts = []
            for idx in indices:
                example = split[int(idx)]
                text = get_text_from_example(example)
                if text and len(text) > 20:
                    # Truncate individual examples to avoid too long aggregation
                    texts.append(text[:500])

            if not texts:
                print(f"  Skipping task {task_id}: no valid texts")
                continue

            # Aggregate texts into ONE document per task
            # Include description + sampled examples
            description = info.get('description', '')
            aggregated_text = f"Task: {description}\n\nExamples:\n" + "\n---\n".join(
                texts[:10])  # Limit to 10 examples in text

            # Truncate to reasonable length for embedding
            aggregated_text = aggregated_text[:4000]

            doc = Document(page_content=aggregated_text,
                           metadata={
                               'task_id': task_id,
                               'description': description,
                               'num_train_examples': len(split),
                               'hf_lora': info.get('hf_lora', ''),
                               'hf_dataset': info.get('hf_dataset', ''),
                           })
            all_documents.append(doc)

            metadata[task_id] = {
                'description': description,
                'num_train_examples': len(split),
            }

        except Exception as e:
            print(f"  Error processing task {task_id}: {e}")
            continue

    if not all_documents:
        print("No documents collected!")
        return

    print(
        f"\nTotal tasks embedded: {len(all_documents)} (one embedding per task)"
    )

    # Build FAISS index using LangChain
    print("\nBuilding FAISS index...")
    vectorstore = LangchainFAISS.from_documents(all_documents, embeddings)
    print(f"✓ FAISS index built with {len(all_documents)} vectors")

    # Save everything
    print("\nSaving RAG index...")

    # Save FAISS index using LangChain's save method
    vectorstore.save_local(str(output_dir))

    # Save metadata
    with open(output_dir / "metadata.yaml", 'w') as f:
        yaml.dump(
            {
                'model_name': model_path,
                'model_preset':
                model_name if model_name in EMBEDDING_MODELS else None,
                'num_tasks': len(all_documents),
                'samples_per_task': samples_per_task,
                'tasks': metadata,
            },
            f,
            default_flow_style=False)

    print(f"✓ RAG index saved to {output_dir}")
    print(f"  - index.faiss: FAISS index")
    print(f"  - index.pkl: Document store")
    print(f"  - metadata.yaml: Metadata and statistics")


def query_rag(
    query: str,
    index_dir: str,
    model_name: str = "qwen",
    top_k: int = 5,
):
    """Query the RAG index.

    Args:
        query: Query text
        index_dir: Directory containing RAG index
        model_name: Embedding model preset or HF path (must match index)
        top_k: Number of results to return
    """
    index_dir = Path(index_dir)
    model_path = resolve_model_name(model_name)

    # Load metadata
    with open(index_dir / "metadata.yaml", 'r') as f:
        metadata = yaml.safe_load(f)

    # Create embeddings wrapper
    embeddings = HuggingFaceEmbeddings(model_path)

    # Load FAISS index using LangChain
    vectorstore = LangchainFAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True  # Required for loading pickled data
    )

    # Search
    docs_with_scores = vectorstore.similarity_search_with_score(query, k=top_k)

    # Format results
    results = []
    for doc, score in docs_with_scores:
        text = doc.page_content
        results.append({
            'task_id': doc.metadata.get('task_id', 'unknown'),
            'score': float(score),
            'text': text[:200] + "..." if len(text) > 200 else text,
            'description': doc.metadata.get('description', ''),
        })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Build RAG index from datasets")
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Build command
    build_parser = subparsers.add_parser('build', help='Build RAG index')
    build_parser.add_argument(
        "--data-dir",
        type=str,
        default="/n/netscratch/sham_lab/Lab/rrinberg/compression/LORAS",
        help="Directory containing downloaded datasets")
    build_parser.add_argument(
        "--output-dir",
        type=str,
        default=
        "/n/netscratch/sham_lab/Lab/rrinberg/compression/LORAS/rag_index",
        help="Directory to save RAG index")
    build_parser.add_argument(
        "--model",
        type=str,
        default="qwen",
        help=
        "Embedding model: 'nemotron', 'gemma', 'qwen' (presets) or full HF path"
    )
    build_parser.add_argument("--samples-per-task",
                              type=int,
                              default=100,
                              help="Number of samples to embed per task")
    build_parser.add_argument("--batch-size",
                              type=int,
                              default=8,
                              help="Batch size for embedding")

    # Query command
    query_parser = subparsers.add_parser('query', help='Query RAG index')
    query_parser.add_argument("query", type=str, help="Query text")
    query_parser.add_argument(
        "--index-dir",
        type=str,
        default=
        "/n/netscratch/sham_lab/Lab/rrinberg/compression/LORAS/rag_index",
        help="Directory containing RAG index")
    query_parser.add_argument(
        "--model",
        type=str,
        default="qwen",
        help=
        "Embedding model: 'nemotron', 'gemma', 'qwen' (presets) or full HF path"
    )
    query_parser.add_argument("--top-k",
                              type=int,
                              default=5,
                              help="Number of results")

    # List available presets
    list_parser = subparsers.add_parser('list-models',
                                        help='List available model presets')

    args = parser.parse_args()

    if args.command == 'list-models':
        print("\nAvailable embedding model presets:")
        for name, path in EMBEDDING_MODELS.items():
            print(f"  {name}: {path}")
    elif args.command == 'build':
        build_rag_index(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            model_name=args.model,
            samples_per_task=args.samples_per_task,
            batch_size=args.batch_size,
        )
    elif args.command == 'query':
        results = query_rag(
            query=args.query,
            index_dir=args.index_dir,
            model_name=args.model,
            top_k=args.top_k,
        )
        print(f"\nTop {len(results)} results for: '{args.query}'")
        print("=" * 60)
        for i, r in enumerate(results):
            print(f"\n{i+1}. Task {r['task_id']} (score: {r['score']:.4f})")
            print(f"   Description: {r['description']}")
            print(f"   Text: {r['text']}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
