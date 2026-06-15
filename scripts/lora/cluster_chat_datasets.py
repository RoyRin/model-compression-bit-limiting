#!/usr/bin/env python3
"""
Cluster chat datasets (WildChat-1M, lmsys-chat-1m) using K-means on embeddings.

Uses the same embedding model as the LoRA RAG system (Qwen by default).

Steps:
1. Load datasets
2. Extract text from conversations
3. Embed text using HuggingFace model
4. Run K-means clustering
5. Split each cluster into train/test
6. Save results
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from datasets import load_from_disk, Dataset, DatasetDict
from transformers import AutoModel, AutoTokenizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import Optional
import pickle
from datetime import datetime


def make_json_serializable(obj):
    """Convert non-JSON-serializable objects to serializable form."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# Preset embedding models (same as build_rag.py)
EMBEDDING_MODELS = {
    'nemotron': 'nvidia/NV-Embed-v2',
    'gemma': 'google/embeddinggemma-300m',
    'qwen': 'Qwen/Qwen3-Embedding-0.6B',
}


def extract_conversation_text(conversation: list, max_turns: int = 5) -> str:
    """Extract text from a conversation (list of messages).

    Args:
        conversation: List of message dicts with 'role' and 'content'
        max_turns: Maximum number of turns to include

    Returns:
        Concatenated conversation text
    """
    if not conversation:
        return ""

    texts = []
    for i, msg in enumerate(conversation[:max_turns]):
        if isinstance(msg, dict):
            content = msg.get('content', '')
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                # Handle multi-part content (e.g., text + images)
                for part in content:
                    if isinstance(part, dict) and part.get('type') == 'text':
                        texts.append(part.get('text', ''))
                    elif isinstance(part, str):
                        texts.append(part)

    return "\n".join(texts)


class EmbeddingModel:
    """Embedding model wrapper (same approach as build_rag.py)."""

    def __init__(self,
                 model_name: str = "qwen",
                 device: str = None,
                 batch_size: int = 32):
        self.model_name = model_name
        self.model_path = EMBEDDING_MODELS.get(model_name, model_name)
        self.device = device or ("cuda"
                                 if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        print(f"Loading embedding model: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path,
                                                       trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16
            if self.device == "cuda" else torch.float32,
            trust_remote_code=True,
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}")

        # Get embedding dimension
        with torch.no_grad():
            dummy = self.tokenizer("test", return_tensors="pt").to(self.device)
            out = self.model(**dummy)
            self.embedding_dim = out.last_hidden_state.shape[-1]
        print(f"Embedding dimension: {self.embedding_dim}")

    def embed(self,
              texts: list[str],
              show_progress: bool = True) -> np.ndarray:
        """Embed a list of texts.

        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress bar

        Returns:
            NumPy array of shape (len(texts), embedding_dim)
        """
        all_embeddings = []

        iterator = range(0, len(texts), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator,
                            desc="Embedding",
                            total=len(texts) // self.batch_size + 1)

        for i in iterator:
            batch = texts[i:i + self.batch_size]

            # Handle empty strings
            batch = [t if t.strip() else "empty" for t in batch]

            inputs = self.tokenizer(batch,
                                    padding=True,
                                    truncation=True,
                                    max_length=512,
                                    return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                attention_mask = inputs['attention_mask']
                embeddings = outputs.last_hidden_state

                # Mean pooling
                mask_expanded = attention_mask.unsqueeze(-1).expand(
                    embeddings.size()).float()
                sum_embeddings = torch.sum(embeddings * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                mean_embeddings = sum_embeddings / sum_mask

                # Normalize
                mean_embeddings = torch.nn.functional.normalize(
                    mean_embeddings, p=2, dim=1)
                all_embeddings.append(mean_embeddings.cpu().numpy())

        return np.vstack(all_embeddings)


def load_and_prepare_dataset(
        dataset_path: str,
        max_samples: Optional[int] = None) -> tuple[list[dict], list[str]]:
    """Load dataset and extract conversation texts.

    Args:
        dataset_path: Path to saved dataset
        max_samples: Maximum number of samples to load (None for all)

    Returns:
        Tuple of (samples list, texts list)
    """
    print(f"Loading dataset from {dataset_path}")
    dataset = load_from_disk(dataset_path)

    # Get train split
    if isinstance(dataset, DatasetDict):
        if 'train' in dataset:
            data = dataset['train']
        else:
            data = dataset[list(dataset.keys())[0]]
    else:
        data = dataset

    print(f"Dataset size: {len(data)}")

    if max_samples and max_samples < len(data):
        print(f"Sampling {max_samples} examples...")
        indices = np.random.choice(len(data), max_samples, replace=False)
        data = data.select(indices)

    samples = []
    texts = []

    print("Extracting conversation texts...")
    for i in tqdm(range(len(data))):
        example = data[i]

        # Extract conversation
        conversation = example.get('conversation', [])
        text = extract_conversation_text(conversation)

        if text and len(text) > 50:  # Filter very short conversations
            samples.append({
                'index': i,
                'conversation': conversation,
                'model': example.get('model', ''),
                'language': example.get('language', ''),
                'turn': example.get('turn', 0),
            })
            texts.append(text)

    print(f"Extracted {len(texts)} valid conversations")
    return samples, texts


def cluster_embeddings(
    embeddings: np.ndarray,
    n_clusters: int = 50,
    random_state: int = 42,
) -> np.ndarray:
    """Run K-means clustering on embeddings.

    Args:
        embeddings: NumPy array of shape (n_samples, embedding_dim)
        n_clusters: Number of clusters
        random_state: Random seed

    Returns:
        Cluster labels array
    """
    print(f"Running K-means with {n_clusters} clusters...")

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=1024,
        n_init=3,
        verbose=1,
    )

    labels = kmeans.fit_predict(embeddings)

    # Print cluster distribution
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\nCluster distribution:")
    print(f"  Min cluster size: {counts.min()}")
    print(f"  Max cluster size: {counts.max()}")
    print(f"  Mean cluster size: {counts.mean():.1f}")
    print(f"  Std cluster size: {counts.std():.1f}")

    return labels, kmeans


def split_by_cluster(
    samples: list[dict],
    texts: list[str],
    labels: np.ndarray,
    test_ratio: float = 0.1,
    random_state: int = 42,
) -> dict:
    """Split samples into train/test per cluster.

    Args:
        samples: List of sample dicts
        texts: List of texts
        labels: Cluster labels
        test_ratio: Fraction for test set
        random_state: Random seed

    Returns:
        Dict with cluster info and splits
    """
    clusters = {}

    unique_labels = np.unique(labels)

    for cluster_id in tqdm(unique_labels, desc="Splitting clusters"):
        mask = labels == cluster_id
        cluster_indices = np.where(mask)[0]

        cluster_samples = [samples[i] for i in cluster_indices]
        cluster_texts = [texts[i] for i in cluster_indices]

        if len(cluster_samples) < 10:
            # Too small for split, put all in train
            train_samples = cluster_samples
            train_texts = cluster_texts
            test_samples = []
            test_texts = []
        else:
            # Split
            train_idx, test_idx = train_test_split(
                range(len(cluster_samples)),
                test_size=test_ratio,
                random_state=random_state,
            )

            train_samples = [cluster_samples[i] for i in train_idx]
            train_texts = [cluster_texts[i] for i in train_idx]
            test_samples = [cluster_samples[i] for i in test_idx]
            test_texts = [cluster_texts[i] for i in test_idx]

        clusters[int(cluster_id)] = {
            'train': {
                'samples': train_samples,
                'texts': train_texts,
            },
            'test': {
                'samples': test_samples,
                'texts': test_texts,
            },
            'total_size': len(cluster_samples),
            'train_size': len(train_samples),
            'test_size': len(test_samples),
        }

    return clusters


def save_clusters(
    clusters: dict,
    output_dir: Path,
    dataset_name: str,
    kmeans_model,
    embedding_model_name: str,
):
    """Save clustered data.

    Args:
        clusters: Dict of cluster data
        output_dir: Output directory
        dataset_name: Name of source dataset
        kmeans_model: Fitted K-means model
        embedding_model_name: Name of embedding model used
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save each cluster as a separate dataset
    clusters_dir = output_dir / "clusters"
    clusters_dir.mkdir(exist_ok=True)

    print(f"\nSaving clusters to {clusters_dir}")

    for cluster_id, data in tqdm(clusters.items(), desc="Saving clusters"):
        cluster_dir = clusters_dir / f"cluster_{cluster_id:03d}"
        cluster_dir.mkdir(exist_ok=True)

        # Save train split
        if data['train']['samples']:
            train_data = {
                'samples': make_json_serializable(data['train']['samples']),
                'texts': data['train']['texts'],
            }
            with open(cluster_dir / "train.json", 'w') as f:
                json.dump(train_data, f)

        # Save test split
        if data['test']['samples']:
            test_data = {
                'samples': make_json_serializable(data['test']['samples']),
                'texts': data['test']['texts'],
            }
            with open(cluster_dir / "test.json", 'w') as f:
                json.dump(test_data, f)

    # Save K-means model
    with open(output_dir / "kmeans_model.pkl", 'wb') as f:
        pickle.dump(kmeans_model, f)

    # Save metadata
    metadata = {
        'dataset_name': dataset_name,
        'embedding_model': embedding_model_name,
        'n_clusters': len(clusters),
        'cluster_stats': {
            int(k): {
                'total': v['total_size'],
                'train': v['train_size'],
                'test': v['test_size'],
            }
            for k, v in clusters.items()
        },
        'total_train': sum(v['train_size'] for v in clusters.values()),
        'total_test': sum(v['test_size'] for v in clusters.values()),
    }

    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved:")
    print(f"  - {len(clusters)} cluster directories")
    print(f"  - kmeans_model.pkl")
    print(f"  - metadata.json")
    print(f"  - Total train samples: {metadata['total_train']}")
    print(f"  - Total test samples: {metadata['total_test']}")


def main():
    parser = argparse.ArgumentParser(
        description="Cluster chat datasets using K-means on embeddings")

    parser.add_argument("--dataset",
                        type=str,
                        required=True,
                        help="Path to dataset (e.g., /path/to/wildchat-1m)")
    parser.add_argument("--output-dir",
                        type=str,
                        required=True,
                        help="Output directory for clustered data")
    parser.add_argument("--n-clusters",
                        type=int,
                        default=50,
                        help="Number of K-means clusters (default: 50)")
    parser.add_argument("--max-samples",
                        type=int,
                        default=None,
                        help="Maximum samples to use (default: all)")
    parser.add_argument("--test-ratio",
                        type=float,
                        default=0.1,
                        help="Test set ratio (default: 0.1)")
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="qwen",
        help=
        "Embedding model: qwen, gemma, nemotron, or HF path (default: qwen)")
    parser.add_argument("--batch-size",
                        type=int,
                        default=32,
                        help="Batch size for embedding (default: 32)")
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="Random seed (default: 42)")

    args = parser.parse_args()

    np.random.seed(args.seed)

    # Load dataset
    samples, texts = load_and_prepare_dataset(args.dataset, args.max_samples)

    if not samples:
        print("No valid samples found!")
        return

    # Create embedding model
    embed_model = EmbeddingModel(
        model_name=args.embedding_model,
        batch_size=args.batch_size,
    )

    # Embed texts
    print(f"\nEmbedding {len(texts)} texts...")
    embeddings = embed_model.embed(texts)
    print(f"Embeddings shape: {embeddings.shape}")

    # Cluster
    labels, kmeans = cluster_embeddings(
        embeddings,
        n_clusters=args.n_clusters,
        random_state=args.seed,
    )

    # Split by cluster
    clusters = split_by_cluster(
        samples,
        texts,
        labels,
        test_ratio=args.test_ratio,
        random_state=args.seed,
    )

    # Save
    dataset_name = Path(args.dataset).name
    save_clusters(
        clusters,
        args.output_dir,
        dataset_name,
        kmeans,
        args.embedding_model,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
