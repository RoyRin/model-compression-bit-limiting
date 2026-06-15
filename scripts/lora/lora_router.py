#!/usr/bin/env python3
"""
LoRA Router with RAG - Routes inputs to the most relevant LoRA using retrieval.

Embeds training examples from each cluster and uses FAISS for fast retrieval.
Routes new inputs based on the cluster labels of retrieved similar examples.
"""

import json
import pickle
import numpy as np
import torch
import faiss
from pathlib import Path
from typing import Optional, Union
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from collections import Counter


class LoRARouterRAG:
    """Routes text inputs to the most relevant LoRA using RAG over training examples."""

    def __init__(
        self,
        index_path: Optional[str] = None,
        loras_root:
        str = "/n/netscratch/sham_lab/Lab/rrinberg/compression/lmsys-loras",
        clusters_root:
        str = "/n/netscratch/sham_lab/Lab/rrinberg/compression/lmsys-clustered/clusters",
        embedding_model: str = "BAAI/bge-large-en-v1.5",
        device: str = None,
    ):
        """Initialize the LoRA router.

        Args:
            index_path: Path to pre-built FAISS index. If None, must call build_index()
            loras_root: Root directory containing LoRA folders
            clusters_root: Root directory containing cluster data folders
            embedding_model: SentenceTransformer model name for computing embeddings
            device: Device to use for embeddings
        """
        self.loras_root = Path(loras_root)
        self.clusters_root = Path(clusters_root)
        self.device = device or ("cuda"
                                 if torch.cuda.is_available() else "cpu")
        self.embedding_model_name = embedding_model

        # Will be populated by build_index() or load_index()
        self.index = None
        self.cluster_labels = None  # Maps index position to cluster ID
        self.embed_model = None
        self.tokenizer = None  # For token counting only

    def _load_embedding_model(self):
        """Lazy load the embedding model."""
        if self.embed_model is None:
            print(f"Loading embedding model: {self.embedding_model_name}")
            self.embed_model = SentenceTransformer(self.embedding_model_name,
                                                   device=self.device)
            # Load a tokenizer for token counting (use mistral since that's what LoRAs use)
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                "mistralai/Mistral-7B-Instruct-v0.2")

    def get_embedding(self, text: str) -> np.ndarray:
        """Compute embedding for a single text."""
        self._load_embedding_model()
        embedding = self.embed_model.encode(text, convert_to_numpy=True)
        return embedding.astype(np.float32)

    def get_embeddings_batch(self,
                             texts: list[str],
                             batch_size: int = 32) -> np.ndarray:
        """Compute embeddings for a batch of texts."""
        self._load_embedding_model()
        embeddings = self.embed_model.encode(texts,
                                             batch_size=batch_size,
                                             show_progress_bar=True,
                                             convert_to_numpy=True)
        return embeddings.astype(np.float32)

    def build_index(
        self,
        cluster_ids: list[int] = None,
        samples_per_cluster: int = 500,
        min_tokens: int = 150,
        output_path: str = None,
        batch_size: int = 8,
    ):
        """Build FAISS index from training examples.

        Args:
            cluster_ids: List of cluster IDs to include. Defaults to 0-49.
            samples_per_cluster: Max samples to embed per cluster
            min_tokens: Minimum number of tokens for a text to be included
            output_path: Where to save the index. Defaults to clusters_root/lora_rag_index/
            batch_size: Batch size for embedding
        """
        self._load_embedding_model()  # Need tokenizer for filtering

        if cluster_ids is None:
            cluster_ids = list(range(50))

        output_dir = Path(
            output_path
        ) if output_path else self.clusters_root.parent / "lora_rag_index"
        output_dir.mkdir(parents=True, exist_ok=True)

        all_embeddings = []
        all_labels = []

        for cluster_id in tqdm(cluster_ids, desc="Processing clusters"):
            cluster_dir = self.clusters_root / f"cluster_{cluster_id:03d}"
            train_path = cluster_dir / "train.json"

            if not train_path.exists():
                print(f"  Skipping cluster {cluster_id}: no train.json")
                continue

            with open(train_path) as f:
                data = json.load(f)

            # Filter by minimum token count
            all_texts = data['texts']
            filtered_texts = []
            for t in all_texts:
                if len(t.strip()) > 0:
                    num_tokens = len(
                        self.tokenizer.encode(t, add_special_tokens=False))
                    if num_tokens >= min_tokens:
                        filtered_texts.append(t)
                if len(filtered_texts) >= samples_per_cluster:
                    break

            texts = filtered_texts
            if not texts:
                print(
                    f"  Skipping cluster {cluster_id}: no texts with >={min_tokens} tokens"
                )
                continue

            print(
                f"  Cluster {cluster_id}: embedding {len(texts)} texts (filtered from {len(all_texts)})..."
            )
            embeddings = self.get_embeddings_batch(texts,
                                                   batch_size=batch_size)

            all_embeddings.append(embeddings)
            all_labels.extend([cluster_id] * len(embeddings))

        # Stack all embeddings
        all_embeddings = np.vstack(all_embeddings)
        all_labels = np.array(all_labels, dtype=np.int32)

        print(f"\nTotal embeddings: {len(all_embeddings)}")
        print(f"Embedding dimension: {all_embeddings.shape[1]}")

        # Normalize for cosine similarity
        faiss.normalize_L2(all_embeddings)

        # Build FAISS index (Inner Product = cosine similarity after normalization)
        dim = all_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(all_embeddings)
        self.cluster_labels = all_labels

        # Save index and labels
        faiss.write_index(self.index, str(output_dir / "index.faiss"))
        np.save(output_dir / "cluster_labels.npy", self.cluster_labels)

        print(f"Saved index to {output_dir}")

    def load_index(self, index_path: str):
        """Load a pre-built FAISS index."""
        index_dir = Path(index_path)
        self.index = faiss.read_index(str(index_dir / "index.faiss"))
        self.cluster_labels = np.load(index_dir / "cluster_labels.npy")
        print(f"Loaded index with {self.index.ntotal} vectors")

    def route(self,
              text: str,
              k: int = 10,
              return_details: bool = False) -> Union[Path, tuple[Path, dict]]:
        """Route a single text to the most relevant LoRA using k-NN voting.

        Args:
            text: Input text to route
            k: Number of neighbors to retrieve for voting
            return_details: If True, return detailed retrieval info

        Returns:
            Path to the best LoRA directory, optionally with details
        """
        if self.index is None:
            raise RuntimeError(
                "Index not loaded. Call load_index() or build_index() first.")

        embedding = self.get_embedding(text).reshape(1, -1)
        faiss.normalize_L2(embedding)

        # Retrieve k nearest neighbors
        distances, indices = self.index.search(embedding, k)
        distances = distances[0]
        indices = indices[0]

        # Get cluster labels of retrieved examples
        retrieved_labels = self.cluster_labels[indices]

        # Vote for best cluster (weighted by similarity)
        cluster_scores = Counter()
        for label, dist in zip(retrieved_labels, distances):
            cluster_scores[label] += dist  # Higher similarity = higher vote

        best_cluster = cluster_scores.most_common(1)[0][0]
        lora_path = self.loras_root / f"cluster_{best_cluster:03d}"

        if return_details:
            details = {
                'best_cluster': best_cluster,
                'cluster_scores': dict(cluster_scores),
                'retrieved_clusters': retrieved_labels.tolist(),
                'similarities': distances.tolist(),
            }
            return lora_path, details

        return lora_path

    def route_batch(self, texts: list[str], k: int = 10) -> list[Path]:
        """Route multiple texts to their most relevant LoRAs."""
        if self.index is None:
            raise RuntimeError(
                "Index not loaded. Call load_index() or build_index() first.")

        embeddings = self.get_embeddings_batch(texts)
        faiss.normalize_L2(embeddings)

        distances, indices = self.index.search(embeddings, k)

        lora_paths = []
        for i in range(len(texts)):
            retrieved_labels = self.cluster_labels[indices[i]]
            cluster_scores = Counter()
            for label, dist in zip(retrieved_labels, distances[i]):
                cluster_scores[label] += dist
            best_cluster = cluster_scores.most_common(1)[0][0]
            lora_paths.append(self.loras_root / f"cluster_{best_cluster:03d}")

        return lora_paths

    def get_top_k_loras(self,
                        text: str,
                        k_retrieve: int = 20,
                        k_loras: int = 3) -> list[tuple[Path, float]]:
        """Get top-k most relevant LoRAs for a text.

        Args:
            text: Input text
            k_retrieve: Number of examples to retrieve for voting
            k_loras: Number of top LoRAs to return

        Returns:
            List of (lora_path, score) tuples, sorted by score
        """
        _, details = self.route(text, k=k_retrieve, return_details=True)

        sorted_clusters = sorted(details['cluster_scores'].items(),
                                 key=lambda x: x[1],
                                 reverse=True)[:k_loras]

        return [(self.loras_root / f"cluster_{c:03d}", score)
                for c, score in sorted_clusters]


def main():
    """Build index or demo the router."""
    import argparse

    parser = argparse.ArgumentParser(description="LoRA Router with RAG")
    parser.add_argument("--build-index",
                        action="store_true",
                        help="Build the FAISS index")
    parser.add_argument("--samples-per-cluster",
                        type=int,
                        default=500,
                        help="Samples per cluster for index")
    parser.add_argument("--min-tokens",
                        type=int,
                        default=150,
                        help="Minimum tokens per sample")
    parser.add_argument("--index-path",
                        type=str,
                        default=None,
                        help="Path to index directory")
    parser.add_argument(
        "--clusters-root",
        type=str,
        default=
        "/n/netscratch/sham_lab/Lab/rrinberg/compression/lmsys-clustered/clusters",
        help="Root directory containing cluster folders")
    parser.add_argument(
        "--loras-root",
        type=str,
        default="/n/netscratch/sham_lab/Lab/rrinberg/compression/lmsys-loras",
        help="Root directory containing LoRA folders")
    parser.add_argument("--text", type=str, help="Text to route")
    parser.add_argument("--k",
                        type=int,
                        default=10,
                        help="Number of neighbors for voting")
    parser.add_argument("--top-k",
                        type=int,
                        default=3,
                        help="Show top-k LoRAs")
    parser.add_argument("--cluster-ids",
                        type=str,
                        default=None,
                        help="Comma-separated cluster IDs (default: 0-49)")
    args = parser.parse_args()

    default_index = "/n/netscratch/sham_lab/Lab/rrinberg/compression/lmsys-clustered/lora_rag_index"

    router = LoRARouterRAG(
        clusters_root=args.clusters_root,
        loras_root=args.loras_root,
    )

    if args.build_index:
        cluster_ids = None
        if args.cluster_ids:
            cluster_ids = [int(x.strip()) for x in args.cluster_ids.split(",")]

        router.build_index(
            cluster_ids=cluster_ids,
            samples_per_cluster=args.samples_per_cluster,
            min_tokens=args.min_tokens,
            output_path=args.index_path or default_index,
        )
        return

    # Load index for routing
    index_path = args.index_path or default_index
    print(f"Loading index from {index_path}")
    router.load_index(index_path)

    if args.text:
        texts_to_test = [args.text]
    else:
        # Demo texts from different domains
        texts_to_test = [
            "Write an article about the safety of 4-bromobenzonitrile in chemical industry",
            "¿Cuál es la capital de España?",
            "How can I improve my sleep quality?",
            "Write a SQL query to join two tables",
            "What are the symptoms of diabetes?",
        ]

    for text in texts_to_test:
        print(f"\n{'='*60}")
        print(f"Text: {text[:80]}...")
        print(f"{'='*60}")

        top_loras = router.get_top_k_loras(text,
                                           k_retrieve=args.k,
                                           k_loras=args.top_k)
        for i, (path, score) in enumerate(top_loras):
            print(f"  {i+1}. {path.name} (score: {score:.4f})")


if __name__ == "__main__":
    main()
