#!/usr/bin/env python3
"""
Cluster plain text datasets (enwik8, enwik9) using K-means on embeddings.

Parses Wikipedia XML dumps, extracts articles, cleans text, embeds with
an HF embedding model, and clusters with K-means. Output format matches
the lmsys/wildchat pipeline: cluster_XXX/{train,test}.json.

Usage:
    # enwik8 (first 10^8 bytes)
    python scripts/lora/cluster_text_datasets.py \
        --input data/enwiki9/enwik9 \
        --output-dir /path/to/enwik8-clustered \
        --n-clusters 10 --max-bytes 100000000

    # enwik9 (full file)
    python scripts/lora/cluster_text_datasets.py \
        --input data/enwiki9/enwik9 \
        --output-dir /path/to/enwik9-clustered \
        --n-clusters 10
"""

import argparse
import json
import re
import sys
import numpy as np
import torch
import pickle
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import Optional
from datetime import datetime

# Add this script's directory to sys.path so we can import sibling modules
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cluster_chat_datasets import EmbeddingModel, make_json_serializable


def clean_wikipedia_text(xml_text: str) -> str:
    """Extract and clean text from Wikipedia XML.

    Removes XML tags, templates, and metadata while preserving article content.
    (Adapted from scripts/create_enwiki9_dataset.py)
    """
    # Remove XML tags
    text = re.sub(r'<[^>]+>', '', xml_text)

    # Remove Wikipedia templates {{...}}
    text = re.sub(r'\{\{[^}]+\}\}', '', text)

    # Remove references [[Category:...]], [[Image:...]], etc.
    text = re.sub(r'\[\[(Category|Image|File):[^\]]+\]\]', '', text)

    # Convert [[link|text]] to just text, [[link]] to link
    text = re.sub(r'\[\[([^|\]]+)\|([^\]]+)\]\]', r'\2', text)
    text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)

    # Remove citations and refs
    text = re.sub(r'&lt;ref[^&]*&lt;/ref&gt;', '', text)

    # Clean up excessive whitespace
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    text = re.sub(r' +', ' ', text)

    # Remove lines that are mostly metadata
    lines = []
    for line in text.split('\n'):
        line = line.strip()
        if (len(line) > 20 and not line.startswith('<?')
                and not line.startswith('xmlns')
                and 'mediawiki' not in line.lower()
                and 'timestamp' not in line.lower()):
            lines.append(line)

    return '\n'.join(lines)


def extract_articles(file_path: str,
                     max_bytes: Optional[int] = None,
                     min_words: int = 100) -> list[dict]:
    """Extract individual Wikipedia articles from the XML dump.

    Args:
        file_path: Path to the enwik file (raw XML)
        max_bytes: Maximum bytes to read (None for full file)
        min_words: Minimum word count to keep an article

    Returns:
        List of dicts with 'title' and 'text' keys
    """
    print(f"Reading {file_path}" +
          (f" (first {max_bytes:,} bytes)" if max_bytes else "") + "...")

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        if max_bytes:
            raw = f.read(max_bytes)
        else:
            raw = f.read()

    print(f"Read {len(raw):,} characters")

    # Extract <page>...</page> blocks
    # Use non-greedy matching to get individual pages
    page_pattern = re.compile(r'<page>(.*?)</page>', re.DOTALL)
    title_pattern = re.compile(r'<title>(.*?)</title>')
    text_pattern = re.compile(r'<text[^>]*>(.*?)</text>', re.DOTALL)

    pages = page_pattern.findall(raw)
    print(f"Found {len(pages)} pages")

    articles = []
    skipped_short = 0
    skipped_special = 0

    for page_content in tqdm(pages, desc="Extracting articles"):
        # Get title
        title_match = title_pattern.search(page_content)
        title = title_match.group(1) if title_match else ""

        # Skip special pages (redirects, categories, templates, etc.)
        if any(
                title.startswith(prefix) for prefix in [
                    'Wikipedia:',
                    'Template:',
                    'Category:',
                    'Image:',
                    'MediaWiki:',
                    'Help:',
                    'Portal:',
                    'User:',
                    'Talk:',
                    'File:',
                ]):
            skipped_special += 1
            continue

        # Get text content
        text_match = text_pattern.search(page_content)
        if not text_match:
            continue

        raw_text = text_match.group(1)

        # Skip redirects
        if raw_text.strip().lower().startswith('#redirect'):
            skipped_special += 1
            continue

        # Clean the article text
        cleaned = clean_wikipedia_text(raw_text)

        # Filter by word count
        word_count = len(cleaned.split())
        if word_count < min_words:
            skipped_short += 1
            continue

        articles.append({
            'title': title,
            'text': cleaned,
            'word_count': word_count,
        })

    print(f"\nExtracted {len(articles)} articles")
    print(f"  Skipped {skipped_special} special/redirect pages")
    print(f"  Skipped {skipped_short} articles with <{min_words} words")
    if articles:
        word_counts = [a['word_count'] for a in articles]
        print(
            f"  Word count stats: min={min(word_counts)}, max={max(word_counts)}, "
            f"mean={np.mean(word_counts):.0f}, median={np.median(word_counts):.0f}"
        )

    return articles


def cluster_embeddings(
    embeddings: np.ndarray,
    n_clusters: int = 10,
    random_state: int = 42,
) -> tuple[np.ndarray, MiniBatchKMeans]:
    """Run K-means clustering on embeddings.

    Args:
        embeddings: NumPy array of shape (n_samples, embedding_dim)
        n_clusters: Number of clusters
        random_state: Random seed

    Returns:
        Tuple of (cluster labels, fitted kmeans model)
    """
    print(
        f"Running K-means with {n_clusters} clusters on {len(embeddings)} samples..."
    )

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=1024,
        n_init=3,
        verbose=1,
    )

    labels = kmeans.fit_predict(embeddings)

    unique, counts = np.unique(labels, return_counts=True)
    print(f"\nCluster distribution:")
    for cid, cnt in zip(unique, counts):
        print(f"  Cluster {cid}: {cnt} articles")
    print(f"  Min cluster size: {counts.min()}")
    print(f"  Max cluster size: {counts.max()}")
    print(f"  Mean cluster size: {counts.mean():.1f}")
    print(f"  Std cluster size: {counts.std():.1f}")

    return labels, kmeans


def split_by_cluster(
    articles: list[dict],
    texts: list[str],
    labels: np.ndarray,
    test_ratio: float = 0.1,
    random_state: int = 42,
) -> dict:
    """Split articles into train/test per cluster.

    Args:
        articles: List of article dicts (with title, text, word_count)
        texts: List of article texts (for the output format)
        labels: Cluster labels
        test_ratio: Fraction for test set
        random_state: Random seed

    Returns:
        Dict mapping cluster_id to {train: {samples, texts}, test: {samples, texts}, ...}
    """
    clusters = {}
    unique_labels = np.unique(labels)

    for cluster_id in tqdm(unique_labels, desc="Splitting clusters"):
        mask = labels == cluster_id
        cluster_indices = np.where(mask)[0]

        cluster_articles = [articles[i] for i in cluster_indices]
        cluster_texts = [texts[i] for i in cluster_indices]

        # Build sample metadata
        cluster_samples = [{
            'index': int(idx),
            'title': cluster_articles[j]['title'],
            'word_count': cluster_articles[j]['word_count'],
        } for j, idx in enumerate(cluster_indices)]

        if len(cluster_samples) < 10:
            # Too small for split, put all in train
            train_samples = cluster_samples
            train_texts = cluster_texts
            test_samples = []
            test_texts = []
        else:
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
                'texts': train_texts
            },
            'test': {
                'samples': test_samples,
                'texts': test_texts
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
    kmeans_model: MiniBatchKMeans,
    embedding_model_name: str,
    extra_metadata: dict = None,
):
    """Save clustered data to disk.

    Args:
        clusters: Dict of cluster data
        output_dir: Output directory
        dataset_name: Name of source dataset
        kmeans_model: Fitted K-means model
        embedding_model_name: Name of embedding model used
        extra_metadata: Additional metadata to include
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clusters_dir = output_dir / "clusters"
    clusters_dir.mkdir(exist_ok=True)

    print(f"\nSaving clusters to {clusters_dir}")

    for cluster_id, data in tqdm(clusters.items(), desc="Saving clusters"):
        cluster_dir = clusters_dir / f"cluster_{cluster_id:03d}"
        cluster_dir.mkdir(exist_ok=True)

        if data['train']['samples']:
            train_data = {
                'samples': make_json_serializable(data['train']['samples']),
                'texts': data['train']['texts'],
            }
            with open(cluster_dir / "train.json", 'w') as f:
                json.dump(train_data, f)

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
        'created_at': datetime.now().isoformat(),
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
    if extra_metadata:
        metadata.update(extra_metadata)

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
        description=
        "Cluster plain text datasets (enwik8/9) using K-means on embeddings")
    parser.add_argument("--input",
                        type=str,
                        required=True,
                        help="Path to input file (e.g., data/enwiki9/enwik9)")
    parser.add_argument("--output-dir",
                        type=str,
                        required=True,
                        help="Output directory for clustered data")
    parser.add_argument("--n-clusters",
                        type=int,
                        default=10,
                        help="Number of K-means clusters (default: 10)")
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=None,
        help=
        "Max bytes to read (e.g., 100000000 for enwik8). None for full file.")
    parser.add_argument(
        "--min-words",
        type=int,
        default=100,
        help="Minimum words per article to keep (default: 100)")
    parser.add_argument("--max-articles",
                        type=int,
                        default=None,
                        help="Maximum articles to use (default: all)")
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

    # Step 1: Extract articles from XML
    articles = extract_articles(
        args.input,
        max_bytes=args.max_bytes,
        min_words=args.min_words,
    )

    if not articles:
        print("No valid articles found!")
        return

    # Optionally limit number of articles
    if args.max_articles and len(articles) > args.max_articles:
        print(f"Sampling {args.max_articles} articles from {len(articles)}...")
        indices = np.random.choice(len(articles),
                                   args.max_articles,
                                   replace=False)
        articles = [articles[i] for i in sorted(indices)]

    texts = [a['text'] for a in articles]
    print(f"\nUsing {len(articles)} articles for clustering")

    # Step 2: Embed articles
    embed_model = EmbeddingModel(
        model_name=args.embedding_model,
        batch_size=args.batch_size,
    )
    print(f"\nEmbedding {len(texts)} articles...")
    embeddings = embed_model.embed(texts)
    print(f"Embeddings shape: {embeddings.shape}")

    # Step 3: Cluster
    labels, kmeans = cluster_embeddings(
        embeddings,
        n_clusters=args.n_clusters,
        random_state=args.seed,
    )

    # Step 4: Split by cluster
    clusters = split_by_cluster(
        articles,
        texts,
        labels,
        test_ratio=args.test_ratio,
        random_state=args.seed,
    )

    # Step 5: Save
    dataset_name = "enwik8" if args.max_bytes and args.max_bytes <= 100_000_000 else "enwik9"
    save_clusters(
        clusters,
        args.output_dir,
        dataset_name,
        kmeans,
        args.embedding_model,
        extra_metadata={
            'source_file': args.input,
            'max_bytes': args.max_bytes,
            'min_words': args.min_words,
            'total_articles_extracted': len(articles),
        },
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
