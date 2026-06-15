#!/usr/bin/env python3
"""Convert enwiki9 dataset to compression dataset format.

This script processes the enwiki9 XML dump (first 1GB of Wikipedia) and
extracts clean text chunks for compression benchmarking.

Usage:
    python scripts/create_enwiki9_dataset.py
    python scripts/create_enwiki9_dataset.py --chunk-size 500 --num-chunks 100
"""

import re
import argparse
from pathlib import Path
from datetime import datetime
import yaml
from typing import List


def clean_wikipedia_text(xml_text: str) -> str:
    """Extract and clean text from Wikipedia XML.

    Removes XML tags, templates, and metadata while preserving article content.
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
        # Skip empty lines, XML artifacts, and metadata
        if (len(line) > 20 and not line.startswith('<?')
                and not line.startswith('xmlns')
                and 'mediawiki' not in line.lower()
                and 'timestamp' not in line.lower()):
            lines.append(line)

    return '\n'.join(lines)


def extract_text_chunks(xml_path: Path,
                        chunk_size: int = 500,
                        max_chunks: int = 200) -> List[str]:
    """Extract chunks of clean text from enwiki9 XML.

    Args:
        xml_path: Path to enwiki9.xml file
        chunk_size: Target size for each chunk in tokens (approximate, based on words)
        max_chunks: Maximum number of chunks to extract

    Returns:
        List of text chunks
    """
    print(f"Reading {xml_path}...")
    with open(xml_path, 'r', encoding='utf-8', errors='ignore') as f:
        xml_content = f.read()

    print(f"Cleaning Wikipedia XML...")
    clean_text = clean_wikipedia_text(xml_content)

    # Split into sentences (approximate)
    sentences = re.split(r'(?<=[.!?])\s+', clean_text)

    print(f"Extracted {len(sentences)} sentences")

    # Group sentences into chunks of approximately chunk_size words
    chunks = []
    current_chunk = []
    current_size = 0

    for sentence in sentences:
        words = sentence.split()
        sentence_size = len(words)

        if current_size + sentence_size > chunk_size and current_chunk:
            # Finish this chunk
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_size = sentence_size
        else:
            current_chunk.append(sentence)
            current_size += sentence_size

        if len(chunks) >= max_chunks:
            break

    # Don't forget the last chunk
    if current_chunk and len(chunks) < max_chunks:
        chunks.append(' '.join(current_chunk))

    return chunks[:max_chunks]


def create_compression_dataset(chunks: List[str], output_path: Path):
    """Create compression dataset from text chunks.

    For enwiki9, we don't have natural prompt/continuation splits like in HumanEval.
    Instead, we'll use a sliding window: first 20% of each chunk as "prompt",
    remaining 80% as "generated_text" to compress.
    """
    samples = []

    for i, chunk in enumerate(chunks):
        # Split chunk into prompt (context) and text to compress
        words = chunk.split()
        split_point = len(words) // 5  # 20% for prompt, 80% for compression

        prompt = ' '.join(words[:split_point])
        generated = ' '.join(words[split_point:])

        # Skip if generated portion is too short
        if len(generated.split()) < 50:
            continue

        sample = {
            'prompt_id': i,
            'prompt': prompt,
            'generated_text': generated,
            'max_new_tokens': len(generated.split()),  # Approximate
            'source': 'enwiki9',
            'chunk_id': i
        }
        samples.append(sample)

    # Create dataset
    dataset = {
        'metadata': {
            'created_at': datetime.now().isoformat(),
            'source': 'enwiki9',
            'dataset_url':
            'http://download.wikipedia.org/enwiki/20060303/enwiki-20060303-pages-articles.xml.bz2',
            'description':
            'enwiki9 (first 1GB of Wikipedia 2006) - standard compression benchmark',
            'total_samples': len(samples),
            'prompt_source': 'enwiki9',
            'generation_backend': 'wikipedia_extract',
            'chunk_size_target': 'varies',
        },
        'samples': samples
    }

    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Convert enwiki9 to compression dataset format")
    parser.add_argument('--input',
                        type=Path,
                        default=Path('data/enwiki9/enwik9'),
                        help='Path to enwik9 file')
    parser.add_argument('--output',
                        type=Path,
                        default=Path('data/compression_dataset_enwiki9.yaml'),
                        help='Output YAML file path')
    parser.add_argument('--chunk-size',
                        type=int,
                        default=500,
                        help='Target chunk size in words (default: 500)')
    parser.add_argument('--num-chunks',
                        type=int,
                        default=200,
                        help='Number of chunks to extract (default: 200)')

    args = parser.parse_args()

    # Check input exists
    if not args.input.exists():
        print(f"❌ Input file not found: {args.input}")
        print(f"   Run: bash scripts/download_enwiki9.sh")
        return 1

    print(f"Processing enwiki9 from {args.input}")
    print(f"Target: {args.num_chunks} chunks of ~{args.chunk_size} words each")
    print()

    # Extract text chunks
    chunks = extract_text_chunks(args.input, args.chunk_size, args.num_chunks)
    print(f"✓ Extracted {len(chunks)} text chunks")
    print()

    # Show sample
    if chunks:
        print("Sample chunk (first 200 chars):")
        print(chunks[0][:200] + "...")
        print()

    # Create dataset
    print("Creating compression dataset...")
    dataset = create_compression_dataset(chunks, args.output)

    # Save to YAML
    print(f"Saving to {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        yaml.dump(dataset, f, default_flow_style=False, sort_keys=False)

    print(
        f"✓ Created compression dataset with {len(dataset['samples'])} samples"
    )
    print()
    print("Dataset statistics:")
    total_words = sum(
        len(s['generated_text'].split()) for s in dataset['samples'])
    avg_words = total_words / len(dataset['samples'])
    print(f"  Total samples: {len(dataset['samples'])}")
    print(f"  Avg words per sample: {avg_words:.1f}")
    print(f"  Total words to compress: {total_words:,}")
    print()
    print("You can now run:")
    print(f"  python scripts/measure_baselines.py {args.output}")
    print(
        f"  python scripts/measure_compression.py 8b {args.output} --limit-samples 50"
    )

    return 0


if __name__ == '__main__':
    exit(main())
