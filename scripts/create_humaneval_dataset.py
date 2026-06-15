#!/usr/bin/env python3
"""Convert HumanEval dataset to compression dataset format.

This script converts the HumanEval coding benchmark dataset to the format
used by measure_compression.py and measure_baselines.py.

Usage:
    python scripts/create_humaneval_dataset.py
    python scripts/create_humaneval_dataset.py --output data/compression_dataset_humaneval.yaml
"""

import json
import gzip
from pathlib import Path
from datetime import datetime
import yaml
import argparse


def load_humaneval(filepath: Path):
    """Load HumanEval dataset from gzipped JSONL file."""
    problems = []
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        for line in f:
            problems.append(json.loads(line.strip()))
    return problems


def convert_to_compression_dataset(problems, max_problems=None):
    """Convert HumanEval to compression dataset format.

    Args:
        problems: List of HumanEval problem dicts
        max_problems: Optional limit on number of problems to include

    Returns:
        Dictionary in compression dataset format
    """
    if max_problems:
        problems = problems[:max_problems]

    samples = []
    for i, problem in enumerate(problems):
        # Use the prompt (function signature + docstring) as context
        # Use the canonical solution as the text to compress
        sample = {
            'prompt_id': i,
            'task_id': problem['task_id'],
            'prompt': problem['prompt'],
            'generated_text': problem['canonical_solution'],
            'max_new_tokens':
            len(problem['canonical_solution'].split()),  # Approximate
            'entry_point': problem['entry_point']
        }
        samples.append(sample)

    # Create metadata
    dataset = {
        'metadata': {
            'created_at': datetime.now().isoformat(),
            'source': 'HumanEval',
            'dataset_url': 'https://github.com/openai/human-eval',
            'description':
            'HumanEval coding benchmark - using canonical solutions as compression targets',
            'total_samples': len(samples),
            'prompt_source': 'humaneval',
            'generation_backend': 'humaneval_canonical',
        },
        'samples': samples
    }

    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Convert HumanEval to compression dataset format")
    parser.add_argument('--input',
                        type=Path,
                        default=Path('data/HumanEval.jsonl.gz'),
                        help='Path to HumanEval.jsonl.gz file')
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/compression_dataset_humaneval.yaml'),
        help='Output YAML file path')
    parser.add_argument(
        '--max-problems',
        type=int,
        default=None,
        help='Maximum number of problems to include (default: all)')

    args = parser.parse_args()

    # Check input exists
    if not args.input.exists():
        print(f"❌ Input file not found: {args.input}")
        print(
            f"   Download with: curl -L -o {args.input} https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"
        )
        return 1

    # Load and convert
    print(f"Loading HumanEval from {args.input}")
    problems = load_humaneval(args.input)
    print(f"  Loaded {len(problems)} problems")

    print(f"Converting to compression dataset format...")
    dataset = convert_to_compression_dataset(problems, args.max_problems)

    if args.max_problems:
        print(f"  Limited to {args.max_problems} problems")

    # Save to YAML
    print(f"Saving to {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        yaml.dump(dataset, f, default_flow_style=False, sort_keys=False)

    print(
        f"✓ Created compression dataset with {len(dataset['samples'])} samples"
    )
    print(f"\nYou can now run:")
    print(f"  python scripts/measure_baselines.py {args.output}")
    print(f"  python scripts/measure_compression.py 8b {args.output}")

    return 0


if __name__ == '__main__':
    exit(main())
