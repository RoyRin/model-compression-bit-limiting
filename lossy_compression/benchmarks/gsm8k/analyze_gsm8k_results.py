#!/usr/bin/env python3
"""
Analyze GSM8K results to categorize problems into easy, medium, and hard.

Usage:
    python analyze_gsm8k_results.py path/to/gsm8k_results.json
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple


def categorize_gsm8k_problems(results_file: str) -> Dict[str, List[int]]:
    """
    Categorize GSM8K problems based on which models solved them.
    
    Categories:
    - easy: All models get it (haiku, sonnet, and opus)
    - medium: Haiku doesn't get it, but sonnet and opus do
    - hard: Only opus gets it
    - very_hard: No model gets it
    - other: Any other combination (e.g., haiku gets it but opus doesn't - unexpected!)
    
    Args:
        results_file: Path to the combined results JSON file
        
    Returns:
        Dictionary mapping category to list of problem indices
    """
    # Load results
    with open(results_file, 'r') as f:
        data = json.load(f)

    # Extract results for each model
    haiku_results = {}
    sonnet_results = {}
    opus_results = {}

    if 'haiku' in data:
        for result in data['haiku']['results']:
            haiku_results[result['problem_id']] = result['is_correct']

    if 'sonnet' in data:
        for result in data['sonnet']['results']:
            sonnet_results[result['problem_id']] = result['is_correct']

    if 'opus' in data:
        for result in data['opus']['results']:
            opus_results[result['problem_id']] = result['is_correct']

    # Get all problem IDs
    all_problem_ids = set(haiku_results.keys()) | set(
        sonnet_results.keys()) | set(opus_results.keys())

    # Categorize problems
    categories = {
        'easy': [],  # All models pass
        'medium': [],  # Sonnet and Opus pass, Haiku fails
        'hard': [],  # Only Opus passes
        'very_hard': [],  # No model passes
        'other': []  # Unexpected patterns
    }

    for problem_id in sorted(all_problem_ids):
        haiku_passed = haiku_results.get(problem_id, False)
        sonnet_passed = sonnet_results.get(problem_id, False)
        opus_passed = opus_results.get(problem_id, False)

        # Categorize based on pass pattern
        if haiku_passed and sonnet_passed and opus_passed:
            categories['easy'].append(problem_id)
        elif not haiku_passed and sonnet_passed and opus_passed:
            categories['medium'].append(problem_id)
        elif not haiku_passed and not sonnet_passed and opus_passed:
            categories['hard'].append(problem_id)
        elif not haiku_passed and not sonnet_passed and not opus_passed:
            categories['very_hard'].append(problem_id)
        else:
            # Unexpected pattern (e.g., haiku passes but opus doesn't)
            categories['other'].append(problem_id)

    return categories


def get_problem_difficulty_indices(results_file: str) -> Dict[str, List[int]]:
    """
    Get easy, medium, and hard problem indices from GSM8K results.
    This is the main function to call from other scripts.
    
    Args:
        results_file: Path to the combined results JSON file
        
    Returns:
        Dictionary with 'easy', 'medium', 'hard' keys mapping to lists of problem indices
    """
    categories = categorize_gsm8k_problems(results_file)

    # Return just the three main categories
    return {
        'easy': categories['easy'],
        'medium': categories['medium'],
        'hard': categories['hard']
    }


def print_analysis(results_file: str):
    """Print detailed analysis of GSM8K results."""

    # Load results for detailed stats
    with open(results_file, 'r') as f:
        data = json.load(f)

    print(f"\nAnalyzing: {results_file}")
    print("=" * 60)

    # Get categories
    categories = categorize_gsm8k_problems(results_file)

    # Calculate total problems
    total_problems = sum(len(problems) for problems in categories.values())

    # Print model accuracies
    print("\nMODEL ACCURACIES:")
    print("-" * 40)
    for model_name in ['haiku', 'sonnet', 'opus']:
        if model_name in data:
            model_data = data[model_name]
            accuracy = model_data['accuracy']
            correct = model_data['correct_count']
            total = model_data['total_problems']
            print(
                f"{model_name.upper():8} {accuracy:6.1%} ({correct}/{total})")

    # Print category breakdown
    print("\nPROBLEM CATEGORIES:")
    print("-" * 40)
    for category, problems in categories.items():
        count = len(problems)
        if count > 0:
            pct = count / total_problems * 100
            print(f"{category:10} {count:4} problems ({pct:5.1f}%)")

    # Print sample problems from each category
    print("\nSAMPLE PROBLEM IDs:")
    print("-" * 40)
    for category in ['easy', 'medium', 'hard', 'very_hard']:
        if categories[category]:
            sample = categories[category][:5]
            sample_str = ', '.join(str(p) for p in sample)
            if len(categories[category]) > 5:
                sample_str += f", ... ({len(categories[category])-5} more)"
            print(f"{category:10} {sample_str}")

    # Print unexpected patterns if any
    if categories['other']:
        print("\nUNEXPECTED PATTERNS:")
        print("-" * 40)
        print(
            f"Found {len(categories['other'])} problems with unexpected pass/fail patterns"
        )
        print("(e.g., easier model passes but harder model fails)")

        # Show details for first few
        with open(results_file, 'r') as f:
            data = json.load(f)

        for problem_id in categories['other'][:3]:
            results = []
            for model in ['haiku', 'sonnet', 'opus']:
                if model in data:
                    for r in data[model]['results']:
                        if r['problem_id'] == problem_id:
                            results.append(
                                f"{model}:{'✓' if r['is_correct'] else '✗'}")
                            break
            print(f"  Problem {problem_id}: {' '.join(results)}")

    return categories


def main():
    parser = argparse.ArgumentParser(
        description='Analyze GSM8K results to find easy/medium/hard problems')
    parser.add_argument('results_file', help='Path to GSM8K results JSON file')
    parser.add_argument(
        '--output',
        '-o',
        help='Output file to save categorization (JSON format)')

    args = parser.parse_args()

    # Check if file exists
    if not Path(args.results_file).exists():
        print(f"Error: File not found: {args.results_file}")
        return

    # Analyze and print results
    categories = print_analysis(args.results_file)

    # Save categorization if requested
    if args.output:
        save_data = {
            'source_file': args.results_file,
            'categories': categories,
            'summary': {
                'easy_count': len(categories['easy']),
                'medium_count': len(categories['medium']),
                'hard_count': len(categories['hard']),
                'very_hard_count': len(categories['very_hard']),
                'other_count': len(categories['other']),
                'total': sum(len(p) for p in categories.values())
            }
        }

        with open(args.output, 'w') as f:
            json.dump(save_data, f, indent=2)
        print(f"\nCategorization saved to: {args.output}")

    # Print usage hint
    print("\n" + "=" * 60)
    print("TO USE IN OTHER SCRIPTS:")
    print("=" * 60)
    print("from analyze_gsm8k_results import get_problem_difficulty_indices")
    print(f"indices = get_problem_difficulty_indices('{args.results_file}')")
    print("easy = indices['easy']")
    print("medium = indices['medium']")
    print("hard = indices['hard']")


if __name__ == "__main__":
    main()
