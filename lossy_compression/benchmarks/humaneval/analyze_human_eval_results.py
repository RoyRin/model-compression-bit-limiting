#!/usr/bin/env python3
"""
Analyze HumanEval results across different models and categorize problems by difficulty.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict


def load_results_from_folder(results_path: str) -> Dict[str, bool]:
    """
    Load results from a results folder and return a dict of task_id -> passed.
    
    Args:
        results_path: Path to results folder (e.g., "results/claude-3-haiku-20240307/20250914_175801")
    
    Returns:
        Dictionary mapping task_id to whether it passed
    """
    results_path = Path(results_path)

    # Check if the path exists
    if not results_path.exists():
        raise ValueError(f"Results path does not exist: {results_path}")

    # Load the detailed results
    detailed_results_path = results_path / "detailed_results.json"
    if detailed_results_path.exists():
        with open(detailed_results_path, 'r') as f:
            data = json.load(f)

        # Extract task results
        task_results = {}
        for result in data.get('results', []):
            task_id = result['task_id']
            passed = result['passed']
            task_results[task_id] = passed

        return task_results

    # Alternative: load from individual problem files
    problems_dir = results_path / "problems"
    if problems_dir.exists():
        task_results = {}
        for problem_file in problems_dir.glob("*.json"):
            with open(problem_file, 'r') as f:
                problem_data = json.load(f)
                task_id = problem_data['task_id']
                passed = problem_data['passed']
                task_results[task_id] = passed
        return task_results

    raise ValueError(f"Could not find results in {results_path}")


def categorize_problems(haiku_results: Dict[str, bool],
                        sonnet_results: Dict[str, bool],
                        opus_results: Dict[str, bool]) -> Dict[str, List[str]]:
    """
    Categorize problems based on which models solved them.
    
    Categories:
    - easy: All models get it (haiku, sonnet, and opus)
    - medium: Haiku doesn't get it, but sonnet and opus do
    - hard: Only opus gets it
    - very_hard: No model gets it
    - other: Any other combination (e.g., haiku gets it but opus doesn't - unexpected!)
    
    Args:
        haiku_results: Dict of task_id -> passed for Haiku model
        sonnet_results: Dict of task_id -> passed for Sonnet model
        opus_results: Dict of task_id -> passed for Opus model
    
    Returns:
        Dictionary mapping category to list of task IDs
    """
    categories = {
        'easy': [],  # All models pass
        'medium': [],  # Sonnet and Opus pass, Haiku fails
        'hard': [],  # Only Opus passes
        'very_hard': [],  # No model passes
        'other': []  # Unexpected patterns
    }

    # Get all unique task IDs
    all_tasks = set(haiku_results.keys()) | set(sonnet_results.keys()) | set(
        opus_results.keys())

    for task_id in sorted(all_tasks):
        # Get results for each model (default to False if not found)
        haiku_passed = haiku_results.get(task_id, False)
        sonnet_passed = sonnet_results.get(task_id, False)
        opus_passed = opus_results.get(task_id, False)

        # Categorize based on pattern
        if haiku_passed and sonnet_passed and opus_passed:
            categories['easy'].append(task_id)
        elif not haiku_passed and sonnet_passed and opus_passed:
            categories['medium'].append(task_id)
        elif not haiku_passed and not sonnet_passed and opus_passed:
            categories['hard'].append(task_id)
        elif not haiku_passed and not sonnet_passed and not opus_passed:
            categories['very_hard'].append(task_id)
        else:
            # Unexpected patterns (e.g., haiku passes but opus doesn't)
            categories['other'].append(task_id)

    return categories


def analyze_results(
        haiku_path: str,
        sonnet_path: str,
        opus_path: str,
        verbose: bool = True) -> Tuple[Dict[str, List[str]], Dict[str, Dict]]:
    """
    Main analysis function that loads results and categorizes problems.
    
    Args:
        haiku_path: Path to Haiku results folder
        sonnet_path: Path to Sonnet results folder
        opus_path: Path to Opus results folder
        verbose: Whether to print analysis
    
    Returns:
        Tuple of (categories dict, statistics dict)
    """
    # Load results from each model
    if verbose:
        print("Loading results...")
        print(f"  Haiku: {haiku_path}")
        print(f"  Sonnet: {sonnet_path}")
        print(f"  Opus: {opus_path}")

    haiku_results = load_results_from_folder(haiku_path)
    sonnet_results = load_results_from_folder(sonnet_path)
    opus_results = load_results_from_folder(opus_path)

    # Categorize problems
    categories = categorize_problems(haiku_results, sonnet_results,
                                     opus_results)

    # Calculate statistics
    stats = {
        'total_problems':
        len(
            set(haiku_results.keys()) | set(sonnet_results.keys())
            | set(opus_results.keys())),
        'haiku_passed':
        sum(haiku_results.values()),
        'sonnet_passed':
        sum(sonnet_results.values()),
        'opus_passed':
        sum(opus_results.values()),
        'category_counts': {
            k: len(v)
            for k, v in categories.items()
        }
    }

    if verbose:
        print("\n" + "=" * 60)
        print("PROBLEM CATEGORIZATION")
        print("=" * 60)

        for category, task_ids in categories.items():
            if task_ids:  # Only show categories with problems
                print(f"\n{category.upper()} ({len(task_ids)} problems):")
                if category == 'other':
                    # For 'other', show the pattern for each
                    for task_id in task_ids[:5]:  # Show first 5
                        h = "✓" if haiku_results.get(task_id, False) else "✗"
                        s = "✓" if sonnet_results.get(task_id, False) else "✗"
                        o = "✓" if opus_results.get(task_id, False) else "✗"
                        print(f"  {task_id}: Haiku={h}, Sonnet={s}, Opus={o}")
                    if len(task_ids) > 5:
                        print(f"  ... and {len(task_ids) - 5} more")
                else:
                    # For regular categories, just list the task IDs
                    for task_id in task_ids[:10]:  # Show first 10
                        print(f"  {task_id}")
                    if len(task_ids) > 10:
                        print(f"  ... and {len(task_ids) - 10} more")

        print("\n" + "=" * 60)
        print("STATISTICS")
        print("=" * 60)
        print(f"Total problems analyzed: {stats['total_problems']}")
        print(
            f"Haiku passed: {stats['haiku_passed']}/{stats['total_problems']} ({stats['haiku_passed']/stats['total_problems']*100:.1f}%)"
        )
        print(
            f"Sonnet passed: {stats['sonnet_passed']}/{stats['total_problems']} ({stats['sonnet_passed']/stats['total_problems']*100:.1f}%)"
        )
        print(
            f"Opus passed: {stats['opus_passed']}/{stats['total_problems']} ({stats['opus_passed']/stats['total_problems']*100:.1f}%)"
        )
        print("\nCategory distribution:")
        for category, count in stats['category_counts'].items():
            if count > 0:
                pct = count / stats['total_problems'] * 100
                print(f"  {category}: {count} ({pct:.1f}%)")

    return categories, stats


def get_problem_details(categories: Dict[str, List[str]],
                        problems_dir: str,
                        category: str = 'medium') -> List[Dict]:
    """
    Get detailed information about problems in a specific category.
    
    Args:
        categories: Category dict from categorize_problems
        problems_dir: Path to a problems directory to get problem details
        category: Which category to get details for
    
    Returns:
        List of problem details including prompts
    """
    if category not in categories:
        raise ValueError(f"Unknown category: {category}")

    task_ids = categories[category]
    problems_dir = Path(problems_dir)

    details = []
    for task_id in task_ids:
        problem_file = problems_dir / f"{task_id.replace('/', '_')}.json"
        if problem_file.exists():
            with open(problem_file, 'r') as f:
                problem_data = json.load(f)
                details.append({
                    'task_id':
                    task_id,
                    'status':
                    problem_data.get('status'),
                    'solution_preview':
                    problem_data.get('solution', '')[:200] +
                    '...' if problem_data.get('solution') else None
                })

    return details


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze HumanEval results across models")
    parser.add_argument("--haiku",
                        required=True,
                        help="Path to Haiku results folder")
    parser.add_argument("--sonnet",
                        required=True,
                        help="Path to Sonnet results folder")
    parser.add_argument("--opus",
                        required=True,
                        help="Path to Opus results folder")
    parser.add_argument("--quiet",
                        action="store_true",
                        help="Suppress verbose output")
    parser.add_argument("--save", help="Save analysis to JSON file")

    args = parser.parse_args()

    categories, stats = analyze_results(args.haiku,
                                        args.sonnet,
                                        args.opus,
                                        verbose=not args.quiet)

    if args.save:
        output = {
            'categories': categories,
            'statistics': stats,
            'paths': {
                'haiku': args.haiku,
                'sonnet': args.sonnet,
                'opus': args.opus
            }
        }
        with open(args.save, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nAnalysis saved to: {args.save}")
