#!/usr/bin/env python3
"""
Extract headline QA compression results for paper tables.

Usage:
    python extract_headline_results.py                    # Both versions
    python extract_headline_results.py --version v4.5    # Specific version
    python extract_headline_results.py --latex           # Include LaTeX output
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

DATASETS = [
    'gsm8k', 'math_algebra', 'math_geometry', 'math_number_theory', 'gpqa_mc',
    'mbpp', 'aime', 'hle'
]

DATASET_NAMES = {
    'gsm8k': 'GSM8K',
    'math_algebra': 'MATH (Algebra)',
    'math_geometry': 'MATH (Geometry)',
    'math_number_theory': 'MATH (Num. Theory)',
    'gpqa_mc': 'GPQA (MC)',
    'gpqa_freeform': 'GPQA (Freeform)',
    'mbpp': 'MBPP',
    'aime': 'AIME',
    'hle': 'HLE',
}

# Configurations to compare
CONFIGS = {
    'BLC':
    ('haiku', 'haiku', 'haiku'),  # Bit-Limited COT: SLM, LLM, Q all haiku
    'QA':
    ('haiku', 'opus', 'haiku'),  # QA Compression: haiku asks, opus answers
}


def load_baseline_difficulties(dataset: str,
                               baseline_dir: Path) -> Dict[int, str]:
    """Load problem difficulties from baseline file."""
    for pattern in [f"{dataset}_v*.json", f"{dataset}.json"]:
        files = list(baseline_dir.glob(pattern))
        if files:
            with open(files[0]) as f:
                data = json.load(f)
            return {
                r.get('problem_idx', r.get('idx')):
                r.get('difficulty', 'unknown')
                for r in data['results']
            }
    return {}


def load_qa_results(results_dir: Path,
                    dataset: str) -> Dict[Tuple[str, str, str], List]:
    """Load QA sweep results for a dataset."""
    results = {}
    patterns = [f"{dataset}_SLM-*.json", f"{dataset}_v*_SLM-*.json"]

    json_files = []
    for pattern in patterns:
        json_files.extend(results_dir.glob(pattern))
    json_files = [f for f in json_files if 'summary' not in f.name]
    json_files = list(set(json_files))

    for json_file in json_files:
        try:
            with open(json_file) as f:
                data = json.load(f)

            parts = json_file.stem.split('_')
            slm = llm = q_model = None
            for part in parts:
                if part.startswith('SLM-'):
                    slm = part[4:]
                elif part.startswith('LLM-'):
                    llm = part[4:]
                elif part.startswith('Q-'):
                    q_model = part[2:]

            if slm and llm and q_model and 'oracle' not in llm:
                key = (slm, llm, q_model)
                results[key] = data.get('problems', data.get('results', []))
        except Exception:
            pass

    return results


def compute_accuracy(results: List, difficulties: Dict[int, str],
                     diff_list: List[str]) -> Tuple[Optional[float], int]:
    """Compute accuracy for specific difficulty levels."""
    correct = total = 0
    for pr in results:
        idx = pr.get('problem_idx') or pr.get('idx')
        diff = pr.get('difficulty', difficulties.get(idx, 'unknown'))
        if diff in diff_list:
            total += 1
            if pr.get('final_correct', False):
                correct += 1

    if total > 0:
        return (correct / total * 100, total)
    return (None, 0)


def extract_results(version: str) -> Dict:
    """Extract all results for a version."""
    baseline_dir = Path(f'results/model-baselines/{version}')
    results_base = Path(f'results/qa-sweep/{version}')

    all_results = {}

    for dataset in DATASETS:
        results_dir = results_base / f'{dataset}_qa_sweep' / 'data'
        if not results_dir.exists():
            continue

        difficulties = load_baseline_difficulties(dataset, baseline_dir)
        qa_results = load_qa_results(results_dir, dataset)

        if not qa_results:
            continue

        all_results[dataset] = {
            'difficulties': difficulties,
            'qa_results': qa_results
        }

    return all_results


def print_table(version: str, all_results: Dict):
    """Print results table with separate columns for each difficulty."""
    print(f"\n{'='*130}")
    print(f"QA COMPRESSION HEADLINE RESULTS ({version})")
    print(f"{'='*130}")
    print("BLC = Bit-Limited COT (haiku→haiku→haiku)")
    print(
        "QA  = QA Compression  (haiku→opus→haiku)  [haiku asks, opus answers]")
    print()

    # Header
    print(
        f"{'Dataset':<20} | {'Medium':^21} | {'Hard':^21} | {'Very Hard':^21} | {'All Non-Easy':^21}"
    )
    print(
        f"{'':20} | {'n':>4} {'BLC':>7} {'QA':>7} | {'n':>4} {'BLC':>7} {'QA':>7} | {'n':>4} {'BLC':>7} {'QA':>7} | {'n':>4} {'BLC':>7} {'QA':>7}"
    )
    print("-" * 130)

    for dataset in DATASETS:
        if dataset not in all_results:
            continue

        difficulties = all_results[dataset]['difficulties']
        qa_results = all_results[dataset]['qa_results']

        blc_key = CONFIGS['BLC']
        qa_key = CONFIGS['QA']

        if blc_key not in qa_results or qa_key not in qa_results:
            continue

        row_parts = [f"{DATASET_NAMES.get(dataset, dataset):<20}"]

        for diff_list in [['medium'], ['hard'], ['very_hard'],
                          ['medium', 'hard', 'very_hard']]:
            blc_acc, n = compute_accuracy(qa_results[blc_key], difficulties,
                                          diff_list)
            qa_acc, _ = compute_accuracy(qa_results[qa_key], difficulties,
                                         diff_list)

            def fmt(x):
                return f"{x:5.1f}%" if x is not None else "   -- "

            row_parts.append(f"{n:>4} {fmt(blc_acc)} {fmt(qa_acc)}")

        print(" | ".join(row_parts))

    print("-" * 130)


def print_latex_table(version: str, all_results: Dict):
    """Print LaTeX table."""
    print(f"\n% LaTeX table for {version}")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(
        r"\caption{QA Compression Results (" + version +
        r"). BLC = Bit-Limited COT (haiku$\to$haiku$\to$haiku), QA = QA Compression (haiku$\to$opus$\to$haiku).}"
    )
    print(r"\label{tab:qa-results-" + version.replace('.', '') + "}")
    print(r"\small")
    print(r"\begin{tabular}{l rr rr rr rr}")
    print(r"\toprule")
    print(
        r" & \multicolumn{2}{c}{Medium} & \multicolumn{2}{c}{Hard} & \multicolumn{2}{c}{Very Hard} & \multicolumn{2}{c}{All Non-Easy} \\"
    )
    print(
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}"
    )
    print(r"Dataset & BLC & QA & BLC & QA & BLC & QA & BLC & QA \\")
    print(r"\midrule")

    for dataset in DATASETS:
        if dataset not in all_results:
            continue

        difficulties = all_results[dataset]['difficulties']
        qa_results = all_results[dataset]['qa_results']

        blc_key = CONFIGS['BLC']
        qa_key = CONFIGS['QA']

        if blc_key not in qa_results or qa_key not in qa_results:
            continue

        def fmt(x):
            return f"{x:.1f}" if x is not None else "--"

        values = []
        for diff_list in [['medium'], ['hard'], ['very_hard'],
                          ['medium', 'hard', 'very_hard']]:
            blc_acc, _ = compute_accuracy(qa_results[blc_key], difficulties,
                                          diff_list)
            qa_acc, _ = compute_accuracy(qa_results[qa_key], difficulties,
                                         diff_list)
            values.extend([fmt(blc_acc), fmt(qa_acc)])

        name = DATASET_NAMES.get(dataset, dataset)
        print(f"{name} & {' & '.join(values)} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract QA compression headline results')
    parser.add_argument('--version',
                        type=str,
                        choices=['v3.5', 'v4.5', 'both'],
                        default='both',
                        help='Model version to analyze')
    parser.add_argument('--latex',
                        action='store_true',
                        help='Also print LaTeX tables')
    args = parser.parse_args()

    versions = ['v4.5', 'v3.5'] if args.version == 'both' else [args.version]

    for version in versions:
        all_results = extract_results(version)
        print_table(version, all_results)

        if args.latex:
            print_latex_table(version, all_results)


if __name__ == '__main__':
    main()
