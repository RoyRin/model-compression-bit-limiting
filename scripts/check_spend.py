#!/usr/bin/env python3
"""Quick script to check API spending from SPEND/ logs."""

import csv
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.api_cost_tracker import ANTHROPIC_PRICING, OPENAI_PRICING

SPEND_DIR = Path(__file__).parent.parent / "SPEND"


def parse_args():
    parser = argparse.ArgumentParser(description="Check API spending")
    parser.add_argument("--days",
                        type=int,
                        default=None,
                        help="Only show spending from last N days")
    parser.add_argument("--daily",
                        action="store_true",
                        help="Show daily breakdown")
    parser.add_argument("--by-model",
                        action="store_true",
                        default=True,
                        help="Show breakdown by model (default)")
    parser.add_argument("--csv",
                        type=str,
                        default=None,
                        help="Path to specific CSV file")
    return parser.parse_args()


def calculate_cost(provider: str, model: str, input_tokens: int,
                   output_tokens: int) -> float:
    """Calculate cost based on pricing dictionaries."""
    if provider.lower() == 'openai':
        pricing = OPENAI_PRICING.get(model, {"input": 0, "output": 0})
    else:
        pricing = ANTHROPIC_PRICING.get(model, {"input": 0, "output": 0})

    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost


def load_spending_data(csv_path: Path,
                       days_filter: int = None,
                       recalculate: bool = True):
    """Load spending data from CSV.

    Args:
        csv_path: Path to CSV file
        days_filter: Only include last N days
        recalculate: Recalculate costs from current pricing (default True)
    """
    data = []
    cutoff = None
    if days_filter:
        cutoff = datetime.now() - timedelta(days=days_filter)

    with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts = datetime.fromisoformat(row['timestamp'])
                if cutoff and ts < cutoff:
                    continue

                provider = row.get('provider', 'anthropic')
                model = row['model']
                input_tokens = int(row['input_tokens'])
                output_tokens = int(row['output_tokens'])

                # Recalculate cost from current pricing or use logged cost
                if recalculate:
                    total_cost = calculate_cost(provider, model, input_tokens,
                                                output_tokens)
                else:
                    total_cost = float(row['total_cost'])

                data.append({
                    'timestamp': ts,
                    'provider': provider,
                    'model': model,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'total_cost': total_cost
                })
            except (ValueError, KeyError) as e:
                continue
    return data


def summarize_by_model(data):
    """Summarize spending by model."""
    by_model = defaultdict(lambda: {
        'cost': 0.0,
        'calls': 0,
        'input_tokens': 0,
        'output_tokens': 0
    })

    for row in data:
        model = row['model']
        by_model[model]['cost'] += row['total_cost']
        by_model[model]['calls'] += 1
        by_model[model]['input_tokens'] += row['input_tokens']
        by_model[model]['output_tokens'] += row['output_tokens']

    return by_model


def summarize_by_day(data):
    """Summarize spending by day."""
    by_day = defaultdict(lambda: {'cost': 0.0, 'calls': 0})

    for row in data:
        day = row['timestamp'].strftime('%Y-%m-%d')
        by_day[day]['cost'] += row['total_cost']
        by_day[day]['calls'] += 1

    return by_day


def format_tokens(n):
    """Format token count with K/M suffix."""
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


def main():
    args = parse_args()

    # Find CSV file
    if args.csv:
        csv_path = Path(args.csv)
    else:
        csv_path = SPEND_DIR / "all_api_spending_log.csv"

    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        return

    # Load data
    data = load_spending_data(csv_path, args.days)

    if not data:
        print("No spending data found.")
        return

    # Calculate totals
    total_cost = sum(row['total_cost'] for row in data)
    total_calls = len(data)
    total_input = sum(row['input_tokens'] for row in data)
    total_output = sum(row['output_tokens'] for row in data)

    # Date range
    min_date = min(row['timestamp'] for row in data)
    max_date = max(row['timestamp'] for row in data)

    print("=" * 70)
    print("API SPENDING SUMMARY")
    print("=" * 70)

    if args.days:
        print(f"Period: Last {args.days} days")
    else:
        print(
            f"Period: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
        )

    print(f"\nTOTAL SPEND: ${total_cost:.2f}")
    print(f"Total API Calls: {total_calls:,}")
    print(
        f"Total Tokens: {format_tokens(total_input)} input, {format_tokens(total_output)} output"
    )

    # By model breakdown
    print("\n" + "-" * 70)
    print("BREAKDOWN BY MODEL")
    print("-" * 70)

    by_model = summarize_by_model(data)
    sorted_models = sorted(by_model.items(),
                           key=lambda x: x[1]['cost'],
                           reverse=True)

    print(f"\n{'Model':<45} {'Cost':>10} {'Calls':>8} {'Tokens':>12}")
    print("-" * 75)

    for model, stats in sorted_models:
        tokens = f"{format_tokens(stats['input_tokens'])}/{format_tokens(stats['output_tokens'])}"
        print(
            f"{model:<45} ${stats['cost']:>8.2f} {stats['calls']:>8,} {tokens:>12}"
        )

    # Daily breakdown if requested
    if args.daily:
        print("\n" + "-" * 70)
        print("DAILY BREAKDOWN")
        print("-" * 70)

        by_day = summarize_by_day(data)
        sorted_days = sorted(by_day.items(), reverse=True)

        print(f"\n{'Date':<15} {'Cost':>10} {'Calls':>8}")
        print("-" * 35)

        for day, stats in sorted_days[:14]:  # Last 14 days
            print(f"{day:<15} ${stats['cost']:>8.2f} {stats['calls']:>8,}")

    print("\n" + "=" * 70)
    print(f"Data from: {csv_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
