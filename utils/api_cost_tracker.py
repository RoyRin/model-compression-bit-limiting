"""API cost tracking for OpenAI and Anthropic APIs."""

import csv
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict

# Pricing per 1M tokens (as of late 2024)
ANTHROPIC_PRICING = {
    # Claude Opus 4.1 models
    "claude-opus-4-1-20250805": {
        "input": 15.00,  # Opus 4.1 pricing - same as Opus 3
        "output": 75.00
    },
    "claude-opus-4-20250514": {
        "input": 15.00,
        "output": 75.00
    },
    # Claude Sonnet 4 models
    "claude-sonnet-4-20250514": {
        "input": 3.00,
        "output": 15.00
    },
    # Claude 3.7 Sonnet
    "claude-3-7-sonnet-20250219": {
        "input": 3.00,  # Claude 3.7 Sonnet pricing - same as 3.5 Sonnet
        "output": 15.00
    },
    # Claude 3.5 models
    "claude-3-5-sonnet-20241022": {
        "input": 3.00,
        "output": 15.00
    },
    "claude-3-5-sonnet-20240620": {
        "input": 3.00,
        "output": 15.00
    },
    "claude-3-5-haiku-20241022": {
        "input": 1.00,  # Claude 3.5 Haiku pricing
        "output": 5.00
    },
    # Claude 4.5 models
    "claude-haiku-4-5-20251001": {
        "input": 1.00,  # Claude 4.5 Haiku - same as 3.5 Haiku
        "output": 5.00
    },
    "claude-sonnet-4-5-20250929": {
        "input": 3.00,  # Claude 4.5 Sonnet - same as 3.5 Sonnet
        "output": 15.00
    },
    "claude-opus-4-5-20251101": {
        "input": 15.00,  # Claude 4.5 Opus - same as Opus 3/4
        "output": 75.00
    },
    # Claude 3 models
    "claude-3-opus-20240229": {
        "input": 15.00,
        "output": 75.00
    },
    "claude-3-sonnet-20240229": {
        "input": 3.00,
        "output": 15.00
    },
    "claude-3-haiku-20240307": {
        "input": 0.25,
        "output": 1.25
    },
    # Claude 2 models
    "claude-2.1": {
        "input": 8.00,
        "output": 24.00
    },
    "claude-2.0": {
        "input": 8.00,
        "output": 24.00
    },
    "claude-instant-1.2": {
        "input": 0.80,
        "output": 2.40
    }
}

OPENAI_PRICING = {
    # GPT-5 models
    "gpt-5-2025-08-07": {
        "input": 30.00,
        "output": 120.00
    },  # Estimated pricing
    "gpt-5": {
        "input": 30.00,
        "output": 120.00
    },  # Estimated pricing
    "gpt-5-mini-2025-08-07": {
        "input": 2.50,
        "output": 10.00
    },  # Estimated pricing - similar to GPT-4o
    "gpt-5-nano-2025-08-07": {
        "input": 0.50,
        "output": 2.00
    },  # Estimated pricing - budget tier
    "gpt-4.1-2025-04-14": {
        "input": 20.00,
        "output": 60.00
    },  # Estimated pricing
    "o4-mini-2025-04-16": {
        "input": 5.00,
        "output": 15.00
    },  # Estimated pricing
    "gpt-5-turbo": {
        "input": 10.00,
        "output": 30.00
    },  # Estimated pricing
    # GPT-4o models
    "gpt-4o": {
        "input": 2.50,
        "output": 10.00
    },
    "gpt-4o-2024-11-20": {
        "input": 2.50,
        "output": 10.00
    },
    "gpt-4o-2024-08-06": {
        "input": 2.50,
        "output": 10.00
    },
    "gpt-4o-2024-05-13": {
        "input": 5.00,
        "output": 15.00
    },
    "gpt-4o-mini": {
        "input": 0.15,
        "output": 0.60
    },
    "gpt-4o-mini-2024-07-18": {
        "input": 0.15,
        "output": 0.60
    },
    # GPT-4 models
    "gpt-4-turbo": {
        "input": 10.00,
        "output": 30.00
    },
    "gpt-4-turbo-2024-04-09": {
        "input": 10.00,
        "output": 30.00
    },
    "gpt-4-turbo-preview": {
        "input": 10.00,
        "output": 30.00
    },
    "gpt-4": {
        "input": 30.00,
        "output": 60.00
    },
    "gpt-4-32k": {
        "input": 60.00,
        "output": 120.00
    },
    # GPT-3.5 models
    "gpt-3.5-turbo": {
        "input": 0.50,
        "output": 1.50
    },
    "gpt-3.5-turbo-0125": {
        "input": 0.50,
        "output": 1.50
    },
    "gpt-3.5-turbo-1106": {
        "input": 1.00,
        "output": 2.00
    },
    "gpt-3.5-turbo-16k": {
        "input": 3.00,
        "output": 4.00
    },
}


# Default log file locations
def get_project_root() -> Path:
    """Get the project root directory."""
    current = Path(__file__).resolve()
    # Go up two levels from utils/api_cost_tracker.py to project root
    return current.parent.parent


SPENDING_LOG_DIR = get_project_root() / "SPEND"
SPENDING_LOG_DIR.mkdir(exist_ok=True)
ANTHROPIC_LOG_FILE = SPENDING_LOG_DIR / "anthropic_spending_log.csv"
OPENAI_LOG_FILE = SPENDING_LOG_DIR / "openai_spending_log.csv"
COMBINED_LOG_FILE = SPENDING_LOG_DIR / "all_api_spending_log.csv"


def init_spending_log(log_file: Path):
    """Initialize the spending log CSV file if it doesn't exist."""
    if not log_file.exists():
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'provider', 'model', 'input_tokens',
                'output_tokens', 'input_cost', 'output_cost', 'total_cost',
                'prompt_preview', 'response_preview'
            ])


def log_api_spending(provider: str,
                     model: str,
                     input_tokens: int,
                     output_tokens: int,
                     prompt: str = "",
                     response: str = "",
                     log_file: Optional[Path] = None):
    """Log API spending to CSV file.
    
    Args:
        provider: 'openai' or 'anthropic'
        model: Model name
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        prompt: The prompt sent (optional)
        response: The response received (optional)
        log_file: Custom log file path (optional)
    
    Returns:
        Total cost for this API call
    """
    # Determine log files
    if log_file is None:
        if provider.lower() == 'openai':
            provider_log = OPENAI_LOG_FILE
        else:
            provider_log = ANTHROPIC_LOG_FILE
    else:
        provider_log = log_file

    # Initialize logs if needed
    init_spending_log(provider_log)
    init_spending_log(COMBINED_LOG_FILE)

    # Get pricing
    if provider.lower() == 'openai':
        pricing = OPENAI_PRICING.get(model, {"input": 0, "output": 0})
    else:
        pricing = ANTHROPIC_PRICING.get(model, {"input": 0, "output": 0})

    # Calculate costs (pricing is per 1M tokens)
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    total_cost = input_cost + output_cost

    # Prepare previews (first 100 chars)
    prompt_preview = prompt[:100].replace('\n', ' ').replace(
        ',', ';') if prompt else ""
    response_preview = response[:100].replace('\n', ' ').replace(
        ',', ';') if response else ""

    # Prepare row data
    row_data = [
        datetime.now().isoformat(), provider, model, input_tokens,
        output_tokens, f"{input_cost:.6f}", f"{output_cost:.6f}",
        f"{total_cost:.6f}", prompt_preview, response_preview
    ]

    # Log to provider-specific file
    with open(provider_log, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row_data)

    # Log to combined file
    with open(COMBINED_LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row_data)

    return total_cost


def get_total_spending(log_file: Optional[Path] = None) -> dict:
    """Get total spending from the log file.
    
    Args:
        log_file: Path to log file. If None, uses combined log.
    
    Returns:
        Dictionary with spending statistics
    """
    if log_file is None:
        log_file = COMBINED_LOG_FILE

    if not log_file.exists():
        return {
            "total": 0.0,
            "by_provider": {},
            "by_model": {},
            "total_input_tokens": 0,
            "total_output_tokens": 0
        }

    total = 0.0
    by_provider = {}
    by_model = {}
    total_input_tokens = 0
    total_output_tokens = 0

    with open(log_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cost = float(row['total_cost'])
            total += cost

            # By provider
            provider = row.get('provider', 'unknown')
            if provider not in by_provider:
                by_provider[provider] = {
                    "cost": 0.0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "calls": 0
                }

            by_provider[provider]["cost"] += cost
            by_provider[provider]["input_tokens"] += int(
                row.get('input_tokens', 0))
            by_provider[provider]["output_tokens"] += int(
                row.get('output_tokens', 0))
            by_provider[provider]["calls"] += 1

            # By model
            model = row['model']
            if model not in by_model:
                by_model[model] = {
                    "cost": 0.0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "calls": 0,
                    "provider": provider
                }

            by_model[model]["cost"] += cost
            by_model[model]["input_tokens"] += int(row.get('input_tokens', 0))
            by_model[model]["output_tokens"] += int(row.get(
                'output_tokens', 0))
            by_model[model]["calls"] += 1

            total_input_tokens += int(row.get('input_tokens', 0))
            total_output_tokens += int(row.get('output_tokens', 0))

    return {
        "total": total,
        "by_provider": by_provider,
        "by_model": by_model,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens
    }


def print_spending_summary(log_file: Optional[Path] = None):
    """Print a formatted summary of API spending.
    
    Args:
        log_file: Path to log file. If None, uses combined log.
    """
    spending = get_total_spending(log_file)

    print("\n" + "=" * 70)
    print("API SPENDING SUMMARY")
    print("=" * 70)

    if spending['total'] == 0:
        print("No API calls logged yet.")
        return

    print(f"\n💰 TOTAL SPENDING: ${spending['total']:.2f}")
    print(
        f"📝 Total Tokens: {spending['total_input_tokens']:,} input, {spending['total_output_tokens']:,} output"
    )

    # By provider
    if spending['by_provider']:
        print("\n🏢 BREAKDOWN BY PROVIDER:")
        print("-" * 70)

        for provider, stats in sorted(spending['by_provider'].items()):
            print(f"\n{provider.upper()}:")
            print(f"  💰 Cost: ${stats['cost']:.2f}")
            print(f"  📞 API Calls: {stats['calls']:,}")
            print(f"  📥 Input Tokens: {stats['input_tokens']:,}")
            print(f"  📤 Output Tokens: {stats['output_tokens']:,}")

            if stats['calls'] > 0:
                avg_cost = stats['cost'] / stats['calls']
                avg_input = stats['input_tokens'] / stats['calls']
                avg_output = stats['output_tokens'] / stats['calls']
                print(
                    f"  📊 Averages per call: ${avg_cost:.4f}, {avg_input:.0f} in, {avg_output:.0f} out"
                )

    # By model
    if spending['by_model']:
        print("\n🤖 BREAKDOWN BY MODEL:")
        print("-" * 70)

        # Sort by cost descending
        sorted_models = sorted(spending['by_model'].items(),
                               key=lambda x: x[1]['cost'],
                               reverse=True)

        for model, stats in sorted_models[:10]:  # Show top 10 models
            print(f"\n{model} ({stats.get('provider', 'unknown')}):")
            print(f"  💰 Cost: ${stats['cost']:.2f}")
            print(f"  📞 API Calls: {stats['calls']:,}")
            print(f"  📥 Input Tokens: {stats['input_tokens']:,}")
            print(f"  📤 Output Tokens: {stats['output_tokens']:,}")

            if stats['calls'] > 0:
                avg_cost = stats['cost'] / stats['calls']
                avg_input = stats['input_tokens'] / stats['calls']
                avg_output = stats['output_tokens'] / stats['calls']
                print(
                    f"  📊 Averages per call: ${avg_cost:.4f}, {avg_input:.0f} in, {avg_output:.0f} out"
                )

    print("\n" + "=" * 70)
    if log_file:
        print(f"Log file: {log_file}")
    else:
        print(f"Log directory: {SPENDING_LOG_DIR}")
    print("=" * 70)


def log_batch_spending(model: str,
                       batch_results: list,
                       description: str = "batch") -> float:
    """Log spending for a batch API request.

    Args:
        model: Model name used in the batch
        batch_results: List of batch result objects from client.messages.batches.results()
        description: Description of the batch for logging

    Returns:
        Total cost for all requests in the batch
    """
    total_cost = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    successful_requests = 0

    for result in batch_results:
        if result.result.type == 'succeeded':
            usage = result.result.message.usage
            input_tokens = usage.input_tokens
            output_tokens = usage.output_tokens

            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            successful_requests += 1

    # Batch API is 50% cheaper
    pricing = ANTHROPIC_PRICING.get(model, {"input": 0, "output": 0})
    batch_discount = 0.5  # 50% discount for batch API

    input_cost = (total_input_tokens /
                  1_000_000) * pricing["input"] * batch_discount
    output_cost = (total_output_tokens /
                   1_000_000) * pricing["output"] * batch_discount
    total_cost = input_cost + output_cost

    # Log to files
    init_spending_log(ANTHROPIC_LOG_FILE)
    init_spending_log(COMBINED_LOG_FILE)

    row_data = [
        datetime.now().isoformat(), 'anthropic', f"{model} (batch)",
        total_input_tokens, total_output_tokens, f"{input_cost:.6f}",
        f"{output_cost:.6f}", f"{total_cost:.6f}",
        f"[BATCH: {successful_requests} requests] {description}", ""
    ]

    with open(ANTHROPIC_LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row_data)

    with open(COMBINED_LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row_data)

    return total_cost


def estimate_cost(provider: str, model: str, input_tokens: int,
                  output_tokens: int) -> float:
    """Estimate cost for a given number of tokens without logging.
    
    Args:
        provider: 'openai' or 'anthropic'
        model: Model name
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
    
    Returns:
        Estimated cost in dollars
    """
    if provider.lower() == 'openai':
        pricing = OPENAI_PRICING.get(model, {"input": 0, "output": 0})
    else:
        pricing = ANTHROPIC_PRICING.get(model, {"input": 0, "output": 0})

    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]

    return input_cost + output_cost


if __name__ == "__main__":
    # Example usage and testing
    print("API Cost Tracker initialized")
    print(f"Spending logs will be saved to: {SPENDING_LOG_DIR}")

    # Print current spending summary
    print_spending_summary()
