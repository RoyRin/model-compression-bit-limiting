#!/usr/bin/env python3
"""
Debug script to test Anthropic Message Batches API.

Tests:
1. Submit a minimal batch (1-2 requests)
2. Poll with verbose status output
3. Download and display results

Usage:
    python scripts/debug_batch_api.py
    python scripts/debug_batch_api.py --batch-id msgbatch_xxx  # Resume polling existing batch
"""

import argparse
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import anthropic
from utils.llm_api import get_anthropic_key

POLL_INTERVAL = 10  # Poll every 10 seconds for debugging


def submit_test_batch(client: anthropic.Anthropic) -> str:
    """Submit a minimal test batch."""
    requests = [
        {
            'custom_id': 'test_request_1',
            'params': {
                'model':
                'claude-haiku-4-5-20251001',
                'max_tokens':
                100,
                'messages': [{
                    'role':
                    'user',
                    'content':
                    'What is 2+2? Reply with just the number.'
                }],
            },
        },
        {
            'custom_id': 'test_request_2',
            'params': {
                'model':
                'claude-haiku-4-5-20251001',
                'max_tokens':
                100,
                'messages': [{
                    'role':
                    'user',
                    'content':
                    'What is the capital of France? Reply with just the city name.'
                }],
            },
        },
    ]

    print(f"Submitting batch with {len(requests)} requests...")
    batch = client.messages.batches.create(requests=requests)
    print(f"Batch submitted!")
    print(f"  Batch ID: {batch.id}")
    print(f"  Created at: {batch.created_at}")
    print(f"  Processing status: {batch.processing_status}")
    return batch.id


def poll_batch_verbose(client: anthropic.Anthropic, batch_id: str) -> dict:
    """Poll batch with verbose output."""
    print(f"\nPolling batch: {batch_id}")
    print("-" * 60)

    poll_count = 0
    start_time = time.time()

    while True:
        poll_count += 1
        elapsed = time.time() - start_time

        try:
            batch = client.messages.batches.retrieve(batch_id)

            status = batch.processing_status
            counts = batch.request_counts

            print(f"\n[Poll #{poll_count}] Elapsed: {elapsed:.0f}s")
            print(f"  Status: {status}")
            print(f"  Request counts:")
            print(f"    - processing: {counts.processing}")
            print(f"    - succeeded: {counts.succeeded}")
            print(f"    - errored: {counts.errored}")
            print(f"    - canceled: {counts.canceled}")
            print(f"    - expired: {counts.expired}")

            # Also print raw batch object attributes
            print(f"  Batch attributes:")
            print(f"    - id: {batch.id}")
            print(f"    - type: {batch.type}")
            print(f"    - created_at: {batch.created_at}")
            if hasattr(batch, 'ended_at'):
                print(f"    - ended_at: {batch.ended_at}")
            if hasattr(batch, 'expires_at'):
                print(f"    - expires_at: {batch.expires_at}")
            if hasattr(batch, 'results_url'):
                print(f"    - results_url: {batch.results_url}")

            if status == 'ended':
                print("\n" + "=" * 60)
                print("BATCH COMPLETED!")
                print("=" * 60)
                return {
                    'status': status,
                    'succeeded': counts.succeeded,
                    'errored': counts.errored,
                    'elapsed': elapsed,
                }

            # Check for other terminal states
            if status in ['canceled', 'expired']:
                print(f"\nBatch ended with status: {status}")
                return {
                    'status': status,
                    'succeeded': counts.succeeded,
                    'errored': counts.errored,
                    'elapsed': elapsed,
                }

        except anthropic.APIError as e:
            print(f"\n[Poll #{poll_count}] API Error: {e}")
            print(f"  Error type: {type(e).__name__}")
            if hasattr(e, 'status_code'):
                print(f"  Status code: {e.status_code}")

        print(f"\nWaiting {POLL_INTERVAL}s before next poll...")
        time.sleep(POLL_INTERVAL)


def download_results(client: anthropic.Anthropic, batch_id: str) -> dict:
    """Download and display batch results."""
    print(f"\nDownloading results for batch: {batch_id}")
    print("-" * 60)

    results = {}
    result_count = 0

    try:
        for result in client.messages.batches.results(batch_id):
            result_count += 1
            print(f"\nResult #{result_count}:")
            print(f"  Custom ID: {result.custom_id}")
            print(f"  Result type: {result.result.type}")

            if result.result.type == 'succeeded':
                message = result.result.message
                content = message.content[
                    0].text if message.content else "(empty)"
                print(f"  Model: {message.model}")
                print(f"  Stop reason: {message.stop_reason}")
                print(f"  Content: {content}")
                results[result.custom_id] = content
            elif result.result.type == 'errored':
                error = result.result.error
                print(f"  Error type: {error.type}")
                print(f"  Error message: {error.message}")
                results[result.custom_id] = f"ERROR: {error.message}"
            else:
                print(f"  Unknown result type: {result.result}")
                results[result.custom_id] = f"UNKNOWN: {result.result.type}"

    except anthropic.APIError as e:
        print(f"Error downloading results: {e}")

    print(f"\nTotal results downloaded: {result_count}")
    return results


def list_batches(client: anthropic.Anthropic, limit: int = 10):
    """List recent batches."""
    print(f"\nListing recent batches (limit={limit}):")
    print("-" * 60)

    try:
        batches = client.messages.batches.list(limit=limit)
        for i, batch in enumerate(batches.data):
            counts = batch.request_counts
            print(f"\n{i+1}. Batch ID: {batch.id}")
            print(f"   Status: {batch.processing_status}")
            print(f"   Created: {batch.created_at}")
            print(
                f"   Requests: {counts.processing} processing, {counts.succeeded} succeeded, {counts.errored} errored"
            )
    except anthropic.APIError as e:
        print(f"Error listing batches: {e}")


def main():
    parser = argparse.ArgumentParser(description='Debug Anthropic Batch API')
    parser.add_argument('--batch-id',
                        type=str,
                        default=None,
                        help='Resume polling an existing batch ID')
    parser.add_argument('--list',
                        action='store_true',
                        help='List recent batches')
    parser.add_argument('--download-only',
                        action='store_true',
                        help='Only download results (skip polling)')
    args = parser.parse_args()

    print("=" * 60)
    print("Anthropic Message Batches API Debug Script")
    print("=" * 60)

    client = anthropic.Anthropic(api_key=get_anthropic_key())
    print(f"Client initialized")

    if args.list:
        list_batches(client)
        return

    if args.batch_id:
        batch_id = args.batch_id
        print(f"Using existing batch ID: {batch_id}")
    else:
        batch_id = submit_test_batch(client)

    if not args.download_only:
        result = poll_batch_verbose(client, batch_id)
        print(f"\nPoll result: {result}")

    results = download_results(client, batch_id)

    print("\n" + "=" * 60)
    print("FINAL RESULTS:")
    print("=" * 60)
    for custom_id, content in results.items():
        print(f"  {custom_id}: {content}")


if __name__ == '__main__':
    main()
