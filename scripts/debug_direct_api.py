#!/usr/bin/env python3
"""
Test direct (non-batch) API call to verify the API key and model work.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import anthropic
from utils.llm_api import get_anthropic_key

# Models to test
MODELS = [
    'claude-haiku-4-5-20251001',
    'claude-3-5-haiku-20241022',  # older haiku
]


def test_direct_call(client: anthropic.Anthropic, model: str):
    """Test a direct API call."""
    print(f"\nTesting model: {model}")
    print("-" * 40)

    start = time.time()
    try:
        response = client.messages.create(
            model=model,
            max_tokens=50,
            messages=[{
                'role': 'user',
                'content': 'What is 2+2? Reply with just the number.'
            }])
        elapsed = time.time() - start

        content = response.content[0].text if response.content else "(empty)"
        print(f"  Success! Response: {content}")
        print(f"  Model: {response.model}")
        print(f"  Stop reason: {response.stop_reason}")
        print(f"  Time: {elapsed:.2f}s")
        return True

    except anthropic.APIError as e:
        elapsed = time.time() - start
        print(f"  ERROR: {e}")
        print(f"  Error type: {type(e).__name__}")
        if hasattr(e, 'status_code'):
            print(f"  Status code: {e.status_code}")
        print(f"  Time: {elapsed:.2f}s")
        return False


def main():
    print("=" * 60)
    print("Direct API Test (non-batch)")
    print("=" * 60)

    client = anthropic.Anthropic(api_key=get_anthropic_key())
    print("Client initialized")

    for model in MODELS:
        test_direct_call(client, model)

    print("\n" + "=" * 60)
    print("Direct API tests complete")
    print("=" * 60)


if __name__ == '__main__':
    main()
