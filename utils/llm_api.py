"""LLM API utilities for Anthropic and OpenAI."""

import os
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
import json
from .api_cost_tracker import log_api_spending
import anthropic

openai_models = [
    "gpt-5-2025-08-07"
    "gpt-5-mini-2025-08-07", "gpt-5-nano-2025-08-07", "gpt-4.1-2025-04-14",
    "o4-mini-2025-04-16"

    ###
    #    "gpt-4o",
    #    "gpt-4o-mini"
]

anthropic_models = [
    # Claude 4.5 models (latest)
    "claude-opus-4-5-20251101",
    "claude-sonnet-4-5-20250929",
    "claude-haiku-4-5-20251001",
    # Older models
    "claude-opus-4-1-20250805",
    "claude-opus-4-20250514",
    "claude-sonnet-4-20250514",
    "claude-3-7-sonnet-20250219",
    "claude-3-5-haiku-20241022",
    "claude-3-haiku-20240307"
]


def _get_secrets_dir() -> Path:
    """Get the SECRETS directory path."""
    from . import SECRETS_DIR
    return SECRETS_DIR


def _load_api_key(filename: str) -> str:
    """Load API key from SECRETS directory."""
    secrets_dir = _get_secrets_dir()
    key_path = secrets_dir / filename

    if not key_path.exists():
        raise FileNotFoundError(
            f"API key not found at {key_path}. "
            f"Please create {filename} in the SECRETS/ directory "
            f"with your API key.")

    with open(key_path, 'r') as f:
        key = f.read().strip()

    if not key:
        raise ValueError(f"API key file {filename} is empty. "
                         f"Please add your API key to the file.")

    return key


def get_anthropic_key() -> str:
    """Get Anthropic API key. Uses ANTHROPIC_KEY_FILE env var if set, else anthropic.key."""
    key_file = os.environ.get("ANTHROPIC_KEY_FILE", "anthropic.key")
    return _load_api_key(key_file)


def get_anthropic_low_prio_key() -> str:
    """Get Anthropic low priority API key."""
    return _load_api_key("anthropic_low_prio.key")


def get_openai_key() -> str:
    """Get OpenAI API key from SECRETS/openai.key."""
    return _load_api_key("openai.key")


def get_openrouter_key() -> str:
    """Get OpenRouter API key from SECRETS/openrouter__mats.key."""
    return _load_api_key("openrouter__mats.key")


def check_api_keys() -> Dict[str, bool]:
    """Check if API keys are available and valid."""
    secrets_dir = _get_secrets_dir()
    results = {}

    for provider, filename in [("anthropic", "anthropic.key"),
                               ("openai", "openai.key"),
                               ("openrouter", "openrouter__mats.key")]:
        key_path = secrets_dir / filename
        if key_path.exists():
            try:
                with open(key_path, 'r') as f:
                    key = f.read().strip()
                results[provider] = bool(key)
            except Exception:
                results[provider] = False
        else:
            results[provider] = False

    return results


def anthropic_completion(
        prompt: str,
        model: str = "claude-3-5-sonnet-20241022",
        max_tokens: int = 1024,
        temperature: float = 1.0,
        seed: Optional[int] = None,  # Kept for compatibility but ignored
        system: Optional[str] = None,
        prefill: Optional[str] = None,
        **kwargs) -> str:
    """Call Anthropic API for text completion.
    
    Args:
        prompt: The user prompt
        model: Model to use
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        seed: Ignored - Anthropic API doesn't support seed
        system: System prompt
        prefill: Optional assistant message prefill to guide response format
        **kwargs: Additional parameters
    
    Returns:
        The model's response text (including prefill if provided)
    """

    api_key = get_anthropic_key()
    client = anthropic.Anthropic(api_key=api_key)

    messages = [{"role": "user", "content": prompt}]

    # Add assistant prefill if provided
    if prefill is not None:
        messages.append({"role": "assistant", "content": prefill})

    create_params = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        **kwargs
    }

    if system is not None:
        create_params["system"] = system

    # Anthropic API doesn't support seed - don't add it

    # Retry with exponential backoff on rate limits
    max_retries = 8
    base_delay = 30
    for attempt in range(max_retries + 1):
        try:
            response = client.messages.create(**create_params)
            break
        except Exception as e:
            error_msg = str(e).lower()
            is_rate_limit = ("rate" in error_msg and "limit" in error_msg) or \
                            "429" in error_msg or "too many requests" in error_msg or \
                            "overloaded" in error_msg
            if is_rate_limit and attempt < max_retries:
                delay = base_delay * (2**attempt)  # 30, 60, 120, 240, ...
                print(
                    f"  Rate limited, waiting {delay}s (attempt {attempt+1}/{max_retries})..."
                )
                time.sleep(delay)
                continue
            raise

    # Track spending
    if hasattr(response, 'usage'):
        log_api_spending(provider='anthropic',
                         model=model,
                         input_tokens=response.usage.input_tokens,
                         output_tokens=response.usage.output_tokens,
                         prompt=prompt,
                         response=response.content[0].text)

    # If prefill was used, prepend it to the response
    if prefill is not None:
        return prefill + response.content[0].text
    else:
        return response.content[0].text


def anthropic_continue(
    prompt: str,
    model: str,
    max_tokens: int = 128,
    temperature: float = 0.7,
    seed: Optional[int] = None,
    stop_sequences: Optional[List[str]] = None,
    prefill_assistant: Optional[str] = None,
    system: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    """
    Continue text for up to `max_tokens` using Anthropic Messages API.

    Two modes:
      - Plain continuation:   anthropic_continue_text("...your text...")
      - Prefill continuation: anthropic_continue_text("USER TEXT", prefill_assistant="ASSISTANT: partial...")

    Returns the concatenated text blocks from Claude's reply.
    """
    client = anthropic.Anthropic(api_key=api_key)

    # Build message list
    messages = [{"role": "user", "content": prompt}]
    if prefill_assistant:
        # "Putting words in Claude's mouth": last input is assistant; Claude continues it
        messages.append({"role": "assistant", "content": prefill_assistant})

    create_params = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop_sequences": stop_sequences or None,
    }

    # Only add system if it's not None
    if system is not None:
        create_params["system"] = system

    # Note: seed and metadata parameters are not supported by Anthropic API

    try:
        resp = client.messages.create(**create_params)
    except Exception as e:
        error_msg = str(e).lower()
        # Check for rate limit errors
        if "rate" in error_msg and "limit" in error_msg:
            raise Exception(
                f"RATE_LIMIT_ERROR: Anthropic API rate limit hit. Error: {e}")
        elif "429" in error_msg or "too many requests" in error_msg:
            raise Exception(
                f"RATE_LIMIT_ERROR: Anthropic API rate limit hit (429). Error: {e}"
            )
        else:
            raise  # Re-raise the original exception if not rate limit

    # Join all text blocks from the response
    return "".join(block.text for block in resp.content
                   if block.type == "text").strip()


def openai_completion(prompt: str,
                      model: str = "gpt-4o-mini",
                      max_tokens: int = 1024,
                      temperature: float = 1.0,
                      seed: Optional[int] = None,
                      system: Optional[str] = None,
                      **kwargs) -> str:
    """Call OpenAI API for text completion."""
    from openai import OpenAI
    import time

    api_key = get_openai_key()
    client = OpenAI(api_key=api_key)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    # GPT-5 uses max_completion_tokens instead of max_tokens
    completion_params = {"model": model, "messages": messages, **kwargs}

    if seed is not None:
        completion_params["seed"] = seed

    # GPT-5 doesn't support temperature parameter
    if not model.startswith("gpt-5"):
        completion_params["temperature"] = temperature

    if model.startswith("gpt-5"):
        completion_params["max_completion_tokens"] = max_tokens
    else:
        completion_params["max_tokens"] = max_tokens

    try:
        response = client.chat.completions.create(**completion_params)
    except Exception as e:
        print(f"[OpenAI API] Error with model {model}: {e}")
        raise

    # Track spending
    if hasattr(response, 'usage'):
        log_api_spending(provider='openai',
                         model=model,
                         input_tokens=response.usage.prompt_tokens,
                         output_tokens=response.usage.completion_tokens,
                         prompt=prompt,
                         response=response.choices[0].message.content)

    return response.choices[0].message.content


def openrouter_completion(prompt: str,
                          model: str = "openai/gpt-oss-120b",
                          max_tokens: int = 1024,
                          temperature: float = 1.0,
                          seed: Optional[int] = None,
                          system: Optional[str] = None,
                          site_url: Optional[str] = None,
                          site_name: Optional[str] = None,
                          **kwargs) -> str:
    """Call OpenRouter API for text completion.

    Args:
        prompt: The user prompt
        model: Model to use (e.g., "openai/gpt-oss-120b", "anthropic/claude-3-opus")
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        seed: Random seed for reproducibility (if supported by model)
        system: System prompt
        site_url: Optional site URL for OpenRouter rankings
        site_name: Optional site name for OpenRouter rankings
        **kwargs: Additional parameters

    Returns:
        The model's response text
    """
    from openai import OpenAI

    api_key = get_openrouter_key()

    # Build headers for OpenRouter
    headers = {}
    if site_url:
        headers["HTTP-Referer"] = site_url
    if site_name:
        headers["X-Title"] = site_name

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers=headers if headers else None,
    )

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    completion_params = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        **kwargs
    }

    if seed is not None:
        completion_params["seed"] = seed

    try:
        response = client.chat.completions.create(**completion_params)
    except Exception as e:
        error_msg = str(e).lower()
        if "rate" in error_msg and "limit" in error_msg:
            raise Exception(
                f"RATE_LIMIT_ERROR: OpenRouter API rate limit hit. Error: {e}")
        elif "429" in error_msg or "too many requests" in error_msg:
            raise Exception(
                f"RATE_LIMIT_ERROR: OpenRouter API rate limit hit (429). Error: {e}"
            )
        else:
            print(f"[OpenRouter API] Error with model {model}: {e}")
            raise

    # Track spending
    if hasattr(response, 'usage') and response.usage:
        log_api_spending(provider='openrouter',
                         model=model,
                         input_tokens=response.usage.prompt_tokens,
                         output_tokens=response.usage.completion_tokens,
                         prompt=prompt,
                         response=response.choices[0].message.content or "")

    return response.choices[0].message.content or ""


def openrouter_messages(messages: List[Dict[str, str]],
                        model: str = "openai/gpt-oss-120b",
                        max_tokens: int = 1024,
                        temperature: float = 1.0,
                        seed: Optional[int] = None,
                        site_url: Optional[str] = None,
                        site_name: Optional[str] = None,
                        **kwargs) -> str:
    """Call OpenRouter API with message format.

    Args:
        messages: List of message dictionaries with 'role' and 'content'
        model: Model to use (e.g., "openai/gpt-oss-120b")
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        seed: Random seed for reproducibility (if supported by model)
        site_url: Optional site URL for OpenRouter rankings
        site_name: Optional site name for OpenRouter rankings
        **kwargs: Additional parameters

    Returns:
        The model's response text
    """
    from openai import OpenAI

    api_key = get_openrouter_key()

    headers = {}
    if site_url:
        headers["HTTP-Referer"] = site_url
    if site_name:
        headers["X-Title"] = site_name

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers=headers if headers else None,
    )

    completion_params = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        **kwargs
    }

    if seed is not None:
        completion_params["seed"] = seed

    try:
        response = client.chat.completions.create(**completion_params)
    except Exception as e:
        error_msg = str(e).lower()
        if "rate" in error_msg and "limit" in error_msg:
            raise Exception(
                f"RATE_LIMIT_ERROR: OpenRouter API rate limit hit. Error: {e}")
        elif "429" in error_msg or "too many requests" in error_msg:
            raise Exception(
                f"RATE_LIMIT_ERROR: OpenRouter API rate limit hit (429). Error: {e}"
            )
        else:
            raise

    # Track spending
    if hasattr(response, 'usage') and response.usage:
        prompt_text = ' '.join([m.get('content', '') for m in messages])
        log_api_spending(provider='openrouter',
                         model=model,
                         input_tokens=response.usage.prompt_tokens,
                         output_tokens=response.usage.completion_tokens,
                         prompt=prompt_text[:1000],
                         response=response.choices[0].message.content or "")

    return response.choices[0].message.content or ""


def anthropic_messages(
        messages: List[Dict[str, str]],
        model: str = "claude-3-5-sonnet-20241022",
        max_tokens: int = 1024,
        temperature: float = 1.0,
        seed: Optional[int] = None,  # Kept for compatibility but ignored
        system: Optional[str] = None,
        prefill: Optional[str] = None,
        **kwargs) -> str:
    """Call Anthropic API with message format.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        model: Model to use
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        seed: Ignored - Anthropic API doesn't support seed
        system: System prompt
        prefill: Optional assistant message prefill to guide response format
        **kwargs: Additional parameters
        
    Returns:
        The model's response text (including prefill if provided)
    """
    import anthropic

    api_key = get_anthropic_key()
    client = anthropic.Anthropic(api_key=api_key)

    # If prefill is provided, add it as the last assistant message
    if prefill is not None:
        messages = messages.copy()  # Don't modify the original
        messages.append({"role": "assistant", "content": prefill})

    create_params = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        **kwargs
    }

    if system is not None:
        create_params["system"] = system

    # Anthropic API doesn't support seed - don't add it

    response = client.messages.create(**create_params)

    # Track spending
    if hasattr(response, 'usage'):
        prompt_text = ' '.join([m.get('content', '') for m in messages])
        log_api_spending(
            provider='anthropic',
            model=model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            prompt=prompt_text[:1000],  # Truncate for logging
            response=response.content[0].text)

    # If prefill was used, prepend it to the response
    if prefill is not None:
        return prefill + response.content[0].text
    else:
        return response.content[0].text


def openai_messages(messages: List[Dict[str, str]],
                    model: str = "gpt-4o-mini",
                    max_tokens: int = 1024,
                    temperature: float = 1.0,
                    seed: Optional[int] = None,
                    **kwargs) -> str:
    """Call OpenAI API with message format."""
    from openai import OpenAI

    api_key = get_openai_key()
    client = OpenAI(api_key=api_key)

    # GPT-5 uses max_completion_tokens instead of max_tokens
    completion_params = {"model": model, "messages": messages, **kwargs}

    if seed is not None:
        completion_params["seed"] = seed

    # GPT-5 doesn't support temperature parameter
    if not model.startswith("gpt-5"):
        completion_params["temperature"] = temperature

    if model.startswith("gpt-5"):
        completion_params["max_completion_tokens"] = max_tokens
    else:
        completion_params["max_tokens"] = max_tokens

    try:
        response = client.chat.completions.create(**completion_params)
    except Exception as e:
        error_msg = str(e).lower()
        # Check for rate limit errors
        if "rate" in error_msg and "limit" in error_msg:
            raise Exception(
                f"RATE_LIMIT_ERROR: OpenAI API rate limit hit. Error: {e}")
        elif "429" in error_msg or "too many requests" in error_msg:
            raise Exception(
                f"RATE_LIMIT_ERROR: OpenAI API rate limit hit (429). Error: {e}"
            )
        else:
            raise  # Re-raise the original exception if not rate limit

    # Track spending
    if hasattr(response, 'usage'):
        log_api_spending(provider='openai',
                         model=model,
                         input_tokens=response.usage.prompt_tokens,
                         output_tokens=response.usage.completion_tokens,
                         prompt=prompt,
                         response=response.choices[0].message.content)

    return response.choices[0].message.content


def anthropic_stream(prompt: str,
                     model: str = "claude-3-5-sonnet-20241022",
                     max_tokens: int = 1024,
                     temperature: float = 1.0,
                     system: Optional[str] = None,
                     **kwargs):
    """Stream responses from Anthropic API."""
    import anthropic

    api_key = get_anthropic_key()
    client = anthropic.Anthropic(api_key=api_key)

    messages = [{"role": "user", "content": prompt}]

    stream_params = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        **kwargs
    }

    if system is not None:
        stream_params["system"] = system

    with client.messages.stream(**stream_params) as stream:
        for text in stream.text_stream:
            yield text


def openai_stream(prompt: str,
                  model: str = "gpt-4o-mini",
                  max_tokens: int = 1024,
                  temperature: float = 1.0,
                  seed: Optional[int] = None,
                  system: Optional[str] = None,
                  **kwargs):
    """Stream responses from OpenAI API."""
    from openai import OpenAI

    api_key = get_openai_key()
    client = OpenAI(api_key=api_key)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    # GPT-5 uses max_completion_tokens instead of max_tokens
    completion_params = {
        "model": model,
        "messages": messages,
        "seed": seed,
        "stream": True,
        **kwargs
    }

    # GPT-5 doesn't support temperature parameter
    if not model.startswith("gpt-5"):
        completion_params["temperature"] = temperature

    if model.startswith("gpt-5"):
        completion_params["max_completion_tokens"] = max_tokens
    else:
        completion_params["max_tokens"] = max_tokens

    stream = client.chat.completions.create(**completion_params)

    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content


def anthropic_batch_submit_and_wait(
    client: anthropic.Anthropic,
    requests: List[Dict],
    description: str = "batch",
    poll_interval: int = 5,
) -> Dict[str, str]:
    """Submit batch requests and wait for completion with spend tracking.

    Args:
        client: Anthropic client instance
        requests: List of batch request dicts with custom_id and params
        description: Description for logging
        poll_interval: Seconds between status checks

    Returns:
        Dict mapping custom_id -> response text
    """
    import time
    from .api_cost_tracker import log_batch_spending

    if not requests:
        return {}

    # Submit batch
    batch = client.messages.batches.create(requests=requests)
    batch_id = batch.id

    # Get model from first request for logging
    model = requests[0]['params'].get('model', 'unknown')

    # Poll for completion
    while True:
        status = client.messages.batches.retrieve(batch_id)

        if status.processing_status == 'ended':
            break

        time.sleep(poll_interval)

    # Retrieve results and track spending
    results = {}
    all_results = list(client.messages.batches.results(batch_id))

    for result in all_results:
        if result.result.type == 'succeeded':
            content = result.result.message.content[0].text
            results[result.custom_id] = content

    # Log batch spending
    log_batch_spending(model, all_results, description)

    return results


def get_available_models(provider: str = "all") -> Dict[str, List[str]]:
    """Return available models for each provider."""
    models = {
        "anthropic": [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
        ],
        "openai": [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo",
            "o1-preview",
            "o1-mini",
        ]
    }

    if provider == "all":
        return models
    elif provider in models:
        return {provider: models[provider]}
    else:
        raise ValueError(f"Unknown provider: {provider}. "
                         f"Use 'anthropic', 'openai', or 'all'.")
