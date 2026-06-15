USE_ANTHROPIC = True  # True  # Set to False to use OpenAI
from utils.llm_api import anthropic_completion, openai_completion, openrouter_completion


def model_completion(prompt: str, model: str, **kwargs) -> str:
    """Dispatch to appropriate API based on model name."""
    # OpenRouter models (e.g., "openai/gpt-oss-120b")
    if model.startswith("openai/") or model.startswith("openrouter/"):
        return openrouter_completion(prompt=prompt, model=model, **kwargs)
    # OpenAI models
    elif model.startswith("gpt-") or model.startswith(
            "o1-") or model.startswith("o4-"):
        return openai_completion(prompt=prompt, model=model, **kwargs)
    # Default to Anthropic
    else:
        return anthropic_completion(prompt=prompt, model=model, **kwargs)


# Default seed for reproducibility
DEFAULT_SEED = 42

# Default temperatures
DEFAULT_LLM_TEMPERATURE = 1.0
DEFAULT_SLM_TEMPERATURE = 0.0

MODEL_ALIAS_MAP_old = {
    "haiku": "claude-3-5-haiku-20241022",
    "sonnet": "claude-sonnet-4-20250514",
    "opus": "claude-opus-4-20250514",
    "gpt-oss": "openai/gpt-oss-120b",
}
MODEL_ALIAS_MAP = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-5-20250929",
    "opus": "claude-opus-4-5-20251101",
    "gpt-5": "gpt-5-2025-08-07",
    "gpt-oss": "openai/gpt-oss-120b",
}

# Model constants
if USE_ANTHROPIC:
    LLM = "claude-opus-4-5-20251101"
    SLM = "claude-haiku-4-5-20251001"  # Claude Haiku as the small model
    QUESTION_SLM = "claude-sonnet-4-5-20250929"  # Using Sonnet for question generation
else:
    LLM = "gpt-5-2025-08-07"  # or "gpt-4o" for GPT-4
    SLM = "gpt-3.5-turbo"  # Main small model for answer generation
    QUESTION_SLM = "gpt-4o-mini"  # Question-generating small model

LOCAL_SLM = "meta-llama/Meta-Llama-3-8B"
