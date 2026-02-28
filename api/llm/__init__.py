"""LLM backends for semantic block refinement.

Unified call() dispatches to Gemini, OpenAI, or Anthropic based on provider string.
All backends return: {text, input_tokens, output_tokens, model}
"""

from api.utils.logging import get_logger

log = get_logger(__name__)


def call(prompt: str, provider: str, api_key: str) -> dict:
    """Route to the correct LLM backend.

    provider: "gemini", "openai", "anthropic"
    Returns: {text, input_tokens, output_tokens, model}
    """
    if provider == "gemini":
        from api.llm.gemini import call as _call
        return _call(prompt, api_key=api_key)
    elif provider == "openai":
        from api.llm.openai import call as _call
        return _call(prompt, api_key=api_key)
    elif provider == "anthropic":
        from api.llm.anthropic import call as _call
        return _call(prompt, api_key=api_key)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
