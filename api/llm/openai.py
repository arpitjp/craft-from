"""OpenAI-compatible API client. Works with OpenAI, Azure, and any
OpenAI-compatible endpoint (e.g. Cursor, local Ollama with --api)."""

import json
import os
import urllib.request
import urllib.error

from api.utils.logging import get_logger

log = get_logger(__name__)

DEFAULT_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-4o-mini"


def call(
    prompt: str,
    api_key: str = None,
    base_url: str = None,
    model: str = None,
) -> dict:
    """Returns {text, input_tokens, output_tokens, model}."""
    key = api_key or os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise ValueError("OPENAI_API_KEY not set")

    url = (base_url or os.getenv("OPENAI_BASE_URL", DEFAULT_URL)).rstrip("/")
    if not url.endswith("/chat/completions"):
        url = url.rstrip("/") + "/chat/completions"

    mdl = model or os.getenv("OPENAI_MODEL", DEFAULT_MODEL)

    payload = {
        "model": mdl,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 2048,
        "response_format": {"type": "json_object"},
    }

    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        error_body = e.read().decode() if e.fp else ""
        log.error(f"OpenAI API error {e.code}: {error_body[:300]}")
        raise RuntimeError(f"OpenAI API returned {e.code}") from e

    try:
        text = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise RuntimeError("Could not parse OpenAI response") from e

    usage = body.get("usage", {})
    return {
        "text": text,
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "model": mdl,
    }
