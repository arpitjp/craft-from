"""Anthropic Claude API client."""

import json
import os
import urllib.request
import urllib.error

from api.utils.logging import get_logger

log = get_logger(__name__)

API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = "claude-sonnet-4-20250514"


def call(prompt: str, api_key: str = None, model: str = None) -> dict:
    """Returns {text, input_tokens, output_tokens, model}."""
    key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    mdl = model or os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL)

    payload = {
        "model": mdl,
        "max_tokens": 2048,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
    }

    req = urllib.request.Request(
        API_URL, data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        error_body = e.read().decode() if e.fp else ""
        log.error(f"Anthropic API error {e.code}: {error_body[:300]}")
        raise RuntimeError(f"Anthropic API returned {e.code}") from e

    try:
        text = body["content"][0]["text"]
    except (KeyError, IndexError) as e:
        raise RuntimeError("Could not parse Anthropic response") from e

    usage = body.get("usage", {})
    return {
        "text": text,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "model": mdl,
    }
