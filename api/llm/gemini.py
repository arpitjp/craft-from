"""Gemini API client. Returns structured result with token usage."""

import json
import os
import random
import time
import urllib.request
import urllib.error

from api.utils.logging import get_logger

log = get_logger(__name__)

DEFAULT_MODEL = "gemini-2.5-flash-lite"
API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
MAX_RETRIES = 3
INITIAL_DELAY = 15
MAX_DELAY = 65


def call(prompt: str, api_key: str = None, model: str = None) -> dict:
    """Returns {text, input_tokens, output_tokens, model, latency_ms, retries}."""
    key = api_key or os.getenv("GEMINI_API_KEY", "")
    if not key:
        raise ValueError("GEMINI_API_KEY not set")

    model_id = model or os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
    url = f"{API_BASE}/{model_id}:generateContent?key={key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 1024,
            "responseMimeType": "application/json",
        },
    }

    body = None
    retries_used = 0
    t0 = time.monotonic()

    for attempt in range(MAX_RETRIES):
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = json.loads(resp.read().decode())
            break
        except urllib.error.HTTPError as e:
            error_body = e.read().decode() if e.fp else ""
            retries_used = attempt + 1

            if e.code in (429, 503) and attempt < MAX_RETRIES - 1:
                retry_after = e.headers.get("Retry-After") if e.headers else None
                if retry_after and retry_after.isdigit():
                    wait = int(retry_after) + random.uniform(1, 3)
                else:
                    wait = min(INITIAL_DELAY * (2 ** attempt) + random.uniform(0, 5), MAX_DELAY)
                label = "Rate limited" if e.code == 429 else "Service unavailable"
                log.warning(f"{label}, retry in {wait:.1f}s ({attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait)
                continue

            log.error(f"Gemini API error {e.code}: {error_body[:300]}")
            raise RuntimeError(f"Gemini API returned {e.code}") from e

    latency_ms = int((time.monotonic() - t0) * 1000)

    if body is None:
        raise RuntimeError("Gemini API: no response after retries")

    try:
        text = body["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as e:
        raise RuntimeError("Could not parse Gemini response") from e

    usage = body.get("usageMetadata", {})
    return {
        "text": text,
        "input_tokens": usage.get("promptTokenCount", 0),
        "output_tokens": usage.get("candidatesTokenCount", 0),
        "model": model_id,
        "latency_ms": latency_ms,
        "retries": retries_used,
    }
