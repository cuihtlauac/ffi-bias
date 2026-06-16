"""
api_client.py — minimal Anthropic Messages API wrapper used by both the
experiment runner and the judge. Reads ANTHROPIC_API_KEY from the environment.
"""

import os, json, time, urllib.request, urllib.error

BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1/messages")
ANTHROPIC_VERSION = "2023-06-01"


class APIError(RuntimeError):
    pass


def call_model(model: str, user: str, system: str = "", temperature: float = 0.7,
               max_tokens: int = 4000, retries: int = 4) -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise APIError("Set ANTHROPIC_API_KEY in the environment.")

    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": user}],
    }
    if system:
        payload["system"] = system

    data = json.dumps(payload).encode()
    headers = {
        "content-type": "application/json",
        "x-api-key": key,
        "anthropic-version": ANTHROPIC_VERSION,
    }

    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(BASE_URL, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode())
            # concatenate all text blocks
            return "".join(b.get("text", "") for b in body.get("content", [])
                           if b.get("type") == "text")
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429, 500, 502, 503, 529):
                time.sleep(min(2 ** attempt + 0.5, 30))
                continue
            raise APIError(f"HTTP {e.code}: {e.read().decode()[:300]}")
        except urllib.error.URLError as e:
            last = e
            time.sleep(min(2 ** attempt + 0.5, 30))
    raise APIError(f"exhausted retries: {last}")
