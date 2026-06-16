"""
api_client.py — minimal multi-vendor completion wrapper.

ONE stable interface for the rest of the harness:

  call_model(model, user, ...) -> str          # text only (judge.py relies on this)
  call_model_meta(model, user, ...) -> (str, dict)  # text + provenance (run_experiment.py)

Behind it, one thin adapter per vendor. The vendor is inferred from the model id
("claude-*" -> Anthropic, "gemini-*" -> Google); callers never branch on vendor.
Each adapter owns its own wire format, auth, retryable status codes, AND its own
parameter quirks (e.g. Opus 4.8/4.7 reject `temperature`; Gemini accepts it), so
those differences never leak into the experiment driver.

Stdlib only (urllib), no SDK — every byte on the wire is auditable, which matters
for a probe whose validity rests on exact request params. Add a vendor by writing
one `_adapter` function and registering it in _ADAPTERS.

Keys are read from the environment per vendor:
  Anthropic -> ANTHROPIC_API_KEY        (endpoint override: ANTHROPIC_BASE_URL)
  Google    -> GEMINI_API_KEY / GOOGLE_API_KEY  (override: GEMINI_BASE_URL)
"""

import os, json, time, urllib.request, urllib.error

ANTHROPIC_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL",
                                    "https://api.anthropic.com/v1/messages")
ANTHROPIC_VERSION = "2023-06-01"
GEMINI_BASE_URL = os.environ.get("GEMINI_BASE_URL",
                                 "https://generativelanguage.googleapis.com/v1beta")

# Status codes worth retrying with backoff (transient / overloaded / rate-limited).
_RETRY_CODES = (429, 500, 502, 503, 529)


class APIError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
#  Shared transport: POST JSON with bounded exponential backoff on _RETRY_CODES.
# ---------------------------------------------------------------------------
def _post_json(url, headers, payload, retries):
    data = json.dumps(payload).encode()
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            last = e
            if e.code in _RETRY_CODES:
                time.sleep(min(2 ** attempt + 0.5, 30))
                continue
            raise APIError(f"HTTP {e.code}: {e.read().decode()[:300]}")
        except urllib.error.URLError as e:
            last = e
            time.sleep(min(2 ** attempt + 0.5, 30))
    raise APIError(f"exhausted retries: {last}")


# ---------------------------------------------------------------------------
#  Adapters. Each returns (text, meta). `meta` records exactly how the call was
#  driven so two runs can be proven identical except for the model.
# ---------------------------------------------------------------------------
def _anthropic(model, user, system, temperature, max_tokens, retries):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise APIError("Set ANTHROPIC_API_KEY in the environment.")

    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": user}],
    }
    # Sampling params are REMOVED on Opus 4.8/4.7 and Fable 5 — sending any of
    # them returns a 400. Include temperature only when explicitly requested
    # (i.e. an older model that still accepts it). These models stay stochastic
    # across independent calls, so per-cell sampling diversity is preserved.
    if temperature is not None:
        payload["temperature"] = temperature
    if system:
        payload["system"] = system

    headers = {
        "content-type": "application/json",
        "x-api-key": key,
        "anthropic-version": ANTHROPIC_VERSION,
    }
    body = _post_json(ANTHROPIC_BASE_URL, headers, payload, retries)
    text = "".join(b.get("text", "") for b in body.get("content", [])
                   if b.get("type") == "text")
    meta = dict(provider="anthropic", model=model, endpoint=ANTHROPIC_BASE_URL,
                anthropic_version=ANTHROPIC_VERSION, max_tokens=max_tokens,
                temperature=temperature, temperature_sent=temperature is not None,
                has_system=bool(system))
    return text, meta


def _gemini(model, user, system, temperature, max_tokens, retries):
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise APIError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment.")

    gen_cfg = {"maxOutputTokens": max_tokens}
    if temperature is not None:          # Gemini accepts temperature; omit -> its default
        gen_cfg["temperature"] = temperature
    payload = {
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": gen_cfg,
    }
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}

    # Key goes in a header, not the query string, so it never lands in URLs/logs.
    url = f"{GEMINI_BASE_URL}/models/{model}:generateContent"
    headers = {"content-type": "application/json", "x-goog-api-key": key}
    body = _post_json(url, headers, payload, retries)

    cands = body.get("candidates", [])
    text = ""
    if cands:
        parts = cands[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts if "text" in p)
    meta = dict(provider="google", model=model, endpoint=url,
                max_tokens=max_tokens, temperature=temperature,
                temperature_sent=temperature is not None, has_system=bool(system),
                finish_reason=(cands[0].get("finishReason") if cands else None))
    return text, meta


_ADAPTERS = {"anthropic": _anthropic, "google": _gemini}


def provider_for(model: str) -> str:
    """Infer the vendor from a model id. Extend the prefixes as vendors are added."""
    m = model.lower()
    if m.startswith("claude") or m.startswith("anthropic"):
        return "anthropic"
    if m.startswith("gemini") or m.startswith("models/gemini"):
        return "google"
    raise APIError(f"no adapter for model {model!r}; register one in api_client._ADAPTERS")


# ---------------------------------------------------------------------------
#  Public interface — stable for all callers.
# ---------------------------------------------------------------------------
def call_model_meta(model: str, user: str, system: str = "", temperature: float = None,
                    max_tokens: int = 4000, retries: int = 4):
    """Return (text, provenance_dict). Use when you want to record how the call ran."""
    return _ADAPTERS[provider_for(model)](model, user, system, temperature,
                                           max_tokens, retries)


def call_model(model: str, user: str, system: str = "", temperature: float = None,
               max_tokens: int = 4000, retries: int = 4) -> str:
    """Return the completion text only. Unchanged contract for existing callers."""
    text, _meta = call_model_meta(model, user, system, temperature, max_tokens, retries)
    return text
