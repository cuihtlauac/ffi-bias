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

import os, json, time, threading, urllib.request, urllib.error
from functools import partial


def _load_dotenv(path=None):
    """Populate os.environ from a local, gitignored `.env` (KEY=VALUE lines) so keys
    live in one file instead of the shell profile — no per-run ceremony, and nothing
    leaks into shell history or `ps`. Existing env vars WIN (setdefault), so an explicit
    export or an inline `KEY=val python ...` still overrides. Stdlib-only, best-effort:
    a missing/unreadable file is a silent no-op. The file must stay gitignored."""
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    try:
        with open(path) as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key:
                    os.environ.setdefault(key, val)
    except OSError:
        pass


_load_dotenv()   # before any os.environ.get below, so .env feeds the constants too

ANTHROPIC_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL",
                                    "https://api.anthropic.com/v1/messages")
ANTHROPIC_VERSION = "2023-06-01"
GEMINI_BASE_URL = os.environ.get("GEMINI_BASE_URL",
                                 "https://generativelanguage.googleapis.com/v1beta")

# Status codes worth retrying with backoff (transient / overloaded / rate-limited).
_RETRY_CODES = (429, 500, 502, 503, 529)

# Optional global request pacing, vendor-agnostic. Free-tier endpoints (e.g. the
# Gemini free tier ~10-15 RPM) throttle hard; firing faster just burns the daily
# quota on 429 retries. Set API_MIN_INTERVAL_S to enforce a minimum gap between
# *every* outbound request (runner AND judge, since both funnel through _post_json).
# Default 0 = no pacing, so paid/high-limit models are unaffected.
_MIN_INTERVAL = float(os.environ.get("API_MIN_INTERVAL_S", "0") or 0)
_rate_lock = threading.Lock()
_last_request = [0.0]

# Set once a per-day free-tier quota is hit, so the remaining jobs in this run
# short-circuit instantly instead of each waiting out the throttle before failing.
# Process-scoped (a fresh run clears it); assumes one model per process, as we run.
_daily_exhausted = [False]


def _throttle():
    if _MIN_INTERVAL <= 0:
        return
    with _rate_lock:
        wait = _MIN_INTERVAL - (time.monotonic() - _last_request[0])
        if wait > 0:
            time.sleep(wait)
        _last_request[0] = time.monotonic()


class APIError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
#  Shared transport: POST JSON with bounded exponential backoff on _RETRY_CODES.
# ---------------------------------------------------------------------------
def _post_json(url, headers, payload, retries):
    data = json.dumps(payload).encode()
    # Default User-Agent: some providers sit behind Cloudflare, which blocks the
    # stock "Python-urllib/x" signature (HTTP 403, CF error 1010). A plain custom
    # UA passes; caller-supplied headers still win.
    headers = {"user-agent": "ffi-bias-probe/1.0", **headers}
    last = None
    for attempt in range(retries):
        if _daily_exhausted[0]:   # a prior call already hit the per-day wall — don't even wait
            raise APIError("daily free-tier quota exhausted (short-circuit)")
        _throttle()   # honor the global min-interval before every outbound request
        try:
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            last = e
            if e.code in _RETRY_CODES:
                body = ""
                if e.code == 429:
                    try:
                        body = e.read().decode()
                    except Exception:
                        body = ""
                # A per-DAY free-tier exhaustion won't clear by retrying — fail fast so
                # the run ends quickly (and resumes after reset) instead of grinding the
                # full retry budget on every remaining job.
                if e.code == 429 and "PerDay" in body:
                    _daily_exhausted[0] = True   # short-circuit the rest of this run
                    raise APIError(f"daily free-tier quota exhausted: {body[:200]}")
                time.sleep(min(2 ** attempt + 0.5, 60))   # cap raised: clears a full RPM window
                continue
            raise APIError(f"HTTP {e.code}: {e.read().decode()[:300]}")
        except urllib.error.URLError as e:
            last = e
            time.sleep(min(2 ** attempt + 0.5, 60))
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
    u = body.get("usage") or {}
    meta = dict(provider="anthropic", model=model,
                model_served=body.get("model"),   # concrete version behind the alias
                endpoint=ANTHROPIC_BASE_URL, anthropic_version=ANTHROPIC_VERSION,
                max_tokens=max_tokens, temperature=temperature,
                temperature_sent=temperature is not None, has_system=bool(system),
                input_tokens=u.get("input_tokens"), output_tokens=u.get("output_tokens"))
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
    u = body.get("usageMetadata") or {}
    meta = dict(provider="google", model=model,
                model_served=body.get("modelVersion"),   # concrete version Google served
                endpoint=url, max_tokens=max_tokens, temperature=temperature,
                temperature_sent=temperature is not None, has_system=bool(system),
                finish_reason=(cands[0].get("finishReason") if cands else None),
                input_tokens=u.get("promptTokenCount"),
                output_tokens=u.get("candidatesTokenCount"))
    return text, meta


def _openai_compat(model, user, system, temperature, max_tokens, retries, *,
                   base_url, key_env, provider):
    """One adapter for every OpenAI /chat/completions-shaped host (Together, the
    Hugging Face Inference router, Fireworks, Groq, OpenRouter, local vLLM/Ollama).
    The model id carries a `provider:` prefix for routing; we strip it before sending.
    This is what unlocks the open-source ecosystem with no per-model code."""
    key = os.environ.get(key_env)
    if not key:
        raise APIError(f"Set {key_env} in the environment.")
    sent = model.split(":", 1)[1] if model.startswith(provider + ":") else model
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": user}]
    payload = {"model": sent, "max_tokens": max_tokens, "messages": msgs}
    if temperature is not None:          # open models accept temperature (unlike Opus 4.8)
        payload["temperature"] = temperature
    headers = {"content-type": "application/json", "authorization": f"Bearer {key}"}
    body = _post_json(base_url, headers, payload, retries)
    choice = (body.get("choices") or [{}])[0]
    text = (choice.get("message") or {}).get("content", "") or ""
    u = body.get("usage") or {}
    meta = dict(provider=provider, model=sent, model_served=body.get("model"),
                endpoint=base_url, max_tokens=max_tokens, temperature=temperature,
                temperature_sent=temperature is not None, has_system=bool(system),
                finish_reason=choice.get("finish_reason"),
                input_tokens=u.get("prompt_tokens"), output_tokens=u.get("completion_tokens"))
    return text, meta


_ADAPTERS = {
    "anthropic": _anthropic,
    "google": _gemini,
    "together": partial(_openai_compat,
                        base_url=os.environ.get("TOGETHER_BASE_URL",
                                                "https://api.together.xyz/v1/chat/completions"),
                        key_env="TOGETHER_API_KEY", provider="together"),
    "hf": partial(_openai_compat,
                  base_url=os.environ.get("HF_BASE_URL",
                                          "https://router.huggingface.co/v1/chat/completions"),
                  key_env="HF_TOKEN", provider="hf"),
    "cerebras": partial(_openai_compat,
                        base_url=os.environ.get("CEREBRAS_BASE_URL",
                                                "https://api.cerebras.ai/v1/chat/completions"),
                        key_env="CEREBRAS_API_KEY", provider="cerebras"),
}


def provider_for(model: str) -> str:
    """Infer the vendor from a model id. Extend the prefixes as vendors are added.
    OpenAI-compatible hosts use an explicit `provider:` prefix (e.g.
    `together:Qwen/Qwen2.5-Coder-32B-Instruct-Turbo`) since open model ids collide."""
    m = model.lower()
    if m.startswith("claude") or m.startswith("anthropic"):
        return "anthropic"
    if m.startswith("gemini") or m.startswith("models/gemini"):
        return "google"
    if m.startswith("together:"):
        return "together"
    if m.startswith("hf:"):
        return "hf"
    if m.startswith("cerebras:"):
        return "cerebras"
    raise APIError(f"no adapter for model {model!r}; register one in api_client._ADAPTERS")


# ---------------------------------------------------------------------------
#  Public interface — stable for all callers.
# ---------------------------------------------------------------------------
def call_model_meta(model: str, user: str, system: str = "", temperature: float = None,
                    max_tokens: int = 4000, retries: int = 6):
    """Return (text, provenance_dict). Use when you want to record how the call ran."""
    return _ADAPTERS[provider_for(model)](model, user, system, temperature,
                                           max_tokens, retries)


def call_model(model: str, user: str, system: str = "", temperature: float = None,
               max_tokens: int = 4000, retries: int = 6) -> str:
    """Return the completion text only. Unchanged contract for existing callers."""
    text, _meta = call_model_meta(model, user, system, temperature, max_tokens, retries)
    return text
