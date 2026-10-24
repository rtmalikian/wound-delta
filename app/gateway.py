"""Embedded OpenAI-compatible inference gateway.

This module is the architectural core of WoundDelta, and it is the shared
platform skeleton carried over from Week 1's Discharge Decoded and Week 2's
Prior-Auth Co-Pilot: instead of the product calling an outside AI API
directly, every AI feature goes through this gateway, which speaks the
OpenAI API shapes (``/v1/models``, ``/v1/chat/completions``,
``/v1/embeddings``) and routes to a pluggable backend selected by the
``AI_BACKEND`` environment variable:

- ``stub`` (default): deterministic, rule-based responses. Zero model needed.
  Used by tests, CI, and the demo when no model is configured.
- ``local``: forwards to an OpenAI-compatible server on the clinic's own
  hardware (Ollama, llama.cpp server, vLLM, ...) via ``LOCAL_MODEL_URL``.
  Photos never leave the building: the self-hosting story.
- ``upstream``: forwards to any OpenAI-compatible provider via
  ``UPSTREAM_BASE_URL`` + ``UPSTREAM_API_KEY``.

Two product tiers ride on the same gateway (see ``TIERS`` below). The tier
is a routing profile, selected by the ``model`` name (``wound-delta-frugal``
vs ``wound-delta-full``) or an ``X-Tier`` header. It is one product with the
escalation dial in one config dict, never two products.

The gateway also keeps an honest token ledger. Vision tokens are estimated
with the published tile maths for tiled vision models (a ``detail:low``
image is a flat 85 tokens; ``detail:high`` is 85 + 170 per 512px tile), so
the frugal budget (<=300 tokens per wound pair) is a number the eval harness
can assert, not a vibe.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import uuid

import httpx
from fastapi import HTTPException

STUB_MODEL_ID = "wound-delta-stub-1"
FRUGAL_MODEL_ID = "wound-delta-frugal"
FULL_MODEL_ID = "wound-delta-full"
_REQUEST_TIMEOUT = 120.0

# ---------------------------------------------------------------------------
# Tiers -- one gateway, one config dict, two budgets
# ---------------------------------------------------------------------------

TIERS: dict[str, dict] = {
    "frugal": {
        "model_alias": FRUGAL_MODEL_ID,
        "vision_detail": "low",
        "roi_only": True,
        "low_detail_tokens": 85,
        "token_budget_per_pair": 300,
        "verifier": False,
        "memory": "last_pair",
        "summary_max_tokens": 160,
    },
    "full": {
        "model_alias": FULL_MODEL_ID,
        "vision_detail": "high",
        "roi_only": False,
        "low_detail_tokens": 85,
        "token_budget_per_pair": 4000,
        "verifier": True,
        "memory": "full_history",
        "summary_max_tokens": 500,
    },
}


def resolve_tier(model: str | None = None, x_tier: str | None = None) -> str:
    """Resolve the tier from an X-Tier header or a model alias."""
    if x_tier and x_tier.strip().lower() in TIERS:
        return x_tier.strip().lower()
    if model:
        m = model.strip().lower()
        if m.endswith("-full") or m == FULL_MODEL_ID:
            return "full"
        if m.endswith("-frugal") or m == FRUGAL_MODEL_ID:
            return "frugal"
    env = (os.environ.get("WOUND_DELTA_TIER", "frugal") or "frugal").strip().lower()
    return env if env in TIERS else "frugal"


def tier_profile(tier: str) -> dict:
    return dict(TIERS[tier])


def estimate_image_tokens(detail: str, width: int, height: int) -> int:
    """Published tile maths: low detail is flat; high detail tiles at 512px."""
    if detail == "low":
        return 85
    tiles = max(1, math.ceil(width / 512)) * max(1, math.ceil(height / 512))
    return 85 + 170 * tiles


def estimate_text_tokens(text: str) -> int:
    return max(1, len(text or "") // 4)


# ---------------------------------------------------------------------------
# Call ledger -- tests assert on this (e.g. unchanged pair == zero calls)
# ---------------------------------------------------------------------------

CALL_LEDGER: dict[str, int] = {"chat": 0, "embeddings": 0}
CALL_LOG: list[dict] = []


def reset_ledger() -> None:
    CALL_LEDGER["chat"] = 0
    CALL_LEDGER["embeddings"] = 0
    CALL_LOG.clear()


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

def get_backend() -> str:
    """Return the configured backend name: stub | local | upstream."""
    return (os.environ.get("AI_BACKEND", "stub") or "stub").strip().lower()


def _local_base() -> str:
    return (os.environ.get("LOCAL_MODEL_URL", "http://localhost:11434") or "").rstrip("/")


def _upstream_base() -> str:
    return (os.environ.get("UPSTREAM_BASE_URL", "https://api.openai.com") or "").rstrip("/")


def _upstream_headers() -> dict:
    key = os.environ.get("UPSTREAM_API_KEY", "")
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


# ---------------------------------------------------------------------------
# Public dispatch API -- the single code path for all AI calls
# ---------------------------------------------------------------------------

def list_models() -> dict:
    backend = get_backend()
    if backend == "stub":
        data = [
            {"id": STUB_MODEL_ID, "object": "model",
             "created": int(time.time()), "owned_by": "wound-delta"},
            {"id": FRUGAL_MODEL_ID, "object": "model",
             "created": int(time.time()), "owned_by": "wound-delta",
             "tier": "frugal"},
            {"id": FULL_MODEL_ID, "object": "model",
             "created": int(time.time()), "owned_by": "wound-delta",
             "tier": "full"},
        ]
        return {"object": "list", "data": data}
    if backend == "local":
        try:
            r = httpx.get(f"{_local_base()}/v1/models", timeout=10.0)
            r.raise_for_status()
            return r.json()
        except Exception:
            return {"object": "list", "data": [
                {"id": "local-model", "object": "model",
                 "created": int(time.time()), "owned_by": "local"}]}
    if backend == "upstream":
        try:
            r = httpx.get(f"{_upstream_base()}/v1/models",
                          headers=_upstream_headers(), timeout=15.0)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            raise HTTPException(status_code=502,
                                detail=f"upstream /v1/models failed: {exc}")
    raise HTTPException(status_code=500, detail=f"unknown AI_BACKEND: {backend}")


def dispatch_chat(messages: list[dict], model: str | None = None,
                  temperature: float = 0.2, max_tokens: int = 2000,
                  response_format: dict | None = None,
                  tier: str | None = None) -> dict:
    """OpenAI-shaped chat completion through the configured backend.

    ``messages`` may carry multimodal content parts (text + image_url).
    The usage block reports *estimated* vision tokens using the tile maths
    above, so per-pair budgets are assertable in tests on every backend.
    """
    tier = tier or resolve_tier(model)
    profile = tier_profile(tier)
    backend = get_backend()
    CALL_LEDGER["chat"] += 1
    CALL_LOG.append({"kind": "chat", "tier": tier, "backend": backend,
                     "model": model})
    if backend == "stub":
        content = _stub_chat(messages)
        model_id = model or profile["model_alias"]
    elif backend in ("local", "upstream"):
        content, model_id = _forward_chat(
            messages, model, temperature, max_tokens, response_format, backend)
    else:
        raise HTTPException(status_code=500, detail=f"unknown AI_BACKEND: {backend}")

    prompt_tokens = _count_prompt_tokens(messages, profile)
    completion_tokens = estimate_text_tokens(content)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "tier": tier,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _count_prompt_tokens(messages: list[dict], profile: dict) -> int:
    total = 0
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            total += estimate_text_tokens(content)
        elif isinstance(content, list):
            for part in content:
                if part.get("type") == "text":
                    total += estimate_text_tokens(part.get("text", ""))
                elif part.get("type") == "image_url":
                    detail = (part.get("image_url", {}) or {}).get(
                        "detail", profile["vision_detail"])
                    # Synthetic parts may carry explicit pixel dims for the
                    # ledger; real data URLs are counted at the tier detail
                    # rate (low = flat 85, high = one 1024px frame = 765).
                    w = part.get("_width", 1024)
                    h = part.get("_height", 1024)
                    total += estimate_image_tokens(detail, w, h)
    return total


def dispatch_embeddings(input: str | list[str], model: str | None = None) -> dict:
    """OpenAI-shaped embeddings through the configured backend."""
    backend = get_backend()
    CALL_LEDGER["embeddings"] += 1
    texts = [input] if isinstance(input, str) else list(input)
    if backend == "stub":
        vectors = [_stub_embedding(t) for t in texts]
        model_id = model or STUB_MODEL_ID
    elif backend in ("local", "upstream"):
        vectors, model_id = _forward_embeddings(texts, model, backend)
    else:
        raise HTTPException(status_code=500, detail=f"unknown AI_BACKEND: {backend}")

    return {
        "object": "list",
        "data": [
            {"object": "embedding", "embedding": vec, "index": i}
            for i, vec in enumerate(vectors)
        ],
        "model": model_id,
        "usage": {
            "prompt_tokens": max(1, sum(len(t) for t in texts) // 4),
            "total_tokens": max(1, sum(len(t) for t in texts) // 4),
        },
    }


# ---------------------------------------------------------------------------
# Forwarding backends
# ---------------------------------------------------------------------------

def _clean_messages(messages: list[dict]) -> list[dict]:
    """Strip ledger-only keys (``_width``/``_height``) before forwarding."""
    out = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            content = [{k: v for k, v in part.items()
                        if not k.startswith("_")} for part in content]
        out.append({**m, "content": content})
    return out


def _forward_chat(messages, model, temperature, max_tokens, response_format, backend):
    if backend == "local":
        url = f"{_local_base()}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        model_id = model or os.environ.get("LOCAL_MODEL", "local-model")
    else:
        url = f"{_upstream_base()}/v1/chat/completions"
        headers = _upstream_headers()
        model_id = model or os.environ.get("UPSTREAM_MODEL", "gpt-4o-mini")
    payload = {"model": model_id, "messages": _clean_messages(messages),
               "temperature": temperature, "max_tokens": max_tokens}
    if response_format:
        payload["response_format"] = response_format
    try:
        r = httpx.post(url, json=payload, headers=headers, timeout=_REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        raise HTTPException(status_code=502,
                            detail=f"{backend} chat completions failed: {exc}")
    try:
        content = data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        raise HTTPException(status_code=502,
                            detail=f"{backend} returned an unexpected chat shape")
    return content, data.get("model", model_id)


def _forward_embeddings(texts, model, backend):
    if backend == "local":
        url = f"{_local_base()}/v1/embeddings"
        headers = {"Content-Type": "application/json"}
        model_id = model or os.environ.get("LOCAL_MODEL", "local-model")
    else:
        url = f"{_upstream_base()}/v1/embeddings"
        headers = _upstream_headers()
        model_id = model or os.environ.get("UPSTREAM_MODEL", "text-embedding-3-small")
    try:
        r = httpx.post(url, json={"model": model_id, "input": texts},
                       headers=headers, timeout=_REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502,
                            detail=f"{backend} embeddings failed: {exc}")
    try:
        vectors = [d["embedding"] for d in sorted(data["data"], key=lambda d: d["index"])]
    except (KeyError, TypeError):
        raise HTTPException(status_code=502,
                            detail=f"{backend} returned an unexpected embeddings shape")
    return vectors, data.get("model", model_id)


# ---------------------------------------------------------------------------
# Stub backend -- deterministic, zero-model, used by tests/CI/demo
#
# The stub enforces the product's core contract in miniature: the model only
# ever *phrases* measurements that deterministic code produced. Given a
# [MEASUREMENTS] JSON block it writes a plain summary using exactly those
# numbers. It cannot invent a number, because it has no other source.
# ---------------------------------------------------------------------------

def _stub_embedding(text: str, dims: int = 16) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [round(b / 255.0, 6) for b in digest[:dims]]


def _find_block(messages: list[dict], tag: str) -> str:
    blob = ""
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            blob += content + "\n"
        elif isinstance(content, list):
            for part in content:
                if part.get("type") == "text":
                    blob += part.get("text", "") + "\n"
    mm = re.search(rf"\[{tag}\]\s*(.*?)\s*\[/{tag}\]", blob, re.S)
    return mm.group(1) if mm else ""


def _stub_chat(messages: list[dict]) -> str:
    task_m = re.search(r"\[TASK:\s*([A-Z_]+)\]",
                       _find_block(messages, "TASK") or
                       " ".join(str(m.get("content", "")) for m in messages))
    task = task_m.group(1) if task_m else None
    if task == "SUMMARISE_CHANGE":
        return _stub_summarise(_find_block(messages, "MEASUREMENTS"))
    if task == "VERIFY_SUMMARY":
        return _stub_verify(_find_block(messages, "MEASUREMENTS"),
                            _find_block(messages, "SUMMARY"))
    blob = _find_block(messages, "MEASUREMENTS")
    if blob:
        return _stub_summarise(blob)
    return "Stub response."


def _stub_summarise(measurements_blob: str) -> str:
    try:
        m = json.loads(measurements_blob)
    except Exception:
        return "No measurements were provided, so no summary can be written."
    pct = m.get("area_change_pct")
    direction = "smaller" if (pct is not None and pct < 0) else (
        "larger" if (pct is not None and pct > 0) else "about the same size")
    days = m.get("days_between")
    when = f"over {days} days" if days else "since the last photo"
    lines = [
        f"Compared with the previous photo, the wound area measured "
        f"{abs(pct):.1f}% {direction} {when}." if pct is not None else
        "The wound area could not be measured on this pair.",
    ]
    flags = m.get("red_flags") or []
    if flags:
        names = ", ".join(f.get("name", "change") for f in flags)
        lines.append(
            f"Patterns to raise with your care team: {names}. "
            "This app does not diagnose. If any of these match how the wound "
            "feels or looks to you, contact your care team.")
    else:
        lines.append("No red-flag patterns were detected by the change "
                     "checks on this pair.")
    lines.append("Measurements come from photo alignment and pixel "
                 "difference done on this device/server, not from the "
                 "language model.")
    return " ".join(lines)


def _stub_verify(measurements_blob: str, summary: str) -> str:
    """Second-pass check (full tier): every number in the summary must
    appear in the measurements block. Returns JSON {ok, stray_numbers}."""
    try:
        m = json.loads(measurements_blob)
    except Exception:
        m = {}
    allowed = set()
    def collect(v):
        if isinstance(v, (int, float)):
            # Summaries phrase magnitudes ("30.0% smaller" for -30.0),
            # so the check compares absolute values.
            allowed.add(abs(round(float(v), 1)))
        elif isinstance(v, dict):
            for x in v.values():
                collect(x)
        elif isinstance(v, list):
            for x in v:
                collect(x)
    collect(m)
    nums = [float(x) for x in re.findall(r"\d+\.\d+|\d+", summary or "")]
    stray = [n for n in nums if abs(round(n, 1)) not in allowed]
    return json.dumps({"ok": not stray, "stray_numbers": stray})
