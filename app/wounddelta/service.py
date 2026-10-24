"""The compare pipeline -- the product in one function.

Order matters, and the order *is* the token story:

1. Hash check. Identical photo -> templated answer, **zero model calls**.
2. Align + segment + diff in deterministic code (``imaging.analyse_pair``).
   Every number in the result is produced here, never by a model.
3. Frugal tier: if the measured change is negligible, a templated answer
   again costs zero calls. Otherwise one gateway chat call phrases the
   measurements, carrying only the changed-region crop at ``detail:low``
   (85 vision tokens). Budget: <=300 tokens per pair, asserted in CI.
4. Full tier: the same measurements, but the gateway call carries the full
   frame at ``detail:high`` and a second verifier call checks that every
   number in the summary actually appears in the measurements (the
   verified-generation pattern from Weeks 1-2, applied to phrasing).
"""

from __future__ import annotations

import base64
import json
from datetime import date

from app import gateway
from app.wounddelta import imaging

NEGLIGIBLE_CHANGE_PCT = 1.0


def _days_between(a: str | None, b: str | None) -> float | None:
    if not a or not b:
        return None
    try:
        return float((date.fromisoformat(b) - date.fromisoformat(a)).days)
    except ValueError:
        return None


def _image_part(png_bytes: bytes, detail: str, width: int, height: int) -> dict:
    return {
        "type": "image_url",
        "image_url": {
            "url": "data:image/png;base64,"
                   + base64.b64encode(png_bytes).decode(),
            "detail": detail,
        },
        "_width": width,
        "_height": height,
    }


def _templated_message(m: dict, note: str) -> str:
    return note


def compare_pair(prev_bytes: bytes, new_bytes: bytes, tier: str = "frugal",
                 prev_date: str | None = None,
                 new_date: str | None = None) -> dict:
    """Compare two wound photos. Returns measurements + phrasing + ledger."""
    profile = gateway.tier_profile(tier)
    prev = imaging.load_image(prev_bytes)
    new = imaging.load_image(new_bytes)

    # 1. Hash check first -- an unchanged photo costs nothing.
    if imaging.image_sha(prev) == imaging.image_sha(new):
        return {
            "changed": False,
            "unchanged_reason": "identical_photo",
            "measurements": None,
            "summary": _templated_message(
                {}, "This photo is identical to the previous one. "
                    "No model call was made."),
            "tier": tier,
            "usage": {"model_calls": 0, "est_tokens": 0},
        }

    # 2. Deterministic measurement.
    days = _days_between(prev_date, new_date)
    analysis = imaging.analyse_pair(prev, new, days_between=days)
    m = imaging.public_measurements(analysis)

    # 3/4. Phrasing through the gateway -- numbers only from step 2.
    aligned = analysis["_aligned"]
    if tier == "frugal":
        crop_arr = imaging.roi_crop(aligned, analysis["roi"])
        img_part = _image_part(imaging.encode_png(crop_arr), "low",
                               crop_arr.shape[1], crop_arr.shape[0])
        negligible = (abs(m["area_change_pct"]) <= NEGLIGIBLE_CHANGE_PCT
                      and not m["red_flags"])
    else:
        img_part = _image_part(imaging.encode_png(aligned), "high",
                               aligned.shape[1], aligned.shape[0])
        negligible = False

    if negligible:
        summary = ("The wound area measured about the same as the previous "
                   "photo, and no red-flag patterns were detected. "
                   "No model call was made.")
        return {"changed": True, "measurements": m, "summary": summary,
                "tier": tier, "usage": {"model_calls": 0, "est_tokens": 0}}

    messages = [
        {"role": "system",
         "content": "You phrase wound-change measurements for a patient. "
                    "Use only the numbers given. Never diagnose, never "
                    "recommend treatment."},
        {"role": "user", "content": [
            {"type": "text",
             "text": "[TASK: SUMMARISE_CHANGE]\n[MEASUREMENTS]\n"
                     + json.dumps(m) + "\n[/MEASUREMENTS]"},
            img_part,
        ]},
    ]
    before = gateway.CALL_LEDGER["chat"]
    resp = gateway.dispatch_chat(messages, model=profile["model_alias"],
                                 tier=tier,
                                 max_tokens=profile["summary_max_tokens"])
    summary = resp["choices"][0]["message"]["content"]
    est_tokens = resp["usage"]["total_tokens"]
    verifier = None
    if profile["verifier"]:
        v_messages = [
            {"role": "user", "content":
                "[TASK: VERIFY_SUMMARY]\n[MEASUREMENTS]\n"
                + json.dumps(m) + "\n[/MEASUREMENTS]\n[SUMMARY]\n"
                + summary + "\n[/SUMMARY]"},
        ]
        v_resp = gateway.dispatch_chat(v_messages, tier=tier,
                                       model=profile["model_alias"])
        est_tokens += v_resp["usage"]["total_tokens"]
        try:
            verifier = json.loads(
                v_resp["choices"][0]["message"]["content"])
        except Exception:
            verifier = {"ok": None, "note": "verifier returned non-JSON"}

    return {
        "changed": True,
        "measurements": m,
        "summary": summary,
        "verifier": verifier,
        "tier": tier,
        "usage": {
            "model_calls": gateway.CALL_LEDGER["chat"] - before,
            "est_tokens": est_tokens,
            "budget": profile["token_budget_per_pair"],
        },
    }
