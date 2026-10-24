"""WoundDelta -- FastAPI application.

Routes:
  /v1/*    Embedded OpenAI-compatible inference gateway (the product's AI
           layer -- every AI feature calls through here). Tier is selected
           by the ``model`` alias (wound-delta-frugal | wound-delta-full)
           or an ``X-Tier: frugal|full`` header.
  /api/*   Product API: wound cases, photo captures, pair comparison,
           trend, ghost overlay, synthetic demo.
  /        Static single-page demo UI.
  /healthz Health check.

Storage is in-memory in v0.1.0. Case labels pass the PHI redaction gate
before storage. Photos are health data: run the local backend and they
never leave the building (see SECURITY.md).
"""

from __future__ import annotations

import logging
import os
from datetime import date

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import gateway, safety
from app.wounddelta import imaging, models, service, synth

logger = logging.getLogger("wound-delta")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")

MAX_UPLOAD_BYTES = 8 * 1024 * 1024

app = FastAPI(
    title="WoundDelta",
    description="Open-source longitudinal wound-change measurement. "
                "Photos in, change measured in code, an embedded "
                "OpenAI-compatible gateway phrases the result.",
    version="0.1.0",
)

CASES: dict[str, models.WoundCase] = {}


def _get_case(case_id: str) -> models.WoundCase:
    case = CASES.get(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="case not found")
    return case


# ---------------------------------------------------------------------------
# Gateway routes -- OpenAI-compatible shapes, served by this app
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[dict] = Field(default_factory=list)
    temperature: float = 0.2
    max_tokens: int = 2000
    response_format: dict | None = None


class EmbeddingsRequest(BaseModel):
    model: str | None = None
    input: str | list[str] = ""


@app.get("/v1/models")
def v1_models():
    return gateway.list_models()


@app.post("/v1/chat/completions")
def v1_chat_completions(req: ChatRequest,
                        x_tier: str | None = Header(default=None)):
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")
    tier = gateway.resolve_tier(req.model, x_tier)
    return gateway.dispatch_chat(
        req.messages, model=req.model, temperature=req.temperature,
        max_tokens=req.max_tokens, response_format=req.response_format,
        tier=tier)


@app.post("/v1/embeddings")
def v1_embeddings(req: EmbeddingsRequest):
    return gateway.dispatch_embeddings(req.input, model=req.model)


# ---------------------------------------------------------------------------
# Product API
# ---------------------------------------------------------------------------

class CaseCreateRequest(BaseModel):
    label: str = Field(min_length=1, max_length=120)


@app.get("/healthz")
def healthz():
    return {"status": "ok", "backend": gateway.get_backend(),
            "tier_default": gateway.resolve_tier(),
            "cases": len(CASES)}


@app.post("/api/cases")
def api_create_case(req: CaseCreateRequest):
    label, _report = safety.redact_phi(req.label)
    case = models.new_case(label=label)
    CASES[case.id] = case
    return case.to_dict()


@app.get("/api/cases/{case_id}")
def api_get_case(case_id: str):
    return _get_case(case_id).to_dict()


@app.post("/api/cases/{case_id}/captures")
async def api_add_capture(case_id: str, file: UploadFile = File(...),
                          taken_on: str = Form(default="")):
    case = _get_case(case_id)
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="file too large (max 8 MB)")
    try:
        arr = imaging.load_image(data)
    except Exception:
        raise HTTPException(status_code=400,
                            detail="not a readable PNG/JPEG image")
    cap = case.add_capture(data, taken_on or date.today().isoformat(),
                           imaging.image_sha(arr))
    # Measure this capture's area against itself-free segmentation so the
    # trend has an area series even before a comparison is requested.
    try:
        cap.area_px = int(imaging.segment_wound(arr).sum()) or None
    except Exception:
        cap.area_px = None
    return {"case_id": case.id, "capture": cap.to_dict(),
            "captures": len(case.captures)}


@app.post("/api/cases/{case_id}/compare")
def api_compare(case_id: str, x_tier: str | None = Header(default=None)):
    """Compare the two most recent captures of a case."""
    case = _get_case(case_id)
    if len(case.captures) < 2:
        raise HTTPException(status_code=400,
                            detail="add two photo captures first")
    prev_cap, new_cap = case.captures[-2], case.captures[-1]
    tier = gateway.resolve_tier(None, x_tier)
    try:
        result = service.compare_pair(
            case.image_for(prev_cap.id), case.image_for(new_cap.id),
            tier=tier, prev_date=prev_cap.taken_on,
            new_date=new_cap.taken_on)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    record = {"prev_capture": prev_cap.id, "new_capture": new_cap.id,
              "tier": tier, **result}
    case.comparisons.append(
        {k: v for k, v in record.items()})
    logger.info("pair compared", extra={
        "case": case.id, "tier": tier,
        "model_calls": result["usage"]["model_calls"]})
    return record


@app.get("/api/cases/{case_id}/overlay", response_class=HTMLResponse)
def api_overlay_info(case_id: str):
    case = _get_case(case_id)
    if not case.captures:
        raise HTTPException(status_code=400, detail="no captures yet")
    return ("<p>The ghost overlay is the previous photo at 50% opacity over "
            "the live preview in the demo UI. Line the wound up with its "
            "ghost before taking the next photo: same angle, same distance, "
            "same light. That one habit is what makes the pixel difference "
            "a measurement instead of a guess.</p>")


@app.get("/api/cases/{case_id}/captures/{capture_id}/ghost")
def api_ghost(case_id: str, capture_id: str):
    """Previous capture as a standalone ghost image (for capture guides)."""
    case = _get_case(case_id)
    data = case.image_for(capture_id)
    if data is None:
        raise HTTPException(status_code=404, detail="capture not found")
    return Response(content=data, media_type="image/png")


@app.get("/api/cases/{case_id}/trend")
def api_trend(case_id: str):
    return _get_case(case_id).trend()


# ---------------------------------------------------------------------------
# Synthetic demo -- the same fixtures CI measures
# ---------------------------------------------------------------------------

@app.get("/api/demo/pair")
def api_demo_pair(kind: str = "healing"):
    kinds = {
        "healing": dict(area_change_pct=-30.0, seed=31),
        "stable": dict(area_change_pct=0.0, days=3, seed=32),
        "worse": dict(area_change_pct=38.0, redness_halo=True, seed=33),
    }
    if kind not in kinds:
        raise HTTPException(status_code=400, detail="kind: healing|stable|worse")
    return synth.make_pair(**kinds[kind])["truth"]


@app.post("/api/demo/compare")
def api_demo_compare(kind: str = "healing",
                     x_tier: str | None = Header(default=None)):
    kinds = {
        "healing": dict(area_change_pct=-30.0, seed=31),
        "stable": dict(area_change_pct=0.0, days=3, seed=32),
        "worse": dict(area_change_pct=38.0, redness_halo=True, seed=33),
    }
    if kind not in kinds:
        raise HTTPException(status_code=400, detail="kind: healing|stable|worse")
    pair = synth.make_pair(**kinds[kind])
    tier = gateway.resolve_tier(None, x_tier)
    result = service.compare_pair(
        pair["prev_png"], pair["new_png"], tier=tier,
        prev_date="2026-10-16",
        new_date="2026-10-23")
    return {"truth": pair["truth"], "tier": tier, **result}


# ---------------------------------------------------------------------------
# Static demo UI
# ---------------------------------------------------------------------------

_STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.isdir(_STATIC_DIR):
    app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")
