"""Deterministic image mechanics: alignment, segmentation, diff, ROI.

Everything measurable in WoundDelta is computed here, in plain numpy, with
no model in the loop:

- ``align`` -- translation alignment by FFT phase correlation.
- ``segment_wound`` -- a redness-index heuristic mask. This is a measurement
  heuristic, not a trained segmentation model and not a diagnosis. It is
  tuned and *proven only on the synthetic fixtures in the eval harness*;
  real-photo accuracy is unproven (see SECURITY.md).
- ``analyse_pair`` -- aligns the new photo to the previous one, segments
  both, and produces the change mask, area-change %, redness spread,
  darkening, the change ROI crop, and red-flag pattern checks.

The language model never sees a full frame in the frugal tier and never
produces a number. It phrases numbers this module produced.
"""

from __future__ import annotations

import hashlib
import io

import numpy as np
from PIL import Image

# Red-flag pattern thresholds (deterministic, stated, adjustable in one place)
RAPID_EXPANSION_PCT = 25.0        # area growth at/above this = pattern
REDNESS_SPREAD_PCT = 15.0         # new red area outside wound, % of wound area
DARKENING_DELTA = 18.0            # mean wound luminance drop (0-255 scale)
STALL_DAYS = 14                   # no measurable change over >= this many days
STALL_MAX_CHANGE_PCT = 2.0


# ---------------------------------------------------------------------------
# Loading / hashing
# ---------------------------------------------------------------------------

def load_image(data: bytes) -> np.ndarray:
    """Load PNG/JPEG bytes to an HxWx3 uint8 RGB array."""
    with Image.open(io.BytesIO(data)) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def image_sha(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()


def encode_png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Alignment -- FFT phase correlation (translation only)
# ---------------------------------------------------------------------------

def _to_gray(arr: np.ndarray) -> np.ndarray:
    return arr.astype(np.float64).mean(axis=2)


def estimate_shift(ref: np.ndarray, img: np.ndarray) -> tuple[int, int]:
    """Return (dy, dx) such that shifting ``img`` by it aligns to ``ref``."""
    a = _to_gray(ref)
    b = _to_gray(img)
    fa = np.fft.rfft2(a)
    fb = np.fft.rfft2(b)
    cross = fa * np.conj(fb)
    cross /= (np.abs(cross) + 1e-9)
    corr = np.fft.irfft2(cross, s=a.shape)
    dy, dx = np.unravel_index(np.argmax(corr), corr.shape)
    if dy > a.shape[0] // 2:
        dy -= a.shape[0]
    if dx > a.shape[1] // 2:
        dx -= a.shape[1]
    return int(dy), int(dx)


def shift_image(img: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Translate an array by (dy, dx); vacated pixels copy the nearest edge."""
    out = np.empty_like(img)
    h, w = img.shape[:2]
    ys_src = np.clip(np.arange(h) - dy, 0, h - 1)
    xs_src = np.clip(np.arange(w) - dx, 0, w - 1)
    return img[np.ix_(ys_src, xs_src)] if img.ndim == 2 else \
        img[ys_src][:, xs_src]


# ---------------------------------------------------------------------------
# Wound segmentation -- redness-index heuristic (measurement, not diagnosis)
# ---------------------------------------------------------------------------

def redness_index(arr: np.ndarray) -> np.ndarray:
    f = arr.astype(np.float64)
    return f[..., 0] - 0.5 * (f[..., 1] + f[..., 2])


def _morph(mask: np.ndarray, fn, iters: int = 1) -> np.ndarray:
    m = mask
    for _ in range(iters):
        padded = np.pad(m, 1, mode="constant")
        neigh = [padded[1 + dy:1 + dy + m.shape[0], 1 + dx:1 + dx + m.shape[1]]
                 for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
        m = fn(neigh)
    return m


def _clean(mask: np.ndarray) -> np.ndarray:
    opened = _morph(mask, lambda n: np.logical_and.reduce(n), iters=1)
    closed = _morph(opened, lambda n: np.logical_or.reduce(n), iters=1)
    return closed


SPREAD_THRESHOLD = 85.0     # moderate redness: inflamed margin / spread
WOUND_REDNESS = 105.0       # strong redness: wound bed
WOUND_DARK_GRAY = 140.0     # or much darker than skin: darkened wound bed


def segment_wound(arr: np.ndarray) -> np.ndarray:
    """Boolean wound mask: strong redness OR much darker than skin.

    Two channels because wound beds fail in opposite directions: a healthy
    granulating bed is strongly red (synthetic index ~120 vs skin ~54),
    while a darkened bed loses redness but drops far below skin brightness
    (gray ~44-88 vs skin ~186). The inflamed halo in the fixtures sits
    between the channels (redness ~98, gray ~149) and is deliberately NOT
    part of the wound mask; it is measured separately as redness spread.
    All constants are stated here and proven in CI only against synthetic
    fixtures with known masks. Real skin tones vary, which is one reason
    real-photo accuracy is unproven (SECURITY.md).
    """
    mask = (redness_index(arr) > WOUND_REDNESS) | \
           (_to_gray(arr) < WOUND_DARK_GRAY)
    return _clean(mask)


# ---------------------------------------------------------------------------
# Pair analysis -- the diff is the measurement
# ---------------------------------------------------------------------------

def _bbox(mask: np.ndarray, pad: int = 8) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    h, w = mask.shape
    return [max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad),
            min(w, int(xs.max()) + pad + 1), min(h, int(ys.max()) + pad + 1)]


def analyse_pair(prev: np.ndarray, new: np.ndarray,
                 days_between: float | None = None) -> dict:
    """Full deterministic comparison of a previous and new wound photo."""
    if prev.shape != new.shape:
        raise ValueError("photo sizes differ; capture at the same size")
    dy, dx = estimate_shift(prev, new)
    aligned = shift_image(new, dy, dx)

    prev_mask = segment_wound(prev)
    new_mask = segment_wound(aligned)
    prev_area = int(prev_mask.sum())
    new_area = int(new_mask.sum())
    if prev_area == 0:
        raise ValueError("no wound region detected in the previous photo")

    area_change_pct = round(100.0 * (new_area - prev_area) / prev_area, 2)
    grew = new_mask & ~prev_mask
    healed = prev_mask & ~new_mask
    change_mask = grew | healed

    # Redness spread: red pixels in the new frame outside both wound masks.
    red = redness_index(aligned) > SPREAD_THRESHOLD
    spread_px = int((red & ~new_mask & ~prev_mask).sum())
    spread_pct = round(100.0 * spread_px / prev_area, 2)

    gray_prev = _to_gray(prev)
    gray_new = _to_gray(aligned)
    lum_prev = float(gray_prev[prev_mask].mean()) if prev_area else 0.0
    lum_new = float(gray_new[new_mask].mean()) if new_area else 0.0
    darkening = round(lum_prev - lum_new, 2)

    roi = _bbox(change_mask)
    result = {
        "shift_px": [dy, dx],
        "prev_area_px": prev_area,
        "new_area_px": new_area,
        "area_change_pct": area_change_pct,
        "grew_px": int(grew.sum()),
        "healed_px": int(healed.sum()),
        "redness_spread_pct": spread_pct,
        "darkening": darkening,
        "days_between": days_between,
        "roi": roi,
        "red_flags": [],
    }
    result["red_flags"] = detect_red_flags(result)
    result["escalate"] = bool(result["red_flags"])
    result["_aligned"] = aligned
    result["_change_mask"] = change_mask
    result["_new_mask"] = new_mask
    result["_prev_mask"] = prev_mask
    return result


def detect_red_flags(m: dict) -> list[dict]:
    """Pattern checks on measurements. Patterns, not diagnoses: each flag
    is phrased as something to raise with the care team."""
    flags = []
    if m["area_change_pct"] >= RAPID_EXPANSION_PCT:
        flags.append({
            "name": "rapid_expansion",
            "detail": f"area grew {m['area_change_pct']:.1f}% between photos",
        })
    if m["redness_spread_pct"] >= REDNESS_SPREAD_PCT:
        flags.append({
            "name": "redness_spread",
            "detail": f"new red area outside the wound equals "
                      f"{m['redness_spread_pct']:.1f}% of the wound area",
        })
    if m["darkening"] >= DARKENING_DELTA:
        flags.append({
            "name": "darkening",
            "detail": f"mean wound brightness dropped {m['darkening']:.1f} "
                      "points between photos",
        })
    if (m.get("days_between") is not None and m["days_between"] >= STALL_DAYS
            and abs(m["area_change_pct"]) <= STALL_MAX_CHANGE_PCT):
        flags.append({
            "name": "no_measurable_change",
            "detail": f"area changed {m['area_change_pct']:.1f}% over "
                      f"{m['days_between']:.0f} days",
        })
    return flags


def roi_crop(arr: np.ndarray, roi: list[int] | None) -> np.ndarray:
    if roi is None:
        return arr
    x0, y0, x1, y1 = roi
    return arr[y0:y1, x0:x1]


def ghost_overlay(prev: np.ndarray, new: np.ndarray,
                  alpha: float = 0.5) -> np.ndarray:
    """The capture guide as an image: previous photo at ``alpha`` over the
    new one, so misalignment is visible before the photo is accepted."""
    a = prev.astype(np.float64)
    b = new.astype(np.float64)
    return ((1 - alpha) * b + alpha * a).astype(np.uint8)


def public_measurements(analysis: dict) -> dict:
    """Strip private array payloads; what an API response / model may see."""
    return {k: v for k, v in analysis.items() if not k.startswith("_")}
