"""Synthetic wound-photo pairs with known ground truth.

The eval harness needs pairs where the true area change is *known*, not
estimated. This module draws them: a skin-tone field, an ellipse wound bed
(the exact pixels drawn are the ground-truth mask), and a second frame with
the ellipse scaled by a chosen factor, optionally shifted, relit, darkened,
or given a redness halo. The same generator seeds the demo UI, so what
visitors see is exactly what CI measures.
"""

from __future__ import annotations

import numpy as np

from app.wounddelta import imaging

SIZE = 256
SKIN = np.array([222.0, 184.0, 152.0])
WOUND = np.array([168.0, 52.0, 44.0])
# Inflamed margin: redness ~96 (above the spread threshold 85, below the
# wound-bed threshold 105), gray ~152 (above the dark-wound channel 140).
# It must sit between the channels or segmentation would swallow it.
HALO = np.array([216.0, 132.0, 108.0])
CENTER = (128, 128)
BASE_RADII = (46, 34)  # rx, ry


def _ellipse_mask(shape, center, radii) -> np.ndarray:
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return ((xx - center[0]) / radii[0]) ** 2 + \
           ((yy - center[1]) / radii[1]) ** 2 <= 1.0


def _render(rng: np.random.Generator, radii, wound_rgb=WOUND,
            halo_px: int = 0, lighting: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    img = np.tile(SKIN, (SIZE, SIZE, 1)).astype(np.float64)
    img += rng.normal(0, 4.0, img.shape)
    mask = _ellipse_mask((SIZE, SIZE), CENTER, radii)
    wound = np.tile(np.asarray(wound_rgb, dtype=np.float64), (SIZE, SIZE, 1))
    wound += rng.normal(0, 5.0, img.shape)
    img[mask] = wound[mask]
    if halo_px > 0:
        # A red annulus hugging the wound margin, sized to ~halo_px pixels.
        grow = 1.0
        outer = _ellipse_mask((SIZE, SIZE), CENTER,
                              (radii[0] * grow, radii[1] * grow))
        while int(outer.sum() - mask.sum()) < halo_px and grow < 2.5:
            grow += 0.05
            outer = _ellipse_mask((SIZE, SIZE), CENTER,
                                  (radii[0] * grow, radii[1] * grow))
        halo = outer & ~mask
        halo_img = np.tile(HALO, (SIZE, SIZE, 1)).astype(np.float64)
        halo_img += rng.normal(0, 4.0, img.shape)
        img[halo] = halo_img[halo]
    img = np.clip(img * lighting, 0, 255).astype(np.uint8)
    return img, mask


def make_pair(area_change_pct: float = -20.0, shift: tuple[int, int] = (0, 0),
              lighting: float = 1.0, redness_halo: bool = False,
              darken: float = 0.0, days: float = 7.0,
              seed: int = 7) -> dict:
    """One synthetic pair + ground truth.

    ``darken`` is a 0..1 multiplier applied to the new wound's colour
    (0.55 simulates a strongly darkened wound bed).
    """
    rng = np.random.default_rng(seed)
    prev, prev_mask = _render(rng, BASE_RADII)
    scale = float(np.sqrt(max(0.05, 1.0 + area_change_pct / 100.0)))
    new_radii = (BASE_RADII[0] * scale, BASE_RADII[1] * scale)
    wound_rgb = WOUND * (1.0 - darken) if darken else WOUND
    halo_px = int(0.35 * prev_mask.sum()) if redness_halo else 0
    new, new_mask = _render(rng, new_radii, wound_rgb=wound_rgb,
                            halo_px=halo_px, lighting=lighting)
    if shift != (0, 0):
        new = imaging.shift_image(new, shift[0], shift[1])
    truth_area_new = int(new_mask.sum())
    truth = {
        "prev_area_px": int(prev_mask.sum()),
        "new_area_px": truth_area_new,
        "area_change_pct": round(
            100.0 * (truth_area_new - int(prev_mask.sum()))
            / int(prev_mask.sum()), 2),
        "shift_px": list(shift),
        "days_between": days,
        "redness_halo": redness_halo,
        "darken": darken,
    }
    return {
        "prev_png": imaging.encode_png(prev),
        "new_png": imaging.encode_png(new),
        "truth": truth,
    }


def deterioration_fixtures() -> list[dict]:
    """Pairs whose ground truth contains a red-flag pattern, for the
    recall gate, plus benign pairs that should stay quiet or nearly so."""
    bad = [
        make_pair(area_change_pct=40.0, seed=11),                    # expansion
        make_pair(area_change_pct=55.0, seed=12),                    # expansion
        make_pair(area_change_pct=-5.0, redness_halo=True, seed=13),  # spread
        make_pair(area_change_pct=-8.0, redness_halo=True, seed=14),  # spread
        make_pair(area_change_pct=-10.0, darken=0.5, seed=15),        # darkening
        make_pair(area_change_pct=-6.0, darken=0.55, seed=16),       # darkening
        make_pair(area_change_pct=0.0, days=21, seed=17),            # stall
        make_pair(area_change_pct=1.0, days=28, seed=18),            # stall
        make_pair(area_change_pct=35.0, shift=(5, -4), seed=19),     # expansion
        make_pair(area_change_pct=-4.0, redness_halo=True,
                  darken=0.4, seed=20),                              # both
    ]
    good = [
        make_pair(area_change_pct=-20.0, seed=21),
        make_pair(area_change_pct=-35.0, seed=22),
        make_pair(area_change_pct=-15.0, shift=(3, 3), seed=23),
        make_pair(area_change_pct=5.0, days=3, seed=24),
    ]
    return [{"pair": p, "should_flag": True} for p in bad] + \
           [{"pair": p, "should_flag": False} for p in good]
