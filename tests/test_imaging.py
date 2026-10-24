"""Deterministic mechanics: alignment, segmentation, diff, ROI, overlay."""

import numpy as np
import pytest

from app.wounddelta import imaging, synth


def test_identical_images_have_identical_hash():
    pair = synth.make_pair(seed=41)
    a = imaging.load_image(pair["prev_png"])
    b = imaging.load_image(pair["prev_png"])
    assert imaging.image_sha(a) == imaging.image_sha(b)


def test_alignment_recovers_known_shift():
    pair = synth.make_pair(area_change_pct=-20.0, shift=(6, -5), seed=42)
    prev = imaging.load_image(pair["prev_png"])
    new = imaging.load_image(pair["new_png"])
    result = imaging.analyse_pair(prev, new)
    assert abs(result["area_change_pct"] - pair["truth"]["area_change_pct"]) < 10


@pytest.mark.parametrize("pct", [-40.0, -25.0, -10.0, 0.0, 15.0, 30.0])
def test_area_change_accuracy_across_range(pct):
    pair = synth.make_pair(area_change_pct=pct, seed=43)
    prev = imaging.load_image(pair["prev_png"])
    new = imaging.load_image(pair["new_png"])
    result = imaging.analyse_pair(prev, new)
    err = abs(result["area_change_pct"] - pair["truth"]["area_change_pct"])
    assert err < 10.0, f"pct={pct} err={err}"


def test_segmentation_area_close_to_drawn_mask():
    pair = synth.make_pair(area_change_pct=0.0, seed=44)
    arr = imaging.load_image(pair["prev_png"])
    mask = imaging.segment_wound(arr)
    truth = pair["truth"]["prev_area_px"]
    assert abs(int(mask.sum()) - truth) / truth < 0.15


def test_roi_contains_the_change():
    pair = synth.make_pair(area_change_pct=45.0, seed=45)
    prev = imaging.load_image(pair["prev_png"])
    new = imaging.load_image(pair["new_png"])
    result = imaging.analyse_pair(prev, new)
    roi = result["roi"]
    assert roi is not None
    x0, y0, x1, y1 = roi
    # The wound sits at the centre of the synthetic frame; the change
    # region's box must cover the centre.
    assert x0 <= 128 <= x1 and y0 <= 128 <= y1


def test_ghost_overlay_blends_both_frames():
    pair = synth.make_pair(seed=46)
    prev = imaging.load_image(pair["prev_png"])
    new = imaging.load_image(pair["new_png"])
    ghost = imaging.ghost_overlay(prev, new, alpha=0.5)
    assert ghost.shape == prev.shape
    assert not np.array_equal(ghost, prev)
    assert not np.array_equal(ghost, new)


def test_size_mismatch_is_an_error():
    pair = synth.make_pair(seed=47)
    prev = imaging.load_image(pair["prev_png"])
    with pytest.raises(ValueError):
        imaging.analyse_pair(prev, prev[:100, :100])


def test_lighting_change_does_not_fabricate_growth():
    pair = synth.make_pair(area_change_pct=-20.0, lighting=1.06, seed=48)
    prev = imaging.load_image(pair["prev_png"])
    new = imaging.load_image(pair["new_png"])
    result = imaging.analyse_pair(prev, new)
    assert abs(result["area_change_pct"] - (-20.0)) < 10
