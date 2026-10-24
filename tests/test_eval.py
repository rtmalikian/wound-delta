"""Eval harness -- the CI gates from the product brief.

These are not unit tests of convenience; they are the claims in the
README, enforced:

- area-change MAE < 10 percentage points vs synthetic ground truth
- red-flag pattern recall >= 0.90 on deterioration fixtures
- benign fixtures stay (mostly) quiet
- frugal tier median estimated tokens per pair <= 300
- unchanged pair == zero model calls (also covered in test_service)
"""

import statistics

from app import gateway
from app.wounddelta import imaging, service, synth


def _errors():
    errs = []
    for pct in (-45.0, -30.0, -15.0, 0.0, 10.0, 25.0, 40.0):
        pair = synth.make_pair(area_change_pct=pct, seed=100 + int(pct))
        result = imaging.analyse_pair(imaging.load_image(pair["prev_png"]),
                                      imaging.load_image(pair["new_png"]))
        errs.append(abs(result["area_change_pct"]
                        - pair["truth"]["area_change_pct"]))
    return errs


def test_eval_area_change_mae_below_10_points():
    errs = _errors()
    mae = sum(errs) / len(errs)
    assert mae < 10.0, f"MAE {mae:.2f} over errors {errs}"


def test_eval_red_flag_recall_at_least_090():
    fixtures = synth.deterioration_fixtures()
    bad = [f for f in fixtures if f["should_flag"]]
    hits = 0
    for f in bad:
        pair = f["pair"]
        result = imaging.analyse_pair(
            imaging.load_image(pair["prev_png"]),
            imaging.load_image(pair["new_png"]),
            days_between=pair["truth"]["days_between"])
        if result["red_flags"]:
            hits += 1
    recall = hits / len(bad)
    assert recall >= 0.90, f"recall {recall:.2f} ({hits}/{len(bad)})"


def test_eval_benign_fixtures_mostly_quiet():
    fixtures = synth.deterioration_fixtures()
    good = [f for f in fixtures if not f["should_flag"]]
    quiet = 0
    for f in good:
        pair = f["pair"]
        result = imaging.analyse_pair(
            imaging.load_image(pair["prev_png"]),
            imaging.load_image(pair["new_png"]),
            days_between=pair["truth"]["days_between"])
        if not result["red_flags"]:
            quiet += 1
    assert quiet >= len(good) - 1


def test_eval_frugal_token_budget_median():
    tokens = []
    for pct, seed in ((-30.0, 61), (25.0, 62), (-12.0, 63)):
        pair = synth.make_pair(area_change_pct=pct, seed=seed)
        out = service.compare_pair(pair["prev_png"], pair["new_png"],
                                   tier="frugal",
                                   prev_date="2026-10-16",
                                   new_date="2026-10-23")
        if out["usage"]["model_calls"]:
            tokens.append(out["usage"]["est_tokens"])
    assert tokens, "expected at least one frugal model call to measure"
    assert statistics.median(tokens) <= 300
