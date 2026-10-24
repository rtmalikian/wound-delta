"""The compare pipeline: hash gate, token budget, tiers, verifier."""

from app import gateway
from app.wounddelta import service, synth


def test_identical_pair_costs_zero_model_calls():
    pair = synth.make_pair(seed=51)
    out = service.compare_pair(pair["prev_png"], pair["prev_png"])
    assert out["changed"] is False
    assert out["usage"]["model_calls"] == 0
    assert gateway.CALL_LEDGER["chat"] == 0


def test_frugal_pair_within_token_budget():
    pair = synth.make_pair(area_change_pct=-30.0, seed=52)
    out = service.compare_pair(pair["prev_png"], pair["new_png"],
                               tier="frugal",
                               prev_date="2026-10-16", new_date="2026-10-23")
    assert out["measurements"]["area_change_pct"] < 0
    assert out["usage"]["model_calls"] == 1
    assert out["usage"]["est_tokens"] <= 300, out["usage"]
    assert out["usage"]["budget"] == 300


def test_frugal_negligible_change_costs_zero_calls():
    pair = synth.make_pair(area_change_pct=0.0, days=3, seed=53)
    out = service.compare_pair(pair["prev_png"], pair["new_png"],
                               tier="frugal",
                               prev_date="2026-10-20", new_date="2026-10-23")
    assert out["usage"]["model_calls"] == 0
    assert gateway.CALL_LEDGER["chat"] == 0


def test_summary_contains_measured_number():
    pair = synth.make_pair(area_change_pct=-30.0, seed=54)
    out = service.compare_pair(pair["prev_png"], pair["new_png"],
                               prev_date="2026-10-16", new_date="2026-10-23")
    measured = abs(out["measurements"]["area_change_pct"])
    assert f"{measured:.1f}%" in out["summary"]


def test_full_tier_runs_verifier_and_costs_more():
    pair = synth.make_pair(area_change_pct=-30.0, seed=55)
    frugal = service.compare_pair(pair["prev_png"], pair["new_png"],
                                  tier="frugal",
                                  prev_date="2026-10-16",
                                  new_date="2026-10-23")
    full = service.compare_pair(pair["prev_png"], pair["new_png"],
                                tier="full",
                                prev_date="2026-10-16",
                                new_date="2026-10-23")
    assert full["usage"]["model_calls"] == 2  # summary + verifier
    assert full["verifier"] is not None
    assert full["verifier"]["ok"] is True
    assert full["usage"]["est_tokens"] > frugal["usage"]["est_tokens"]


def test_deteriorating_pair_escalates():
    pair = synth.make_pair(area_change_pct=40.0, redness_halo=True, seed=56)
    out = service.compare_pair(pair["prev_png"], pair["new_png"],
                               prev_date="2026-10-16", new_date="2026-10-23")
    assert out["measurements"]["escalate"] is True
    names = {f["name"] for f in out["measurements"]["red_flags"]}
    assert "rapid_expansion" in names
    assert "care team" in out["summary"]
