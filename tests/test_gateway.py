"""Gateway shape, tier resolution, and the token ledger."""

from app import gateway


def test_models_list_exposes_both_tier_aliases(client):
    data = client.get("/v1/models").json()
    ids = {m["id"] for m in data["data"]}
    assert gateway.FRUGAL_MODEL_ID in ids
    assert gateway.FULL_MODEL_ID in ids
    assert gateway.STUB_MODEL_ID in ids


def test_tier_resolution_from_model_alias():
    assert gateway.resolve_tier("wound-delta-full") == "full"
    assert gateway.resolve_tier("wound-delta-frugal") == "frugal"
    assert gateway.resolve_tier("anything-full") == "full"
    assert gateway.resolve_tier(None) == "frugal"


def test_tier_resolution_header_wins():
    assert gateway.resolve_tier("wound-delta-frugal", "full") == "full"
    assert gateway.resolve_tier("wound-delta-full", "frugal") == "frugal"


def test_image_token_tile_maths():
    # Published tile maths: low detail is a flat 85 at any size;
    # high detail at 1024x1024 is 85 + 4 tiles * 170 = 765.
    assert gateway.estimate_image_tokens("low", 4000, 3000) == 85
    assert gateway.estimate_image_tokens("high", 1024, 1024) == 765
    assert gateway.estimate_image_tokens("high", 512, 512) == 255


def test_chat_completion_shape(client):
    r = client.post("/v1/chat/completions", json={
        "model": "wound-delta-frugal",
        "messages": [{"role": "user", "content":
                      "[MEASUREMENTS]\n{\"area_change_pct\": -12.0, "
                      "\"red_flags\": []}\n[/MEASUREMENTS]"}],
    })
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert "12.0%" in body["choices"][0]["message"]["content"]
    assert body["usage"]["total_tokens"] > 0


def test_chat_requires_messages(client):
    r = client.post("/v1/chat/completions", json={"messages": []})
    assert r.status_code == 400


def test_embeddings_shape(client):
    r = client.post("/v1/embeddings", json={"input": ["pair summary a",
                                                      "pair summary b"]})
    assert r.status_code == 200
    data = r.json()["data"]
    assert len(data) == 2
    assert len(data[0]["embedding"]) == 16


def test_stub_summary_uses_only_measured_numbers():
    m = {"area_change_pct": -23.5, "days_between": 7, "red_flags": []}
    out = gateway._stub_summarise(__import__("json").dumps(m))
    assert "23.5%" in out
    assert "7 days" in out


def test_stub_verifier_flags_invented_numbers():
    import json
    m = json.dumps({"area_change_pct": -10.0, "red_flags": []})
    ok = json.loads(gateway._stub_verify(m, "The wound shrank 10.0%."))
    bad = json.loads(gateway._stub_verify(m, "The wound shrank 42.0%."))
    assert ok["ok"] is True
    assert bad["ok"] is False
    assert 42.0 in bad["stray_numbers"]
