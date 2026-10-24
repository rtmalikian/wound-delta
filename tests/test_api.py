"""Product API end-to-end, in stub mode."""

import io

from app.wounddelta import synth


def _upload(client, case_id, png, taken_on):
    return client.post(f"/api/cases/{case_id}/captures",
                       files={"file": ("wound.png", io.BytesIO(png),
                                       "image/png")},
                       data={"taken_on": taken_on})


def test_healthz(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["backend"] == "stub"


def test_case_flow_compare_and_trend(client):
    case = client.post("/api/cases", json={"label": "left heel, demo"}).json()
    pair = synth.make_pair(area_change_pct=-25.0, seed=71)
    r1 = _upload(client, case["id"], pair["prev_png"], "2026-10-16")
    assert r1.status_code == 200
    r2 = _upload(client, case["id"], pair["new_png"], "2026-10-23")
    assert r2.status_code == 200

    comp = client.post(f"/api/cases/{case['id']}/compare")
    assert comp.status_code == 200
    body = comp.json()
    assert body["measurements"]["area_change_pct"] < 0
    assert body["usage"]["model_calls"] == 1

    trend = client.get(f"/api/cases/{case['id']}/trend").json()
    assert len(trend["points"]) == 2
    assert trend["total_change_pct"] < 0


def test_compare_requires_two_captures(client):
    case = client.post("/api/cases", json={"label": "demo"}).json()
    r = client.post(f"/api/cases/{case['id']}/compare")
    assert r.status_code == 400


def test_unknown_case_404(client):
    assert client.get("/api/cases/wnd-nope").status_code == 404


def test_upload_rejects_non_image(client):
    case = client.post("/api/cases", json={"label": "demo"}).json()
    r = client.post(f"/api/cases/{case['id']}/captures",
                    files={"file": ("x.txt", io.BytesIO(b"not an image"),
                                    "text/plain")},
                    data={"taken_on": "2026-10-23"})
    assert r.status_code == 400


def test_full_tier_via_header(client):
    case = client.post("/api/cases", json={"label": "demo full"}).json()
    pair = synth.make_pair(area_change_pct=-25.0, seed=72)
    _upload(client, case["id"], pair["prev_png"], "2026-10-16")
    _upload(client, case["id"], pair["new_png"], "2026-10-23")
    comp = client.post(f"/api/cases/{case['id']}/compare",
                       headers={"X-Tier": "full"})
    body = comp.json()
    assert body["tier"] == "full"
    assert body["usage"]["model_calls"] == 2
    assert body["verifier"]["ok"] is True


def test_demo_compare_endpoint(client):
    r = client.post("/api/demo/compare?kind=worse")
    assert r.status_code == 200
    body = r.json()
    assert body["measurements"]["escalate"] is True
    assert abs(body["truth"]["area_change_pct"] - 38.0) < 0.5


def test_demo_pair_truth_endpoint(client):
    r = client.get("/api/demo/pair?kind=healing")
    assert r.status_code == 200
    assert abs(r.json()["area_change_pct"] - (-30.0)) < 0.5
