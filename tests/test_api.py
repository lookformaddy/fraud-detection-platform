"""API tests (FastAPI TestClient). These run only when fastapi and
httpx are installed; otherwise they are skipped, not failed.
"""
import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from frauddet.service import app  # noqa: E402
from tests.conftest import BASE_TS  # noqa: E402

client = TestClient(app)


def _payload(**overrides):
    body = {
        "txn_id": "txn-api-1",
        "account_id": "acct-api-1",
        "device_id": "dev-api-1",
        "ip": "10.0.1.1",
        "amount": 120.0,
        "timestamp": BASE_TS + 3600,
    }
    body.update(overrides)
    return body


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert resp.json()["scorer_trained"] is True


def test_score_returns_decision_and_explanation():
    resp = client.post("/score", json=_payload())
    assert resp.status_code == 200
    data = resp.json()
    assert data["decision"] in ("approve", "step_up", "review", "block")
    assert 0.0 <= data["calibrated_score"] <= 1.0
    assert isinstance(data["top_features"], list)
    assert data["top_features"]  # explanations are always present


def test_score_duplicate_is_conflict():
    client.post("/score", json=_payload(txn_id="txn-api-dup"))
    resp = client.post("/score", json=_payload(txn_id="txn-api-dup"))
    assert resp.status_code == 409


def test_score_validates_amount():
    resp = client.post("/score", json=_payload(amount=-5.0))
    assert resp.status_code == 422


def test_alert_case_disposition_flow():
    # A blatantly fraudulent transaction should open an alert.
    resp = client.post("/score", json=_payload(
        txn_id="txn-api-fraud", device_id="dev-fraud-api",
        ip="203.0.9.9", amount=5000.0, timestamp=BASE_TS + 7200,
        latitude=51.5074, longitude=-0.1278, city="London", country="GB",
    ))
    assert resp.status_code == 200
    data = resp.json()
    assert data["decision"] in ("review", "block")
    alert_id = data["alert_id"]
    assert alert_id is not None

    alerts = client.get("/alerts").json()["alerts"]
    assert any(a["alert_id"] == alert_id for a in alerts)

    ack = client.post(f"/alerts/{alert_id}/acknowledge")
    assert ack.status_code == 200
    assert ack.json()["status"] == "acknowledged"

    case = client.post(f"/alerts/{alert_id}/case").json()
    case_id = case["case_id"]
    assigned = client.post(f"/cases/{case_id}/assign", params={"assignee": "analyst-1"})
    assert assigned.status_code == 200
    assert assigned.json()["status"] == "assigned"

    resolved = client.post(f"/cases/{case_id}/resolve", json={
        "disposition": "confirmed_fraud", "analyst": "analyst-1",
        "device_id": "dev-fraud-api", "ip": "203.0.9.9",
        "rule_ids": ["NEW_DEVICE_AND_IP"],
    })
    assert resolved.status_code == 200
    assert resolved.json()["case"]["disposition"] == "confirmed_fraud"
    assert resolved.json()["case"]["status"] == "resolved"


def test_unknown_alert_404():
    assert client.post("/alerts/nope/acknowledge").status_code == 404
    assert client.post("/alerts/nope/case").status_code == 404


def test_metrics_and_models_endpoints():
    assert client.get("/metrics").status_code == 200
    models = client.get("/models").json()
    assert models["champion"] == "v1-seed42"
    assert models["versions"]
