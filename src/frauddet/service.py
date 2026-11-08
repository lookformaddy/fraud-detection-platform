"""FastAPI service — thin wrapper over the pure-stdlib core.

The service owns no decision logic: every endpoint delegates to
``StreamProcessor`` / ``CaseQueue`` / ``ModelRegistry``. A scorer is
trained once at startup on seeded synthetic data so ``/score`` works
out of the box; replace it with a persisted model in production.
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .cases import CaseQueue, FeedbackLoop
from .models import Disposition, Transaction
from .registry import ModelRegistry, ModelVersion
from .scorer import FraudScorer, generate_synthetic_transactions
from .streaming import StreamProcessor

app = FastAPI(title="fraud-detection-platform", version="1.0.0")

_scorer = FraudScorer()
_txns, _labels = generate_synthetic_transactions(n=800, seed=42)
_scorer.train(_txns, _labels, seed=42)
processor = StreamProcessor(_scorer)
feedback = FeedbackLoop(rule_engine=processor.rules, store=processor.store)
case_queue = CaseQueue(feedback)
registry = ModelRegistry()
registry.register(ModelVersion(
    version_id="v1-seed42", seed=42,
    metrics={"roc_auc": 0.0, "precision_at_review": 0.0,
             "recall_at_review": 0.0, "f1_at_review": 0.0},
))
registry.set_champion("v1-seed42")


class TransactionIn(BaseModel):
    txn_id: str
    account_id: str
    device_id: str
    ip: str
    amount: float = Field(gt=0)
    timestamp: int
    country: str = "US"
    city: str = "San Francisco"
    latitude: float = 37.7749
    longitude: float = -122.4194
    merchant_category: str = "general"
    channel: str = "online"
    currency: str = "USD"

    def to_transaction(self) -> Transaction:
        return Transaction(**self.model_dump())


class DispositionIn(BaseModel):
    disposition: Disposition
    analyst: str
    notes: str = ""
    device_id: str = ""
    ip: str = ""
    rule_ids: list[str] = []


def _alert_view(alert) -> dict:
    return {
        "alert_id": alert.alert_id, "txn_id": alert.txn_id,
        "account_id": alert.account_id, "score": round(alert.score, 4),
        "band": alert.band.value, "status": alert.status.value,
        "created_ts": alert.created_ts, "rule_ids": alert.rule_ids,
    }


def _case_view(case) -> dict:
    return {
        "case_id": case.case_id, "alert_id": case.alert_id,
        "txn_id": case.txn_id, "account_id": case.account_id,
        "score": round(case.score, 4), "status": case.status.value,
        "assignee": case.assignee,
        "disposition": case.disposition.value if case.disposition else None,
        "analyst": case.analyst, "notes": case.notes,
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "scorer_trained": _scorer.trained,
        "feature_store_available": processor.store.available,
        "open_alerts": len(processor.open_alerts()),
        "open_cases": len(case_queue.open_cases()),
    }


@app.post("/score")
def score(payload: TransactionIn) -> dict:
    result = processor.process(payload.to_transaction())
    scored = result.scored
    if scored is None:
        raise HTTPException(status_code=409, detail="duplicate transaction")
    return {
        "txn_id": scored.transaction.txn_id,
        "raw_score": round(scored.raw_score, 4),
        "calibrated_score": round(scored.calibrated_score, 4),
        "decision": scored.decision.value,
        "decision_reason": scored.decision_reason,
        "is_duplicate": result.is_duplicate,
        "is_late": result.is_late,
        "rule_hits": [
            {"rule_id": h.rule_id, "message": h.message,
             "score_boost": round(h.score_boost, 3),
             "blocking": h.blocking, "severity": h.severity}
            for h in scored.rule_hits
        ],
        "top_features": [
            {"feature": name, "contribution": round(c, 4)}
            for name, c in scored.top_features
        ],
        "alert_id": result.alert.alert_id if result.alert else None,
    }


@app.get("/alerts")
def list_alerts(status: str | None = None) -> dict:
    alerts = list(processor.alerts.values())
    if status:
        alerts = [a for a in alerts if a.status.value == status]
    return {"alerts": [_alert_view(a) for a in alerts]}


@app.post("/alerts/{alert_id}/acknowledge")
def acknowledge_alert(alert_id: str) -> dict:
    alert = processor.acknowledge_alert(alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail="unknown alert")
    return _alert_view(alert)


@app.post("/alerts/{alert_id}/case")
def open_case(alert_id: str) -> dict:
    alert = processor.alerts.get(alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail="unknown alert")
    return _case_view(case_queue.open_case(alert))


@app.get("/cases")
def list_cases(status: str | None = None) -> dict:
    cases = list(case_queue.cases.values())
    if status:
        cases = [c for c in cases if c.status.value == status]
    return {"cases": [_case_view(c) for c in cases]}


@app.post("/cases/{case_id}/assign")
def assign_case(case_id: str, assignee: str) -> dict:
    try:
        return _case_view(case_queue.assign(case_id, assignee))
    except KeyError:
        raise HTTPException(status_code=404, detail="unknown case")


@app.post("/cases/{case_id}/resolve")
def resolve_case(case_id: str, payload: DispositionIn) -> dict:
    try:
        result = case_queue.resolve(
            case_id, payload.disposition, payload.analyst, payload.notes,
            device_id=payload.device_id, ip=payload.ip,
            rule_ids=payload.rule_ids,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="unknown case")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    case = result["case"]
    processor.resolve_alert(case.alert_id)
    return {"case": _case_view(case), "feedback": result["feedback"]}


@app.get("/metrics")
def metrics() -> dict:
    return {
        "stream": {
            "processed": processor.stats.processed,
            "duplicates": processor.stats.duplicates,
            "late": processor.stats.late,
            "feature_store_errors": processor.stats.feature_store_errors,
            "alerts_opened": processor.stats.alerts_opened,
            "by_decision": processor.stats.by_decision,
            "watermark": processor.watermark,
        },
        "label_quality": case_queue.quality.as_dict(),
        "retrain_due": feedback.retrain_due,
        "labels_since_retrain": feedback.labels_since_retrain,
    }


@app.get("/models")
def list_models() -> dict:
    return {
        "champion": registry.champion_id,
        "versions": [
            {"version_id": v.version_id, "seed": v.seed,
             "status": v.status, "metrics": v.metrics,
             "gate_report": v.gate_report}
            for v in registry.versions.values()
        ],
    }


@app.post("/models/{version_id}/promote")
def promote_model(version_id: str, seed_aucs: list[float]) -> dict:
    if version_id not in registry.versions:
        raise HTTPException(status_code=404, detail="unknown model version")
    return registry.promote(version_id, seed_aucs)
