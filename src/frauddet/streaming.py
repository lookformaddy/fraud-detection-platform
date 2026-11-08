"""Streaming scoring: in-order event processing with watermark,
late-event handling, dedup, and the alert lifecycle.

Semantics:
- The watermark is ``max_event_ts - allowed_lateness``. An event
  with ``timestamp < watermark`` is *late*: it is still scored (the
  cardholder's money is at stake either way) but flagged, counted,
  and it does **not** update feature-store state — late data must not
  rewrite history that in-order decisions already used.
- A repeated ``txn_id`` is a duplicate: it is counted and returned
  without re-scoring, so retries never double-alert.
- REVIEW / BLOCK decisions open an alert (deduplicated per txn).
  Alerts move OPEN -> ACKNOWLEDGED -> RESOLVED; resolution happens
  through the case queue (see cases.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .decisioning import score_transaction
from .features import FeatureStore, FeatureStoreUnavailable
from .models import (
    Alert, AlertStatus, Decision, ScoredTransaction, Thresholds, Transaction,
)
from .rules import RuleEngine
from .scorer import FraudScorer


@dataclass
class ProcessedEvent:
    scored: ScoredTransaction | None
    is_duplicate: bool = False
    is_late: bool = False
    alert: Alert | None = None
    error: str | None = None


@dataclass
class StreamStats:
    processed: int = 0
    duplicates: int = 0
    late: int = 0
    feature_store_errors: int = 0
    alerts_opened: int = 0
    by_decision: dict[str, int] = field(
        default_factory=lambda: {d.value: 0 for d in Decision}
    )


class StreamProcessor:
    """Stateful, single-partition stream scorer."""

    def __init__(self, scorer: FraudScorer,
                 thresholds: Thresholds | None = None,
                 allowed_lateness_seconds: int = 300) -> None:
        self.scorer = scorer
        self.thresholds = thresholds or Thresholds()
        self.allowed_lateness = allowed_lateness_seconds
        self.store = FeatureStore()
        self.rules = RuleEngine()
        self.watermark = 0
        self._seen_txn_ids: set[str] = set()
        self.alerts: dict[str, Alert] = {}
        self.stats = StreamStats()

    def process(self, txn: Transaction) -> ProcessedEvent:
        if txn.txn_id in self._seen_txn_ids:
            self.stats.duplicates += 1
            return ProcessedEvent(scored=None, is_duplicate=True)
        self._seen_txn_ids.add(txn.txn_id)

        is_late = txn.timestamp < self.watermark
        if is_late:
            self.stats.late += 1

        try:
            features = self.store.compute_features(txn)
        except FeatureStoreUnavailable:
            self.stats.feature_store_errors += 1
            # Fail closed: no features, no auto-approve.
            scored = score_transaction(
                txn, _empty_features(txn), 0.0, 0.0, [], [],
                self.thresholds, feature_store_available=False,
            )
            alert = self._maybe_alert(scored, txn)
            self.stats.processed += 1
            self.stats.by_decision[scored.decision.value] += 1
            return ProcessedEvent(scored=scored, is_late=is_late, alert=alert)

        raw, calibrated, top = self.scorer.score(features)
        rule_hits = self.rules.evaluate(txn, features)
        scored = score_transaction(
            txn, features, raw, calibrated, rule_hits, top,
            self.thresholds, feature_store_available=True,
        )
        if not is_late:
            self.store.update(txn)
            self.watermark = max(self.watermark, txn.timestamp - self.allowed_lateness)
        alert = self._maybe_alert(scored, txn)
        self.stats.processed += 1
        self.stats.by_decision[scored.decision.value] += 1
        return ProcessedEvent(scored=scored, is_late=is_late, alert=alert)

    def _maybe_alert(self, scored: ScoredTransaction, txn: Transaction) -> Alert | None:
        if scored.decision not in (Decision.REVIEW, Decision.BLOCK):
            return None
        alert_id = f"alert-{txn.txn_id}"
        if alert_id in self.alerts:
            return self.alerts[alert_id]
        alert = Alert(
            alert_id=alert_id, txn_id=txn.txn_id, account_id=txn.account_id,
            score=scored.calibrated_score, band=scored.decision,
            created_ts=txn.timestamp,
            rule_ids=[h.rule_id for h in scored.rule_hits],
        )
        self.alerts[alert_id] = alert
        self.stats.alerts_opened += 1
        return alert

    def acknowledge_alert(self, alert_id: str) -> Alert | None:
        alert = self.alerts.get(alert_id)
        if alert is not None and alert.status == AlertStatus.OPEN:
            alert.status = AlertStatus.ACKNOWLEDGED
        return alert

    def resolve_alert(self, alert_id: str) -> Alert | None:
        alert = self.alerts.get(alert_id)
        if alert is not None:
            alert.status = AlertStatus.RESOLVED
        return alert

    def open_alerts(self) -> list[Alert]:
        return [a for a in self.alerts.values() if a.status == AlertStatus.OPEN]


def _empty_features(txn: Transaction):
    from .features import FEATURE_NAMES
    from .models import FeatureVector

    return FeatureVector(txn_id=txn.txn_id, values={n: 0.0 for n in FEATURE_NAMES})
