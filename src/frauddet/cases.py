"""Analyst case management and the feedback loop.

An alert becomes a case; an analyst assigns it and resolves it with
a disposition (confirmed fraud / false positive / undetermined).
Dispositions feed back into the system in three ways:

1. **Blocklists** — confirmed fraud adds the device / IP / account
   to the rule engine's blocklists, so the next attempt is blocked
   outright.
2. **Reputation** — confirmed outcomes update device/IP fraud rates
   in the feature store, so future scores see the reputation.
3. **Rule weights** — each disposition nudges the weight multiplier
   of the rules that fired: confirmed fraud raises them (up to 2x),
   false positives lower them (down to 0.5x). The model itself is
   retrained on the relabelled set once enough new labels arrive
   (``retrain_threshold``), which is the explicit retrain trigger.

Label quality is tracked, not assumed: the queue records
analyst agreement (two analysts disposing the same case
differently), the fraud-confirmation rate, and the share of
undetermined cases — the three numbers that say whether the labels
feeding retraining can be trusted.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .features import FeatureStore
from .models import Alert, Case, CaseStatus, Disposition
from .rules import RuleEngine


@dataclass
class FeedbackLoop:
    rule_engine: RuleEngine
    store: FeatureStore
    retrain_threshold: int = 50
    labels_since_retrain: int = 0
    # rule_id -> multiplier, mirrored onto the rule engine
    rule_weight_multipliers: dict[str, float] = field(default_factory=dict)

    def apply(self, case: Case, rule_ids: list[str],
              device_id: str, ip: str) -> dict:
        """Apply one resolved disposition. Returns a summary dict."""
        if case.disposition is None:
            raise ValueError("case has no disposition yet")
        summary: dict = {"case_id": case.case_id,
                         "disposition": case.disposition.value}
        is_fraud = case.disposition == Disposition.CONFIRMED_FRAUD
        is_fp = case.disposition == Disposition.FALSE_POSITIVE
        if is_fraud:
            self.rule_engine.blocked_devices.add(device_id)
            self.rule_engine.blocked_ips.add(ip)
            summary["blocklisted"] = [device_id, ip]
        self.store.record_outcome(device_id, ip, is_fraud=is_fraud)
        for rule_id in rule_ids:
            current = self.rule_weight_multipliers.get(rule_id, 1.0)
            if is_fraud:
                new = min(2.0, current * 1.10)
            elif is_fp:
                new = max(0.5, current * 0.90)
            else:
                new = current
            self.rule_weight_multipliers[rule_id] = round(new, 4)
            self.rule_engine.weight_multipliers[rule_id] = round(new, 4)
        summary["rule_weight_updates"] = {
            r: self.rule_weight_multipliers[r] for r in rule_ids
        }
        self.labels_since_retrain += 1
        summary["labels_since_retrain"] = self.labels_since_retrain
        summary["retrain_due"] = self.labels_since_retrain >= self.retrain_threshold
        return summary

    @property
    def retrain_due(self) -> bool:
        return self.labels_since_retrain >= self.retrain_threshold

    def mark_retrained(self) -> None:
        self.labels_since_retrain = 0


@dataclass
class LabelQuality:
    total_resolved: int = 0
    confirmed_fraud: int = 0
    false_positives: int = 0
    undetermined: int = 0
    disagreements: int = 0  # two analysts, same case, different dispositions

    @property
    def confirmation_rate(self) -> float:
        return self.confirmed_fraud / self.total_resolved if self.total_resolved else 0.0

    @property
    def disagreement_rate(self) -> float:
        return self.disagreements / self.total_resolved if self.total_resolved else 0.0

    def as_dict(self) -> dict:
        return {
            "total_resolved": self.total_resolved,
            "confirmed_fraud": self.confirmed_fraud,
            "false_positives": self.false_positives,
            "undetermined": self.undetermined,
            "confirmation_rate": round(self.confirmation_rate, 4),
            "disagreement_rate": round(self.disagreement_rate, 4),
        }


class CaseQueue:
    """Analyst work queue over alerts."""

    def __init__(self, feedback: FeedbackLoop) -> None:
        self.feedback = feedback
        self.cases: dict[str, Case] = {}
        self.quality = LabelQuality()
        self._seen_dispositions: dict[str, tuple[str, Disposition]] = {}

    def open_case(self, alert: Alert) -> Case:
        case_id = f"case-{alert.alert_id}"
        if case_id in self.cases:
            return self.cases[case_id]
        case = Case(
            case_id=case_id, alert_id=alert.alert_id, txn_id=alert.txn_id,
            account_id=alert.account_id, score=alert.score,
            created_ts=alert.created_ts,
        )
        self.cases[case_id] = case
        return case

    def assign(self, case_id: str, assignee: str) -> Case:
        case = self._get(case_id)
        case.assignee = assignee
        case.status = CaseStatus.ASSIGNED
        return case

    def resolve(self, case_id: str, disposition: Disposition,
                analyst: str, notes: str = "", resolved_ts: int = 0,
                device_id: str = "", ip: str = "",
                rule_ids: list[str] | None = None) -> dict:
        case = self._get(case_id)
        if case.status == CaseStatus.RESOLVED:
            raise ValueError(f"case {case_id} is already resolved")
        previous = self._seen_dispositions.get(case_id)
        if previous is not None and previous[1] != disposition:
            self.quality.disagreements += 1
        self._seen_dispositions[case_id] = (analyst, disposition)
        case.disposition = disposition
        case.analyst = analyst
        case.notes = notes
        case.status = CaseStatus.RESOLVED
        case.resolved_ts = resolved_ts
        self.quality.total_resolved += 1
        if disposition == Disposition.CONFIRMED_FRAUD:
            self.quality.confirmed_fraud += 1
        elif disposition == Disposition.FALSE_POSITIVE:
            self.quality.false_positives += 1
        else:
            self.quality.undetermined += 1
        feedback = self.feedback.apply(
            case, rule_ids or [], device_id, ip,
        )
        return {"case": case, "feedback": feedback}

    def open_cases(self) -> list[Case]:
        return [c for c in self.cases.values() if c.status != CaseStatus.RESOLVED]

    def _get(self, case_id: str) -> Case:
        case = self.cases.get(case_id)
        if case is None:
            raise KeyError(f"unknown case {case_id}")
        return case
