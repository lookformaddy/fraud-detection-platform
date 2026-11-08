"""Core domain models: transactions, features, scores, decisions,
alerts, cases, and configuration.

Everything is a plain dataclass so the whole pipeline can be tested
without a server, a database, or a network.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field


class Decision(str, enum.Enum):
    APPROVE = "approve"
    STEP_UP = "step_up"
    REVIEW = "review"
    BLOCK = "block"


class AlertStatus(str, enum.Enum):
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


class CaseStatus(str, enum.Enum):
    OPEN = "open"
    ASSIGNED = "assigned"
    RESOLVED = "resolved"


class Disposition(str, enum.Enum):
    CONFIRMED_FRAUD = "confirmed_fraud"
    FALSE_POSITIVE = "false_positive"
    UNDETERMINED = "undetermined"


@dataclass(frozen=True)
class Transaction:
    """One payment transaction. All fields are synthetic / generic."""

    txn_id: str
    account_id: str
    device_id: str
    ip: str
    amount: float
    timestamp: int  # epoch seconds
    country: str = "US"
    city: str = "San Francisco"
    latitude: float = 37.7749
    longitude: float = -122.4194
    merchant_category: str = "general"
    channel: str = "online"
    currency: str = "USD"


@dataclass
class FeatureVector:
    """Deterministic, point-in-time features for one transaction.

    ``values`` is keyed by the fixed feature names in
    :data:`frauddet.features.FEATURE_NAMES`; the model consumes them
    in that order.
    """

    txn_id: str
    values: dict[str, float] = field(default_factory=dict)

    def as_ordered(self, order: list[str]) -> list[float]:
        return [self.values.get(name, 0.0) for name in order]


@dataclass
class RuleHit:
    rule_id: str
    message: str
    score_boost: float  # additive boost to the calibrated score
    blocking: bool = False
    severity: str = "medium"  # low | medium | high | critical


@dataclass
class ScoredTransaction:
    transaction: Transaction
    features: FeatureVector
    raw_score: float  # logistic output before calibration
    calibrated_score: float  # Platt-calibrated probability of fraud
    rule_hits: list[RuleHit] = field(default_factory=list)
    decision: Decision = Decision.APPROVE
    decision_reason: str = ""
    top_features: list[tuple[str, float]] = field(default_factory=list)
    # top_features: (feature name, signed contribution) sorted by |contribution| desc


@dataclass
class Alert:
    alert_id: str
    txn_id: str
    account_id: str
    score: float
    band: Decision
    status: AlertStatus = AlertStatus.OPEN
    created_ts: int = 0
    rule_ids: list[str] = field(default_factory=list)


@dataclass
class Case:
    case_id: str
    alert_id: str
    txn_id: str
    account_id: str
    score: float
    status: CaseStatus = CaseStatus.OPEN
    assignee: str | None = None
    disposition: Disposition | None = None
    analyst: str | None = None
    notes: str = ""
    created_ts: int = 0
    resolved_ts: int | None = None


@dataclass
class Thresholds:
    """Decision bands on the calibrated score.

    step_up <= score < review -> step-up challenge (e.g. 3-D Secure);
    review <= score < block -> analyst review queue;
    score >= block (or any blocking rule) -> block.
    """

    step_up: float = 0.40
    review: float = 0.65
    block: float = 0.90

    def validate(self) -> None:
        if not 0.0 < self.step_up < self.review < self.block <= 1.0:
            raise ValueError(
                "thresholds must satisfy 0 < step_up < review < block <= 1"
            )
