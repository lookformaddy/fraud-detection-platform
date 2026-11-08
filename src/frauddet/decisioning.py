"""Banded decisioning.

A scored transaction lands in exactly one band:

- ``approve`` — below the step-up threshold;
- ``step_up`` — challenge the cardholder (e.g. 3-D Secure);
- ``review`` — analyst queue;
- ``block`` — score at/above the block threshold, or any blocking rule.

Fail-closed rule: if the feature store is unavailable we cannot
trust the score, so the transaction goes to ``review`` — never to
auto-approve. Rule boosts are added to the calibrated score (capped
at 1.0) before banding, so a borderline model score plus strong
rules can push a transaction into review.
"""
from __future__ import annotations

from .models import Decision, RuleHit, ScoredTransaction, Thresholds


def decide(scored: ScoredTransaction,
           thresholds: Thresholds | None = None,
           feature_store_available: bool = True) -> Decision:
    thresholds = thresholds or Thresholds()
    thresholds.validate()
    if not feature_store_available:
        scored.decision_reason = "feature_store_unavailable: routed to review"
        return Decision.REVIEW
    if any(hit.blocking for hit in scored.rule_hits):
        blocking = [h.rule_id for h in scored.rule_hits if h.blocking]
        scored.decision_reason = f"blocking rule hit: {', '.join(blocking)}"
        return Decision.BLOCK
    score = scored.calibrated_score
    if score >= thresholds.block:
        scored.decision_reason = f"score {score:.3f} >= block {thresholds.block}"
        return Decision.BLOCK
    if score >= thresholds.review:
        scored.decision_reason = f"score {score:.3f} >= review {thresholds.review}"
        return Decision.REVIEW
    if score >= thresholds.step_up:
        scored.decision_reason = f"score {score:.3f} >= step-up {thresholds.step_up}"
        return Decision.STEP_UP
    scored.decision_reason = f"score {score:.3f} below step-up {thresholds.step_up}"
    return Decision.APPROVE


def apply_rule_boosts(scored: ScoredTransaction) -> ScoredTransaction:
    """Add non-blocking rule boosts to the calibrated score (cap 1.0)."""
    boost = sum(h.score_boost for h in scored.rule_hits if not h.blocking)
    scored.calibrated_score = min(1.0, scored.calibrated_score + boost)
    return scored


def score_transaction(transaction, features, raw, calibrated, rule_hits,
                      top_features, thresholds=None,
                      feature_store_available=True) -> ScoredTransaction:
    """Assemble a ScoredTransaction and run it through decisioning."""
    scored = ScoredTransaction(
        transaction=transaction, features=features,
        raw_score=raw, calibrated_score=calibrated,
        rule_hits=list(rule_hits), top_features=list(top_features),
    )
    apply_rule_boosts(scored)
    scored.decision = decide(scored, thresholds, feature_store_available)
    return scored
