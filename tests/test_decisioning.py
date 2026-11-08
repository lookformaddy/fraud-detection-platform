"""Tests for banded decisioning, including fail-closed behaviour."""
from frauddet.decisioning import apply_rule_boosts, decide, score_transaction
from frauddet.features import FEATURE_NAMES
from frauddet.models import (
    Decision, FeatureVector, RuleHit, ScoredTransaction, Thresholds,
)
from tests.conftest import make_txn


def _scored(score, rule_hits=None):
    return ScoredTransaction(
        transaction=make_txn(),
        features=FeatureVector(txn_id="txn-1",
                               values={n: 0.0 for n in FEATURE_NAMES}),
        raw_score=score, calibrated_score=score,
        rule_hits=rule_hits or [],
    )


def test_bands_follow_thresholds():
    assert decide(_scored(0.10)) == Decision.APPROVE
    assert decide(_scored(0.45)) == Decision.STEP_UP
    assert decide(_scored(0.70)) == Decision.REVIEW
    assert decide(_scored(0.95)) == Decision.BLOCK


def test_custom_thresholds_move_the_bands():
    thresholds = Thresholds(step_up=0.2, review=0.5, block=0.8)
    assert decide(_scored(0.30), thresholds) == Decision.STEP_UP
    assert decide(_scored(0.60), thresholds) == Decision.REVIEW
    assert decide(_scored(0.85), thresholds) == Decision.BLOCK


def test_invalid_thresholds_rejected():
    import pytest
    with pytest.raises(ValueError):
        decide(_scored(0.5), Thresholds(step_up=0.9, review=0.5, block=0.8))


def test_blocking_rule_overrides_low_score():
    hit = RuleHit("BLOCKLIST_DEVICE", "device blocklisted", 1.0, blocking=True)
    assert decide(_scored(0.01, [hit])) == Decision.BLOCK


def test_feature_store_outage_fails_closed_to_review():
    # Even a near-zero score must not auto-approve without features.
    scored = _scored(0.01)
    assert decide(scored, feature_store_available=False) == Decision.REVIEW
    assert "feature_store_unavailable" in scored.decision_reason


def test_rule_boosts_push_borderline_into_review():
    scored = _scored(0.60, [RuleHit("VELOCITY_COUNT_1H", "fast", 0.25)])
    apply_rule_boosts(scored)
    assert scored.calibrated_score == 0.85
    assert decide(scored) == Decision.REVIEW


def test_rule_boosts_capped_at_one():
    scored = _scored(0.95, [RuleHit("VELOCITY_COUNT_1H", "fast", 0.25)])
    apply_rule_boosts(scored)
    assert scored.calibrated_score == 1.0


def test_score_transaction_wires_decision_and_reason():
    scored = score_transaction(
        make_txn(), FeatureVector(txn_id="txn-1",
                                  values={n: 0.0 for n in FEATURE_NAMES}),
        0.5, 0.72, [], [("amount_zscore", 1.2)],
    )
    assert scored.decision == Decision.REVIEW
    assert scored.decision_reason
    assert scored.top_features == [("amount_zscore", 1.2)]
