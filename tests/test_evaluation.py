"""Tests for evaluation metrics."""
from frauddet.evaluation import (
    alert_precision_by_band, confusion_at_threshold, detection_delay,
    evaluate_scored, precision_recall_curve, roc_auc,
)


def test_roc_auc_perfect_and_random():
    assert roc_auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert roc_auc([0.5, 0.5, 0.5, 0.5], [1, 1, 0, 0]) == 0.5
    assert roc_auc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0.0


def test_roc_auc_single_class_is_neutral():
    assert roc_auc([0.1, 0.9], [1, 1]) == 0.5


def test_confusion_at_threshold_counts():
    result = confusion_at_threshold([0.9, 0.7, 0.3, 0.1], [1, 0, 1, 0], 0.5)
    assert result == {"threshold": 0.5, "tp": 1, "fp": 1, "tn": 1, "fn": 1,
                      "precision": 0.5, "recall": 0.5, "f1": 0.5}


def test_precision_recall_curve_covers_thresholds():
    curve = precision_recall_curve([0.9, 0.1], [1, 0])
    assert len(curve) == 19
    assert curve[0]["threshold"] == 0.05
    assert curve[-1]["threshold"] == 0.95


def test_detection_delay_median_and_missed():
    result = detection_delay(
        {"acct-a": 1000, "acct-b": 2000, "acct-c": 3000},
        {"acct-a": 1060, "acct-b": 2300},
    )
    assert result["accounts_with_fraud"] == 3
    assert result["accounts_alerted"] == 2
    assert result["accounts_missed"] == 1
    assert result["missed_accounts"] == ["acct-c"]
    assert result["median_delay_seconds"] == 300
    assert result["mean_delay_seconds"] == 180.0


def test_detection_delay_no_alerts():
    result = detection_delay({"acct-a": 1000}, {})
    assert result["median_delay_seconds"] is None
    assert result["accounts_missed"] == 1


def test_alert_precision_by_band():
    alerts = [
        {"band": "review", "confirmed_fraud": True},
        {"band": "review", "confirmed_fraud": False},
        {"band": "block", "confirmed_fraud": True},
    ]
    result = alert_precision_by_band(alerts)
    assert result["review"]["precision"] == 0.5
    assert result["block"]["precision"] == 1.0
    assert result["review"]["alerts"] == 2


def test_evaluate_scored_summary(scorer, synthetic_data):
    from frauddet.features import FeatureStore
    txns, labels = synthetic_data
    store = FeatureStore()
    scored = []
    for txn in txns:
        feats = store.compute_features(txn)
        raw, calibrated, top = scorer.score(feats)
        from frauddet.decisioning import score_transaction
        scored.append(score_transaction(txn, feats, raw, calibrated, [], top))
        store.update(txn)
    summary = evaluate_scored(scored, labels)
    assert summary["n"] == len(txns)
    assert summary["roc_auc"] > 0.85
    assert summary["at_review_threshold"]["precision"] > 0.4
