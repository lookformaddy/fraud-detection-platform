"""Tests for the from-scratch scorer and the synthetic generator."""
import pytest

from frauddet.evaluation import roc_auc
from frauddet.features import FEATURE_NAMES, FeatureStore
from frauddet.scorer import (
    FraudScorer, LogisticRegression, PlattScaler,
    generate_synthetic_transactions,
)


def test_generator_is_deterministic_and_labelled():
    txns_a, labels_a = generate_synthetic_transactions(n=500, seed=7)
    txns_b, labels_b = generate_synthetic_transactions(n=500, seed=7)
    assert labels_a == labels_b
    assert [t.txn_id for t in txns_a] == [t.txn_id for t in txns_b]
    assert 0 < sum(labels_a) < len(labels_a)  # both classes present
    assert all(t.timestamp >= 1_700_000_000 for t in txns_a)
    # Chronological order.
    assert all(txns_a[i].timestamp <= txns_a[i + 1].timestamp
               for i in range(len(txns_a) - 1))


def test_logistic_regression_separates_toy_data():
    # Two clearly separable clusters.
    X = [[0.0, 0.0]] * 20 + [[5.0, 5.0]] * 20
    y = [0] * 20 + [1] * 20
    model = LogisticRegression()
    model.fit(X, y, epochs=200)
    assert model.predict_proba([0.0, 0.0]) < 0.3
    assert model.predict_proba([5.0, 5.0]) > 0.7


def test_platt_scaler_is_monotone():
    scaler = PlattScaler()
    scaler.fit([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1])
    assert scaler.calibrate(0.1) < scaler.calibrate(0.5) < scaler.calibrate(0.9)


def test_scorer_trains_and_beats_chance(scorer, synthetic_data):
    txns, labels = synthetic_data
    assert scorer.trained
    assert scorer.train_size > 0
    store = FeatureStore()
    scores = []
    for txn, label in zip(txns, labels):
        feats = store.compute_features(txn)
        raw, calibrated, top = scorer.score(feats)
        assert 0.0 <= raw <= 1.0
        assert 0.0 <= calibrated <= 1.0
        assert len(top) == 5
        scores.append(calibrated)
        store.update(txn)
    auc = roc_auc(scores, labels)
    assert auc > 0.85, f"expected a strong synthetic signal, got AUC {auc}"


def test_scorer_explanations_reference_real_features(scorer):
    from tests.conftest import make_txn
    store = FeatureStore()
    txn = make_txn()
    feats = store.compute_features(txn)
    _raw, _cal, top = scorer.score(feats)
    assert all(name in FEATURE_NAMES for name, _ in top)
    assert all(isinstance(c, float) for _, c in top)


def test_untrained_scorer_refuses_to_score():
    scorer = FraudScorer()
    from frauddet.models import FeatureVector
    feats = FeatureVector(txn_id="x", values={n: 0.0 for n in FEATURE_NAMES})
    with pytest.raises(ValueError):
        scorer.score(feats)


def test_retraining_is_reproducible(synthetic_data):
    txns, labels = synthetic_data
    first = FraudScorer()
    first.train(txns, labels, seed=42)
    second = FraudScorer()
    second.train(txns, labels, seed=42)
    assert first.model.weights == second.model.weights
    assert first.platt.a == second.platt.a
