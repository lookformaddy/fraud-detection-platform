"""Tests for deterministic feature engineering."""
import math

import pytest

from frauddet.features import (
    FEATURE_NAMES, FeatureStore, FeatureStoreUnavailable, haversine_km,
)
from tests.conftest import BASE_TS, make_txn


def test_haversine_san_francisco_to_new_york():
    # ~4,130 km; tolerance is generous because this is a smoke check.
    dist = haversine_km(37.7749, -122.4194, 40.7128, -74.0060)
    assert 4000 < dist < 4300


def test_feature_names_stable_and_complete():
    assert len(FEATURE_NAMES) == 17
    assert len(set(FEATURE_NAMES)) == 17
    assert "amount_zscore" in FEATURE_NAMES
    assert "geo_velocity_kmh" in FEATURE_NAMES


def test_first_transaction_is_all_novel_and_neutral(store):
    txn = make_txn()
    feats = store.compute_features(txn)
    assert feats.values["new_device"] == 1.0
    assert feats.values["new_ip"] == 1.0
    assert feats.values["new_country"] == 1.0
    assert feats.values["txn_count_1h"] == 0.0
    assert feats.values["amount_zscore"] == 0.0  # no baseline yet
    assert feats.values["geo_velocity_kmh"] == 0.0
    assert feats.values["device_fraud_rate"] == 0.0


def test_velocity_counts_exclude_current_transaction(store):
    for i in range(3):
        txn = make_txn(txn_id=f"txn-{i}", timestamp=BASE_TS + 3600 + i * 60)
        store.update(txn)
    probe = make_txn(txn_id="txn-probe", timestamp=BASE_TS + 3600 + 3 * 60)
    feats = store.compute_features(probe)
    assert feats.values["txn_count_1h"] == 3.0
    assert feats.values["txn_count_24h"] == 3.0
    assert feats.values["new_device"] == 0.0  # same device seen before


def test_amount_zscore_uses_account_baseline(store):
    for i in range(10):
        store.update(make_txn(txn_id=f"txn-{i}", amount=100.0,
                              timestamp=BASE_TS + i * 60))
    big = make_txn(txn_id="txn-big", amount=1000.0,
                   timestamp=BASE_TS + 3600)
    feats = store.compute_features(big)
    # Baseline is flat 100 with zero variance -> std is 0 -> z-score 0.
    assert feats.values["amount_zscore"] == 0.0
    assert feats.values["amount_vs_median_ratio"] == 10.0


def test_amount_zscore_with_variance(store):
    amounts = [80.0, 90.0, 100.0, 110.0, 120.0]
    for i, amount in enumerate(amounts):
        store.update(make_txn(txn_id=f"txn-{i}", amount=amount,
                              timestamp=BASE_TS + i * 60))
    probe = make_txn(txn_id="txn-probe", amount=200.0,
                     timestamp=BASE_TS + 3600)
    feats = store.compute_features(probe)
    mean = sum(amounts) / len(amounts)
    var = sum((a - mean) ** 2 for a in amounts) / len(amounts)
    expected = (200.0 - mean) / math.sqrt(var)
    assert feats.values["amount_zscore"] == pytest.approx(expected)


def test_geo_velocity_impossible_travel(store):
    first = make_txn(txn_id="txn-1", timestamp=BASE_TS,
                     latitude=37.7749, longitude=-122.4194)
    store.update(first)
    # New York one hour later: ~4,100 km/h implied.
    far = make_txn(txn_id="txn-2", timestamp=BASE_TS + 3600,
                   latitude=40.7128, longitude=-74.0060)
    feats = store.compute_features(far)
    assert feats.values["geo_velocity_kmh"] > 900.0


def test_device_reputation_from_feedback(store):
    store.record_outcome("dev-x", "1.2.3.4", is_fraud=True)
    store.record_outcome("dev-x", "1.2.3.4", is_fraud=False)
    txn = make_txn(device_id="dev-x", ip="1.2.3.4")
    feats = store.compute_features(txn)
    assert feats.values["device_fraud_rate"] == pytest.approx(0.5)
    assert feats.values["ip_fraud_rate"] == pytest.approx(0.5)


def test_store_outage_raises(store):
    store.available = False
    with pytest.raises(FeatureStoreUnavailable):
        store.compute_features(make_txn())


def test_night_flag():
    # 02:00 UTC -> night; 14:00 UTC -> day.
    store = FeatureStore()
    night = make_txn(timestamp=BASE_TS + 2 * 3600)
    day = make_txn(timestamp=BASE_TS + 14 * 3600)
    assert store.compute_features(night).values["is_night"] == 1.0
    assert store.compute_features(day).values["is_night"] == 0.0
