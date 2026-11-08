"""Tests for streaming: ordering, watermark/late events, dedup,
alert lifecycle, and fail-closed outage handling."""
from frauddet.models import AlertStatus, Decision
from tests.conftest import BASE_TS, make_txn


def _fraud_like_txn(txn_id, ts, account_id="acct-0001"):
    # New device + new IP + far away + large amount: maximally suspicious.
    return make_txn(txn_id=txn_id, account_id=account_id,
                    device_id=f"dev-fraud-{txn_id}", ip="203.0.9.9",
                    amount=2500.0, timestamp=ts,
                    latitude=51.5074, longitude=-0.1278, city="London",
                    country="GB")


def test_in_order_processing_scores_and_counts(processor):
    result = processor.process(make_txn())
    assert result.scored is not None
    assert result.is_duplicate is False
    assert result.is_late is False
    assert processor.stats.processed == 1


def test_duplicate_transaction_is_not_rescored(processor):
    txn = make_txn()
    first = processor.process(txn)
    second = processor.process(txn)
    assert first.scored is not None
    assert second.is_duplicate is True
    assert second.scored is None
    assert processor.stats.duplicates == 1
    assert processor.stats.processed == 1


def test_late_event_flagged_and_not_state_updating(processor):
    processor.process(make_txn(txn_id="txn-1", timestamp=BASE_TS + 3600))
    processor.process(make_txn(txn_id="txn-2", timestamp=BASE_TS + 7200))
    # Watermark is now BASE_TS + 7200 - 300. An event at BASE_TS + 100
    # is late: scored, flagged, and excluded from feature state.
    late = make_txn(txn_id="txn-late", timestamp=BASE_TS + 100)
    result = processor.process(late)
    assert result.is_late is True
    assert processor.stats.late == 1
    probe = make_txn(txn_id="txn-probe", timestamp=BASE_TS + 7300)
    feats = processor.store.compute_features(probe)
    assert feats.values["txn_count_24h"] == 2.0  # late txn not counted


def test_high_risk_transaction_opens_alert(processor):
    # Seed some history so velocity features fire, then send a burst.
    for i in range(6):
        processor.process(make_txn(
            txn_id=f"txn-seed-{i}", timestamp=BASE_TS + 3600 + i * 30,
            amount=100.0))
    result = processor.process(_fraud_like_txn("txn-bad", BASE_TS + 4000))
    assert result.scored.decision in (Decision.REVIEW, Decision.BLOCK)
    assert result.alert is not None
    assert result.alert.status == AlertStatus.OPEN
    assert processor.stats.alerts_opened == 1


def test_alert_lifecycle_acknowledge_and_resolve(processor):
    for i in range(6):
        processor.process(make_txn(
            txn_id=f"txn-seed-{i}", timestamp=BASE_TS + 3600 + i * 30))
    result = processor.process(_fraud_like_txn("txn-bad", BASE_TS + 4000))
    alert = result.alert
    assert processor.acknowledge_alert(alert.alert_id).status == AlertStatus.ACKNOWLEDGED
    assert processor.resolve_alert(alert.alert_id).status == AlertStatus.RESOLVED
    assert processor.open_alerts() == []


def test_feature_store_outage_routes_to_review(processor):
    processor.store.available = False
    result = processor.process(make_txn())
    assert result.scored.decision == Decision.REVIEW
    assert processor.stats.feature_store_errors == 1
    assert result.alert is not None  # review opens an alert, never approves


def test_watermark_advances_with_events(processor):
    processor.process(make_txn(txn_id="txn-1", timestamp=BASE_TS + 3600))
    assert processor.watermark == BASE_TS + 3600 - 300
    processor.process(make_txn(txn_id="txn-2", timestamp=BASE_TS + 7200))
    assert processor.watermark == BASE_TS + 7200 - 300


def test_by_decision_counts_add_up(processor, synthetic_data):
    txns, _labels = synthetic_data
    for txn in txns[:100]:
        processor.process(txn)
    total = sum(processor.stats.by_decision.values())
    assert total == processor.stats.processed == 100
