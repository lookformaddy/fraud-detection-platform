"""Tests for the deterministic rule engine."""
from frauddet.features import FEATURE_NAMES
from frauddet.models import FeatureVector
from tests.conftest import BASE_TS, make_txn


def _features(**overrides):
    values = {name: 0.0 for name in FEATURE_NAMES}
    values.update(overrides)
    return FeatureVector(txn_id="txn-1", values=values)


def test_no_hits_on_quiet_transaction(rules):
    txn = make_txn()
    assert rules.evaluate(txn, _features()) == []


def test_blocklist_device_blocks(rules):
    rules.blocked_devices.add("dev-bad")
    txn = make_txn(device_id="dev-bad")
    hits = rules.evaluate(txn, _features())
    assert len(hits) == 1
    assert hits[0].rule_id == "BLOCKLIST_DEVICE"
    assert hits[0].blocking is True


def test_blocklist_ip_and_account_block(rules):
    rules.blocked_ips.add("1.2.3.4")
    rules.blocked_accounts.add("acct-0001")
    hits = rules.evaluate(make_txn(ip="1.2.3.4"), _features())
    assert [h.rule_id for h in hits] == ["BLOCKLIST_ACCOUNT", "BLOCKLIST_IP"]


def test_velocity_count_rule(rules):
    txn = make_txn()
    hits = rules.evaluate(txn, _features(txn_count_1h=6.0))
    assert any(h.rule_id == "VELOCITY_COUNT_1H" for h in hits)
    assert not any(h.rule_id == "VELOCITY_COUNT_1H"
                   for h in rules.evaluate(txn, _features(txn_count_1h=4.0)))


def test_impossible_travel_rule(rules):
    hits = rules.evaluate(make_txn(), _features(geo_velocity_kmh=2500.0))
    assert any(h.rule_id == "IMPOSSIBLE_TRAVEL" for h in hits)


def test_night_high_amount_rule(rules):
    txn = make_txn(amount=5000.0, timestamp=BASE_TS + 2 * 3600)
    hits = rules.evaluate(txn, _features(is_night=1.0))
    assert any(h.rule_id == "NIGHT_HIGH_AMOUNT" for h in hits)


def test_new_device_and_ip_rule(rules):
    txn = make_txn(amount=600.0)
    hits = rules.evaluate(
        txn, _features(new_device=1.0, new_ip=1.0))
    assert any(h.rule_id == "NEW_DEVICE_AND_IP" for h in hits)
    # Small amount does not trigger it.
    small = make_txn(amount=50.0)
    assert not any(h.rule_id == "NEW_DEVICE_AND_IP"
                   for h in rules.evaluate(small, _features(new_device=1.0, new_ip=1.0)))


def test_total_boost_excludes_blocking(rules):
    rules.blocked_devices.add("dev-bad")
    hits = rules.evaluate(make_txn(device_id="dev-bad"),
                          _features(txn_count_1h=6.0))
    # Blocking hit contributes 1.0 but total_boost counts only non-blocking.
    assert rules.total_boost(hits) == 0.25


def test_feedback_multiplier_scales_boost(rules):
    rules.weight_multipliers["VELOCITY_COUNT_1H"] = 2.0
    hits = rules.evaluate(make_txn(), _features(txn_count_1h=6.0))
    assert hits[0].score_boost == 0.5
