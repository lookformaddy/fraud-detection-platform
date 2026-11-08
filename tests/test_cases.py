"""Tests for the case queue, feedback loop, and label quality."""
import pytest

from frauddet.models import Alert, CaseStatus, Decision, Disposition
from tests.conftest import BASE_TS


def _alert(alert_id="alert-txn-1"):
    return Alert(alert_id=alert_id, txn_id="txn-1", account_id="acct-0001",
                 score=0.8, band=Decision.REVIEW, created_ts=BASE_TS)


def test_case_open_assign_resolve(case_queue):
    case = case_queue.open_case(_alert())
    assert case.status == CaseStatus.OPEN
    case_queue.assign(case.case_id, "analyst-1")
    assert case.status == CaseStatus.ASSIGNED
    result = case_queue.resolve(
        case.case_id, Disposition.CONFIRMED_FRAUD, "analyst-1",
        device_id="dev-bad", ip="203.0.9.9", rule_ids=["VELOCITY_COUNT_1H"],
    )
    assert result["case"].status == CaseStatus.RESOLVED
    assert result["case"].disposition == Disposition.CONFIRMED_FRAUD
    assert case_queue.quality.confirmed_fraud == 1
    assert case_queue.quality.confirmation_rate == 1.0


def test_confirmed_fraud_blocklists_device_and_ip(case_queue, processor):
    case = case_queue.open_case(_alert())
    case_queue.resolve(case.case_id, Disposition.CONFIRMED_FRAUD, "analyst-1",
                       device_id="dev-bad", ip="203.0.9.9")
    assert "dev-bad" in processor.rules.blocked_devices
    assert "203.0.9.9" in processor.rules.blocked_ips


def test_false_positive_downweights_fired_rules(case_queue):
    case = case_queue.open_case(_alert())
    result = case_queue.resolve(
        case.case_id, Disposition.FALSE_POSITIVE, "analyst-1",
        device_id="dev-x", ip="10.0.0.9", rule_ids=["VELOCITY_COUNT_1H"])
    assert result["feedback"]["rule_weight_updates"]["VELOCITY_COUNT_1H"] < 1.0
    assert case_queue.quality.false_positives == 1


def test_confirmed_fraud_upweights_fired_rules(case_queue):
    case = case_queue.open_case(_alert())
    result = case_queue.resolve(
        case.case_id, Disposition.CONFIRMED_FRAUD, "analyst-1",
        device_id="dev-x", ip="10.0.0.9", rule_ids=["VELOCITY_COUNT_1H"])
    assert result["feedback"]["rule_weight_updates"]["VELOCITY_COUNT_1H"] > 1.0


def test_retrain_trigger_after_threshold_labels(case_queue):
    feedback = case_queue.feedback
    feedback.retrain_threshold = 2
    for i in range(2):
        case = case_queue.open_case(_alert(f"alert-txn-{i}"))
        case_queue.resolve(case.case_id, Disposition.FALSE_POSITIVE,
                           "analyst-1", device_id="dev-x", ip="10.0.0.9")
    assert feedback.retrain_due is True
    feedback.mark_retrained()
    assert feedback.retrain_due is False
    assert feedback.labels_since_retrain == 0


def test_label_disagreement_is_tracked(case_queue):
    case = case_queue.open_case(_alert())
    case_queue.resolve(case.case_id, Disposition.CONFIRMED_FRAUD, "analyst-1",
                       device_id="dev-x", ip="10.0.0.9")
    # A second analyst re-disposing the same case differently is a
    # disagreement. Reopen by hand to simulate a review pass.
    case.status = CaseStatus.ASSIGNED
    case.disposition = None
    case_queue.resolve(case.case_id, Disposition.FALSE_POSITIVE, "analyst-2",
                       device_id="dev-x", ip="10.0.0.9")
    assert case_queue.quality.disagreements == 1


def test_resolving_twice_is_rejected(case_queue):
    case = case_queue.open_case(_alert())
    case_queue.resolve(case.case_id, Disposition.UNDETERMINED, "analyst-1")
    with pytest.raises(ValueError):
        case_queue.resolve(case.case_id, Disposition.CONFIRMED_FRAUD, "analyst-1")


def test_open_case_is_idempotent(case_queue):
    first = case_queue.open_case(_alert())
    second = case_queue.open_case(_alert())
    assert first is second
    assert len(case_queue.cases) == 1
