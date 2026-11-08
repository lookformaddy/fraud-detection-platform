"""Pytest configuration: put the src layout on the import path so plain
``pytest`` passes from the project root, with or without an install.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from frauddet.cases import CaseQueue, FeedbackLoop  # noqa: E402
from frauddet.features import FeatureStore  # noqa: E402
from frauddet.models import Transaction  # noqa: E402
from frauddet.rules import RuleEngine  # noqa: E402
from frauddet.scorer import FraudScorer, generate_synthetic_transactions  # noqa: E402
from frauddet.streaming import StreamProcessor  # noqa: E402

SEED = 42
BASE_TS = 1_700_000_000


def make_txn(txn_id="txn-000001", account_id="acct-0001",
             device_id="dev-0001-a", ip="10.0.0.1", amount=120.0,
             timestamp=BASE_TS + 3600, **kwargs) -> Transaction:
    return Transaction(txn_id=txn_id, account_id=account_id,
                       device_id=device_id, ip=ip, amount=amount,
                       timestamp=timestamp, **kwargs)


@pytest.fixture()
def store() -> FeatureStore:
    return FeatureStore()


@pytest.fixture()
def rules() -> RuleEngine:
    return RuleEngine()


@pytest.fixture()
def synthetic_data():
    return generate_synthetic_transactions(n=1200, seed=SEED)


@pytest.fixture()
def scorer(synthetic_data) -> FraudScorer:
    txns, labels = synthetic_data
    scorer = FraudScorer()
    scorer.train(txns, labels, seed=SEED)
    return scorer


@pytest.fixture()
def processor(scorer) -> StreamProcessor:
    return StreamProcessor(scorer)


@pytest.fixture()
def case_queue(processor) -> CaseQueue:
    feedback = FeedbackLoop(rule_engine=processor.rules,
                            store=processor.store)
    return CaseQueue(feedback)
