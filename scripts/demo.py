"""End-to-end demo: train -> stream-score -> cases -> evaluate -> registry.

Deterministic (seed 42). Prints verifiable numbers: training set
size, validation AUC, test AUC, precision/recall/F1 at the review
threshold, alerts by decision band, detection delay, and a
champion/challenger promotion verdict.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from frauddet.cases import CaseQueue, FeedbackLoop  # noqa: E402
from frauddet.evaluation import (  # noqa: E402
    alert_precision_by_band, detection_delay, evaluate_scored,
)
from frauddet.models import Decision, Disposition  # noqa: E402
from frauddet.registry import ModelRegistry, ModelVersion  # noqa: E402
from frauddet.scorer import FraudScorer, generate_synthetic_transactions  # noqa: E402
from frauddet.streaming import StreamProcessor  # noqa: E402


def main() -> None:
    print("== fraud-detection-platform demo (seed 42, synthetic data) ==")

    # 1. Train on 2,000 seeded synthetic transactions.
    train_txns, train_labels = generate_synthetic_transactions(n=2000, seed=42)
    scorer = FraudScorer()
    train_metrics = scorer.train(train_txns, train_labels, seed=42)
    print(f"training set: {train_metrics['train_size']} txns "
          f"(val {train_metrics['val_size']}), "
          f"train AUC {train_metrics['train_auc']:.3f}, "
          f"val AUC {train_metrics['val_auc']:.3f}")

    # 2. Stream-score 500 fresh transactions (different seed).
    test_txns, test_labels = generate_synthetic_transactions(n=500, seed=123)
    processor = StreamProcessor(scorer)
    scored, labels = [], []
    first_fraud: dict[str, int] = {}
    first_alert: dict[str, int] = {}
    for txn, label in zip(test_txns, test_labels):
        if label == 1 and txn.account_id not in first_fraud:
            first_fraud[txn.account_id] = txn.timestamp
        result = processor.process(txn)
        scored.append(result.scored)
        labels.append(label)
        if result.alert is not None and txn.account_id not in first_alert:
            first_alert[txn.account_id] = txn.timestamp

    summary = evaluate_scored(scored, labels)
    print(f"test set: {summary['n']} txns, fraud rate {summary['fraud_rate']:.3f}, "
          f"test ROC-AUC {summary['roc_auc']:.3f}")
    cm = summary["at_review_threshold"]
    print(f"at review threshold 0.65: precision {cm['precision']:.3f}, "
          f"recall {cm['recall']:.3f}, F1 {cm['f1']:.3f} "
          f"(TP {cm['tp']} FP {cm['fp']} FN {cm['fn']})")
    print("decisions:", {k: v for k, v in processor.stats.by_decision.items() if v})
    print(f"alerts opened: {processor.stats.alerts_opened}, "
          f"duplicates: {processor.stats.duplicates}, "
          f"late events: {processor.stats.late}")

    delay = detection_delay(first_fraud, first_alert)
    print(f"detection delay: median {delay['median_delay_seconds']}s, "
          f"mean {delay['mean_delay_seconds']}s, "
          f"missed {delay['accounts_missed']}/{delay['accounts_with_fraud']} accounts")

    # 3. Analyst workflow: open cases for the first few alerts, dispose
    #    them against the known synthetic labels.
    feedback = FeedbackLoop(rule_engine=processor.rules, store=processor.store)
    queue = CaseQueue(feedback)
    label_by_txn = {t.txn_id: y for t, y in zip(test_txns, test_labels)}
    alert_records = []
    for alert in list(processor.alerts.values())[:10]:
        case = queue.open_case(alert)
        queue.assign(case.case_id, "analyst-1")
        truth = label_by_txn[alert.txn_id]
        disposition = (Disposition.CONFIRMED_FRAUD if truth == 1
                       else Disposition.FALSE_POSITIVE)
        txn = next(t for t in test_txns if t.txn_id == alert.txn_id)
        queue.resolve(case.case_id, disposition, "analyst-1",
                      device_id=txn.device_id, ip=txn.ip,
                      rule_ids=alert.rule_ids)
        alert_records.append({"band": alert.band.value,
                              "confirmed_fraud": truth == 1})
    print("label quality:", queue.quality.as_dict())
    print("alert precision by band:", alert_precision_by_band(alert_records))
    print(f"retrain due after feedback: {feedback.retrain_due} "
          f"({feedback.labels_since_retrain} labels)")

    # 4. Champion/challenger: register the demo model and a challenger
    #    trained on a different seed, then run the promotion gates.
    registry = ModelRegistry()
    registry.register(ModelVersion(
        version_id="champion-seed42", seed=42, metrics={
            "roc_auc": summary["roc_auc"],
            "precision_at_review": cm["precision"],
            "recall_at_review": cm["recall"], "f1_at_review": cm["f1"]}))
    registry.set_champion("champion-seed42")
    challenger = FraudScorer()
    challenger.train(train_txns, train_labels, seed=7)
    chal_scores = []
    from frauddet.features import FeatureStore
    store = FeatureStore()
    for txn in test_txns:
        feats = store.compute_features(txn)
        _raw, calibrated, _top = challenger.score(feats)
        chal_scores.append(calibrated)
        store.update(txn)
    chal_metrics = ModelRegistry.evaluate_version(chal_scores, test_labels)
    registry.register(ModelVersion(
        version_id="challenger-seed7", seed=7, metrics=chal_metrics))
    seed_aucs = []
    for seed in (7, 8, 9):
        s = FraudScorer()
        s.train(train_txns, train_labels, seed=seed)
        store = FeatureStore()
        scores = []
        for txn in test_txns:
            feats = store.compute_features(txn)
            _r, cal, _t = s.score(feats)
            scores.append(cal)
            store.update(txn)
        from frauddet.evaluation import roc_auc
        seed_aucs.append(roc_auc(scores, test_labels))
    report = registry.promote("challenger-seed7", seed_aucs)
    print(f"challenger seed AUCs: {[round(a, 3) for a in seed_aucs]}")
    print(f"promotion verdict: passed={report['passed']} "
          f"promoted={report['promoted']} reasons={report['reasons']}")
    print("demo complete.")


if __name__ == "__main__":
    main()
