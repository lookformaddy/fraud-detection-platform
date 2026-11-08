"""Offline evaluation: precision/recall/F1, ROC-AUC, detection
delay, and alert precision by decision band.

All metrics are computed from (score, label) pairs or from scored
transactions — no model internals required, so the same code
evaluates a champion and a challenger identically.
"""
from __future__ import annotations

import math


def confusion_at_threshold(scores: list[float], labels: list[int],
                           threshold: float) -> dict:
    tp = fp = tn = fn = 0
    for s, y in zip(scores, labels):
        pred = 1 if s >= threshold else 0
        if pred == 1 and y == 1:
            tp += 1
        elif pred == 1 and y == 0:
            fp += 1
        elif pred == 0 and y == 0:
            tn += 1
        else:
            fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": threshold, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(precision, 4), "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def roc_auc(scores: list[float], labels: list[int]) -> float:
    """Rank-based ROC-AUC (Mann-Whitney U), pure stdlib."""
    pairs = sorted(zip(scores, labels), key=lambda p: p[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    # Average ranks for ties.
    ranks: dict[float, float] = {}
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based
        ranks[pairs[i][0]] = avg_rank
        i = j + 1
    rank_sum_pos = sum(ranks[s] for s, y in pairs if y == 1)
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return round(min(max(auc, 0.0), 1.0), 4)


def precision_recall_curve(scores: list[float], labels: list[int],
                           thresholds: list[float] | None = None) -> list[dict]:
    thresholds = thresholds or [round(0.05 * k, 2) for k in range(1, 20)]
    return [confusion_at_threshold(scores, labels, t) for t in thresholds]


def detection_delay(first_fraud_ts: dict[str, int],
                    first_alert_ts: dict[str, int]) -> dict:
    """Delay from the first fraudulent txn to the first alert, per account.

    Accounts with fraud but no alert are reported as missed; accounts
    with an alert are reported with their delay in seconds.
    """
    delays = []
    missed = []
    for account, fraud_ts in first_fraud_ts.items():
        alert_ts = first_alert_ts.get(account)
        if alert_ts is None:
            missed.append(account)
        else:
            delays.append(max(0, alert_ts - fraud_ts))
    delays.sort()
    median = delays[len(delays) // 2] if delays else None
    mean = round(sum(delays) / len(delays), 1) if delays else None
    return {
        "accounts_with_fraud": len(first_fraud_ts),
        "accounts_alerted": len(delays),
        "accounts_missed": len(missed),
        "missed_accounts": sorted(missed),
        "median_delay_seconds": median,
        "mean_delay_seconds": mean,
        "max_delay_seconds": delays[-1] if delays else None,
    }


def alert_precision_by_band(alerts: list[dict]) -> dict:
    """Per-band precision from alert records.

    Each record: {"band": "review"|"block"|..., "confirmed_fraud": bool}.
    """
    by_band: dict[str, dict[str, int]] = {}
    for alert in alerts:
        entry = by_band.setdefault(alert["band"], {"alerts": 0, "confirmed": 0})
        entry["alerts"] += 1
        if alert.get("confirmed_fraud"):
            entry["confirmed"] += 1
    return {
        band: {
            "alerts": v["alerts"],
            "confirmed_fraud": v["confirmed"],
            "precision": round(v["confirmed"] / v["alerts"], 4) if v["alerts"] else 0.0,
        }
        for band, v in sorted(by_band.items())
    }


def evaluate_scored(scored: list, labels: list[int],
                    review_threshold: float = 0.65) -> dict:
    """One-shot summary for a list of ScoredTransaction + labels."""
    scores = [s.calibrated_score for s in scored]
    return {
        "n": len(scores),
        "fraud_rate": round(sum(labels) / len(labels), 4) if labels else 0.0,
        "roc_auc": roc_auc(scores, labels),
        "at_review_threshold": confusion_at_threshold(
            scores, labels, review_threshold),
    }
