# Architecture — Fraud Detection Platform

## Overview

A real-time transaction fraud scoring platform for a generic
payments context. The pipeline is a straight line with a feedback
loop closing it:

```
transaction ──▶ feature store (point-in-time velocity / reputation)
        │                │
        ▼                ▼
   rule engine ──▶ from-scratch logistic scorer (Platt-calibrated)
        │                │
        ▼                ▼
   decisioning (approve / step-up / review / block, fail-closed)
        │                │
        ▼                ▼
   streaming processor (watermark, dedup) ──▶ alerts ──▶ case queue
        │                                                  │
        ▼                                                  ▼
   evaluation (AUC, P/R, delay)            feedback loop (blocklists,
                                            reputation, rule weights,
                                            retrain trigger)
        │
        ▼
   champion/challenger registry (promotion gates)
```

All decision logic is pure stdlib (`src/frauddet/`); FastAPI
(`service.py`) and the CLI (`cli.py`) are thin edges over it, so the
service can never become a second source of truth.

## Components

### Feature store (`features.py`)
In-memory point-in-time state: per-account transaction history,
per-device and per-IP timestamp histories, running amount baselines,
seen-device/IP/country sets, and device/IP fraud rates fed by analyst
feedback. Features are computed **before** the current transaction
is added to the store, so velocity features never leak the event
being scored. Seventeen deterministic features:

- velocity: txn counts and amount sums in 1h/24h windows, per
  account, per device, per IP;
- reputation: observed fraud rate per device and per IP (0.0 at cold
  start — the absence of history is explicit, not assumed);
- amount vs baseline: z-score and ratio vs the account's running
  mean/median;
- time: night flag (22:00–06:00);
- geo-velocity: km/h implied by distance from the previous
  transaction divided by elapsed time (impossible travel > 900 km/h);
- novelty: first-seen device / IP / country for the account.

`FeatureStore.available = False` simulates an outage; every read
then raises `FeatureStoreUnavailable`, and decisioning routes to
review (fail closed).

### Rule engine (`rules.py`)
Deterministic, explainable rules: account/device/IP blocklists
(blocking), velocity thresholds, impossible travel, amount z-score,
night high-amount, and new-device-and-IP. Non-blocking rules add a
score boost; blocking rules decide outright. Rule weights carry a
multiplier that the feedback loop adjusts from analyst dispositions.

### Scorer (`scorer.py`)
Logistic regression trained by batch gradient descent on standardized
features — written from scratch, no sklearn/numpy — plus Platt
scaling fitted on a held-out split for calibration. `score()`
returns the raw score, the calibrated probability, and the top-5
signed feature contributions (weight × standardized value), so every
decision is explainable. Training data is generated in-process by
`generate_synthetic_transactions`: fraud is a latent behaviour
(bursts of fast transactions from a new device/IP, at night, for
multiples of the account baseline, often far from home), so the
model must discover the signal from the features.

### Decisioning (`decisioning.py`)
Score bands on the calibrated score (defaults: step-up 0.40, review
0.65, block 0.90), all configurable. Rule boosts are added before
banding (capped at 1.0). Two hard rules: any blocking rule hit means
block; a feature-store outage means review, never auto-approve.

### Streaming (`streaming.py`)
`StreamProcessor` holds the store, rules, and scorer as one
single-partition stream. The watermark is
`max_event_ts − allowed_lateness` (default 300s). Late events
(`timestamp < watermark`) are still scored — the cardholder's money
is at stake either way — but flagged, counted, and excluded from
feature-store state, so late data cannot rewrite history that
in-order decisions already used. Duplicate `txn_id`s are counted
and returned without re-scoring, so retries never double-alert.
REVIEW/BLOCK decisions open an alert (one per transaction);
alerts move OPEN → ACKNOWLEDGED → RESOLVED.

### Case management (`cases.py`)
`CaseQueue` turns alerts into analyst cases with assign/resolve and
three dispositions. Resolution feeds back three ways: confirmed
fraud blocklists the device/IP, updates device/IP reputation in the
feature store, and nudges the weights of the rules that fired
(confirmed fraud ×1.10 up to 2×; false positive ×0.90 down to 0.5×).
After `retrain_threshold` new labels the feedback loop flags a
retrain as due. `LabelQuality` tracks confirmation rate,
disagreement rate (two analysts, same case, different dispositions),
and undetermined share — the numbers that say whether the labels
feeding retraining can be trusted.

### Evaluation (`evaluation.py`)
Precision/recall/F1 at any threshold, rank-based ROC-AUC
(Mann-Whitney), precision/recall curves, detection delay (first
fraudulent transaction → first alert, per account, with missed
accounts listed), and alert precision by decision band.

### Registry (`registry.py`)
Champion/challenger registry. A challenger promotes only if it
clears **all** gates: AUC floor (0.80), precision-at-review floor
(0.50), must beat the champion's AUC by ≥ 0.01, and multi-seed
stability (worst-seed AUC ≥ floor and cross-seed AUC std ≤ 0.02).
Promotion is recorded with its gate report, so the decision is
auditable.

## Data flow (one transaction)

1. `StreamProcessor.process(txn)` — dedup check, watermark check.
2. `FeatureStore.compute_features(txn)` — point-in-time features.
3. `FraudScorer.score(features)` — raw + calibrated score + top features.
4. `RuleEngine.evaluate(txn, features)` — rule hits + boosts.
5. `decisioning.score_transaction(...)` — boost, band, decision.
6. `FeatureStore.update(txn)` — only if not late.
7. REVIEW/BLOCK → alert; alert → case → disposition → feedback.

## Failure modes (deliberate)

| Failure | Behaviour |
| ------- | --------- |
| Feature store down | Score is 0, decision REVIEW, alert opened |
| Late event | Scored + flagged, state untouched |
| Duplicate event | Counted, not re-scored, no second alert |
| Blocking rule hit | BLOCK regardless of model score |
| Challenger fails a gate | Stays challenger; champion untouched |
| No labelled feedback yet | Reputation 0.0; rules run at configured weights |

## What is intentionally not here

No real payments data, no external feature store, no durable
database (state is in-memory; persistence is the explicit next
step), no real key-based signatures. The demo and tests run entirely
offline on seeded synthetic data.
