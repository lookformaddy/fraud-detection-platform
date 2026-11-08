# Fraud Detection Platform

A **real-time transaction fraud scoring platform** for a generic
payments context: deterministic velocity features, a rule engine and
a from-scratch logistic-regression scorer combined into a calibrated,
explainable risk score, banded decisioning that fails closed,
streaming scoring with watermark and dedup, an analyst case queue
with a feedback loop, and champion/challenger promotion gates.

Fraud teams live with a tension: block too much and good customers
can't pay; block too little and losses pile up; and when a decision
is challenged — by a customer, an auditor, or a regulator — "the
model said so" is not an answer. This platform closes that gap with
deterministic, explainable machinery: every decision carries its
score, the rules that fired, and the top contributing features, and
every failure mode (store outage, late event, duplicate) has a
defined, fail-closed behaviour. All data is synthetic and generated
in-process; nothing here touches a real payments system.

## What it does

- **Feature engineering** — 17 deterministic point-in-time features:
  velocity (txn counts / amount sums in 1h and 24h windows per
  account, device, and IP), device/IP reputation from analyst
  feedback, amount z-score and ratio vs the account baseline,
  night flag, and geo-velocity (km/h implied by distance from the
  previous transaction — the impossible-travel signal). Features are
  computed *before* the current transaction enters the store, so
  nothing leaks.
- **Detectors from scratch (no sklearn)** — a rule engine
  (blocklists, velocity thresholds, impossible travel, amount
  z-score, night high-amount, new device + IP) plus a logistic
  regression scorer trained by gradient descent on seeded synthetic
  labelled data, Platt-calibrated, combined into one risk score with
  the top-5 contributing features per decision.
- **Decisioning** — score bands (approve / step-up challenge /
  review / block) with configurable thresholds. Fail-closed on a
  feature-store outage: the transaction goes to review, never to
  auto-approve. Blocking rules decide outright regardless of score.
- **Streaming scoring** — in-order event processing with a watermark
  (default 300s allowed lateness): late events are still scored but
  flagged and excluded from feature state; duplicate `txn_id`s are
  counted and never re-scored; REVIEW/BLOCK decisions open alerts
  with an OPEN → ACKNOWLEDGED → RESOLVED lifecycle.
- **Case management** — analyst queue with dispositions (confirmed
  fraud / false positive / undetermined). The feedback loop
  blocklists confirmed-fraud devices/IPs, updates device/IP
  reputation, nudges the weights of the rules that fired, and flags
  a retrain once enough new labels arrive. Label quality is tracked:
  confirmation rate, analyst disagreement rate, undetermined share.
- **Evaluation** — precision/recall/F1 at any threshold, ROC-AUC,
  detection delay (first fraudulent transaction → first alert, per
  account), and alert precision by decision band.
- **Champion/challenger registry** — promotion gates: absolute AUC
  and precision floors, must beat the champion's AUC, and
  multi-seed stability (worst-seed AUC above the floor, cross-seed
  std below 0.02). A single strong seed never ships a model.
- **FastAPI service + CLI + demo script**, deterministic core (no
  network, no keys).

## Architecture

See [docs/architecture.md](docs/architecture.md).

```
transaction ──▶ feature store (velocity / reputation / geo-velocity)
        │                              │
        ▼                              ▼
   rule engine ──▶ logistic scorer (from scratch, Platt-calibrated)
        │                              │
        ▼                              ▼
   decisioning (bands, fail-closed) ──▶ streaming (watermark, dedup)
        │                              │
        ▼                              ▼
   alerts ──▶ case queue ──▶ feedback (blocklists, reputation,
        │                    rule weights, retrain trigger)
        ▼                              │
   evaluation (AUC, P/R, delay)       ▼
        │                    champion/challenger registry (gates)
        ▼
   promotion decision (auditable gate report)
```

## How to run

```bash
pip install -r requirements.txt
make test          # pytest suite (core tests run on stdlib only)
make demo          # train -> stream-score -> cases -> evaluate -> registry
uvicorn frauddet.service:app --host 0.0.0.0 --port 8000   # with PYTHONPATH=src
```

CLI (trains on seeded synthetic data):

```bash
PYTHONPATH=src python -m frauddet.cli train --n 2000 --seed 42
PYTHONPATH=src python -m frauddet.cli score txn.json
PYTHONPATH=src python -m frauddet.cli stream txns.jsonl
```

## API surface

| Method | Endpoint | Purpose |
| ------ | -------- | ------- |
| `GET` | `/health` | Liveness + scorer/store/alert counts |
| `POST` | `/score` | Score one transaction (409 on duplicate) |
| `GET` | `/alerts`, `/alerts?status=open` | Alert inventory |
| `POST` | `/alerts/{id}/acknowledge` | Acknowledge an alert |
| `POST` | `/alerts/{id}/case` | Open an analyst case for an alert |
| `GET` | `/cases` | Case inventory |
| `POST` | `/cases/{id}/assign` | Assign a case to an analyst |
| `POST` | `/cases/{id}/resolve` | Resolve with a disposition (drives feedback) |
| `GET` | `/metrics` | Stream stats, label quality, retrain flag |
| `GET` | `/models` | Champion/challenger registry |
| `POST` | `/models/{id}/promote` | Run promotion gates for a challenger |

## Key design decisions and trade-offs

- **Fail-closed everywhere.** A store outage routes to review, a
  blocking rule blocks regardless of score, a challenger that fails
  any gate stays a challenger. A fraud system that fails open
  manufactures losses and false confidence in equal measure.
- **Features before the event enters state.** Velocity features
  exclude the transaction being scored, so the model cannot learn
  to recognise fraud from the fraud itself. The trade-off is one
  extra ordering rule in the stream processor that every caller
  must respect — enforced by the processor, not by convention.
- **Rules + model, not model alone.** Rules are instant, explainable,
  and adjustable by analysts without a retrain; the model catches
  the combinations no single rule names. The trade-off is two
  systems to maintain and a boost that must be calibrated so rules
  cannot silently swamp the model — boosts are capped and the
  calibrated score stays capped at 1.0.
- **Late events are scored, not dropped.** Money is at stake either
  way, so a late transaction still gets a decision; it just cannot
  rewrite the feature state that in-order decisions already used.
  The trade-off is that late decisions use slightly stale features,
  which is the honest price of ordering.
- **From-scratch model, pure stdlib.** Gradient descent and Platt
  scaling in ~150 lines means the whole scoring path is readable,
  testable, and auditable without a framework. The trade-off is
  speed and feature breadth — this is the right call for a
  reference implementation, and the registry gates are where a
  heavier model would have to prove itself before promotion.
- **Feedback moves rules immediately, the model on a schedule.**
  Blocklists and reputation update on the next transaction; rule
  weights nudge per disposition; the model retrains only after a
  label threshold. That separation keeps analyst feedback fast
  without retraining on noisy single labels.
- **Synthetic fraud as latent behaviour.** Fraud in the generator is
  a behaviour (bursts, new devices, night hours, far-away
  locations), not a label pasted on a row, so the evaluation
  measures whether the features and model actually discover the
  signal.

## What it demonstrates

Real-time fraud detection end to end: deterministic point-in-time
feature engineering (velocity, reputation, geo-velocity), a rule
engine plus a from-scratch calibrated logistic scorer with
per-decision explanations, banded decisioning with fail-closed
outage handling, watermarked streaming with dedup and alert
lifecycle, analyst case management with a feedback loop that moves
blocklists, reputation, and rule weights, offline evaluation
(ROC-AUC, precision/recall, detection delay, alert precision by
band), and champion/challenger promotion gates with multi-seed
stability — built as a typed, tested, enterprise-shaped Python
codebase with a FastAPI service, CLI, Docker image, and CI
definition.
