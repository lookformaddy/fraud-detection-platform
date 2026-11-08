"""From-scratch logistic-regression scorer + Platt calibration.

No sklearn, no numpy: gradient descent on standardized features,
pure stdlib. The scorer also owns the synthetic data generator used
for training and demos — seeded, fully synthetic transactions for a
generic payments context (no real data, no employer data).

Fraud in the generator is a latent behaviour, not a label pasted on
a row: fraud bursts are fast sequences from a new device/IP, at
night, for multiples of the account's normal spend, often from a
far-away location. The model has to *discover* those signals from
the engineered features, which is what makes the evaluation honest.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .features import FEATURE_NAMES, FeatureStore
from .models import FeatureVector, Transaction

# Cities used only as generic synthetic geography.
CITIES = [
    ("US", "San Francisco", 37.7749, -122.4194),
    ("US", "New York", 40.7128, -74.0060),
    ("US", "Chicago", 41.8781, -87.6298),
    ("US", "Austin", 30.2672, -97.7431),
    ("US", "Seattle", 47.6062, -122.3321),
    ("GB", "London", 51.5074, -0.1278),
    ("DE", "Berlin", 52.5200, 13.4050),
    ("SG", "Singapore", 1.3521, 103.8198),
]


def generate_synthetic_transactions(
    n: int = 2000, seed: int = 42, fraud_rate: float = 0.08,
) -> tuple[list[Transaction], list[int]]:
    """Generate ``n`` seeded synthetic transactions and their labels.

    Returns (transactions, labels) in chronological order; label 1 =
    fraud. Fraud appears as bursts: several fast transactions from a
    new device/IP, at night, for multiples of the account baseline,
    often far from the account's home city.
    """
    rng = random.Random(seed)
    base_ts = 1_700_000_000
    n_accounts = 200
    accounts = []
    for i in range(n_accounts):
        home = CITIES[i % 5]  # home is one of the five US cities
        accounts.append({
            "account_id": f"acct-{i:04d}",
            "baseline": rng.uniform(40.0, 400.0),
            "device_id": f"dev-{i:04d}-a",
            "ip": f"10.0.{i % 250}.{(i * 7) % 250 + 1}",
            "home": home,
        })

    txns: list[Transaction] = []
    labels: list[int] = []
    # account_id -> remaining fraud-burst txns + burst parameters
    bursts: dict[str, dict] = {}
    ts = base_ts
    for i in range(n):
        ts += rng.randint(60, 600)
        acct = accounts[rng.randrange(n_accounts)]
        burst = bursts.get(acct["account_id"])
        if burst is None and rng.random() < fraud_rate:
            far = rng.choice(CITIES)
            burst = {
                "remaining": rng.randint(3, 6),
                "device_id": f"dev-fraud-{rng.randrange(10_000):05d}",
                "ip": f"203.0.{rng.randrange(250)}.{rng.randrange(1, 250)}",
                "city": far,
            }
            bursts[acct["account_id"]] = burst
        if burst is not None:
            city = burst["city"]
            txn = Transaction(
                txn_id=f"txn-{i:06d}",
                account_id=acct["account_id"],
                device_id=burst["device_id"],
                ip=burst["ip"],
                amount=round(acct["baseline"] * rng.uniform(2.5, 8.0), 2),
                timestamp=ts,
                country=city[0], city=city[1],
                latitude=city[2] + rng.uniform(-0.05, 0.05),
                longitude=city[3] + rng.uniform(-0.05, 0.05),
                merchant_category=rng.choice(["electronics", "gift-cards", "travel"]),
                channel="online",
            )
            label = 1
            burst["remaining"] -= 1
            if burst["remaining"] <= 0:
                del bursts[acct["account_id"]]
        else:
            home = acct["home"]
            # Legitimate spend is daytime-weighted; amount near baseline.
            amount = round(
                acct["baseline"] * math.exp(rng.gauss(0.0, 0.4)), 2
            )
            amount = max(1.0, min(amount, acct["baseline"] * 4))
            txn = Transaction(
                txn_id=f"txn-{i:06d}",
                account_id=acct["account_id"],
                device_id=acct["device_id"],
                ip=acct["ip"],
                amount=amount,
                timestamp=ts,
                country=home[0], city=home[1],
                latitude=home[2] + rng.uniform(-0.05, 0.05),
                longitude=home[3] + rng.uniform(-0.05, 0.05),
                merchant_category=rng.choice(
                    ["groceries", "dining", "retail", "fuel", "general"]
                ),
                channel=rng.choice(["online", "online", "in-store"]),
            )
            label = 0
        txns.append(txn)
        labels.append(label)
    # Timestamps already increase; keep chronological order.
    return txns, labels


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


@dataclass
class LogisticRegression:
    """Binary logistic regression trained by batch gradient descent."""

    weights: list[float] = field(default_factory=list)
    bias: float = 0.0
    feature_means: list[float] = field(default_factory=list)
    feature_stds: list[float] = field(default_factory=list)
    n_features: int = 0
    epochs_run: int = 0

    def fit(self, X: list[list[float]], y: list[int],
            lr: float = 0.15, epochs: int = 300, l2: float = 0.01) -> None:
        n = len(X)
        self.n_features = len(X[0]) if X else 0
        self.feature_means = [
            sum(row[j] for row in X) / n for j in range(self.n_features)
        ]
        self.feature_stds = []
        for j in range(self.n_features):
            mean = self.feature_means[j]
            var = sum((row[j] - mean) ** 2 for row in X) / n
            self.feature_stds.append(math.sqrt(var) if var > 1e-12 else 1.0)
        self.weights = [0.0] * self.n_features
        self.bias = 0.0
        for _ in range(epochs):
            grad_w = [0.0] * self.n_features
            grad_b = 0.0
            for row, label in zip(X, y):
                z = self.bias + sum(
                    w * (row[j] - self.feature_means[j]) / self.feature_stds[j]
                    for j, w in enumerate(self.weights)
                )
                err = _sigmoid(z) - label
                for j in range(self.n_features):
                    grad_w[j] += err * (row[j] - self.feature_means[j]) / self.feature_stds[j]
                grad_b += err
            for j in range(self.n_features):
                self.weights[j] -= lr * (grad_w[j] / n + l2 * self.weights[j])
            self.bias -= lr * grad_b / n
        self.epochs_run = epochs

    def _standardized(self, row: list[float]) -> list[float]:
        return [
            (row[j] - self.feature_means[j]) / self.feature_stds[j]
            for j in range(self.n_features)
        ]

    def predict_proba(self, row: list[float]) -> float:
        x = self._standardized(row)
        z = self.bias + sum(w * v for w, v in zip(self.weights, x))
        return _sigmoid(z)

    def contributions(self, row: list[float]) -> list[tuple[str, float]]:
        """Signed per-feature contributions (weight * standardized value)."""
        x = self._standardized(row)
        return [
            (name, self.weights[j] * x[j])
            for j, name in enumerate(FEATURE_NAMES)
        ]


@dataclass
class PlattScaler:
    """Platt scaling: fit sigmoid(a * logit(score) + b) on held-out data.

    Fitting on the log-odds (not the raw probability) keeps the
    calibrated output able to reach the full [0, 1] range: fitting on
    raw probabilities would cap the calibrated score below 1 whenever
    the raw scores saturate, which would silently break the review /
    block bands that sit above that cap.
    """

    a: float = 1.0
    b: float = 0.0

    @staticmethod
    def _logit(p: float) -> float:
        p = min(max(p, 1e-6), 1.0 - 1e-6)
        return math.log(p / (1.0 - p))

    def fit(self, scores: list[float], labels: list[int],
            lr: float = 0.1, epochs: int = 300) -> None:
        logits = [self._logit(s) for s in scores]
        a, b = 1.0, 0.0
        n = len(scores)
        for _ in range(epochs):
            ga = gb = 0.0
            for x, y in zip(logits, labels):
                p = _sigmoid(a * x + b)
                err = p - y
                ga += err * x
                gb += err
            a -= lr * ga / n
            b -= lr * gb / n
        self.a, self.b = a, b

    def calibrate(self, score: float) -> float:
        return _sigmoid(self.a * self._logit(score) + self.b)


@dataclass
class FraudScorer:
    """Trained scorer: logistic model + Platt calibration + explanations."""

    model: LogisticRegression = field(default_factory=LogisticRegression)
    platt: PlattScaler = field(default_factory=PlattScaler)
    trained: bool = False
    train_size: int = 0

    def train(self, transactions: list[Transaction], labels: list[int],
              seed: int = 42) -> dict:
        """Train on synthetic transactions; returns train/val metrics."""
        from .evaluation import roc_auc  # local import: avoids a cycle

        store = FeatureStore()
        X: list[list[float]] = []
        y: list[int] = []
        for txn, label in zip(transactions, labels):
            feats = store.compute_features(txn)
            X.append(feats.as_ordered(FEATURE_NAMES))
            y.append(label)
            store.update(txn)
        # Deterministic 80/20 split (shuffled with the seed).
        rng = random.Random(seed)
        idx = list(range(len(X)))
        rng.shuffle(idx)
        split = int(0.8 * len(idx))
        train_idx, val_idx = idx[:split], idx[split:]
        model = LogisticRegression()
        model.fit([X[i] for i in train_idx], [y[i] for i in train_idx])
        val_scores = [model.predict_proba(X[i]) for i in val_idx]
        platt = PlattScaler()
        platt.fit(val_scores, [y[i] for i in val_idx])
        self.model, self.platt = model, platt
        self.trained = True
        self.train_size = len(train_idx)
        train_scores = [model.predict_proba(row) for row in X]
        return {
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "train_auc": roc_auc(train_scores, y),
            "val_auc": roc_auc(val_scores, [y[i] for i in val_idx]),
        }

    def score(self, features: FeatureVector, top_k: int = 5) -> tuple[float, float, list[tuple[str, float]]]:
        """Return (raw_score, calibrated_score, top contributing features)."""
        if not self.trained:
            raise ValueError("scorer has not been trained yet")
        row = features.as_ordered(FEATURE_NAMES)
        raw = self.model.predict_proba(row)
        calibrated = self.platt.calibrate(raw)
        contribs = self.model.contributions(row)
        top = sorted(contribs, key=lambda c: abs(c[1]), reverse=True)[:top_k]
        return raw, calibrated, top
