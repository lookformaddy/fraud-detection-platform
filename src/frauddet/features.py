"""Deterministic feature engineering.

Features are computed point-in-time from an in-memory
:class:`FeatureStore` that holds per-account / per-device / per-IP
history. The current transaction is *not* included in its own
velocity features (it is added to the store after scoring), so
features never leak the label they are scored against.

Feature families:
- velocity: txn counts / amounts in rolling 1h and 24h windows, per
  account, per device, per IP;
- reputation: observed fraud rate per device and per IP (from analyst
  feedback, seeded at 0.0 — cold start is explicit, not assumed);
- amount vs baseline: z-score and ratio against the account's running
  mean / median;
- time: night flag + cyclical hour encoding;
- geo-velocity: km/h implied by the distance from the previous
  transaction and the time since it (impossible-travel signal);
- novelty: first-seen device / IP / country for the account.
"""
from __future__ import annotations

import math
from collections import deque

from .models import FeatureVector, Transaction

FEATURE_NAMES: list[str] = [
    "amount_log",
    "amount_zscore",
    "amount_vs_median_ratio",
    "txn_count_1h",
    "txn_count_24h",
    "amount_sum_1h",
    "amount_sum_24h",
    "device_txn_count_24h",
    "ip_txn_count_24h",
    "device_fraud_rate",
    "ip_fraud_rate",
    "geo_velocity_kmh",
    "is_night",
    "new_device",
    "new_ip",
    "new_country",
    "time_since_last_log",
]

ONE_HOUR = 3600
ONE_DAY = 86400

# Above this implied speed, travel between two consecutive
# transactions is physically implausible for a single cardholder.
IMPOSSIBLE_TRAVEL_KMH = 900.0


class FeatureStoreUnavailable(Exception):
    """Raised when the feature store is down. Callers must fail closed."""


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return 2 * r * math.asin(math.sqrt(a))


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


class FeatureStore:
    """In-memory point-in-time feature state.

    ``available`` simulates a store outage: when False every read
    raises :class:`FeatureStoreUnavailable` and the decisioning layer
    must route to review instead of auto-approving.
    """

    def __init__(self) -> None:
        self.available = True
        # account_id -> deque of (ts, amount, lat, lon, device_id, ip, country)
        self._account_history: dict[str, deque] = {}
        # device_id / ip -> deque of timestamps
        self._device_history: dict[str, deque] = {}
        self._ip_history: dict[str, deque] = {}
        # device_id / ip -> (fraud_count, total_count) from analyst feedback
        self._device_reputation: dict[str, list[int]] = {}
        self._ip_reputation: dict[str, list[int]] = {}
        # account_id -> list of past amounts (for baseline stats)
        self._account_amounts: dict[str, list[float]] = {}
        # account_id -> set of seen devices / ips / countries
        self._account_devices: dict[str, set[str]] = {}
        self._account_ips: dict[str, set[str]] = {}
        self._account_countries: dict[str, set[str]] = {}

    # -- writes ---------------------------------------------------------
    def update(self, txn: Transaction) -> None:
        hist = self._account_history.setdefault(txn.account_id, deque())
        hist.append(
            (txn.timestamp, txn.amount, txn.latitude, txn.longitude,
             txn.device_id, txn.ip, txn.country)
        )
        self._device_history.setdefault(txn.device_id, deque()).append(txn.timestamp)
        self._ip_history.setdefault(txn.ip, deque()).append(txn.timestamp)
        self._account_amounts.setdefault(txn.account_id, []).append(txn.amount)
        self._account_devices.setdefault(txn.account_id, set()).add(txn.device_id)
        self._account_ips.setdefault(txn.account_id, set()).add(txn.ip)
        self._account_countries.setdefault(txn.account_id, set()).add(txn.country)

    def record_outcome(self, device_id: str, ip: str, is_fraud: bool) -> None:
        """Feed analyst-confirmed outcomes back into reputation."""
        for store, key in ((self._device_reputation, device_id),
                           (self._ip_reputation, ip)):
            entry = store.setdefault(key, [0, 0])
            entry[1] += 1
            if is_fraud:
                entry[0] += 1

    # -- reads ----------------------------------------------------------
    def _window(self, history: deque, now: int, window: int) -> list:
        # Account history rows are tuples (ts first); device/IP
        # histories store bare timestamps.
        cutoff = now - window
        return [
            row for row in history
            if (row[0] if isinstance(row, tuple) else row) > cutoff
        ]

    def compute_features(self, txn: Transaction) -> FeatureVector:
        if not self.available:
            raise FeatureStoreUnavailable("feature store is unavailable")
        now = txn.timestamp
        hist = self._account_history.get(txn.account_id, deque())
        amounts = self._account_amounts.get(txn.account_id, [])

        h1 = self._window(hist, now, ONE_HOUR)
        h24 = self._window(hist, now, ONE_DAY)
        dev24 = self._window(self._device_history.get(txn.device_id, deque()), now, ONE_DAY)
        ip24 = self._window(self._ip_history.get(txn.ip, deque()), now, ONE_DAY)

        # Amount vs account baseline.
        if len(amounts) >= 2:
            mean = sum(amounts) / len(amounts)
            var = sum((a - mean) ** 2 for a in amounts) / len(amounts)
            std = math.sqrt(var) if var > 0 else 0.0
            zscore = (txn.amount - mean) / std if std > 0 else 0.0
            median = _median(amounts)
            ratio = txn.amount / median if median > 0 else 1.0
        else:
            zscore, ratio = 0.0, 1.0

        # Geo-velocity vs the previous transaction on this account.
        geo_velocity = 0.0
        time_since_last: float | None = None
        if hist:
            prev_ts, _amt, prev_lat, prev_lon, *_rest = hist[-1]
            dt_hours = (now - prev_ts) / 3600.0
            time_since_last = float(now - prev_ts)
            if dt_hours > 0:
                dist = haversine_km(prev_lat, prev_lon, txn.latitude, txn.longitude)
                geo_velocity = dist / dt_hours

        def fraud_rate(store: dict[str, list[int]], key: str) -> float:
            entry = store.get(key)
            if not entry or entry[1] == 0:
                return 0.0
            return entry[0] / entry[1]

        hour = (now // 3600) % 24
        is_night = 1.0 if hour < 6 or hour >= 22 else 0.0

        seen_devices = self._account_devices.get(txn.account_id, set())
        seen_ips = self._account_ips.get(txn.account_id, set())
        seen_countries = self._account_countries.get(txn.account_id, set())
        values = {
            "amount_log": math.log1p(max(txn.amount, 0.0)),
            "amount_zscore": zscore,
            "amount_vs_median_ratio": ratio,
            "txn_count_1h": float(len(h1)),
            "txn_count_24h": float(len(h24)),
            "amount_sum_1h": float(sum(row[1] for row in h1)),
            "amount_sum_24h": float(sum(row[1] for row in h24)),
            "device_txn_count_24h": float(len(dev24)),
            "ip_txn_count_24h": float(len(ip24)),
            "device_fraud_rate": fraud_rate(self._device_reputation, txn.device_id),
            "ip_fraud_rate": fraud_rate(self._ip_reputation, txn.ip),
            "geo_velocity_kmh": geo_velocity,
            "is_night": is_night,
            "new_device": 0.0 if txn.device_id in seen_devices else 1.0,
            "new_ip": 0.0 if txn.ip in seen_ips else 1.0,
            "new_country": 0.0 if txn.country in seen_countries else 1.0,
            "time_since_last_log": (
                math.log1p(max(time_since_last, 0.0))
                if time_since_last is not None else 0.0
            ),
        }
        return FeatureVector(txn_id=txn.txn_id, values=values)
