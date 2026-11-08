"""Deterministic rule engine.

Rules are the fast, explainable first line of defence: blocklists,
velocity thresholds, and impossible travel. A rule either blocks
outright (blocklist hits) or adds a calibrated score boost. Rule
weights are adjustable by the case-feedback loop (see cases.py), so
analyst dispositions move the rules, not just the model.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .features import IMPOSSIBLE_TRAVEL_KMH, FeatureVector
from .models import RuleHit, Transaction


@dataclass
class RuleConfig:
    max_txn_count_1h: int = 5
    max_amount_sum_24h: float = 10_000.0
    max_device_txn_24h: int = 20
    max_ip_txn_24h: int = 30
    amount_zscore_limit: float = 4.0
    night_amount_limit: float = 2_000.0


@dataclass
class RuleEngine:
    config: RuleConfig = field(default_factory=RuleConfig)
    blocked_accounts: set[str] = field(default_factory=set)
    blocked_devices: set[str] = field(default_factory=set)
    blocked_ips: set[str] = field(default_factory=set)
    # Multipliers adjusted by analyst feedback; 1.0 = as configured.
    weight_multipliers: dict[str, float] = field(default_factory=dict)

    def _boost(self, rule_id: str, base: float) -> float:
        return base * self.weight_multipliers.get(rule_id, 1.0)

    def evaluate(self, txn: Transaction, features: FeatureVector) -> list[RuleHit]:
        v = features.values
        hits: list[RuleHit] = []

        if txn.account_id in self.blocked_accounts:
            hits.append(RuleHit("BLOCKLIST_ACCOUNT", "account is blocklisted",
                                1.0, blocking=True, severity="critical"))
        if txn.device_id in self.blocked_devices:
            hits.append(RuleHit("BLOCKLIST_DEVICE", "device is blocklisted",
                                1.0, blocking=True, severity="critical"))
        if txn.ip in self.blocked_ips:
            hits.append(RuleHit("BLOCKLIST_IP", "IP is blocklisted",
                                1.0, blocking=True, severity="critical"))

        if v["txn_count_1h"] >= self.config.max_txn_count_1h:
            hits.append(RuleHit(
                "VELOCITY_COUNT_1H",
                f"{int(v['txn_count_1h'])} txns in the last hour "
                f"(limit {self.config.max_txn_count_1h})",
                self._boost("VELOCITY_COUNT_1H", 0.25), severity="high"))
        if v["amount_sum_24h"] >= self.config.max_amount_sum_24h:
            hits.append(RuleHit(
                "VELOCITY_AMOUNT_24H",
                f"${v['amount_sum_24h']:,.0f} in the last 24h "
                f"(limit ${self.config.max_amount_sum_24h:,.0f})",
                self._boost("VELOCITY_AMOUNT_24H", 0.25), severity="high"))
        if v["device_txn_count_24h"] >= self.config.max_device_txn_24h:
            hits.append(RuleHit(
                "VELOCITY_DEVICE_24H",
                f"{int(v['device_txn_count_24h'])} txns from this device in 24h",
                self._boost("VELOCITY_DEVICE_24H", 0.20), severity="high"))
        if v["ip_txn_count_24h"] >= self.config.max_ip_txn_24h:
            hits.append(RuleHit(
                "VELOCITY_IP_24H",
                f"{int(v['ip_txn_count_24h'])} txns from this IP in 24h",
                self._boost("VELOCITY_IP_24H", 0.15), severity="medium"))
        if v["geo_velocity_kmh"] >= IMPOSSIBLE_TRAVEL_KMH:
            hits.append(RuleHit(
                "IMPOSSIBLE_TRAVEL",
                f"implied travel {v['geo_velocity_kmh']:,.0f} km/h since last txn",
                self._boost("IMPOSSIBLE_TRAVEL", 0.35), severity="high"))
        if v["amount_zscore"] >= self.config.amount_zscore_limit:
            hits.append(RuleHit(
                "AMOUNT_ZSCORE",
                f"amount z-score {v['amount_zscore']:.1f} vs account baseline",
                self._boost("AMOUNT_ZSCORE", 0.20), severity="medium"))
        if v["is_night"] == 1.0 and txn.amount >= self.config.night_amount_limit:
            hits.append(RuleHit(
                "NIGHT_HIGH_AMOUNT",
                f"${txn.amount:,.0f} transaction at night",
                self._boost("NIGHT_HIGH_AMOUNT", 0.10), severity="low"))
        if v["new_device"] == 1.0 and v["new_ip"] == 1.0 and txn.amount >= 500.0:
            hits.append(RuleHit(
                "NEW_DEVICE_AND_IP",
                "first transaction from a new device and a new IP",
                self._boost("NEW_DEVICE_AND_IP", 0.15), severity="medium"))
        return hits

    def total_boost(self, hits: list[RuleHit]) -> float:
        return sum(h.score_boost for h in hits if not h.blocking)
