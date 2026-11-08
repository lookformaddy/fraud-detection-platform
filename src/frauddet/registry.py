"""Champion/challenger model registry with promotion gates.

A challenger may replace the champion only if it clears *all*
gates — a single strong seed is never enough:

1. **Metric floor** — AUC and precision-at-review must clear absolute
   floors (a model that is merely better than a bad champion still
   does not ship).
2. **Must beat the champion** — challenger AUC must exceed the
   champion's AUC by at least ``min_auc_improvement``.
3. **Multi-seed stability** — the same training recipe run across
   several seeds must stay above the AUC floor on *every* seed and
   its AUC standard deviation must stay below ``max_auc_std``.

Promotion is a record, not a mutation of the model: the registry
keeps the full history of versions and their gate evaluations, so a
promotion decision is auditable after the fact.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .evaluation import confusion_at_threshold, roc_auc


@dataclass
class ModelVersion:
    version_id: str
    seed: int
    metrics: dict
    status: str = "candidate"  # candidate | champion | challenger | retired
    gate_report: dict | None = None


@dataclass
class PromotionGates:
    min_auc: float = 0.80
    min_precision_at_review: float = 0.50
    min_auc_improvement: float = 0.01
    max_auc_std: float = 0.02
    review_threshold: float = 0.65


class ModelRegistry:
    def __init__(self, gates: PromotionGates | None = None) -> None:
        self.gates = gates or PromotionGates()
        self.versions: dict[str, ModelVersion] = {}
        self.champion_id: str | None = None

    def register(self, version: ModelVersion) -> ModelVersion:
        self.versions[version.version_id] = version
        return version

    def champion(self) -> ModelVersion | None:
        return self.versions.get(self.champion_id) if self.champion_id else None

    def set_champion(self, version_id: str) -> ModelVersion:
        version = self.versions[version_id]
        if self.champion_id and self.champion_id in self.versions:
            self.versions[self.champion_id].status = "retired"
        version.status = "champion"
        self.champion_id = version_id
        return version

    @staticmethod
    def evaluate_version(scores: list[float], labels: list[int],
                         review_threshold: float = 0.65) -> dict:
        cm = confusion_at_threshold(scores, labels, review_threshold)
        return {
            "roc_auc": roc_auc(scores, labels),
            "precision_at_review": cm["precision"],
            "recall_at_review": cm["recall"],
            "f1_at_review": cm["f1"],
        }

    def evaluate_promotion(self, challenger_id: str,
                           seed_aucs: list[float]) -> dict:
        """Evaluate all promotion gates for a challenger.

        ``seed_aucs`` are the challenger's AUCs across the seeds the
        same recipe was trained on (multi-seed stability gate).
        """
        gates = self.gates
        challenger = self.versions[challenger_id]
        champion = self.champion()
        metrics = challenger.metrics
        checks: dict[str, bool] = {}
        reasons: list[str] = []

        checks["auc_floor"] = metrics["roc_auc"] >= gates.min_auc
        if not checks["auc_floor"]:
            reasons.append(
                f"AUC {metrics['roc_auc']:.3f} below floor {gates.min_auc}")
        checks["precision_floor"] = (
            metrics["precision_at_review"] >= gates.min_precision_at_review
        )
        if not checks["precision_floor"]:
            reasons.append(
                "precision at review "
                f"{metrics['precision_at_review']:.3f} below floor "
                f"{gates.min_precision_at_review}")
        if champion is not None:
            improvement = metrics["roc_auc"] - champion.metrics["roc_auc"]
            checks["beats_champion"] = improvement >= gates.min_auc_improvement
            if not checks["beats_champion"]:
                reasons.append(
                    f"AUC improvement {improvement:+.3f} below required "
                    f"{gates.min_auc_improvement}")
        else:
            checks["beats_champion"] = True  # no champion yet: any floor-passing model may lead
        if seed_aucs:
            mean_auc = sum(seed_aucs) / len(seed_aucs)
            var = sum((a - mean_auc) ** 2 for a in seed_aucs) / len(seed_aucs)
            std = math.sqrt(var)
            checks["multi_seed_floor"] = min(seed_aucs) >= gates.min_auc
            checks["multi_seed_stability"] = std <= gates.max_auc_std
            if not checks["multi_seed_floor"]:
                reasons.append(f"worst-seed AUC {min(seed_aucs):.3f} below floor")
            if not checks["multi_seed_stability"]:
                reasons.append(f"cross-seed AUC std {std:.3f} above {gates.max_auc_std}")
        else:
            checks["multi_seed_floor"] = False
            checks["multi_seed_stability"] = False
            reasons.append("no multi-seed results supplied")
        passed = all(checks.values())
        report = {
            "challenger": challenger_id,
            "champion": self.champion_id,
            "checks": checks,
            "passed": passed,
            "reasons": reasons,
        }
        challenger.gate_report = report
        return report

    def promote(self, challenger_id: str, seed_aucs: list[float]) -> dict:
        report = self.evaluate_promotion(challenger_id, seed_aucs)
        if report["passed"]:
            challenger = self.versions[challenger_id]
            challenger.status = "champion"
            if self.champion_id and self.champion_id in self.versions:
                self.versions[self.champion_id].status = "retired"
            self.champion_id = challenger_id
            report["promoted"] = True
        else:
            report["promoted"] = False
        return report
