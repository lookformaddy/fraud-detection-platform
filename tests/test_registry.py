"""Tests for the champion/challenger registry and promotion gates."""
from frauddet.registry import ModelRegistry, ModelVersion, PromotionGates


def _version(version_id, auc, precision):
    return ModelVersion(
        version_id=version_id, seed=42,
        metrics={"roc_auc": auc, "precision_at_review": precision,
                 "recall_at_review": 0.7, "f1_at_review": 0.6},
    )


def test_first_model_can_become_champion():
    registry = ModelRegistry()
    registry.register(_version("v1", 0.90, 0.7))
    registry.set_champion("v1")
    assert registry.champion_id == "v1"
    assert registry.champion().status == "champion"


def test_promotion_requires_beating_champion():
    registry = ModelRegistry()
    registry.register(_version("v1", 0.90, 0.7))
    registry.set_champion("v1")
    registry.register(_version("v2", 0.905, 0.7))  # +0.005 < 0.01 required
    report = registry.promote("v2", seed_aucs=[0.905, 0.902, 0.908])
    assert report["passed"] is False
    assert report["promoted"] is False
    assert report["checks"]["beats_champion"] is False
    assert registry.champion_id == "v1"


def test_promotion_passes_when_all_gates_clear():
    registry = ModelRegistry()
    registry.register(_version("v1", 0.88, 0.7))
    registry.set_champion("v1")
    registry.register(_version("v2", 0.92, 0.75))
    report = registry.promote("v2", seed_aucs=[0.92, 0.915, 0.925])
    assert report["passed"] is True
    assert report["promoted"] is True
    assert registry.champion_id == "v2"
    assert registry.versions["v1"].status == "retired"


def test_auc_floor_blocks_promotion():
    registry = ModelRegistry()
    registry.register(_version("v1", 0.85, 0.7))
    registry.set_champion("v1")
    registry.register(_version("v2", 0.79, 0.7))
    report = registry.promote("v2", seed_aucs=[0.79, 0.79, 0.79])
    assert report["checks"]["auc_floor"] is False
    assert report["promoted"] is False


def test_precision_floor_blocks_promotion():
    registry = ModelRegistry()
    registry.register(_version("v1", 0.85, 0.7))
    registry.set_champion("v1")
    registry.register(_version("v2", 0.90, 0.40))  # precision below 0.50
    report = registry.promote("v2", seed_aucs=[0.90, 0.90, 0.90])
    assert report["checks"]["precision_floor"] is False


def test_unstable_across_seeds_blocks_promotion():
    registry = ModelRegistry()
    registry.register(_version("v1", 0.85, 0.7))
    registry.set_champion("v1")
    registry.register(_version("v2", 0.92, 0.75))
    # One seed dips below the floor and std is large.
    report = registry.promote("v2", seed_aucs=[0.92, 0.78, 0.93])
    assert report["checks"]["multi_seed_floor"] is False
    assert report["checks"]["multi_seed_stability"] is False
    assert report["promoted"] is False


def test_missing_seed_results_block_promotion():
    registry = ModelRegistry()
    registry.register(_version("v1", 0.85, 0.7))
    registry.set_champion("v1")
    registry.register(_version("v2", 0.92, 0.75))
    report = registry.promote("v2", seed_aucs=[])
    assert report["passed"] is False
    assert "no multi-seed results" in " ".join(report["reasons"])


def test_evaluate_version_metrics():
    scores = [0.9, 0.8, 0.2, 0.1]
    labels = [1, 1, 0, 0]
    metrics = ModelRegistry.evaluate_version(scores, labels)
    assert metrics["roc_auc"] == 1.0
    assert metrics["precision_at_review"] == 1.0
    assert metrics["recall_at_review"] == 1.0
