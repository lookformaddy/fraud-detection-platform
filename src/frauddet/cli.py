"""CLI for the fraud detection platform.

Usage (from the project root, with PYTHONPATH=src):

    python -m frauddet.cli train --n 2000 --seed 42
    python -m frauddet.cli score txn.json
    python -m frauddet.cli stream txns.jsonl
    python -m frauddet.cli demo
"""
from __future__ import annotations

import argparse
import json
import sys

from .models import Decision, Transaction
from .scorer import FraudScorer, generate_synthetic_transactions
from .streaming import StreamProcessor


def _train(n: int, seed: int) -> FraudScorer:
    txns, labels = generate_synthetic_transactions(n=n, seed=seed)
    scorer = FraudScorer()
    metrics = scorer.train(txns, labels, seed=seed)
    print(f"trained on {metrics['train_size']} txns "
          f"(val {metrics['val_size']}); "
          f"train AUC {metrics['train_auc']:.3f}, "
          f"val AUC {metrics['val_auc']:.3f}")
    return scorer


def _txn_from_dict(data: dict) -> Transaction:
    return Transaction(
        txn_id=data["txn_id"], account_id=data["account_id"],
        device_id=data["device_id"], ip=data["ip"],
        amount=float(data["amount"]), timestamp=int(data["timestamp"]),
        country=data.get("country", "US"), city=data.get("city", "San Francisco"),
        latitude=float(data.get("latitude", 37.7749)),
        longitude=float(data.get("longitude", -122.4194)),
        merchant_category=data.get("merchant_category", "general"),
        channel=data.get("channel", "online"),
        currency=data.get("currency", "USD"),
    )


def cmd_train(args: argparse.Namespace) -> int:
    _train(args.n, args.seed)
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    scorer = _train(2000, 42)
    processor = StreamProcessor(scorer)
    with open(args.file) as fh:
        data = json.load(fh)
    result = processor.process(_txn_from_dict(data))
    scored = result.scored
    print(json.dumps({
        "txn_id": scored.transaction.txn_id,
        "raw_score": round(scored.raw_score, 4),
        "calibrated_score": round(scored.calibrated_score, 4),
        "decision": scored.decision.value,
        "decision_reason": scored.decision_reason,
        "rule_hits": [h.rule_id for h in scored.rule_hits],
        "top_features": scored.top_features,
    }, indent=2))
    return 0


def cmd_stream(args: argparse.Namespace) -> int:
    scorer = _train(2000, 42)
    processor = StreamProcessor(scorer)
    counts = {d.value: 0 for d in Decision}
    with open(args.file) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            result = processor.process(_txn_from_dict(json.loads(line)))
            if result.scored is not None:
                counts[result.scored.decision.value] += 1
    print(json.dumps({"processed": processor.stats.processed,
                      "duplicates": processor.stats.duplicates,
                      "late": processor.stats.late,
                      "alerts_opened": processor.stats.alerts_opened,
                      "by_decision": counts}, indent=2))
    return 0


def cmd_demo(_args: argparse.Namespace) -> int:
    import importlib.util
    from pathlib import Path

    demo_path = Path(__file__).resolve().parents[2] / "scripts" / "demo.py"
    spec = importlib.util.spec_from_file_location("frauddet_demo", demo_path)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    module.main()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="frauddet",
                                     description="Fraud detection platform CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="train the scorer on synthetic data")
    p_train.add_argument("--n", type=int, default=2000)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.set_defaults(func=cmd_train)

    p_score = sub.add_parser("score", help="score one transaction JSON file")
    p_score.add_argument("file")
    p_score.set_defaults(func=cmd_score)

    p_stream = sub.add_parser("stream", help="score a JSONL stream of transactions")
    p_stream.add_argument("file")
    p_stream.set_defaults(func=cmd_stream)

    p_demo = sub.add_parser("demo", help="run the end-to-end demo")
    p_demo.set_defaults(func=cmd_demo)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
