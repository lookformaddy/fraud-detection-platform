"""Fraud detection platform (frauddet).

Real-time transaction fraud scoring for a generic payments context:
deterministic velocity features, a from-scratch rule engine plus a
from-scratch logistic-regression scorer, calibrated risk scores with
explainable top-contributing features, banded decisioning that fails
closed, streaming scoring with watermark / late-event handling, an
analyst case queue with a feedback loop, and champion/challenger
model promotion gates. Pure-stdlib core; FastAPI at the edge.

All data used here is synthetic and generated in-process. Nothing in
this package touches a real payments system or any employer's data.
"""
from __future__ import annotations

__version__ = "1.0.0"
