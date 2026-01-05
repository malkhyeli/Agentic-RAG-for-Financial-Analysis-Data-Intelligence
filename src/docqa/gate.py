from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol

from src.docqa.features import RetrievalFeatures


class AnswerabilityModel(Protocol):
    def predict_proba(self, features: RetrievalFeatures) -> float: ...


@dataclass(frozen=True)
class SklearnAnswerabilityModel:
    """Pickle-friendly wrapper around an sklearn classifier."""

    feature_names: list[str]
    classifier: Any

    def predict_proba(self, features: RetrievalFeatures) -> float:
        row = [[float(getattr(features, k, 0.0) or 0.0) for k in self.feature_names]]
        try:
            proba = self.classifier.predict_proba(row)
            return float(proba[0][1])
        except Exception:
            return 0.0


@dataclass(frozen=True)
class GateDecision:
    p_supported: float
    threshold: float
    supported: bool
    reason: str
    model: str


def load_answerability_model(model_path: Path = Path("models/answerability.pkl")) -> Optional[AnswerabilityModel]:
    # Phase 4: optional trainable model. Default to baseline gate when missing.
    if not model_path.exists():
        return None
    try:
        import pickle
    except Exception:
        return None
    try:
        with model_path.open("rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _normalize_score(s: float) -> float:
    if not math.isfinite(s):
        return 0.0
    if s <= 0:
        return 0.0
    # GroundX scores are typically 0..1, but be defensive in case it's 0..100.
    if s > 1.0 and s <= 100.0:
        return s / 100.0
    if s > 100.0:
        return 1.0
    return s


def baseline_gate(features: RetrievalFeatures, *, threshold: float) -> GateDecision:
    # Conservative baseline: turns retrieval signals into a probability-like score.
    top_score = _normalize_score(features.top_score)
    score_gap = _normalize_score(features.score_gap)

    num_quotes = max(0, int(features.num_quotes))
    total_chars = max(0, int(features.total_quote_chars))
    total_chars = min(total_chars, 2000)

    if num_quotes <= 0 or total_chars <= 0:
        return GateDecision(
            p_supported=0.0,
            threshold=threshold,
            supported=False,
            reason="no_evidence",
            model="baseline",
        )

    # Score ranges are uncertain; use a relatively steep slope with a negative bias
    # to reduce false accepts. Adjusted for better recall while maintaining precision.
    a = 4.0
    b = 2.0
    c = 0.45
    d = 0.002  # Increased from 0.0008 - more evidence chars = more confidence
    bias = -2.5  # Reduced from -3.5 to accept more borderline evidence

    z = a * top_score + b * score_gap + c * float(num_quotes) + d * float(total_chars) + bias
    p = float(_sigmoid(z))
    return GateDecision(
        p_supported=p,
        threshold=threshold,
        supported=(p >= threshold),
        reason="ok" if p >= threshold else "below_threshold",
        model="baseline",
    )


def decide_answerability(features: RetrievalFeatures, *, threshold: float) -> GateDecision:
    model = load_answerability_model()
    if model is not None:
        try:
            p = float(model.predict_proba(features))
        except Exception:
            p = 0.0
        return GateDecision(
            p_supported=p,
            threshold=threshold,
            supported=(p >= threshold),
            reason="ok" if p >= threshold else "below_threshold",
            model="trained",
        )
    return baseline_gate(features, threshold=threshold)
