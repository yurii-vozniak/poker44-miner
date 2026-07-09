"""Piecewise Isolation Forest calibration for low-FPR bot scoring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import IsolationForest


@dataclass(frozen=True)
class IsoCalibration:
    p_min: float
    p1: float
    p_max: float

    def to_dict(self) -> dict[str, float]:
        return {"p_min": self.p_min, "p1": self.p1, "p_max": self.p_max}

    @classmethod
    def from_dict(cls, payload: dict) -> "IsoCalibration":
        return cls(
            p_min=float(payload["p_min"]),
            p1=float(payload["p1"]),
            p_max=float(payload["p_max"]),
        )


def fit_iso_calibration(
    iso_forest: IsolationForest,
    human_features: np.ndarray,
    *,
    low_percentile: float = 1.0,
) -> IsoCalibration:
    raw_scores = iso_forest.score_samples(human_features)
    return IsoCalibration(
        p_min=float(np.min(raw_scores)),
        p1=float(np.percentile(raw_scores, low_percentile)),
        p_max=float(np.max(raw_scores)),
    )


def iso_bot_probability(raw_scores: np.ndarray, calibration: IsoCalibration) -> np.ndarray:
    scores = np.asarray(raw_scores, dtype=np.float64)
    span_normal = max(0.001, calibration.p_max - calibration.p1)
    span_anom = max(0.001, calibration.p1 - calibration.p_min)
    below_p1 = 0.5 + (calibration.p1 - scores) / span_anom * 0.5
    above_p1 = 0.5 - (scores - calibration.p1) / span_normal * 0.5
    probabilities = np.where(scores >= calibration.p1, above_p1, below_p1)
    return np.clip(probabilities, 0.0, 1.0)
