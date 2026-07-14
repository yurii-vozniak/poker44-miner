"""Stacked ensemble detector for Poker44 chunk scoring."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from deploy.batch_calibration import apply_batch_calibration
from deploy.features import FEATURE_NAMES, chunk_features


class StackedDetector:
    def __init__(
        self,
        *,
        scaler,
        base_models: list[tuple[str, Any]],
        meta,
        calibrator=None,
        metadata: dict[str, Any] | None = None,
        model_path: str | Path | None = None,
    ) -> None:
        self.model_path = Path(model_path).resolve() if model_path else None
        self.scaler = scaler
        self.base_models = list(base_models)
        self.meta = meta
        self.calibrator = calibrator
        self.metadata = dict(metadata or {})

    @classmethod
    def from_artifact(cls, artifact: dict[str, Any], *, model_path: str | Path) -> "StackedDetector":
        return cls(
            scaler=artifact["scaler"],
            base_models=list(artifact["base_models"]),
            meta=artifact["meta"],
            calibrator=artifact.get("calibrator"),
            metadata=artifact.get("metadata"),
            model_path=model_path,
        )

    def _base_matrix(self, frame: pd.DataFrame) -> np.ndarray:
        columns: list[np.ndarray] = []
        for name, model in self.base_models:
            columns.append(model.predict_proba(frame)[:, 1])
        return np.column_stack(columns)

    def score_features(self, features: np.ndarray) -> np.ndarray:
        scaled = self.scaler.transform(features)
        frame = pd.DataFrame(scaled, columns=FEATURE_NAMES)
        meta_input = self._base_matrix(frame)
        scores = self.meta.predict_proba(meta_input)[:, 1]
        if self.calibrator is not None:
            scores = np.clip(self.calibrator.predict(scores), 0.0, 1.0)
        return np.clip(scores, 0.0, 1.0)

    def score_chunk(self, chunk: list[dict]) -> float:
        scores = self.score_chunks([chunk])
        return scores[0] if scores else 0.0

    def score_chunks(self, chunks: list[list[dict]]) -> list[float]:
        if not chunks:
            return []
        features = np.vstack([chunk_features(chunk, for_training=False) for chunk in chunks])
        scores = self.score_features(features)
        if len(scores) > 1:
            scores = apply_batch_calibration(scores)
        return [round(max(0.0, min(1.0, float(score))), 6) for score in scores]
