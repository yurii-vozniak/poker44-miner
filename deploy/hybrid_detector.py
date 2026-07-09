"""Load and run the hybrid LightGBM + Isolation Forest detector."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from deploy.features import FEATURE_NAMES, chunk_features


class HybridDetector:
    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path).resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Hybrid model not found at {self.model_path}. "
                "Run: poker44-miner train-hybrid"
            )

        artifact: dict[str, Any] = joblib.load(self.model_path)
        self.scaler = artifact["scaler"]
        self.lgbm = artifact["lgbm"]
        self.iso_forest = artifact["iso_forest"]
        self.calibrator = artifact.get("calibrator")
        self.metadata = artifact.get("metadata", {})
        self.invert_scores = bool(artifact.get("invert_scores"))
        self.iso_min = float(artifact.get("iso_min", 0.0))
        self.iso_max = float(artifact.get("iso_max", 1.0))
        self.iso_blend_weight = float(artifact.get("iso_blend_weight", 0.0))
        self.fusion_mode = str(artifact.get("fusion_mode", "supervised"))

    def _normalize_iso_scores(self, raw_scores: np.ndarray) -> np.ndarray:
        span = max(self.iso_max - self.iso_min, 1e-8)
        normalized = (raw_scores - self.iso_min) / span
        return np.clip(normalized, 0.0, 1.0)

    def _supervised_probability(self, features: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(features, columns=FEATURE_NAMES)
        probabilities = self.lgbm.predict_proba(frame)[:, 1]
        if self.invert_scores:
            probabilities = 1.0 - probabilities
        if self.calibrator is not None:
            probabilities = self.calibrator.predict(probabilities)
        return np.clip(probabilities, 0.0, 1.0)

    def _anomaly_probability(self, features: np.ndarray) -> np.ndarray:
        raw = -self.iso_forest.score_samples(features)
        return self._normalize_iso_scores(raw)

    def _fuse_scores(self, supervised: np.ndarray, anomaly: np.ndarray) -> np.ndarray:
        if self.fusion_mode == "max":
            return np.maximum(supervised, anomaly)
        if self.fusion_mode == "blend":
            blended = supervised + self.iso_blend_weight * anomaly * (1.0 - supervised)
            return np.clip(blended, 0.0, 1.0)
        return supervised

    def score_features(self, features: np.ndarray) -> np.ndarray:
        scaled = self.scaler.transform(features)
        supervised = self._supervised_probability(scaled)
        if self.fusion_mode == "supervised":
            return supervised
        anomaly = self._anomaly_probability(scaled)
        return self._fuse_scores(supervised, anomaly)

    def score_chunk(self, chunk: list[dict]) -> float:
        features = chunk_features(chunk, for_training=False).reshape(1, -1)
        score = float(self.score_features(features)[0])
        return round(max(0.0, min(1.0, score)), 6)

    def score_chunks(self, chunks: list[list[dict]]) -> list[float]:
        if not chunks:
            return []
        features = np.vstack([chunk_features(chunk, for_training=False) for chunk in chunks])
        scores = self.score_features(features)
        return [round(max(0.0, min(1.0, float(score))), 6) for score in scores]
