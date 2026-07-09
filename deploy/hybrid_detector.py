"""Load and run the hybrid LightGBM + Isolation Forest detector."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from deploy.features import FEATURE_NAMES, chunk_features
from deploy.iso_calibration import IsoCalibration, iso_bot_probability


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
        self.iso_calibration = artifact.get("iso_calibration")
        self.metadata = artifact.get("metadata", {})
        self.invert_scores = bool(artifact.get("invert_scores"))
        self.fusion_mode = str(artifact.get("fusion_mode", "max"))
        self.iso_blend_weight = float(artifact.get("iso_blend_weight", 0.0))
        self.hand_boost_weight = float(artifact.get("hand_boost_weight", 0.12))

    def _supervised_probability(self, features: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(features, columns=FEATURE_NAMES)
        probabilities = self.lgbm.predict_proba(frame)[:, 1]
        if self.invert_scores:
            probabilities = 1.0 - probabilities
        if self.calibrator is not None:
            probabilities = self.calibrator.predict(probabilities)
        return np.clip(probabilities, 0.0, 1.0)

    def _anomaly_probability(self, features: np.ndarray) -> np.ndarray:
        raw_scores = self.iso_forest.score_samples(features)
        if self.iso_calibration is not None:
            calibration = IsoCalibration.from_dict(self.iso_calibration)
            return iso_bot_probability(raw_scores, calibration)
        span = max(float(self.metadata.get("iso_span", 1.0)), 1e-8)
        normalized = (raw_scores - float(self.metadata.get("iso_min", 0.0))) / span
        return np.clip(1.0 - normalized, 0.0, 1.0)

    def _hand_heuristic_boost(self, chunks: list[list[dict]]) -> np.ndarray:
        boosts = []
        for chunk in chunks:
            if not chunk:
                boosts.append(0.0)
                continue
            from deploy.features import _heuristic_score

            scores = [_heuristic_score(hand) for hand in chunk]
            boosts.append(float(np.percentile(scores, 75)))
        return np.asarray(boosts, dtype=np.float64)

    def _fuse_scores(
        self,
        supervised: np.ndarray,
        anomaly: np.ndarray,
        *,
        hand_boost: np.ndarray | None = None,
    ) -> np.ndarray:
        if self.fusion_mode == "blend":
            fused = supervised + self.iso_blend_weight * anomaly * (1.0 - supervised)
        elif self.fusion_mode == "supervised":
            fused = supervised
        else:
            fused = np.maximum(supervised, anomaly)
        if hand_boost is not None and self.hand_boost_weight > 0:
            fused = np.clip(
                fused + self.hand_boost_weight * hand_boost * (1.0 - fused),
                0.0,
                1.0,
            )
        return fused

    def score_features(self, features: np.ndarray) -> np.ndarray:
        scaled = self.scaler.transform(features)
        supervised = self._supervised_probability(scaled)
        if self.fusion_mode == "supervised":
            return supervised
        anomaly = self._anomaly_probability(scaled)
        return self._fuse_scores(supervised, anomaly)

    def score_chunk(self, chunk: list[dict]) -> float:
        features = chunk_features(chunk, for_training=False).reshape(1, -1)
        supervised = self._supervised_probability(self.scaler.transform(features))
        if self.fusion_mode == "supervised":
            score = float(supervised[0])
        else:
            scaled = self.scaler.transform(features)
            anomaly = self._anomaly_probability(scaled)
            from deploy.features import _heuristic_score

            hand_boost = np.asarray(
                [float(np.percentile([_heuristic_score(h) for h in chunk], 75)) if chunk else 0.0]
            )
            score = float(
                self._fuse_scores(supervised, anomaly, hand_boost=hand_boost)[0]
            )
        return round(max(0.0, min(1.0, score)), 6)

    def score_chunks(self, chunks: list[list[dict]]) -> list[float]:
        if not chunks:
            return []
        features = np.vstack([chunk_features(chunk, for_training=False) for chunk in chunks])
        scaled = self.scaler.transform(features)
        supervised = self._supervised_probability(scaled)
        if self.fusion_mode == "supervised":
            scores = supervised
        else:
            anomaly = self._anomaly_probability(scaled)
            hand_boost = self._hand_heuristic_boost(chunks)
            scores = self._fuse_scores(supervised, anomaly, hand_boost=hand_boost)
        return [round(max(0.0, min(1.0, float(score))), 6) for score in scores]
