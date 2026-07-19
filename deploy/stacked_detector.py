"""Stacked ensemble detector for Poker44 chunk scoring."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from deploy.features import FEATURE_NAMES, HAND_KEYS, chunk_features, hand_features
from deploy.inference_postprocess import finalize_batch_scores
from deploy.iso_calibration import IsoCalibration, iso_bot_probability


class StackedDetector:
    def __init__(
        self,
        *,
        scaler,
        base_models: list[tuple[str, Any]],
        meta,
        calibrator=None,
        iso_forest=None,
        iso_calibration: dict[str, Any] | None = None,
        hand_lgbm=None,
        hand_calibrator=None,
        hand_aggregate_mode: str = "p90",
        hand_mix_weight: float = 0.0,
        hand_boost_weight: float = 0.0,
        rank_blend: float | None = None,
        adaptive_rank: bool = True,
        iso_blend_weight: float = 0.0,
        fusion_mode: str = "max",
        metadata: dict[str, Any] | None = None,
        model_path: str | Path | None = None,
    ) -> None:
        self.model_path = Path(model_path).resolve() if model_path else None
        self.scaler = scaler
        self.base_models = list(base_models)
        self.meta = meta
        self.calibrator = calibrator
        self.iso_forest = iso_forest
        self.iso_calibration = iso_calibration
        self.hand_lgbm = hand_lgbm
        self.hand_calibrator = hand_calibrator
        self.hand_aggregate_mode = hand_aggregate_mode
        self.hand_mix_weight = float(hand_mix_weight)
        self.hand_boost_weight = float(hand_boost_weight)
        self.rank_blend = rank_blend
        self.adaptive_rank = bool(adaptive_rank)
        self.iso_blend_weight = float(iso_blend_weight)
        self.fusion_mode = fusion_mode
        self.metadata = dict(metadata or {})

    @classmethod
    def from_artifact(cls, artifact: dict[str, Any], *, model_path: str | Path) -> "StackedDetector":
        return cls(
            scaler=artifact["scaler"],
            base_models=list(artifact["base_models"]),
            meta=artifact["meta"],
            calibrator=artifact.get("calibrator"),
            iso_forest=artifact.get("iso_forest"),
            iso_calibration=artifact.get("iso_calibration"),
            hand_lgbm=artifact.get("hand_lgbm"),
            hand_calibrator=artifact.get("hand_calibrator"),
            hand_aggregate_mode=str(artifact.get("hand_aggregate_mode", "p90")),
            hand_mix_weight=float(artifact.get("hand_mix_weight", 0.0)),
            hand_boost_weight=float(artifact.get("hand_boost_weight", 0.0)),
            rank_blend=artifact.get("rank_blend"),
            adaptive_rank=bool(artifact.get("adaptive_rank", True)),
            iso_blend_weight=float(artifact.get("iso_blend_weight", 0.0)),
            fusion_mode=str(artifact.get("fusion_mode", "max")),
            metadata=artifact.get("metadata"),
            model_path=model_path,
        )

    def _base_matrix(self, frame: pd.DataFrame) -> np.ndarray:
        columns: list[np.ndarray] = []
        for name, model in self.base_models:
            columns.append(model.predict_proba(frame)[:, 1])
        return np.column_stack(columns)

    def _supervised_probability(self, features: np.ndarray) -> np.ndarray:
        scaled = self.scaler.transform(features)
        frame = pd.DataFrame(scaled, columns=FEATURE_NAMES)
        meta_input = self._base_matrix(frame)
        scores = self.meta.predict_proba(meta_input)[:, 1]
        if self.calibrator is not None:
            scores = np.clip(self.calibrator.predict(scores), 0.0, 1.0)
        return np.clip(scores, 0.0, 1.0)

    def _anomaly_probability(self, features: np.ndarray) -> np.ndarray:
        if self.iso_forest is None:
            return np.zeros(features.shape[0], dtype=np.float64)
        scaled = self.scaler.transform(features)
        raw_scores = self.iso_forest.score_samples(scaled)
        if self.iso_calibration is not None:
            payload = self.iso_calibration
            if hasattr(payload, "to_dict"):
                payload = payload.to_dict()
            calibration = IsoCalibration.from_dict(payload)
            return iso_bot_probability(raw_scores, calibration)
        metadata = self.metadata or {}
        span = max(float(metadata.get("iso_span", 1.0)), 1e-8)
        normalized = (raw_scores - float(metadata.get("iso_min", 0.0))) / span
        return np.clip(1.0 - normalized, 0.0, 1.0)

    def _hand_aggregate_for_chunks(self, chunks: list[list[dict]]) -> np.ndarray:
        if self.hand_lgbm is None or self.hand_mix_weight <= 0.0:
            return np.zeros(len(chunks), dtype=np.float64)

        rows: list[np.ndarray] = []
        chunk_sizes: list[int] = []
        for chunk in chunks:
            chunk_sizes.append(len(chunk or []))
            for hand in chunk or []:
                rows.append(hand_features(hand, for_training=False))

        if not rows:
            return np.zeros(len(chunks), dtype=np.float64)

        frame = pd.DataFrame(np.vstack(rows), columns=HAND_KEYS)
        probs = self.hand_lgbm.predict_proba(frame)[:, 1]
        if self.hand_calibrator is not None:
            probs = np.clip(self.hand_calibrator.predict(probs), 0.0, 1.0)

        aggregated: list[float] = []
        offset = 0
        for hand_count in chunk_sizes:
            if hand_count <= 0:
                aggregated.append(0.0)
                continue
            chunk_probs = probs[offset : offset + hand_count]
            offset += hand_count
            if self.hand_aggregate_mode == "max":
                aggregated.append(float(np.max(chunk_probs)))
            elif self.hand_aggregate_mode == "p75":
                aggregated.append(float(np.percentile(chunk_probs, 75)))
            else:
                aggregated.append(float(np.percentile(chunk_probs, 90)))
        return np.asarray(aggregated, dtype=np.float64)

    def _fuse_scores(self, supervised: np.ndarray, anomaly: np.ndarray) -> np.ndarray:
        if self.iso_forest is None or self.fusion_mode == "supervised":
            return supervised
        if self.fusion_mode == "blend":
            return np.clip(
                supervised + self.iso_blend_weight * anomaly * (1.0 - supervised),
                0.0,
                1.0,
            )
        return np.maximum(supervised, anomaly)

    def score_features(self, features: np.ndarray) -> np.ndarray:
        supervised = self._supervised_probability(features)
        if self.iso_forest is None:
            return supervised
        anomaly = self._anomaly_probability(features)
        return self._fuse_scores(supervised, anomaly)

    def score_chunk(self, chunk: list[dict]) -> float:
        scores = self.score_chunks([chunk])
        return scores[0] if scores else 0.0

    def score_chunks(self, chunks: list[list[dict]]) -> list[float]:
        if not chunks:
            return []
        features = np.vstack([chunk_features(chunk, for_training=False) for chunk in chunks])
        scores = self.score_features(features)
        if self.hand_mix_weight > 0.0:
            hand_scores = self._hand_aggregate_for_chunks(chunks)
            scores = np.clip(
                np.maximum(scores, self.hand_mix_weight * hand_scores),
                0.0,
                1.0,
            )
        if len(scores) > 1:
            scores = finalize_batch_scores(
                scores,
                chunks,
                hand_boost_weight=self.hand_boost_weight,
                rank_blend=self.rank_blend,
                adaptive_rank=self.adaptive_rank,
            )
        return [round(max(0.0, min(1.0, float(score))), 6) for score in scores]
