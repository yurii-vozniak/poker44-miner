"""Dual-model ensemble detector for live-eval robustness."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from deploy.features import chunk_features
from deploy.inference_postprocess import finalize_batch_scores, hand_heuristic_boost


class EnsembleDetector:
    def __init__(
        self,
        *,
        stacked,
        hybrid,
        stacked_weight: float = 0.55,
        hybrid_weight: float = 0.45,
        iso_weight: float = 0.20,
        hand_mix_weight: float = 0.0,
        hand_boost_weight: float = 0.10,
        rank_blend: float = 0.35,
        adaptive_rank: bool = True,
        max_pos_frac: float | None = None,
        adaptive_max_pos_frac: bool = True,
        metadata: dict[str, Any] | None = None,
        model_path: str | Path | None = None,
    ) -> None:
        self.stacked = stacked
        self.hybrid = hybrid
        total = stacked_weight + hybrid_weight
        self.stacked_weight = stacked_weight / total if total > 0 else 0.5
        self.hybrid_weight = hybrid_weight / total if total > 0 else 0.5
        self.iso_weight = float(iso_weight)
        self.hand_mix_weight = float(hand_mix_weight)
        self.hand_boost_weight = float(hand_boost_weight)
        self.rank_blend = float(rank_blend)
        self.adaptive_rank = bool(adaptive_rank)
        self.max_pos_frac = max_pos_frac
        self.adaptive_max_pos_frac = bool(adaptive_max_pos_frac)
        self.model_path = Path(model_path).resolve() if model_path else None
        self.metadata = dict(metadata or {})

    @classmethod
    def from_artifact(cls, artifact: dict[str, Any], *, model_path: str | Path) -> "EnsembleDetector":
        root = Path(model_path).resolve().parent

        def _resolve_model_path(raw: str | Path, default_name: str) -> Path:
            candidate = Path(raw or default_name)
            if candidate.is_absolute():
                return candidate
            by_name = (root / candidate.name).resolve()
            if by_name.is_file():
                return by_name
            nested = (root / candidate).resolve()
            if nested.is_file():
                return nested
            repo_root = root.parent
            repo_candidate = (repo_root / candidate).resolve()
            return repo_candidate

        stacked_path = _resolve_model_path(
            artifact.get("stacked_path") or "stacked.joblib",
            "stacked.joblib",
        )
        hybrid_path = _resolve_model_path(
            artifact.get("hybrid_path") or "hybrid.joblib",
            "hybrid.joblib",
        )

        from deploy.chunk_detector import load_chunk_detector

        stacked = load_chunk_detector(stacked_path)
        hybrid = load_chunk_detector(hybrid_path)
        return cls(
            stacked=stacked,
            hybrid=hybrid,
            stacked_weight=float(artifact.get("stacked_weight", 0.55)),
            hybrid_weight=float(artifact.get("hybrid_weight", 0.45)),
            iso_weight=float(artifact.get("iso_weight", 0.20)),
            hand_mix_weight=float(artifact.get("hand_mix_weight", 0.0)),
            hand_boost_weight=float(artifact.get("hand_boost_weight", 0.10)),
            rank_blend=float(artifact.get("rank_blend", 0.35)),
            adaptive_rank=bool(artifact.get("adaptive_rank", True)),
            max_pos_frac=artifact.get("max_pos_frac"),
            adaptive_max_pos_frac=bool(artifact.get("adaptive_max_pos_frac", True)),
            metadata=artifact.get("metadata"),
            model_path=model_path,
        )

    def _hybrid_supervised(self, features: np.ndarray) -> np.ndarray:
        scaled = self.hybrid.scaler.transform(features)
        return self.hybrid._supervised_probability(scaled)

    def _iso_signal(self, features: np.ndarray) -> np.ndarray:
        signals: list[np.ndarray] = []
        if self.stacked.iso_forest is not None:
            signals.append(self.stacked._anomaly_probability(features))
        scaled = self.hybrid.scaler.transform(features)
        if self.hybrid.iso_forest is not None:
            signals.append(self.hybrid._anomaly_probability(scaled))
        if not signals:
            return np.zeros(features.shape[0], dtype=np.float64)
        return np.maximum.reduce(signals)

    def _hand_mix_signal(self, chunks: list[list[dict]]) -> np.ndarray:
        if self.hand_mix_weight <= 0.0 or self.stacked.hand_lgbm is None:
            return np.zeros(len(chunks), dtype=np.float64)
        return self.stacked._hand_aggregate_for_chunks(chunks)

    def score_chunks(self, chunks: list[list[dict]]) -> list[float]:
        if not chunks:
            return []
        features = np.vstack([chunk_features(chunk, for_training=False) for chunk in chunks])
        stacked_scores = self.stacked.score_features(features)
        hybrid_scores = self._hybrid_supervised(features)
        fused = np.clip(
            self.stacked_weight * stacked_scores + self.hybrid_weight * hybrid_scores,
            0.0,
            1.0,
        )
        if self.iso_weight > 0.0:
            iso_scores = self._iso_signal(features)
            fused = np.clip(np.maximum(fused, self.iso_weight * iso_scores), 0.0, 1.0)
        heuristic = hand_heuristic_boost(chunks) if len(chunks) > 1 else None
        if heuristic is not None and float(np.std(fused)) < 0.12:
            fused = np.clip(np.maximum(fused, 0.28 * heuristic), 0.0, 1.0)
        if self.hand_mix_weight > 0.0:
            hand_scores = self._hand_mix_signal(chunks)
            fused = np.clip(np.maximum(fused, self.hand_mix_weight * hand_scores), 0.0, 1.0)
        if len(fused) > 1:
            fused = finalize_batch_scores(
                fused,
                chunks,
                hand_boost_weight=self.hand_boost_weight,
                rank_blend=self.rank_blend,
                adaptive_rank=self.adaptive_rank,
                max_pos_frac=self.max_pos_frac,
                adaptive_max_pos_frac=self.adaptive_max_pos_frac,
            )
        return [round(max(0.0, min(1.0, float(score))), 6) for score in fused]
