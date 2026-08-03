"""Micro-session v2 ensemble: reference-v2 + proxy-trained ML with fusion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from deploy.micro_session_features import FEATURE_NAMES, sessions_to_matrix
from deploy.train_micro_session_v2 import _fuse
from poker44.miner.config import MinerModelConfig
from poker44.miner.model import BotDetectionModel, ReferenceSessionModel


class MicroSessionEnsembleModel:
    def __init__(self, config: MinerModelConfig):
        self.config = config
        self.version = config.version
        self._artifact: dict[str, Any] | None = None
        self._reference = ReferenceSessionModel(config)
        self._artifact_path = Path(
            config.model_path or "models/micro_session_v2.joblib"
        )

    def load(self) -> None:
        if not self._artifact_path.exists():
            raise FileNotFoundError(
                f"Micro-session v2 artifact not found: {self._artifact_path}"
            )
        payload = joblib.load(self._artifact_path)
        if not isinstance(payload, dict) or "ml_model" not in payload:
            raise ValueError(f"Invalid v2 artifact: {self._artifact_path}")
        self._artifact = payload
        self._reference.load()

    def predict(self, sessions: list[dict[str, Any]]) -> list[float]:
        if self._artifact is None:
            raise RuntimeError("Model not loaded")
        if not sessions:
            return []

        ref_probs = np.asarray(self._reference.predict(sessions), dtype=np.float64)
        mode = str(self._artifact.get("fusion_mode") or "reference")
        weight = float(self._artifact.get("fusion_weight") or 1.0)

        if mode == "reference":
            probs = ref_probs
        else:
            matrix = sessions_to_matrix(sessions)
            if self._artifact.get("reference_feature"):
                matrix = np.hstack([matrix, ref_probs.reshape(-1, 1)])
            ml_probs = self._artifact["ml_model"].predict_proba(matrix)[:, 1]
            probs = _fuse(ref_probs, ml_probs, mode=mode, weight=weight)

        return [float(round(max(0.0, min(1.0, score)), 6)) for score in probs]


def create_model(config: MinerModelConfig) -> BotDetectionModel:
    return MicroSessionEnsembleModel(config)
