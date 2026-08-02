"""Poker44 v3 micro-session BotDetectionModel (schema v4.1)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from deploy.micro_session_features import sessions_to_matrix
from poker44.miner.config import MinerModelConfig
from poker44.miner.model import BotDetectionModel


class MicroSessionBotDetectionModel:
    def __init__(self, config: MinerModelConfig):
        self.config = config
        self.version = config.version
        self._model = None
        self._artifact_path = Path(
            config.model_path or "models/micro_session_v1.joblib"
        )

    def load(self) -> None:
        if not self._artifact_path.exists():
            raise FileNotFoundError(
                f"Micro-session model artifact not found: {self._artifact_path}"
            )
        payload = joblib.load(self._artifact_path)
        self._model = payload["model"] if isinstance(payload, dict) else payload

    def predict(self, sessions: list[dict[str, Any]]) -> list[float]:
        if self._model is None:
            raise RuntimeError("Model not loaded")
        if not sessions:
            return []
        matrix = sessions_to_matrix(sessions)
        probs = self._model.predict_proba(matrix)[:, 1]
        return [float(round(max(0.0, min(1.0, score)), 6)) for score in probs]


def create_model(config: MinerModelConfig) -> BotDetectionModel:
    return MicroSessionBotDetectionModel(config)
