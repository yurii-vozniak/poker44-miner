"""BotDetectionModel implementation for Poker44 v3.0 (SessionDetectionSynapse).

Conforms to the miner-owned model protocol documented on the `dev` branch:

    class BotDetectionModel(Protocol):
        version: str
        def load(self) -> None: ...
        def predict(self, sessions: list[dict]) -> list[float]: ...

Wired up via POKER44_MODEL_FACTORY=deploy.session_model:create_model once the
v3.0 miner runtime (SessionDetectionSynapse, poker44.miner.* modules) is
vendored into this repo. See docs/v3_migration_plan.md for the remaining
steps once v3.0 ships.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from deploy.session_features import session_features

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "session_model_v1.joblib"


class SessionBotDetectionModel:
    """Scores subject-session v2 payloads using hand + telemetry features."""

    def __init__(self, model_path: str | Path | None = None, version: str | None = None):
        self._model_path = Path(model_path or os.getenv("POKER44_SESSION_MODEL_PATH") or DEFAULT_MODEL_PATH)
        self.version = version or os.getenv("POKER44_MODEL_VERSION", "session-v1-synthetic")
        self._bundle: dict[str, Any] | None = None

    def load(self) -> None:
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Session model artifact not found at {self._model_path}. "
                "Run `python -m deploy.train_session_model` first."
            )
        self._bundle = joblib.load(self._model_path)
        self.version = self._bundle.get("version", self.version)

    @staticmethod
    def _clamp01(value: float) -> float:
        if not np.isfinite(value):
            return 0.5
        return float(max(0.0, min(1.0, value)))

    def predict(self, sessions: list[dict[str, Any]]) -> list[float]:
        if self._bundle is None:
            self.load()
        assert self._bundle is not None
        model = self._bundle["model"]

        features = np.vstack([session_features(session) for session in sessions]) if sessions else np.zeros((0, 0))
        if features.shape[0] == 0:
            return []
        raw = model.predict_proba(features)[:, 1]
        return [self._clamp01(v) for v in raw]


def create_model(config: Any = None) -> SessionBotDetectionModel:
    """Factory entrypoint for POKER44_MODEL_FACTORY=deploy.session_model:create_model."""
    model_path = getattr(config, "model_path", None) if config is not None else None
    version = getattr(config, "version", None) if config is not None else None
    return SessionBotDetectionModel(model_path=model_path, version=version)
