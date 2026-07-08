"""Load and run the trained baseline bot-detection model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from deploy.features import chunk_features


class BaselineDetector:
    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path).resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Baseline model not found at {self.model_path}. "
                "Run: poker44-miner train"
            )
        artifact: dict[str, Any] = joblib.load(self.model_path)
        self.pipeline = artifact["pipeline"]
        self.metadata = artifact.get("metadata", {})

    def score_chunk(self, chunk: list[dict]) -> float:
        features = chunk_features(chunk).reshape(1, -1)
        if hasattr(self.pipeline, "predict_proba"):
            probability = float(self.pipeline.predict_proba(features)[0, 1])
        else:
            probability = float(self.pipeline.predict(features)[0])
        if self.metadata.get("invert_scores"):
            probability = 1.0 - probability
        return round(max(0.0, min(1.0, probability)), 6)

    def score_chunks(self, chunks: list[list[dict]]) -> list[float]:
        return [self.score_chunk(chunk) for chunk in chunks]
