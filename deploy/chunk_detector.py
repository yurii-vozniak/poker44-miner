"""Factory for loading chunk-level detector artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import joblib


class ChunkDetector(Protocol):
    metadata: dict[str, Any]

    def score_chunks(self, chunks: list[list[dict]]) -> list[float]: ...


def load_chunk_detector(model_path: str | Path) -> ChunkDetector:
    path = Path(model_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Model not found at {path}")

    artifact: dict[str, Any] = joblib.load(path)
    model_type = str(artifact.get("model_type") or "hybrid")
    if model_type == "stacked":
        from deploy.stacked_detector import StackedDetector

        return StackedDetector.from_artifact(artifact, model_path=path)

    from deploy.hybrid_detector import HybridDetector

    return HybridDetector(path)
