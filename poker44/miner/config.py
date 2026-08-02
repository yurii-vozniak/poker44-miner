"""Environment-backed miner inference configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class MinerModelConfig:
    factory: str
    model_path: str | None
    device: str
    version: str
    max_sessions_per_request: int
    max_request_bytes: int = 16 * 1024 * 1024

    @classmethod
    def from_env(cls) -> "MinerModelConfig":
        max_sessions = int(os.getenv("POKER44_MAX_SESSIONS_PER_REQUEST", "256"))
        if max_sessions <= 0:
            raise ValueError("POKER44_MAX_SESSIONS_PER_REQUEST must be positive")
        max_request_bytes = int(
            os.getenv("POKER44_MAX_REQUEST_BYTES", str(16 * 1024 * 1024))
        )
        if max_request_bytes <= 0:
            raise ValueError("POKER44_MAX_REQUEST_BYTES must be positive")
        return cls(
            factory=os.getenv(
                "POKER44_MODEL_FACTORY",
                "poker44.miner.model:create_reference_model",
            ).strip(),
            model_path=os.getenv("POKER44_MODEL_PATH") or None,
            device=os.getenv("POKER44_MODEL_DEVICE", "cpu").strip() or "cpu",
            version=os.getenv("POKER44_MODEL_VERSION", "reference-v2").strip()
            or "unknown",
            max_sessions_per_request=max_sessions,
            max_request_bytes=max_request_bytes,
        )
