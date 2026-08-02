"""Validation and execution around a miner-owned inference model."""

from __future__ import annotations

import asyncio
import inspect
import json
import math
from typing import Any

from poker44.miner.config import MinerModelConfig
from poker44.contracts import find_forbidden, validate_v4_micro_session
from poker44.miner.model import BotDetectionModel


class MinerInferenceService:
    def __init__(self, model: BotDetectionModel, config: MinerModelConfig):
        self.model = model
        self.config = config
        self._model_lock = asyncio.Lock()

    @staticmethod
    def _find_forbidden(value: Any, path: str = "session") -> list[str]:
        return find_forbidden(value, path)

    @staticmethod
    def _validate_micro_session(item: Any, index: int) -> dict[str, Any]:
        if not isinstance(item, dict):
            raise ValueError(f"items[{index}] must be an object")
        if str(item.get("schema_version") or "") != "4.1":
            raise ValueError(f"items[{index}] must use micro-session schema 4.1")
        leaked = MinerInferenceService._find_forbidden(item, f"items[{index}]")
        if leaked:
            raise ValueError(f"items[{index}] contains ground-truth fields: {sorted(leaked)}")
        validate_v4_micro_session(item, index)
        return item

    async def _predict_validated(self, items: list[dict[str, Any]]) -> list[float]:
        if len(items) > self.config.max_sessions_per_request:
            raise ValueError(
                f"Request contains {len(items)} items; maximum is "
                f"{self.config.max_sessions_per_request}"
            )
        request_bytes = len(
            json.dumps(items, separators=(",", ":"), ensure_ascii=False).encode(
                "utf-8"
            )
        )
        if request_bytes > self.config.max_request_bytes:
            raise ValueError(
                f"Request is {request_bytes} bytes; maximum is "
                f"{self.config.max_request_bytes}"
            )
        async with self._model_lock:
            if inspect.iscoroutinefunction(self.model.predict):
                result = await self.model.predict(items)
            else:
                result = await asyncio.to_thread(self.model.predict, items)
        if inspect.isawaitable(result):
            result = await result
        scores = [float(score) for score in result]
        if len(scores) != len(items):
            raise ValueError(
                f"Model returned {len(scores)} scores for {len(items)} items"
            )
        if any(
            not math.isfinite(score) or score < 0.0 or score > 1.0 for score in scores
        ):
            raise ValueError("Model risk scores must be finite values within [0, 1]")
        return scores

    async def predict_micro_sessions(self, items: list[dict[str, Any]]) -> list[float]:
        if not items:
            raise ValueError("Micro-session request contains no items")
        validated = [
            self._validate_micro_session(item, index)
            for index, item in enumerate(items)
        ]
        return await self._predict_validated(validated)
