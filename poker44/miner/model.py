"""Stable model interface implemented by every Poker44 miner."""

from __future__ import annotations

from collections import Counter
from typing import Any, Protocol, runtime_checkable

from poker44.miner.config import MinerModelConfig


@runtime_checkable
class BotDetectionModel(Protocol):
    """A model resident in a miner process and served through its axon."""

    version: str

    def load(self) -> None: ...

    def predict(self, sessions: list[dict[str, Any]]) -> list[float]: ...


class ReferenceSessionModel:
    """Deterministic baseline using the four miner-visible poker decisions.

    This is intentionally only a runnable reference. Production miners provide
    their own factory through ``POKER44_MODEL_FACTORY``.
    """

    def __init__(self, config: MinerModelConfig):
        self.config = config
        self.version = config.version

    def load(self) -> None:
        return None

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @classmethod
    def _session_score(cls, session: dict[str, Any]) -> float:
        decisions = [
            decision
            for decision in (session.get("decisions") or [])
            if isinstance(decision, dict)
        ]
        counts = Counter(str(decision.get("action_type") or "") for decision in decisions)
        meaningful = max(1, len(decisions))
        aggression = (counts["bet"] + counts["raise"] + counts["all_in"]) / meaningful
        passive = (counts["check"] + counts["call"]) / meaningful
        folds = counts["fold"] / meaningful
        overbets = sum(
            decision.get("size_bucket") in {"overbet", "all_in"}
            for decision in decisions
        ) / meaningful
        score = (
            0.42 * cls._clamp(aggression / 0.45)
            + 0.24 * cls._clamp(folds / 0.45)
            + 0.20 * cls._clamp(overbets / 0.20)
            + 0.14 * (1.0 - cls._clamp(passive / 0.75))
        )
        return round(cls._clamp(score), 6)

    def predict(self, sessions: list[dict[str, Any]]) -> list[float]:
        return [self._session_score(session) for session in sessions]


def create_reference_model(config: MinerModelConfig) -> BotDetectionModel:
    return ReferenceSessionModel(config)
