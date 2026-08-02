"""Load a miner-owned model factory without imposing repository metadata."""

from __future__ import annotations

import importlib
from collections.abc import Callable

from poker44.miner.config import MinerModelConfig
from poker44.miner.model import BotDetectionModel


def _resolve_factory(path: str) -> Callable[[MinerModelConfig], BotDetectionModel]:
    module_name, separator, attribute = path.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError(
            "POKER44_MODEL_FACTORY must use the form 'python.module:create_model'"
        )
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute)
    if not callable(factory):
        raise TypeError(f"Configured model factory is not callable: {path}")
    return factory


def load_model(config: MinerModelConfig) -> BotDetectionModel:
    factory = _resolve_factory(config.factory)
    model = factory(config)
    if not isinstance(model, BotDetectionModel):
        raise TypeError(
            "Poker44 model must expose version, load() and predict(sessions)"
        )
    model.load()
    return model
