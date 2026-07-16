"""Shared batch post-processing used at inference and during model selection."""

from __future__ import annotations

import os

import numpy as np

from deploy.batch_calibration import apply_batch_calibration
from deploy.features import _heuristic_score


def _parse_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def hand_heuristic_boost(chunks: list[list[dict]]) -> np.ndarray:
    boosts: list[float] = []
    for chunk in chunks:
        if not chunk:
            boosts.append(0.0)
            continue
        scores = [_heuristic_score(hand) for hand in chunk]
        boosts.append(float(np.percentile(scores, 75)))
    return np.asarray(boosts, dtype=np.float64)


def apply_hand_boost(
    scores: np.ndarray,
    chunks: list[list[dict]],
    *,
    weight: float,
) -> np.ndarray:
    if weight <= 0.0 or not chunks:
        return scores
    boost = hand_heuristic_boost(chunks)
    fused = np.asarray(scores, dtype=np.float64)
    return np.clip(fused + weight * boost * (1.0 - fused), 0.0, 1.0)


def rank_coherent_blend(scores: np.ndarray, *, alpha: float | None = None) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    n = values.size
    if n <= 1:
        return values
    if alpha is None:
        alpha = _parse_float("POKER44_RANK_BLEND", 0.25)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.linspace(0.0, 1.0, n)
    return np.clip((1.0 - alpha) * values + alpha * ranks, 0.0, 1.0)


def finalize_batch_scores(
    scores: np.ndarray,
    chunks: list[list[dict]] | None = None,
    *,
    hand_boost_weight: float = 0.0,
    rank_blend: float | None = None,
    batch_calibrate: bool = True,
) -> np.ndarray:
    result = np.asarray(scores, dtype=np.float64)
    if chunks is not None and hand_boost_weight > 0.0:
        result = apply_hand_boost(result, chunks, weight=hand_boost_weight)
    if result.size > 1:
        result = rank_coherent_blend(result, alpha=rank_blend)
        if batch_calibrate:
            result = apply_batch_calibration(result)
    return np.clip(result, 0.0, 1.0)
