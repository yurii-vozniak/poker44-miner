"""Batch-aware score remapping for validator query batches."""

from __future__ import annotations

import os

import numpy as np


def _parse_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def apply_batch_calibration(
    scores: np.ndarray | list[float],
    *,
    max_pos_frac: float | None = None,
    squeeze_below: float | None = None,
    rank_remap: bool | None = None,
) -> np.ndarray:
    """
    Remap chunk scores using the full validator query batch.

    Top miners cap how many chunks in a batch can sit above the decision cliff
    and spread ranks across the unit interval. This mirrors that behavior without
    changing per-chunk feature scoring.
    """
    if max_pos_frac is None:
        max_pos_frac = _parse_float("POKER44_BATCH_MAX_POS_FRAC", 0.12)
    if squeeze_below is None:
        squeeze_below = _parse_float("POKER44_BATCH_SQUEEZE_BELOW", 0.48)
    if rank_remap is None:
        rank_remap = os.getenv("POKER44_BATCH_RANK_REMAP", "true").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    result = np.asarray(scores, dtype=np.float64).copy()
    n = result.size
    if n <= 1:
        return np.clip(result, 0.0, 1.0)

    order = np.argsort(-result, kind="mergesort")
    k = max(1, int(round(n * max_pos_frac)))
    cutoff = float(result[order[k - 1]])
    high_mask = result >= cutoff
    low_mask = ~high_mask & (result >= 0.5)
    result[low_mask] = np.minimum(result[low_mask], squeeze_below)

    if rank_remap:
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.linspace(1.0, 0.0, n)
        # Blend rank spread with calibrated scores to preserve local ordering.
        spread = 0.08 + 0.84 * ranks
        result = 0.55 * result + 0.45 * spread

    return np.clip(result, 0.0, 1.0)
