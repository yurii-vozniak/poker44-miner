"""Multi-signal rank fusion for live validator batches (rank-detector style)."""

from __future__ import annotations

from typing import Callable

import numpy as np

from deploy.inference_postprocess import hand_heuristic_boost


def multi_signal_rank_scores(
    *signals: np.ndarray,
    weights: tuple[float, ...] | None = None,
) -> np.ndarray:
    """Fuse normalized rank orders from one or more score signals."""
    components = [np.asarray(signal, dtype=np.float64) for signal in signals if signal.size]
    if not components:
        return np.zeros(0, dtype=np.float64)

    n = components[0].size
    if weights is None:
        unit = 1.0 / len(components)
        resolved = tuple(unit for _ in components)
    else:
        total = sum(weights[: len(components)]) or 1.0
        resolved = tuple(weight / total for weight in weights[: len(components)])

    fused = np.zeros(n, dtype=np.float64)
    for weight, signal in zip(resolved, components):
        order = np.argsort(signal, kind="mergesort")
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.linspace(0.0, 1.0, n)
        fused += weight * ranks
    return np.clip(fused, 0.0, 1.0)


def apply_live_rank_fusion(
    base_scores: np.ndarray,
    rank_scores: np.ndarray,
    *,
    weight: float,
    batch_std: float | None = None,
) -> np.ndarray:
    """Blend supervised/iso scores with a rank-first signal on weak-separation batches."""
    base = np.asarray(base_scores, dtype=np.float64)
    ranks = np.asarray(rank_scores, dtype=np.float64)
    if base.size <= 1 or weight <= 0.0:
        return base

    resolved = float(weight)
    if batch_std is not None:
        if batch_std < 0.04:
            resolved = min(0.95, resolved + 0.30)
        elif batch_std < 0.08:
            resolved = min(0.88, resolved + 0.20)
        elif batch_std < 0.12:
            resolved = min(0.80, resolved + 0.10)

    return np.clip((1.0 - resolved) * base + resolved * ranks, 0.0, 1.0)


def benchmark_supervised_rank_fusion(
    *,
    iso_scores: np.ndarray,
    stacked_scores: np.ndarray,
    hybrid_scores: np.ndarray,
    hand_scores: np.ndarray,
    heuristic: np.ndarray | None = None,
) -> np.ndarray:
    """Pure rank blend of anomaly/heuristic/hand signals (top live-miner style)."""
    components = [
        np.asarray(iso_scores, dtype=np.float64),
        np.asarray(stacked_scores, dtype=np.float64),
        np.asarray(hybrid_scores, dtype=np.float64),
        np.asarray(hand_scores, dtype=np.float64),
    ]
    if heuristic is not None and heuristic.size:
        components.append(np.asarray(heuristic, dtype=np.float64))
    return multi_signal_rank_scores(
        *components,
        weights=(0.30, 0.18, 0.18, 0.18, 0.16),
    )


def apply_batch_ensemble_fusion(
    fused: np.ndarray,
    chunks: list[list[dict]],
    *,
    iso_scores: np.ndarray,
    hand_scores: np.ndarray | None = None,
    stacked_scores: np.ndarray | None = None,
    hybrid_scores: np.ndarray | None = None,
    hand_mix_weight: float = 0.0,
    live_rank_weight: float = 0.0,
    benchmark_supervised_weight: float = 0.0,
    heuristic_fn: Callable[[list[list[dict]]], np.ndarray] = hand_heuristic_boost,
) -> np.ndarray:
    """Apply heuristic + live-rank fusion before finalize_batch_scores."""
    scores = np.asarray(fused, dtype=np.float64).copy()
    if scores.size <= 1:
        return scores

    heuristic = heuristic_fn(chunks) if scores.size > 1 else None
    batch_std = float(np.std(scores))
    if heuristic is not None and batch_std < 0.14:
        scores = np.clip(np.maximum(scores, 0.42 * heuristic), 0.0, 1.0)

    hand_rank = (
        np.asarray(hand_scores, dtype=np.float64)
        if hand_scores is not None and hand_scores.size == scores.size
        else np.zeros(scores.size, dtype=np.float64)
    )

    if benchmark_supervised_weight > 0.0 and stacked_scores is not None and hybrid_scores is not None:
        rank_supervised = benchmark_supervised_rank_fusion(
            iso_scores=iso_scores,
            stacked_scores=stacked_scores,
            hybrid_scores=hybrid_scores,
            hand_scores=hand_rank,
            heuristic=heuristic,
        )
        scores = apply_live_rank_fusion(
            scores,
            rank_supervised,
            weight=benchmark_supervised_weight,
            batch_std=batch_std,
        )

    if live_rank_weight > 0.0:
        rank_inputs = [scores, np.asarray(iso_scores, dtype=np.float64)]
        if heuristic is not None:
            rank_inputs.append(heuristic)
        if np.any(hand_rank):
            rank_inputs.append(hand_rank)
        rank_signal = multi_signal_rank_scores(
            *rank_inputs,
            weights=(0.30, 0.30, 0.20, 0.20),
        )
        scores = apply_live_rank_fusion(
            scores,
            rank_signal,
            weight=live_rank_weight,
            batch_std=float(np.std(scores)),
        )

    if hand_mix_weight > 0.0 and np.any(hand_rank):
        scores = np.clip(np.maximum(scores, hand_mix_weight * hand_rank), 0.0, 1.0)
    return scores
