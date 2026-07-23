"""Round-stability metrics for model selection (proxy via per-date holdout)."""

from __future__ import annotations

from typing import Any

import numpy as np

from deploy.eval_metrics import evaluate_scores
from deploy.inference_postprocess import finalize_batch_scores
from deploy.live_rank_fusion import apply_batch_ensemble_fusion
from poker44.score.scoring import reward


def per_date_flat_rewards(
    scores: np.ndarray,
    y_true: np.ndarray,
    val_examples,
) -> dict[str, float]:
    labels = np.asarray(y_true, dtype=int)
    values = np.asarray(scores, dtype=float)
    rewards: dict[str, float] = {}
    for source_date in sorted({example.source_date for example in val_examples}):
        indices = [
            index
            for index, example in enumerate(val_examples)
            if example.source_date == source_date
        ]
        if not indices:
            continue
        metrics = evaluate_scores(values[indices], labels[indices])
        rewards[source_date] = float(metrics.get("reward") or -1.0)
    return rewards


def per_date_batched_rewards(
    scores: np.ndarray,
    y_true: np.ndarray,
    val_examples,
    *,
    hand_boost_weight: float = 0.0,
    rank_blend: float | None = None,
    adaptive_rank: bool = True,
    max_pos_frac: float | None = None,
    adaptive_max_pos_frac: bool = True,
    batch_size: int = 100,
    iso_scores: np.ndarray | None = None,
    hand_scores: np.ndarray | None = None,
    hand_mix_weight: float = 0.0,
    live_rank_weight: float = 0.0,
) -> dict[str, float]:
    """Simulate validator 100-chunk forwards separately for each release date."""
    labels = np.asarray(y_true, dtype=int)
    values = np.asarray(scores, dtype=float)
    chunks = [example.chunk for example in val_examples]
    rewards: dict[str, float] = {}

    for source_date in sorted({example.source_date for example in val_examples}):
        indices = np.asarray(
            [
                index
                for index, example in enumerate(val_examples)
                if example.source_date == source_date
            ],
            dtype=int,
        )
        if indices.size < 20:
            continue

        date_labels = labels[indices]
        date_values = values[indices]
        date_chunks = [chunks[index] for index in indices]
        batch_rewards: list[float] = []

        for start in range(0, indices.size, batch_size):
            part = np.arange(start, min(start + batch_size, indices.size))
            if part.size < 20:
                continue
            batch_chunks = [date_chunks[index] for index in part]
            batch_base = date_values[part]
            batch_iso = (
                np.asarray(iso_scores, dtype=np.float64)[indices[part]]
                if iso_scores is not None and iso_scores.size == values.size
                else np.zeros(part.size, dtype=np.float64)
            )
            batch_hand = (
                np.asarray(hand_scores, dtype=np.float64)[indices[part]]
                if hand_scores is not None and hand_scores.size == values.size
                else np.zeros(part.size, dtype=np.float64)
            )
            batch_base = apply_batch_ensemble_fusion(
                batch_base,
                batch_chunks,
                iso_scores=batch_iso,
                hand_scores=batch_hand,
                hand_mix_weight=hand_mix_weight,
                live_rank_weight=live_rank_weight,
            )
            batch_scores = finalize_batch_scores(
                batch_base,
                batch_chunks,
                hand_boost_weight=hand_boost_weight,
                rank_blend=rank_blend,
                adaptive_rank=adaptive_rank,
                max_pos_frac=max_pos_frac,
                adaptive_max_pos_frac=adaptive_max_pos_frac,
            )
            _, metrics = reward(batch_scores, date_labels[part])
            batch_rewards.append(float(metrics["reward"]))

        if batch_rewards:
            rewards[source_date] = float(np.mean(batch_rewards))
    return rewards


def stability_summary(per_date_rewards: dict[str, float]) -> dict[str, float | None]:
    if not per_date_rewards:
        return {
            "mean": None,
            "min": None,
            "max": None,
            "std": None,
        }
    values = np.asarray(list(per_date_rewards.values()), dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "std": float(np.std(values)),
    }


def stability_selection_reward(
    per_date_rewards: dict[str, float],
    *,
    floor: float = 0.55,
    batch_mean: float | None = None,
    recent_dates: int = 7,
) -> float:
    """
    Prefer configs with high worst-date score and low cross-date variance.

    Configs that miss the stability floor are heavily penalized so tuning
    cannot chase a high average on one lucky date. Recent release dates are
    weighted more heavily because they better match live validator rounds.
    """
    if not per_date_rewards:
        return -1.0

    values = np.asarray(list(per_date_rewards.values()), dtype=np.float64)
    sorted_dates = sorted(per_date_rewards.keys())
    recent = sorted_dates[-recent_dates:] if recent_dates > 0 else sorted_dates
    recent_values = np.asarray([per_date_rewards[source_date] for source_date in recent], dtype=np.float64)
    min_reward = float(np.min(values))
    min_recent = float(np.min(recent_values)) if recent_values.size else min_reward
    mean_reward = float(np.mean(values))
    mean_recent = float(np.mean(recent_values)) if recent_values.size else mean_reward
    std_reward = float(np.std(values))

    if min_reward < floor or min_recent < floor:
        return min(min_reward, min_recent) - 1.0

    score = (
        0.50 * min_recent
        + 0.10 * min_reward
        + 0.20 * mean_recent
        + 0.05 * mean_reward
        - 0.15 * std_reward
    )
    if batch_mean is not None:
        score += 0.10 * batch_mean
    return score


def meets_stability_floor(
    per_date_rewards: dict[str, float],
    *,
    floor: float = 0.55,
) -> bool:
    return bool(per_date_rewards) and min(per_date_rewards.values()) >= floor - 1e-9


def format_stability_report(per_date_rewards: dict[str, float]) -> dict[str, Any]:
    summary = stability_summary(per_date_rewards)
    return {
        "per_date": {key: round(value, 4) for key, value in sorted(per_date_rewards.items())},
        "summary": summary,
        "meets_floor_0_55": meets_stability_floor(per_date_rewards),
    }
