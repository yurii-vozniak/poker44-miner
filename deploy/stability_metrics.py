"""Round-stability metrics for model selection (proxy via per-date holdout)."""

from __future__ import annotations

from typing import Any

import numpy as np

from deploy.eval_metrics import evaluate_scores
from deploy.inference_postprocess import finalize_batch_scores
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
    batch_size: int = 100,
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
            batch_scores = finalize_batch_scores(
                date_values[part],
                [date_chunks[index] for index in part],
                hand_boost_weight=hand_boost_weight,
                rank_blend=rank_blend,
                adaptive_rank=adaptive_rank,
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
) -> float:
    """
    Prefer configs with high worst-date score and low cross-date variance.

    Configs that miss the stability floor are heavily penalized so tuning
    cannot chase a high average on one lucky date.
    """
    if not per_date_rewards:
        return -1.0

    values = np.asarray(list(per_date_rewards.values()), dtype=np.float64)
    min_reward = float(np.min(values))
    mean_reward = float(np.mean(values))
    std_reward = float(np.std(values))

    if min_reward < floor:
        return min_reward - 1.0

    score = 0.55 * min_reward + 0.30 * mean_reward - 0.15 * std_reward
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
