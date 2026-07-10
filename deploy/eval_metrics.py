"""Evaluation metrics aligned with Poker44 validator scoring."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from poker44.score.scoring import reward


def recall_at_fpr(
    y_score: np.ndarray,
    y_true: np.ndarray,
    *,
    max_fpr: float = 0.05,
) -> tuple[float, float]:
    """Return best bot recall while human FPR stays at or below ``max_fpr``."""
    _, metrics = reward(np.asarray(y_score, dtype=float), np.asarray(y_true, dtype=int))
    return float(metrics["bot_recall"]), float(metrics["fpr"])


def windowed_reward(
    y_score: np.ndarray,
    y_true: np.ndarray,
    *,
    window_size: int = 100,
    n_trials: int = 8,
    seed: int = 42,
) -> float | None:
    """Average validator-style reward over random fixed-size windows."""
    scores = np.asarray(y_score, dtype=float)
    labels = np.asarray(y_true, dtype=int)
    if labels.size == 0 or len(set(labels.tolist())) < 2:
        return None
    if labels.size <= window_size:
        _, metrics = reward(scores, labels)
        return float(metrics["reward"])

    rng = np.random.default_rng(seed)
    rewards: list[float] = []
    for _ in range(n_trials):
        idx = rng.choice(labels.size, size=window_size, replace=False)
        _, metrics = reward(scores[idx], labels[idx])
        rewards.append(float(metrics["reward"]))
    return float(np.mean(rewards))


def evaluate_scores(
    y_score: np.ndarray,
    y_true: np.ndarray,
    *,
    max_fpr: float = 0.05,
) -> dict[str, float | None]:
    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_score, dtype=float)
    if labels.size == 0 or len(set(labels.tolist())) < 2:
        return {
            "roc_auc": None,
            "average_precision": None,
            "bot_recall_at_fpr": None,
            "fpr_at_recall": None,
            "reward": None,
        }

    roc_auc = float(roc_auc_score(labels, scores))
    average_precision = float(average_precision_score(labels, scores))
    bot_recall, fpr = recall_at_fpr(scores, labels, max_fpr=max_fpr)
    rew, _ = reward(scores, labels)
    return {
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "bot_recall_at_fpr": bot_recall,
        "fpr_at_recall": fpr,
        "reward": float(rew),
    }
