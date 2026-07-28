#!/usr/bin/env python3
"""Tune coherent rank ensemble for >=0.60 batched holdout stability."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from deploy.benchmark_client import BenchmarkClient
from deploy.benchmark_dataset import (
    download_releases,
    iter_training_examples,
    split_examples_by_date,
)
from deploy.chunk_detector import load_chunk_detector
from deploy.features import chunk_features
from deploy.live_rank_fusion import build_rank_only_batch_scores
from deploy.stability_metrics import (
    format_stability_report,
    imbalance_robustness_rewards,
    meets_stability_floor,
    per_date_batched_rewards,
    stability_selection_reward,
    stability_summary,
)
from deploy.train_stacked import _batched_window_reward
from poker44.validator.payload_view import prepare_hand_for_miner

DEFAULT_MODEL_VERSION = "23"
STABILITY_FLOOR = 0.60

# iso, stacked, hybrid, hand, heuristic — tuned toward top live miners
RANK_WEIGHT_PRESETS: tuple[tuple[float, ...], ...] = (
    (0.28, 0.10, 0.10, 0.30, 0.22),
    (0.30, 0.08, 0.08, 0.28, 0.26),
    (0.32, 0.10, 0.10, 0.26, 0.22),
    (0.26, 0.12, 0.12, 0.24, 0.26),
    (0.34, 0.12, 0.12, 0.22, 0.20),
    (0.24, 0.08, 0.08, 0.32, 0.28),
)


def _selection_reward(
    scores: np.ndarray,
    labels: np.ndarray,
    val_examples,
    *,
    hand_boost_weight: float,
    rank_blend: float,
    max_pos_frac: float | None,
    adaptive_max_pos_frac: bool,
) -> tuple[float, dict[str, float]]:
    per_date = per_date_batched_rewards(
        scores,
        labels,
        val_examples,
        hand_boost_weight=hand_boost_weight,
        rank_blend=rank_blend,
        adaptive_rank=True,
        max_pos_frac=max_pos_frac,
        adaptive_max_pos_frac=adaptive_max_pos_frac,
        skip_pre_finalize_fusion=True,
    )
    batch = _batched_window_reward(
        scores,
        labels,
        val_examples,
        hand_boost_weight=hand_boost_weight,
        rank_blend=rank_blend,
        max_pos_frac=max_pos_frac,
        adaptive_max_pos_frac=adaptive_max_pos_frac,
        skip_pre_finalize_fusion=True,
    )
    selection = stability_selection_reward(
        per_date,
        floor=STABILITY_FLOOR,
        batch_mean=batch,
        recent_dates=5,
    )
    return selection, per_date


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stacked-path", type=Path, default=Path("models/stacked.joblib"))
    parser.add_argument("--hybrid-path", type=Path, default=Path("models/hybrid.joblib"))
    parser.add_argument("--output", type=Path, default=Path("models/ensemble.joblib"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/benchmark"))
    parser.add_argument("--dates", type=int, default=30)
    parser.add_argument("--holdout-dates", type=int, default=8)
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args()

    client = BenchmarkClient()
    source_dates = client.list_source_dates()[: args.dates]
    records = download_releases(
        client,
        source_dates,
        cache_dir=args.cache_dir,
        refresh=args.refresh_cache,
    )
    examples = list(iter_training_examples(records))
    _, val_examples = split_examples_by_date(examples, holdout_dates=args.holdout_dates)
    if not val_examples:
        raise RuntimeError("No validation examples.")

    stacked = load_chunk_detector(args.stacked_path)
    hybrid = load_chunk_detector(args.hybrid_path)

    prepared = [
        [prepare_hand_for_miner(h) for h in example.chunk]
        for example in val_examples
    ]
    features = np.vstack([chunk_features(chunk, for_training=False) for chunk in prepared])
    labels = np.asarray([example.label for example in val_examples], dtype=int)
    stacked_scores = stacked.score_features(features)
    hybrid_scores = hybrid._supervised_probability(hybrid.scaler.transform(features))
    iso_scores = np.maximum(
        stacked._anomaly_probability(features)
        if hasattr(stacked, "_anomaly_probability")
        else np.zeros(len(val_examples)),
        hybrid._anomaly_probability(hybrid.scaler.transform(features))
        if hasattr(hybrid, "_anomaly_probability")
        else np.zeros(len(val_examples)),
    )
    hand_scores = (
        stacked._hand_aggregate_for_chunks(prepared)
        if hasattr(stacked, "_hand_aggregate_for_chunks")
        else np.zeros(len(val_examples))
    )

    best: dict[str, float | dict | list | bool] = {"selection_reward": -2.0}
    best_per_date: dict[str, float] = {}
    best_robustness: dict[float, float] = {}
    # Real validator batches almost certainly do NOT share the benchmark's
    # artificial 50/50 bot/human balance. Because the reward's hard-threshold
    # calibration term is extremely sensitive to over-flagging humans, a much
    # smaller max_pos_frac dominates larger ones across every plausible live
    # bot rate (see deploy/batch_calibration.py). Search a low, non-adaptive
    # range instead of assuming the benchmark's balance holds live.
    for rank_weights in RANK_WEIGHT_PRESETS:
        rank_scores = build_rank_only_batch_scores(
            prepared,
            iso_scores=iso_scores,
            stacked_scores=stacked_scores,
            hybrid_scores=hybrid_scores,
            hand_scores=hand_scores,
            rank_signal_weights=rank_weights,
        )
        for hand_boost_w in (0.18, 0.22, 0.26):
            for rank_blend in (0.85, 0.90, 0.92):
                for max_pos_frac in (0.05, 0.08, 0.10, 0.12, 0.15):
                    selection, per_date = _selection_reward(
                        rank_scores,
                        labels,
                        val_examples,
                        hand_boost_weight=hand_boost_w,
                        rank_blend=rank_blend,
                        max_pos_frac=max_pos_frac,
                        adaptive_max_pos_frac=False,
                    )
                    robustness = imbalance_robustness_rewards(
                        rank_scores,
                        labels,
                        val_examples,
                        hand_boost_weight=hand_boost_w,
                        rank_blend=rank_blend,
                        max_pos_frac=max_pos_frac,
                        adaptive_max_pos_frac=False,
                    )
                    worst_case = min(robustness.values()) if robustness else -1.0
                    mean_case = (
                        float(np.mean(list(robustness.values()))) if robustness else -1.0
                    )
                    # Blend the date-based selection reward with worst-case and
                    # mean robustness across assumed live bot-prevalence rates,
                    # so we don't just re-overfit to the 50/50 benchmark split.
                    combined = 0.40 * selection + 0.40 * worst_case + 0.20 * mean_case
                    if combined > float(best["selection_reward"]):
                        best = {
                            "selection_reward": combined,
                            "rank_only": True,
                            "rank_signal_weights": list(rank_weights),
                            "hand_boost_weight": hand_boost_w,
                            "rank_blend": rank_blend,
                            "max_pos_frac": max_pos_frac,
                            "adaptive_max_pos_frac": False,
                            "stability": stability_summary(per_date),
                            "meets_floor_0_60": meets_stability_floor(
                                per_date, floor=STABILITY_FLOOR
                            ),
                            "date_selection_reward": selection,
                            "robustness_worst_case": worst_case,
                            "robustness_mean_case": mean_case,
                        }
                        best_per_date = per_date
                        best_robustness = robustness

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_name": "poker44-coherent-rank",
        "model_version": DEFAULT_MODEL_VERSION,
        "framework": "coherent-rank-benchmark-supervised",
        "validation_rows": len(val_examples),
        "selection_reward": best["selection_reward"],
        "fusion": {k: v for k, v in best.items() if k != "selection_reward"},
        "stability_report": format_stability_report(best_per_date),
        "imbalance_robustness_report": {
            f"bot_frac_{frac}": round(value, 4) for frac, value in sorted(best_robustness.items())
        },
        "stacked_model_version": stacked.metadata.get("model_version"),
        "hybrid_model_version": hybrid.metadata.get("model_version"),
    }

    artifact = {
        "model_type": "ensemble",
        "stacked_path": "stacked.joblib",
        "hybrid_path": "hybrid.joblib",
        "rank_only": True,
        "rank_signal_weights": best["rank_signal_weights"],
        "hand_boost_weight": best["hand_boost_weight"],
        "rank_blend": best["rank_blend"],
        "max_pos_frac": best.get("max_pos_frac"),
        "adaptive_max_pos_frac": best.get("adaptive_max_pos_frac", True),
        "adaptive_rank": True,
        "stacked_weight": 0.0,
        "hybrid_weight": 0.0,
        "iso_weight": 0.0,
        "hand_mix_weight": 0.0,
        "live_rank_weight": 0.0,
        "benchmark_supervised_weight": 0.0,
        "metadata": metadata,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output)
    sidecar = args.output.with_suffix(".json")
    sidecar.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Stability report:", json.dumps(format_stability_report(best_per_date), indent=2))
    print(
        "Imbalance robustness report (reward at assumed live bot rates):",
        json.dumps({f"bot_frac_{f}": round(v, 4) for f, v in sorted(best_robustness.items())}, indent=2),
    )
    print("Best fusion:", json.dumps({k: v for k, v in best.items() if k != "selection_reward"}, indent=2, default=str))
    print(f"Saved ensemble to {args.output}")


if __name__ == "__main__":
    main()
