#!/usr/bin/env python3
"""Tune ensemble fusion weights on date holdout with batched validator scoring."""

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
from deploy.stability_metrics import (
    format_stability_report,
    meets_stability_floor,
    per_date_batched_rewards,
    stability_selection_reward,
    stability_summary,
)
from deploy.train_stacked import _batched_window_reward
from poker44.score.scoring import reward
from poker44.validator.payload_view import prepare_hand_for_miner

DEFAULT_MODEL_VERSION = "16"
STABILITY_FLOOR = 0.55


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
    )
    batch = _batched_window_reward(
        scores,
        labels,
        val_examples,
        hand_boost_weight=hand_boost_weight,
        rank_blend=rank_blend,
        max_pos_frac=max_pos_frac,
        adaptive_max_pos_frac=adaptive_max_pos_frac,
    )
    selection = stability_selection_reward(per_date, floor=STABILITY_FLOOR, batch_mean=batch)
    return selection, per_date


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stacked-path", type=Path, default=Path("models/stacked.joblib"))
    parser.add_argument("--hybrid-path", type=Path, default=Path("models/hybrid.joblib"))
    parser.add_argument("--output", type=Path, default=Path("models/ensemble.joblib"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/benchmark"))
    parser.add_argument("--dates", type=int, default=37)
    parser.add_argument("--holdout-dates", type=int, default=10)
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

    best: dict[str, float | dict] = {"selection_reward": -2.0}
    best_per_date: dict[str, float] = {}
    for stacked_w in (0.45, 0.55, 0.65):
        hybrid_w = 1.0 - stacked_w
        for iso_w in (0.20, 0.30, 0.40):
            for hand_mix_w in (0.0, 0.18, 0.26):
                for hand_boost_w in (0.10, 0.14, 0.18):
                    for rank_blend in (0.40, 0.50, 0.60):
                        for max_pos_frac in (0.30, 0.35, 0.40):
                            adaptive_max_pos_frac = True
                            fused = np.clip(
                                stacked_w * stacked_scores + hybrid_w * hybrid_scores,
                                0.0,
                                1.0,
                            )
                            if iso_w > 0.0:
                                fused = np.clip(np.maximum(fused, iso_w * iso_scores), 0.0, 1.0)
                            if hand_mix_w > 0.0:
                                fused = np.clip(np.maximum(fused, hand_mix_w * hand_scores), 0.0, 1.0)
                            selection, per_date = _selection_reward(
                                fused,
                                labels,
                                val_examples,
                                hand_boost_weight=hand_boost_w,
                                rank_blend=rank_blend,
                                max_pos_frac=max_pos_frac,
                                adaptive_max_pos_frac=adaptive_max_pos_frac,
                            )
                            if selection > float(best["selection_reward"]):
                                best = {
                                    "selection_reward": selection,
                                    "stacked_weight": stacked_w,
                                    "hybrid_weight": hybrid_w,
                                    "iso_weight": iso_w,
                                    "hand_mix_weight": hand_mix_w,
                                    "hand_boost_weight": hand_boost_w,
                                    "rank_blend": rank_blend,
                                    "max_pos_frac": max_pos_frac,
                                    "adaptive_max_pos_frac": adaptive_max_pos_frac,
                                    "stability": stability_summary(per_date),
                                    "meets_floor_0_55": meets_stability_floor(per_date),
                                }
                                best_per_date = per_date

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_name": "poker44-dual-ensemble",
        "model_version": DEFAULT_MODEL_VERSION,
        "framework": "stacked+hybrid+iso",
        "validation_rows": len(val_examples),
        "selection_reward": best["selection_reward"],
        "fusion": {k: v for k, v in best.items() if k != "selection_reward"},
        "stability_report": format_stability_report(best_per_date),
        "stacked_model_version": stacked.metadata.get("model_version"),
        "hybrid_model_version": hybrid.metadata.get("model_version"),
    }

    artifact = {
        "model_type": "ensemble",
        "stacked_path": "stacked.joblib",
        "hybrid_path": "hybrid.joblib",
        "stacked_weight": best["stacked_weight"],
        "hybrid_weight": best["hybrid_weight"],
        "iso_weight": best["iso_weight"],
        "hand_mix_weight": best["hand_mix_weight"],
        "hand_boost_weight": best["hand_boost_weight"],
        "rank_blend": best["rank_blend"],
        "max_pos_frac": best.get("max_pos_frac"),
        "adaptive_max_pos_frac": best.get("adaptive_max_pos_frac", True),
        "adaptive_rank": True,
        "metadata": metadata,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output)
    sidecar = args.output.with_suffix(".json")
    sidecar.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Stability report:", json.dumps(format_stability_report(best_per_date), indent=2))
    print("Best fusion:", json.dumps({k: v for k, v in best.items() if k != "selection_reward"}, indent=2, default=str))
    print(f"Saved ensemble to {args.output}")


if __name__ == "__main__":
    main()
