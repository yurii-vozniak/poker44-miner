#!/usr/bin/env python3
"""Tune hybrid batch postprocess for validator-style 100-chunk scoring."""

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
from deploy.features import chunk_features
from deploy.hybrid_detector import HybridDetector
from deploy.stability_metrics import (
    format_stability_report,
    meets_stability_floor,
    per_date_batched_rewards,
    stability_selection_reward,
)
from deploy.train_hybrid import _batched_window_reward
from poker44.validator.payload_view import prepare_hand_for_miner

STABILITY_FLOOR = 0.55


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path("models/hybrid.joblib"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/benchmark"))
    parser.add_argument("--dates", type=int, default=41)
    parser.add_argument("--holdout-dates", type=int, default=10)
    args = parser.parse_args()

    client = BenchmarkClient()
    source_dates = client.list_source_dates()[: args.dates]
    records = download_releases(client, source_dates, cache_dir=args.cache_dir, refresh=False)
    examples = list(iter_training_examples(records))
    _, val_examples = split_examples_by_date(examples, holdout_dates=args.holdout_dates)

    prepared = [[prepare_hand_for_miner(h) for h in ex.chunk] for ex in val_examples]
    features = np.vstack([chunk_features(c, for_training=False) for c in prepared])
    labels = np.asarray([ex.label for ex in val_examples], dtype=int)

    detector = HybridDetector(args.model_path)
    base_scores = detector.score_features(features)

    best = {"selection_reward": -2.0}
    best_per_date: dict[str, float] = {}
    for hand_boost in (0.12, 0.16, 0.20, 0.24):
        for rank_blend in (0.72, 0.80, 0.88):
            for max_pos in (0.44, 0.48, 0.52, 0.56):
                per_date = per_date_batched_rewards(
                    base_scores,
                    labels,
                    val_examples,
                    hand_boost_weight=hand_boost,
                    rank_blend=rank_blend,
                    max_pos_frac=max_pos,
                    adaptive_max_pos_frac=True,
                    skip_pre_finalize_fusion=True,
                )
                batch = _batched_window_reward(
                    base_scores,
                    labels,
                    val_examples,
                    hand_boost_weight=hand_boost,
                    rank_blend=rank_blend,
                    max_pos_frac=max_pos,
                )
                selection = stability_selection_reward(
                    per_date,
                    floor=STABILITY_FLOOR,
                    batch_mean=batch,
                    recent_dates=5,
                )
                if selection > float(best["selection_reward"]):
                    best = {
                        "selection_reward": selection,
                        "hand_boost_weight": hand_boost,
                        "rank_blend": rank_blend,
                        "max_pos_frac": max_pos,
                        "meets_floor_0_55": meets_stability_floor(per_date),
                    }
                    best_per_date = per_date

    artifact: dict = joblib.load(args.model_path)
    artifact["hand_boost_weight"] = best["hand_boost_weight"]
    artifact["rank_blend"] = best["rank_blend"]
    artifact["max_pos_frac"] = best["max_pos_frac"]
    artifact["adaptive_max_pos_frac"] = True
    metadata = dict(artifact.get("metadata") or {})
    metadata["batch_tune"] = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "selection_reward": best["selection_reward"],
        "params": {k: v for k, v in best.items() if k != "selection_reward"},
        "stability_report": format_stability_report(best_per_date),
    }
    artifact["metadata"] = metadata
    joblib.dump(artifact, args.model_path)
    sidecar = args.model_path.with_suffix(".json")
    if sidecar.is_file():
        sidecar.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(best, indent=2))
    print(json.dumps(format_stability_report(best_per_date), indent=2))


if __name__ == "__main__":
    main()
