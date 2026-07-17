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
from deploy.ensemble_detector import EnsembleDetector
from deploy.eval_metrics import evaluate_scores
from deploy.features import chunk_features
from deploy.inference_postprocess import finalize_batch_scores
from deploy.train_stacked import _batched_window_reward
from poker44.score.scoring import reward
from poker44.validator.payload_view import prepare_hand_for_miner

DEFAULT_MODEL_VERSION = "13"


def _selection_reward(
    scores: np.ndarray,
    labels: np.ndarray,
    val_examples,
    *,
    hand_boost_weight: float,
    rank_blend: float,
) -> float:
    flat = float(evaluate_scores(scores, labels).get("reward") or -1.0)
    batch = _batched_window_reward(
        scores,
        labels,
        val_examples,
        hand_boost_weight=hand_boost_weight,
        rank_blend=rank_blend,
    )
    if batch is None:
        batch = flat
    return 0.15 * flat + 0.85 * batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stacked-path", type=Path, default=Path("models/stacked.joblib"))
    parser.add_argument("--hybrid-path", type=Path, default=Path("models/hybrid.joblib"))
    parser.add_argument("--output", type=Path, default=Path("models/ensemble.joblib"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/benchmark"))
    parser.add_argument("--dates", type=int, default=36)
    parser.add_argument("--holdout-dates", type=int, default=7)
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
    iso_scores = (
        stacked._anomaly_probability(features)
        if hasattr(stacked, "_anomaly_probability")
        else np.zeros(len(val_examples))
    )
    hand_scores = (
        stacked._hand_aggregate_for_chunks(prepared)
        if hasattr(stacked, "_hand_aggregate_for_chunks")
        else np.zeros(len(val_examples))
    )

    best: dict[str, float] = {"selection_reward": -1.0}
    for stacked_w in (0.45, 0.55, 0.65):
        hybrid_w = 1.0 - stacked_w
        for iso_w in (0.0, 0.15, 0.25):
            for hand_mix_w in (0.0, 0.15, 0.22):
                for hand_boost_w in (0.08, 0.12, 0.16):
                    for rank_blend in (0.25, 0.35, 0.45):
                        fused = np.clip(
                            stacked_w * stacked_scores + hybrid_w * hybrid_scores,
                            0.0,
                            1.0,
                        )
                        if iso_w > 0.0:
                            fused = np.clip(np.maximum(fused, iso_w * iso_scores), 0.0, 1.0)
                        if hand_mix_w > 0.0:
                            fused = np.clip(np.maximum(fused, hand_mix_w * hand_scores), 0.0, 1.0)
                        selection = _selection_reward(
                            fused,
                            labels,
                            val_examples,
                            hand_boost_weight=hand_boost_w,
                            rank_blend=rank_blend,
                        )
                        if selection > best["selection_reward"]:
                            best = {
                                "selection_reward": selection,
                                "stacked_weight": stacked_w,
                                "hybrid_weight": hybrid_w,
                                "iso_weight": iso_w,
                                "hand_mix_weight": hand_mix_w,
                                "hand_boost_weight": hand_boost_w,
                                "rank_blend": rank_blend,
                            }

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_name": "poker44-dual-ensemble",
        "model_version": DEFAULT_MODEL_VERSION,
        "framework": "stacked+hybrid+iso",
        "validation_rows": len(val_examples),
        "selection_reward": best["selection_reward"],
        "fusion": best,
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
        "metadata": metadata,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output)
    sidecar = args.output.with_suffix(".json")
    sidecar.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Best fusion:", json.dumps(best, indent=2))
    print(f"Saved ensemble to {args.output}")


if __name__ == "__main__":
    main()
