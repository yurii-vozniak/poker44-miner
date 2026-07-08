#!/usr/bin/env python3
"""Train a hybrid LightGBM + Isolation Forest detector with date holdout."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from deploy.benchmark_client import BenchmarkClient
from deploy.benchmark_dataset import (
    download_releases,
    examples_to_arrays,
    iter_training_examples,
    split_examples_by_date,
    summarize_examples,
)
from deploy.eval_metrics import evaluate_scores
from deploy.features import FEATURE_NAMES


def _maybe_invert_scores(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, bool]:
    metrics = evaluate_scores(scores, y_true)
    roc_auc = metrics.get("roc_auc")
    if roc_auc is None or roc_auc >= 0.5:
        return scores, False
    return 1.0 - scores, True


def _fit_hybrid(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[StandardScaler, lgb.LGBMClassifier, IsolationForest, dict]:
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_train_frame = pd.DataFrame(x_train_scaled, columns=FEATURE_NAMES)
    x_val_frame = pd.DataFrame(x_val_scaled, columns=FEATURE_NAMES)

    positive_rate = float(np.mean(y_train)) if y_train.size else 0.1
    lgbm = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )
    lgbm.fit(x_train_frame, y_train)

    human_mask = y_train == 0
    iso_train = x_train_scaled[human_mask] if np.any(human_mask) else x_train_scaled
    iso_forest = IsolationForest(
        n_estimators=200,
        contamination=min(max(positive_rate, 0.01), 0.49),
        random_state=42,
    )
    iso_forest.fit(iso_train)

    supervised_val = lgbm.predict_proba(x_val_frame)[:, 1]
    raw_iso_val = -iso_forest.score_samples(x_val_scaled)
    raw_iso_train = -iso_forest.score_samples(x_train_scaled)
    iso_min = float(np.min(raw_iso_train))
    iso_max = float(np.max(raw_iso_train))
    span = max(iso_max - iso_min, 1e-8)
    anomaly_val = np.clip((raw_iso_val - iso_min) / span, 0.0, 1.0)

    hybrid_val = np.maximum(supervised_val, anomaly_val)
    hybrid_val, invert_scores = _maybe_invert_scores(y_val, hybrid_val)
    if invert_scores:
        supervised_val = 1.0 - supervised_val
        anomaly_val = 1.0 - anomaly_val

    calibration = {
        "invert_scores": invert_scores,
        "iso_min": iso_min,
        "iso_max": iso_max,
        "validation": {
            "lgbm": evaluate_scores(supervised_val, y_val),
            "iso": evaluate_scores(anomaly_val, y_val),
            "hybrid": evaluate_scores(hybrid_val, y_val),
        },
    }
    return scaler, lgbm, iso_forest, calibration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/hybrid.joblib"),
        help="Path to write the trained hybrid model artifact.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/benchmark"),
        help="Directory for cached benchmark chunk records.",
    )
    parser.add_argument(
        "--dates",
        type=int,
        default=7,
        help="Number of most recent release dates to use.",
    )
    parser.add_argument(
        "--source-dates",
        nargs="*",
        default=None,
        help="Explicit source dates (YYYY-MM-DD). Overrides --dates.",
    )
    parser.add_argument(
        "--holdout-dates",
        type=int,
        default=1,
        help="Hold out the most recent N release dates for validation.",
    )
    parser.add_argument(
        "--max-chunks-per-date",
        type=int,
        default=None,
        help="Optional cap on chunk records downloaded per release date.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Re-download benchmark records even when cache files exist.",
    )
    args = parser.parse_args()

    client = BenchmarkClient()
    if args.source_dates:
        source_dates = list(args.source_dates)
    else:
        source_dates = client.list_source_dates()[: args.dates]

    if len(source_dates) <= args.holdout_dates:
        raise RuntimeError(
            f"Need more than {args.holdout_dates} source dates for holdout validation; "
            f"got {len(source_dates)}."
        )

    print(f"Using source dates: {', '.join(source_dates)}")
    records_by_date = download_releases(
        client,
        source_dates,
        cache_dir=args.cache_dir,
        max_chunks_per_date=args.max_chunks_per_date,
        refresh=args.refresh_cache,
    )
    examples = list(iter_training_examples(records_by_date))
    train_examples, val_examples = split_examples_by_date(
        examples,
        holdout_dates=args.holdout_dates,
    )

    if not train_examples or not val_examples:
        raise RuntimeError("Train/validation split produced empty partitions.")

    x_train, y_train, train_meta = examples_to_arrays(train_examples)
    x_val, y_val, val_meta = examples_to_arrays(val_examples)

    print(
        "Train summary:",
        json.dumps(summarize_examples(train_examples), indent=2),
    )
    print(
        "Validation summary:",
        json.dumps(summarize_examples(val_examples), indent=2),
    )

    scaler, lgbm, iso_forest, calibration = _fit_hybrid(x_train, y_train, x_val, y_val)

    per_date_metrics: dict[str, dict] = {}
    for source_date in sorted({example.source_date for example in val_examples}):
        date_examples = [example for example in val_examples if example.source_date == source_date]
        _, date_labels, _ = examples_to_arrays(date_examples)
        date_features = np.vstack([example.feature_row for example in date_examples])
        scaled = scaler.transform(date_features)
        date_frame = pd.DataFrame(scaled, columns=FEATURE_NAMES)
        supervised = lgbm.predict_proba(date_frame)[:, 1]
        if calibration["invert_scores"]:
            supervised = 1.0 - supervised
        raw_iso = -iso_forest.score_samples(scaled)
        span = max(calibration["iso_max"] - calibration["iso_min"], 1e-8)
        anomaly = np.clip((raw_iso - calibration["iso_min"]) / span, 0.0, 1.0)
        hybrid_scores = np.maximum(supervised, anomaly)
        per_date_metrics[source_date] = evaluate_scores(hybrid_scores, date_labels)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_name": "poker44-hybrid-lgbm-iso",
        "model_version": "1",
        "framework": "lightgbm+sklearn",
        "feature_names": FEATURE_NAMES,
        "source_dates": source_dates,
        "holdout_dates": args.holdout_dates,
        "train": summarize_examples(train_examples),
        "validation": summarize_examples(val_examples),
        "calibration": calibration,
        "per_date_validation": per_date_metrics,
        "sample_train_meta": train_meta[:3],
        "sample_val_meta": val_meta[:3],
    }
    artifact = {
        "scaler": scaler,
        "lgbm": lgbm,
        "iso_forest": iso_forest,
        "invert_scores": calibration["invert_scores"],
        "iso_min": calibration["iso_min"],
        "iso_max": calibration["iso_max"],
        "metadata": metadata,
    }
    joblib.dump(artifact, args.output)

    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved model to {args.output}")
    print(f"Saved metadata to {summary_path}")
    print(
        "Validation hybrid metrics:",
        json.dumps(calibration["validation"]["hybrid"], indent=2),
    )
    print("Per-date validation metrics:", json.dumps(per_date_metrics, indent=2))


if __name__ == "__main__":
    main()
