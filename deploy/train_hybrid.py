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
from sklearn.isotonic import IsotonicRegression
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


def _anomaly_scores(
    iso_forest: IsolationForest,
    features: np.ndarray,
    *,
    iso_min: float,
    iso_max: float,
) -> np.ndarray:
    raw = -iso_forest.score_samples(features)
    span = max(iso_max - iso_min, 1e-8)
    return np.clip((raw - iso_min) / span, 0.0, 1.0)


def _fuse_scores(
    supervised: np.ndarray,
    anomaly: np.ndarray,
    *,
    fusion_mode: str,
    iso_blend_weight: float,
) -> np.ndarray:
    if fusion_mode == "max":
        return np.maximum(supervised, anomaly)
    if fusion_mode == "blend":
        return np.clip(supervised + iso_blend_weight * anomaly * (1.0 - supervised), 0.0, 1.0)
    return supervised


def _select_fusion(
    supervised: np.ndarray,
    anomaly: np.ndarray,
    y_true: np.ndarray,
) -> tuple[str, float, dict]:
    candidates: list[tuple[str, float, np.ndarray]] = []
    candidates.append(("supervised", 0.0, supervised.copy()))
    for weight in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40):
        candidates.append(
            (
                "blend",
                weight,
                _fuse_scores(supervised, anomaly, fusion_mode="blend", iso_blend_weight=weight),
            )
        )
    candidates.append(("max", 0.0, _fuse_scores(supervised, anomaly, fusion_mode="max", iso_blend_weight=0.0)))

    best_mode = "supervised"
    best_weight = 0.0
    best_metrics: dict = {}
    best_reward = -1.0
    best_scores = supervised

    for mode, weight, scores in candidates:
        metrics = evaluate_scores(scores, y_true)
        reward = float(metrics.get("reward") or -1.0)
        if reward > best_reward:
            best_reward = reward
            best_mode = mode
            best_weight = weight
            best_metrics = metrics
            best_scores = scores

    return best_mode, best_weight, {
        "fusion_mode": best_mode,
        "iso_blend_weight": best_weight,
        "metrics": best_metrics,
        "scores": best_scores,
    }


def _fit_hybrid(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[StandardScaler, lgb.LGBMClassifier, IsolationForest, IsotonicRegression | None, dict]:
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_train_frame = pd.DataFrame(x_train_scaled, columns=FEATURE_NAMES)
    x_val_frame = pd.DataFrame(x_val_scaled, columns=FEATURE_NAMES)

    positive_rate = float(np.mean(y_train)) if y_train.size else 0.1
    lgbm = lgb.LGBMClassifier(
        n_estimators=800,
        learning_rate=0.02,
        num_leaves=63,
        min_child_samples=15,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.15,
        reg_lambda=0.15,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )
    lgbm.fit(
        x_train_frame,
        y_train,
        eval_set=[(x_val_frame, y_val)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(stopping_rounds=60, verbose=False)],
    )

    train_probs = lgbm.predict_proba(x_train_frame)[:, 1]
    sample_weights = np.ones(len(y_train), dtype=np.float32)
    sample_weights[(y_train == 0) & (train_probs > 0.35)] = 3.0
    sample_weights[(y_train == 1) & (train_probs < 0.35)] = 2.0
    lgbm.fit(
        x_train_frame,
        y_train,
        sample_weight=sample_weights,
        eval_set=[(x_val_frame, y_val)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(stopping_rounds=60, verbose=False)],
    )

    human_mask = y_train == 0
    iso_train = x_train_scaled[human_mask] if np.any(human_mask) else x_train_scaled
    iso_forest = IsolationForest(
        n_estimators=300,
        contamination=min(max(positive_rate, 0.01), 0.49),
        random_state=42,
    )
    iso_forest.fit(iso_train)

    supervised_val = lgbm.predict_proba(x_val_frame)[:, 1]
    supervised_val, invert_scores = _maybe_invert_scores(y_val, supervised_val)

    calibrator: IsotonicRegression | None = None
    if len(np.unique(y_val)) > 1:
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(supervised_val, y_val)
        supervised_val = np.clip(calibrator.predict(supervised_val), 0.0, 1.0)

    raw_iso_train = -iso_forest.score_samples(x_train_scaled)
    iso_min = float(np.min(raw_iso_train))
    iso_max = float(np.max(raw_iso_train))
    anomaly_val = _anomaly_scores(iso_forest, x_val_scaled, iso_min=iso_min, iso_max=iso_max)

    fusion_mode, iso_blend_weight, fusion = _select_fusion(supervised_val, anomaly_val, y_val)
    final_scores = fusion["scores"]

    calibration = {
        "invert_scores": invert_scores,
        "iso_min": iso_min,
        "iso_max": iso_max,
        "fusion_mode": fusion_mode,
        "iso_blend_weight": iso_blend_weight,
        "validation": {
            "supervised": evaluate_scores(supervised_val, y_val),
            "iso": evaluate_scores(anomaly_val, y_val),
            "selected": fusion["metrics"],
        },
    }
    return scaler, lgbm, iso_forest, calibrator, calibration


def _score_examples(
    examples,
    *,
    scaler: StandardScaler,
    lgbm: lgb.LGBMClassifier,
    iso_forest: IsolationForest,
    calibrator: IsotonicRegression | None,
    calibration: dict,
) -> np.ndarray:
    features = np.vstack([example.feature_row for example in examples])
    scaled = scaler.transform(features)
    frame = pd.DataFrame(scaled, columns=FEATURE_NAMES)
    supervised = lgbm.predict_proba(frame)[:, 1]
    if calibration["invert_scores"]:
        supervised = 1.0 - supervised
    if calibrator is not None:
        supervised = np.clip(calibrator.predict(supervised), 0.0, 1.0)
    if calibration["fusion_mode"] == "supervised":
        return supervised
    anomaly = _anomaly_scores(
        iso_forest,
        scaled,
        iso_min=float(calibration["iso_min"]),
        iso_max=float(calibration["iso_max"]),
    )
    return _fuse_scores(
        supervised,
        anomaly,
        fusion_mode=str(calibration["fusion_mode"]),
        iso_blend_weight=float(calibration["iso_blend_weight"]),
    )


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
        default=90,
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
        default=2,
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

    print("Train summary:", json.dumps(summarize_examples(train_examples), indent=2))
    print("Validation summary:", json.dumps(summarize_examples(val_examples), indent=2))

    scaler, lgbm, iso_forest, calibrator, calibration = _fit_hybrid(
        x_train,
        y_train,
        x_val,
        y_val,
    )

    per_date_metrics: dict[str, dict] = {}
    for source_date in sorted({example.source_date for example in val_examples}):
        date_examples = [example for example in val_examples if example.source_date == source_date]
        date_labels = np.asarray([example.label for example in date_examples], dtype=np.int32)
        date_scores = _score_examples(
            date_examples,
            scaler=scaler,
            lgbm=lgbm,
            iso_forest=iso_forest,
            calibrator=calibrator,
            calibration=calibration,
        )
        per_date_metrics[source_date] = evaluate_scores(date_scores, date_labels)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_name": "poker44-hybrid-lgbm-iso",
        "model_version": "3",
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
        "calibrator": calibrator,
        "invert_scores": calibration["invert_scores"],
        "iso_min": calibration["iso_min"],
        "iso_max": calibration["iso_max"],
        "fusion_mode": calibration["fusion_mode"],
        "iso_blend_weight": calibration["iso_blend_weight"],
        "metadata": metadata,
    }
    joblib.dump(artifact, args.output)

    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved model to {args.output}")
    print(f"Saved metadata to {summary_path}")
    print(
        "Validation selected metrics:",
        json.dumps(calibration["validation"]["selected"], indent=2),
    )
    print("Per-date validation metrics:", json.dumps(per_date_metrics, indent=2))


if __name__ == "__main__":
    main()
