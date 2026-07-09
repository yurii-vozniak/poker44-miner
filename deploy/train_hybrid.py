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
from deploy.features import FEATURE_NAMES, _heuristic_score
from deploy.iso_calibration import fit_iso_calibration, iso_bot_probability


def _maybe_invert_scores(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, bool]:
    metrics = evaluate_scores(scores, y_true)
    roc_auc = metrics.get("roc_auc")
    if roc_auc is None or roc_auc >= 0.5:
        return scores, False
    return 1.0 - scores, True


def _hand_boost_scores(examples) -> np.ndarray:
    boosts = []
    for example in examples:
        if not example.chunk:
            boosts.append(0.0)
            continue
        scores = [_heuristic_score(hand) for hand in example.chunk]
        boosts.append(float(np.percentile(scores, 75)))
    return np.asarray(boosts, dtype=np.float64)


def _fuse_scores(
    supervised: np.ndarray,
    anomaly: np.ndarray,
    *,
    fusion_mode: str,
    iso_blend_weight: float,
    hand_boost: np.ndarray | None = None,
    hand_boost_weight: float = 0.0,
) -> np.ndarray:
    if fusion_mode == "blend":
        fused = supervised + iso_blend_weight * anomaly * (1.0 - supervised)
    elif fusion_mode == "supervised":
        fused = supervised
    else:
        fused = np.maximum(supervised, anomaly)
    if hand_boost is not None and hand_boost_weight > 0:
        fused = np.clip(fused + hand_boost_weight * hand_boost * (1.0 - fused), 0.0, 1.0)
    return fused


def _select_fusion(
    supervised: np.ndarray,
    anomaly: np.ndarray,
    y_true: np.ndarray,
    hand_boost: np.ndarray,
) -> tuple[str, float, float, dict]:
    candidates: list[tuple[str, float, float, np.ndarray]] = []
    candidates.append(("supervised", 0.0, 0.0, supervised.copy()))
    candidates.append(("max", 0.0, 0.0, np.maximum(supervised, anomaly)))
    for weight in (0.05, 0.10, 0.15, 0.20, 0.25):
        candidates.append(
            (
                "blend",
                weight,
                0.0,
                _fuse_scores(
                    supervised,
                    anomaly,
                    fusion_mode="blend",
                    iso_blend_weight=weight,
                ),
            )
        )
    for boost_weight in (0.0, 0.08, 0.12, 0.16):
        for mode, blend_weight, _, base_scores in list(candidates):
            if mode == "blend" and blend_weight not in {0.0, 0.10}:
                continue
            if mode == "supervised" and boost_weight == 0.0:
                continue
            candidates.append(
                (
                    mode,
                    blend_weight,
                    boost_weight,
                    _fuse_scores(
                        supervised,
                        anomaly,
                        fusion_mode=mode,
                        iso_blend_weight=blend_weight,
                        hand_boost=hand_boost,
                        hand_boost_weight=boost_weight,
                    ),
                )
            )

    best_mode = "max"
    best_blend = 0.0
    best_boost = 0.12
    best_metrics: dict = {}
    best_reward = -1.0

    for mode, blend_weight, boost_weight, scores in candidates:
        metrics = evaluate_scores(scores, y_true)
        reward = float(metrics.get("reward") or -1.0)
        if reward > best_reward:
            best_reward = reward
            best_mode = mode
            best_blend = blend_weight
            best_boost = boost_weight
            best_metrics = metrics

    return best_mode, best_blend, best_boost, {
        "fusion_mode": best_mode,
        "iso_blend_weight": best_blend,
        "hand_boost_weight": best_boost,
        "metrics": best_metrics,
    }


def _fit_hybrid(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    val_examples,
) -> tuple[StandardScaler, lgb.LGBMClassifier, IsolationForest, IsotonicRegression | None, dict]:
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_train_frame = pd.DataFrame(x_train_scaled, columns=FEATURE_NAMES)
    x_val_frame = pd.DataFrame(x_val_scaled, columns=FEATURE_NAMES)

    positive_rate = float(np.mean(y_train)) if y_train.size else 0.1
    lgbm = lgb.LGBMClassifier(
        n_estimators=900,
        learning_rate=0.02,
        num_leaves=63,
        min_child_samples=12,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_alpha=0.2,
        reg_lambda=0.2,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )
    lgbm.fit(
        x_train_frame,
        y_train,
        eval_set=[(x_val_frame, y_val)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(stopping_rounds=80, verbose=False)],
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
        callbacks=[lgb.early_stopping(stopping_rounds=80, verbose=False)],
    )

    human_mask = y_train == 0
    iso_train = x_train_scaled[human_mask] if np.any(human_mask) else x_train_scaled
    iso_forest = IsolationForest(
        n_estimators=400,
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

    iso_calibration = fit_iso_calibration(iso_forest, iso_train)
    anomaly_val = iso_bot_probability(iso_forest.score_samples(x_val_scaled), iso_calibration)
    hand_boost = _hand_boost_scores(val_examples)
    fusion_mode, iso_blend_weight, hand_boost_weight, fusion = _select_fusion(
        supervised_val,
        anomaly_val,
        y_val,
        hand_boost,
    )

    calibration = {
        "invert_scores": invert_scores,
        "iso_calibration": iso_calibration.to_dict(),
        "fusion_mode": fusion_mode,
        "iso_blend_weight": iso_blend_weight,
        "hand_boost_weight": hand_boost_weight,
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
    if calibration["fusion_mode"] == "supervised" and calibration["hand_boost_weight"] <= 0:
        return supervised
    from deploy.iso_calibration import IsoCalibration

    iso_cal = IsoCalibration.from_dict(calibration["iso_calibration"])
    anomaly = iso_bot_probability(iso_forest.score_samples(scaled), iso_cal)
    hand_boost = _hand_boost_scores(examples)
    return _fuse_scores(
        supervised,
        anomaly,
        fusion_mode=str(calibration["fusion_mode"]),
        iso_blend_weight=float(calibration["iso_blend_weight"]),
        hand_boost=hand_boost,
        hand_boost_weight=float(calibration["hand_boost_weight"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/hybrid.joblib"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/benchmark"))
    parser.add_argument("--dates", type=int, default=90)
    parser.add_argument("--source-dates", nargs="*", default=None)
    parser.add_argument("--holdout-dates", type=int, default=2)
    parser.add_argument("--max-chunks-per-date", type=int, default=None)
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args()

    client = BenchmarkClient()
    source_dates = list(args.source_dates or client.list_source_dates()[: args.dates])
    if len(source_dates) <= args.holdout_dates:
        raise RuntimeError(
            f"Need more than {args.holdout_dates} source dates; got {len(source_dates)}."
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
        val_examples,
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
        "model_version": "4",
        "framework": "lightgbm+sklearn",
        "feature_names": FEATURE_NAMES,
        "feature_count": len(FEATURE_NAMES),
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
        "iso_calibration": calibration["iso_calibration"],
        "fusion_mode": calibration["fusion_mode"],
        "iso_blend_weight": calibration["iso_blend_weight"],
        "hand_boost_weight": calibration["hand_boost_weight"],
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
