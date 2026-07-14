#!/usr/bin/env python3
"""Train a stacked tree ensemble with date-group OOF meta learner."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from deploy.batch_calibration import apply_batch_calibration
from deploy.benchmark_client import BenchmarkClient
from deploy.benchmark_dataset import (
    download_releases,
    examples_to_arrays,
    iter_training_examples,
    split_examples_by_date,
    summarize_examples,
)
from deploy.eval_metrics import evaluate_scores, windowed_reward
from deploy.features import FEATURE_NAMES
from poker44.score.scoring import reward

DEFAULT_MODEL_VERSION = "10"
MAX_HUMAN_FPR = 0.05


def _recency_weights(examples, *, half_life_days: float = 21.0) -> np.ndarray:
    dates = sorted({example.source_date for example in examples})
    date_rank = {source_date: index for index, source_date in enumerate(dates)}
    max_rank = max(len(dates) - 1, 1)
    weights = np.ones(len(examples), dtype=np.float32)
    for index, example in enumerate(examples):
        age_days = max_rank - date_rank[example.source_date]
        weights[index] = 0.5 ** (age_days / half_life_days)
    return weights


def _batched_window_reward(
    scores: np.ndarray,
    y_true: np.ndarray,
    *,
    batch_size: int = 100,
    n_trials: int = 8,
    seed: int = 42,
) -> float | None:
    labels = np.asarray(y_true, dtype=int)
    values = np.asarray(scores, dtype=float)
    if labels.size < 20 or len(set(labels.tolist())) < 2:
        return None

    rng = np.random.default_rng(seed)
    rewards: list[float] = []
    for _ in range(n_trials):
        order = rng.permutation(labels.size)
        batch_rewards: list[float] = []
        for start in range(0, labels.size, batch_size):
            part = order[start : start + batch_size]
            if part.size < 20:
                continue
            batch_scores = apply_batch_calibration(values[part])
            _, metrics = reward(batch_scores, labels[part])
            batch_rewards.append(float(metrics["reward"]))
        if batch_rewards:
            rewards.append(float(np.mean(batch_rewards)))
    return float(np.mean(rewards)) if rewards else None


def _selection_reward(scores: np.ndarray, y_true: np.ndarray) -> float:
    metrics = evaluate_scores(scores, y_true)
    flat_reward = float(metrics.get("reward") or -1.0)
    window_reward = windowed_reward(scores, y_true, window_size=200, n_trials=8)
    if window_reward is None:
        window_reward = flat_reward
    batch_reward = _batched_window_reward(scores, y_true)
    if batch_reward is None:
        batch_reward = flat_reward
    return 0.35 * flat_reward + 0.25 * window_reward + 0.40 * batch_reward


def _make_base_specs() -> list[tuple[str, Any]]:
    return [
        (
            "lgbm",
            lgb.LGBMClassifier(
                n_estimators=700,
                learning_rate=0.03,
                num_leaves=63,
                min_child_samples=12,
                subsample=0.85,
                colsample_bytree=0.8,
                reg_alpha=0.2,
                reg_lambda=0.2,
                class_weight="balanced",
                random_state=42,
                verbose=-1,
            ),
        ),
        (
            "extra_trees",
            ExtraTreesClassifier(
                n_estimators=500,
                max_features="sqrt",
                min_samples_leaf=4,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            ),
        ),
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=500,
                max_features="sqrt",
                min_samples_leaf=4,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            ),
        ),
        (
            "hist_gbm",
            HistGradientBoostingClassifier(
                max_depth=8,
                learning_rate=0.05,
                max_iter=400,
                random_state=42,
            ),
        ),
    ]


def _fit_oof_stack(
    x_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
    *,
    sample_weight: np.ndarray | None,
) -> tuple[list[tuple[str, Any]], np.ndarray]:
    specs = _make_base_specs()
    oof = np.zeros((y_train.size, len(specs)), dtype=np.float64)
    n_splits = min(5, len(set(groups.tolist())))
    if n_splits < 2:
        n_splits = 2
    splitter = GroupKFold(n_splits=n_splits)

    for fold, (fit_idx, val_idx) in enumerate(splitter.split(x_train, y_train, groups)):
        x_fit = x_train[fit_idx]
        y_fit = y_train[fit_idx]
        x_val = x_train[val_idx]
        fit_weight = sample_weight[fit_idx] if sample_weight is not None else None
        frame_fit = pd.DataFrame(x_fit, columns=FEATURE_NAMES)
        frame_val = pd.DataFrame(x_val, columns=FEATURE_NAMES)

        for model_idx, (name, prototype) in enumerate(specs):
            model = prototype.__class__(**prototype.get_params())
            fit_kwargs: dict = {}
            if fit_weight is not None and name == "lgbm":
                fit_kwargs["sample_weight"] = fit_weight
            elif fit_weight is not None and hasattr(model, "fit") and "sample_weight" in model.fit.__code__.co_varnames:
                fit_kwargs["sample_weight"] = fit_weight
            model.fit(frame_fit, y_fit, **fit_kwargs)
            oof[val_idx, model_idx] = model.predict_proba(frame_val)[:, 1]

    fitted: list[tuple[str, Any]] = []
    frame_train = pd.DataFrame(x_train, columns=FEATURE_NAMES)
    for name, prototype in specs:
        model = prototype.__class__(**prototype.get_params())
        fit_kwargs = {}
        if sample_weight is not None and name == "lgbm":
            fit_kwargs["sample_weight"] = sample_weight
        model.fit(frame_train, y_train, **fit_kwargs)
        fitted.append((name, model))
    return fitted, oof


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/stacked.joblib"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/benchmark"))
    parser.add_argument("--dates", type=int, default=30)
    parser.add_argument("--holdout-dates", type=int, default=5)
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args()

    client = BenchmarkClient()
    source_dates = client.list_source_dates()[: args.dates]
    if len(source_dates) <= args.holdout_dates:
        raise RuntimeError(f"Need more than {args.holdout_dates} source dates.")

    print(f"Using source dates: {', '.join(source_dates)}")
    records_by_date = download_releases(
        client,
        source_dates,
        cache_dir=args.cache_dir,
        refresh=args.refresh_cache,
    )
    examples = list(iter_training_examples(records_by_date))
    train_examples, val_examples = split_examples_by_date(
        examples,
        holdout_dates=args.holdout_dates,
    )
    if not train_examples or not val_examples:
        raise RuntimeError("Train/validation split produced empty partitions.")

    x_train, y_train, _ = examples_to_arrays(train_examples)
    x_val, y_val, _ = examples_to_arrays(val_examples)
    groups = np.asarray([example.source_date for example in train_examples])
    print("Train summary:", json.dumps(summarize_examples(train_examples), indent=2))
    print("Validation summary:", json.dumps(summarize_examples(val_examples), indent=2))

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    sample_weight = _recency_weights(train_examples)

    base_models, oof = _fit_oof_stack(
        x_train_scaled,
        y_train,
        groups,
        sample_weight=sample_weight,
    )
    meta = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    meta.fit(oof, y_train)

    val_frame = pd.DataFrame(x_val_scaled, columns=FEATURE_NAMES)
    val_base = np.column_stack([model.predict_proba(val_frame)[:, 1] for _, model in base_models])
    val_scores = meta.predict_proba(val_base)[:, 1]

    calibrator: IsotonicRegression | None = None
    if len(np.unique(y_val)) > 1:
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(val_scores, y_val)
        val_scores = np.clip(calibrator.predict(val_scores), 0.0, 1.0)

    metrics = evaluate_scores(val_scores, y_val)
    selection = _selection_reward(val_scores, y_val)
    print("Validation metrics:", json.dumps(metrics, indent=2))
    print(f"Selection reward: {selection:.4f}")

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_name": "poker44-stacked-ensemble",
        "model_version": DEFAULT_MODEL_VERSION,
        "framework": "lightgbm+sklearn-stack",
        "feature_count": len(FEATURE_NAMES),
        "base_models": [name for name, _ in base_models],
        "validation": metrics,
        "selection_reward": selection,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model_type": "stacked",
            "scaler": scaler,
            "base_models": base_models,
            "meta": meta,
            "calibrator": calibrator,
            "metadata": metadata,
        },
        args.output,
    )
    meta_path = args.output.with_suffix(".json")
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved stacked model to {args.output}")


if __name__ == "__main__":
    main()
