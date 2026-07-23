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

from deploy.inference_postprocess import finalize_batch_scores
from deploy.live_rank_fusion import apply_batch_ensemble_fusion
from deploy.benchmark_client import BenchmarkClient
from deploy.benchmark_dataset import (
    download_releases,
    examples_to_arrays,
    iter_training_examples,
    split_examples_by_date,
    summarize_examples,
)
from deploy.eval_metrics import evaluate_scores, windowed_reward
from deploy.stability_metrics import (
    format_stability_report,
    meets_stability_floor,
    per_date_batched_rewards,
    stability_selection_reward,
)
from deploy.features import FEATURE_NAMES, HAND_KEYS, hand_features
from deploy.iso_calibration import IsoCalibration, fit_iso_calibration, iso_bot_probability
from poker44.score.scoring import reward
from sklearn.ensemble import IsolationForest

DEFAULT_MODEL_VERSION = "18"
STABILITY_FLOOR = 0.55
MAX_HUMAN_FPR = 0.05


def _recency_weights(examples, *, half_life_days: float = 14.0) -> np.ndarray:
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
    val_examples,
    *,
    hand_boost_weight: float = 0.0,
    rank_blend: float | None = None,
    adaptive_rank: bool = True,
    max_pos_frac: float | None = None,
    adaptive_max_pos_frac: bool = True,
    batch_size: int = 100,
    n_trials: int = 8,
    seed: int = 42,
    iso_scores: np.ndarray | None = None,
    hand_scores: np.ndarray | None = None,
    hand_mix_weight: float = 0.0,
    live_rank_weight: float = 0.0,
) -> float | None:
    labels = np.asarray(y_true, dtype=int)
    values = np.asarray(scores, dtype=float)
    if labels.size < 20 or len(set(labels.tolist())) < 2:
        return None

    chunks = [example.chunk for example in val_examples]
    rng = np.random.default_rng(seed)
    rewards: list[float] = []
    for _ in range(n_trials):
        order = rng.permutation(labels.size)
        batch_rewards: list[float] = []
        for start in range(0, labels.size, batch_size):
            part = order[start : start + batch_size]
            if part.size < 20:
                continue
            batch_chunks = [chunks[index] for index in part]
            batch_base = values[part]
            batch_iso = (
                np.asarray(iso_scores, dtype=np.float64)[part]
                if iso_scores is not None and iso_scores.size == values.size
                else np.zeros(part.size, dtype=np.float64)
            )
            batch_hand = (
                np.asarray(hand_scores, dtype=np.float64)[part]
                if hand_scores is not None and hand_scores.size == values.size
                else np.zeros(part.size, dtype=np.float64)
            )
            batch_base = apply_batch_ensemble_fusion(
                batch_base,
                batch_chunks,
                iso_scores=batch_iso,
                hand_scores=batch_hand,
                hand_mix_weight=hand_mix_weight,
                live_rank_weight=live_rank_weight,
            )
            batch_scores = finalize_batch_scores(
                batch_base,
                batch_chunks,
                hand_boost_weight=hand_boost_weight,
                rank_blend=rank_blend,
                adaptive_rank=adaptive_rank,
                max_pos_frac=max_pos_frac,
                adaptive_max_pos_frac=adaptive_max_pos_frac,
            )
            _, metrics = reward(batch_scores, labels[part])
            batch_rewards.append(float(metrics["reward"]))
        if batch_rewards:
            rewards.append(float(np.mean(batch_rewards)))
    return float(np.mean(rewards)) if rewards else None


def _selection_reward(
    scores: np.ndarray,
    y_true: np.ndarray,
    val_examples,
    *,
    hand_boost_weight: float = 0.0,
    rank_blend: float | None = None,
) -> float:
    metrics = evaluate_scores(scores, y_true)
    flat_reward = float(metrics.get("reward") or -1.0)
    window_reward = windowed_reward(scores, y_true, window_size=200, n_trials=8)
    if window_reward is None:
        window_reward = flat_reward
    batch_reward = _batched_window_reward(
        scores,
        y_true,
        val_examples,
        hand_boost_weight=hand_boost_weight,
        rank_blend=rank_blend,
    )
    if batch_reward is None:
        batch_reward = flat_reward
    per_date = per_date_batched_rewards(
        scores,
        y_true,
        val_examples,
        hand_boost_weight=hand_boost_weight,
        rank_blend=rank_blend,
        adaptive_rank=True,
    )
    return stability_selection_reward(
        per_date,
        floor=STABILITY_FLOOR,
        batch_mean=batch_reward,
    )


def _hand_aggregate_scores(
    examples,
    hand_probs: np.ndarray,
    *,
    mode: str = "p90",
) -> np.ndarray:
    aggregated: list[float] = []
    offset = 0
    for example in examples:
        hand_count = len(example.chunk or [])
        if hand_count <= 0:
            aggregated.append(0.0)
            continue
        chunk_probs = hand_probs[offset : offset + hand_count]
        offset += hand_count
        if mode == "max":
            aggregated.append(float(np.max(chunk_probs)))
        elif mode == "p75":
            aggregated.append(float(np.percentile(chunk_probs, 75)))
        else:
            aggregated.append(float(np.percentile(chunk_probs, 90)))
    return np.asarray(aggregated, dtype=np.float64)


def _fit_hand_lgbm(train_examples, val_examples, *, train_weights: np.ndarray | None = None):
    x_train_rows: list[np.ndarray] = []
    y_train: list[int] = []
    hand_weights: list[float] = []
    for index, example in enumerate(train_examples):
        chunk_weight = float(train_weights[index]) if train_weights is not None else 1.0
        for hand in example.chunk or []:
            x_train_rows.append(hand_features(hand, for_training=True))
            y_train.append(int(example.label))
            hand_weights.append(chunk_weight)

    if not x_train_rows or len(set(y_train)) < 2:
        return None, None

    x_train = np.vstack(x_train_rows)
    y_train_arr = np.asarray(y_train, dtype=np.int32)
    x_train_frame = pd.DataFrame(x_train, columns=HAND_KEYS)

    x_val_rows: list[np.ndarray] = []
    y_val: list[int] = []
    for example in val_examples:
        for hand in example.chunk or []:
            x_val_rows.append(hand_features(hand, for_training=True))
            y_val.append(int(example.label))
    x_val = np.vstack(x_val_rows) if x_val_rows else x_train[:1]
    y_val_arr = np.asarray(y_val if y_val else y_train[:1], dtype=np.int32)
    x_val_frame = pd.DataFrame(x_val, columns=HAND_KEYS)

    hand_lgbm = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.3,
        reg_lambda=0.3,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )
    hand_lgbm.fit(
        x_train_frame,
        y_train_arr,
        sample_weight=np.asarray(hand_weights, dtype=np.float32),
        eval_set=[(x_val_frame, y_val_arr)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(stopping_rounds=60, verbose=False)],
    )

    val_probs = hand_lgbm.predict_proba(x_val_frame)[:, 1]
    hand_calibrator: IsotonicRegression | None = None
    if len(np.unique(y_val_arr)) > 1:
        hand_calibrator = IsotonicRegression(out_of_bounds="clip")
        hand_calibrator.fit(val_probs, y_val_arr)
    return hand_lgbm, hand_calibrator


def _hand_probs_for_examples(
    examples,
    *,
    hand_lgbm: lgb.LGBMClassifier,
    hand_calibrator: IsotonicRegression | None,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for example in examples:
        for hand in example.chunk or []:
            rows.append(hand_features(hand, for_training=True))
    if not rows:
        return np.zeros(0, dtype=np.float64)
    frame = pd.DataFrame(np.vstack(rows), columns=HAND_KEYS)
    probs = hand_lgbm.predict_proba(frame)[:, 1]
    if hand_calibrator is not None:
        probs = np.clip(hand_calibrator.predict(probs), 0.0, 1.0)
    return probs


def _fuse_validation_scores(
    supervised: np.ndarray,
    anomaly: np.ndarray,
    hand_chunk: np.ndarray,
    *,
    fusion_mode: str,
    iso_blend_weight: float,
    hand_mix_weight: float,
) -> np.ndarray:
    if fusion_mode == "blend":
        fused = supervised + iso_blend_weight * anomaly * (1.0 - supervised)
    elif fusion_mode == "supervised":
        fused = supervised
    else:
        fused = np.maximum(supervised, anomaly)
    if hand_mix_weight > 0.0:
        fused = np.clip(np.maximum(fused, hand_mix_weight * hand_chunk), 0.0, 1.0)
    return fused


def _tune_fusion(
    val_scores: np.ndarray,
    val_anomaly: np.ndarray,
    val_hand_chunk: np.ndarray,
    y_val: np.ndarray,
    val_examples,
) -> dict[str, float | str | None]:
    best: dict[str, float | str | None] = {
        "selection_reward": -1.0,
        "fusion_mode": "max",
        "iso_blend_weight": 0.0,
        "hand_mix_weight": 0.0,
        "hand_boost_weight": 0.0,
        "rank_blend": 0.25,
    }
    for fusion_mode in ("max", "blend"):
        iso_weights = (0.0, 0.15, 0.25) if fusion_mode == "blend" else (0.0,)
        for iso_w in iso_weights:
            for hand_w in (0.0, 0.12, 0.18, 0.24):
                fused = _fuse_validation_scores(
                    val_scores,
                    val_anomaly,
                    val_hand_chunk,
                    fusion_mode=fusion_mode,
                    iso_blend_weight=iso_w,
                    hand_mix_weight=hand_w,
                )
                for hand_boost_w in (0.10, 0.14, 0.18):
                    for rank_blend in (0.30, 0.40, 0.50, 0.60):
                        selection = _selection_reward(
                            fused,
                            y_val,
                            val_examples,
                            hand_boost_weight=hand_boost_w,
                            rank_blend=rank_blend,
                        )
                        if selection > float(best["selection_reward"]):
                            best = {
                                "selection_reward": selection,
                                "fusion_mode": fusion_mode,
                                "iso_blend_weight": iso_w,
                                "hand_mix_weight": hand_w,
                                "hand_boost_weight": hand_boost_w,
                                "rank_blend": rank_blend,
                            }
    return best


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
    parser.add_argument("--dates", type=int, default=37)
    parser.add_argument("--holdout-dates", type=int, default=10)
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

    human_mask = y_train == 0
    iso_train = x_train_scaled[human_mask] if np.any(human_mask) else x_train_scaled
    iso_forest = IsolationForest(
        n_estimators=300,
        contamination=min(max(float(np.mean(y_train)), 0.01), 0.49),
        random_state=42,
        n_jobs=-1,
    )
    iso_forest.fit(iso_train)
    iso_calibration = fit_iso_calibration(iso_forest, iso_train)
    iso_calibration_dict = iso_calibration.to_dict()
    val_anomaly = iso_bot_probability(iso_forest.score_samples(x_val_scaled), iso_calibration)

    hand_lgbm, hand_calibrator = _fit_hand_lgbm(
        train_examples,
        val_examples,
        train_weights=sample_weight,
    )
    val_hand_chunk = np.zeros(len(val_examples), dtype=np.float64)
    if hand_lgbm is not None:
        hand_probs = _hand_probs_for_examples(
            val_examples,
            hand_lgbm=hand_lgbm,
            hand_calibrator=hand_calibrator,
        )
        val_hand_chunk = _hand_aggregate_scores(val_examples, hand_probs, mode="p90")

    fusion = _tune_fusion(val_scores, val_anomaly, val_hand_chunk, y_val, val_examples)
    fused_val = _fuse_validation_scores(
        val_scores,
        val_anomaly,
        val_hand_chunk,
        fusion_mode=str(fusion["fusion_mode"]),
        iso_blend_weight=float(fusion["iso_blend_weight"]),
        hand_mix_weight=float(fusion["hand_mix_weight"]),
    )
    selection = float(fusion["selection_reward"])
    per_date_report = per_date_batched_rewards(
        fused_val,
        y_val,
        val_examples,
        hand_boost_weight=float(fusion["hand_boost_weight"]),
        rank_blend=float(fusion["rank_blend"]) if fusion["rank_blend"] is not None else None,
        adaptive_rank=True,
    )
    metrics = evaluate_scores(fused_val, y_val)
    print("Fusion config:", json.dumps(fusion, indent=2))
    print("Stability report:", json.dumps(format_stability_report(per_date_report), indent=2))
    print("Validation metrics:", json.dumps(metrics, indent=2))
    print(f"Selection reward: {selection:.4f}")

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_name": "poker44-stacked-ensemble",
        "model_version": DEFAULT_MODEL_VERSION,
        "framework": "lightgbm+sklearn-stack+iso+hand",
        "feature_count": len(FEATURE_NAMES),
        "base_models": [name for name, _ in base_models],
        "validation": metrics,
        "selection_reward": selection,
        "fusion": fusion,
        "stability_report": format_stability_report(per_date_report),
        "iso_min": float(np.min(iso_forest.score_samples(iso_train))),
        "iso_span": max(
            float(np.max(iso_forest.score_samples(iso_train)))
            - float(np.min(iso_forest.score_samples(iso_train))),
            1e-8,
        ),
        "hand_aggregate_mode": "p90",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model_type": "stacked",
            "scaler": scaler,
            "base_models": base_models,
            "meta": meta,
            "calibrator": calibrator,
            "iso_forest": iso_forest,
            "iso_calibration": iso_calibration_dict,
            "hand_lgbm": hand_lgbm,
            "hand_calibrator": hand_calibrator,
            "hand_aggregate_mode": "p90",
            "hand_mix_weight": float(fusion["hand_mix_weight"]),
            "hand_boost_weight": float(fusion["hand_boost_weight"]),
            "rank_blend": fusion["rank_blend"],
            "adaptive_rank": True,
            "iso_blend_weight": float(fusion["iso_blend_weight"]),
            "fusion_mode": str(fusion["fusion_mode"]),
            "metadata": metadata,
        },
        args.output,
    )
    meta_path = args.output.with_suffix(".json")
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved stacked model to {args.output}")


if __name__ == "__main__":
    main()
