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
from deploy.batch_calibration import apply_batch_calibration
from deploy.eval_metrics import evaluate_scores, windowed_reward
from deploy.features import FEATURE_NAMES, HAND_KEYS, _heuristic_score, hand_features
from deploy.iso_calibration import fit_iso_calibration, iso_bot_probability
from poker44.score.scoring import reward

DEFAULT_MODEL_VERSION = "12"
SELECTION_WINDOW_SIZE = 200
MAX_HUMAN_FPR = 0.05


def _maybe_invert_scores(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, bool]:
    metrics = evaluate_scores(scores, y_true)
    roc_auc = metrics.get("roc_auc")
    if roc_auc is None or roc_auc >= 0.5:
        return scores, False
    return 1.0 - scores, True


def _recency_weights(examples, *, half_life_days: float = 21.0) -> np.ndarray:
    dates = sorted({example.source_date for example in examples})
    date_rank = {source_date: index for index, source_date in enumerate(dates)}
    max_rank = max(len(dates) - 1, 1)
    weights = np.ones(len(examples), dtype=np.float32)
    for index, example in enumerate(examples):
        age_days = max_rank - date_rank[example.source_date]
        weights[index] = 0.5 ** (age_days / half_life_days)
    return weights


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


def _hand_aggregate_scores(
    examples,
    hand_probs: np.ndarray,
    *,
    mode: str,
) -> np.ndarray:
    """Pool per-hand probabilities into one score per chunk."""
    if not examples:
        return np.asarray([], dtype=np.float64)
    if hand_probs.size == 0:
        return np.zeros(len(examples), dtype=np.float64)

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


def _fit_hand_lgbm(
    train_examples,
    val_examples,
    *,
    train_weights: np.ndarray | None = None,
) -> tuple[lgb.LGBMClassifier | None, IsotonicRegression | None, dict]:
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
        return None, None, {"hand_aggregate_mode": "p90", "hand_mix_weight": 0.0}

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
        val_probs = np.clip(hand_calibrator.predict(val_probs), 0.0, 1.0)

    hand_info = {
        "hand_aggregate_mode": "p90",
        "hand_mix_weight": 0.0,
        "validation": evaluate_scores(
            _hand_aggregate_scores(val_examples, val_probs, mode="p90"),
            np.asarray([example.label for example in val_examples], dtype=np.int32),
        ),
    }
    return hand_lgbm, hand_calibrator, hand_info


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
        return np.asarray([], dtype=np.float64)
    frame = pd.DataFrame(np.vstack(rows), columns=HAND_KEYS)
    probs = hand_lgbm.predict_proba(frame)[:, 1]
    if hand_calibrator is not None:
        probs = np.clip(hand_calibrator.predict(probs), 0.0, 1.0)
    return probs


def _per_date_rewards(
    scores: np.ndarray,
    y_true: np.ndarray,
    val_examples,
) -> list[float]:
    rewards: list[float] = []
    for source_date in sorted({example.source_date for example in val_examples}):
        indices = [
            index
            for index, example in enumerate(val_examples)
            if example.source_date == source_date
        ]
        if not indices:
            continue
        date_metrics = evaluate_scores(scores[indices], y_true[indices])
        rewards.append(float(date_metrics.get("reward") or -1.0))
    return rewards


def _max_human_fpr_by_date(
    scores: np.ndarray,
    y_true: np.ndarray,
    val_examples,
) -> float:
    """Worst per-date human FPR at the reward-optimal operating point."""
    max_fpr = 0.0
    for source_date in sorted({example.source_date for example in val_examples}):
        indices = [
            index
            for index, example in enumerate(val_examples)
            if example.source_date == source_date
        ]
        if not indices:
            continue
        date_metrics = evaluate_scores(scores[indices], y_true[indices])
        max_fpr = max(max_fpr, float(date_metrics.get("fpr_at_recall") or 1.0))
    return max_fpr


def _passes_human_fpr_guard(
    scores: np.ndarray,
    y_true: np.ndarray,
    val_examples,
    *,
    max_fpr: float = MAX_HUMAN_FPR,
) -> bool:
    return _max_human_fpr_by_date(scores, y_true, val_examples) <= max_fpr + 1e-9


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


def _selection_reward(
    scores: np.ndarray,
    y_true: np.ndarray,
    val_examples,
) -> float:
    """Stability-first objective aligned with v2.2 multi-round competition."""
    metrics = evaluate_scores(scores, y_true)
    flat_reward = float(metrics.get("reward") or -1.0)
    window_reward = windowed_reward(
        scores,
        y_true,
        window_size=SELECTION_WINDOW_SIZE,
        n_trials=8,
    )
    if window_reward is None:
        window_reward = flat_reward
    batch_reward = _batched_window_reward(scores, y_true)
    if batch_reward is None:
        batch_reward = flat_reward

    per_date_rewards = _per_date_rewards(scores, y_true, val_examples)
    mean_date_reward = flat_reward
    min_date_reward = flat_reward
    if per_date_rewards:
        mean_date_reward = float(np.mean(per_date_rewards))
        min_date_reward = float(np.min(per_date_rewards))

    return (
        0.30 * flat_reward
        + 0.20 * window_reward
        + 0.25 * batch_reward
        + 0.15 * mean_date_reward
        + 0.10 * min_date_reward
    )


def _select_fusion(
    supervised: np.ndarray,
    anomaly: np.ndarray,
    y_true: np.ndarray,
    hand_boost: np.ndarray,
    val_examples,
    hand_aggregate: np.ndarray | None = None,
) -> tuple[str, float, float, float, str, dict]:
    candidates: list[tuple[str, float, float, float, str, np.ndarray]] = []
    candidates.append(("supervised", 0.0, 0.0, 0.0, "none", supervised.copy()))
    candidates.append(("max", 0.0, 0.0, 0.0, "none", np.maximum(supervised, anomaly)))
    for weight in (0.05, 0.10, 0.12, 0.15, 0.20):
        candidates.append(
            (
                "blend",
                weight,
                0.0,
                0.0,
                "none",
                _fuse_scores(
                    supervised,
                    anomaly,
                    fusion_mode="blend",
                    iso_blend_weight=weight,
                ),
            )
        )

    base_candidates = list(candidates)
    for boost_weight in (0.0, 0.08, 0.12, 0.16):
        for mode, blend_weight, _, _, _, base_scores in base_candidates:
            if mode == "blend" and blend_weight not in {0.0, 0.10, 0.12}:
                continue
            if mode == "supervised" and boost_weight == 0.0:
                continue
            candidates.append(
                (
                    mode,
                    blend_weight,
                    boost_weight,
                    0.0,
                    "none",
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

    if hand_aggregate is not None and hand_aggregate.size:
        flat_hand_reward = float(
            (evaluate_scores(hand_aggregate, y_true).get("reward") or -1.0)
        )
        if flat_hand_reward >= 0.55:
            enriched: list[tuple[str, float, float, float, str, np.ndarray]] = []
            for mode, blend_weight, boost_weight, _, _, base_scores in candidates:
                for mix_weight in (0.35, 0.55, 0.75):
                    for aggregate_mode in ("p90", "max"):
                        mixed = np.clip(
                            np.maximum(base_scores, mix_weight * hand_aggregate),
                            0.0,
                            1.0,
                        )
                        enriched.append(
                            (
                                mode,
                                blend_weight,
                                boost_weight,
                                mix_weight,
                                aggregate_mode,
                                mixed,
                            )
                        )
            candidates.extend(enriched)

    best_mode = "max"
    best_blend = 0.0
    best_boost = 0.12
    best_hand_mix = 0.0
    best_hand_aggregate = "p90"
    best_metrics: dict = {}
    best_reward = -1.0

    def _consider_candidates(*, enforce_fpr_guard: bool) -> None:
        nonlocal best_mode, best_blend, best_boost, best_hand_mix
        nonlocal best_hand_aggregate, best_metrics, best_reward
        for mode, blend_weight, boost_weight, hand_mix, aggregate_mode, scores in candidates:
            if enforce_fpr_guard and not _passes_human_fpr_guard(scores, y_true, val_examples):
                continue
            selection_reward = _selection_reward(scores, y_true, val_examples)
            metrics = evaluate_scores(scores, y_true)
            if selection_reward > best_reward:
                best_reward = selection_reward
                best_mode = mode
                best_blend = blend_weight
                best_boost = boost_weight
                best_hand_mix = hand_mix
                best_hand_aggregate = aggregate_mode
                best_metrics = metrics

    _consider_candidates(enforce_fpr_guard=True)
    if best_reward < 0.0:
        _consider_candidates(enforce_fpr_guard=False)

    return best_mode, best_blend, best_boost, best_hand_mix, best_hand_aggregate, {
        "fusion_mode": best_mode,
        "iso_blend_weight": best_blend,
        "hand_boost_weight": best_boost,
        "hand_mix_weight": best_hand_mix,
        "hand_aggregate_mode": best_hand_aggregate,
        "metrics": best_metrics,
        "selection_reward": best_reward,
    }


def _fit_hybrid(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    train_examples,
    val_examples,
) -> tuple[
    StandardScaler,
    lgb.LGBMClassifier,
    IsolationForest,
    IsotonicRegression | None,
    lgb.LGBMClassifier | None,
    IsotonicRegression | None,
    dict,
]:
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
    recency_weights = _recency_weights(train_examples)
    sample_weights = recency_weights.astype(np.float32)
    sample_weights[(y_train == 0) & (train_probs > 0.35)] *= 3.0
    sample_weights[(y_train == 1) & (train_probs < 0.35)] *= 2.0
    lgbm.fit(
        x_train_frame,
        y_train,
        sample_weight=sample_weights,
        eval_set=[(x_val_frame, y_val)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(stopping_rounds=80, verbose=False)],
    )

    hand_lgbm, hand_calibrator, hand_info = _fit_hand_lgbm(
        train_examples,
        val_examples,
        train_weights=recency_weights,
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
    hand_aggregate = None
    if hand_lgbm is not None:
        hand_probs = _hand_probs_for_examples(
            val_examples,
            hand_lgbm=hand_lgbm,
            hand_calibrator=hand_calibrator,
        )
        hand_aggregate = _hand_aggregate_scores(val_examples, hand_probs, mode="p90")
    (
        fusion_mode,
        iso_blend_weight,
        hand_boost_weight,
        hand_mix_weight,
        hand_aggregate_mode,
        fusion,
    ) = _select_fusion(
        supervised_val,
        anomaly_val,
        y_val,
        hand_boost,
        val_examples,
        hand_aggregate=hand_aggregate,
    )

    calibration = {
        "invert_scores": invert_scores,
        "iso_calibration": iso_calibration.to_dict(),
        "fusion_mode": fusion_mode,
        "iso_blend_weight": iso_blend_weight,
        "hand_boost_weight": hand_boost_weight,
        "hand_mix_weight": hand_mix_weight,
        "hand_aggregate_mode": hand_aggregate_mode,
        "validation": {
            "supervised": evaluate_scores(supervised_val, y_val),
            "iso": evaluate_scores(anomaly_val, y_val),
            "selected": fusion["metrics"],
            "selection_reward": fusion.get("selection_reward"),
            "hand_model": hand_info.get("validation"),
        },
    }
    return (
        scaler,
        lgbm,
        iso_forest,
        calibrator,
        hand_lgbm,
        hand_calibrator,
        calibration,
    )


def _score_examples(
    examples,
    *,
    scaler: StandardScaler,
    lgbm: lgb.LGBMClassifier,
    iso_forest: IsolationForest,
    calibrator: IsotonicRegression | None,
    calibration: dict,
    hand_lgbm: lgb.LGBMClassifier | None = None,
    hand_calibrator: IsotonicRegression | None = None,
) -> np.ndarray:
    features = np.vstack([example.feature_row for example in examples])
    scaled = scaler.transform(features)
    frame = pd.DataFrame(scaled, columns=FEATURE_NAMES)
    supervised = lgbm.predict_proba(frame)[:, 1]
    if calibration["invert_scores"]:
        supervised = 1.0 - supervised
    if calibrator is not None:
        supervised = np.clip(calibrator.predict(supervised), 0.0, 1.0)
    hand_mix_weight = float(calibration.get("hand_mix_weight") or 0.0)
    if calibration["fusion_mode"] == "supervised" and calibration["hand_boost_weight"] <= 0:
        fused = supervised
    else:
        from deploy.iso_calibration import IsoCalibration

        iso_cal = IsoCalibration.from_dict(calibration["iso_calibration"])
        anomaly = iso_bot_probability(iso_forest.score_samples(scaled), iso_cal)
        hand_boost = _hand_boost_scores(examples)
        fused = _fuse_scores(
            supervised,
            anomaly,
            fusion_mode=str(calibration["fusion_mode"]),
            iso_blend_weight=float(calibration["iso_blend_weight"]),
            hand_boost=hand_boost,
            hand_boost_weight=float(calibration["hand_boost_weight"]),
        )
    if hand_lgbm is not None and hand_mix_weight > 0.0:
        hand_probs = _hand_probs_for_examples(
            examples,
            hand_lgbm=hand_lgbm,
            hand_calibrator=hand_calibrator,
        )
        hand_aggregate = _hand_aggregate_scores(
            examples,
            hand_probs,
            mode=str(calibration.get("hand_aggregate_mode") or "p90"),
        )
        fused = np.clip(np.maximum(fused, hand_mix_weight * hand_aggregate), 0.0, 1.0)
    return fused


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/hybrid.joblib"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/benchmark"))
    parser.add_argument("--dates", type=int, default=37)
    parser.add_argument("--source-dates", nargs="*", default=None)
    parser.add_argument("--holdout-dates", type=int, default=10)
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

    scaler, lgbm, iso_forest, calibrator, hand_lgbm, hand_calibrator, calibration = _fit_hybrid(
        x_train,
        y_train,
        x_val,
        y_val,
        train_examples,
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
            hand_lgbm=hand_lgbm,
            hand_calibrator=hand_calibrator,
        )
        per_date_metrics[source_date] = evaluate_scores(date_scores, date_labels)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_name": "poker44-hybrid-lgbm-iso",
        "model_version": DEFAULT_MODEL_VERSION,
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
        "hand_lgbm": hand_lgbm,
        "hand_calibrator": hand_calibrator,
        "invert_scores": calibration["invert_scores"],
        "iso_calibration": calibration["iso_calibration"],
        "fusion_mode": calibration["fusion_mode"],
        "iso_blend_weight": calibration["iso_blend_weight"],
        "hand_boost_weight": calibration["hand_boost_weight"],
        "hand_mix_weight": calibration.get("hand_mix_weight", 0.0),
        "hand_aggregate_mode": calibration.get("hand_aggregate_mode", "p90"),
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
