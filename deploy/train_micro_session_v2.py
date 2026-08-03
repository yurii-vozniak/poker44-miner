#!/usr/bin/env python3
"""Train micro-v2: HGB + reference fusion tuned for validator reward."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from deploy.micro_session_dataset import (
    LabeledMicroSession,
    discover_labeled_sessions,
    split_by_source_date,
)
from deploy.micro_session_features import FEATURE_NAMES, sessions_to_matrix
from deploy.micro_session_reward import reward
from poker44.miner.config import MinerModelConfig
from poker44.miner.model import ReferenceSessionModel


def _reference_probs(sessions: list[dict]) -> np.ndarray:
    ref = ReferenceSessionModel(
        MinerModelConfig(
            factory="reference",
            version="reference-v2",
            model_path="",
            device="cpu",
            max_sessions_per_request=256,
        )
    )
    return np.asarray(ref.predict(sessions), dtype=np.float64)


def _evaluate_probs(probs: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    metrics = reward(probs, labels)
    return {
        "reward": round(metrics.reward, 4),
        "ap": round(metrics.average_precision, 4),
        "ap_skill": round(metrics.average_precision_skill, 4),
        "recall_at_fpr_05": round(metrics.recall_at_fpr_05, 4),
        "brier_skill": round(metrics.brier_skill, 4),
        "accuracy": round(metrics.accuracy, 4),
        "auc": round(float(roc_auc_score(labels, probs)), 4),
    }


def _fuse(
    ref_probs: np.ndarray,
    ml_probs: np.ndarray,
    *,
    mode: str,
    weight: float,
) -> np.ndarray:
    if mode == "reference":
        return ref_probs.copy()
    if mode == "ml":
        return ml_probs.copy()
    if mode == "blend":
        w = float(weight)
        return np.clip(w * ref_probs + (1.0 - w) * ml_probs, 0.0, 1.0)
    if mode == "max":
        return np.maximum(ref_probs, ml_probs)
    if mode == "ref_boost":
        # Monotone lift: never score below reference, add ML excess when confident.
        return np.clip(ref_probs + float(weight) * np.maximum(0.0, ml_probs - ref_probs), 0.0, 1.0)
    raise ValueError(f"Unknown fusion mode: {mode}")


def _search_fusion(
    ref_probs: np.ndarray,
    ml_probs: np.ndarray,
    labels: np.ndarray,
) -> tuple[str, float, dict[str, float]]:
    best: tuple[float, str, float, dict[str, float]] = (-1.0, "reference", 1.0, {})
    candidates: list[tuple[str, float]] = [("reference", 1.0), ("ml", 0.0)]
    for weight in (0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5):
        candidates.append(("blend", weight))
    for weight in (0.25, 0.5, 0.75, 1.0):
        candidates.append(("ref_boost", weight))
    candidates.append(("max", 0.0))

    for mode, weight in candidates:
        probs = _fuse(ref_probs, ml_probs, mode=mode, weight=weight)
        metrics = _evaluate_probs(probs, labels)
        if metrics["reward"] > best[0]:
            best = (metrics["reward"], mode, weight, metrics)
    return best[1], best[2], best[3]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/micro_session_v2.joblib"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/benchmark"))
    parser.add_argument("--holdout-dates", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows, dataset_summary = discover_labeled_sessions(cache_dir=args.cache_dir)
    if not rows:
        raise RuntimeError("No labeled proxy rows found under data/benchmark/")

    train_rows, val_rows = split_by_source_date(rows, holdout_dates=args.holdout_dates)
    train_sessions = [r.session for r in train_rows]
    val_sessions = [r.session for r in val_rows]
    y_train = np.asarray([r.label for r in train_rows], dtype=int)
    y_val = np.asarray([r.label for r in val_rows], dtype=int)

    x_train = sessions_to_matrix(train_sessions)
    x_val = sessions_to_matrix(val_sessions)
    ref_train = _reference_probs(train_sessions).reshape(-1, 1)
    ref_val = _reference_probs(val_sessions)

    x_train_aug = np.hstack([x_train, ref_train])
    x_val_aug = np.hstack([x_val, ref_val.reshape(-1, 1)])

    base = HistGradientBoostingClassifier(
        random_state=args.seed,
        max_depth=6,
        max_iter=500,
        learning_rate=0.06,
        min_samples_leaf=12,
        l2_regularization=0.5,
    )
    ml_model = CalibratedClassifierCV(base, method="isotonic", cv=3)
    ml_model.fit(x_train_aug, y_train)
    ml_val = ml_model.predict_proba(x_val_aug)[:, 1]

    fusion_mode, fusion_weight, holdout_metrics = _search_fusion(ref_val, ml_val, y_val)

    ref_only = _evaluate_probs(ref_val, y_val)
    ml_only = _evaluate_probs(ml_val, y_val)

    # Retrain ML on all proxy rows for the deployment artifact.
    all_sessions = [r.session for r in rows]
    y_all = np.asarray([r.label for r in rows], dtype=int)
    x_all = sessions_to_matrix(all_sessions)
    ref_all = _reference_probs(all_sessions).reshape(-1, 1)
    x_all_aug = np.hstack([x_all, ref_all])
    deploy_model = CalibratedClassifierCV(
        HistGradientBoostingClassifier(
            random_state=args.seed,
            max_depth=6,
            max_iter=500,
            learning_rate=0.06,
            min_samples_leaf=12,
            l2_regularization=0.5,
        ),
        method="isotonic",
        cv=3,
    )
    deploy_model.fit(x_all_aug, y_all)

    artifact = {
        "ml_model": deploy_model,
        "feature_names": FEATURE_NAMES,
        "reference_feature": True,
        "fusion_mode": fusion_mode,
        "fusion_weight": fusion_weight,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output)

    manifest = {
        "version": "micro-v2",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "trained_on": {
            **dataset_summary,
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "holdout_dates": args.holdout_dates,
        },
        "fusion": {"mode": fusion_mode, "weight": fusion_weight},
        "holdout_metrics": {
            "selected": holdout_metrics,
            "reference_only": ref_only,
            "ml_only": ml_only,
        },
        "live_target": {
            "rank_10_reward": 0.2432,
            "rank_1_reward": 0.3386,
            "note": (
                "Holdout metrics use a train-split ML model only. The saved "
                "artifact retrains ML on all proxy rows for deployment."
            ),
        },
    }

    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
