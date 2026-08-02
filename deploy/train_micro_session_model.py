#!/usr/bin/env python3
"""Train a calibrated micro-session model on available labeled v4.1 data."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from deploy.micro_session_dataset import (
    LabeledMicroSession,
    discover_labeled_sessions,
    split_by_source_date,
)
from deploy.micro_session_features import FEATURE_NAMES, sessions_to_matrix
from deploy.micro_session_reward import reward
from deploy.synthetic_micro_sessions import make_dataset


def _rows_to_arrays(rows: list[LabeledMicroSession]) -> tuple[np.ndarray, np.ndarray]:
    sessions = [row.session for row in rows]
    labels = np.asarray([row.label for row in rows], dtype=int)
    return sessions_to_matrix(sessions), labels


def _evaluate(model, matrix: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    probs = model.predict_proba(matrix)[:, 1]
    metrics = reward(probs, labels)
    return {
        "reward": round(metrics.reward, 4),
        "ap": round(metrics.average_precision, 4),
        "ap_skill": round(metrics.average_precision_skill, 4),
        "recall_at_fpr_05": round(metrics.recall_at_fpr_05, 4),
        "brier_skill": round(metrics.brier_skill, 4),
        "accuracy": round(metrics.accuracy, 4),
        "auc": round(float(roc_auc_score(labels, probs)), 4),
        "raw_ap": round(float(average_precision_score(labels, probs)), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/micro_session_v1.joblib"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/benchmark"))
    parser.add_argument("--v41-jsonl", type=Path, default=None)
    parser.add_argument("--holdout-dates", type=int, default=3)
    parser.add_argument("--synthetic-sessions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    labeled_rows, dataset_summary = discover_labeled_sessions(
        cache_dir=args.cache_dir,
        v41_jsonl=args.v41_jsonl,
    )
    if not labeled_rows:
        raise RuntimeError(
            "No labeled v4.1 training rows found. Expected data/micro_session_benchmark.jsonl "
            "or cached legacy benchmark under data/benchmark/."
        )

    train_rows, val_rows = split_by_source_date(
        labeled_rows,
        holdout_dates=args.holdout_dates,
    )
    if args.synthetic_sessions > 0:
        synth_sessions, synth_labels = make_dataset(
            n_sessions=args.synthetic_sessions,
            seed=args.seed,
        )
        for session, label in zip(synth_sessions, synth_labels):
            train_rows.append(
                LabeledMicroSession(
                    session=session,
                    label=int(label),
                    source="synthetic-v4.1-prior-v2",
                )
            )

    x_train, y_train = _rows_to_arrays(train_rows)
    x_val, y_val = _rows_to_arrays(val_rows)

    base = GradientBoostingClassifier(
        random_state=args.seed,
        n_estimators=400,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.85,
        min_samples_leaf=8,
    )
    model = CalibratedClassifierCV(base, method="isotonic", cv=3)
    model.fit(x_train, y_train)

    train_metrics = _evaluate(model, x_train, y_train)
    val_metrics = _evaluate(model, x_val, y_val)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_names": FEATURE_NAMES}, args.output)

    manifest = {
        "version": "micro-v1",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "trained_on": {
            **dataset_summary,
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "synthetic_rows": args.synthetic_sessions,
            "holdout_dates": args.holdout_dates,
        },
        "train_metrics": train_metrics,
        "holdout_metrics": val_metrics,
        "caveat": (
            "No public labeled v4.1 corpus exists yet. Primary training data is "
            "legacy chunk groups converted to v4.1 strategic decisions; treat "
            "holdout reward as a proxy, not live WTA performance."
        ),
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(manifest, indent=2))
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
