#!/usr/bin/env python3
"""Download benchmark chunks and train a baseline sklearn detector."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from deploy.benchmark_client import BenchmarkClient
from deploy.features import chunks_to_matrix


def _load_training_rows(
    client: BenchmarkClient,
    source_date: str | None,
    max_chunks: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    feature_rows: list[np.ndarray] = []
    labels: list[int] = []
    chunk_meta: list[dict] = []

    for record in client.iter_chunks(source_date=source_date, max_chunks=max_chunks):
        model_inputs = record.get("chunks") or []
        ground_truth = record.get("groundTruth") or []
        if not model_inputs or not ground_truth:
            continue

        if len(model_inputs) != len(ground_truth):
            continue

        for chunk_group, label in zip(model_inputs, ground_truth):
            if not chunk_group:
                continue
            feature_rows.append(chunks_to_matrix([chunk_group])[0])
            labels.append(int(label))
            chunk_meta.append(
                {
                    "chunkId": record.get("chunkId"),
                    "sourceDate": record.get("sourceDate"),
                    "split": record.get("split"),
                }
            )

    if not feature_rows:
        raise RuntimeError("No training rows downloaded from the benchmark API.")

    return (
        np.vstack(feature_rows),
        np.asarray(labels, dtype=np.int32),
        {
            "source_date": source_date or client.latest_source_date(),
            "rows": len(labels),
            "positive_rate": float(np.mean(labels)),
            "sample_meta": chunk_meta[:5],
        },
    )

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/baseline.joblib"),
        help="Path to write the trained model artifact.",
    )
    parser.add_argument(
        "--source-date",
        default=None,
        help="Benchmark source date (YYYY-MM-DD). Defaults to latest.",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=200,
        help="Maximum number of benchmark chunk records to download.",
    )
    args = parser.parse_args()

    client = BenchmarkClient()
    source_date = args.source_date or client.latest_source_date()
    print(f"Downloading benchmark chunks for source_date={source_date} ...")

    features, labels, download_meta = _load_training_rows(
        client,
        source_date=source_date,
        max_chunks=args.max_chunks,
    )
    print(
        f"Loaded {download_meta['rows']} chunk groups "
        f"(positive_rate={download_meta['positive_rate']:.3f})"
    )

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels if len(set(labels)) > 1 else None,
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)

    metrics: dict[str, float | None] = {}
    invert_scores = False
    if len(set(y_test)) > 1:
        probabilities = pipeline.predict_proba(x_test)[:, 1]
        roc_auc = float(roc_auc_score(y_test, probabilities))
        if roc_auc < 0.5:
            invert_scores = True
            probabilities = 1.0 - probabilities
            roc_auc = 1.0 - roc_auc
        metrics["roc_auc"] = roc_auc
        metrics["average_precision"] = float(average_precision_score(y_test, probabilities))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "pipeline": pipeline,
        "metadata": {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "source_date": source_date,
            "download": download_meta,
            "metrics": metrics,
            "invert_scores": invert_scores,
            "model_name": "poker44-baseline-sklearn",
            "model_version": "1",
            "framework": "scikit-learn",
        },
    }
    joblib.dump(artifact, args.output)

    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(artifact["metadata"], indent=2), encoding="utf-8")
    print(f"Saved model to {args.output}")
    print(f"Saved metadata to {summary_path}")
    if metrics:
        print(f"Validation metrics: {metrics}")


if __name__ == "__main__":
    main()
