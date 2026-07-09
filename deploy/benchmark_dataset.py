"""Download, cache, and split Poker44 benchmark releases by source date."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from deploy.benchmark_client import BenchmarkClient
from deploy.features import chunks_to_matrix
import numpy as np


@dataclass(frozen=True)
class TrainingExample:
    source_date: str
    chunk_id: str
    chunk_hash: str
    split: str
    label: int
    chunk: list[dict]
    feature_row: np.ndarray


def _record_cache_path(cache_dir: Path, source_date: str, chunk_id: str) -> Path:
    return cache_dir / source_date / f"{chunk_id}.json"


def download_release(
    client: BenchmarkClient,
    source_date: str,
    *,
    cache_dir: Path,
    max_chunks: int | None = None,
    refresh: bool = False,
) -> list[dict[str, Any]]:
    """Download one release date, caching each chunk record by ``chunkId``."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for record in client.iter_chunks(source_date=source_date, max_chunks=max_chunks):
        chunk_id = str(record.get("chunkId") or "")
        if not chunk_id:
            continue

        cache_path = _record_cache_path(cache_dir, source_date, chunk_id)
        if cache_path.is_file() and not refresh:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            records.append(cached)
            continue

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(record, ensure_ascii=True), encoding="utf-8")
        records.append(record)

    return records


def download_releases(
    client: BenchmarkClient,
    source_dates: list[str],
    *,
    cache_dir: Path,
    max_chunks_per_date: int | None = None,
    refresh: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    downloaded: dict[str, list[dict[str, Any]]] = {}
    for source_date in source_dates:
        downloaded[source_date] = download_release(
            client,
            source_date,
            cache_dir=cache_dir,
            max_chunks=max_chunks_per_date,
            refresh=refresh,
        )
    return downloaded


def iter_training_examples(records_by_date: dict[str, list[dict[str, Any]]]) -> Iterator[TrainingExample]:
    for source_date, records in sorted(records_by_date.items()):
        for record in records:
            model_inputs = record.get("chunks") or []
            ground_truth = record.get("groundTruth") or []
            if not model_inputs or not ground_truth:
                continue
            if len(model_inputs) != len(ground_truth):
                continue

            for chunk_group, label in zip(model_inputs, ground_truth):
                if not chunk_group:
                    continue
                feature_row = chunks_to_matrix([chunk_group], for_training=True)[0]
                yield TrainingExample(
                    source_date=source_date,
                    chunk_id=str(record.get("chunkId") or ""),
                    chunk_hash=str(record.get("chunkHash") or ""),
                    split=str(record.get("split") or ""),
                    label=int(label),
                    chunk=chunk_group,
                    feature_row=feature_row,
                )


def split_examples_by_date(
    examples: list[TrainingExample],
    *,
    holdout_dates: int = 1,
) -> tuple[list[TrainingExample], list[TrainingExample]]:
    """Hold out the most recent ``holdout_dates`` release dates for validation."""
    if holdout_dates <= 0 or not examples:
        return examples, []

    unique_dates = sorted({example.source_date for example in examples})
    if len(unique_dates) <= holdout_dates:
        holdout = set(unique_dates[-1:])
    else:
        holdout = set(unique_dates[-holdout_dates:])

    train_rows = [example for example in examples if example.source_date not in holdout]
    val_rows = [example for example in examples if example.source_date in holdout]
    return train_rows, val_rows


def examples_to_arrays(
    examples: list[TrainingExample],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    if not examples:
        raise RuntimeError("No training examples available.")

    features = np.vstack([example.feature_row for example in examples])
    labels = np.asarray([example.label for example in examples], dtype=np.int32)
    metadata = [
        {
            "sourceDate": example.source_date,
            "chunkId": example.chunk_id,
            "chunkHash": example.chunk_hash,
            "split": example.split,
        }
        for example in examples
    ]
    return features, labels, metadata


def summarize_examples(examples: list[TrainingExample]) -> dict[str, Any]:
    if not examples:
        return {"rows": 0, "positive_rate": 0.0, "source_dates": []}

    labels = np.asarray([example.label for example in examples], dtype=np.int32)
    return {
        "rows": int(labels.size),
        "positive_rate": float(labels.mean()),
        "source_dates": sorted({example.source_date for example in examples}),
    }
