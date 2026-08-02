"""Load labeled micro-session training rows from available sources."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from deploy.chunk_to_micro_session import chunk_group_to_micro_session


@dataclass(frozen=True)
class LabeledMicroSession:
    session: dict[str, Any]
    label: int
    source: str
    source_date: str = ""
    split: str = ""


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number} must be a JSON object")
        yield row


def load_v41_jsonl(path: Path) -> list[LabeledMicroSession]:
    rows: list[LabeledMicroSession] = []
    for row in _iter_jsonl(path):
        label = int(row.get("label", -1))
        payload = row.get("payload") or row.get("session") or row.get("item")
        if label not in {0, 1} or not isinstance(payload, dict):
            continue
        rows.append(
            LabeledMicroSession(
                session=payload,
                label=label,
                source=f"v41-jsonl:{path.name}",
                source_date=str(row.get("source_date") or ""),
                split=str(row.get("split") or ""),
            )
        )
    return rows


def load_legacy_chunk_benchmark(cache_dir: Path) -> list[LabeledMicroSession]:
    rows: list[LabeledMicroSession] = []
    for source_date_dir in sorted(cache_dir.iterdir()):
        if not source_date_dir.is_dir():
            continue
        source_date = source_date_dir.name
        if not source_date[:4].isdigit():
            continue
        for record_path in sorted(source_date_dir.glob("*.json")):
            record = json.loads(record_path.read_text(encoding="utf-8"))
            chunk_groups = record.get("chunks") or []
            labels = record.get("groundTruth") or []
            split = str(record.get("split") or "")
            if len(chunk_groups) != len(labels):
                continue
            for index, (chunk_group, label) in enumerate(zip(chunk_groups, labels)):
                session = chunk_group_to_micro_session(
                    chunk_group,
                    window_id=f"legacy-{source_date}",
                    item_id=f"{record.get('chunkId', record_path.stem)}:{index}",
                )
                if session is None:
                    continue
                rows.append(
                    LabeledMicroSession(
                        session=session,
                        label=int(label),
                        source="legacy-chunk-proxy",
                        source_date=source_date,
                        split=split,
                    )
                )
    return rows


def discover_labeled_sessions(
    *,
    cache_dir: Path = Path("data/benchmark"),
    v41_jsonl: Path | None = None,
) -> tuple[list[LabeledMicroSession], dict[str, Any]]:
    """Return all labeled rows plus a summary of sources used."""
    rows: list[LabeledMicroSession] = []
    sources: dict[str, int] = {}

    candidates = [
        v41_jsonl,
        Path("data/micro_session_benchmark.jsonl"),
        Path("data/v41_labeled.jsonl"),
    ]
    for candidate in candidates:
        if candidate is None or not candidate.is_file():
            continue
        loaded = load_v41_jsonl(candidate)
        rows.extend(loaded)
        sources[str(candidate)] = len(loaded)

    if cache_dir.is_dir():
        legacy = load_legacy_chunk_benchmark(cache_dir)
        rows.extend(legacy)
        sources["legacy-chunk-proxy"] = len(legacy)

    summary = {
        "rows": len(rows),
        "positive_rate": float(sum(row.label for row in rows) / max(len(rows), 1)),
        "sources": sources,
        "source_dates": sorted({row.source_date for row in rows if row.source_date}),
    }
    return rows, summary


def split_by_source_date(
    rows: list[LabeledMicroSession],
    *,
    holdout_dates: int = 3,
) -> tuple[list[LabeledMicroSession], list[LabeledMicroSession]]:
    dated = [row for row in rows if row.source_date]
    undated = [row for row in rows if not row.source_date]
    unique_dates = sorted({row.source_date for row in dated})
    if len(unique_dates) <= holdout_dates:
        holdout = set(unique_dates[-1:]) if unique_dates else set()
    else:
        holdout = set(unique_dates[-holdout_dates:])

    train = [row for row in dated if row.source_date not in holdout]
    val = [row for row in dated if row.source_date in holdout]
    split_point = max(1, int(len(undated) * 0.85))
    train.extend(undated[:split_point])
    val.extend(undated[split_point:])
    return train, val
