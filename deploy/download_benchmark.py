#!/usr/bin/env python3
"""Download and cache multiple Poker44 benchmark release dates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from deploy.benchmark_client import BenchmarkClient
from deploy.benchmark_dataset import download_releases, iter_training_examples, summarize_examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/benchmark"),
        help="Directory for cached chunk JSON records.",
    )
    parser.add_argument(
        "--dates",
        type=int,
        default=7,
        help="Number of most recent release dates to download.",
    )
    parser.add_argument(
        "--source-dates",
        nargs="*",
        default=None,
        help="Explicit source dates (YYYY-MM-DD). Overrides --dates.",
    )
    parser.add_argument(
        "--max-chunks-per-date",
        type=int,
        default=None,
        help="Optional cap on chunk records downloaded per release date.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-download records even when cache files already exist.",
    )
    args = parser.parse_args()

    client = BenchmarkClient()
    if args.source_dates:
        source_dates = list(args.source_dates)
    else:
        source_dates = client.list_source_dates()[: args.dates]

    if not source_dates:
        raise RuntimeError("No benchmark source dates available to download.")

    print(f"Downloading {len(source_dates)} release dates: {', '.join(source_dates)}")
    records_by_date = download_releases(
        client,
        source_dates,
        cache_dir=args.cache_dir,
        max_chunks_per_date=args.max_chunks_per_date,
        refresh=args.refresh,
    )

    examples = list(iter_training_examples(records_by_date))
    summary = {
        "cache_dir": str(args.cache_dir.resolve()),
        "source_dates": source_dates,
        "records_by_date": {date: len(records) for date, records in records_by_date.items()},
        "examples": summarize_examples(examples),
    }
    index_path = args.cache_dir / "index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Saved cache index to {index_path}")


if __name__ == "__main__":
    main()
