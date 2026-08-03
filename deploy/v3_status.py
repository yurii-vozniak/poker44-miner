#!/usr/bin/env python3
"""Poker44 v3 dashboard + window status for UID 79."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any

import requests

DEFAULT_BASE = "https://api.poker44.net/api/v1"
DEFAULT_UID = 79


def _get(url: str, timeout: int = 30) -> dict[str, Any]:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    return payload.get("data", payload)


def collect_status(*, uid: int = DEFAULT_UID, base_url: str = DEFAULT_BASE) -> dict[str, Any]:
    miners = _get(f"{base_url}/dashboard/miners")
    miner_list = miners if isinstance(miners, list) else miners.get("miners") or []
    miner = next((row for row in miner_list if row.get("uid") == uid), None)
    readiness = _get(f"{base_url}/dashboard/round-readiness")

    ranked = sorted(miner_list, key=lambda row: float(row.get("average_reward") or 0.0), reverse=True)
    rank = next((index + 1 for index, row in enumerate(ranked) if row.get("uid") == uid), None)
    top10_threshold = float(ranked[9]["average_reward"]) if len(ranked) >= 10 else None

    return {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "uid": uid,
        "rank": rank,
        "miner_count": len(miner_list),
        "top10_threshold_reward": top10_threshold,
        "miner": miner,
        "readiness": readiness,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uid", type=int, default=int(os.getenv("POKER44_UID", DEFAULT_UID)))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    status = collect_status(uid=args.uid)
    if args.json:
        print(json.dumps(status, indent=2))
        return 0

    miner = status.get("miner") or {}
    readiness = status.get("readiness") or {}
    print(f"Checked: {status['checked_at']}")
    print(f"UID {args.uid} rank: {status.get('rank')} / {status.get('miner_count')}")
    if status.get("top10_threshold_reward") is not None:
        gap = float(status["top10_threshold_reward"]) - float(miner.get("average_reward") or 0.0)
        print(f"Top-10 threshold reward: {status['top10_threshold_reward']:.4f} (gap {gap:+.4f})")
    print(
        "Score:",
        f"reward={float(miner.get('average_reward') or 0):.4f}",
        f"ap={float(miner.get('average_precision') or 0):.4f}",
        f"recall={float(miner.get('average_recall_at_fpr_05') or 0):.4f}",
        f"model={miner.get('model_version')}",
        f"evals={miner.get('evaluation_count')}",
        f"last={miner.get('last_evaluated_at')}",
    )
    print(
        "Window:",
        readiness.get("window_id"),
        f"status={readiness.get('status')}",
        f"progress={readiness.get('progress_percent')}%",
        f"collection_active={readiness.get('collection_active')}",
        f"sealed_at={readiness.get('sealed_at')}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
