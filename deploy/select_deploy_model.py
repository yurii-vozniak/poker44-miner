#!/usr/bin/env python3
"""Select and apply the best micro-session model for live deployment."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from deploy.micro_session_dataset import discover_labeled_sessions, load_v41_jsonl, split_by_source_date
from deploy.micro_session_features import sessions_to_matrix
from deploy.micro_session_reward import reward
from deploy.train_micro_session_v2 import _fuse, _reference_probs
from deploy.v41_benchmark_client import V41BenchmarkClient
from poker44.miner.config import MinerModelConfig
from poker44.miner.model import ReferenceSessionModel

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = REPO_ROOT / ".env"
STATE_FILE = REPO_ROOT / "data" / "benchmark" / "v3_deploy_state.json"
V41_JSONL = REPO_ROOT / "data" / "micro_session_benchmark.jsonl"

# Live Aug-1 evidence: reference-v2 ~= 0.243 on tournament windows.
LIVE_REFERENCE_REWARD = 0.2432
PROXY_BEAT_REFERENCE_MARGIN = 0.08
PROXY_MIN_REWARD = 0.22


def _evaluate_reference(sessions: list[dict], labels: np.ndarray) -> float:
    ref = ReferenceSessionModel(
        MinerModelConfig(
            factory="reference",
            version="reference-v2",
            model_path="",
            device="cpu",
            max_sessions_per_request=256,
        )
    )
    probs = np.asarray(ref.predict(sessions), dtype=np.float64)
    return float(reward(probs, labels).reward)


def _evaluate_v2_holdout(artifact_path: Path, val_sessions: list[dict], labels: np.ndarray) -> float:
    manifest_path = artifact_path.with_suffix(".manifest.json")
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        holdout = (manifest.get("holdout_metrics") or {}).get("ml_only") or {}
        if holdout.get("reward") is not None:
            return float(holdout["reward"])

    payload = joblib.load(artifact_path)
    ref_probs = _reference_probs(val_sessions)
    mode = str(payload.get("fusion_mode") or "reference")
    weight = float(payload.get("fusion_weight") or 1.0)
    if mode == "reference":
        probs = ref_probs
    else:
        matrix = sessions_to_matrix(val_sessions)
        if payload.get("reference_feature"):
            matrix = np.hstack([matrix, ref_probs.reshape(-1, 1)])
        ml_probs = payload["ml_model"].predict_proba(matrix)[:, 1]
        probs = _fuse(ref_probs, ml_probs, mode=mode, weight=weight)
    return float(reward(probs, labels).reward)


def _probe_v41_corpus() -> int:
    client = V41BenchmarkClient()
    status = client.discover_status()
    if not status.get("endpoint"):
        return 0
    return client.download_jsonl(output_path=V41_JSONL)


def _choose_strategy(*, holdout_dates: int = 5) -> dict:
    v41_rows = 0
    if V41_JSONL.is_file():
        v41_rows = len(load_v41_jsonl(V41_JSONL))
    if v41_rows == 0:
        v41_rows = _probe_v41_corpus()

    rows, summary = discover_labeled_sessions(v41_jsonl=V41_JSONL if v41_rows else None)
    train_rows, val_rows = split_by_source_date(rows, holdout_dates=holdout_dates)
    val_sessions = [row.session for row in val_rows]
    val_labels = np.asarray([row.label for row in val_rows], dtype=int)

    ref_holdout = _evaluate_reference(val_sessions, val_labels) if val_rows else 0.0
    v2_path = REPO_ROOT / "models" / "micro_session_v2.joblib"
    v2_holdout = _evaluate_v2_holdout(v2_path, val_sessions, val_labels) if v2_path.is_file() and val_rows else 0.0

    use_reference = True
    reason = "reference-v2 proven ~0.243 live; proxy holdout is not fully predictive"
    factory = "poker44.miner.model:create_reference_model"
    version = "reference-v2"
    model_path = ""

    if v41_rows >= 100 and v2_holdout >= PROXY_MIN_REWARD and v2_holdout >= ref_holdout + PROXY_BEAT_REFERENCE_MARGIN:
        use_reference = False
        reason = (
            f"public v4.1 corpus ({v41_rows} rows) + honest holdout "
            f"v2={v2_holdout:.4f} beats reference proxy={ref_holdout:.4f}"
        )
        factory = "deploy.micro_session_ensemble_model:create_model"
        version = "micro-v2"
        model_path = "./models/micro_session_v2.joblib"
    elif v41_rows >= 100:
        reason = (
            f"v4.1 corpus found ({v41_rows} rows) but holdout v2={v2_holdout:.4f} "
            f"did not beat reference proxy={ref_holdout:.4f} by {PROXY_BEAT_REFERENCE_MARGIN}; keeping reference"
        )

    return {
        "selected_at": datetime.now(timezone.utc).isoformat(),
        "strategy": "reference-v2" if use_reference else "micro-v2",
        "reason": reason,
        "factory": factory,
        "version": version,
        "model_path": model_path,
        "metrics": {
            "v41_rows": v41_rows,
            "proxy_rows": summary.get("rows", 0),
            "holdout_rows": len(val_rows),
            "reference_holdout_reward": round(ref_holdout, 4),
            "v2_holdout_reward": round(v2_holdout, 4),
            "live_reference_baseline": LIVE_REFERENCE_REWARD,
        },
    }


def _update_env(decision: dict) -> bool:
    if not ENV_FILE.is_file():
        return False
    text = ENV_FILE.read_text(encoding="utf-8")
    replacements = {
        "POKER44_MODEL_FACTORY": decision["factory"],
        "POKER44_MODEL_VERSION": decision["version"],
        "POKER44_MODEL_PATH": decision["model_path"],
    }
    for key, value in replacements.items():
        pattern = rf"^{re.escape(key)}=.*$"
        replacement = f"{key}={value}"
        if re.search(pattern, text, flags=re.MULTILINE):
            text = re.sub(pattern, replacement, text, count=1, flags=re.MULTILINE)
        else:
            text = text.rstrip() + f"\n{replacement}\n"
    ENV_FILE.write_text(text, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Write selection to .env")
    parser.add_argument("--holdout-dates", type=int, default=5)
    args = parser.parse_args()

    decision = _choose_strategy(holdout_dates=args.holdout_dates)
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")

    changed = False
    if args.apply:
        changed = _update_env(decision)

    print(json.dumps({**decision, "env_updated": changed}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
