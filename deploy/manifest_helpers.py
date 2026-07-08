"""Helpers for building transparent Poker44 miner manifests."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Iterable

from poker44.utils.model_manifest import (
    GIT_COMMIT_RE,
    evaluate_manifest_compliance,
    build_local_model_manifest,
)

BENCHMARK_API_URL = "https://api.poker44.net/api/v1/benchmark"


def resolve_repo_commit(repo_root: Path) -> str:
    """Return a verifiable git commit from env or the local repository."""
    env_commit = os.getenv("POKER44_MODEL_REPO_COMMIT", "").strip()
    if env_commit and GIT_COMMIT_RE.fullmatch(env_commit):
        return env_commit

    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""

    commit = completed.stdout.strip()
    return commit if GIT_COMMIT_RE.fullmatch(commit) else ""


def resolve_repo_url(default: str = "") -> str:
    return os.getenv("POKER44_MODEL_REPO_URL", default).strip()


def hybrid_implementation_files(repo_root: Path) -> list[Path]:
    return [
        repo_root / "neurons" / "miner_hybrid.py",
        repo_root / "deploy" / "hybrid_detector.py",
        repo_root / "deploy" / "features.py",
        repo_root / "deploy" / "manifest_helpers.py",
    ]


def build_hybrid_model_manifest(
    *,
    repo_root: Path,
    implementation_files: Iterable[Path] | None = None,
    model_version: str | None = None,
) -> dict:
    repo_commit = resolve_repo_commit(repo_root)
    repo_url = resolve_repo_url()
    files = list(implementation_files or hybrid_implementation_files(repo_root))

    return build_local_model_manifest(
        repo_root=repo_root,
        implementation_files=files,
        defaults={
            "open_source": True,
            "model_name": os.getenv("POKER44_MODEL_NAME", "poker44-hybrid-lgbm-iso"),
            "model_version": model_version or os.getenv("POKER44_MODEL_VERSION", "1"),
            "framework": os.getenv("POKER44_MODEL_FRAMEWORK", "lightgbm+sklearn"),
            "license": os.getenv("POKER44_MODEL_LICENSE", "MIT"),
            "repo_url": repo_url,
            "repo_commit": repo_commit,
            "inference_mode": os.getenv("POKER44_MODEL_INFERENCE_MODE", "local"),
            "training_data_statement": (
                "Trained on labeled chunk groups from the public Poker44 benchmark API "
                f"({BENCHMARK_API_URL}). Training uses multiple release dates with "
                "date-based holdout validation. No validator-only live evaluation labels "
                "are used for training."
            ),
            "training_data_sources": [BENCHMARK_API_URL],
            "private_data_attestation": (
                "This miner trains only on the public benchmark API and does not use "
                "validator-only live evaluation labels or private hand histories."
            ),
            "notes": (
                "Hybrid LightGBM classifier plus Isolation Forest anomaly detector. "
                "Chunk-level risk scores use max(supervised probability, anomaly score)."
            ),
        },
    )


def manifest_startup_report(manifest: dict) -> dict:
    compliance = evaluate_manifest_compliance(manifest)
    compliance["repo_commit_present"] = bool(manifest.get("repo_commit"))
    compliance["repo_url_present"] = bool(manifest.get("repo_url"))
    return compliance
