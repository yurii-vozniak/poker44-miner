#!/usr/bin/env python3
"""Print transparent-manifest compliance for the hybrid miner."""

from __future__ import annotations

import json
from pathlib import Path

from deploy.manifest_helpers import build_hybrid_model_manifest, manifest_startup_report


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    manifest = build_hybrid_model_manifest(repo_root=repo_root)
    report = manifest_startup_report(manifest)
    payload = {
        "compliance": report,
        "manifest": manifest,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
