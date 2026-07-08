"""Poker44 miner using a benchmark-trained sklearn baseline detector."""

import os
import time
from pathlib import Path
from typing import Tuple

import bittensor as bt

from deploy.baseline_detector import BaselineDetector
from poker44.base.miner import BaseMinerNeuron
from poker44.utils.model_manifest import (
    build_local_model_manifest,
    evaluate_manifest_compliance,
    manifest_digest,
)
from poker44.validator.synapse import DetectionSynapse


class Miner(BaseMinerNeuron):
    def __init__(self, config=None):
        super().__init__(config=config)
        repo_root = Path(__file__).resolve().parents[1]
        model_path = Path(
            os.getenv("POKER44_MODEL_PATH", repo_root / "models" / "baseline.joblib")
        )
        self.detector = BaselineDetector(model_path)
        bt.logging.info(f"Loaded baseline model from {model_path}")

        implementation_files = [
            Path(__file__).resolve(),
            repo_root / "deploy" / "baseline_detector.py",
            repo_root / "deploy" / "features.py",
        ]
        self.model_manifest = build_local_model_manifest(
            repo_root=repo_root,
            implementation_files=implementation_files,
            defaults={
                "model_name": os.getenv("POKER44_MODEL_NAME", "poker44-baseline-sklearn"),
                "model_version": os.getenv("POKER44_MODEL_VERSION", "1"),
                "framework": "scikit-learn",
                "license": "MIT",
                "repo_url": "https://github.com/Poker44/Poker44-subnet",
                "notes": "Baseline logistic regression trained on the public benchmark API.",
                "open_source": True,
                "inference_mode": "remote",
                "training_data_statement": (
                    "Trained on labeled chunks from the public Poker44 benchmark API "
                    "(https://api.poker44.net/api/v1/benchmark). "
                    "No validator-only live evaluation labels are used for training."
                ),
                "training_data_sources": [
                    "https://api.poker44.net/api/v1/benchmark",
                ],
                "private_data_attestation": (
                    "This miner trains only on the public benchmark API and does not "
                    "use validator-only live evaluation labels."
                ),
            },
        )
        self.manifest_compliance = evaluate_manifest_compliance(self.model_manifest)
        self.manifest_digest = manifest_digest(self.model_manifest)
        self._log_manifest_startup(repo_root)

    def _log_manifest_startup(self, repo_root: Path) -> None:
        bt.logging.info(
            f"Miner transparency status: {self.manifest_compliance['status']} "
            f"(missing_fields={self.manifest_compliance['missing_fields']})"
        )
        bt.logging.info(
            f"Manifest summary | model={self.model_manifest.get('model_name', '')} "
            f"version={self.model_manifest.get('model_version', '')} "
            f"commit={self.model_manifest.get('repo_commit', '')}"
        )
        bt.logging.info(f"Manifest digest={self.manifest_digest}")
        bt.logging.info(f"Miner docs: {repo_root / 'docs' / 'miner.md'}")

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        chunks = synapse.chunks or []
        scores = self.detector.score_chunks(chunks)
        synapse.risk_scores = scores
        synapse.predictions = [score >= 0.5 for score in scores]
        synapse.model_manifest = dict(self.model_manifest)
        bt.logging.info(f"Scored {len(chunks)} chunks with baseline model.")
        return synapse

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        return self.common_blacklist(synapse)

    async def priority(self, synapse: DetectionSynapse) -> float:
        return self.caller_priority(synapse)


if __name__ == "__main__":
    with Miner() as miner:
        bt.logging.info("Poker44 baseline miner running...")
        while True:
            bt.logging.info(
                f"Miner UID: {miner.uid} | Incentive: {miner.metagraph.I[miner.uid]}"
            )
            time.sleep(5 * 60)
