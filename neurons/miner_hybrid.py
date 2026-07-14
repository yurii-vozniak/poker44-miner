"""Poker44 miner using a hybrid LightGBM + Isolation Forest detector."""

import os
import time
from pathlib import Path
from typing import Tuple

import bittensor as bt
from dotenv import load_dotenv

from deploy.chunk_detector import load_chunk_detector
from deploy.manifest_helpers import build_hybrid_model_manifest, manifest_startup_report
from poker44.base.miner import BaseMinerNeuron
from poker44.utils.model_manifest import manifest_digest
from poker44.validator.synapse import DetectionSynapse


class Miner(BaseMinerNeuron):
    def __init__(self, config=None):
        repo_root = Path(__file__).resolve().parents[1]
        load_dotenv(repo_root / ".env", override=False)
        super().__init__(config=config)
        model_path = Path(
            os.getenv("POKER44_MODEL_PATH", repo_root / "models" / "hybrid.joblib")
        )
        self.detector = load_chunk_detector(model_path)
        bt.logging.info(f"Loaded detector from {model_path}")

        artifact_version = str(
            self.detector.metadata.get("model_version")
            or os.getenv("POKER44_MODEL_VERSION", "1")
        )
        artifact_name = str(
            self.detector.metadata.get("model_name")
            or os.getenv("POKER44_MODEL_NAME", "poker44-hybrid-lgbm-iso")
        )
        self.model_manifest = build_hybrid_model_manifest(
            repo_root=repo_root,
            model_version=artifact_version,
        )
        if artifact_name:
            self.model_manifest["model_name"] = artifact_name
        self.manifest_compliance = manifest_startup_report(self.model_manifest)
        self.manifest_digest = manifest_digest(self.model_manifest)
        self._manifest_warned = False
        self._log_manifest_startup(repo_root)

    def _log_manifest_startup(self, repo_root: Path) -> None:
        bt.logging.info(
            f"Miner transparency status: {self.manifest_compliance['status']} "
            f"(missing_fields={self.manifest_compliance['missing_fields']}, "
            f"policy_violations={self.manifest_compliance['policy_violations']})"
        )
        if self.manifest_compliance["status"] != "transparent":
            bt.logging.warning(
                "Manifest is not transparent. Set POKER44_MODEL_REPO_URL to your public "
                "model repository and POKER44_MODEL_REPO_COMMIT to a verifiable git SHA."
            )
        bt.logging.info(
            f"Manifest summary | model={self.model_manifest.get('model_name', '')} "
            f"version={self.model_manifest.get('model_version', '')} "
            f"repo={self.model_manifest.get('repo_url', '')} "
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
        if (
            not self._manifest_warned
            and self.manifest_compliance.get("status") != "transparent"
        ):
            self._manifest_warned = True
            bt.logging.warning(
                "Sending opaque manifest to validators; dashboard scores may stay at 0 "
                f"until manifest is transparent: missing={self.manifest_compliance.get('missing_fields')}"
            )
        bt.logging.info(f"Scored {len(chunks)} chunks with {self.detector.metadata.get('model_name', 'detector')} model "
            f"(manifest_digest={self.manifest_digest[:12]}…, "
            f"status={self.manifest_compliance.get('status')})."
        )
        return synapse

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        return self.common_blacklist(synapse)

    async def priority(self, synapse: DetectionSynapse) -> float:
        return self.caller_priority(synapse)


if __name__ == "__main__":
    with Miner() as miner:
        bt.logging.info("Poker44 hybrid miner running...")
        while True:
            bt.logging.info(
                f"Miner UID: {miner.uid} | Incentive: {miner.metagraph.I[miner.uid]}"
            )
            time.sleep(5 * 60)
