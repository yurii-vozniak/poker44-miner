"""Canonical Poker44 micro-session miner request contract."""

from __future__ import annotations

from typing import Any, ClassVar, Literal

import bittensor as bt
from pydantic import ConfigDict, Field

from poker44.contracts import find_forbidden, validate_v4_micro_session


class MicroSessionDetectionSynapse(bt.Synapse):
    """Classify audited schema-v4.1 tournament micro-sessions."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    contract_version: Literal["microsession-v1"] = "microsession-v1"
    window_id: str = ""
    dataset_hash: str = ""
    query_id: str = ""
    items: list[dict[str, Any]] = Field(default_factory=list)
    risk_scores: list[float] | None = None
    predictions: list[bool] | None = None
    model_version: str | None = None

    required_hash_fields: ClassVar[list[str]] = [
        "contract_version",
        "window_id",
        "dataset_hash",
        "query_id",
        "items",
    ]

    def deserialize(self) -> "MicroSessionDetectionSynapse":
        return self


def _validate_envelope(window_id: str, dataset_hash: str, query_id: str) -> None:
    if not window_id.strip():
        raise ValueError("window_id is required")
    digest = dataset_hash.strip().lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError("dataset_hash must be a 64-character SHA-256 hex digest")
    if not query_id.strip():
        raise ValueError("query_id is required")


def validate_micro_session_request(synapse: MicroSessionDetectionSynapse) -> None:
    _validate_envelope(synapse.window_id, synapse.dataset_hash, synapse.query_id)
    if not synapse.items:
        raise ValueError("micro-session request contains no items")
    for index, item in enumerate(synapse.items):
        if not isinstance(item, dict) or str(item.get("schema_version")) != "4.1":
            raise ValueError(f"items[{index}] must use schema 4.1")
        leaked = find_forbidden(item, f"items[{index}]")
        if leaked:
            raise ValueError(f"items[{index}] contains forbidden fields: {sorted(leaked)}")
        validate_v4_micro_session(item, index)
