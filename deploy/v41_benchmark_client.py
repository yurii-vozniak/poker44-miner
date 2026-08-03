"""Probe and download a public labeled v4.1 micro-session corpus when available."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests

DEFAULT_TIMEOUT = 60
CANDIDATE_STATUS_PATHS = (
    "/api/v1/training/micro-sessions",
    "/api/v1/benchmark/micro-sessions",
    "/api/v1/training/benchmark",
    "/api/v1/benchmark",
)
DEFAULT_BASE = "https://api.poker44.net"


class V41BenchmarkClient:
    def __init__(self, base_url: str = DEFAULT_BASE, timeout: int = DEFAULT_TIMEOUT) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
        try:
            response = requests.get(
                f"{self.base_url}{path}",
                params=params,
                timeout=self.timeout,
            )
            if response.status_code == 404:
                return None
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict) and payload.get("success") is False:
                return None
            return payload.get("data", payload) if isinstance(payload, dict) else payload
        except requests.RequestException:
            return None

    def discover_status(self) -> dict[str, Any]:
        """Return the first reachable status payload describing a v4.1 corpus."""
        for path in CANDIDATE_STATUS_PATHS:
            data = self._get(path)
            if not isinstance(data, dict):
                continue
            schema = str(data.get("schemaVersion") or data.get("schema_version") or "")
            if schema.startswith("4") or data.get("corpusAvailable") or data.get("latestSourceDate"):
                return {"endpoint": path, **data}
        return {"endpoint": None, "available": False}

    def download_jsonl(
        self,
        *,
        output_path: Path,
        source_url: str | None = None,
    ) -> int:
        """Download labeled rows to JSONL. Returns number of rows written."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []

        if source_url:
            response = requests.get(source_url, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
            raw_rows = payload.get("data", payload)
            if isinstance(raw_rows, dict):
                raw_rows = raw_rows.get("rows") or raw_rows.get("sessions") or []
            if not isinstance(raw_rows, list):
                raise RuntimeError(f"Unexpected corpus payload from {source_url}")
            rows = [row for row in raw_rows if isinstance(row, dict)]
        else:
            for path in CANDIDATE_STATUS_PATHS:
                data = self._get(path, params={"limit": 5000})
                if not isinstance(data, dict):
                    continue
                raw_rows = data.get("rows") or data.get("sessions") or data.get("items") or []
                if isinstance(raw_rows, list) and raw_rows:
                    rows = [row for row in raw_rows if isinstance(row, dict)]
                    break

        if not rows:
            return 0

        written = 0
        with output_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                label = row.get("label")
                payload = row.get("payload") or row.get("session") or row.get("item")
                if label not in {0, 1} or not isinstance(payload, dict):
                    continue
                handle.write(
                    json.dumps(
                        {
                            "label": int(label),
                            "payload": payload,
                            "source_date": row.get("source_date") or row.get("sourceDate") or "",
                            "split": row.get("split") or "",
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )
                written += 1
        return written
