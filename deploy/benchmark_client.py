"""Client for the public Poker44 training benchmark API."""

from __future__ import annotations

from typing import Any, Iterator

import requests

DEFAULT_BASE_URL = "https://api.poker44.net/api/v1/benchmark"
DEFAULT_TIMEOUT = 60


class BenchmarkClient:
    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: int = DEFAULT_TIMEOUT) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        response = requests.get(
            f"{self.base_url}{path}",
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return payload.get("data", payload)

    def status(self) -> dict[str, Any]:
        return self._get("")

    def latest_source_date(self) -> str:
        return str(self.status()["latestSourceDate"])

    def list_releases(self) -> list[dict[str, Any]]:
        data = self._get("/releases")
        releases = data.get("releases") or []
        return sorted(
            releases,
            key=lambda item: str(item.get("sourceDate", "")),
            reverse=True,
        )

    def list_source_dates(self) -> list[str]:
        return [str(release["sourceDate"]) for release in self.list_releases() if release.get("sourceDate")]

    def iter_chunks(
        self,
        source_date: str | None = None,
        *,
        limit: int = 24,
        max_chunks: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        if source_date is None:
            source_date = self.latest_source_date()

        cursor: str | None = None
        downloaded = 0

        while True:
            params: dict[str, Any] = {"sourceDate": source_date, "limit": limit}
            if cursor:
                params["cursor"] = cursor

            data = self._get("/chunks", params=params)
            for chunk in data.get("chunks", []):
                yield chunk
                downloaded += 1
                if max_chunks is not None and downloaded >= max_chunks:
                    return

            cursor = data.get("nextCursor")
            if not cursor:
                return
