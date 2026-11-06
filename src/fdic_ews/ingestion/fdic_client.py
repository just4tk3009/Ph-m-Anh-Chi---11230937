"""Thin, robust client for the public FDIC BankFind Suite API (https://api.fdic.gov/banks).

Design choices (each is discussed in Part 1 of the report):

* **Retries with exponential backoff** on HTTP 429/5xx and connection errors,
  using ``urllib3.Retry`` mounted on a ``requests.Session`` (connection reuse).
* **Offset pagination** with the API maximum page size (10,000 rows), and a
  completeness check: the number of rows received must equal ``meta.total``.
* **Raw bytes are returned unchanged**, so the Bronze layer stores exactly what
  the API sent (provenance + re-processing without re-downloading).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..utils.io_utils import sha256_bytes, utc_now
from ..utils.logging_utils import get_logger

log = get_logger("ingestion.client")


@dataclass
class Page:
    """One API response page, kept as raw bytes plus metadata for the manifest."""

    endpoint: str
    params: dict[str, Any]
    raw: bytes
    n_rows: int
    total: int
    fetched_at: str = field(default_factory=utc_now)

    @property
    def sha256(self) -> str:
        return sha256_bytes(self.raw)


class FDICClient:
    def __init__(self, base_url: str, page_limit: int = 10000, timeout_s: int = 90,
                 max_retries: int = 6, backoff_factor: float = 2.0,
                 sleep_between_requests_s: float = 0.25) -> None:
        self.base_url = base_url.rstrip("/")
        self.page_limit = page_limit
        self.timeout_s = timeout_s
        self.sleep = sleep_between_requests_s
        retry = Retry(
            total=max_retries, connect=max_retries, read=max_retries,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",), backoff_factor=backoff_factor,
            respect_retry_after_header=True, raise_on_status=False,
        )
        self.session = requests.Session()
        self.session.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=4))
        self.session.headers.update({"Accept": "application/json",
                                     "User-Agent": "bft65b-fdic-ews/0.1 (academic project)"})

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "FDICClient":
        a = cfg["api"]
        return cls(a["base_url"], a["page_limit"], a["timeout_s"], a["max_retries"],
                   a["backoff_factor"], a["sleep_between_requests_s"])

    # ------------------------------------------------------------------ core
    def _get(self, endpoint: str, params: dict[str, Any]) -> bytes:
        url = f"{self.base_url}/{endpoint}"
        resp = self.session.get(url, params=params, timeout=self.timeout_s)
        if resp.status_code != 200:
            raise RuntimeError(f"GET {resp.url} -> HTTP {resp.status_code}: {resp.text[:300]}")
        return resp.content

    @staticmethod
    def parse(raw: bytes) -> tuple[list[dict[str, Any]], int]:
        """Return (records, total) from an FDIC JSON payload.

        FDIC wraps each row as ``{"data": {...}, "score": ...}``; the total row
        count is in ``meta.total`` (or ``totals.count`` on some endpoints).
        """
        obj = json.loads(raw)
        rows = [r.get("data", r) for r in obj.get("data", [])]
        total = obj.get("meta", {}).get("total")
        if total is None:
            total = obj.get("totals", {}).get("count", len(rows))
        return rows, int(total)

    def iter_pages(self, endpoint: str, filters: str | None, fields: list[str] | None,
                   sort_by: str = "CERT") -> Iterator[Page]:
        """Yield every page for one query, checking completeness at the end."""
        offset, received, total = 0, 0, None
        while True:
            params: dict[str, Any] = {"limit": self.page_limit, "offset": offset,
                                      "sort_by": sort_by, "sort_order": "ASC",
                                      "format": "json"}
            if filters:
                params["filters"] = filters
            if fields:
                params["fields"] = ",".join(fields)
            t0 = time.perf_counter()
            raw = self._get(endpoint, params)
            rows, total = self.parse(raw)
            received += len(rows)
            log.info("%s filters=%s offset=%d rows=%d/%d (%.1fs)", endpoint, filters,
                     offset, received, total, time.perf_counter() - t0)
            yield Page(endpoint, params, raw, len(rows), total)
            if not rows or received >= total:
                break
            offset += self.page_limit
            time.sleep(self.sleep)
        if total is not None and received != total:
            raise RuntimeError(f"Incomplete download for {endpoint} {filters}: "
                               f"{received} rows received, API reports {total}")

    def probe_fields(self, endpoint: str, filters: str | None) -> list[str]:
        """Fetch one row with *all* fields: used to validate the configured field list."""
        params: dict[str, Any] = {"limit": 1, "format": "json"}
        if filters:
            params["filters"] = filters
        rows, _ = self.parse(self._get(endpoint, params))
        return sorted(rows[0].keys()) if rows else []
