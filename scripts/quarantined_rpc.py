"""
quarantined_rpc.py — JSON-RPC caller with endpoint quarantine (429/Retry-After aware).

Public RPC endpoints often enforce per-IP and per-key limits. When an endpoint
returns HTTP 429 with a `Retry-After` header (or similar), blindly retrying the
same endpoint wastes time and can keep you locked out for longer.

This module implements a tiny JSON-RPC client that:
  - cycles endpoints in a deterministic round-robin
  - quarantines endpoints on 429 according to `Retry-After`
  - retries on transient network/5xx errors by switching endpoint

It is intentionally lightweight and script-friendly (not a full provider
replacement).
"""

from __future__ import annotations

import datetime as _dt
import json
import random
import time
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import requests


class QuarantinedRPCError(RuntimeError):
    """Raised when a JSON-RPC call fails after exhausting attempts."""


@dataclass
class EndpointStatus:
    """Track per-endpoint quarantine state."""

    url: str
    quarantine_until: float = 0.0
    last_error: Optional[str] = None
    last_status_code: Optional[int] = None


class QuarantinedRPC:
    """
    Call JSON-RPC methods across multiple endpoints with quarantine on 429.

    Parameters
    ----------
    urls:
        Sequence of endpoint URLs (HTTPS recommended). Empty/whitespace entries
        are ignored.
    timeout_seconds:
        Per-request timeout in seconds for HTTP requests.
    max_attempts:
        Maximum number of attempts per call. Each attempt picks a non-quarantined
        endpoint (round-robin). If all endpoints are quarantined, the caller
        will wait briefly (up to `max_wait_when_all_quarantined_seconds`) and
        then re-check.
    backoff_base_seconds:
        Base for exponential backoff between attempts on transient failures.
    max_wait_when_all_quarantined_seconds:
        If all endpoints are quarantined and the soonest release time is further
        than this threshold, raise immediately instead of sleeping for a long time.

    Returns
    -------
    QuarantinedRPC
        Instance.

    Notes
    -----
    - This client is designed to avoid hammering endpoints that explicitly ask
      us to slow down. It does not attempt to perfectly classify all provider
      error messages.
    - Determinism: endpoint selection is round-robin (not random), so behavior is
      reproducible given the same failures/quarantine timings.

    Examples
    --------
    >>> rpc = QuarantinedRPC([\"http://localhost:8545\"], max_attempts=1)
    >>> isinstance(rpc, QuarantinedRPC)
    True
    """

    def __init__(
        self,
        urls: Sequence[str],
        timeout_seconds: float = 30.0,
        max_attempts: int = 12,
        backoff_base_seconds: float = 0.6,
        max_wait_when_all_quarantined_seconds: float = 30.0,
        user_agent: str = "MEV_analysis/QuarantinedRPC",
    ) -> None:
        cleaned = [str(u).strip() for u in urls if str(u).strip()]
        if not cleaned:
            raise ValueError("At least one RPC URL is required")

        self._session = requests.Session()
        self._session.headers.update({"User-Agent": user_agent})

        self._endpoints: List[EndpointStatus] = [EndpointStatus(url=u) for u in cleaned]
        self._rr_index = 0
        self._timeout_seconds = float(timeout_seconds)
        self._max_attempts = int(max_attempts)
        self._backoff_base_seconds = float(backoff_base_seconds)
        self._max_wait_all_quarantined = float(max_wait_when_all_quarantined_seconds)

        # JSON-RPC id is arbitrary; keep it incrementing for debuggability.
        self._rpc_id = random.randint(1, 1_000_000)
        self._last_wait_print = 0.0
        self._last_quarantine_print: Dict[str, float] = {}

    def _short_name(self, url: str) -> str:
        """Return a safe, non-secret identifier for logs (domain only)."""
        try:
            return urlparse(url).netloc or url
        except Exception:
            return url

    def endpoints(self) -> List[str]:
        """
        List configured endpoint URLs.

        Parameters
        ----------
        None

        Returns
        -------
        list[str]
            Endpoint URLs in the order used for round-robin selection.

        Notes
        -----
        This reflects the configuration passed at construction time.

        Examples
        --------
        >>> rpc = QuarantinedRPC([\"http://localhost:8545\"], max_attempts=1)
        >>> rpc.endpoints() == [\"http://localhost:8545\"]
        True
        """
        return [e.url for e in self._endpoints]

    def _parse_retry_after_seconds(self, retry_after: Optional[str]) -> Optional[float]:
        """Parse Retry-After header (seconds or HTTP-date)."""
        if not retry_after:
            return None
        s = str(retry_after).strip()
        if not s:
            return None
        if s.isdigit():
            return float(int(s))
        try:
            dt = parsedate_to_datetime(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=_dt.timezone.utc)
            now = _dt.datetime.now(tz=_dt.timezone.utc)
            return max(0.0, (dt - now).total_seconds())
        except Exception:
            return None

    def _pick_endpoint(self) -> Tuple[int, EndpointStatus]:
        """Pick next non-quarantined endpoint (round-robin)."""
        now = time.time()
        n = len(self._endpoints)
        for k in range(n):
            idx = (self._rr_index + k) % n
            ep = self._endpoints[idx]
            if now >= ep.quarantine_until:
                self._rr_index = (idx + 1) % n
                return idx, ep
        # all quarantined
        soonest = min(self._endpoints, key=lambda e: e.quarantine_until)
        wait_for = max(0.0, soonest.quarantine_until - now)
        if wait_for > 0.5 and (now - self._last_wait_print) > 30.0:
            # Keep this very lightweight: one line every 30s at most.
            print(f"RPC: all endpoints quarantined; waiting {wait_for:.1f}s for the next one to reopen…", flush=True)
            self._last_wait_print = now
        if wait_for > self._max_wait_all_quarantined:
            raise QuarantinedRPCError(
                "All RPC endpoints are quarantined "
                f"(soonest release in {wait_for:.1f}s, threshold {self._max_wait_all_quarantined:.1f}s). "
                "Remove/replace the rate-limited endpoint(s) or wait and retry."
            )
        time.sleep(wait_for)
        # After sleeping, pick again (should succeed unless quarantine extended).
        return self._pick_endpoint()

    def _quarantine(self, idx: int, seconds: float, reason: str) -> None:
        ep = self._endpoints[idx]
        until = time.time() + max(0.0, float(seconds))
        ep.quarantine_until = max(ep.quarantine_until, until)
        ep.last_error = reason

    def call(self, method: str, params: List[Any]) -> Any:
        """
        Call a JSON-RPC method with endpoint quarantine on 429.

        Parameters
        ----------
        method:
            JSON-RPC method name (e.g., \"eth_getLogs\").
        params:
            JSON-RPC params list.

        Returns
        -------
        Any
            The `result` field from the JSON-RPC response.

        Notes
        -----
        - On HTTP 429, quarantines the endpoint for `Retry-After` seconds when present;
          otherwise uses a conservative default (60s).
        - On transient network errors (timeouts, connection errors) and HTTP 5xx,
          quarantines the endpoint briefly and tries another.
        - If the server returns a JSON-RPC `error` payload, this raises `QuarantinedRPCError`
          (caller decides whether to split ranges, etc.).

        Examples
        --------
        >>> rpc = QuarantinedRPC([\"http://localhost:8545\"], max_attempts=1)
        >>> isinstance(rpc.call, object)
        True
        """
        if not isinstance(params, list):
            raise TypeError("params must be a list")

        last_exc: Optional[Exception] = None
        current_sleep = self._backoff_base_seconds

        for attempt in range(self._max_attempts):
            idx, ep = self._pick_endpoint()
            url = ep.url

            self._rpc_id += 1
            payload = {"jsonrpc": "2.0", "id": self._rpc_id, "method": method, "params": params}

            try:
                resp = self._session.post(url, json=payload, timeout=self._timeout_seconds)
                ep.last_status_code = int(resp.status_code)

                # HTTP-level rate limit / bans
                if resp.status_code == 429:
                    retry_after = self._parse_retry_after_seconds(resp.headers.get("Retry-After"))
                    quarantine_for = retry_after if retry_after is not None else 60.0
                    self._quarantine(idx, quarantine_for, f"HTTP 429 Retry-After={resp.headers.get('Retry-After')!r}")
                    now = time.time()
                    short = self._short_name(url)
                    last = self._last_quarantine_print.get(short, 0.0)
                    if quarantine_for >= 5.0 and (now - last) > 15.0:
                        print(
                            f"RPC: quarantining {short} for {quarantine_for:.0f}s (HTTP 429, Retry-After={resp.headers.get('Retry-After')!r})",
                            flush=True,
                        )
                        self._last_quarantine_print[short] = now
                    last_exc = QuarantinedRPCError(f"HTTP 429 from {url}")
                    time.sleep(current_sleep)
                    current_sleep *= 1.25
                    continue

                # Retry on transient 5xx
                if 500 <= resp.status_code < 600:
                    self._quarantine(idx, 10.0, f"HTTP {resp.status_code}")
                    last_exc = QuarantinedRPCError(f"HTTP {resp.status_code} from {url}")
                    time.sleep(current_sleep)
                    current_sleep *= 1.25
                    continue

                resp.raise_for_status()

                try:
                    data = resp.json()
                except json.JSONDecodeError as exc:
                    # Some providers return HTML bodies on errors; quarantine briefly.
                    self._quarantine(idx, 30.0, "Non-JSON response body")
                    last_exc = exc
                    time.sleep(current_sleep)
                    current_sleep *= 1.25
                    continue

                if "error" in data and data["error"] is not None:
                    err = data["error"]
                    code = err.get("code")
                    msg = str(err.get("message", ""))
                    msg_l = msg.lower()
                    # Rate limit reported as JSON-RPC error payload (provider-specific).
                    if ("rate limit" in msg_l) or ("too many request" in msg_l) or (code in (-32090,)):
                        self._quarantine(idx, 30.0, f"JSON-RPC rate limit: {code} {msg}")
                        last_exc = QuarantinedRPCError(f"JSON-RPC rate limit from {url}: {code} {msg}")
                        time.sleep(current_sleep)
                        current_sleep *= 1.25
                        continue
                    raise QuarantinedRPCError(f"JSON-RPC error from {url}: {code} {msg}")

                if "result" not in data:
                    raise QuarantinedRPCError(f"Malformed JSON-RPC response from {url}: missing 'result'")

                return data["result"]

            except (requests.Timeout, requests.ConnectionError) as exc:
                self._quarantine(idx, 10.0, exc.__class__.__name__)
                last_exc = exc
                time.sleep(current_sleep)
                current_sleep *= 1.25
                continue
            except requests.HTTPError as exc:
                # Any other 4xx: quarantine briefly and try other endpoint.
                self._quarantine(idx, 30.0, f"HTTPError {getattr(exc.response, 'status_code', None)}")
                last_exc = exc
                time.sleep(current_sleep)
                current_sleep *= 1.25
                continue

        raise QuarantinedRPCError(f"JSON-RPC call failed after {self._max_attempts} attempts: {last_exc}")
