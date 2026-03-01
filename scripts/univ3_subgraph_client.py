"""
univ3_subgraph_client.py — Subgraph client + merged event stream for Uniswap v3.

This module is intentionally small and explicit. It provides:
  - A retrying GraphQL client for The Graph endpoints
  - Paging helpers for swaps/mints/burns
  - A merged event iterator yielding events in chronological order

The merged iterator is designed for long-running harvesters that need to stream
events without loading the full time window into memory.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import requests


@dataclass
class Cursor:
    """Cursor for GraphQL pagination."""

    last_id: str = ""
    exhausted: bool = False


@dataclass
class PoolState:
    """Pool state snapshot used to initialize the running-state computation."""

    liquidity: int
    sqrt_price_x96: int
    tick: int
    token0_decimals: Optional[int]
    token1_decimals: Optional[int]


class SubgraphClient:
    """
    Minimal GraphQL client with retries/backoff.

    Parameters
    ----------
    graph_url:
        Full HTTPS endpoint of the subgraph (e.g., The Graph gateway URL).
    timeout:
        Per-request timeout in seconds.
    max_retries:
        Number of retries for transient failures.
    backoff_base:
        Base delay for exponential backoff (seconds).

    Notes
    -----
    - Retries on network errors, HTTP 5xx, and GraphQL "errors" payloads.
    - Uses exponential backoff with jitter: base * 2**attempt + U(0, base).
    """

    def __init__(
        self,
        graph_url: str,
        timeout: float = 60.0,
        max_retries: int = 6,
        backoff_base: float = 0.8,
    ) -> None:
        self.graph_url = str(graph_url).strip()
        if not self.graph_url:
            raise ValueError("graph_url must be a non-empty URL")
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self.backoff_base = float(backoff_base)

    def post(self, query: str, variables: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a GraphQL POST request with retries.

        Parameters
        ----------
        query:
            GraphQL query string.
        variables:
            Query variables dict.

        Returns
        -------
        dict
            The `data` field from GraphQL response.

        Notes
        -----
        Raises on final failure, including when GraphQL returns an "errors" field.

        Examples
        --------
        >>> client = SubgraphClient("https://example.com/subgraphs/id/...")
        >>> data = client.post("query($x:Int!){_meta{block{number}}}", {"x": 1})
        """
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    self.graph_url,
                    json={"query": query, "variables": variables},
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                payload = resp.json()
                if "errors" in payload and payload["errors"]:
                    raise RuntimeError(str(payload["errors"]))
                return payload.get("data") or {}
            except Exception as exc:
                last_err = exc
                if attempt >= self.max_retries - 1:
                    break
                delay = self.backoff_base * (2**attempt) + random.uniform(0.0, self.backoff_base)
                print(
                    f"Subgraph: retry {attempt + 1}/{self.max_retries} in {delay:.1f}s ({exc.__class__.__name__}: {exc})",
                    flush=True,
                )
                time.sleep(delay)
        assert last_err is not None
        raise last_err


# ---------- GraphQL queries ----------

Q_FIRST_EVENT_BLOCKS = """
query($pool: Bytes!, $ts: BigInt!) {
  s: swaps(first:1, orderBy: timestamp, orderDirection: asc, where:{pool:$pool, timestamp_gte:$ts}) {
    transaction{ blockNumber }
  }
  m: mints(first:1, orderBy: timestamp, orderDirection: asc, where:{pool:$pool, timestamp_gte:$ts}) {
    transaction{ blockNumber }
  }
  b: burns(first:1, orderBy: timestamp, orderDirection: asc, where:{pool:$pool, timestamp_gte:$ts}) {
    transaction{ blockNumber }
  }
}
"""

Q_POOL_AT_BLOCK = """
query($pool: ID!, $b: Int!) {
  pool(id: $pool, block: { number: $b }) {
    liquidity
    sqrtPrice
    tick
    token0 { decimals }
    token1 { decimals }
  }
}
"""

Q_SWAPS_PAGE = """
query($pool: Bytes!, $lo: BigInt!, $hi: BigInt!, $afterId: ID!, $n: Int!) {
  swaps(first: $n, orderBy: timestamp, orderDirection: asc,
        where: { pool: $pool, timestamp_gte: $lo, timestamp_lte: $hi, id_gt: $afterId }) {
    id
    timestamp
    logIndex
    origin
    transaction { id blockNumber }
    sender
    recipient
    amount0
    amount1
    sqrtPriceX96
    tick
  }
}
"""

Q_MINTS_PAGE = """
query($pool: Bytes!, $lo: BigInt!, $hi: BigInt!, $afterId: ID!, $n: Int!) {
  mints(first: $n, orderBy: timestamp, orderDirection: asc,
        where: { pool: $pool, timestamp_gte: $lo, timestamp_lte: $hi, id_gt: $afterId }) {
    id
    timestamp
    logIndex
    origin
    transaction { id blockNumber }
    sender
    owner
    amount
    amount0
    amount1
    tickLower
    tickUpper
  }
}
"""

Q_BURNS_PAGE = """
query($pool: Bytes!, $lo: BigInt!, $hi: BigInt!, $afterId: ID!, $n: Int!) {
  burns(first: $n, orderBy: timestamp, orderDirection: asc,
        where: { pool: $pool, timestamp_gte: $lo, timestamp_lte: $hi, id_gt: $afterId }) {
    id
    timestamp
    logIndex
    origin
    transaction { id blockNumber }
    owner
    amount
    amount0
    amount1
    tickLower
    tickUpper
  }
}
"""


# ---------- Public helpers ----------

def find_first_event_block(client: SubgraphClient, pool: str, start_ts: int) -> Optional[int]:
    """
    Find the first block (min over swaps/mints/burns) after a start timestamp.

    Parameters
    ----------
    client:
        Subgraph client.
    pool:
        Pool address (0x...) as Bytes string for Graph (lowercase is fine).
    start_ts:
        Inclusive window start timestamp (unix seconds).

    Returns
    -------
    int | None
        Block number of the earliest event at/after `start_ts`, or None if no events exist.

    Notes
    -----
    This is used to initialize the running state at `first_block - 1`.
    """
    d = client.post(Q_FIRST_EVENT_BLOCKS, {"pool": pool, "ts": int(start_ts)})
    candidates: List[int] = []
    for key in ("s", "m", "b"):
        arr = d.get(key) or []
        if not arr:
            continue
        try:
            candidates.append(int(arr[0]["transaction"]["blockNumber"]))
        except Exception:
            continue
    return min(candidates) if candidates else None


def fetch_pool_state_and_decimals_at_block(
    client: SubgraphClient, pool_id: str, block_num: int
) -> PoolState:
    """
    Fetch pool liquidity/sqrtPrice/tick and token decimals at a given block.

    Parameters
    ----------
    client:
        Subgraph client.
    pool_id:
        Pool ID in the subgraph (lowercase 0x address).
    block_num:
        Block number for the historical query.

    Returns
    -------
    PoolState
        Snapshot containing liquidity, sqrtPriceX96, tick and token decimals.

    Notes
    -----
    - The Uniswap v3 subgraph exposes `sqrtPrice` (Q64.96) and `liquidity` as strings.
    - If the pool is not yet indexed at `block_num`, this raises.
    """
    d = client.post(Q_POOL_AT_BLOCK, {"pool": pool_id, "b": int(block_num)})
    p = d.get("pool")
    if not p:
        raise RuntimeError(f"Subgraph returned no pool state at block {block_num}")
    return PoolState(
        liquidity=int(p["liquidity"]),
        sqrt_price_x96=int(p["sqrtPrice"]),
        tick=int(p["tick"]),
        token0_decimals=int(p["token0"]["decimals"]) if p.get("token0") and p["token0"].get("decimals") is not None else None,
        token1_decimals=int(p["token1"]["decimals"]) if p.get("token1") and p["token1"].get("decimals") is not None else None,
    )


def page_swaps(
    client: SubgraphClient, pool: str, lo_ts: int, hi_ts: int, after_id: str, page_size: int
) -> List[Dict[str, Any]]:
    """
    Fetch one page of swaps after a cursor id.

    Parameters
    ----------
    client:
        Subgraph client.
    pool:
        Pool address (Bytes) string (0x..., typically lowercase).
    lo_ts, hi_ts:
        Inclusive time window in unix seconds.
    after_id:
        Cursor id (Graph entity id) to page after (exclusive). Use "" for first page.
    page_size:
        Maximum number of entities to return.

    Returns
    -------
    list[dict]
        List of swap entity dicts, possibly empty.

    Notes
    -----
    The returned dicts include `id`, `timestamp`, `logIndex`, `origin`, and nested
    `transaction { id blockNumber }`.

    Examples
    --------
    >>> cur = ""
    >>> page = page_swaps(client, pool, lo_ts, hi_ts, cur, 1000)
    >>> bool(page) in (True, False)
    True
    """
    d = client.post(
        Q_SWAPS_PAGE,
        {"pool": pool, "lo": int(lo_ts), "hi": int(hi_ts), "afterId": after_id or "", "n": int(page_size)},
    )
    return d.get("swaps") or []


def page_mints(
    client: SubgraphClient, pool: str, lo_ts: int, hi_ts: int, after_id: str, page_size: int
) -> List[Dict[str, Any]]:
    """
    Fetch one page of mints after a cursor id.

    Parameters
    ----------
    client:
        Subgraph client.
    pool:
        Pool address (Bytes) string (0x..., typically lowercase).
    lo_ts, hi_ts:
        Inclusive time window in unix seconds.
    after_id:
        Cursor id (Graph entity id) to page after (exclusive). Use "" for first page.
    page_size:
        Maximum number of entities to return.

    Returns
    -------
    list[dict]
        List of mint entity dicts, possibly empty.

    Notes
    -----
    The returned dicts include `id`, `timestamp`, `logIndex`, `origin`, and nested
    `transaction { id blockNumber }`.

    Examples
    --------
    >>> cur = ""
    >>> page = page_mints(client, pool, lo_ts, hi_ts, cur, 1000)
    >>> isinstance(page, list)
    True
    """
    d = client.post(
        Q_MINTS_PAGE,
        {"pool": pool, "lo": int(lo_ts), "hi": int(hi_ts), "afterId": after_id or "", "n": int(page_size)},
    )
    return d.get("mints") or []


def page_burns(
    client: SubgraphClient, pool: str, lo_ts: int, hi_ts: int, after_id: str, page_size: int
) -> List[Dict[str, Any]]:
    """
    Fetch one page of burns after a cursor id.

    Parameters
    ----------
    client:
        Subgraph client.
    pool:
        Pool address (Bytes) string (0x..., typically lowercase).
    lo_ts, hi_ts:
        Inclusive time window in unix seconds.
    after_id:
        Cursor id (Graph entity id) to page after (exclusive). Use "" for first page.
    page_size:
        Maximum number of entities to return.

    Returns
    -------
    list[dict]
        List of burn entity dicts, possibly empty.

    Notes
    -----
    The returned dicts include `id`, `timestamp`, `logIndex`, `origin`, and nested
    `transaction { id blockNumber }`.

    Examples
    --------
    >>> cur = ""
    >>> page = page_burns(client, pool, lo_ts, hi_ts, cur, 1000)
    >>> len(page) >= 0
    True
    """
    d = client.post(
        Q_BURNS_PAGE,
        {"pool": pool, "lo": int(lo_ts), "hi": int(hi_ts), "afterId": after_id or "", "n": int(page_size)},
    )
    return d.get("burns") or []


def merged_event_stream(
    client: SubgraphClient,
    pool: str,
    lo_ts: int,
    hi_ts: int,
    page_size: int,
    cur_swap: Cursor,
    cur_mint: Cursor,
    cur_burn: Cursor,
) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """
    Merge swaps/mints/burns into a single chronological iterator.

    Parameters
    ----------
    client:
        Subgraph client.
    pool:
        Pool address (Bytes) to query (lowercase 0x address).
    lo_ts, hi_ts:
        Inclusive time window (unix seconds).
    page_size:
        Page size for each entity query.
    cur_swap, cur_mint, cur_burn:
        Mutable cursors to support resume.

    Returns
    -------
    iterator
        Yields `(etype, payload)` where `etype ∈ {"swap","mint","burn"}` and payload is the
        GraphQL entity dict.

    Notes
    -----
    - This uses small buffers per entity type and always yields the next event by
      `(timestamp, blockNumber, logIndex)` ordering, which matches on-chain order.
    - The cursors are advanced as pages are fetched; persist `cursor.last_id` in checkpoints.
    """
    buf_swap: List[Dict[str, Any]] = []
    buf_mint: List[Dict[str, Any]] = []
    buf_burn: List[Dict[str, Any]] = []

    def refill(kind: str) -> None:
        cur = {"swap": cur_swap, "mint": cur_mint, "burn": cur_burn}[kind]
        buf = {"swap": buf_swap, "mint": buf_mint, "burn": buf_burn}[kind]
        if cur.exhausted or len(buf) >= max(1, page_size // 4):
            return

        # IMPORTANT: use the last buffered id for paging (if any), otherwise the
        # last yielded id from the cursor. This avoids advancing the cursor on
        # fetch and therefore makes checkpointing safe.
        after_id = str(buf[-1].get("id", "")) if buf else str(cur.last_id or "")
        page_fn = {"swap": page_swaps, "mint": page_mints, "burn": page_burns}[kind]
        page = page_fn(client, pool, lo_ts, hi_ts, after_id, page_size)
        if not page:
            cur.exhausted = True
            return
        buf.extend(page)

    # initial fill
    refill("swap")
    refill("mint")
    refill("burn")

    while True:
        if (
            cur_swap.exhausted
            and cur_mint.exhausted
            and cur_burn.exhausted
            and not (buf_swap or buf_mint or buf_burn)
        ):
            return

        candidates: List[Tuple[str, Dict[str, Any], Tuple[int, int, int]]] = []
        if buf_swap:
            s = buf_swap[0]
            candidates.append(
                (
                    "swap",
                    s,
                    (int(s["timestamp"]), int(s["transaction"]["blockNumber"]), int(s.get("logIndex") or 0)),
                )
            )
        if buf_mint:
            m = buf_mint[0]
            candidates.append(
                (
                    "mint",
                    m,
                    (int(m["timestamp"]), int(m["transaction"]["blockNumber"]), int(m.get("logIndex") or 0)),
                )
            )
        if buf_burn:
            b = buf_burn[0]
            candidates.append(
                (
                    "burn",
                    b,
                    (int(b["timestamp"]), int(b["transaction"]["blockNumber"]), int(b.get("logIndex") or 0)),
                )
            )

        if not candidates:
            refill("swap")
            refill("mint")
            refill("burn")
            continue

        candidates.sort(key=lambda x: x[2])
        etype, item, _ = candidates[0]
        if etype == "swap":
            buf_swap.pop(0)
            if item.get("id"):
                cur_swap.last_id = str(item["id"])
            refill("swap")
        elif etype == "mint":
            buf_mint.pop(0)
            if item.get("id"):
                cur_mint.last_id = str(item["id"])
            refill("mint")
        else:
            buf_burn.pop(0)
            if item.get("id"):
                cur_burn.last_id = str(item["id"])
            refill("burn")
        yield etype, item
