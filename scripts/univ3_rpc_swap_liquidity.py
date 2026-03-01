"""
univ3_rpc_swap_liquidity.py — Fetch Swap.liquidity via eth_getLogs with retries.

The Uniswap v3 Swap event includes the pool's active liquidity after the swap.
This value is NOT reliably available in the subgraph, but it is required to
compute correct running-state columns (L_before/L_after, x/y virtual reserves).

This module fetches Swap logs for a pool over a block range and decodes
`liquidity` from the event data. It is designed to be resilient to common
public-RPC limitations (range too large, too many results, 413, 429, timeouts).
"""

from __future__ import annotations

import random
import time
from typing import Any, Dict, Iterable, Optional, Tuple, Union

from web3 import Web3

from quarantined_rpc import QuarantinedRPC, QuarantinedRPCError


SWAP_TOPIC0 = "0x" + Web3.keccak(text="Swap(address,address,int256,int256,uint160,uint128,int24)").hex()


RETRYABLE_SUBSTRINGS = (
    "timeout",
    "timed out",
    "429",
    "rate limit",
    "too many request",
    "limit exceeded",
    "503",
    "502",
    "500",
    "gateway",
    "temporarily unavailable",
    "413",
    "payload too large",
    "request entity too large",
    "entity too large",
    "content too big",
    "range is too large",
    "query returned more than",
    "exceeds max results",
    "-32005",
    "-32603",
    "-32602",
)


def fetch_swap_liquidity_map(
    w3: Any,
    pool: str,
    start_block: int,
    end_block: int,
    chunk_blocks: int = 500,
    soft_log_limit: int = 9000,
    max_retries: int = 6,
    backoff_base_seconds: float = 0.8,
    politeness_sleep_range: Tuple[float, float] = (0.05, 0.15),
    rpc: Optional[QuarantinedRPC] = None,
) -> Dict[Tuple[int, int], int]:
    """
    Fetch a mapping (blockNumber, logIndex) -> Swap.liquidity for a block range.

    Parameters
    ----------
    w3:
        A Web3 instance (ideally created via `create_multi_provider_web3` for failover).
    pool:
        Pool address (0x...) - checksum or lowercase is fine.
    start_block, end_block:
        Inclusive block range.
    chunk_blocks:
        Initial chunk size in blocks. The function will adaptively split further
        on provider limits.
    soft_log_limit:
        If a successful `eth_getLogs` call returns more than this number of logs,
        the range is split to avoid exceeding provider result caps.
    max_retries:
        Maximum retries for transient errors before range splitting.
    backoff_base_seconds:
        Base for exponential backoff on retryable errors.
    politeness_sleep_range:
        Sleep interval (min,max) after each successful `eth_getLogs` call to reduce
        burst 429s on public RPCs.
    rpc:
        Optional quarantine-aware JSON-RPC caller. If provided, this function uses
        `rpc.call("eth_getLogs", ...)` instead of `w3.eth.get_logs(...)` and will
        respect `Retry-After` on HTTP 429 by quarantining the offending endpoint.

    Returns
    -------
    dict
        Mapping from (blockNumber, logIndex) to decoded `liquidity` (int).

    Notes
    -----
    - Units of work parallelism: none. This function intentionally avoids concurrent
      getLogs calls because they often trigger rate limits across endpoints.
    - Determinism: deterministic given RPC responses; ordering is irrelevant because
      the output is a dict keyed by (blockNumber, logIndex).

    Examples
    --------
    >>> # Typical usage (constructed once and reused across calls)
    >>> from quarantined_rpc import QuarantinedRPC
    >>> rpc = QuarantinedRPC([\"http://localhost:8545\"], max_attempts=1)
    >>> isinstance(rpc, QuarantinedRPC)
    True
    """
    if end_block < start_block:
        return {}

    pool_checksum = Web3.to_checksum_address(pool)
    out: Dict[Tuple[int, int], int] = {}

    def _sleep_backoff(attempt: int) -> None:
        delay = backoff_base_seconds * (2**attempt) + random.uniform(0.0, backoff_base_seconds)
        time.sleep(delay)

    def _is_retryable(msg_lower: str) -> bool:
        return any(s in msg_lower for s in RETRYABLE_SUBSTRINGS)

    def _to_int(x: Union[int, str, None]) -> Optional[int]:
        if x is None:
            return None
        if isinstance(x, int):
            return x
        s = str(x)
        if s.startswith("0x"):
            return int(s, 16)
        try:
            return int(s)
        except Exception:
            return None

    def _decode_liquidity_from_data(data: Any) -> Optional[int]:
        """
        Decode Swap.liquidity from the ABI-encoded log data.

        Parameters
        ----------
        data:
            Log `data` field (0x-hex string or bytes-like).

        Returns
        -------
        int | None
            Decoded liquidity as int, or None if data is malformed.

        Notes
        -----
        Swap event data encodes 5 words (5*32 bytes):
          amount0, amount1, sqrtPriceX96, liquidity, tick
        Liquidity is the 4th word at offset 96..128 bytes.
        """
        if data is None:
            return None
        if isinstance(data, (bytes, bytearray)):
            b = bytes(data)
        else:
            s = str(data)
            if s.startswith("0x"):
                s = s[2:]
            try:
                b = bytes.fromhex(s)
            except Exception:
                return None
        if len(b) < 160:
            return None
        return int.from_bytes(b[96:128], byteorder="big", signed=False)

    def _get_logs_range(a: int, b: int) -> Optional[Iterable[Dict[str, Any]]]:
        if rpc is not None:
            # Direct JSON-RPC call with quarantine-aware endpoint rotation.
            filt = {
                "fromBlock": hex(int(a)),
                "toBlock": hex(int(b)),
                "address": pool_checksum,
                "topics": [SWAP_TOPIC0],
            }
            logs = rpc.call("eth_getLogs", [filt])
            lo, hi = politeness_sleep_range
            if hi > 0:
                time.sleep(random.uniform(max(0.0, lo), max(0.0, hi)))
            return logs

        # Fallback: use Web3 provider (may be randomised by eth_defi's fallback provider).
        filt = {
            "fromBlock": int(a),
            "toBlock": int(b),
            "address": pool_checksum,
            "topics": [SWAP_TOPIC0],
        }
        for attempt in range(max_retries):
            try:
                logs = w3.eth.get_logs(filt)
                # small sleep to reduce burstiness across providers
                lo, hi = politeness_sleep_range
                if hi > 0:
                    time.sleep(random.uniform(max(0.0, lo), max(0.0, hi)))
                return logs
            except Exception as exc:
                msg = str(exc).lower()
                if attempt < max_retries - 1 and _is_retryable(msg):
                    _sleep_backoff(attempt)
                    continue
                raise
        return None

    def _yield_logs_adaptive(a: int, b: int, depth: int = 0) -> None:
        if a > b:
            return
        # Avoid pathological recursion; at worst go down to single blocks.
        if depth > 40 and a < b:
            mid = (a + b) // 2
            _yield_logs_adaptive(a, mid, depth + 1)
            _yield_logs_adaptive(mid + 1, b, depth + 1)
            return

        try:
            logs = _get_logs_range(a, b)
        except QuarantinedRPCError:
            # All endpoints exhausted/quarantined — splitting the range won't
            # help (it doubles requests to the same rate-limited endpoints).
            # Re-raise so the caller can decide how to handle it.
            if a == b:
                print(f"  ⚠️  RPC: giving up on block {a} (all endpoints rate-limited)", flush=True)
                return
            raise
        except Exception as exc:
            msg = str(exc).lower()
            if a == b:
                print(f"  ⚠️  RPC: giving up on block {a} ({exc.__class__.__name__})", flush=True)
                return
            if _is_retryable(msg):
                mid = (a + b) // 2
                _yield_logs_adaptive(a, mid, depth + 1)
                _yield_logs_adaptive(mid + 1, b, depth + 1)
                return
            # non-retryable errors should surface
            raise

        if logs is None:
            return

        if len(logs) > soft_log_limit and a < b:
            mid = (a + b) // 2
            _yield_logs_adaptive(a, mid, depth + 1)
            _yield_logs_adaptive(mid + 1, b, depth + 1)
            return

        for lg in logs:
            liq = _decode_liquidity_from_data(lg.get("data") if isinstance(lg, dict) else getattr(lg, "data", None))
            if liq is None:
                continue
            bn = _to_int(lg.get("blockNumber") if isinstance(lg, dict) else getattr(lg, "blockNumber", None))
            li = _to_int(lg.get("logIndex") if isinstance(lg, dict) else getattr(lg, "logIndex", None))
            if bn is None or li is None:
                continue
            out[(bn, li)] = liq

    cur = int(start_block)
    while cur <= end_block:
        to_blk = min(cur + int(chunk_blocks) - 1, int(end_block))
        _yield_logs_adaptive(cur, to_blk, depth=0)
        cur = to_blk + 1

    return out
