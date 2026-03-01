#!/usr/bin/env python3
"""
data_fetch.py — Subgraph-first Uniswap v3 pool event harvester (RPC-minimized)

What this script does
---------------------
Fetches Swap/Mint/Burn events for a single Uniswap v3 pool over a time window,
computes the pool *running state* before/after each event, and streams the
result to a CSV with a resume-safe checkpoint.

Reliability strategy (why this exists)
--------------------------------------
Public RPC providers often rate-limit or hard-cap `eth_getLogs` queries over
large ranges. The Uniswap v3 subgraph is typically more reliable for the *event
list* (tx hashes, logIndex, timestamps, amounts, ticks, sqrtPriceX96, origin).

This script therefore:
  1) Uses the subgraph for the canonical event stream (swap/mint/burn).
  2) Uses RPC only for fields that are not reliably stored in the subgraph but
     are required for correctness:
       - Swap event `liquidity` (active liquidity after the swap), decoded from
         on-chain Swap logs via `eth_getLogs`.
  3) Leaves gas fields empty by default (use `scripts/add_gas.py` later).

Configuration (no secrets in repo)
---------------------------------
Set endpoints via environment variables (recommended):
  - UNIV3_GRAPH_URL : subgraph GraphQL endpoint
  - MEV_RPC_URLS    : space-separated RPC endpoints (for failover)
  - WEB3_PROVIDER_URI: single RPC endpoint fallback

You can also pass `--graph-url` and `--rpc-urls`.

Output
------
CSV is written to `data/raw/univ3_<POOL>.csv` by default, with schema compatible
with the existing MEV pipeline (e.g. `scripts/mev_collect.py`).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd
from eth_defi.provider.multi_provider import create_multi_provider_web3
from web3 import Web3

from quarantined_rpc import QuarantinedRPC
from univ3_amounts import to_raw_units
from univ3_checkpoint import load_checkpoint, save_checkpoint_atomic
from univ3_rpc_swap_liquidity import fetch_swap_liquidity_map
from univ3_subgraph_client import (
    Cursor,
    SubgraphClient,
    fetch_pool_state_and_decimals_at_block,
    find_first_event_block,
    merged_event_stream,
)


# ---------------- Defaults (no secrets) ----------------

DEFAULT_GRAPH_URL = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3"
DEFAULT_RPC_URLS = [
    "https://eth.llamarpc.com",
    "https://rpc.ankr.com/eth",
]

DEFAULT_POOL_ADDR = "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"  # USDC/WETH 0.05%
DEFAULT_START_TS: Union[int, str] = "2023-01-01T00:00:00Z"
DEFAULT_END_TS: Union[int, str] = "2024-12-31T23:59:59Z"

DEFAULT_SUBGRAPH_PAGE_SIZE = 1000
DEFAULT_FLUSH_EVERY_EVENTS = 10_000
DEFAULT_RPC_SWAP_CHUNK_BLOCKS = 10

Q96 = 1 << 96


# ---------------- Small helpers ----------------

def to_unix(ts: Union[int, str, _dt.datetime, pd.Timestamp]) -> int:
    """
    Convert a timestamp input to UNIX seconds (UTC).

    Parameters
    ----------
    ts:
        Either an int (already UNIX seconds) or a datetime-like / ISO8601 string.

    Returns
    -------
    int
        UNIX timestamp in seconds.

    Notes
    -----
    Uses pandas parsing for ISO8601 strings and timezone normalization.

    Examples
    --------
    >>> to_unix(0)
    0
    """
    if isinstance(ts, int):
        return ts
    if isinstance(ts, str):
        s = ts.strip()
        if s.lstrip("-").isdigit():
            return int(s)
        return int(pd.to_datetime(s, utc=True).value // 10**9)
    return int(pd.to_datetime(ts, utc=True).value // 10**9)


def checksum_or_none(addr: Optional[str]) -> Optional[str]:
    """
    Normalize an address string to EIP-55 checksum (or return None).

    Parameters
    ----------
    addr:
        Address string (0x...) or None.

    Returns
    -------
    str | None
        Checksum address if parseable, else None.

    Notes
    -----
    Subgraph fields are often lowercase; downstream datasets in this repo use
    checksum addresses.

    Examples
    --------
    >>> checksum_or_none(None) is None
    True
    """
    if addr is None:
        return None
    a = str(addr)
    if not a.startswith("0x") or len(a) != 42:
        return a
    try:
        return Web3.to_checksum_address(a)
    except Exception:
        return a


def virt_x(L: int, sqrtP_x96: int) -> Optional[int]:
    """
    Compute virtual reserve x = L / sqrt(P) in Q64.96 convention.

    Parameters
    ----------
    L:
        Active liquidity (uint128 as int).
    sqrtP_x96:
        sqrtPriceX96 (Q64.96) as int.

    Returns
    -------
    int | None
        Virtual x reserve in token0 raw units (approx), or None if sqrt is zero.

    Notes
    -----
    This is a v2-like virtual reserve mapping used throughout the repo.

    Examples
    --------
    >>> virt_x(1, 1 << 96)  # sqrtP=1.0 in Q96
    1
    """
    if not sqrtP_x96:
        return None
    return (int(L) * Q96) // int(sqrtP_x96)


def virt_y(L: int, sqrtP_x96: int) -> Optional[int]:
    """
    Compute virtual reserve y = L * sqrt(P) in Q64.96 convention.

    Parameters
    ----------
    L:
        Active liquidity (uint128 as int).
    sqrtP_x96:
        sqrtPriceX96 (Q64.96) as int.

    Returns
    -------
    int | None
        Virtual y reserve in token1 raw units (approx), or None if sqrt is zero.

    Notes
    -----
    This is a v2-like virtual reserve mapping used throughout the repo.

    Examples
    --------
    >>> virt_y(1, 1 << 96)  # sqrtP=1.0 in Q96
    1
    """
    if not sqrtP_x96:
        return None
    return (int(L) * int(sqrtP_x96)) // Q96


def ensure_schema_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure output schema columns exist (stability for downstream scripts).

    Parameters
    ----------
    df:
        DataFrame containing event rows.

    Returns
    -------
    pd.DataFrame
        Same DataFrame, with missing columns added (filled with None).

    Notes
    -----
    This keeps the dataset compatible with existing CSV readers and scripts.

    Examples
    --------
    >>> import pandas as pd
    >>> out = ensure_schema_columns(pd.DataFrame([{\"eventType\":\"Swap\",\"blockNumber\":1,\"logIndex\":0,\"timestamp\":0,\"transactionHash\":\"0x\"}]))
    >>> \"L_before\" in out.columns
    True
    """
    cols = [
        "eventType",
        "blockNumber",
        "logIndex",
        "timestamp",
        "transactionHash",
        "gasUsed",
        "gasPrice",
        "effectiveGasPrice",
        "origin",
        "sender",
        "owner",
        "recipient",
        "amount0",
        "amount1",
        "sqrtPriceX96_event",
        "tick_event",
        "liquidityAfter_event",
        "tickLower",
        "tickUpper",
        "liquidityDelta",
        "L_before",
        "sqrt_before",
        "tick_before",
        "x_before",
        "y_before",
        "L_after",
        "sqrt_after",
        "tick_after",
        "x_after",
        "y_after",
        "affectsActive",
        "deltaL_applied",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols]


def compute_running_state(
    df: pd.DataFrame, cur_L: int, cur_sqrt: int, cur_tick: int
) -> Tuple[pd.DataFrame, int, int, int]:
    """
    Compute running state columns (before/after) for an ordered event DataFrame.

    Parameters
    ----------
    df:
        DataFrame sorted by (blockNumber, logIndex), containing event rows.
    cur_L, cur_sqrt, cur_tick:
        Initial pool state at the block immediately before the first row.

    Returns
    -------
    (df, new_L, new_sqrt, new_tick):
        DataFrame with derived columns filled, and the final pool state after
        processing all rows.

    Notes
    -----
    - For Mint/Burn, active liquidity changes only if the current tick is inside
      [tickLower, tickUpper).
    - For Swap, active liquidity is set from Swap.liquidityAfter_event, and
      sqrt/tick are set from the swap event.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame([{\"eventType\":\"Mint\",\"blockNumber\":1,\"logIndex\":0,\"tickLower\":0,\"tickUpper\":10,\"liquidityDelta\":5}]).sort_values([\"blockNumber\",\"logIndex\"])
    >>> df2, L2, sp2, tk2 = compute_running_state(df, cur_L=1, cur_sqrt=1<<96, cur_tick=5)
    >>> int(df2.iloc[0][\"L_after\"]) == 6
    True
    """
    pre_L: List[int] = []
    pre_sqrt: List[int] = []
    pre_tick: List[int] = []
    post_L: List[int] = []
    post_sqrt: List[int] = []
    post_tick: List[int] = []
    x_before: List[Optional[int]] = []
    y_before: List[Optional[int]] = []
    x_after: List[Optional[int]] = []
    y_after: List[Optional[int]] = []
    affects_active: List[Optional[bool]] = []
    delta_applied: List[Optional[int]] = []

    curL, curSP, curTk = int(cur_L), int(cur_sqrt), int(cur_tick)

    for _, row in df.iterrows():
        etype = row["eventType"]

        pre_L.append(curL)
        pre_sqrt.append(curSP)
        pre_tick.append(curTk)
        x_before.append(virt_x(curL, curSP))
        y_before.append(virt_y(curL, curSP))

        if etype in ("Mint", "Burn"):
            hit = bool(int(row["tickLower"]) <= curTk < int(row["tickUpper"]))
            affects_active.append(hit)
            dL = int(row["liquidityDelta"]) if hit else 0
            delta_applied.append(int(dL))
            curL = curL + dL
            post_L.append(curL)
            post_sqrt.append(curSP)
            post_tick.append(curTk)
        else:  # Swap
            affects_active.append(None)
            delta_applied.append(None)

            liq_after = row.get("liquidityAfter_event")
            if liq_after is None or (isinstance(liq_after, float) and pd.isna(liq_after)):
                raise ValueError("Missing liquidityAfter_event for Swap row; cannot compute running state.")

            curL = int(liq_after)
            curSP = int(row["sqrtPriceX96_event"])
            curTk = int(row["tick_event"])

            post_L.append(curL)
            post_sqrt.append(curSP)
            post_tick.append(curTk)

        x_after.append(virt_x(post_L[-1], post_sqrt[-1]))
        y_after.append(virt_y(post_L[-1], post_sqrt[-1]))

    df["L_before"] = pre_L
    df["sqrt_before"] = pre_sqrt
    df["tick_before"] = pre_tick
    df["x_before"] = x_before
    df["y_before"] = y_before
    df["L_after"] = post_L
    df["sqrt_after"] = post_sqrt
    df["tick_after"] = post_tick
    df["x_after"] = x_after
    df["y_after"] = y_after
    df["affectsActive"] = affects_active
    df["deltaL_applied"] = delta_applied

    return df, int(curL), int(curSP), int(curTk)


def csv_append(df: pd.DataFrame, path: str) -> None:
    """
    Append a DataFrame to a CSV file (writes header only once).

    Parameters
    ----------
    df:
        DataFrame to append.
    path:
        Target CSV path.

    Returns
    -------
    None

    Notes
    -----
    The header is written only if the target file does not exist.

    Examples
    --------
    >>> import pandas as pd
    >>> csv_append(pd.DataFrame([{\"a\":1}]), \"/tmp/example_append.csv\")
    """
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=header, index=False)


def _tail_last_csv_row(path: str) -> Optional[Dict[str, str]]:
    """
    Read the last non-empty row of a CSV (no pandas, O(1) seek).

    Parameters
    ----------
    path:
        CSV file path.

    Returns
    -------
    dict | None
        Mapping column->string value for the last row, or None if file has no data rows.

    Notes
    -----
    Assumes fields are simple (no embedded newlines/commas). This holds for the
    datasets produced by this repo.
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None

    with open(path, "rb") as fh:
        header = fh.readline().decode("utf-8").rstrip("\n")
        cols = header.split(",")
        # seek from end to find last newline
        fh.seek(0, os.SEEK_END)
        end = fh.tell()
        if end <= len(header) + 1:
            return None
        # read last up to 64k bytes
        read_size = min(65_536, end)
        fh.seek(end - read_size)
        chunk = fh.read(read_size)
        lines = chunk.splitlines()
        if not lines:
            return None
        # last line might be empty if file ends with newline
        last = lines[-1].decode("utf-8").strip()
        if not last and len(lines) >= 2:
            last = lines[-2].decode("utf-8").strip()
        if not last:
            return None
        values = last.split(",")
        if len(values) != len(cols):
            return None
        return dict(zip(cols, values))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """
    Parse CLI arguments.

    Parameters
    ----------
    argv:
        Optional argv list (defaults to sys.argv).

    Returns
    -------
    argparse.Namespace
        Parsed arguments.

    Notes
    -----
    Defaults are chosen to make `python scripts/data_fetch.py` runnable without
    extra flags (endpoints can be set via env vars).

    Examples
    --------
    >>> ns = parse_args([\"--pool\", \"0x0000000000000000000000000000000000000000\"])
    >>> hasattr(ns, \"pool\")
    True
    """
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"

    p = argparse.ArgumentParser(description="Subgraph-first Uniswap v3 pool harvester (CSV + checkpoint)")
    p.add_argument("--pool", default=DEFAULT_POOL_ADDR, help="Pool address (0x...)")
    p.add_argument("--start-ts", default=DEFAULT_START_TS, help="Start timestamp (unix seconds or ISO8601)")
    p.add_argument("--end-ts", default=DEFAULT_END_TS, help="End timestamp (unix seconds or ISO8601)")

    p.add_argument(
        "--graph-url",
        default=os.environ.get("UNIV3_GRAPH_URL", "").strip() or DEFAULT_GRAPH_URL,
        help="Subgraph GraphQL endpoint (env: UNIV3_GRAPH_URL)",
    )
    p.add_argument(
        "--rpc-urls",
        default=(os.environ.get("MEV_RPC_URLS") or os.environ.get("WEB3_PROVIDER_URI") or "").strip()
        or " ".join(DEFAULT_RPC_URLS),
        help="Space-separated RPC URLs (env: MEV_RPC_URLS or WEB3_PROVIDER_URI)",
    )

    p.add_argument(
        "--out-csv",
        default=str(data_dir / "raw" / f"univ3_{DEFAULT_POOL_ADDR}.csv"),
        help="Output CSV path",
    )
    p.add_argument(
        "--checkpoint",
        default=str(data_dir / "checkpoints" / "univ3_checkpoint.json"),
        help="Checkpoint JSON path",
    )

    p.add_argument("--subgraph-page-size", type=int, default=DEFAULT_SUBGRAPH_PAGE_SIZE)
    p.add_argument("--flush-every-events", type=int, default=DEFAULT_FLUSH_EVERY_EVENTS)
    p.add_argument("--rpc-swap-log-chunk-blocks", type=int, default=DEFAULT_RPC_SWAP_CHUNK_BLOCKS)
    p.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=15.0,
        help="Print a progress heartbeat while streaming events (0 disables).",
    )
    p.add_argument(
        "--max-events",
        type=int,
        default=0,
        help="If >0, stop after writing this many events (debug/smoke runs).",
    )
    p.add_argument(
        "--strict-amount-conversion",
        action="store_true",
        default=True,
        help="Require exact BigDecimal->raw conversion using token decimals (recommended).",
    )
    p.add_argument(
        "--no-strict-amount-conversion",
        action="store_false",
        dest="strict_amount_conversion",
        help="Truncate BigDecimal->raw conversion toward zero if not integral (not recommended).",
    )

    args = p.parse_args(argv)
    # Make output paths pool-dependent unless user explicitly set them.
    pool_checksum = Web3.to_checksum_address(args.pool)
    pool_lower = pool_checksum.lower()
    if args.out_csv.endswith(f"univ3_{DEFAULT_POOL_ADDR}.csv"):
        args.out_csv = str(data_dir / "raw" / f"univ3_{pool_lower}.csv")
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    """
    Main entry point.

    Parameters
    ----------
    argv:
        Optional argv list (defaults to sys.argv).

    Returns
    -------
    None

    Notes
    -----
    Intended usage is via environment variables:
      - UNIV3_GRAPH_URL
      - MEV_RPC_URLS

    Examples
    --------
    >>> isinstance(main, object)
    True
    """
    args = parse_args(argv)

    pool_checksum = Web3.to_checksum_address(args.pool)
    pool_lower = pool_checksum.lower()
    start_ts = to_unix(args.start_ts)
    end_ts = to_unix(args.end_ts)
    if end_ts < start_ts:
        raise ValueError("end-ts must be >= start-ts")

    out_csv = str(Path(args.out_csv))
    ckpt_path = str(Path(args.checkpoint))
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)

    # Print config first so the user sees something immediately.
    print(f"Pool: {pool_checksum}", flush=True)
    print(f"Window: {pd.to_datetime(start_ts, unit='s', utc=True)} -> {pd.to_datetime(end_ts, unit='s', utc=True)}", flush=True)
    print(f"Graph: {args.graph_url}", flush=True)
    print(f"RPC URLs: {len(args.rpc_urls.split())} endpoint(s)", flush=True)
    print(f"Output: {out_csv}", flush=True)
    print(f"Checkpoint: {ckpt_path}", flush=True)
    print(f"Settings: page={args.subgraph_page_size}, flush={args.flush_every_events}, rpc_chunk_blocks={args.rpc_swap_log_chunk_blocks}", flush=True)

    # Subgraph + RPC setup
    sg = SubgraphClient(args.graph_url)
    print("Connecting to RPC…", flush=True)
    w3 = create_multi_provider_web3(args.rpc_urls, request_kwargs={"timeout": 60.0})
    if w3.eth.chain_id != 1:
        raise RuntimeError(f"Connected chain is not Ethereum mainnet (chain_id={w3.eth.chain_id}).")
    print("RPC connected (mainnet).", flush=True)

    # Quarantine-aware direct JSON-RPC caller used for eth_getLogs (swap liquidity).
    # This avoids repeatedly hitting an endpoint that returns HTTP 429/Retry-After.
    rpc = QuarantinedRPC(args.rpc_urls.split())

    ckpt = load_checkpoint(ckpt_path)

    # Resume-safe cursors and running state.
    cur_swap = Cursor()
    cur_mint = Cursor()
    cur_burn = Cursor()
    cur_L: Optional[int] = None
    cur_sqrt: Optional[int] = None
    cur_tick: Optional[int] = None
    token0_decimals: Optional[int] = None
    token1_decimals: Optional[int] = None
    events_written = 0
    last_written_key: Optional[Tuple[int, int]] = None

    def ckpt_matches(c: Dict[str, Any]) -> bool:
        return (
            c.get("ckpt_version") == 2
            and str(c.get("pool", "")).lower() == pool_lower
            and int(c.get("start_ts", -1)) == int(start_ts)
            and int(c.get("end_ts", -1)) == int(end_ts)
            and str(c.get("graph_url", "")).strip() == str(args.graph_url).strip()
        )

    resuming = bool(ckpt and ckpt_matches(ckpt))

    if resuming:
        print("Resuming from checkpoint.", flush=True)
        cur_swap.last_id = str(ckpt.get("cursor", {}).get("swap_last_id", "") or "")
        cur_mint.last_id = str(ckpt.get("cursor", {}).get("mint_last_id", "") or "")
        cur_burn.last_id = str(ckpt.get("cursor", {}).get("burn_last_id", "") or "")
        cur_L = int(ckpt["cur_L"])
        cur_sqrt = int(ckpt["cur_sqrt"])
        cur_tick = int(ckpt["cur_tick"])
        token0_decimals = ckpt.get("token0_decimals")
        token1_decimals = ckpt.get("token1_decimals")
        events_written = int(ckpt.get("events_written", 0))
        if ckpt.get("last_written", None):
            lw = ckpt["last_written"]
            try:
                last_written_key = (int(lw["blockNumber"]), int(lw["logIndex"]))
            except Exception:
                last_written_key = None
    else:
        print("No matching checkpoint found; starting fresh.", flush=True)
        # Avoid accidentally appending to an old output file from a different run.
        if os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
            ts = _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            bak = out_csv + f".bak_{ts}"
            os.replace(out_csv, bak)
            print(f"  ⚠️  Existing output moved aside: {bak}", flush=True)
        # Same for a mismatching checkpoint (keep it for debugging).
        if ckpt and os.path.exists(ckpt_path) and os.path.getsize(ckpt_path) > 0:
            ts = _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            bak = ckpt_path + f".bak_{ts}"
            os.replace(ckpt_path, bak)
            print(f"  ⚠️  Existing checkpoint moved aside: {bak}", flush=True)

    # If resuming, prefer the last row of the CSV as the source of truth for the
    # last-written key and state. This protects against crashes between CSV append
    # and checkpoint save.
    if resuming:
        if not os.path.exists(out_csv):
            if events_written == 0:
                # Checkpoint was saved before any rows were written (e.g., crash
                # during first flush).  Safe to treat as a fresh start.
                print("Checkpoint has 0 events and CSV is missing; restarting fresh.", flush=True)
                resuming = False
            else:
                raise RuntimeError("Checkpoint exists but output CSV is missing; cannot resume safely.")

        last_row = _tail_last_csv_row(out_csv)
        if last_row:
            try:
                file_key = (int(last_row["blockNumber"]), int(last_row["logIndex"]))
                file_L = int(last_row["L_after"])
                file_sqrt = int(last_row["sqrt_after"])
                file_tick = int(last_row["tick_after"])
                if last_written_key is None or file_key > last_written_key:
                    last_written_key = file_key
                    cur_L, cur_sqrt, cur_tick = file_L, file_sqrt, file_tick
                print(f"Last CSV row key: block={file_key[0]}, logIndex={file_key[1]} (state recovered from file).", flush=True)
            except Exception:
                pass

    if token0_decimals is None or token1_decimals is None or cur_L is None or cur_sqrt is None or cur_tick is None:
        # Initialize state/decimals from subgraph (fallback to RPC if needed).
        first_block = find_first_event_block(sg, pool_lower, start_ts)
        if first_block is None:
            print("No events found in the requested window; nothing to do.", flush=True)
            return
        prev_block = max(int(first_block) - 1, 0)
        print(f"Initializing state at block {prev_block} (first event block is {first_block}).", flush=True)

        try:
            st = fetch_pool_state_and_decimals_at_block(sg, pool_lower, prev_block)
            cur_L, cur_sqrt, cur_tick = st.liquidity, st.sqrt_price_x96, st.tick
            token0_decimals, token1_decimals = st.token0_decimals, st.token1_decimals
        except Exception as exc:
            print(f"⚠️  Subgraph pool-at-block query failed ({exc}); falling back to RPC calls.", flush=True)
            # RPC fallback: slot0/liquidity at prev_block
            POOL_STATE_ABI = [
                {
                    "name": "slot0",
                    "inputs": [],
                    "outputs": [
                        {"type": "uint160", "name": "sqrtPriceX96"},
                        {"type": "int24", "name": "tick"},
                        {"type": "uint16", "name": "observationIndex"},
                        {"type": "uint16", "name": "observationCardinality"},
                        {"type": "uint16", "name": "observationCardinalityNext"},
                        {"type": "uint8", "name": "feeProtocol"},
                        {"type": "bool", "name": "unlocked"},
                    ],
                    "stateMutability": "view",
                    "type": "function",
                },
                {
                    "name": "liquidity",
                    "inputs": [],
                    "outputs": [{"type": "uint128"}],
                    "stateMutability": "view",
                    "type": "function",
                },
            ]
            state_c = w3.eth.contract(address=pool_checksum, abi=POOL_STATE_ABI)
            s0 = state_c.functions.slot0().call(block_identifier=prev_block)
            L0 = state_c.functions.liquidity().call(block_identifier=prev_block)
            cur_sqrt, cur_tick, cur_L = int(s0[0]), int(s0[1]), int(L0)

            # decimals fallback: token0/token1 decimals
            POOL_INFO_ABI = [
                {
                    "inputs": [],
                    "name": "token0",
                    "outputs": [{"internalType": "address", "name": "", "type": "address"}],
                    "stateMutability": "view",
                    "type": "function",
                },
                {
                    "inputs": [],
                    "name": "token1",
                    "outputs": [{"internalType": "address", "name": "", "type": "address"}],
                    "stateMutability": "view",
                    "type": "function",
                },
            ]
            ERC20_DECIMALS_ABI = [
                {
                    "inputs": [],
                    "name": "decimals",
                    "outputs": [{"internalType": "uint8", "name": "", "type": "uint8"}],
                    "stateMutability": "view",
                    "type": "function",
                }
            ]
            pool_contract = w3.eth.contract(address=pool_checksum, abi=POOL_INFO_ABI)
            t0 = pool_contract.functions.token0().call()
            t1 = pool_contract.functions.token1().call()
            token0_decimals = int(w3.eth.contract(address=t0, abi=ERC20_DECIMALS_ABI).functions.decimals().call())
            token1_decimals = int(w3.eth.contract(address=t1, abi=ERC20_DECIMALS_ABI).functions.decimals().call())

        # Start fresh: write a new checkpoint header (does not overwrite CSV).
        ckpt = {
            "ckpt_version": 2,
            "pool": pool_lower,
            "start_ts": int(start_ts),
            "end_ts": int(end_ts),
            "graph_url": str(args.graph_url).strip(),
            "out_csv": out_csv,
            "token0_decimals": int(token0_decimals) if token0_decimals is not None else None,
            "token1_decimals": int(token1_decimals) if token1_decimals is not None else None,
            "cur_L": int(cur_L),
            "cur_sqrt": int(cur_sqrt),
            "cur_tick": int(cur_tick),
            "cursor": {"swap_last_id": "", "mint_last_id": "", "burn_last_id": ""},
            "events_written": int(events_written),
            "last_written": None,
            "updated_at": int(time.time()),
        }
        save_checkpoint_atomic(ckpt_path, ckpt)

    assert cur_L is not None and cur_sqrt is not None and cur_tick is not None
    assert token0_decimals is not None and token1_decimals is not None

    # Stream and flush
    buf: List[Dict[str, Any]] = []
    t0 = time.time()
    last_print = t0
    last_heartbeat = t0
    heartbeat_s = float(args.heartbeat_seconds or 0.0)
    streamed_total = 0

    stream = merged_event_stream(
        sg,
        pool_lower,
        int(start_ts),
        int(end_ts),
        int(args.subgraph_page_size),
        cur_swap,
        cur_mint,
        cur_burn,
    )

    def flush_buffer() -> None:
        nonlocal buf, cur_L, cur_sqrt, cur_tick, events_written, last_written_key, last_print

        if not buf:
            return

        # Convert buffer to DataFrame and enforce ordering.
        df = pd.DataFrame(buf)
        df = df.sort_values(["blockNumber", "logIndex"]).reset_index(drop=True)
        # Defensive: subgraph paging glitches can yield duplicates. A pool log is
        # uniquely identified by (blockNumber, logIndex).
        df = df.drop_duplicates(subset=["blockNumber", "logIndex"], keep="first").reset_index(drop=True)

        print(
            f"Flushing {len(df):,} events (blocks {int(df['blockNumber'].min()):,}–{int(df['blockNumber'].max()):,})…",
            flush=True,
        )

        # Skip anything already written (resume robustness).
        if last_written_key is not None:
            b0, l0 = last_written_key
            mask = (df["blockNumber"] > b0) | ((df["blockNumber"] == b0) & (df["logIndex"] > l0))
            df = df.loc[mask].reset_index(drop=True)
            if df.empty:
                buf = []
                return

        # Normalize addresses.
        for col in ("origin", "sender", "recipient", "owner"):
            if col in df.columns:
                df[col] = df[col].map(checksum_or_none)

        # Convert BigDecimal amounts -> raw units (ints).
        def _conv_amount0(x: Any) -> Optional[int]:
            if x is None or (isinstance(x, float) and pd.isna(x)):
                return None
            return to_raw_units(str(x), int(token0_decimals), strict=bool(args.strict_amount_conversion))

        def _conv_amount1(x: Any) -> Optional[int]:
            if x is None or (isinstance(x, float) and pd.isna(x)):
                return None
            return to_raw_units(str(x), int(token1_decimals), strict=bool(args.strict_amount_conversion))

        df["amount0"] = df["amount0"].map(_conv_amount0)
        df["amount1"] = df["amount1"].map(_conv_amount1)

        # Fill Swap liquidity via RPC logs (required for running-state).
        swaps = df["eventType"] == "Swap"
        if swaps.any():
            start_block = int(df.loc[swaps, "blockNumber"].min())
            end_block = int(df.loc[swaps, "blockNumber"].max())
            n_swaps = int(swaps.sum())
            print(f"  RPC: fetching Swap.liquidity for {n_swaps:,} swaps (blocks {start_block:,}–{end_block:,})…", flush=True)
            liq_map = fetch_swap_liquidity_map(
                w3,
                pool_checksum,
                start_block,
                end_block,
                chunk_blocks=int(args.rpc_swap_log_chunk_blocks),
                rpc=rpc,
            )
            # IMPORTANT: keep dtype=object to avoid float rounding of huge uint128 values.
            swap_index = df.index[swaps]
            liq_values = []
            for r in df.loc[swaps, ["blockNumber", "logIndex"]].itertuples(index=False):
                liq_values.append(liq_map.get((int(r.blockNumber), int(r.logIndex))))
            df.loc[swap_index, "liquidityAfter_event"] = pd.Series(liq_values, index=swap_index, dtype=object)

            missing = int(df.loc[swaps, "liquidityAfter_event"].isna().sum())
            if missing:
                # Fallback: per-block fetch for blocks that still have missing swaps.
                miss_blocks = sorted(df.loc[swaps & df["liquidityAfter_event"].isna(), "blockNumber"].unique().tolist())
                for bn in miss_blocks:
                    # Reuse the already-decoded range fetcher for this single block (cheap).
                    one = fetch_swap_liquidity_map(w3, pool_checksum, int(bn), int(bn), chunk_blocks=1, rpc=rpc)
                    for (bb, li), v in one.items():
                        liq_map[(bb, li)] = v
                    # Re-fill missing swaps for that block using dtype=object.
                    block_mask = swaps & (df["blockNumber"] == bn) & df["liquidityAfter_event"].isna()
                    if block_mask.any():
                        idx = df.index[block_mask]
                        vals = []
                        for r in df.loc[block_mask, ["blockNumber", "logIndex"]].itertuples(index=False):
                            vals.append(liq_map.get((int(r.blockNumber), int(r.logIndex))))
                        df.loc[idx, "liquidityAfter_event"] = pd.Series(vals, index=idx, dtype=object)

            filled = int(df.loc[swaps, "liquidityAfter_event"].notna().sum())
            still_missing = int(df.loc[swaps, "liquidityAfter_event"].isna().sum())

            # Last-resort fallback: use the Web3 multi-provider (independent from
            # QuarantinedRPC) for any blocks still missing after quarantine exhaustion.
            if still_missing:
                print(f"  RPC: {still_missing:,} swaps still missing; trying w3 provider fallback…", flush=True)
                miss_blocks2 = sorted(
                    df.loc[swaps & df["liquidityAfter_event"].isna(), "blockNumber"].unique().tolist()
                )
                for bn in miss_blocks2:
                    try:
                        one = fetch_swap_liquidity_map(
                            w3, pool_checksum, int(bn), int(bn), chunk_blocks=1, rpc=None,
                        )
                        for (bb, li), v in one.items():
                            liq_map[(bb, li)] = v
                        block_mask = swaps & (df["blockNumber"] == bn) & df["liquidityAfter_event"].isna()
                        if block_mask.any():
                            idx = df.index[block_mask]
                            vals = [
                                liq_map.get((int(r.blockNumber), int(r.logIndex)))
                                for r in df.loc[block_mask, ["blockNumber", "logIndex"]].itertuples(index=False)
                            ]
                            df.loc[idx, "liquidityAfter_event"] = pd.Series(vals, index=idx, dtype=object)
                    except Exception as exc:
                        print(f"  ⚠️  w3 fallback failed for block {bn}: {exc}", flush=True)

                filled = int(df.loc[swaps, "liquidityAfter_event"].notna().sum())
                still_missing = int(df.loc[swaps, "liquidityAfter_event"].isna().sum())

            if still_missing:
                print(f"  RPC: filled {filled:,}/{n_swaps:,} swap liquidity (missing {still_missing:,})", flush=True)
            else:
                print(f"  RPC: filled {filled:,}/{n_swaps:,} swap liquidity", flush=True)

        # Ensure gas fields exist (left empty; enrich later via scripts/add_gas.py).
        for c in ("gasUsed", "gasPrice", "effectiveGasPrice"):
            if c not in df.columns:
                df[c] = None

        # Running state columns
        df, cur_L, cur_sqrt, cur_tick = compute_running_state(df, int(cur_L), int(cur_sqrt), int(cur_tick))

        # Schema stability + append
        df_out = ensure_schema_columns(df)
        csv_append(df_out, out_csv)

        events_written += int(len(df_out))
        last_row = df_out.iloc[-1]
        last_written_key = (int(last_row["blockNumber"]), int(last_row["logIndex"]))

        # Checkpoint update (cursor ids correspond to last *streamed* events,
        # but we only persist at flush boundaries so they are safe to resume).
        ckpt_payload = {
            "ckpt_version": 2,
            "pool": pool_lower,
            "start_ts": int(start_ts),
            "end_ts": int(end_ts),
            "graph_url": str(args.graph_url).strip(),
            "out_csv": out_csv,
            "token0_decimals": int(token0_decimals),
            "token1_decimals": int(token1_decimals),
            "cur_L": int(cur_L),
            "cur_sqrt": int(cur_sqrt),
            "cur_tick": int(cur_tick),
            "cursor": {
                "swap_last_id": str(cur_swap.last_id or ""),
                "mint_last_id": str(cur_mint.last_id or ""),
                "burn_last_id": str(cur_burn.last_id or ""),
            },
            "events_written": int(events_written),
            "last_written": {"blockNumber": int(last_row["blockNumber"]), "logIndex": int(last_row["logIndex"])},
            "updated_at": int(time.time()),
        }
        save_checkpoint_atomic(ckpt_path, ckpt_payload)

        # Progress print
        now = time.time()
        if now - last_print > 5.0:
            elapsed = now - t0
            rate = events_written / elapsed if elapsed > 0 else 0.0
            print(f"  ✓ wrote {len(df_out):,} rows (total {events_written:,}) | rate ~{rate:,.0f} rows/s", flush=True)
            last_print = now

        buf = []

    max_events = int(args.max_events or 0)

    try:
        print("Streaming events from subgraph…", flush=True)
        for etype, ev in stream:
            streamed_total += 1
            bn = int(ev["transaction"]["blockNumber"])
            li = int(ev.get("logIndex") or 0)

            # Skip anything already written (resume robustness).
            if last_written_key is not None and (bn, li) <= last_written_key:
                continue

            base = {
                "blockNumber": bn,
                "logIndex": li,
                "timestamp": int(ev["timestamp"]),
                "transactionHash": str(ev["transaction"]["id"]),
                "origin": ev.get("origin"),
                "gasUsed": None,
                "gasPrice": None,
                "effectiveGasPrice": None,
            }

            if etype == "swap":
                row = {
                    **base,
                    "eventType": "Swap",
                    "sender": ev.get("sender"),
                    "owner": None,
                    "recipient": ev.get("recipient"),
                    "amount0": ev.get("amount0"),
                    "amount1": ev.get("amount1"),
                    "sqrtPriceX96_event": int(ev["sqrtPriceX96"]),
                    "tick_event": int(ev["tick"]),
                    "liquidityAfter_event": None,  # filled via RPC logs on flush
                    "tickLower": None,
                    "tickUpper": None,
                    "liquidityDelta": None,
                }
            elif etype == "mint":
                row = {
                    **base,
                    "eventType": "Mint",
                    "sender": ev.get("sender"),
                    "owner": ev.get("owner"),
                    "recipient": None,
                    "amount0": ev.get("amount0"),
                    "amount1": ev.get("amount1"),
                    "sqrtPriceX96_event": None,
                    "tick_event": None,
                    "liquidityAfter_event": None,
                    "tickLower": int(ev["tickLower"]),
                    "tickUpper": int(ev["tickUpper"]),
                    "liquidityDelta": int(ev["amount"]),
                }
            else:  # burn
                row = {
                    **base,
                    "eventType": "Burn",
                    "sender": None,
                    "owner": ev.get("owner"),
                    "recipient": None,
                    "amount0": ev.get("amount0"),
                    "amount1": ev.get("amount1"),
                    "sqrtPriceX96_event": None,
                    "tick_event": None,
                    "liquidityAfter_event": None,
                    "tickLower": int(ev["tickLower"]),
                    "tickUpper": int(ev["tickUpper"]),
                    "liquidityDelta": -int(ev["amount"]),
                }

            buf.append(row)

            if heartbeat_s > 0:
                now = time.time()
                if now - last_heartbeat >= heartbeat_s:
                    ts = int(ev.get("timestamp") or 0)
                    ts_s = str(pd.to_datetime(ts, unit="s", utc=True)) if ts else "?"
                    print(
                        f"  … streamed {streamed_total:,} events | buffered {len(buf):,} | written {events_written:,} | last {bn:,}:{li} @ {ts_s}",
                        flush=True,
                    )
                    last_heartbeat = now

            if len(buf) >= int(args.flush_every_events):
                flush_buffer()
                if max_events and events_written >= max_events:
                    break
    finally:
        flush_buffer()

    elapsed = time.time() - t0
    rate = events_written / elapsed if elapsed > 0 else 0.0
    print(f"✅ Completed. Total rows written: {events_written:,} | elapsed: {elapsed:.1f}s | avg rate: {rate:,.0f} rows/s", flush=True)
    print(f"Output CSV: {out_csv}", flush=True)


if __name__ == "__main__":
    main()
