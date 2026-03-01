#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Incrementally fill the `origin` (tx sender) field for Uniswap v3 event rows.

Summary
-------
Given a CSV containing Uniswap v3 events (with a `transactionHash` column), this
script resolves the sender address for each unique transaction via
`eth_getTransaction` and writes/updates an output CSV with an `origin` column.

Parameters
----------
--in:
    Input CSV path. Must contain `transactionHash`.
--out:
    Output CSV path. If it exists, the script resumes by filling missing origins.
--checkpoint:
    JSON checkpoint path used for safe resume after interruption.
--batch-size:
    Unique tx hashes resolved per batch.
--workers:
    Number of parallel threads used for RPC calls (I/O-bound).
--save-every:
    Persist checkpoint + partial CSV every N processed tx hashes.

Returns
-------
Writes `--out` CSV to disk and (on successful completion) deletes `--checkpoint`.

Notes
-----
- This is I/O-bound (RPC calls), so we use a `ThreadPoolExecutor`.
- Determinism: output ordering is inherited from the input CSV; within a batch,
  RPC completion order does not matter because we map results by tx hash.
- Defaults are repo-relative (safe to run from any working directory).

Examples
--------
conda activate main
python scripts/add_origin.py \
  --in data/processed/univ3_pool_events_with_running_state.csv \
  --out data/processed/univ3_pool_events_with_origin.csv
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence, TypedDict

import pandas as pd

# Optional/heavy dependencies (`web3`, `eth_defi`) are imported lazily so that
# `python scripts/add_origin.py --help` works even before installing them.


REPO_ROOT = Path(__file__).resolve().parents[1]

# Default RPCs are public and rate-limited; override by setting `MEV_RPC_URLS`
# (space-separated list; same format used by eth-defi multi-provider) or
# `WEB3_PROVIDER_URI` (single endpoint).
DEFAULT_RPC_URLS: Sequence[str] = (
    "https://eth.llamarpc.com",
    "https://rpc.ankr.com/eth",
)

DEFAULT_INPUT_CSV = REPO_ROOT / "data" / "processed" / "univ3_pool_events_with_running_state.csv"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "data" / "processed" / "univ3_pool_events_with_origin.csv"
DEFAULT_CHECKPOINT = REPO_ROOT / "data" / "checkpoints" / "origin_fetch_checkpoint.json"


class Checkpoint(TypedDict):
    processed_tx_count: int
    pending_tx_hashes: List[str]
    tx_cache: Dict[str, Optional[str]]


def _build_mainnet_config(urls: Sequence[str], *, timeout: float = 5.0) -> str:
    """
    Filter RPC URLs to those reachable on Ethereum mainnet.

    Parameters
    ----------
    urls:
        Candidate HTTP RPC URLs.
    timeout:
        Per-endpoint timeout (seconds) for reachability checks.

    Returns
    -------
    str
        Space-separated endpoints usable by `create_multi_provider_web3`.

    Notes
    -----
    - We probe `eth_chainId` to ensure we are on mainnet (chain_id == 1).
    - If none are reachable, we raise to avoid silently running on the wrong network.
    """
    from web3 import Web3

    ok: List[str] = []
    for url in urls:
        try:
            tmp = Web3(Web3.HTTPProvider(url, request_kwargs={"timeout": timeout}))
            if tmp.eth.chain_id == 1:
                ok.append(url)
        except Exception:
            continue
    if not ok:
        raise RuntimeError("No Ethereum mainnet endpoints are reachable.")
    return " ".join(ok)


def _get_web3() -> Web3:
    """
    Create a Web3 instance with multi-endpoint failover.

    Returns
    -------
    web3.Web3
        Web3 connected to Ethereum mainnet.

    Notes
    -----
    - Uses `MEV_RPC_URLS` (space-separated) or `WEB3_PROVIDER_URI` (single endpoint) if set;
      otherwise probes `DEFAULT_RPC_URLS`.
    """
    rpc_line = os.environ.get("MEV_RPC_URLS") or os.environ.get("WEB3_PROVIDER_URI")
    from eth_defi.provider.multi_provider import create_multi_provider_web3

    if rpc_line and rpc_line.strip():
        w3 = create_multi_provider_web3(rpc_line.strip(), request_kwargs={"timeout": 30.0})
    else:
        rpc_line = _build_mainnet_config(DEFAULT_RPC_URLS)
        w3 = create_multi_provider_web3(rpc_line, request_kwargs={"timeout": 30.0})
    if w3.eth.chain_id != 1:
        raise RuntimeError(f"Connected chain is not Ethereum mainnet (chain_id={w3.eth.chain_id}).")
    return w3


def _load_checkpoint(path: Path) -> Checkpoint:
    """
    Load a resume checkpoint from disk.

    Parameters
    ----------
    path:
        Path to the checkpoint JSON.

    Returns
    -------
    Checkpoint
        Parsed checkpoint (with backward-compatible defaults).

    Notes
    -----
    - Older formats may store `processed_rows`; we map it to `processed_tx_count`.
    """
    if not path.exists():
        return {"processed_tx_count": 0, "pending_tx_hashes": [], "tx_cache": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        data = {}
    return {
        "processed_tx_count": int(data.get("processed_tx_count", data.get("processed_rows", 0)) or 0),
        "pending_tx_hashes": list(data.get("pending_tx_hashes", []) or []),
        "tx_cache": dict(data.get("tx_cache", {}) or {}),
    }


def _save_checkpoint(path: Path, processed_tx_count: int, pending_tx_hashes: List[str], tx_cache: Dict[str, Optional[str]]) -> None:
    """
    Persist a resume checkpoint to disk.

    Parameters
    ----------
    path:
        Where to write the checkpoint.
    processed_tx_count:
        How many unique tx hashes have been processed.
    pending_tx_hashes:
        Remaining tx hashes to process.
    tx_cache:
        Cache mapping tx hash -> origin (or None on failure).

    Returns
    -------
    None
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "processed_tx_count": int(processed_tx_count),
        "pending_tx_hashes": pending_tx_hashes,
        "tx_cache": tx_cache,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    print(f"  💾 Checkpoint saved — processed {processed_tx_count} / total so far")


def _canonical_missing_mask(series: pd.Series) -> pd.Series:
    """
    Identify missing/empty origin entries.

    Parameters
    ----------
    series:
        Pandas Series for the `origin` column.

    Returns
    -------
    pd.Series
        Boolean mask (True where origin is missing/empty).
    """
    if series.dtype == object:
        s = series.astype(str)
        return series.isna() | (s.str.len() == 0) | (s.str.lower().isin(["none", "nan"]))
    return series.isna()


def _determine_work(df: pd.DataFrame) -> List[str]:
    """
    Compute the list of unique transaction hashes missing an origin.

    Parameters
    ----------
    df:
        Input dataframe. Must have `transactionHash`; may or may not have `origin`.

    Returns
    -------
    list[str]
        Unique tx hashes that still need fetching.
    """
    if "transactionHash" not in df.columns:
        raise ValueError("Input CSV must contain a 'transactionHash' column.")

    if "origin" not in df.columns:
        df["origin"] = None
        return df.loc[df["transactionHash"].notna(), "transactionHash"].astype(str).unique().tolist()

    mask_missing = _canonical_missing_mask(df["origin"]) & df["transactionHash"].notna()
    return df.loc[mask_missing, "transactionHash"].astype(str).unique().tolist()


def _fetch_transaction_origin(w3: Web3, tx_hash: str, tx_cache: Dict[str, Optional[str]], *, retry_count: int = 3) -> Optional[str]:
    """
    Fetch tx sender (`from`) for a transaction hash.

    Parameters
    ----------
    w3:
        Web3 instance (mainnet).
    tx_hash:
        0x-prefixed transaction hash.
    tx_cache:
        Cache mapping tx hash -> origin (or None on previous failures).
    retry_count:
        Retry attempts on transient RPC failures.

    Returns
    -------
    str | None
        Origin address if resolved, else None.
    """
    if tx_hash in tx_cache:
        return tx_cache[tx_hash]

    for attempt in range(retry_count):
        try:
            tx = w3.eth.get_transaction(tx_hash)
            origin = tx.get("from")
            tx_cache[tx_hash] = origin
            return origin
        except Exception as exc:  # noqa: BLE001
            if attempt < retry_count - 1:
                time.sleep(0.5 * (attempt + 1))  # simple backoff
                continue
            print(f"    ⚠️  Failed to fetch origin for {tx_hash}: {str(exc)[:120]}")
            tx_cache[tx_hash] = None
            return None


def _batch_fetch_origins(
    w3: Web3,
    tx_hashes: List[str],
    tx_cache: Dict[str, Optional[str]],
    *,
    max_workers: int,
) -> Dict[str, Optional[str]]:
    """
    Resolve a batch of origins in parallel.

    Parameters
    ----------
    w3:
        Web3 instance.
    tx_hashes:
        Batch of tx hashes to resolve.
    tx_cache:
        Shared cache to reuse already-fetched values.
    max_workers:
        Thread pool size.

    Returns
    -------
    dict[str, str | None]
        Mapping tx hash -> origin.

    Notes
    -----
    - This function is I/O bound; threads avoid blocking on RPC latency.
    - Errors are captured per-hash and returned as None.
    """
    results: Dict[str, Optional[str]] = {}
    to_fetch: List[str] = []

    for h in tx_hashes:
        if h in tx_cache:
            results[h] = tx_cache[h]
        else:
            to_fetch.append(h)

    if not to_fetch:
        return results

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_transaction_origin, w3, h, tx_cache): h for h in to_fetch}
        for fut in as_completed(futures):
            h = futures[fut]
            try:
                results[h] = fut.result()
            except Exception as exc:  # noqa: BLE001
                print(f"    ⚠️  Error fetching {h}: {str(exc)[:120]}")
                results[h] = None

    return results


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Incrementally add `origin` (tx sender) to an events CSV.")
    p.add_argument("--in", dest="in_path", default=str(DEFAULT_INPUT_CSV), help="Input CSV path.")
    p.add_argument("--out", dest="out_path", default=str(DEFAULT_OUTPUT_CSV), help="Output CSV path.")
    p.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="Checkpoint JSON path.")
    p.add_argument("--batch-size", type=int, default=200, help="Unique tx hashes per batch.")
    p.add_argument("--workers", type=int, default=10, help="Parallel threads for RPC calls.")
    p.add_argument("--save-every", type=int, default=1000, help="Save checkpoint/partial CSV every N processed tx hashes.")
    return p.parse_args()


def main() -> None:
    """
    Run incremental origin enrichment.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    -----
    - If `--out` exists, the script resumes by filling missing origins in that file.
    - A checkpoint is written periodically; delete it to force a full rescan.
    """
    args = _parse_args()
    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    checkpoint_path = Path(args.checkpoint)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    print("🚀 Starting origin fetch (incremental-safe) ...")
    print(f"📄 INPUT_CSV     = {in_path}")
    print(f"📄 OUTPUT_CSV    = {out_path}")
    print(f"🧾 CHECKPOINT    = {checkpoint_path}")
    print(f"⚙️  batch_size={args.batch_size}, workers={args.workers}, save_every={args.save_every}")

    w3 = _get_web3()

    checkpoint = _load_checkpoint(checkpoint_path)
    tx_cache: Dict[str, Optional[str]] = checkpoint.get("tx_cache", {})
    if tx_cache:
        print(f"♻️  Loaded cache with {len(tx_cache):,} tx entries")

    # Use output file as the base if it exists (resume-friendly).
    if out_path.exists():
        print("📂 Found existing OUTPUT_CSV. Loading it to continue filling missing origins ...")
        df = pd.read_csv(out_path, low_memory=False)
    else:
        if not in_path.exists():
            raise SystemExit(f"Input CSV not found: {in_path}")
        print("📂 No OUTPUT_CSV found. Loading INPUT_CSV and starting from scratch ...")
        df = pd.read_csv(in_path, low_memory=False)

    # Determine worklist (either from checkpoint or fresh scan).
    if checkpoint.get("pending_tx_hashes"):
        work_tx_hashes = list(checkpoint["pending_tx_hashes"])
        start_index = int(checkpoint.get("processed_tx_count", 0))
        print(f"🔁 Resuming from checkpoint: {start_index} processed / {len(work_tx_hashes)} pending")
    else:
        work_tx_hashes = _determine_work(df)
        start_index = 0
        if not work_tx_hashes:
            print("✅ Nothing to do — all origins already present.")
            return

    total = len(work_tx_hashes)
    print(f"\n📥 Pending unique tx hashes to resolve: {total:,}")

    t0 = time.time()
    processed_tx = start_index
    for i in range(start_index, total, int(args.batch_size)):
        batch = work_tx_hashes[i : i + int(args.batch_size)]
        batch_num = i // int(args.batch_size) + 1
        total_batches = (total + int(args.batch_size) - 1) // int(args.batch_size)
        print(f"  🔄 Batch {batch_num}/{total_batches} — {len(batch)} txs")

        batch_origins = _batch_fetch_origins(w3, batch, tx_cache, max_workers=int(args.workers))

        # Map into the dataframe only where origin is missing.
        if "origin" not in df.columns:
            df["origin"] = None
        mask_missing = _canonical_missing_mask(df["origin"]) & df["transactionHash"].isin(batch)
        df.loc[mask_missing, "origin"] = df.loc[mask_missing, "transactionHash"].map(batch_origins)

        processed_tx = i + len(batch)

        if processed_tx % int(args.save_every) == 0:
            _save_checkpoint(checkpoint_path, processed_tx, work_tx_hashes[processed_tx:], tx_cache)
            df.to_csv(out_path, index=False)
            print(f"  💽 Partial save to {out_path}")

    print(f"\n💾 Saving CSV to {out_path} ...")
    df.to_csv(out_path, index=False)

    elapsed = time.time() - t0
    total_rows = len(df)
    success = int(df["origin"].notna().sum()) if "origin" in df.columns else 0
    failed = int(df["origin"].isna().sum()) if "origin" in df.columns else total_rows

    print("\n" + "=" * 60)
    print("✅ COMPLETED!")
    print(f"📊 Total rows: {total_rows:,}")
    print(f"✅ Successful origin fetches: {success:,}")
    print(f"❌ Failed origin fetches: {failed:,}")
    print(f"⏱️  Time elapsed: {elapsed:.2f} seconds")
    print(f"📄 Output saved to: {out_path}")
    print("=" * 60)

    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print("🧹 Checkpoint file removed")


if __name__ == "__main__":
    main()
