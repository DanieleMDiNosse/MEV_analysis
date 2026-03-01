#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Incrementally fill gas fields for Uniswap v3 event rows.

This script mirrors `add_origin.py` but fetches:
  • gasUsed                (from transaction receipt)
  • gasPrice               (from transaction; may be None on EIP-1559 txs)
  • effectiveGasPrice      (from transaction receipt; actual paid price)

It:
  - Reads an existing CSV (or continues from a partially-filled OUTPUT_CSV)
  - Finds rows with missing gas fields (any of the three)
  - Resolves unique tx hashes in parallel with multi-endpoint failover
  - Periodically checkpoints progress + writes out partial CSV
  - Is safe to resume after interruption
  - **Checks free disk space (>=30%) every time it saves a checkpoint; if below, exits**

CLI
---
python scripts/add_gas.py \
  --in  /path/to/input.csv \
  --out /path/to/output.csv \
  --checkpoint /path/to/checkpoint.json \
  --batch-size 200 --workers 10 --save-every 1000
"""

import os
import json
import time
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

# Optional/heavy dependencies (`web3`, `eth_defi`) are imported lazily so that
# `python scripts/add_gas.py --help` works even before installing them.


# ---------------- Disk space guard ----------------
MIN_FREE_RATIO = 0.30  # 30%

def _check_disk_space_for_path(path: str, min_free_ratio: float = MIN_FREE_RATIO) -> None:
    """
    Check free space on the filesystem containing `path`.
    If free/total < min_free_ratio, terminate the script.
    """
    dir_path = os.path.abspath(os.path.dirname(path) or ".")
    try:
        usage = shutil.disk_usage(dir_path)
        free_ratio = usage.free / usage.total if usage.total else 1.0
    except Exception as e:
        # If we cannot determine disk usage, fail safe and abort.
        print(f"🛑 Could not determine disk usage for '{dir_path}': {e}")
        raise SystemExit(3)

    if free_ratio < min_free_ratio:
        print(
            f"🛑 Low disk space on '{dir_path}': {free_ratio*100:.1f}% free "
            f"(threshold {min_free_ratio*100:.0f}%). Terminating to protect data integrity."
        )
        raise SystemExit(3)


# ---------------- RPC configuration ----------------
def build_mainnet_config(urls, timeout=5.0) -> str:
    """Return a space-joined string of endpoints that respond on chain_id==1."""
    from web3 import Web3

    ok = []
    for u in urls:
        try:
            tmp = Web3(Web3.HTTPProvider(u, request_kwargs={"timeout": timeout}))
            if tmp.eth.chain_id == 1:
                ok.append(u)
        except Exception:
            continue
    if not ok:
        raise RuntimeError("No Ethereum mainnet endpoints are reachable.")
    return " ".join(ok)


DEFAULT_RPC_URLS = [
    "https://eth.llamarpc.com",
    "https://rpc.ankr.com/eth",
]

def get_web3():
    """
    Create a Web3 instance with multi-endpoint failover.

    Uses `MEV_RPC_URLS` (space-separated) or `WEB3_PROVIDER_URI` (single endpoint) if set;
    otherwise probes `DEFAULT_RPC_URLS`.
    """
    from eth_defi.provider.multi_provider import create_multi_provider_web3

    rpc_line = os.environ.get("MEV_RPC_URLS") or os.environ.get("WEB3_PROVIDER_URI")
    if rpc_line and rpc_line.strip():
        w3_local = create_multi_provider_web3(rpc_line.strip(), request_kwargs={"timeout": 30.0})
    else:
        rpc_line = build_mainnet_config(DEFAULT_RPC_URLS)
        w3_local = create_multi_provider_web3(rpc_line, request_kwargs={"timeout": 30.0})
    if w3_local.eth.chain_id != 1:
        raise RuntimeError(f"Connected chain is not Ethereum mainnet (chain_id={w3_local.eth.chain_id}).")
    return w3_local


# Initialized in `main()` to keep `--help` working without optional deps installed.
w3 = None


# ---------------- Defaults (can be overridden via CLI) ----------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_CSV = REPO_ROOT / "data" / "raw" / "usdc_weth_05.csv"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "data" / "processed" / "univ3_pool_events_with_gas.csv"
DEFAULT_CHECKPOINT = REPO_ROOT / "data" / "checkpoints" / "gas_fetch_checkpoint.json"

DEFAULT_BATCH_SIZE = 100
DEFAULT_WORKERS = 1
DEFAULT_SAVE_EVERY = DEFAULT_BATCH_SIZE*5


# ---------------- In-memory caches ----------------
# tx hash -> {"gasUsed": int|None, "gasPrice": int|None, "effectiveGasPrice": int|None}
_tx_cache: Dict[str, Dict[str, Optional[int]]] = {}


# ---------------- Checkpointing ----------------
def load_checkpoint(path: str) -> Dict:
    if os.path.exists(path):
        with open(path, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
        return {
            "processed_tx_count": data.get("processed_tx_count", 0),
            "pending_tx_hashes": data.get("pending_tx_hashes", []),
            "tx_cache": data.get("tx_cache", {}),
        }
    return {"processed_tx_count": 0, "pending_tx_hashes": [], "tx_cache": {}}


def save_checkpoint(path: str, processed_tx_count: int, pending_tx_hashes: List[str]):
    # Disk guard: ensure enough free space on the checkpoint filesystem
    _check_disk_space_for_path(path)

    checkpoint = {
        "processed_tx_count": processed_tx_count,
        "pending_tx_hashes": pending_tx_hashes,
        "tx_cache": _tx_cache,
    }
    with open(path, "w") as f:
        json.dump(checkpoint, f)
    print(f"  💾 Checkpoint saved — processed {processed_tx_count} so far")


# ---------------- Utilities ----------------
def canonical_missing_str(series: pd.Series) -> pd.Series:
    """True where string-like field is missing/empty."""
    if series.dtype == object:
        s = series.astype(str).str.lower()
        return series.isna() | (series.astype(str).str.len() == 0) | s.isin(["none", "nan"])
    return series.isna()


def canonical_missing_num(series: pd.Series) -> pd.Series:
    """True where numeric field is missing/invalid (NaN or 0)."""
    if pd.api.types.is_numeric_dtype(series):
        return series.isna() | (series == 0)
    # Try coercion if object
    coerced = pd.to_numeric(series, errors="coerce")
    return coerced.isna() | (coerced == 0)


def ensure_gas_columns(df: pd.DataFrame) -> None:
    """Create gas columns if absent (as nullable Int64)."""
    for col in ["gasUsed", "gasPrice", "effectiveGasPrice"]:
        if col not in df.columns:
            df[col] = pd.Series([None] * len(df), dtype="Int64")


def determine_work(df: pd.DataFrame) -> List[str]:
    """List unique tx hashes that still need any gas field."""
    if "transactionHash" not in df.columns:
        raise ValueError("Input CSV must contain 'transactionHash' column")

    ensure_gas_columns(df)

    need = (
        canonical_missing_num(df["gasUsed"])
        | canonical_missing_num(df["gasPrice"])
        | canonical_missing_num(df["effectiveGasPrice"])
    ) & df["transactionHash"].notna()

    work = df.loc[need, "transactionHash"].astype(str).unique().tolist()
    return work


# ---------------- RPC fetchers ----------------
def fetch_tx_gas(tx_hash: str, retry_count: int = 3, backoff: float = 0.6) -> Dict[str, Optional[int]]:
    """Fetch gasUsed, gasPrice, effectiveGasPrice for a hash."""
    if w3 is None:
        raise RuntimeError("Web3 is not initialized. Run via `main()` or call `get_web3()` first.")
    if tx_hash in _tx_cache:
        return _tx_cache[tx_hash]

    last_err = None
    for attempt in range(retry_count):
        try:
            tx = w3.eth.get_transaction(tx_hash)
            receipt = w3.eth.get_transaction_receipt(tx_hash)

            gas_used = int(receipt.get("gasUsed")) if receipt and receipt.get("gasUsed") is not None else None
            # Note: On EIP-1559 txs, tx.get('gasPrice') can be None. That's expected.
            gas_price = int(tx.get("gasPrice")) if tx and tx.get("gasPrice") is not None else None
            eff_gas_price = (
                int(receipt.get("effectiveGasPrice"))
                if receipt and receipt.get("effectiveGasPrice") is not None
                else None
            )

            result = {"gasUsed": gas_used, "gasPrice": gas_price, "effectiveGasPrice": eff_gas_price}
            _tx_cache[tx_hash] = result
            return result
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < retry_count - 1:
                time.sleep(backoff * (attempt + 1))
            else:
                print(f"    ⚠️  Failed to fetch gas for {tx_hash}: {str(e)[:140]}")
                result = {"gasUsed": None, "gasPrice": None, "effectiveGasPrice": None}
                _tx_cache[tx_hash] = result
                return result
    # Should not reach here
    raise RuntimeError(f"Unexpected error: {last_err}")


def batch_fetch_gas(tx_hashes: List[str], workers: int) -> Dict[str, Dict[str, Optional[int]]]:
    """Parallel fetch with a small cache passthrough."""
    results: Dict[str, Dict[str, Optional[int]]] = {}
    to_fetch: List[str] = []

    for h in tx_hashes:
        if h in _tx_cache:
            results[h] = _tx_cache[h]
        else:
            to_fetch.append(h)

    if not to_fetch:
        return results

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_hash = {executor.submit(fetch_tx_gas, h): h for h in to_fetch}
        for fut in as_completed(future_to_hash):
            h = future_to_hash[fut]
            try:
                results[h] = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"    ⚠️  Error in thread for {h}: {str(e)[:140]}")
                results[h] = {"gasUsed": None, "gasPrice": None, "effectiveGasPrice": None}

    return results


# ---------------- Orchestration ----------------
def add_gas_incremental(
    df: pd.DataFrame,
    work_tx_hashes: List[str],
    *,
    batch_size: int,
    workers: int,
    save_every: int,
    start_index: int,
    checkpoint_path: str,
    output_csv: str,
) -> Tuple[pd.DataFrame, int]:
    """Fill gas fields for queued hashes from start_index. Returns (df, processed_tx_total)."""
    total = len(work_tx_hashes)
    print(f"\n📥 Pending unique tx hashes to resolve: {total}")

    processed_tx = start_index
    for i in range(start_index, total, batch_size):
        batch = work_tx_hashes[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size
        print(f"  🔄 Batch {batch_num}/{total_batches} — {len(batch)} txs")

        # Fetch
        fetched = batch_fetch_gas(batch, workers=workers)

        # Update only rows that still need any gas field
        need_mask = (
            (canonical_missing_num(df["gasUsed"])
             | canonical_missing_num(df["gasPrice"])
             | canonical_missing_num(df["effectiveGasPrice"]))
            & df["transactionHash"].isin(batch)
        )

        if need_mask.any():
            # Map per-column to keep nullable integers intact
            df.loc[need_mask, "gasUsed"] = df.loc[need_mask, "transactionHash"].map(
                lambda h: fetched.get(h, {}).get("gasUsed")
            )
            df.loc[need_mask, "gasPrice"] = df.loc[need_mask, "transactionHash"].map(
                lambda h: fetched.get(h, {}).get("gasPrice")
            )
            df.loc[need_mask, "effectiveGasPrice"] = df.loc[need_mask, "transactionHash"].map(
                lambda h: fetched.get(h, {}).get("effectiveGasPrice")
            )

        processed_tx = i + len(batch)

        if processed_tx % save_every == 0:
            # This will abort if disk space < 30% BEFORE writing checkpoint/CSV
            save_checkpoint(checkpoint_path, processed_tx, work_tx_hashes[processed_tx:])
            # If we made it past the checkpoint, go ahead and write the partial CSV
            df.to_csv(output_csv, index=False)
            print(f"  💽 Partial save → {output_csv}")

    return df, processed_tx


# ---------------- Main ----------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fill gasUsed / gasPrice / effectiveGasPrice via RPC")
    p.add_argument("--in", dest="input_csv", default=str(DEFAULT_INPUT_CSV), help="Input CSV path")
    p.add_argument("--out", dest="output_csv", default=str(DEFAULT_OUTPUT_CSV), help="Output CSV path")
    p.add_argument("--checkpoint", dest="checkpoint", default=str(DEFAULT_CHECKPOINT), help="Checkpoint JSON path")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY)
    return p.parse_args()


def main():
    args = parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    checkpoint_path = Path(args.checkpoint)
    batch_size = args.batch_size
    workers = args.workers
    save_every = args.save_every

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    print("🚀 Starting gas fetch (incremental-safe) ...")
    print(f"📄 INPUT_CSV  = {input_csv}")
    print(f"📄 OUTPUT_CSV = {output_csv}")
    print(f"🧷 CHECKPOINT = {checkpoint_path}")
    print(f"⚙️  batch={batch_size}  workers={workers}  save_every={save_every}")

    # Initialize Web3 lazily (keeps `--help` working without optional deps installed).
    global w3
    w3 = get_web3()

    # Load checkpoint & cache
    checkpoint = load_checkpoint(str(checkpoint_path))
    global _tx_cache
    _tx_cache = checkpoint.get("tx_cache", {}) or {}
    if _tx_cache:
        print(f"♻️  Loaded cache with {len(_tx_cache)} tx entries")

    # Choose base dataframe
    if output_csv.exists():
        print("📂 Found existing OUTPUT_CSV. Continuing to fill missing gas fields ...")
        df = pd.read_csv(output_csv)
    else:
        print("📂 No OUTPUT_CSV found. Loading INPUT_CSV and starting from scratch ...")
        if not input_csv.exists():
            raise SystemExit(f"Input CSV not found: {input_csv}")
        df = pd.read_csv(input_csv)

    ensure_gas_columns(df)

    # Worklist (checkpoint or fresh)
    if checkpoint.get("pending_tx_hashes"):
        work_tx_hashes = checkpoint["pending_tx_hashes"]
        start_index = checkpoint.get("processed_tx_count", 0)
        print(f"🔁 Resuming: {start_index} processed / {len(work_tx_hashes)} pending")
    else:
        work_tx_hashes = determine_work(df)
        start_index = 0
        if not work_tx_hashes:
            print("✅ Nothing to do — all gas fields already present.")
            # Still write out (ensures unified schema)
            # Extra safety: ensure disk is okay before final write
            _check_disk_space_for_path(str(output_csv))
            df.to_csv(output_csv, index=False)
            return

    t0 = time.time()
    df, processed_tx = add_gas_incremental(
        df,
        work_tx_hashes,
        batch_size=batch_size,
        workers=workers,
        save_every=save_every,
        start_index=start_index,
        checkpoint_path=str(checkpoint_path),
        output_csv=str(output_csv),
    )

    # Final save (guarded)
    print(f"\n💾 Saving CSV to {output_csv} ...")
    _check_disk_space_for_path(str(output_csv))
    df.to_csv(output_csv, index=False)

    # Stats
    elapsed = time.time() - t0
    total_rows = len(df)
    ok_gas_used = df["gasUsed"].notna().sum()
    ok_gas_price = df["gasPrice"].notna().sum()
    ok_eff = df["effectiveGasPrice"].notna().sum()

    print("\n" + "=" * 60)
    print("✅ COMPLETED!")
    print(f"📊 Total rows: {total_rows}")
    print(f"✅ gasUsed filled: {ok_gas_used}")
    print(f"✅ gasPrice filled: {ok_gas_price}  (may be None on EIP-1559 txs)")
    print(f"✅ effectiveGasPrice filled: {ok_eff}")
    print(f"⏱️  Time elapsed: {elapsed:.2f} s")
    print(f"📄 Output saved to: {output_csv}")
    print("=" * 60)

    # Cleanup checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print("🧹 Checkpoint file removed")


if __name__ == "__main__":
    main()
