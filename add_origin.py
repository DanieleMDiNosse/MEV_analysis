# pip install web3 pandas eth-defi
import os
import json
import time
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from web3 import Web3
from eth_defi.provider.multi_provider import create_multi_provider_web3

def build_mainnet_config(urls, timeout=5.0):
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

JSON_RPC_URLS = [
    "https://eth.llamarpc.com/sk_llama_252714c1e64c9873e3b21ff94d7f1a3f",
    "https://mainnet.infura.io/v3/5f38fb376e0548c8a828112252a6a588",
    "https://snowy-broken-spring.quiknode.pro/e1c35cbb709b1cb095d49e42dcd4d40e6cbbfd7a",
    "https://eth.rpc.grove.city/v1/887ffda2",
    "https://lb.nodies.app/v1/c6a2e72646e34fc78d95513a52c4aca6",
]
JSON_RPC_LINE = build_mainnet_config(JSON_RPC_URLS)
w3 = create_multi_provider_web3(JSON_RPC_LINE, request_kwargs={"timeout": 30.0})
assert w3.eth.chain_id == 1, "Connected chain is not Ethereum mainnet"


# Input/output files
INPUT_CSV = "/home/daniele/repositories/ABM_Uni_v3/data/univ3_pool_events_with_running_state.csv"
OUTPUT_CSV = "/home/daniele/repositories/ABM_Uni_v3/data/univ3_pool_events_with_origin.csv"
CHECKPOINT_FILE = "/home/daniele/repositories/ABM_Uni_v3/data/origin_fetch_checkpoint.json"

# Performance settings
BATCH_SIZE = 200            # Process this many transactions in parallel per batch
PARALLEL_WORKERS = 10       # Number of parallel threads
SAVE_EVERY = 1000           # Persist every N processed tx hashes

# ---------------- Setup ----------------
w3 = create_multi_provider_web3(JSON_RPC_LINE, request_kwargs={"timeout": 30.0})

# Cache for transaction data (tx hash -> origin)
_tx_cache: Dict[str, Optional[str]] = {}

# ---------------- Checkpointing ----------------
def load_checkpoint() -> Dict:
    """Load checkpoint if it exists."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
        # Backward compatibility with the previous format
        return {
            "processed_tx_count": data.get("processed_tx_count", data.get("processed_rows", 0)),
            "pending_tx_hashes": data.get("pending_tx_hashes", []),
            "tx_cache": data.get("tx_cache", {}),
        }
    return {"processed_tx_count": 0, "pending_tx_hashes": [], "tx_cache": {}}


def save_checkpoint(processed_tx_count: int, pending_tx_hashes: List[str]):
    """Save checkpoint with progress and cache."""
    checkpoint = {
        "processed_tx_count": processed_tx_count,
        "pending_tx_hashes": pending_tx_hashes,
        "tx_cache": _tx_cache,
    }
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f)
    print(f"  💾 Checkpoint saved — processed {processed_tx_count} / total so far")


# ---------------- RPC helpers ----------------
def fetch_transaction_origin(tx_hash: str, retry_count: int = 3) -> Optional[str]:
    """Fetch the `from` (origin) address for a transaction."""
    if tx_hash in _tx_cache:
        return _tx_cache[tx_hash]

    for attempt in range(retry_count):
        try:
            tx = w3.eth.get_transaction(tx_hash)
            origin = tx["from"]
            _tx_cache[tx_hash] = origin
            return origin
        except Exception as e:  # noqa: BLE001
            if attempt < retry_count - 1:
                time.sleep(0.5 * (attempt + 1))  # simple backoff
                continue
            print(f"    ⚠️  Failed to fetch origin for {tx_hash}: {str(e)[:120]}")
            _tx_cache[tx_hash] = None
            return None


def batch_fetch_origins(tx_hashes: List[str]) -> Dict[str, Optional[str]]:
    """Fetch origins for multiple transactions (parallel)."""
    results: Dict[str, Optional[str]] = {}
    to_fetch: List[str] = []

    # Use cache when available
    for h in tx_hashes:
        if h in _tx_cache:
            results[h] = _tx_cache[h]
        else:
            to_fetch.append(h)

    if not to_fetch:
        return results

    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
        future_to_hash = {executor.submit(fetch_transaction_origin, h): h for h in to_fetch}
        for future in as_completed(future_to_hash):
            h = future_to_hash[future]
            try:
                results[h] = future.result()
            except Exception as e:  # noqa: BLE001
                print(f"    ⚠️  Error fetching {h}: {str(e)[:120]}")
                results[h] = None

    return results


# ---------------- Dataframe orchestration ----------------
def canonical_missing_mask(series: pd.Series) -> pd.Series:
    """True where origin is missing/empty/NaN."""
    if series.dtype == object:
        return series.isna() | (series.astype(str).str.len() == 0) | (series.astype(str).str.lower().isin(["none", "nan"]))
    return series.isna()


def determine_work(df: pd.DataFrame) -> List[str]:
    """Return the list of tx hashes we still need to fetch origins for."""
    if "origin" in df.columns:
        mask_missing = canonical_missing_mask(df["origin"]) & df["transactionHash"].notna()
        work = df.loc[mask_missing, "transactionHash"].astype(str).unique().tolist()
    else:
        df["origin"] = None
        work = df.loc[df["transactionHash"].notna(), "transactionHash"].astype(str).unique().tolist()
    return work


def add_origins_incremental(df: pd.DataFrame, work_tx_hashes: List[str], start_index: int = 0) -> pd.DataFrame:
    """Add origins to `df` for the given `work_tx_hashes` (from `start_index`).

    This function only touches rows whose `transactionHash` is in `work_tx_hashes` and
    whose `origin` is currently missing.
    """
    total = len(work_tx_hashes)
    print(f"\n📥 Pending unique tx hashes to resolve: {total}")

    processed_tx = start_index
    for i in range(start_index, total, BATCH_SIZE):
        batch = work_tx_hashes[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"  🔄 Batch {batch_num}/{total_batches} — {len(batch)} txs")

        # Fetch
        batch_origins = batch_fetch_origins(batch)

        # Map into the dataframe only where origin is missing
        mask_missing = canonical_missing_mask(df["origin"]) & df["transactionHash"].isin(batch)
        # Use map to align values
        df.loc[mask_missing, "origin"] = df.loc[mask_missing, "transactionHash"].map(batch_origins)

        processed_tx = i + len(batch)

        # Periodic persistence
        if processed_tx % SAVE_EVERY == 0:
            save_checkpoint(processed_tx, work_tx_hashes[processed_tx:])
            # Write partial results to disk so we can resume even if interrupted
            df.to_csv(OUTPUT_CSV, index=False)
            print(f"  💽 Partial save to {OUTPUT_CSV}")

    return df


# ---------------- Main ----------------
def main():
    print("🚀 Starting origin fetch (incremental-safe) ...")
    print(f"📄 INPUT_CSV  = {INPUT_CSV}")
    print(f"📄 OUTPUT_CSV = {OUTPUT_CSV}")

    # Load previous checkpoint & cache (if any)
    checkpoint = load_checkpoint()
    global _tx_cache
    _tx_cache = checkpoint.get("tx_cache", {})
    if _tx_cache:
        print(f"♻️  Loaded cache with {len(_tx_cache)} tx entries")

    # Choose the base dataframe
    if os.path.exists(OUTPUT_CSV):
        print("📂 Found existing OUTPUT_CSV. Loading it to continue filling missing origins ...")
        df = pd.read_csv(OUTPUT_CSV)
        base_is_output = True
    else:
        print("📂 No OUTPUT_CSV found. Loading INPUT_CSV and starting from scratch ...")
        df = pd.read_csv(INPUT_CSV)
        base_is_output = False

    # Determine worklist (either from checkpoint or fresh scan)
    if checkpoint.get("pending_tx_hashes"):
        work_tx_hashes = checkpoint["pending_tx_hashes"]
        start_index = checkpoint.get("processed_tx_count", 0)
        print(f"🔁 Resuming from checkpoint: {start_index} processed / {len(work_tx_hashes)} pending")
    else:
        work_tx_hashes = determine_work(df)
        start_index = 0
        if not work_tx_hashes:
            print("✅ Nothing to do — all origins already present.")
            return

    t0 = time.time()
    df = add_origins_incremental(df, work_tx_hashes, start_index)

    # Final save
    print(f"\n💾 Saving CSV to {OUTPUT_CSV} ...")
    df.to_csv(OUTPUT_CSV, index=False)

    # Final stats
    elapsed = time.time() - t0
    total_rows = len(df)
    success = df["origin"].notna().sum()
    failed = df["origin"].isna().sum()

    print("\n" + "=" * 60)
    print("✅ COMPLETED!")
    print(f"📊 Total rows: {total_rows}")
    print(f"✅ Successful origin fetches: {success}")
    print(f"❌ Failed origin fetches: {failed}")
    print(f"⏱️  Time elapsed: {elapsed:.2f} seconds")
    print(f"📄 Output saved to: {OUTPUT_CSV}")
    print("=" * 60)

    # Clean up checkpoint
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print("🧹 Checkpoint file removed")


if __name__ == "__main__":
    main()
