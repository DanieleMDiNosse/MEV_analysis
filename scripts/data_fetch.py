"""
data_fetch.py — Uniswap v3 pool event harvester (origin separated from gas fields)

Overview
--------
This script extracts Swap, Mint, and Burn events for a single Uniswap v3 pool
over a given time window, enriches each event with the pool's *running state*
before/after the event, and (optionally) augments rows with on-chain gas data:
`gasUsed`, `gasPrice`, and `effectiveGasPrice`. Results are streamed to a CSV.

Key features
------------
- Robust, adaptive `get_logs` over large block ranges (auto-splitting on RPC limits).
- Parallel parsing of logs.
- **Separated metadata fetch**:
  - Always fetch `origin` (tx.from) from `eth_getTransaction`.
  - Optionally fetch gas fields from `eth_getTransactionReceipt` (plus legacy `gasPrice`).
- Resume-safe checkpoints (block cursor + last known pool state).
- Running state columns for liquidity and sqrt price before/after each event.

Workflow (high level)
---------------------
1) Convert the requested time window to block numbers via binary search.
2) Fetch Swap/Mint/Burn logs in adaptively sized block chunks.
3) Decode logs into event rows (base columns).
4) **Fetch origins only** for unique `transactionHash` values.
5) If `SKIP_GAS_DATA` is False, batch-fetch gas fields for those hashes.
6) Walk the event sequence in block/log order to compute pool *running state*.
7) Append rows to the CSV and persist a checkpoint.

Configuration knobs
-------------------
- `POOL_ADDR`: Pool address (checksum).
- `START_TS` / `END_TS`: Time window (UNIX seconds or ISO8601 string).
- `CHUNK_SIZE_BLOCKS`: Max block span per log request (auto-splits further on RPC limits).
- `PARALLEL_WORKERS`: Thread fan-out for log parsing.
- `BATCH_RECEIPT_SIZE`: Batch size for metadata lookups.
- `SKIP_GAS_DATA`: If True, skip `gasUsed` / `effectiveGasPrice` / `gasPrice` (faster).

Output CSV schema
-----------------
All rows share a common schema; some fields are `None` when not applicable to the
event type. Types are Python primitives as written by pandas.

Core identifiers and timing:
- `eventType` (str): One of `"Swap"`, `"Mint"`, `"Burn"`.
- `blockNumber` (int): Block height containing the event.
- `logIndex` (int): Index of the log within the block (for strict ordering).
- `timestamp` (int): Block timestamp in UNIX seconds (UTC).
- `transactionHash` (str): Hex string of the tx hash (0x-prefixed).

Gas / tx metadata:
- `origin` (str|None): Transaction sender (`from` address). **Always fetched.**
- `gasUsed` (int|None): Gas units actually consumed by the tx (from receipt).
- `gasPrice` (int|None): Legacy tx gas price in wei (from tx). May be `None` for
  EIP-1559 transactions which instead have `maxFeePerGas`/`maxPriorityFeePerGas`.
- `effectiveGasPrice` (int|None): Actual wei paid per gas (from receipt), i.e.
  `min(maxFeePerGas, baseFee + maxPriorityFeePerGas)` for type-2, or `gasPrice`
  for legacy. Use `gasUsed * effectiveGasPrice / 1e18` to get ETH spent.

Event-specific payload:
- `sender` (str|None): For `Swap`: event `sender`. For `Mint`/`Burn`: `None`.
- `owner` (str|None): For `Mint`/`Burn`: position owner. Else `None`.
- `recipient` (str|None): For `Swap`: event `recipient`. Else `None`.
- `amount0` (int):
  * Swap: signed pool delta of token0. **Sign convention (Uniswap v3):**
    Positive => pool received token0; Negative => pool sent token0.
  * Mint/Burn: unsigned amount of token0 moved due to liquidity change (event field).
- `amount1` (int):
  * Swap: signed pool delta of token1 (same sign convention as above).
  * Mint/Burn: unsigned amount of token1 moved due to liquidity change.
- `sqrtPriceX96_event` (int|None): For `Swap`: the new sqrt price Q64.96 after the swap.
  For `Mint`/`Burn`: `None`.
- `tick_event` (int|None): For `Swap`: the new tick after the swap. Else `None`.
- `liquidityAfter_event` (int|None): For `Swap`: pool active liquidity AFTER the swap event.
  For `Mint`/`Burn`: `None`.
- `tickLower` (int|None): For `Mint`/`Burn`: lower tick of the position. Else `None`.
- `tickUpper` (int|None): For `Mint`/`Burn`: upper tick of the position. Else `None`.

Running-state (derived):
- `L_before`, `sqrt_before`, `tick_before`, `x_before`, `y_before`
- `L_after`,  `sqrt_after`,  `tick_after`,  `x_after`,  `y_after`
- `affectsActive` (bool|None): For Mint/Burn: whether the position spans active tick.
- `deltaL_applied` (int|None): Liquidity change applied to active L (Mint positive, Burn negative when spanning).
"""

import os, json, tempfile, time
from pathlib import Path
from typing import Union, Optional, Dict, Any, List, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

import pandas as pd
from web3 import Web3
from web3._utils.events import get_event_data
from eth_defi.provider.multi_provider import create_multi_provider_web3

# ---------------- RPC configuration ----------------
# Do not hard-code RPC keys/secrets in the repository. Prefer environment variables:
# - MEV_RPC_URLS: space-separated HTTPS endpoints (preferred; enables failover)
# - WEB3_PROVIDER_URI: a single HTTPS endpoint
#
# If neither is set, we fall back to a small list of public endpoints (rate-limited).
DEFAULT_RPC_URLS = [
    "https://eth.llamarpc.com",
    "https://rpc.ankr.com/eth",
]
JSON_RPC_LINE = (os.environ.get("MEV_RPC_URLS") or os.environ.get("WEB3_PROVIDER_URI") or "").strip()
if not JSON_RPC_LINE:
    JSON_RPC_LINE = " ".join(DEFAULT_RPC_URLS)

POOL_ADDR = "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"  # USDC/WETH 0.05%
# POOL_ADDR = "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8"  # USDC/WETH 0.3%
# POOL_ADDR = "0xe0554a476a092703abdb3ef35c80e0d76d32939f"  # USDC/WETH 0.01%

# Provide either UNIX seconds or ISO strings (UTC assumed unless offset specified)
START_TS: Union[int, str] = "2023-01-01T00:00:00Z"
END_TS:   Union[int, str] = "2023-12-31T23:59:59Z"

# OPTIMIZED: Adaptive chunks, parallel processing
CHUNK_SIZE_BLOCKS = 500  # Reduced to avoid provider limits
PARALLEL_WORKERS = 8      # Parallel chunk processing
BATCH_RECEIPT_SIZE = 100  # Batch receipt/tx fetching

# Checkpointing / outputs (repo-relative so this script is runnable from any CWD)
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
CHECKPOINT_PATH = str(DATA_DIR / "checkpoints" / "univ3_checkpoint.json")
OUT_CSV = str(DATA_DIR / "raw" / f"univ3_{POOL_ADDR}.csv")

# Ensure directories exist early (before any writes).
Path(CHECKPOINT_PATH).parent.mkdir(parents=True, exist_ok=True)
Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)

# Option to skip gas data entirely (HUGE speedup if True)
SKIP_GAS_DATA = False  # Set True to skip gasUsed/gasPrice/effectiveGasPrice

# ---------------- Setup ----------------
w3 = create_multi_provider_web3(JSON_RPC_LINE, request_kwargs={"timeout": 60.0})
if w3.eth.chain_id != 1:
    raise RuntimeError(f"Connected chain is not Ethereum mainnet (chain_id={w3.eth.chain_id}).")
POOL = Web3.to_checksum_address(POOL_ADDR)

Q96 = 1 << 96

def to_unix(ts: Union[int, str]) -> int:
    if isinstance(ts, int):
        return ts
    return int(pd.to_datetime(ts, utc=True).value // 10**9)

START_TS = to_unix(START_TS)
END_TS   = to_unix(END_TS)
if END_TS < START_TS:
    raise ValueError("END_TS must be >= START_TS")

# ---------------- ABIs ----------------
SWAP_EVENT_ABI = {
    "anonymous": False,
    "inputs": [
        {"indexed": True,  "internalType":"address","name":"sender","type":"address"},
        {"indexed": True,  "internalType":"address","name":"recipient","type":"address"},
        {"indexed": False, "internalType":"int256", "name":"amount0","type":"int256"},
        {"indexed": False, "internalType":"int256", "name":"amount1","type":"int256"},
        {"indexed": False, "internalType":"uint160","name":"sqrtPriceX96","type":"uint160"},
        {"indexed": False, "internalType":"uint128","name":"liquidity","type":"uint128"},
        {"indexed": False, "internalType":"int24",  "name":"tick","type":"int24"}
    ],
    "name": "Swap", "type": "event"
}
MINT_EVENT_ABI = {
    "anonymous": False,
    "inputs": [
        {"indexed": False, "internalType":"address","name":"sender","type":"address"},
        {"indexed": True,  "internalType":"address","name":"owner","type":"address"},
        {"indexed": True,  "internalType":"int24",  "name":"tickLower","type":"int24"},
        {"indexed": True,  "internalType":"int24",  "name":"tickUpper","type":"int24"},
        {"indexed": False, "internalType":"uint128","name":"amount","type":"uint128"},
        {"indexed": False, "internalType":"uint256","name":"amount0","type":"uint256"},
        {"indexed": False, "internalType":"uint256","name":"amount1","type":"uint256"}
    ],
    "name": "Mint", "type": "event"
}
BURN_EVENT_ABI = {
    "anonymous": False,
    "inputs": [
        {"indexed": True,  "internalType":"address","name":"owner","type":"address"},
        {"indexed": True,  "internalType":"int24",  "name":"tickLower","type":"int24"},
        {"indexed": True,  "internalType":"int24",  "name":"tickUpper","type":"int24"},
        {"indexed": False, "internalType":"uint128","name":"amount","type":"uint128"},
        {"indexed": False, "internalType":"uint256","name":"amount0","type":"uint256"},
        {"indexed": False, "internalType":"uint256","name":"amount1","type":"uint256"}
    ],
    "name": "Burn", "type": "event"
}
POOL_STATE_ABI = [
    {"name":"slot0","inputs":[],"outputs":[
        {"type":"uint160","name":"sqrtPriceX96"},
        {"type":"int24","name":"tick"},
        {"type":"uint16","name":"observationIndex"},
        {"type":"uint16","name":"observationCardinality"},
        {"type":"uint16","name":"observationCardinalityNext"},
        {"type":"uint8","name":"feeProtocol"},
        {"type":"bool","name":"unlocked"}],
     "stateMutability":"view","type":"function"},
    {"name":"liquidity","inputs":[],"outputs":[{"type":"uint128"}],
     "stateMutability":"view","type":"function"},
]

contract = w3.eth.contract(address=POOL, abi=[SWAP_EVENT_ABI, MINT_EVENT_ABI, BURN_EVENT_ABI])
state_c  = w3.eth.contract(address=POOL, abi=POOL_STATE_ABI)

SwapEvent = contract.events.Swap
MintEvent = contract.events.Mint
BurnEvent = contract.events.Burn

SWAP_TOPIC0 = w3.keccak(text="Swap(address,address,int256,int256,uint160,uint128,int24)").hex()
MINT_TOPIC0 = w3.keccak(text="Mint(address,address,int24,int24,uint128,uint256,uint256)").hex()
BURN_TOPIC0 = w3.keccak(text="Burn(address,int24,int24,uint128,uint256,uint256)").hex()

# ---------------- Block helpers with caching ----------------
_block_cache: Dict[int, Any] = {}
_ts_cache: Dict[int, int] = {}

def get_block_cached(block_num: int):
    if block_num not in _block_cache:
        _block_cache[block_num] = w3.eth.get_block(block_num)
    return _block_cache[block_num]

def block_ts(bn: int) -> int:
    if bn not in _ts_cache:
        _ts_cache[bn] = get_block_cached(bn)["timestamp"]
    return _ts_cache[bn]

def block_for_timestamp(target_ts: int, mode: str = "start") -> int:
    latest = w3.eth.block_number
    lo, hi = 0, latest
    ans = None
    while lo <= hi:
        mid = (lo + hi) // 2
        ts = get_block_cached(mid)["timestamp"]
        if mode == "start":
            if ts >= target_ts:
                ans, hi = mid, mid - 1
            else:
                lo = mid + 1
        else:
            if ts <= target_ts:
                ans, lo = mid, mid + 1
            else:
                hi = mid - 1
    return ans if ans is not None else (0 if mode == "start" else 0)

START_BLOCK = block_for_timestamp(START_TS, "start")
END_BLOCK   = block_for_timestamp(END_TS,   "end")

# ---------------- Metadata caches ----------------
_rcpt_cache: Dict[str, Any] = {}
# tx hash -> {"from": str|None, "gasPrice": int|None}
_tx_cache: Dict[str, Dict[str, Optional[Union[str, int]]]] = {}

# ---------------- NEW: focused metadata helpers ----------------
def batch_fetch_origins(tx_hashes: List[str]) -> Dict[str, Optional[str]]:
    """
    Fetch only the transaction sender (`from`) for a set of tx hashes.
    Uses _tx_cache to avoid duplicate RPC calls.
    """
    origin_result: Dict[str, Optional[str]] = {}

    to_fetch = [h for h in tx_hashes if h not in _tx_cache or _tx_cache[h].get("from") is None]

    def fetch_tx(txh: str):
        try:
            tx = w3.eth.get_transaction(txh)
            _tx_cache[txh] = {"from": tx["from"], "gasPrice": tx.get("gasPrice")}
            return txh, _tx_cache[txh]["from"]
        except Exception:
            # cache a miss to avoid refetch storms
            _tx_cache.setdefault(txh, {"from": None, "gasPrice": None})
            return txh, None

    if to_fetch:
        with ThreadPoolExecutor(max_workers=min(10, len(to_fetch))) as ex:
            for fut in as_completed([ex.submit(fetch_tx, h) for h in to_fetch]):
                txh, origin = fut.result()
                origin_result[txh] = origin

    # ensure every requested hash has an entry (from cache or fetch)
    for h in tx_hashes:
        origin_result.setdefault(h, _tx_cache.get(h, {}).get("from"))
    return origin_result


def batch_fetch_gas(tx_hashes: List[str]) -> Tuple[
    Dict[str, Optional[int]],  # gasUsed
    Dict[str, Optional[int]],  # effectiveGasPrice
    Dict[str, Optional[int]],  # gasPrice (legacy; None on EIP-1559)
]:
    """
    Fetch gas metadata via receipts (gasUsed, effectiveGasPrice),
    plus legacy gasPrice via tx (using/warming _tx_cache).
    """
    gas_used: Dict[str, Optional[int]] = {}
    eff_price: Dict[str, Optional[int]] = {}
    gas_price: Dict[str, Optional[int]] = {}

    to_fetch_receipts = [h for h in tx_hashes if h not in _rcpt_cache]

    def fetch_receipt(txh: str):
        try:
            rc = w3.eth.get_transaction_receipt(txh)
            _rcpt_cache[txh] = rc
            return txh, rc
        except Exception:
            return txh, None

    if to_fetch_receipts:
        with ThreadPoolExecutor(max_workers=min(10, len(to_fetch_receipts))) as ex:
            for fut in as_completed([ex.submit(fetch_receipt, h) for h in to_fetch_receipts]):
                txh, rc = fut.result()
                if rc is not None:
                    _rcpt_cache[txh] = rc

    # Fill from caches, warming tx cache for gasPrice if needed
    for h in tx_hashes:
        rc = _rcpt_cache.get(h)
        gas_used[h] = int(rc["gasUsed"]) if rc and rc.get("gasUsed") is not None else None
        eff_price[h] = int(rc["effectiveGasPrice"]) if rc and rc.get("effectiveGasPrice") is not None else None

        if h not in _tx_cache:
            try:
                tx = w3.eth.get_transaction(h)
                _tx_cache[h] = {"from": tx["from"], "gasPrice": tx.get("gasPrice")}
            except Exception:
                _tx_cache[h] = {"from": None, "gasPrice": None}

        gprice = _tx_cache[h].get("gasPrice")
        gas_price[h] = int(gprice) if gprice is not None else None

    return gas_used, eff_price, gas_price

# ---------------- OPTIMIZED: Faster getLogs with better retry ----------------
RETRYABLE = (
    "timeout", "503", "502", "500", "429", "rate limit", "too many",
    "limit exceeded", "gateway", "413", "entity too large", 
    "payload too large", "request entity too large", "content too big",
    "range is too large", "max is 1k blocks",
    "query returned more than 10000 results", "exceeds max results",
    "-32005", "-32603", "-32602",
)

def get_logs_chunked_any(topics_any: List[str], start_block: int, end_block: int):
    """Optimized log fetching with aggressive adaptive chunking"""
    SOFT_LOG_LIMIT = 5000

    def extract_suggested_range(error_msg: str):
        import re
        m = re.search(r'range (\d+)-(\d+)', error_msg)
        if m:
            return int(m.group(1)), int(m.group(2))
        return None
    
    def yield_range(a: int, b: int, retry_count: int = 0):
        if retry_count > 2:
            chunk_size = max((b - a) // 10, 50)
            for chunk_start in range(a, b + 1, chunk_size):
                chunk_end = min(chunk_start + chunk_size - 1, b)
                yield from yield_range(chunk_start, chunk_end, 0)
            return
        
        filt = {
            "fromBlock": a,
            "toBlock": b,
            "address": POOL,
            "topics": [topics_any],
        }
        try:
            logs = w3.eth.get_logs(filt)
        except Exception as e:
            msg = str(e).lower()
            if b > a:
                if "exceeds max results" in msg:
                    suggested = extract_suggested_range(str(e))
                    if suggested:
                        s, e2 = suggested
                        if s == a:
                            yield from yield_range(a, e2, 0)
                            if e2 < b:
                                yield from yield_range(e2 + 1, b, 0)
                            return
                    parts = 8
                    part_size = (b - a) // parts
                    for i in range(parts):
                        s = a + i * part_size
                        e2 = a + (i + 1) * part_size - 1 if i < parts - 1 else b
                        if s <= e2:
                            yield from yield_range(s, e2, retry_count + 1)
                    return
                elif any(s in msg for s in RETRYABLE):
                    parts = 8 if ("10000" in msg or "20000" in msg) else (4 if ("entity too large" in msg or "payload too large" in msg) else 2)
                    part_size = (b - a + 1) // parts
                    for i in range(parts):
                        s = a + i * part_size
                        e2 = a + (i + 1) * part_size - 1 if i < parts - 1 else b
                        if s <= e2:
                            yield from yield_range(s, e2, retry_count + 1)
                    return
            raise
        
        if len(logs) > SOFT_LOG_LIMIT and b > a:
            mid = (a + b) // 2
            yield from yield_range(a, mid, retry_count)
            yield from yield_range(mid + 1, b, retry_count)
            return
        
        for lg in logs:
            yield lg
    
    cur = start_block
    while cur <= end_block:
        to_blk = min(cur + CHUNK_SIZE_BLOCKS - 1, end_block)
        try:
            yield from yield_range(cur, to_blk)
        except Exception as e:
            print(f"  ⚠️  Error in range {cur}-{to_blk}: {str(e)[:100]}")
            smaller_chunk = max(CHUNK_SIZE_BLOCKS // 4, 100)
            inner_cur = cur
            while inner_cur <= to_blk:
                inner_end = min(inner_cur + smaller_chunk - 1, to_blk)
                yield from yield_range(inner_cur, inner_end)
                inner_cur = inner_end + 1
        cur = to_blk + 1

# ---------------- Checkpoint helpers ----------------
def load_checkpoint(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)

def atomic_write_json(path: str, payload: Dict[str, Any]):
    d = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(prefix=".ckpt_", dir=d, text=True)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, separators=(",", ":"), sort_keys=True)
        os.replace(tmp, path)
    except Exception:
        try: os.remove(tmp)
        except Exception: pass
        raise

def save_checkpoint(path: str, data: Dict[str, Any]): 
    atomic_write_json(path, data)

def csv_append(df: pd.DataFrame, path: str):
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=header, index=False)

# ---------------- Virtual balances ----------------
def virt_x(L, sP): return (int(L) * Q96) // int(sP) if sP else None
def virt_y(L, sP): return (int(L) * int(sP)) // Q96 if sP else None

# ---------------- OPTIMIZED: Parallel log processing ----------------
def process_logs_parallel(logs_chunks: List[List], start_block: int, end_block: int):
    """Process multiple log chunks in parallel"""
    all_rows = []
    def process_chunk(logs):
        rows = []
        for log in logs:
            topic0 = log["topics"][0].hex()
            bn = log["blockNumber"]
            ts = block_ts(bn)
            if ts < START_TS or ts > END_TS:
                continue
            txh = log["transactionHash"].hex()
            if topic0 == SWAP_TOPIC0:
                evt = get_event_data(w3.codec, SwapEvent._get_event_abi(), log)
                a = evt["args"]
                rows.append({
                    "eventType": "Swap",
                    "blockNumber": bn, "logIndex": log["logIndex"], "timestamp": ts,
                    "transactionHash": txh,
                    "gasUsed": None, "gasPrice": None, "effectiveGasPrice": None, "origin": None,
                    "sender": a["sender"], "owner": None, "recipient": a["recipient"],
                    "amount0": int(a["amount0"]), "amount1": int(a["amount1"]),
                    "sqrtPriceX96_event": int(a["sqrtPriceX96"]),
                    "tick_event": int(a["tick"]),
                    "liquidityAfter_event": int(a["liquidity"]),
                    "tickLower": None, "tickUpper": None,
                    "liquidityDelta": None
                })
            elif topic0 == MINT_TOPIC0:
                evt = get_event_data(w3.codec, MintEvent._get_event_abi(), log)
                a = evt["args"]
                rows.append({
                    "eventType": "Mint",
                    "blockNumber": bn, "logIndex": log["logIndex"], "timestamp": ts,
                    "transactionHash": txh,
                    "gasUsed": None, "gasPrice": None, "effectiveGasPrice": None, "origin": None,
                    "sender": a["sender"], "owner": a["owner"], "recipient": None,
                    "amount0": int(a["amount0"]), "amount1": int(a["amount1"]),
                    "sqrtPriceX96_event": None, "tick_event": None, "liquidityAfter_event": None,
                    "tickLower": int(a["tickLower"]), "tickUpper": int(a["tickUpper"]),
                    "liquidityDelta": int(a["amount"])
                })
            elif topic0 == BURN_TOPIC0:
                evt = get_event_data(w3.codec, BurnEvent._get_event_abi(), log)
                a = evt["args"]
                rows.append({
                    "eventType": "Burn",
                    "blockNumber": bn, "logIndex": log["logIndex"], "timestamp": ts,
                    "transactionHash": txh,
                    "gasUsed": None, "gasPrice": None, "effectiveGasPrice": None, "origin": None,
                    "sender": None, "owner": a["owner"], "recipient": None,
                    "amount0": int(a["amount0"]), "amount1": int(a["amount1"]),
                    "sqrtPriceX96_event": None, "tick_event": None, "liquidityAfter_event": None,
                    "tickLower": int(a["tickLower"]), "tickUpper": int(a["tickUpper"]),
                    "liquidityDelta": -int(a["amount"])
                })
        return rows
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
        futures = [executor.submit(process_chunk, chunk) for chunk in logs_chunks]
        for future in as_completed(futures):
            all_rows.extend(future.result())
    return all_rows

# ---------------- OPTIMIZED: Main processing function ----------------
def process_window(from_block: int, to_block: int, 
                   cur_L: int, cur_sqrt: int, cur_tick: int) -> Dict[str, Any]:
    """Optimized window processing with parallel operations"""
    print(f"  Fetching logs for blocks {from_block}-{to_block}.")
    all_logs = list(get_logs_chunked_any([SWAP_TOPIC0, MINT_TOPIC0, BURN_TOPIC0], 
                                          from_block, to_block))
    if not all_logs:
        return {
            "df_chunk": pd.DataFrame(),
            "new_cur_L": cur_L, "new_cur_sqrt": cur_sqrt, "new_cur_tick": cur_tick,
            "first_event_block": None, "last_event_block": None, "n_rows": 0
        }
    print(f"  Processing {len(all_logs)} logs.")
    chunk_size = max(len(all_logs) // PARALLEL_WORKERS, 100)
    log_chunks = [all_logs[i:i+chunk_size] for i in range(0, len(all_logs), chunk_size)]
    rows = process_logs_parallel(log_chunks, from_block, to_block)
    if not rows:
        return {
            "df_chunk": pd.DataFrame(),
            "new_cur_L": cur_L, "new_cur_sqrt": cur_sqrt, "new_cur_tick": cur_tick,
            "first_event_block": None, "last_event_block": None, "n_rows": 0
        }
    df = pd.DataFrame(rows).sort_values(["blockNumber","logIndex"]).reset_index(drop=True)

    # ---- NEW: fetch origins always (cheap), gas optionally
    tx_list = df["transactionHash"].unique().tolist()

    # Origins
    print(f"  Fetching origins for {len(tx_list)} transactions...")
    origin_map: Dict[str, Optional[str]] = {}
    for i in range(0, len(tx_list), BATCH_RECEIPT_SIZE):
        batch = tx_list[i:i+BATCH_RECEIPT_SIZE]
        origins = batch_fetch_origins(batch)
        origin_map.update(origins)
    df["origin"] = df["transactionHash"].map(origin_map)

    # Gas fields
    if not SKIP_GAS_DATA:
        print(f"  Fetching gas fields for {len(tx_list)} transactions...")
        gas_used_map: Dict[str, Optional[int]] = {}
        eff_price_map: Dict[str, Optional[int]] = {}
        gas_price_map: Dict[str, Optional[int]] = {}

        for i in range(0, len(tx_list), BATCH_RECEIPT_SIZE):
            batch = tx_list[i:i+BATCH_RECEIPT_SIZE]
            gused, geff, gprice = batch_fetch_gas(batch)
            gas_used_map.update(gused)
            eff_price_map.update(geff)
            gas_price_map.update(gprice)

        df["gasUsed"]           = df["transactionHash"].map(gas_used_map)
        df["effectiveGasPrice"] = df["transactionHash"].map(eff_price_map)
        df["gasPrice"]          = df["transactionHash"].map(gas_price_map)
    else:
        # Ensure gas columns exist with None (schema stability)
        if "gasUsed" not in df.columns: df["gasUsed"] = None
        if "gasPrice" not in df.columns: df["gasPrice"] = None
        if "effectiveGasPrice" not in df.columns: df["effectiveGasPrice"] = None

    # ---- Running state calculation
    print(f"  Computing running state.")
    pre_L, pre_sqrt, pre_tick = [], [], []
    post_L, post_sqrt, post_tick = [], [], []
    x_before, y_before, x_after, y_after = [], [], [], []
    affects_active, delta_applied = [], []
    
    curL, curSP, curTk = int(cur_L), int(cur_sqrt), int(cur_tick)
    
    for _, row in df.iterrows():
        etype = row["eventType"]
        pre_L.append(curL); pre_sqrt.append(curSP); pre_tick.append(curTk)
        x_before.append(virt_x(curL, curSP))
        y_before.append(virt_y(curL, curSP))
        if etype in ("Mint", "Burn"):
            hit = (row["tickLower"] <= curTk) and (curTk < row["tickUpper"])
            affects_active.append(bool(hit))
            dL = int(row["liquidityDelta"]) if hit else 0
            delta_applied.append(dL)
            curL = curL + dL
            post_L.append(curL); post_sqrt.append(curSP); post_tick.append(curTk)
        elif etype == "Swap":
            affects_active.append(None)
            delta_applied.append(None)
            curL = int(row["liquidityAfter_event"])
            curSP = int(row["sqrtPriceX96_event"])
            curTk = int(row["tick_event"])
            post_L.append(curL); post_sqrt.append(curSP); post_tick.append(curTk)
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
    
    return {
        "df_chunk": df,
        "new_cur_L": curL, "new_cur_sqrt": curSP, "new_cur_tick": curTk,
        "first_event_block": int(df.iloc[0]["blockNumber"]),
        "last_event_block": int(df.iloc[-1]["blockNumber"]),
        "n_rows": len(df),
    }

# ---------------- Main execution ----------------
print(f"Starting data fetch for pool {POOL_ADDR}")
print(f"Date range: {pd.to_datetime(START_TS, unit='s')} to {pd.to_datetime(END_TS, unit='s')}")
print(f"Block range: {START_BLOCK} to {END_BLOCK} (~{END_BLOCK - START_BLOCK:,} blocks)")
print(f"Settings: CHUNK_SIZE={CHUNK_SIZE_BLOCKS}, WORKERS={PARALLEL_WORKERS}, SKIP_GAS={SKIP_GAS_DATA}")

ckpt = load_checkpoint(CHECKPOINT_PATH)
fresh = True

if ckpt:
    ok = (
        ckpt.get("pool") == POOL_ADDR
        and ckpt.get("start_ts") == START_TS
        and ckpt.get("end_ts") == END_TS
        and ckpt.get("start_block") == START_BLOCK
        and ckpt.get("end_block") == END_BLOCK
    )
    if ok:
        from_block = int(ckpt["next_from_block"])
        cur_L = int(ckpt["cur_L"])
        cur_sqrt = int(ckpt["cur_sqrt"])
        cur_tick = int(ckpt["cur_tick"])
        fresh = False
        print(f"Resuming from block {from_block} (state: L={cur_L}, sP={cur_sqrt}, tick={cur_tick})")
    else:
        print("Checkpoint params differ. Starting fresh.")
        ckpt = None

if fresh:
    from_block = START_BLOCK
    prev_block = max(from_block - 1, 0)
    s0 = state_c.functions.slot0().call(block_identifier=prev_block)
    L0 = int(state_c.functions.liquidity().call(block_identifier=prev_block))
    cur_sqrt = int(s0[0]); cur_tick = int(s0[1]); cur_L = int(L0)
    
    if os.path.exists(OUT_CSV):
        print(f"Removing previous output {OUT_CSV}")
        os.remove(OUT_CSV)
    
    ckpt = {
        "pool": POOL_ADDR,
        "start_ts": START_TS,
        "end_ts": END_TS,
        "start_block": START_BLOCK,
        "end_block": END_BLOCK,
        "next_from_block": from_block,
        "cur_L": cur_L,
        "cur_sqrt": cur_sqrt,
        "cur_tick": cur_tick,
        "events_written": 0,
        "last_event_block": None,
        "out_csv": OUT_CSV,
    }
    save_checkpoint(CHECKPOINT_PATH, ckpt)
    print(f"Initial state at block {prev_block}: L={cur_L}, sP={cur_sqrt}, tick={cur_tick}")

# Progress tracking
import datetime
start_time = time.time()
initial_block = from_block
total_blocks = END_BLOCK - START_BLOCK + 1

while from_block <= END_BLOCK:
    to_block = min(from_block + CHUNK_SIZE_BLOCKS - 1, END_BLOCK)
    
    blocks_done = from_block - initial_block
    pct_done = (blocks_done / total_blocks) * 100 if total_blocks > 0 else 0
    elapsed = time.time() - start_time
    if blocks_done > 0:
        rate = blocks_done / elapsed
        eta = (total_blocks - blocks_done) / rate if rate > 0 else 0
        eta_str = str(datetime.timedelta(seconds=int(eta)))
        print(f"\n[{pct_done:.1f}%] Processing blocks [{from_block:,}, {to_block:,}] "
              f"(Rate: {rate:.0f} blocks/s, ETA: {eta_str})")
    else:
        print(f"\nProcessing blocks [{from_block:,}, {to_block:,}].")
    
    result = process_window(from_block, to_block, cur_L, cur_sqrt, cur_tick)
    df_chunk = result["df_chunk"]
    n_rows = result["n_rows"]
    
    cur_L = result["new_cur_L"]
    cur_sqrt = result["new_cur_sqrt"]
    cur_tick = result["new_cur_tick"]
    
    if n_rows > 0:
        csv_append(df_chunk, OUT_CSV)
        ckpt.update({
            "cur_L": cur_L,
            "cur_sqrt": cur_sqrt,
            "cur_tick": cur_tick,
            "events_written": int(ckpt.get("events_written", 0)) + n_rows,
            "last_event_block": result["last_event_block"],
            "next_from_block": to_block + 1,
        })
        save_checkpoint(CHECKPOINT_PATH, ckpt)
        print(f"  ✓ Wrote {n_rows} rows | Last event block: {result['last_event_block']}")
    else:
        ckpt.update({
            "cur_L": cur_L,
            "cur_sqrt": cur_sqrt,
            "cur_tick": cur_tick,
            "next_from_block": to_block + 1,
        })
        save_checkpoint(CHECKPOINT_PATH, ckpt)
        print("  ✓ No events in window")
    
    from_block = to_block + 1

# Final summary
elapsed_total = time.time() - start_time
print(f"\n{'='*60}")
print(f"✅ COMPLETED!")
print(f"Time range: {pd.to_datetime(START_TS, unit='s')} to {pd.to_datetime(END_TS, unit='s')}")
print(f"Total events written: {ckpt.get('events_written', 0):,}")
print(f"Output file: {OUT_CSV}")
print(f"Total time: {str(datetime.timedelta(seconds=int(elapsed_total)))}")
print(f"Average rate: {total_blocks/elapsed_total:.0f} blocks/second")
print(f"{'='*60}")
