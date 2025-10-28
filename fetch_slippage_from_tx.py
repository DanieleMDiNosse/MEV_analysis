#!/usr/bin/env python3
"""
fetch_slippage_from_tx.py
=========================

Purpose
-------
Given an Ethereum transaction hash that performs a **Uniswap v3 swap**, this script
recovers the **implied slippage tolerance** that the trader encoded on-chain.

Uniswap v3 does not store a "slippage percent." Instead, swaps include **numeric guards**:
- For *exact input* swaps: `amountOutMinimum`
- For *exact output* swaps: `amountInMaximum`

This script:
1) **Locates and decodes** the swap parameters from either:
   - **SwapRouter v1** (0xE592…61564) or **SwapRouter v2** (0x68b3…5fC45),
     including when the swap is wrapped inside `multicall(bytes[])` or
     `multicall(uint256,bytes[])`; or
   - The **Universal Router** (0xEf1c…BF6B), by parsing its command VM and decoding
     `V3_SWAP_EXACT_IN` and `V3_SWAP_EXACT_OUT` inputs.

2) **Reconstructs the “expected” quote** by calling **Uniswap QuoterV2** at the
   **block immediately before** the transaction (`blockNumber - 1`). This is the best
   on-chain approximation of what the wallet/aggregator saw at signing time.

3) **Computes the implied slippage tolerance**:
   - Exact input:     tolerance = 1 − (amountOutMinimum / expectedOut)
   - Exact output:    tolerance = (amountInMaximum / expectedIn) − 1
   The script reports tolerance in **basis points (bps)** and **percent**.

What “decoding” means here
--------------------------
• **SwapRouter v1/v2 (periphery)**  
  Supports:
  - `exactInputSingle((tokenIn,tokenOut,fee,recipient,amountIn,amountOutMinimum,sqrtPriceLimitX96))`
  - `exactOutputSingle((tokenIn,tokenOut,fee,recipient,amountOut,amountInMaximum,sqrtPriceLimitX96))`
  - `exactInput((path,recipient,deadline,amountIn,amountOutMinimum))`
  - `exactOutput((path,recipient,deadline,amountOut,amountInMaximum))`
  The script also opens `multicall(...)` payloads and decodes inner calls.

• **Universal Router**  
  Decodes `execute(commands, inputs[, deadline])`. It iterates the command bytes,
  masks the “allow-revert” bit, and looks for:
  - `V3_SWAP_EXACT_IN`  → `(recipient, amountIn, amountOutMin, path, payerIsUser)`
  - `V3_SWAP_EXACT_OUT` → `(recipient, amountOut, amountInMax, path, payerIsUser)`
  The **v3 path** format is repeated chunks of `20-byte address + 3-byte fee`,
  e.g. `token0 | fee0 | token1 | fee1 | token2`, so tokens = fees + 1.
  For *exact output* routes, Universal Router encodes the **reverse path**
  (it starts with `tokenOut` and ends with `tokenIn`), matching Uniswap conventions.

How the “expected” quote is obtained
------------------------------------
We call **QuoterV2 (0x61fF…0B21e)** as a read-only `eth_call`:
- For single-hop: `quoteExactInputSingle` or `quoteExactOutputSingle`
- For multi-hop:  `quoteExactInput(path, amountIn)` or `quoteExactOutput(path, amountOut)`

We pass `blockNumber - 1` so pool state reflects *pre-trade* liquidity and tick.
That makes the implied tolerance comparable to the UI’s quoted expectation. If the
transaction is pending (no block yet), we fall back to `latest`.

Important distinctions
----------------------
• **Tolerance vs. Realized slippage**  
  This script computes the **tolerance** encoded by the trader (guard vs. expected).
  It does **not** compute realized slippage from actual fills. (That would require
  reading logs to get `amountIn/amountOut` and comparing to a contemporaneous quote.)

• **Exact input vs. exact output**  
  - *Exact input*: user spends a fixed `amountIn` and guards with `amountOutMinimum`.
  - *Exact output*: user targets a fixed `amountOut` and guards with `amountInMaximum`.

Limitations & edge cases
------------------------
1) **Router coverage**: Only Uniswap v3 **SwapRouter v1/v2** and **Universal Router**.
   It won’t decode other aggregators (e.g., 1inch/0x/Odos) or Uniswap v2/v4 swaps.

2) **Universal Router command scope**: Only `V3_SWAP_EXACT_IN` and `V3_SWAP_EXACT_OUT`
   are handled. Other UR commands (permit2, wrap/unwrap, NFT, etc.) are ignored.

3) **Historical state requirement**: To quote at `block-1`, your RPC should serve
   historical state. Some providers silently serve `latest` instead, slightly
   degrading accuracy (the script prints a note).

4) **Intra-block effects**: MEV/reordering and other swaps **within the same block**
   can move the pool between `block-1` and the actual execution point. Expect small
   deviations vs. the UI’s number, especially during volatile times.

5) **Path correctness**: If calldata is malformed or uses tokens/pools that Quoter
   can’t query (e.g., non-standard tokens), the quote can fail.

6) **No fee-on-transfer support**: Uniswap v3 periphery generally doesn’t support
   fee-on-transfer/rebasing tokens; such tokens may cause decode/quote failures.

7) **Decimals & humanization**: The script works in **raw units** (integers). It
   prints symbols/decimals only for user convenience and doesn’t change the math.

8) **Networks**: Addresses/ABIs are for **Ethereum mainnet**. For other networks you
   must update router/quoter addresses accordingly.

Output at a glance
------------------
- Router used and swap “kind”
- Tokens or full path (for multi-hop) and fee tiers
- Guard amounts: `amountOutMinimum` (exact in) or `amountInMaximum` (exact out)
- QuoterV2 “expected” amount at `block-1`
- **Implied slippage tolerance** in **bps** and **percent**

Usage
-----
1) Install deps:
   `pip install web3==6.* eth-abi hexbytes`
2) Set an RPC URL:
   `export WEB3_PROVIDER_URI="https://mainnet.infura.io/v3/<KEY>"`
3) Run:
   `python fetch_slippage_from_tx.py 0x<tx_hash>`

Troubleshooting
---------------
- If you see “Quoter failed …”, try a different RPC or an archive/full node.
- If you see “Could not locate a Uniswap v3 swap…”, the tx likely isn’t a v3 swap
  through the supported routers/commands, or it’s an unsupported aggregator route.
"""

# https://mainnet.infura.io/v3/5f38fb376e0548c8a828112252a6a588

from __future__ import annotations
import os
import sys
import argparse
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any, Iterable

import pandas as pd

from web3 import Web3
from web3.contract import Contract
from web3.types import TxData, TxReceipt
from hexbytes import HexBytes
from eth_abi import abi

# ---------- Addresses (Ethereum mainnet) ----------
UNISWAP_V3_ROUTER_V1 = Web3.to_checksum_address("0xE592427A0AEce92De3Edee1F18E0157C05861564")
UNISWAP_V3_ROUTER_V2 = Web3.to_checksum_address("0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45")
UNISWAP_V3_ROUTERS = {UNISWAP_V3_ROUTER_V1, UNISWAP_V3_ROUTER_V2}

UNIVERSAL_ROUTER = Web3.to_checksum_address("0xEf1c6E67703c7BD7107eed8303Fbe6EC2554BF6B")

# Quoter V2 (mainnet)
QUOTER_V2 = Web3.to_checksum_address("0x61fFE014bA17989E743c5f6cB21bF9697530B21e")

# ---------- Minimal ABIs ----------
SWAP_ROUTER_ABI: List[Dict[str, Any]] = [
    # exactInputSingle((...))
    {"name":"exactInputSingle","type":"function","stateMutability":"payable",
     "inputs":[{"name":"params","type":"tuple","components":[
         {"name":"tokenIn","type":"address"},
         {"name":"tokenOut","type":"address"},
         {"name":"fee","type":"uint24"},
         {"name":"recipient","type":"address"},
         {"name":"amountIn","type":"uint256"},
         {"name":"amountOutMinimum","type":"uint256"},
         {"name":"sqrtPriceLimitX96","type":"uint160"}]}],
     "outputs":[{"name":"amountOut","type":"uint256"}]},
    # exactOutputSingle((...))
    {"name":"exactOutputSingle","type":"function","stateMutability":"payable",
     "inputs":[{"name":"params","type":"tuple","components":[
         {"name":"tokenIn","type":"address"},
         {"name":"tokenOut","type":"address"},
         {"name":"fee","type":"uint24"},
         {"name":"recipient","type":"address"},
         {"name":"amountOut","type":"uint256"},
         {"name":"amountInMaximum","type":"uint256"},
         {"name":"sqrtPriceLimitX96","type":"uint160"}]}],
     "outputs":[{"name":"amountIn","type":"uint256"}]},
    # exactInput((path,recipient,deadline,amountIn,amountOutMinimum))
    {"name":"exactInput","type":"function","stateMutability":"payable",
     "inputs":[{"name":"params","type":"tuple","components":[
         {"name":"path","type":"bytes"},
         {"name":"recipient","type":"address"},
         {"name":"deadline","type":"uint256"},
         {"name":"amountIn","type":"uint256"},
         {"name":"amountOutMinimum","type":"uint256"}]}],
     "outputs":[{"name":"amountOut","type":"uint256"}]},
    # exactOutput((path,recipient,deadline,amountOut,amountInMaximum))
    {"name":"exactOutput","type":"function","stateMutability":"payable",
     "inputs":[{"name":"params","type":"tuple","components":[
         {"name":"path","type":"bytes"},
         {"name":"recipient","type":"address"},
         {"name":"deadline","type":"uint256"},
         {"name":"amountOut","type":"uint256"},
         {"name":"amountInMaximum","type":"uint256"}]}],
     "outputs":[{"name":"amountIn","type":"uint256"}]},
    # multicall variants
    {"name":"multicall","type":"function","stateMutability":"payable",
     "inputs":[{"name":"data","type":"bytes[]"}],"outputs":[{"name":"results","type":"bytes[]"}]},
    {"name":"multicall","type":"function","stateMutability":"payable",
     "inputs":[{"name":"deadline","type":"uint256"},{"name":"data","type":"bytes[]"}],
     "outputs":[{"name":"results","type":"bytes[]"}]},
]

UNIVERSAL_ROUTER_ABI: List[Dict[str, Any]] = [
    {"name":"execute","type":"function","stateMutability":"payable",
     "inputs":[{"name":"commands","type":"bytes"},
               {"name":"inputs","type":"bytes[]"},
               {"name":"deadline","type":"uint256"}], "outputs":[]},
    {"name":"execute","type":"function","stateMutability":"payable",
     "inputs":[{"name":"commands","type":"bytes"},
               {"name":"inputs","type":"bytes[]"}], "outputs":[]},
]

QUOTER_V2_ABI: List[Dict[str, Any]] = [
    {"name":"quoteExactInputSingle","type":"function","stateMutability":"nonpayable",
     "inputs":[{"name":"params","type":"tuple","components":[
         {"name":"tokenIn","type":"address"},
         {"name":"tokenOut","type":"address"},
         {"name":"amountIn","type":"uint256"},
         {"name":"fee","type":"uint24"},
         {"name":"sqrtPriceLimitX96","type":"uint160"}]}],
     "outputs":[
         {"name":"amountOut","type":"uint256"},
         {"name":"sqrtPriceX96After","type":"uint160"},
         {"name":"initializedTicksCrossed","type":"uint32"},
         {"name":"gasEstimate","type":"uint256"}]},
    {"name":"quoteExactOutputSingle","type":"function","stateMutability":"nonpayable",
     "inputs":[{"name":"params","type":"tuple","components":[
         {"name":"tokenIn","type":"address"},
         {"name":"tokenOut","type":"address"},
         {"name":"amount","type":"uint256"},
         {"name":"fee","type":"uint24"},
         {"name":"sqrtPriceLimitX96","type":"uint160"}]}],
     "outputs":[
         {"name":"amountIn","type":"uint256"},
         {"name":"sqrtPriceX96After","type":"uint160"},
         {"name":"initializedTicksCrossed","type":"uint32"},
         {"name":"gasEstimate","type":"uint256"}]},
    {"name":"quoteExactInput","type":"function","stateMutability":"nonpayable",
     "inputs":[{"name":"path","type":"bytes"},{"name":"amountIn","type":"uint256"}],
     "outputs":[
         {"name":"amountOut","type":"uint256"},
         {"name":"sqrtPriceX96AfterList","type":"uint160[]"},
         {"name":"initializedTicksCrossedList","type":"uint32[]"},
         {"name":"gasEstimate","type":"uint256"}]},
    {"name":"quoteExactOutput","type":"function","stateMutability":"nonpayable",
     "inputs":[{"name":"path","type":"bytes"},{"name":"amountOut","type":"uint256"}],
     "outputs":[
         {"name":"amountIn","type":"uint256"},
         {"name":"sqrtPriceX96AfterList","type":"uint160[]"},
         {"name":"initializedTicksCrossedList","type":"uint32[]"},
         {"name":"gasEstimate","type":"uint256"}]},
]

ERC20_ABI_MINI = [
    {"type":"function","name":"decimals","stateMutability":"view","inputs":[],"outputs":[{"type":"uint8"}]},
    {"type":"function","name":"symbol","stateMutability":"view","inputs":[],"outputs":[{"type":"string"}]},
]

# ---------- Data models ----------
@dataclass
class SwapGuard:
    router_type: str  # "swaprouter" | "universal"
    kind: str         # exactInputSingle | exactOutputSingle | exactInput | exactOutput | UR:V3_SWAP_EXACT_IN | UR:V3_SWAP_EXACT_OUT
    token_in: Optional[str]
    token_out: Optional[str]
    fee: Optional[int]
    path: Optional[bytes]
    amount_in: Optional[int]
    amount_out_min: Optional[int]
    amount_out: Optional[int]
    amount_in_max: Optional[int]
    sqrt_price_limit_x96: Optional[int]

# ---------- Helpers ----------
def get_w3() -> Web3:
    url = os.environ.get("WEB3_PROVIDER_URI")
    if not url:
        raise SystemExit("Please set WEB3_PROVIDER_URI to an Ethereum HTTPS RPC endpoint.")
    return Web3(Web3.HTTPProvider(url))

def fetch_tx(w3: Web3, tx_hash: str) -> Tuple[TxData, Optional[TxReceipt]]:
    tx = w3.eth.get_transaction(tx_hash)
    try:
        rcpt = w3.eth.get_transaction_receipt(tx_hash)
    except Exception:
        rcpt = None
    return tx, rcpt

def get_router(w3: Web3, addr: str) -> Contract:
    return w3.eth.contract(address=addr, abi=SWAP_ROUTER_ABI)

def get_universal_router(w3: Web3) -> Contract:
    return w3.eth.contract(address=UNIVERSAL_ROUTER, abi=UNIVERSAL_ROUTER_ABI)

def get_quoter(w3: Web3) -> Contract:
    return w3.eth.contract(address=QUOTER_V2, abi=QUOTER_V2_ABI)

def _safe_decode(types: List[str], payload: bytes):
    try:
        return True, abi.decode(types, payload)
    except Exception:
        return False, ()

# ---------- V3 path parsing ----------
def parse_v3_path(path: bytes) -> Tuple[List[str], List[int]]:
    i = 0
    tokens: List[str] = []
    fees: List[int] = []
    while True:
        if i + 20 > len(path):
            raise ValueError("Invalid path: truncated address")
        tokens.append(Web3.to_checksum_address("0x"+path[i:i+20].hex()))
        i += 20
        if i == len(path):
            break
        if i + 3 > len(path):
            raise ValueError("Invalid path: truncated fee")
        fees.append(int.from_bytes(path[i:i+3], "big"))
        i += 3
    if len(tokens) != len(fees) + 1:
        raise ValueError("Invalid path: tokens/fees mismatch")
    return tokens, fees

# ---------- SwapRouter decoding (robust) ----------
def decode_swaprouter_swap(w3: Web3, tx: TxData) -> Optional[SwapGuard]:
    to = tx["to"]
    if to is None or Web3.to_checksum_address(to) not in UNISWAP_V3_ROUTERS:
        return None
    router = get_router(w3, Web3.to_checksum_address(to))
    data = HexBytes(tx["input"])

    # decode the top-level function first
    try:
        fn, args = router.decode_function_input(data)
    except Exception:
        return None

    name = fn.fn_name
    if name in {"exactInputSingle","exactOutputSingle","exactInput","exactOutput"}:
        return _guard_from_swaprouter_decoded(name, args)

    if name == "multicall":
        # support both overloads; args is dict with keys 'data' or ('deadline','data')
        inner_list = None
        if "data" in args:
            inner_list = list(args["data"])
        elif isinstance(args, dict) and len(args) == 2:
            # older web3 may map to positional tuple
            try:
                inner_list = list(args[1])
            except Exception:
                pass
        if not inner_list:
            # Last resort: manual ABI decode
            sel = data[:4].hex()
            if sel == "ac9650d8":
                ok, dec = _safe_decode(["bytes[]"], data[4:])
                inner_list = list(dec[0]) if ok else []
            elif sel == "5ae401dc":
                ok, dec = _safe_decode(["uint256","bytes[]"], data[4:])
                inner_list = list(dec[1]) if ok else []

        for sub in inner_list or []:
            try:
                fn2, args2 = router.decode_function_input(sub)
                if fn2.fn_name in {"exactInputSingle","exactOutputSingle","exactInput","exactOutput"}:
                    return _guard_from_swaprouter_decoded(fn2.fn_name, args2)
            except Exception:
                continue
    return None

def _guard_from_swaprouter_decoded(name: str, args: Dict[str, Any]) -> SwapGuard:
    if name == "exactInputSingle":
        p = args["params"]
        return SwapGuard("swaprouter", name,
                         Web3.to_checksum_address(p["tokenIn"]),
                         Web3.to_checksum_address(p["tokenOut"]),
                         int(p["fee"]), None,
                         int(p["amountIn"]), int(p["amountOutMinimum"]),
                         None, None, int(p["sqrtPriceLimitX96"]))
    if name == "exactOutputSingle":
        p = args["params"]
        return SwapGuard("swaprouter", name,
                         Web3.to_checksum_address(p["tokenIn"]),
                         Web3.to_checksum_address(p["tokenOut"]),
                         int(p["fee"]), None,
                         None, None,
                         int(p["amountOut"]), int(p["amountInMaximum"]),
                         int(p["sqrtPriceLimitX96"]))
    if name == "exactInput":
        p = args["params"]
        return SwapGuard("swaprouter", name, None, None, None, bytes(p["path"]),
                         int(p["amountIn"]), int(p["amountOutMinimum"]),
                         None, None, None)
    if name == "exactOutput":
        p = args["params"]
        return SwapGuard("swaprouter", name, None, None, None, bytes(p["path"]),
                         None, None, int(p["amountOut"]), int(p["amountInMaximum"]), None)
    raise ValueError(f"Unsupported {name}")

# ---------- Universal Router decoding ----------
CMD_V3_SWAP_EXACT_IN  = 0x00
CMD_V3_SWAP_EXACT_OUT = 0x01

def decode_universal_router_swap(w3: Web3, tx: TxData) -> Optional[SwapGuard]:
    to = tx["to"]
    if to is None or Web3.to_checksum_address(to) != UNIVERSAL_ROUTER:
        return None
    ur = get_universal_router(w3)
    data = HexBytes(tx["input"])
    try:
        fn, args = ur.decode_function_input(data)
    except Exception:
        return None

    commands: bytes = args["commands"]
    inputs: List[bytes] = list(args["inputs"])

    for i, cmd in enumerate(commands):
        base = cmd & 0x1F  # strip allow-revert bit and reserved
        if base == CMD_V3_SWAP_EXACT_IN:
            ok, dec = _safe_decode(["address","uint256","uint256","bytes","bool"], inputs[i])
            if not ok: continue
            recipient, amountIn, amountOutMin, path, payerIsUser = dec
            # V3 path is forward for exact-in; token0 -> tokenN
            tokens, _ = parse_v3_path(path)
            token_in, token_out = tokens[0], tokens[-1]
            return SwapGuard("universal", "UR:V3_SWAP_EXACT_IN", token_in, token_out, None, path,
                             int(amountIn), int(amountOutMin), None, None, None)
        if base == CMD_V3_SWAP_EXACT_OUT:
            ok, dec = _safe_decode(["address","uint256","uint256","bytes","bool"], inputs[i])
            if not ok: continue
            recipient, amountOut, amountInMax, path, payerIsUser = dec
            # For exact-out, UR encodes the *reverse* path (out -> in)
            tokens, _ = parse_v3_path(path)
            token_out, token_in = tokens[0], tokens[-1]
            return SwapGuard("universal", "UR:V3_SWAP_EXACT_OUT", token_in, token_out, None, path,
                             None, None, int(amountOut), int(amountInMax), None)
    return None

# ---------- Quoter & slippage ----------
def quote_expected_amounts(w3: Web3, g: SwapGuard, block_for_quote) -> Tuple[Optional[int], Optional[int], str]:
    q = get_quoter(w3)
    try:
        if g.kind in {"exactInputSingle"}:
            out, *_ = q.functions.quoteExactInputSingle((g.token_in, g.token_out, int(g.amount_in), int(g.fee), int(g.sqrt_price_limit_x96 or 0))).call(block_identifier=block_for_quote)
            return None, int(out), f"Quoted exactInputSingle @ {block_for_quote}"
        if g.kind in {"exactOutputSingle"}:
            _in, *_ = q.functions.quoteExactOutputSingle((g.token_in, g.token_out, int(g.amount_out), int(g.fee), int(g.sqrt_price_limit_x96 or 0))).call(block_identifier=block_for_quote)
            return int(_in), None, f"Quoted exactOutputSingle @ {block_for_quote}"
        if g.kind in {"exactInput","UR:V3_SWAP_EXACT_IN"}:
            out, *_ = q.functions.quoteExactInput(g.path, int(g.amount_in)).call(block_identifier=block_for_quote)
            return None, int(out), f"Quoted exactInput(path) @ {block_for_quote}"
        if g.kind in {"exactOutput","UR:V3_SWAP_EXACT_OUT"}:
            _in, *_ = q.functions.quoteExactOutput(g.path, int(g.amount_out)).call(block_identifier=block_for_quote)
            return int(_in), None, f"Quoted exactOutput(path) @ {block_for_quote}"
    except Exception as e:
        return None, None, f"Quoter failed: {type(e).__name__}: {e}"
    return None, None, "Unsupported kind"

def compute_slippage_bps(g: SwapGuard, expected_in: Optional[int], expected_out: Optional[int]) -> Optional[float]:
    if g.kind in {"exactInputSingle","exactInput","UR:V3_SWAP_EXACT_IN"}:
        if expected_out and g.amount_out_min is not None and expected_out > 0:
            return (1.0 - g.amount_out_min / expected_out) * 10_000.0
        return None
    if g.kind in {"exactOutputSingle","exactOutput","UR:V3_SWAP_EXACT_OUT"}:
        if expected_in and g.amount_in_max is not None and expected_in > 0:
            return (g.amount_in_max / expected_in - 1.0) * 10_000.0
        return None
    return None

# ---------- Single-tx wrapper ----------
def compute_slippage_for_tx(tx_hash: str, w3: Optional[Web3]=None) -> Dict[str, Any]:
    """
    Returns a dict with fields described in the module docstring.
    Never raises; errors are stored in the 'error' field.
    """
    own_w3 = False
    if w3 is None:
        w3 = get_w3()
        own_w3 = True
    try:
        tx, rcpt = fetch_tx(w3, tx_hash)
        guard = decode_swaprouter_swap(w3, tx) or decode_universal_router_swap(w3, tx)
        if guard is None:
            return {
                "transactionHash": tx_hash,
                "error": "Not a supported Uniswap v3 swap (SwapRouter v1/v2 or Universal Router V3 swap)",
            }

        block = tx.get("blockNumber", None)
        block_for_quote = "latest" if block is None else max(0, block - 1)
        expected_in, expected_out, note = quote_expected_amounts(w3, guard, block_for_quote)
        tol_bps = compute_slippage_bps(guard, expected_in, expected_out)

        # default outputs
        path_tokens, path_fees = [], []
        fee = guard.fee
        token_in = guard.token_in
        token_out = guard.token_out

        if guard.path:
            try:
                tokens, fees = parse_v3_path(guard.path)
                path_tokens = tokens
                path_fees = fees
                # if path supplied, override token_in/out for convenience
                token_in = tokens[0]
                token_out = tokens[-1]
            except Exception:
                pass

        out = {
            "transactionHash": tx_hash,
            "router": Web3.to_checksum_address(tx["to"]) if tx.get("to") else None,
            "kind": guard.kind,
            "block": block,
            "quoted_block": block_for_quote,
            "token_in": token_in,
            "token_out": token_out,
            "fee": fee,
            "path_tokens": path_tokens,
            "path_fees": path_fees,
            "amount_in": guard.amount_in,
            "amount_out_min": guard.amount_out_min,
            "amount_out": guard.amount_out,
            "amount_in_max": guard.amount_in_max,
            "expected_in": expected_in,
            "expected_out": expected_out,
            "slippage_bps": float(tol_bps) if tol_bps is not None else None,
            "slippage_pct": (float(tol_bps)/100.0) if tol_bps is not None else None,
            "note": note,
            "error": None if tol_bps is not None else ("Insufficient data to compute tolerance" if "Quoter failed" not in (note or "") else note),
        }
        return out
    except Exception as e:
        return {
            "transactionHash": tx_hash,
            "error": f"{type(e).__name__}: {e}",
        }
    finally:
        # nothing to clean for web3 HTTP
        pass

# ---------- Batch utilities ----------
def _unique_tx_hashes(df: pd.DataFrame) -> List[str]:
    if "eventType" not in df.columns or "transactionHash" not in df.columns:
        raise ValueError("DataFrame must contain 'eventType' and 'transactionHash' columns.")
    mask = df["eventType"].isin(["swap_x2y","swap_y2x"])
    return (
        df.loc[mask, "transactionHash"]
          .dropna()
          .astype(str)
          .str.lower()
          .drop_duplicates()
          .tolist()
    )

def compute_slippage_for_hashes(hashes: Iterable[str], w3: Optional[Web3]=None) -> pd.DataFrame:
    own_w3 = False
    if w3 is None:
        w3 = get_w3()
        own_w3 = True
    rows = []
    for h in hashes:
        h_norm = h if h.startswith("0x") else ("0x"+h)
        rows.append(compute_slippage_for_tx(h_norm, w3=w3))
    if own_w3:
        # nothing to close
        pass
    return pd.DataFrame(rows)

def compute_slippage_from_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    hashes = _unique_tx_hashes(df)
    return compute_slippage_for_hashes(hashes)

# ---------- CLI ----------
def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Batch fetch implied slippage tolerance for Uniswap v3 swaps from a DataFrame/CSV/Parquet")
    gsrc = p.add_mutually_exclusive_group(required=True)
    gsrc.add_argument("--csv", type=str, help="Path to CSV with columns: eventType, transactionHash")
    gsrc.add_argument("--parquet", type=str, help="Path to Parquet with columns: eventType, transactionHash")
    p.add_argument("--out", type=str, help="Optional path to write CSV results")
    return p.parse_args(argv)

def _load_df_from_args(args) -> pd.DataFrame:
    if args.csv:
        return pd.read_csv(args.csv)
    if args.parquet:
        return pd.read_parquet(args.parquet)
    raise ValueError("No input provided")

def main(argv=None):
    args = _parse_args(argv)
    df = _load_df_from_args(args)
    res = compute_slippage_from_dataframe(df)
    if args.out:
        res.to_csv(args.out, index=False)
        print(f"Wrote {len(res)} rows to {args.out}")
    else:
        # pretty print a small sample to stdout
        with pd.option_context("display.max_columns", None, "display.width", 160):
            print(res.head(20))

if __name__ == "__main__":
    main()
