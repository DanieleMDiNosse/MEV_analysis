#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MEV collector for a single Uniswap v3 pool
==========================================

What this script does
---------------------
• Scans a per-pool, per-event dataset (swaps/mints/burns) and **detects three MEV patterns** in a
  same-block, single-pool, *strict* sense:
  1) **Pure JIT**: `mint(att) → victim swaps (≠att) → burn(att)` (contiguous)  
  2) **Classical Sandwich**: `front(att) → victim swaps (dir D, ≠att) → back(att, dir ¬D)` (contiguous)  
     **JIT-Sandwich** variant is also detected: `front → mint → victim swaps → burn → back`
  3) **Reverse Back-run**: `victim swap by A` immediately followed by `swap by B` in opposite direction

• For each pattern, it **augments rows with Section 3 theory**—normalized sizes (σ, ε), direction-aware
  price-impact I(·), viability thresholds (e.g., σ_min for back-runs), and **profit ceilings** (optimal
  Π⋆_br, sandwich π⋆ under slippage γ, JIT bribe ceilings)—all exactly as derived in
  Section 3. Formulas/eq. numbers below refer to that Section.

• **Outputs computation-ready CSVs** in `--outdir`:
  - `jit_cycles_tidy_<fee_bps>.csv`
  - `sandwich_attacks_tidy_<fee_bps>.csv`
  - `reverse_backruns_tidy_<fee_bps>.csv`
  Each already contains the raw anchors (tx hashes, directions, amounts) **plus** the theory fields.

NEW:
  • `attacker_liq_share` for Pure JIT and JIT-Sandwich rows:
      attacker_liq_share = liquidityDelta / (liquidityDelta + L_before at the mint)
"""

from __future__ import annotations

# Limit BLAS threads before numpy/pandas import
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
import argparse
import math
from pathlib import Path
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

Q96 = 2 ** 96
REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------
# Helpers: schema & numeric accessors
# ---------------------------------------------------------------------

@dataclass
class Schema:
    col_event: str
    col_block: str
    col_log_index: str
    col_txhash: str
    col_origin: Optional[str]
    col_amount0: str
    col_amount1: str
    col_amount: Optional[str]
    col_tick_lower: Optional[str]
    col_tick_upper: Optional[str]
    col_gas_used: Optional[str]
    col_gas_price: Optional[str]
    col_L_before: Optional[str]
    col_sqrt_before: Optional[str]
    col_x_before: Optional[str]
    col_y_before: Optional[str]
    col_L_after: Optional[str]
    col_sqrt_event: Optional[str]
    col_tick_before: Optional[str]
    col_tick_event: Optional[str]
    col_tick_after: Optional[str]


def resolve_schema(df: pd.DataFrame) -> Schema:
    def pick(*cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    col_event = pick('eventType', 'event')
    col_block = pick('blockNumber')
    col_log_index = pick('logIndex')
    col_txhash = pick('transactionHash')
    col_origin = pick('origin', 'owner', 'sender')

    col_amount0 = pick('amount0')
    col_amount1 = pick('amount1')
    col_amount = pick('liquidityDelta')

    col_tick_lower = pick('tickLower')
    col_tick_upper = pick('tickUpper')

    col_gas_used = pick('gasUsed')
    col_gas_price = pick('effectiveGasPrice')

    col_L_before = pick('L_before')
    col_sqrt_before = pick('sqrt_before', 'sqrtPriceX96_before')
    col_x_before = pick('x_before')
    col_y_before = pick('y_before')

    col_L_after = pick('liquidityAfter_event', 'L_after', 'liquidity')
    col_sqrt_event = pick('sqrtPriceX96_event', 'sqrt_after')

    col_tick_before = pick('tick_before')
    col_tick_event = pick('tick_event', 'tick_after')
    col_tick_after = pick('tick_after')

    for must in [col_event, col_block, col_log_index, col_txhash, col_amount0, col_amount1]:
        if must is None:
            raise ValueError("Missing a required column — check your dataset has eventType,event/logIndex/blockNumber/transactionHash, amount0/amount1.")

    return Schema(
        col_event, col_block, col_log_index, col_txhash, col_origin, col_amount0,
        col_amount1, col_amount, col_tick_lower, col_tick_upper, col_gas_used, col_gas_price,
        col_L_before, col_sqrt_before, col_x_before, col_y_before, col_L_after,
        col_sqrt_event, col_tick_before, col_tick_event, col_tick_after
    )


def to_num(x) -> float:
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    # strings of ints/decimals
    try:
        if isinstance(x, str) and x.strip().startswith("0x"):
            return float(int(x, 16))
        return float(x)
    except Exception:
        return np.nan


# ---------------------------------------------------------------------
# Uniswap v3 math helpers (v2-like virtual reserves)
# ---------------------------------------------------------------------

def sqrtp_q96_to_float(sqrt_q96: float) -> float:
    """Convert a Q96 sqrt price to a float sqrt(P)."""
    return float(sqrt_q96) / Q96


def price_from_sqrtq96(sqrt_q96: float) -> float:
    sp = sqrtp_q96_to_float(sqrt_q96)
    return sp * sp


def virtual_xy_from_L_sqrt(L: float, sqrt_q96: float) -> Tuple[float, float]:
    """Map (L, √P_Q96) -> (x, y) via x = L/√P, y = L*√P."""
    sp = sqrtp_q96_to_float(sqrt_q96)
    if sp <= 0 or np.isnan(sp) or np.isnan(L):
        return (np.nan, np.nan)
    return (L / sp, L * sp)


# ---------------------------------------------------------------------
# Section 3 formulas
# ---------------------------------------------------------------------

def price_impact_x2y(r: float, sigma_x_gross: float) -> float:
    return 1.0 / (1.0 + r * sigma_x_gross) ** 2 - 1.0


def price_impact_y2x(r: float, sigma_y_gross: float) -> float:
    return (1.0 + r * sigma_y_gross) ** 2 - 1.0


def backrun_sigma_min(r: float) -> float:
    # Eq. (13): σ ≥ (1/√r - 1)/r
    return (1.0 / math.sqrt(r) - 1.0) / r


def backrun_opt_dy_and_profit(x0: float, y0: float, r: float, sigma_gross_x: float) -> float:
    """Return Π*_br measured in token X units (Eq. 15)."""
    u = 1.0 + r * sigma_gross_x
    Pi_start = x0 * (math.sqrt(r) * u - 1.0)**2 / (r*u) if (u and u > 0) else np.nan
    return Pi_start

def backrun_opt_profit_base(base_reserve: float, r: float, sigma_native_gross: float) -> float:
    u = 1.0 + r * sigma_native_gross
    if not (u > 0):
        return float("nan")
    return base_reserve * (math.sqrt(r) * u - 1.0) ** 2 / (r * u)


def eps_max_under_slippage(sigma_gross: float, gamma: float, r: float) -> float:
    if gamma >= 1.0:
        return float("inf")
    if gamma < 0:
        gamma = 0.0
    A = (r * sigma_gross) ** 2
    B = 4.0 * (1.0 + r * sigma_gross) / (1.0 - gamma)
    return max(0.0, ((-r * sigma_gross + math.sqrt(A + B)) / 2.0 - 1.0) / r)

def eps_max_under_slippage_jit(sigma_gross: float, gamma: float, r: float, alpha: float) -> float:
    """
    ε_max for the mixed (front-run + JIT + back-run) bundle, derived from the victim's min-out
    when the pool is temporarily scaled by (1+α) during the victim leg.

    Implements §3.4 eq. (44):
        ε_max(σ, γ, r, α) = ((-rσ + sqrt((rσ)^2 + 4(1+α)^2(1+rσ)/(1-γ))) / (2(1+α)) - 1)/r

    Returns +inf if gamma >= 1. Clips gamma to [0,1), alpha to ≥0.
    """
    if gamma >= 1.0:
        return float("inf")
    if gamma < 0.0:
        gamma = 0.0
    if not np.isfinite(alpha) or alpha < 0.0:
        alpha = 0.0
    if not np.isfinite(sigma_gross) or sigma_gross < 0.0 or not np.isfinite(r) or r <= 0.0:
        return float("nan")

    A = (r * sigma_gross) ** 2
    B = 4.0 * (1.0 + r * sigma_gross) * (1.0 + alpha) ** 2 / (1.0 - gamma)
    num = -r * sigma_gross + math.sqrt(A + B)
    den = 2.0 * (1.0 + alpha)
    return max(0.0, (num / den - 1.0) / r)


def evaluate_eps_max_series_jit(sigmas: pd.Series, gamma: float, r: float, alphas: pd.Series) -> pd.Series:
    """
    Vectorized wrapper over eps_max_under_slippage_jit for pandas Series inputs.
    Falls back to NaN where inputs are invalid; caller can choose a default.
    """
    sig = sigmas.astype(float)
    alp = alphas.astype(float)
    out = [eps_max_under_slippage_jit(float(s), gamma, r, float(a)) for s, a in zip(sig, alp)]
    return pd.Series(out, index=sig.index)



def sandwich_profit_normalized(eps_gross: float, sigma_gross: float, r: float) -> float:
    if eps_gross < 0:
        return np.nan
    A = r * r * eps_gross * (1.0 + r * eps_gross + r * sigma_gross) ** 2
    B = (1.0 + r * eps_gross) + r * r * eps_gross * (1.0 + r * eps_gross + r * sigma_gross)
    if B == 0:
        return np.nan
    return A / B - eps_gross


def sandwich_profit_star(sigma_gross: float, r: float, gamma: float, grid_n: int = 128) -> Tuple[float, float]:
    eps_max = eps_max_under_slippage(sigma_gross, gamma, r)
    if not np.isfinite(eps_max) or eps_max <= 0:
        return (0.0, 0.0)
    xs = np.linspace(0.0, eps_max, max(8, int(grid_n)))
    vals = [sandwich_profit_normalized(e, sigma_gross, r) for e in xs]
    j = int(np.nanargmax(vals))
    return (float(xs[j]), float(vals[j]))


def sigma_star_backrun_vs_jit(f: float, phi: float, r: float) -> Optional[float]:
    g = f - phi
    a = (r - g)
    b = (g - 2.0 * math.sqrt(r))
    c = 1.0
    disc = b * b - 4.0 * a * c
    if a == 0 or disc < 0:
        return None
    u_pos = (-b + math.sqrt(disc)) / (2.0 * a)
    sigma = (u_pos - 1.0) / r
    return sigma


# ---------------------------------------------------------------------
# Block slicer & multiprocessing
# ---------------------------------------------------------------------

_GDF: Optional[pd.DataFrame] = None
_GSCHEMA: Optional[Schema] = None


def _pool_init(df: pd.DataFrame, schema: Schema):
    global _GDF, _GSCHEMA
    _GDF = df
    _GSCHEMA = schema


def _build_block_slices(df: pd.DataFrame, schema: Schema) -> List[Tuple[int, int, int]]:
    blocks = df[schema.col_block].astype(np.int64).to_numpy()
    uniq = np.unique(blocks)
    order = np.argsort(blocks)
    left = np.searchsorted(blocks[order], uniq, side='left')
    right = list(left[1:]) + [len(df)]
    return list(zip(uniq.tolist(), left.tolist(), right))


def _chunkify(slices: List[Tuple[int, int, int]], k: int) -> List[List[Tuple[int, int, int]]]:
    if k <= 1:
        return [[s] for s in slices]
    return [slices[i:i + k] for i in range(0, len(slices), k)]


# ---------------------------------------------------------------------
# Core detectors (operate on the raw df; *no column renaming*)
# ---------------------------------------------------------------------

def _read_block_arrays(sub: pd.DataFrame, schema: Schema) -> Dict[str, Any]:
    ev = sub[schema.col_event].astype(str).str.lower().to_numpy()
    a0 = sub[schema.col_amount0].apply(to_num).to_numpy()
    a1 = sub[schema.col_amount1].apply(to_num).to_numpy()
    tx = sub[schema.col_txhash].astype(str).to_numpy()
    gas_used = sub[schema.col_gas_used].apply(to_num).to_numpy() if schema.col_gas_used else np.full(len(sub), np.nan)
    gas_price = sub[schema.col_gas_price].apply(to_num).to_numpy() if schema.col_gas_price else np.full(len(sub), np.nan)
    amt = sub[schema.col_amount].apply(to_num).to_numpy() if schema.col_amount else np.full(len(sub), np.nan)

    if schema.col_origin:
        origins = sub[schema.col_origin].astype(str).str.lower().to_numpy()
    else:
        origins = np.array([None] * len(sub), dtype=object)

    tlo = sub[schema.col_tick_lower].apply(to_num).to_numpy() if schema.col_tick_lower else np.full(len(sub), np.nan)
    thi = sub[schema.col_tick_upper].apply(to_num).to_numpy() if schema.col_tick_upper else np.full(len(sub), np.nan)

    # pre-state preferred if present
    Lb = sub[schema.col_L_before].apply(to_num).to_numpy() if schema.col_L_before else np.full(len(sub), np.nan)
    sqb = sub[schema.col_sqrt_before].apply(to_num).to_numpy() if schema.col_sqrt_before else np.full(len(sub), np.nan)
    xb = sub[schema.col_x_before].apply(to_num).to_numpy() if schema.col_x_before else np.full(len(sub), np.nan)
    yb = sub[schema.col_y_before].apply(to_num).to_numpy() if schema.col_y_before else np.full(len(sub), np.nan)

    # fallback to event-time state
    La = sub[schema.col_L_after].apply(to_num).to_numpy() if schema.col_L_after else np.full(len(sub), np.nan)
    sqe = sub[schema.col_sqrt_event].apply(to_num).to_numpy() if schema.col_sqrt_event else np.full(len(sub), np.nan)

    return dict(ev=ev, a0=a0, a1=a1, tx=tx, origins=origins, tlo=tlo, thi=thi,
                gas_used=gas_used, gas_price=gas_price, amt=amt,
                Lb=Lb, sqb=sqb, xb=xb, yb=yb, La=La, sqe=sqe)


def _dir_from_amounts(a0: float, a1: float) -> Optional[str]:
    # Uniswap v3 semantics: pool deltas; X→Y has amount0>0, amount1<0; Y→X vice-versa
    if np.isnan(a0) or np.isnan(a1):
        return None
    if a0 > 0 and a1 < 0:
        return 'swap_x2y'
    if a0 < 0 and a1 > 0:
        return 'swap_y2x'
    return None

def detect_reverse_backrun_in_block(sub: pd.DataFrame, schema: Schema) -> List[Dict[str, Any]]:
    """
    Reverse back-run arbitrage (strict, single-pool, same-block adjacency)
    """
    arr = _read_block_arrays(sub, schema)
    n = len(sub)
    rows: List[Dict[str, Any]] = []

    for i in range(n - 1):
        # victim must be a swap
        is_swap_i = (arr['ev'][i] == 'swap') or arr['ev'][i].startswith('swap_') or (_dir_from_amounts(arr['a0'][i], arr['a1'][i]) is not None)
        if not is_swap_i:
            continue

        # immediately next must be a swap in the opposite direction by a different origin
        j = i + 1
        is_swap_j = (arr['ev'][j] == 'swap') or arr['ev'][j].startswith('swap_') or (_dir_from_amounts(arr['a0'][j], arr['a1'][j]) is not None)
        if not is_swap_j:
            continue

        dir_i = _dir_from_amounts(arr['a0'][i], arr['a1'][i])
        dir_j = _dir_from_amounts(arr['a0'][j], arr['a1'][j])
        if dir_i is None or dir_j is None:
            continue
        if dir_i == dir_j:  # must be opposite
            continue

        origin_i = arr['origins'][i]
        origin_j = arr['origins'][j]
        if origin_i is None or origin_j is None or origin_i == origin_j:
            continue

        # Pre-victim pool state & prices
        L0  = arr['Lb'][i]  if not np.isnan(arr['Lb'][i])  else (arr['La'][i-1]  if i>0 else np.nan)
        SQ0 = arr['sqb'][i] if not np.isnan(arr['sqb'][i]) else (arr['sqe'][i-1] if i>0 else np.nan)
        x0, y0 = (arr['xb'][i], arr['yb'][i]) if not np.isnan(arr['xb'][i]) else virtual_xy_from_L_sqrt(L0, SQ0)

        P_pre  = price_from_sqrtq96(SQ0) if np.isfinite(SQ0) else np.nan
        P_post = price_from_sqrtq96(arr['sqe'][i]) if np.isfinite(arr['sqe'][i]) else np.nan
        P_back_post = price_from_sqrtq96(arr['sqe'][j]) if np.isfinite(arr['sqe'][j]) else np.nan

        rows.append(dict(
            block_number=int(sub.iloc[0][schema.col_block]),
            pattern_type='ReverseBackrun',
            victim_origin=origin_i,
            arb_origin=origin_j,
            victim_dir=('x2y' if dir_i == 'swap_x2y' else 'y2x'),
            victim_tx=arr['tx'][i],
            back_tx=arr['tx'][j],
            L0=L0, sqrtP0_Q96=SQ0, x0=x0, y0=y0,
            P_victim_pre=P_pre, P_victim_post=P_post, P_back_post=P_back_post,
            S_net_token0=float(arr['a0'][i]),
            S_net_token1=float(arr['a1'][i]),
            back_a0=float(arr['a0'][j]),
            back_a1=float(arr['a1'][j]),
            gas_used=[arr['gas_used'][i], arr['gas_used'][j]],
            gas_price=[arr['gas_price'][i], arr['gas_price'][j]],
        ))

    return rows


def detect_jit_in_block(sub: pd.DataFrame, schema: Schema) -> List[Dict[str, Any]]:
    """
    Pure JIT (strict):
      mint(attacker) -> victim swaps (any direction, NOT attacker), all consecutive -> burn(attacker)
    No other actions allowed between anchors. Assumes single-pool input (as in this script).
    """
    arr = _read_block_arrays(sub, schema)
    n = len(sub)
    rows: List[Dict[str, Any]] = []

    i = 0
    while i < n:
        if arr['ev'][i] == 'mint':
            attacker = arr['origins'][i]
            lower    = arr['tlo'][i]
            upper    = arr['thi'][i]
            mint_idx = i

            # --- victims must be consecutive swaps by NON-attacker (any direction) ---
            swaps_idx: List[int] = []
            j = i + 1
            while j < n:
                # victim swap? (swap event & not attacker)
                is_swap = (arr['ev'][j] == 'swap') or arr['ev'][j].startswith('swap_') or (_dir_from_amounts(arr['a0'][j], arr['a1'][j]) is not None)
                if is_swap and arr['origins'][j] != attacker:
                    swaps_idx.append(j)
                    j += 1
                    continue
                break  # first non-victim ends the contiguous victim span

            # need ≥1 victim and next must be the burn by the same attacker (same range), immediately
            if swaps_idx and j < n and arr['ev'][j] == 'burn' and arr['origins'][j] == attacker and arr['tlo'][j] == lower and arr['thi'][j] == upper:
                burn_idx = j

                first_swap = swaps_idx[0]
                last_swap  = swaps_idx[-1]

                # pre-strategy state (use state at before the first swap)
                # L0  = arr['Lb'][first_swap]  if not np.isnan(arr['Lb'][first_swap])  else arr['La'][mint_idx]
                # SQ0 = arr['sqb'][first_swap] if not np.isnan(arr['sqb'][first_swap]) else arr['sqe'][mint_idx]
                # x0, y0 = (arr['xb'][first_swap], arr['yb'][first_swap]) if not np.isnan(arr['xb'][first_swap]) else virtual_xy_from_L_sqrt(L0, SQ0)
                L0  = arr['Lb'][mint_idx]
                SQ0 = arr['sqb'][mint_idx]
                x0, y0 = virtual_xy_from_L_sqrt(L0, SQ0)

                vict_a0 = float(np.nansum(arr['a0'][first_swap:last_swap+1]))
                vict_a1 = float(np.nansum(arr['a1'][first_swap:last_swap+1]))
                mint_amount = float(arr['amt'][mint_idx]) if not np.isnan(arr['amt'][mint_idx]) else np.nan
                burn_amount = float(arr['amt'][burn_idx]) if not np.isnan(arr['amt'][burn_idx]) else np.nan

                # Capture the token amounts for the mint and burn events
                mint_amount0 = float(arr['a0'][mint_idx])
                mint_amount1 = float(arr['a1'][mint_idx])
                burn_amount0 = float(arr['a0'][burn_idx])
                burn_amount1 = float(arr['a1'][burn_idx])

                # ---- attacker's liquidity share right after mint ----
                L_before_mint = arr['Lb'][mint_idx] if not np.isnan(arr['Lb'][mint_idx]) else (arr['La'][mint_idx - 1] if mint_idx > 0 else np.nan)
                L_after_mint  = (mint_amount + L_before_mint) if (np.isfinite(mint_amount) and np.isfinite(L_before_mint)) else np.nan
                attacker_liq_share = (mint_amount / L_after_mint) if (np.isfinite(L_after_mint) and L_after_mint != 0) else np.nan

                rows.append(dict(
                    block_number=int(sub.iloc[0][schema.col_block]),
                    pattern_type='Pure JIT',
                    origin=attacker,
                    mint_tick_lower=lower,
                    mint_tick_upper=upper,
                    mint_tx=arr['tx'][mint_idx],
                    burn_tx=arr['tx'][burn_idx],
                    victim_txs=[arr['tx'][k] for k in swaps_idx],
                    L0=L0, sqrtP0_Q96=SQ0, x0=x0, y0=y0,
                    S_net_token0=vict_a0, S_net_token1=vict_a1,
                    mint_amount=mint_amount,
                    burn_amount=burn_amount,
                    mint_amount0=mint_amount0,
                    mint_amount1=mint_amount1,
                    burn_amount0=burn_amount0,
                    burn_amount1=burn_amount1,
                    attacker_liq_share=attacker_liq_share,
                    gas_used=[arr['gas_used'][mint_idx]] + [arr['gas_used'][k] for k in swaps_idx] + [arr['gas_used'][burn_idx]],
                    gas_price=[arr['gas_price'][mint_idx]] + [arr['gas_price'][k] for k in swaps_idx] + [arr['gas_price'][burn_idx]],
                ))
                i = burn_idx + 1
                continue

        i += 1

    return rows



def detect_sandwich_in_block(sub: pd.DataFrame, schema: Schema, min_victims: int) -> List[Dict[str, Any]]:
    """
    Sandwich (strict, single-pool):
      Classical:     front swap(att) -> victim swaps(dir D, not att), all consecutive -> back swap(att, opposite dir)
      JIT-sandwich:  front swap(att) -> mint(att) -> victim swaps(dir D, not att), all consecutive -> burn(att) -> back swap(att, opposite dir)
    No other actions allowed between anchors; both attacker legs share the same origin.
    """
    arr = _read_block_arrays(sub, schema)
    n = len(sub)
    rows: List[Dict[str, Any]] = []

    i = 0
    while i < n:
        # front-run candidate
        is_swap = (arr['ev'][i] == 'swap') or arr['ev'][i].startswith('swap_') or (_dir_from_amounts(arr['a0'][i], arr['a1'][i]) is not None)
        if not is_swap:
            i += 1
            continue

        front_dir = _dir_from_amounts(arr['a0'][i], arr['a1'][i]) or ('swap_x2y' if (arr['a0'][i] > 0) else 'swap_y2x')
        back_dir  = 'swap_x2y' if front_dir == 'swap_y2x' else 'swap_y2x'
        attacker  = arr['origins'][i]

        # pre-strategy pool state
        L0  = arr['Lb'][i]  if not np.isnan(arr['Lb'][i])  else (arr['La'][i-1]  if i>0 else np.nan)
        SQ0 = arr['sqb'][i] if not np.isnan(arr['sqb'][i]) else (arr['sqe'][i-1] if i>0 else np.nan)
        x0, y0 = (arr['xb'][i], arr['yb'][i]) if not np.isnan(arr['xb'][i]) else virtual_xy_from_L_sqrt(L0, SQ0)

        # ==========================
        # A) JIT-sandwich (strict)
        # ==========================
        mint_idx = i + 1 if (i + 1 < n and arr['ev'][i + 1] == 'mint' and arr['origins'][i + 1] == attacker) else None
        if mint_idx is not None:
            # consecutive victims after mint
            victims, victim_idx, victim_hashes = [], [], []
            k = mint_idx + 1
            while k < n:
                is_swap_k = (arr['ev'][k] == 'swap') or arr['ev'][k].startswith('swap_') or (_dir_from_amounts(arr['a0'][k], arr['a1'][k]) is not None)
                if is_swap_k and (_dir_from_amounts(arr['a0'][k], arr['a1'][k]) == front_dir) and (arr['origins'][k] != attacker):
                    victims.append([arr['a0'][k], arr['a1'][k]])
                    victim_idx.append(k)
                    victim_hashes.append(arr['tx'][k])
                    k += 1
                    continue
                break  # end of the strictly consecutive victim window

            # need ≥ min_victims victims and next must be burn(attacker), then next must be back-run(attacker, back_dir)
            if len(victims) >= min_victims and k < n and arr['ev'][k] == 'burn' and arr['origins'][k] == attacker:
                burn_idx = k
                back_idx = k + 1 if (k + 1 < n and (_dir_from_amounts(arr['a0'][k + 1], arr['a1'][k + 1]) == back_dir) and (arr['origins'][k + 1] == attacker)) else None
                if back_idx is not None:
                    SQv0 = arr['sqe'][i]                 # price right after front-run
                    Pv0  = price_from_sqrtq96(SQv0) if np.isfinite(SQv0) else np.nan
                    SQv1 = arr['sqe'][victim_idx[-1]]    # after last victim
                    Pv1  = price_from_sqrtq96(SQv1) if np.isfinite(SQv1) else np.nan
                    vict_a0 = float(np.nansum(arr['a0'][victim_idx]))
                    vict_a1 = float(np.nansum(arr['a1'][victim_idx]))
                    L_mint = float(arr['amt'][mint_idx]) if not np.isnan(arr['amt'][mint_idx]) else np.nan
                    L_burn = float(arr['amt'][burn_idx]) if not np.isnan(arr['amt'][burn_idx]) else np.nan
                    mint_amount0 = float(arr['a0'][mint_idx]) if not np.isnan(arr['a0'][mint_idx]) else np.nan
                    mint_amount1 = float(arr['a1'][mint_idx]) if not np.isnan(arr['a1'][mint_idx]) else np.nan
                    burn_amount0 = float(arr['a0'][burn_idx]) if not np.isnan(arr['a0'][burn_idx]) else np.nan
                    burn_amount1 = float(arr['a1'][burn_idx]) if not np.isnan(arr['a1'][burn_idx]) else np.nan

                    # ---- attacker's liquidity share right after mint ----
                    L_before_mint = arr['Lb'][mint_idx] if not np.isnan(arr['Lb'][mint_idx]) else (arr['La'][mint_idx - 1] if mint_idx > 0 else np.nan)
                    L_after_mint  = (L_mint + L_before_mint) if (np.isfinite(L_mint) and np.isfinite(L_before_mint)) else np.nan
                    attacker_liq_share = (L_mint / L_after_mint) if (np.isfinite(L_after_mint) and L_after_mint != 0) else np.nan

                    rows.append(dict(
                        block_number=int(sub.iloc[0][schema.col_block]),
                        pattern_type='JIT-Sandwich',
                        origin=attacker,
                        front_dir=front_dir,
                        front_a0=float(arr['a0'][i]),
                        front_a1=float(arr['a1'][i]),
                        back_dir=back_dir,
                        back_a0=float(arr['a0'][back_idx]),
                        back_a1=float(arr['a1'][back_idx]),
                        mint_tx=arr['tx'][mint_idx],
                        burn_tx=arr['tx'][burn_idx],
                        front_tx=arr['tx'][i],
                        back_tx=arr['tx'][back_idx],
                        victim_txs=victim_hashes,
                        L0=L0, sqrtP0_Q96=SQ0, x0=x0, y0=y0,
                        P_victim_pre=Pv0, P_victim_post=Pv1,
                        S_net_token0=vict_a0, S_net_token1=vict_a1,
                        L_mint=L_mint,
                        L_burn=L_burn,
                        mint_amount0=mint_amount0,
                        mint_amount1=mint_amount1,
                        burn_amount0=burn_amount0,
                        burn_amount1=burn_amount1,
                        attacker_liq_share=attacker_liq_share, 
                        gas_used=[arr['gas_used'][i], arr['gas_used'][mint_idx]] + [arr['gas_used'][k] for k in victim_idx] + [arr['gas_used'][burn_idx], arr['gas_used'][back_idx]],
                        gas_price=[arr['gas_price'][i], arr['gas_price'][mint_idx]] + [arr['gas_price'][k] for k in victim_idx] + [arr['gas_price'][burn_idx], arr['gas_price'][back_idx]],
                    ))
                    i = back_idx + 1
                    continue  # finished this pattern

        # ==========================
        # B) Classical (strict)
        # ==========================
        victims, victim_idx, victim_hashes = [], [], []
        k = i + 1
        while k < n:
            is_swap_k = (arr['ev'][k] == 'swap') or arr['ev'][k].startswith('swap_') or (_dir_from_amounts(arr['a0'][k], arr['a1'][k]) is not None)
            if is_swap_k and (_dir_from_amounts(arr['a0'][k], arr['a1'][k]) == front_dir) and (arr['origins'][k] != attacker):
                victims.append([arr['a0'][k], arr['a1'][k]])
                victim_idx.append(k)
                victim_hashes.append(arr['tx'][k])
                k += 1
                continue
            break  # end of strictly consecutive victim window

        if len(victims) >= min_victims:
            # next must be the back-run swap by the SAME attacker in opposite direction
            back_idx = k if (k < n and (_dir_from_amounts(arr['a0'][k], arr['a1'][k]) == back_dir) and (arr['origins'][k] == attacker)) else None
            if back_idx is not None:
                SQv0 = arr['sqe'][i]               # after front-run
                Pv0  = price_from_sqrtq96(SQv0) if np.isfinite(SQv0) else np.nan
                SQv1 = arr['sqe'][victim_idx[-1]]  # after last victim
                Pv1  = price_from_sqrtq96(SQv1) if np.isfinite(SQv1) else np.nan
                vict_a0 = float(np.nansum(arr['a0'][victim_idx]))
                vict_a1 = float(np.nansum(arr['a1'][victim_idx]))

                rows.append(dict(
                    block_number=int(sub.iloc[0][schema.col_block]),
                    pattern_type='Classical',
                    origin=attacker,
                    front_dir=front_dir,
                    front_a0=float(arr['a0'][i]),
                    front_a1=float(arr['a1'][i]),
                    back_dir=back_dir,
                    back_a0=float(arr['a0'][back_idx]),
                    back_a1=float(arr['a1'][back_idx]),
                    front_tx=arr['tx'][i],
                    back_tx=arr['tx'][back_idx],
                    victim_txs=victim_hashes,
                    L0=L0, sqrtP0_Q96=SQ0, x0=x0, y0=y0,
                    P_victim_pre=Pv0, P_victim_post=Pv1,
                    S_net_token0=vict_a0, S_net_token1=vict_a1,
                    L_mint=np.nan,
                    L_burn=np.nan,
                    mint_amount0=np.nan,
                    mint_amount1=np.nan,
                    burn_amount0=np.nan,
                    burn_amount1=np.nan,
                    attacker_liq_share=np.nan, 
                    gas_used=[arr['gas_used'][i]] + [arr['gas_used'][k] for k in victim_idx] + [arr['gas_used'][back_idx]],
                    gas_price=[arr['gas_price'][i]] + [arr['gas_price'][k] for k in victim_idx] + [arr['gas_price'][back_idx]],
                ))
                i = back_idx + 1
                continue

        # no pattern starting at i
        i += 1

    return rows



# ---------------------------------------------------------------------
# Wrappers to run detectors over chunks
# ---------------------------------------------------------------------

def _process_chunk(detector_name: str, chunk: List[Tuple[int, int, int]], min_victims: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for block, lo, hi in chunk:
        sub = _GDF.iloc[lo:hi]
        if detector_name == 'JIT':
            out.extend(detect_jit_in_block(sub, _GSCHEMA))
        elif detector_name == 'RBACKRUN':
            out.extend(detect_reverse_backrun_in_block(sub, _GSCHEMA))
        else:
            out.extend(detect_sandwich_in_block(sub, _GSCHEMA, min_victims))
    return out



def run_detector_mp(df: pd.DataFrame, schema: Schema, name: str, n_jobs: int, chunk_size: int, min_victims: int, quiet: bool) -> pd.DataFrame:
    slices = _build_block_slices(df, schema)
    chunks = _chunkify(slices, chunk_size)
    total = len(chunks)

    if not quiet:
        if name == 'JIT':
            label = 'JIT collector ⚡'
        elif name == 'RBACKRUN':
            label = 'Reverse Arbitrage collector 🔄'
        else:
            label = 'Sandwich collector 🥪'
        print(f"🧪 {label}: {len(slices):,} blocks → {total:,} chunks (size={chunk_size})")

    if n_jobs in (None, 0) or n_jobs < -1:
        n_jobs = -1
    if n_jobs == -1:
        n_jobs = max(1, mp.cpu_count())

    rows: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=n_jobs, initializer=_pool_init, initargs=(df, schema)) as pool, tqdm(total=total, disable=quiet, desc=f"{name} progress", unit="chunk", mininterval=0.2, smoothing=0.1) as pbar:
        futures = [pool.submit(_process_chunk, name, chunk, min_victims) for chunk in chunks]
        for fut in as_completed(futures):
            rows.extend(fut.result())
            pbar.update(1)

    out = pd.DataFrame(rows)
    if not quiet:
        if name == 'JIT':
            print(f"✅ JIT found: {len(out):,} events.")
        elif name == 'RBACKRUN':
            print(f"✅ Reverse Arbitrage found: {len(out):,} events.")
        else:
            print(f"✅ Sandwich found: {len(out):,} events.")
    return out


# ---------------------------------------------------------------------
# Augment with Section 3 metrics (DIRECTION-AWARE)
# ---------------------------------------------------------------------

def _safe_div(n, d) -> np.ndarray:
    n = np.asarray(n, dtype=float)
    d = np.asarray(d, dtype=float)
    shape = np.broadcast(n, d).shape
    out = np.full(shape, np.nan, dtype=float)
    mask = (d != 0) & np.isfinite(n) & np.isfinite(d)
    np.divide(n, d, out=out, where=mask)
    return out


def augment_reverse_backruns(df: pd.DataFrame, fee_bps: Optional[float]) -> pd.DataFrame:
    if df.empty:
        return df

    if fee_bps is None:
        fee_bps = 5.0
    f = float(fee_bps) / 10_000.0
    r = 1.0 - f

    x0 = df['x0'].astype(float).to_numpy()
    y0 = df['y0'].astype(float).to_numpy()
    P0 = _safe_div(y0, x0)

    dir_x2y = (df['victim_dir'].astype(str).values == 'x2y')

    # NOTE: Swap `amount0/amount1` are *gross* pool deltas. In the Section 3 model,
    # the fee is applied on the input side in the invariant via `r = 1 - f`.
    S0_gross = df['S_net_token0'].astype(float).to_numpy()
    S1_gross = df['S_net_token1'].astype(float).to_numpy()

    sigma0_gross = _safe_div(S0_gross, x0)
    sigma1_gross = _safe_div(S1_gross, y0)
    sigma0_net = r * sigma0_gross
    sigma1_net = r * sigma1_gross

    sigma_native_gross = np.where(dir_x2y, sigma0_gross, sigma1_gross)
    sigma_native_net = np.where(dir_x2y, sigma0_net, sigma1_net)

    I_theory = np.where(
        dir_x2y,
        price_impact_x2y(r, sigma_native_gross),
        price_impact_y2x(r, sigma_native_gross),
    )
    sigma_min_br = backrun_sigma_min(r)

    base = np.where(dir_x2y, x0, y0)
    pi_star_native = np.array(
        [
            backrun_opt_profit_base(b, r, s) if (np.isfinite(b) and np.isfinite(s)) else np.nan
            for b, s in zip(base, sigma_native_gross)
        ],
        dtype=float,
    )
    pi_star_token0 = np.where(dir_x2y, pi_star_native, _safe_div(pi_star_native, P0))
    pi_star_per_x0 = _safe_div(pi_star_token0, x0)

    back_a0 = df['back_a0'].astype(float).to_numpy()
    back_a1 = df['back_a1'].astype(float).to_numpy()
    back_x2y = (back_a0 > 0) & (back_a1 < 0)
    back_y2x = ~back_x2y
    # Trader cashflows use the *gross* pool deltas (no `/r` conversion needed).
    x_cost = np.where(back_x2y,  back_a0, 0.0)
    x_get  = np.where(back_y2x, -back_a0,     0.0)
    y_cost = np.where(back_y2x,  back_a1, 0.0)
    y_get  = np.where(back_x2y, -back_a1,     0.0)
    profit_obs_token0 = (x_get - x_cost) + _safe_div((y_get - y_cost), P0)
    profit_obs_per_x0 = _safe_div(profit_obs_token0, x0)

    P_pre  = df['P_victim_pre'].astype(float).to_numpy()
    P_post = df['P_victim_post'].astype(float).to_numpy()
    I_measured = _safe_div(P_post, P_pre) - 1.0

    if 'P_back_post' in df.columns:
        P_back = df['P_back_post'].astype(float).to_numpy()
    else:
        P_back = np.full_like(P0, np.nan, dtype=float)

    viable_sigma = sigma_native_gross >= sigma_min_br
    dir_ok = np.where(dir_x2y, I_measured < 0, I_measured > 0)
    reverted = np.where(dir_x2y, P_back > P_post, P_back < P_post)
    reverted = np.where(np.isfinite(P_back) & np.isfinite(P_post), reverted, False)

    TOL_NEAR_TARGET = 1e-2  # 1%
    near_target = (
        np.isfinite(P0) & np.isfinite(P_back) &
        (np.abs(P_back - r * P0) <= TOL_NEAR_TARGET * np.abs(P0))
    )

    reverse_label = np.where(
        ~viable_sigma, 'subthreshold',
        np.where(
            ~dir_ok, 'wrong_sign',
            np.where(~reverted, 'no_reversion', 'candidate')
        )
    )

    mmax_JIT_token0 = np.where(
        dir_x2y,
        f * S0_gross,
        f * _safe_div(S1_gross, P0),
    )

    return df.assign(
        fee_fraction=f, r=r,
        sigma_net=sigma_native_net,
        sigma_gross=sigma_native_gross,
        sigma0_net=sigma0_net, sigma0_gross=sigma0_gross,
        sigma1_net=sigma1_net, sigma1_gross=sigma1_gross,
        I_theory=I_theory, I_measured=I_measured,
        sigma_min_backrun=sigma_min_br,
        br_pi_star_token0=pi_star_token0,
        br_pi_star_per_x0=pi_star_per_x0,
        profit_obs_token0=profit_obs_token0,
        profit_obs_per_x0=profit_obs_per_x0,
        viable_sigma=viable_sigma,
        reverted=reverted,
        near_target=near_target,
        reverse_label=reverse_label,
        mmax_JIT_token0=mmax_JIT_token0,
    )



def augment_jit(df: pd.DataFrame, fee_bps: Optional[float]) -> pd.DataFrame:
    """
    Augments JIT cycles with Section 3 metrics, including a corrected
    empirical profit calculation for Uniswap v3.

    Profit is calculated as:
    Π_total = Π_fee + Π_IL
    
    where:
      - Π_fee is the directly collected fee revenue.
      - Π_IL is the value of the inventory change (burn amounts - mint amounts).
    """
    if df.empty:
        return df

    if fee_bps is None:
        fee_bps = 5.0
    f = float(fee_bps) / 10_000.0
    r = 1.0 - f

    # --- Pre-computation of state variables ---
    x0 = df['x0'].astype(float).to_numpy()
    y0 = df['y0'].astype(float).to_numpy()
    P0 = _safe_div(y0, x0)

    # NOTE: Swap `amount0/amount1` are *gross* pool deltas. The input fee is
    # applied only in the invariant via `r = 1 - f` (see paper §3 and `change.md`).
    S0_gross = df['S_net_token0'].astype(float).to_numpy()
    S1_gross = df['S_net_token1'].astype(float).to_numpy()

    dir_x2y = S0_gross >= 0
    victim_dir = np.where(dir_x2y, 'x2y', 'y2x')

    sigma0_gross = _safe_div(S0_gross, x0)
    sigma1_gross = _safe_div(S1_gross, y0)
    sigma_native_gross = np.where(dir_x2y, sigma0_gross, sigma1_gross)
    sigma0_net = r * sigma0_gross
    sigma1_net = r * sigma1_gross
    sigma_native_net = r * sigma_native_gross
    share = df['attacker_liq_share'].astype(float).to_numpy()
    
    # --- 1. Calculate Fee Revenue Component ---
    # This is the profit from fees collected, based on the attacker's share of liquidity.
    profit_fee_token0 = np.where(
        dir_x2y,
        f * S0_gross * share,
        f * share * _safe_div(S1_gross, P0)
    )

    # --- 2. Calculate Inventory Change (Impermanent Loss) Component ---
    # This is the PnL from the rebalancing of the LP's assets.
    # Note: For Uniswap v3 events, burn amounts are negative, mints are positive.
    # The change in the LP's holdings is thus (withdrawn - deposited).
    # Since amounts in events are absolute values, we use (burn - mint).
    mint_a0 = df['mint_amount0'].astype(float).to_numpy()
    mint_a1 = df['mint_amount1'].astype(float).to_numpy()
    burn_a0 = df['burn_amount0'].astype(float).to_numpy()
    burn_a1 = df['burn_amount1'].astype(float).to_numpy()
    
    # Value of inventory change, in token0 units
    delta_a0 = burn_a0 - mint_a0
    delta_a1 = burn_a1 - mint_a1
    profit_il_token0 = delta_a0 + _safe_div(delta_a1, P0)

    # --- 3. Calculate Total Empirical Profit ---
    profit_total_token0 = profit_fee_token0 + profit_il_token0
    profit_total_per_x0 = _safe_div(profit_total_token0, x0)

    # --- Theory Metrics (unchanged) ---
    I_theory = np.where(
        dir_x2y,
        price_impact_x2y(r, sigma_native_gross),
        price_impact_y2x(r, sigma_native_gross),
    )

    return df.assign(
        fee_fraction=f,
        r=r,
        victim_dir=victim_dir,
        sigma_net=sigma_native_net,
        sigma_gross=sigma_native_gross,
        sigma0_net=sigma0_net,
        sigma0_gross=sigma0_gross,
        sigma1_net=sigma1_net,
        sigma1_gross=sigma1_gross,
        I_theory=I_theory,
        profit_fee_token0=profit_fee_token0,
        profit_il_token0=profit_il_token0,
        profit_total_token0=profit_total_token0,
        profit_total_per_x0=profit_total_per_x0,
    )

def augment_sandwich(df: pd.DataFrame, fee_bps: Optional[float], gamma: float, grid_n: int) -> pd.DataFrame:
    if df.empty:
        return df
    if fee_bps is None:
        fee_bps = 5.0
    f = float(fee_bps) / 10_000.0
    r = 1.0 - f

    x0 = df['x0'].astype(float).to_numpy()
    y0 = df['y0'].astype(float).to_numpy()
    P0 = _safe_div(y0, x0)

    # Determine victim direction: x2y means victim inputs token0; y2x means victim inputs token1
    if 'front_dir' in df.columns:
        dir_vals = df['front_dir'].astype(str).to_numpy()
        victim_dir = np.where(dir_vals == 'swap_x2y', 'x2y', 'y2x')
    else:
        S0_gross_tmp = df['S_net_token0'].astype(float).to_numpy()
        victim_dir = np.where(S0_gross_tmp >= 0, 'x2y', 'y2x')
    dir_x2y = (victim_dir == 'x2y')

    # Pool deltas (front/back) — use *gross* deltas for trader cashflows.
    front_a0 = df['front_a0'].astype(float).fillna(0.0).to_numpy()
    back_a0  = df['back_a0'].astype(float).fillna(0.0).to_numpy()
    front_a1 = df['front_a1'].astype(float).fillna(0.0).to_numpy()
    back_a1  = df['back_a1'].astype(float).fillna(0.0).to_numpy()

    # Base sandwich PnL in token0 (from the two attacker swaps only).
    a0_in  = np.maximum(front_a0, 0.0) + np.maximum(back_a0, 0.0)
    a0_out = np.maximum(-front_a0, 0.0) + np.maximum(-back_a0, 0.0)
    a1_in  = np.maximum(front_a1, 0.0) + np.maximum(back_a1, 0.0)
    a1_out = np.maximum(-front_a1, 0.0) + np.maximum(-back_a1, 0.0)
    profit_token0_base = (a0_out - a0_in) + _safe_div((a1_out - a1_in), P0)

    # Victim flow and normalized sizes (σ): treat event deltas as *gross*.
    S0_gross = df['S_net_token0'].astype(float).to_numpy()
    S1_gross = df['S_net_token1'].astype(float).to_numpy()

    sigma0_gross = _safe_div(S0_gross, x0)
    sigma1_gross = _safe_div(S1_gross, y0)
    sigma0_net = r * sigma0_gross
    sigma1_net = r * sigma1_gross

    sigma_native_net   = np.where(dir_x2y, sigma0_net,   sigma1_net)
    sigma_native_gross = np.where(dir_x2y, sigma0_gross, sigma1_gross)

    # Add JIT LP fee revenue ONLY for JIT-Sandwich rows (Eq. 48)
    has_lp_share_col = ('attacker_liq_share' in df.columns)
    lp_share = df['attacker_liq_share'].astype(float).to_numpy() if has_lp_share_col else np.full_like(x0, np.nan)

    pt = df.get('pattern_type', pd.Series(index=df.index, dtype=str)).astype(str)
    is_jit_sandwich = (pt.str.contains('JIT', case=False, na=False).to_numpy() |
                       (df.get('mint_tx', pd.Series(index=df.index)).notna().to_numpy() &
                        df.get('burn_tx', pd.Series(index=df.index)).notna().to_numpy()))

    S_input_gross_native = np.where(dir_x2y, S0_gross, S1_gross)
    fee_native = f * S_input_gross_native * lp_share
    jit_fee_token0 = np.where(dir_x2y, fee_native, _safe_div(fee_native, P0))
    jit_fee_token0 = np.where(is_jit_sandwich & np.isfinite(jit_fee_token0) & (lp_share > 0), jit_fee_token0, 0.0)

    # LP token delta term (burn−mint), valued in token0
    mint_amount0 = df.get('mint_amount0', pd.Series(index=df.index)).astype(float).fillna(0.0).to_numpy()
    mint_amount1 = df.get('mint_amount1', pd.Series(index=df.index)).astype(float).fillna(0.0).to_numpy()
    burn_amount0 = df.get('burn_amount0', pd.Series(index=df.index)).astype(float).fillna(0.0).to_numpy()
    burn_amount1 = df.get('burn_amount1', pd.Series(index=df.index)).astype(float).fillna(0.0).to_numpy()

    lp_token_delta_token0 = (burn_amount0 - mint_amount0) + _safe_div((burn_amount1 - mint_amount1), P0)
    # Keep it only on JIT-Sandwich rows (else 0)
    lp_token_delta_token0 = np.where(is_jit_sandwich, lp_token_delta_token0, 0.0)

    # Total profit in token0 now includes: base sandwich + JIT fees + LP token delta
    profit_token0 = profit_token0_base + jit_fee_token0 + lp_token_delta_token0
    profit_per_x0 = _safe_div(profit_token0, x0)

    # Theoretical price impact, backrun thresholds & envelopes
    I_theory = np.where(dir_x2y,
                        price_impact_x2y(r, sigma_native_gross),
                        price_impact_y2x(r, sigma_native_gross))
    sigma_min_br = backrun_sigma_min(r)

    base = np.where(dir_x2y, x0, y0)
    br_pi_star_native_units = np.array(
        [backrun_opt_profit_base(b, r, s) if (np.isfinite(b) and np.isfinite(s)) else np.nan
         for b, s in zip(base, sigma_native_gross)],
        dtype=float,
    )
    br_pi_star_token0 = np.where(dir_x2y, br_pi_star_native_units,
                                 _safe_div(br_pi_star_native_units, P0))
    br_pi_star_per_x0 = _safe_div(br_pi_star_token0, x0)

    eps_gross = np.where(
        dir_x2y,
        _safe_div(np.abs(front_a0), x0),
        _safe_div(np.abs(front_a1), y0),
    )
    eps_net = r * eps_gross

    sand_pi_obs_native = [sandwich_profit_normalized(e, s, r)
                          for e, s in zip(eps_gross, sigma_native_gross)]
    eps_star_list, sand_pi_star_native = [], []
    for s in sigma_native_gross:
        e_star, pi_star = sandwich_profit_star(s, r, gamma, grid_n)
        eps_star_list.append(e_star)
        sand_pi_star_native.append(pi_star)

    sand_obs_native_units  = base * np.array(sand_pi_obs_native,  dtype=float)
    sand_star_native_units = base * np.array(sand_pi_star_native, dtype=float)

    sand_pi_obs_token0  = np.where(dir_x2y, sand_obs_native_units,
                                   _safe_div(sand_obs_native_units,  P0))
    sand_pi_star_token0 = np.where(dir_x2y, sand_star_native_units,
                                   _safe_div(sand_star_native_units, P0))

    sand_pi_obs_per_x0  = _safe_div(sand_pi_obs_token0,  x0)
    sand_pi_star_per_x0 = _safe_div(sand_pi_star_token0, x0)

    P_pre  = df['P_victim_pre'].astype(float).to_numpy()
    P_post = df['P_victim_post'].astype(float).to_numpy()
    I_measured = _safe_div(P_post, P_pre) - 1.0

    # -------------------------
    # ε_max (classical vs JIT)
    # -------------------------
    eps_max_classic_series = evaluate_eps_max_series(pd.Series(sigma_native_gross), gamma, r)
    eps_max_classic = eps_max_classic_series.to_numpy()

    # Infer α from attacker LP share when present: share = α/(α+1) ⇒ α = share/(1-share)
    share = lp_share
    share_clamped = np.where(np.isfinite(share), np.clip(share, 0.0, 1.0 - 1e-12), np.nan)
    alpha_from_share = share_clamped / (1.0 - share_clamped)

    eps_max_jit_series = evaluate_eps_max_series_jit(
        pd.Series(sigma_native_gross), gamma, r, pd.Series(alpha_from_share)
    )
    eps_max_jit = eps_max_jit_series.to_numpy()

    eps_max_final = np.where(is_jit_sandwich & np.isfinite(eps_max_jit), eps_max_jit, eps_max_classic)

    out = df.assign(
        fee_fraction=f,
        r=r,
        victim_dir=victim_dir,
        # components
        profit_token0_base=profit_token0_base,
        jit_fee_token0=jit_fee_token0,
        lp_token_delta_token0=lp_token_delta_token0,
        # totals
        profit_token0=profit_token0,
        profit_per_x0=profit_per_x0,
        eps_net=eps_net,
        eps_gross=eps_gross,
        sigma_net=sigma_native_net,
        sigma_gross=sigma_native_gross,
        sigma0_net=sigma0_net,
        sigma0_gross=sigma0_gross,
        sigma1_net=sigma1_net,
        sigma1_gross=sigma1_gross,
        I_theory=I_theory,
        I_measured=I_measured,
        sigma_min_backrun=sigma_min_br,
        br_pi_star_token0=br_pi_star_token0,
        sand_pi_obs_token0=sand_pi_obs_token0,
        sand_pi_star_token0=sand_pi_star_token0,
        br_pi_star_per_x0=br_pi_star_per_x0,
        sand_pi_obs_per_x0=sand_pi_obs_per_x0,
        sand_pi_star=sand_pi_star_per_x0,
        eps_max_classic=eps_max_classic,
        eps_max_jit=eps_max_jit,
        eps_max=eps_max_final,
        gamma_used=gamma,
        sigma_star_br_vs_jit_phi0=sigma_star_backrun_vs_jit(f, phi=0.0, r=r),
    )
    return out



def evaluate_eps_max_series(sigmas: pd.Series, gamma: float, r: float) -> pd.Series:
    return pd.Series([eps_max_under_slippage(float(s), gamma, r) for s in sigmas], index=sigmas.index)


# ---------------------------------------------------------------------
# CLI & main
# ---------------------------------------------------------------------

def _discover_default_input_path() -> Path:
    """Pick a deterministic, repo-relative default input dataset."""
    candidates = [
        REPO_ROOT / "data" / "processed" / "univ3_pool_events_with_origin.csv",
        REPO_ROOT / "data" / "processed" / "univ3_pool_events_with_running_state.csv",
        REPO_ROOT / "data" / "raw" / "usdc_weth_05.csv",
    ]
    for path in candidates:
        if path.exists():
            return path

    raw_dir = REPO_ROOT / "data" / "raw"
    discovered = sorted(raw_dir.glob("univ3_*.csv")) if raw_dir.exists() else []
    if discovered:
        return discovered[0]

    return candidates[0]


def parse_args() -> argparse.Namespace:
    default_in = _discover_default_input_path()
    p = argparse.ArgumentParser(description="Collect JIT cycles and Sandwich attacks + Section 3 metrics (no column normalization).")
    p.add_argument("--in", dest="in_path", default=str(default_in), help="Input CSV, Parquet or Pickle file (default: %(default)s).")
    p.add_argument("--outdir", default=str(REPO_ROOT / "mev_out"), help="Output directory (default: %(default)s).")
    p.add_argument("--n-jobs", type=int, default=-1, help="Worker processes (-1 = all cores, default: %(default)s).")
    p.add_argument("--chunk-size", type=int, default=64, help="Blocks per task chunk to reduce IPC overhead (default: %(default)s).")
    p.add_argument("--min-victims", type=int, default=1, help="Min victim swaps for a sandwich (default: %(default)s).")
    p.add_argument("--fee-bps", "--fee_bps", dest="fee_bps", type=float, default=5.0, help="Pool fee tier for outputs and formulas; accepts bps (5, 30, 100) or fraction (0.0005) (default: %(default)s).")
    # NOTE: argparse help strings use %-formatting; escape literal '%' as '%%'.
    p.add_argument("--gamma", type=float, default=0.01, help="Victim slippage tolerance used in theory constraints (default: %(default)s = 1%%).")
    p.add_argument("--grid-npoints", type=int, default=128, help="Grid size to maximize π(ε) under γ (default: %(default)s).")
    p.add_argument("--quiet", action="store_true", default=False, help="Less verbose printing (default: %(default)s).")
    p.add_argument("--recompute_jit", action="store_true", default=False, help="Recompute jit detectors even if output files exist (default: %(default)s).")
    p.add_argument("--recompute_sand", action="store_true", default=False, help="Recompute sandwich detectors even if output files exist (default: %(default)s).")
    p.add_argument("--recompute_br", action="store_true", default=False, help="Recompute back run arbitrage detectors even if output files exist (default: %(default)s).")
    return p.parse_args()


def _normalize_fee_bps(fee_value: float) -> float:
    """Return fee in basis points; accept either bps (>=1) or fraction (<1)."""
    fee = float(fee_value)
    if not np.isfinite(fee) or fee < 0.0:
        raise ValueError(f"Invalid fee value: {fee_value!r}. Use a non-negative float.")
    if fee < 1.0:
        return fee * 10_000.0
    return fee


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if not Path(args.in_path).exists():
        raise FileNotFoundError(
            f"Input dataset not found: {args.in_path}. "
            "Pass --in explicitly or place a dataset under data/processed or data/raw."
        )

    fee_bps = _normalize_fee_bps(args.fee_bps)
    FEE = f"{fee_bps:g}"
    outdir = Path(args.outdir)
    jit_cached = outdir / f"jit_cycles_tidy_{FEE}.csv"
    sand_cached = outdir / f"sandwich_attacks_tidy_{FEE}.csv"
    br_cached = outdir / f"reverse_backruns_tidy_{FEE}.csv"

    # Load (no normalization!)
    print("📥 Loading dataset…")
    ext = os.path.splitext(args.in_path)[1].lower()
    if ext in (".parquet", ".pq"):
        df = pd.read_parquet(args.in_path)
    elif ext in (".csv", ".gz"):
        df = pd.read_csv(args.in_path, low_memory=False)
    elif ext in (".pkl", ".pickle"):
        df = pd.read_pickle(args.in_path)
    else:
        raise ValueError("Unsupported input file. Use CSV or Parquet.")
    print(f"✅ Loaded {len(df):,} rows from {args.in_path}")

    # Resolve schema and sort by (block, logIndex)
    schema = resolve_schema(df)
    df = df.sort_values([schema.col_block, schema.col_log_index], kind='mergesort').reset_index(drop=True)

    # Decide worker processes
    n_jobs = args.n_jobs if args.n_jobs != -1 else max(1, mp.cpu_count())

    # Run detectors
    print("🚀 Starting collectors…")
    if jit_cached.exists() and not args.recompute_jit:
        print("JIT cycles already detected, skipping...")
        jit_df = pd.read_csv(jit_cached, low_memory=False)
    else:
        jit_df = run_detector_mp(df, schema, name='JIT', n_jobs=n_jobs, chunk_size=args.chunk_size, min_victims=args.min_victims, quiet=args.quiet)
    if sand_cached.exists() and not args.recompute_sand:
        print("Sandwich attacks already detected, skipping...")
        sand_df = pd.read_csv(sand_cached, low_memory=False)
    else:
        sand_df = run_detector_mp(df, schema, name='SANDWICH', n_jobs=n_jobs, chunk_size=args.chunk_size, min_victims=args.min_victims, quiet=args.quiet)
    if br_cached.exists() and not args.recompute_br:
        print("Reverse back-runs already detected, skipping...")
        rback_df = pd.read_csv(br_cached, low_memory=False)
    else:
        rback_df = run_detector_mp(df, schema, name='RBACKRUN', n_jobs=n_jobs, chunk_size=args.chunk_size, min_victims=args.min_victims, quiet=args.quiet)

    # Exclude any pair whose back-run TX is part of a detected sandwich ending (strict de-dup)
    if not sand_df.empty and not rback_df.empty:
        sand_back = set(sand_df['back_tx'].dropna().astype(str).tolist())
        before = len(rback_df)
        rback_df = rback_df[~rback_df['back_tx'].astype(str).isin(sand_back)].reset_index(drop=True)
        removed = before - len(rback_df)
        if not args.quiet:
            print(f"🧹 Removed {removed:,} reverse pairs that were sandwich back-runs.")

    # Augment with Section 3 metrics
    print("🧮 Computing Section 3 metrics…")
    jit_df  = augment_jit(jit_df, fee_bps=fee_bps)
    sand_df = augment_sandwich(sand_df, fee_bps=fee_bps, gamma=args.gamma, grid_n=args.grid_npoints)
    rback_df = augment_reverse_backruns(rback_df, fee_bps=fee_bps)

    # Save
    def save_csv(df_out: pd.DataFrame, name: str):
        path = os.path.join(args.outdir, f"{name}.csv")
        df_out.to_csv(path, index=False)
        print(f"💾 Saved {len(df_out):,} rows → {path}")

    if not rback_df.empty:
        path = os.path.join(args.outdir, f"reverse_backruns_tidy_{FEE}.csv")
        rback_df.to_csv(path, index=False)
        print(f"💾 Saved {len(rback_df):,} rows → {path}")
    else:
        print("ℹ️ No reverse back-run arbitrages detected.")

    if not jit_df.empty:
        save_csv(jit_df, f"jit_cycles_tidy_{FEE}")
    else:
        print("ℹ️ No JIT cycles detected.")

    if not sand_df.empty:
        # add a compact hash bundle for convenience
        def compact_hashes(row: pd.Series) -> List[str]:
            out = []
            if isinstance(row.get('front_tx'), str): out.append(row['front_tx'])
            if isinstance(row.get('mint_tx'), str): out.append(row['mint_tx'])
            if isinstance(row.get('victim_txs'), list): out.extend(row['victim_txs'])
            if isinstance(row.get('burn_tx'), str): out.append(row['burn_tx'])
            if isinstance(row.get('back_tx'), str): out.append(row['back_tx'])
            return out
        sand_df = sand_df.assign(tx_sequence=sand_df.apply(compact_hashes, axis=1))
        save_csv(sand_df, f"sandwich_attacks_tidy_{FEE}")
    else:
        print("ℹ️ No sandwich attacks detected.")

    print("🏁 Done. 🎉")


if __name__ == "__main__":
    if sys.platform.startswith("win"):
        mp.set_start_method("spawn", force=True)
    main()
