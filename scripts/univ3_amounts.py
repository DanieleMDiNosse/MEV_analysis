"""
univ3_amounts.py — Robust conversion of Uniswap v3 subgraph BigDecimal amounts.

The Uniswap v3 subgraph exposes token amounts as BigDecimal strings (human units),
while on-chain logs and most MEV/math pipelines operate in raw base units (ints).

This module centralizes conversion logic to keep the data pipeline consistent.
"""

from __future__ import annotations

from decimal import Decimal, ROUND_DOWN


def to_raw_units(amount_str: str, decimals: int, strict: bool = True) -> int:
    """
    Convert a BigDecimal string amount into raw integer base units.

    Parameters
    ----------
    amount_str:
        String representation of a decimal number (e.g., "-5039.850865").
        This comes from The Graph BigDecimal fields (Swap/Mint/Burn amount0/amount1).
    decimals:
        Token decimals (e.g., 6 for USDC, 18 for WETH).
    strict:
        If True, require that `amount_str * 10**decimals` is exactly integral.
        If False, truncate toward zero.

    Returns
    -------
    int
        Raw integer amount in base units (signed for swaps, unsigned for mints/burns).

    Notes
    -----
    - Uses `decimal.Decimal` to avoid float rounding.
    - `ROUND_DOWN` truncates toward zero for both positive and negative values.
    - In strict mode, any non-integral scaled result raises `ValueError`, which
      is usually a sign of mismatched decimals or unexpected subgraph formatting.

    Examples
    --------
    >>> to_raw_units("1.5", 6)
    1500000
    >>> to_raw_units("-0.01", 18)
    -10000000000000000
    """
    if decimals < 0:
        raise ValueError("decimals must be >= 0")

    amount = Decimal(str(amount_str))
    scale = Decimal(10) ** int(decimals)
    scaled = amount * scale

    integral = scaled.to_integral_value(rounding=ROUND_DOWN)
    if strict and integral != scaled:
        raise ValueError(
            f"Non-integral scaled amount: amount={amount_str!r}, decimals={decimals}, "
            f"scaled={str(scaled)}"
        )
    return int(integral)

