#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Section 3 — Interactive scatter by STRATEGY (Plotly, color only)

Creates a single interactive scatter plot of Profit vs |σ| where **color** encodes the strategy:
- JIT (realized PnL): blue
- Classical Sandwich (realized PnL): orange
- JIT-Sandwich (realized PnL): green
- Back-run Arbitrage (observed PnL): red

No wallet/origin tracking, no different markers — just color.
Saves as an interactive HTML (default) and optionally a PNG if kaleido is installed.

Inputs (CSV)
------------
• --in-jit       : jit_cycles_tidy_5.csv (preferred; falls back to legacy names)
                    (expects columns: sigma_gross, profit_total_per_x0)
• --in-sand      : sandwich_attacks_tidy_5.csv (preferred; falls back to legacy names)
                    (expects columns: pattern_type in {'Classical','JIT-Sandwich'},
                                     sigma_gross, profit_per_x0)
• --in-backrun   : reverse_backruns_tidy_5.csv (preferred; falls back to legacy names)
                    (expects: sigma_gross, profit_obs_per_x0) [optional]

Usage
-----
python scripts/section3_empirical_simple.py \
  --in-jit ./mev_out/jit_cycles_tidy_5.csv \
  --in-sand ./mev_out/sandwich_attacks_tidy_5.csv \
  --in-backrun ./mev_out/reverse_backruns_tidy_5.csv \
  --out ./mev_out/section3_scatter_by_strategy.html \
  --png-out ./mev_out/section3_scatter_by_strategy.png \
  --show
"""

from __future__ import annotations
import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# Plotly
try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception as e:
    raise SystemExit("Plotly is required. Try: pip install plotly kaleido\n" + str(e))

# ---------------- I/O helpers ----------------

def load_csv(path: str) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    if not os.path.exists(path):
        print(f"[warn] File not found: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as e:
        print(f"[warn] Failed reading {path}: {e}")
        return pd.DataFrame()

def discover_tidy_path(mev_out: Path, stem: str) -> Path:
    """Prefer current `mev_collect.py` outputs, but keep backward-compatible fallbacks."""
    candidates = [
        mev_out / f'{stem}_5.csv',
        mev_out / f'{stem}_5.0.csv',
        mev_out / f'{stem}.csv',
        mev_out / f'{stem}_None.csv',
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]

# ---------------- Builders ----------------

def prepare_jit(df_jit: pd.DataFrame, abs_sigma: bool) -> pd.DataFrame:
    if df_jit.empty:
        return pd.DataFrame()
    if 'sigma_gross' not in df_jit.columns or 'profit_total_per_x0' not in df_jit.columns:
        print("[warn] JIT CSV missing required columns {sigma_gross, profit_total_per_x0}. Skipping.")
        return pd.DataFrame()
    sig = pd.to_numeric(df_jit['sigma_gross'], errors='coerce')
    x = sig.abs() if abs_sigma else sig
    y = pd.to_numeric(df_jit['profit_total_per_x0'], errors='coerce')
    out = pd.DataFrame({'sigma': x, 'profit': y, 'strategy': 'JIT'})
    return out.replace([np.inf, -np.inf], np.nan).dropna()

def prepare_sandwich(df_sand: pd.DataFrame, abs_sigma: bool, kind: str, label: str) -> pd.DataFrame:
    if df_sand.empty:
        return pd.DataFrame()
    if 'pattern_type' not in df_sand.columns:
        print("[warn] Sandwich CSV missing 'pattern_type'. Skipping.")
        return pd.DataFrame()
    dfk = df_sand[df_sand['pattern_type'].astype(str).str.strip().str.lower() == kind]
    if dfk.empty:
        return pd.DataFrame()
    required = ['sigma_gross', 'profit_per_x0']
    if any(c not in dfk.columns for c in required):
        print(f"[warn] Sandwich '{label}' missing columns {required}. Skipping.")
        return pd.DataFrame()
    sig = pd.to_numeric(dfk['sigma_gross'], errors='coerce')
    x = sig.abs() if abs_sigma else sig
    y = pd.to_numeric(dfk['profit_per_x0'], errors='coerce')
    out = pd.DataFrame({'sigma': x, 'profit': y, 'strategy': label})
    return out.replace([np.inf, -np.inf], np.nan).dropna()

def prepare_backrun(df_br: pd.DataFrame, abs_sigma: bool) -> pd.DataFrame:
    if df_br.empty:
        return pd.DataFrame()
    required = ['sigma_gross', 'profit_obs_per_x0']
    if any(c not in df_br.columns for c in required):
        print(f"[warn] Back-run CSV missing columns {required}. Skipping.")
        return pd.DataFrame()
    sig = pd.to_numeric(df_br['sigma_gross'], errors='coerce')
    x = sig.abs() if abs_sigma else sig
    y = pd.to_numeric(df_br['profit_obs_per_x0'], errors='coerce')
    out = pd.DataFrame({'sigma': x, 'profit': y, 'strategy': 'Back-run Arbitrage'})
    return out.replace([np.inf, -np.inf], np.nan).dropna()


# ---------------- Main ----------------

def main():
    repo_root = Path(__file__).resolve().parents[1]
    mev_out = repo_root / "mev_out"
    default_jit = discover_tidy_path(mev_out, 'jit_cycles_tidy')
    default_sand = discover_tidy_path(mev_out, 'sandwich_attacks_tidy')
    default_br = discover_tidy_path(mev_out, 'reverse_backruns_tidy')

    ap = argparse.ArgumentParser(description="Interactive scatter: Profit vs |σ|, color = strategy (Plotly)")
    ap.add_argument('--in-jit', default=str(default_jit), help='Path to JIT tidy CSV (default: %(default)s).')
    ap.add_argument('--in-sand', default=str(default_sand), help='Path to sandwich tidy CSV (default: %(default)s).')
    ap.add_argument('--in-backrun', default=str(default_br), help='Path to reverse back-run tidy CSV (default: %(default)s).')
    ap.add_argument('--out', default=str(mev_out / 'section3_scatter_by_strategy.html'),
                    help='Output HTML path for the interactive figure.')
    ap.add_argument('--png-out', default='',
                    help='Optional static PNG path (requires kaleido).')
    ap.add_argument('--signed-sigma', action='store_true', help='Use signed σ instead of absolute value')
    ap.add_argument('--width', type=int, default=1100, help='Figure width in px (default: 1100)')
    ap.add_argument('--height', type=int, default=700, help='Figure height in px (default: 700)')
    ap.add_argument('--show', action='store_true', help='Open the figure after saving')
    ap.add_argument(
        '--fee-bps', '--fee_bps',
        dest='fee_bps',
        type=float,
        default=5.0,
        help="Pool fee tier used for σ-threshold guide lines. Accepts basis points (e.g., 5, 30, 100) or a fraction (e.g., 0.0005).",
    )
    args = ap.parse_args()

    # Load CSVs
    jit_df  = load_csv(args.in_jit)
    sand_df = load_csv(args.in_sand)
    br_df   = load_csv(args.in_backrun)

    # Build per-strategy frames
    use_abs = not args.signed_sigma
    frames = []

    df_jit  = prepare_jit(jit_df, use_abs)
    if not df_jit.empty: frames.append(df_jit)

    df_sand_class = prepare_sandwich(sand_df, use_abs, kind='classical', label='Classical sandwich')
    if not df_sand_class.empty: frames.append(df_sand_class)

    df_sand_jit = prepare_sandwich(sand_df, use_abs, kind='jit-sandwich', label='JIT-Sandwich')
    if not df_sand_jit.empty: frames.append(df_sand_jit)

    df_back = prepare_backrun(br_df, use_abs)
    if not df_back.empty: frames.append(df_back)

    if not frames:
        raise SystemExit("No data to plot. Check your CSV paths and required columns.")

    df_all = pd.concat(frames, ignore_index=True)

    # Colors per strategy — single marker for all
    color_map = {
        'JIT': '#1f77b4',                # tab:blue
        'Classical sandwich': '#ff7f0e', # tab:orange
        'JIT-Sandwich': '#2ca02c',       # tab:green
        'Back-run Arbitrage': '#d62728', # tab:red
    }

    x_title = 'σ'

    fig = px.scatter(
        df_all,
        x='sigma',
        y='profit',
        color='strategy',
        color_discrete_map=color_map,
        hover_name='strategy',
        hover_data={
            'sigma': ':.6f',
            'profit': ':.6f',
            'strategy': False,  # already in hover_name
        },
        opacity=0.5,
        title='MEV profits vs ' + 'σ',
        width=args.width,
        height=args.height,
    )

    fig.update_traces(marker=dict(size=6, line=dict(width=0)))  # single marker style, no outlines
    fig.update_layout(
        legend_title_text='Strategy',
        xaxis_title=x_title,
        yaxis_title=r'Profit',
        template='plotly_white',
        margin=dict(l=60, r=20, t=60, b=60),
    )

    if args.fee_bps is not None:
        # Interpret `--fee-bps` robustly: users may pass 5 (bps) or 0.0005 (fraction).
        fee_in = float(args.fee_bps)
        f = (fee_in / 10_000.0) if fee_in >= 1.0 else fee_in
        if not (0.0 <= f < 1.0):
            raise SystemExit(f"Invalid fee fraction inferred from --fee-bps={args.fee_bps!r}: f={f}.")
        r = 1.0 - f

        # Paper §3: back-run profitability threshold (Eq. 13):
        #   σ_min^br = (1/sqrt(r) - 1)/r ≈ f/2 for small f.
        sigma_min_br = (1.0 / np.sqrt(r) - 1.0) / r

        # Sandwich profit turns on at σ > σ_min(ε); as ε→0+, σ_min → (1-r)/r^2 ≈ f.
        sigma_min_sand_eps0 = (1.0 - r) / (r * r)

        c_br = '#1f77b4'
        c_sand = '#d62728'
        fig.add_vline(x=sigma_min_br, line_dash='dash', line_color=c_br)
        fig.add_vline(x=sigma_min_sand_eps0, line_dash='dash', line_color=c_sand)

        # Dummy traces to label the vertical guide lines in the legend.
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color=c_br, dash='dash'),
                                 name='σ_min back-run', hoverinfo='skip', showlegend=True))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color=c_sand, dash='dash'),
                                 name='σ_min sandwich (ε→0)', hoverinfo='skip', showlegend=True))

    # Save HTML
    out_html = args.out
    os.makedirs(os.path.dirname(out_html) or '.', exist_ok=True)
    fig.write_html(out_html, include_plotlyjs='cdn', full_html=True)
    print(f"Saved interactive HTML → {out_html}")

    # Optional PNG
    if args.png_out:
        try:
            import plotly.io as pio
            os.makedirs(os.path.dirname(args.png_out) or '.', exist_ok=True)
            pio.write_image(fig, args.png_out, width=args.width, height=args.height, scale=2)
            print(f"Saved static PNG → {args.png_out}")
        except Exception as e:
            print(f"[warn] Could not export PNG. Install 'kaleido'. Error: {e}")

    if args.show:
        fig.show()

if __name__ == '__main__':
    main()
