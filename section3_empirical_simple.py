#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Section 3 — Interactive scatter by STRATEGY (Plotly, color only)

Creates a single interactive scatter plot of Profit vs |σ| where **color** encodes the strategy:
- JIT ceiling (token0-normalized): blue
- Classical Sandwich (realized PnL): orange
- JIT-Sandwich (realized PnL): green
- Back-run Arbitrage (observed PnL): red

No wallet/origin tracking, no different markers — just color.
Saves as an interactive HTML (default) and optionally a PNG if kaleido is installed.

Inputs (CSV)
------------
• --in-jit       : jit_cycles_tidy.csv    (expects columns: sigma_gross, profit_per_x0)
• --in-sand      : sandwich_attacks_tidy.csv
                    (expects columns: pattern_type in {'Classical','JIT-Sandwich'},
                                     sigma_gross, profit_per_x0)
• --in-backrun   : reverse_backruns_tidy.csv (expects: sigma_gross, profit_obs_per_x0) [optional]

Usage
-----
python section3_scatter_plotly_by_strategy.py \
  --in-jit ./mev_out/jit_cycles_tidy.csv \
  --in-sand ./mev_out/sandwich_attacks_tidy.csv \
  --in-backrun ./mev_out/reverse_backruns_tidy.csv \
  --out ./mev_out/section3_scatter_by_strategy.html \
  --png-out ./mev_out/section3_scatter_by_strategy.png \
  --show
"""

from __future__ import annotations
import os
import argparse
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
    ap = argparse.ArgumentParser(description="Interactive scatter: Profit vs |σ|, color = strategy (Plotly)")
    ap.add_argument('--in-jit', default='./mev_out/jit_cycles_tidy_None.csv')
    ap.add_argument('--in-sand', default='./mev_out/sandwich_attacks_tidy_None.csv')
    ap.add_argument('--in-backrun', default='./mev_out/reverse_backruns_tidy_None.csv')
    ap.add_argument('--out', default='./mev_out/section3_scatter_by_strategy.html',
                    help='Output HTML path for the interactive figure.')
    ap.add_argument('--png-out', default='',
                    help='Optional static PNG path (requires kaleido).')
    ap.add_argument('--signed-sigma', action='store_true', help='Use signed σ instead of absolute value')
    ap.add_argument('--width', type=int, default=1100, help='Figure width in px (default: 1100)')
    ap.add_argument('--height', type=int, default=700, help='Figure height in px (default: 700)')
    ap.add_argument('--show', action='store_true', help='Open the figure after saving')
    ap.add_argument('--fee-bps', type=float, default=0.0005,
                    help='Pool fee tier in basis points; used for reference guide lines')
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
        half_fee = args.fee_bps / 2.0
        min_color = '#1f77b4'
        tier_color = '#d62728'
        fig.add_vline(
            x=half_fee,
            line_dash='dash',
            line_color=min_color,
        )
        fig.add_vline(
            x=args.fee_bps,
            line_dash='dash',
            line_color=tier_color,
        )
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode='lines',
                line=dict(color=min_color, dash='dash'),
                name='σ min br',
                hoverinfo='skip',
                showlegend=True,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode='lines',
                line=dict(color=tier_color, dash='dash'),
                name='Fee Tier',
                hoverinfo='skip',
                showlegend=True,
            )
        )

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
