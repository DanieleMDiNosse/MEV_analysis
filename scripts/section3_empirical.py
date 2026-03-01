#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotly version of:
Section 3 — Profit vs σ (JIT, Classical Sandwich, JIT-Sandwich, Back-run Arbitrage)

Keeps the same visualization features as the Matplotlib script:
- top-N per-group highlighted origins (separate knobs for JIT vs Sandwich/Back-run)
- marker shape by pattern, "X" marker for σ < σ_min
- non-highlighted origins in light gray with lower alpha
- same axis labels, title, dashed grid
ADDITION: hover shows 'origin' (and pattern, σ, y).

Outputs:
  • .html → interactive
  • .png/.pdf → static image (requires kaleido)

Usage (example):
  python scripts/section3_empirical.py \
    --in-jit ./mev_out/jit_cycles_tidy_5.csv \
    --in-sand ./mev_out/sandwich_attacks_tidy_5.csv \
    --in-backrun ./mev_out/reverse_backruns_tidy_5.csv \
    --color-top-origins-jit 10 --color-top-origins-sand 30 \
    --legend-max-origins 25 \
    --out ./mev_out/section3_profit_vs_sigma.html \
    --show
"""
from __future__ import annotations
import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

import colorsys

import plotly.graph_objects as go

# ---------------- I/O helpers ----------------

def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path, low_memory=False)

# ---------------- Utilities ----------------

def _sigma(series: pd.Series, use_abs: bool) -> pd.Series:
    s = pd.to_numeric(series, errors='coerce')
    return s.abs() if use_abs else s

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

# ---------------- Series builders ----------------

def prepare_jit(df_jit: pd.DataFrame, use_abs_sigma: bool) -> pd.DataFrame:
    """(σ, y, label, origin, under_threshold=False) for realized JIT PnL in token0 units."""
    if df_jit.empty:
        return df_jit
    if 'sigma_gross' not in df_jit.columns:
        raise ValueError("Expected 'sigma_gross' in JIT tidy file.")
    if 'profit_total_per_x0' not in df_jit.columns:
        raise ValueError("Expected 'profit_total_per_x0' in JIT tidy file.")

    x = _sigma(df_jit['sigma_gross'], use_abs_sigma)
    y = pd.to_numeric(df_jit['profit_total_per_x0'], errors='coerce')
    origin = df_jit['origin'].astype(str) if 'origin' in df_jit.columns else '(unknown)'
    out = pd.DataFrame({'sigma': x, 'y': y, 'label': 'JIT', 'origin': origin, 'under_threshold': False})
    return out.replace([np.inf, -np.inf], np.nan).dropna()

def _prepare_sandwich_kind(df_sand: pd.DataFrame, kind: str, label: str, use_abs_sigma: bool) -> pd.DataFrame:
    """(σ, y, label, origin, under_threshold) for the given sandwich kind using realized token0 PnL."""
    if df_sand.empty:
        return df_sand
    if 'pattern_type' not in df_sand.columns:
        raise ValueError("Expected 'pattern_type' in sandwich tidy file.")

    mask = df_sand['pattern_type'].astype(str).str.lower() == kind
    df_k = df_sand[mask]

    required = ['sigma_gross', 'profit_per_x0', 'sigma_min_backrun']
    missing = [c for c in required if c not in df_k.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in sandwich tidy file.")

    sigma_raw = pd.to_numeric(df_k['sigma_gross'], errors='coerce')
    x = sigma_raw.abs() if use_abs_sigma else sigma_raw
    y = pd.to_numeric(df_k['profit_per_x0'], errors='coerce')
    smin = pd.to_numeric(df_k['sigma_min_backrun'], errors='coerce')
    origin = df_k['origin'].astype(str) if 'origin' in df_k.columns else '(unknown)'
    under = sigma_raw.abs() < smin

    out = pd.DataFrame({
        'sigma': x, 'y': y, 'label': label, 'origin': origin, 'under_threshold': under
    })
    return out.replace([np.inf, -np.inf], np.nan).dropna()

def prepare_sandwich_classical(df_sand: pd.DataFrame, use_abs_sigma: bool) -> pd.DataFrame:
    return _prepare_sandwich_kind(df_sand, 'classical', 'Classical sandwich', use_abs_sigma)

def prepare_sandwich_jit(df_sand: pd.DataFrame, use_abs_sigma: bool) -> pd.DataFrame:
    return _prepare_sandwich_kind(df_sand, 'jit-sandwich', 'JIT-Sandwich', use_abs_sigma)

def prepare_backrun(df_br: pd.DataFrame, use_abs_sigma: bool) -> pd.DataFrame:
    """(σ, y, label, origin, under_threshold) for back-run arbitrage observed PnL."""
    if df_br.empty:
        return df_br

    required = ['sigma_gross', 'profit_obs_per_x0', 'arb_origin', 'viable_sigma']
    missing = [c for c in required if c not in df_br.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in back-run tidy file.")

    x = _sigma(df_br['sigma_gross'], use_abs_sigma)
    y = pd.to_numeric(df_br['profit_obs_per_x0'], errors='coerce')
    origin = df_br['arb_origin'].astype(str) if 'arb_origin' in df_br.columns else '(unknown)'
    # 'viable_sigma' is True when sigma >= sigma_min. We want to flag when it's under.
    under = ~pd.to_numeric(df_br['viable_sigma'], errors='coerce').astype(bool)

    out = pd.DataFrame({
        'sigma': x, 'y': y, 'label': 'Back-run Arbitrage', 'origin': origin, 'under_threshold': under
    })
    return out.replace([np.inf, -np.inf], np.nan).dropna()

# ---------------- Diagnostics ----------------

def summarize(series: pd.Series, name: str) -> None:
    s = pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) == 0:
        print(f"{name}: no data"); return
    print(f"{name}: n={len(s):,}, median={np.median(s):.6g}, mean={np.mean(s):.6g}, p90={np.quantile(s, 0.9):.6g}")

# ---------------- Color/Marker helpers ----------------

def build_origin_color_map(origins: pd.Series) -> dict:
    """Return a {origin: hexcolor} mapping (deterministic HSV palette; no matplotlib dependency)."""
    cats = pd.Index(pd.Series(origins).dropna().astype(str).unique())
    n = len(cats)
    if n == 0:
        return {}

    # Use a stable HSV sweep for distinct colors; keep saturation/value moderate for readability.
    colors = []
    for i in range(n):
        h = (i / max(1, n)) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.65, 0.90)
        colors.append(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")

    return {cats[i]: colors[i] for i in range(n)}

PLOTLY_MARKERS = {
    'JIT': 'circle',
    'Classical sandwich': 'square',
    'JIT-Sandwich': 'triangle-up',
    'Back-run Arbitrage': 'diamond',
}

def filter_self_funded(df: pd.DataFrame) -> pd.DataFrame:
    dec0 = df.get('token0_decimals')
    if dec0 is None:
        dec0 = df.get('dec0')
    if dec0 is None:
        dec0 = pd.Series(6, index=df.index)
    dec0 = dec0.astype(int)

    dec1 = df.get('token1_decimals')
    if dec1 is None:
        dec1 = df.get('dec1')
    if dec1 is None:
        dec1 = pd.Series(18, index=df.index)
    dec1 = dec1.astype(int)

    fa0 = df['front_a0'].astype(float).abs().to_numpy()
    ba0 = df['back_a0'].astype(float).abs().to_numpy()
    fa1 = df['front_a1'].astype(float).abs().to_numpy()
    ba1 = df['back_a1'].astype(float).abs().to_numpy()

    front_dir = df['front_dir'].astype(str).to_numpy()
    is_x2y = front_dir == 'swap_x2y'

    # NOTE: Swap `amount0/amount1` are *gross* pool deltas (not divided by `r`).
    diff_token1 = (ba1 / (10.0 ** dec1)) - (fa1 / (10.0 ** dec1))
    diff_token0 = (ba0 / (10.0 ** dec0)) - (fa0 / (10.0 ** dec0))
    diff_tokens = np.where(is_x2y, diff_token1, diff_token0)

    rtol = 2e-6
    atol_token1 = 5.0 * (10.0 ** (-dec1))
    atol_token0 = 5.0 * (10.0 ** (-dec0))
    atol = np.where(is_x2y, atol_token1, atol_token0)

    is_self_funded = np.isclose(diff_tokens, 0.0, rtol=rtol, atol=atol)
    return df.loc[is_self_funded].copy()

# ---------------- Main ----------------

def main():
    repo_root = Path(__file__).resolve().parents[1]
    mev_out = repo_root / "mev_out"
    default_jit = discover_tidy_path(mev_out, 'jit_cycles_tidy')
    default_sand = discover_tidy_path(mev_out, 'sandwich_attacks_tidy')
    default_br = discover_tidy_path(mev_out, 'reverse_backruns_tidy')

    ap = argparse.ArgumentParser(
        description='Plotly scatter: profit vs |σ| for realized JIT/Sandwich/Back-run PnL — colored by origin.'
    )
    ap.add_argument('--in-jit', default=str(default_jit), help='Path to JIT tidy CSV (default: %(default)s).')
    ap.add_argument('--in-sand', default=str(default_sand), help='Path to sandwich tidy CSV (default: %(default)s).')
    ap.add_argument('--in-backrun', default=str(default_br), help='Path to reverse back-run tidy CSV (default: %(default)s).')

    ap.add_argument('--color-top-origins', type=int, default=3,
                    help='Fallback N used for both groups if group-specific values are not provided. Default: 3')
    ap.add_argument('--color-top-origins-jit', type=int, default=None,
                    help='Highlight top-N origins among JIT attacks. Overrides --color-top-origins for JIT.')
    ap.add_argument('--color-top-origins-sand', type=int, default=None,
                    help='Highlight top-N origins among Sandwich attacks (Classical + JIT-Sandwich). Overrides --color-top-origins for Sandwich.')
    ap.add_argument('--legend-max-origins', type=int, default=None,
                    help='Optional cap for number of highlighted origins shown in the legend. Default: all highlighted.')

    ap.add_argument('--out', default=str(mev_out / 'section3_profit_vs_sigma.html'), help='Output path (.html for interactive, or .png/.pdf if kaleido is installed)')
    ap.add_argument('--show', action='store_true', help='Open the plot in a browser after saving.')
    args = ap.parse_args()

    # Load
    jit_df  = load_csv(args.in_jit)
    sand_df = load_csv(args.in_sand)
    classical_sand = sand_df[sand_df['pattern_type'] == 'Classical']
    # Consider only self funded cases
    classical_sand = filter_self_funded(classical_sand)
    jit_sand = sand_df[sand_df['pattern_type'] == 'JIT-Sandwich']
    br_df = load_csv(args.in_backrun) if args.in_backrun and os.path.exists(args.in_backrun) else pd.DataFrame()

    # Build series (σ, y, label, origin)
    frames = []
    jit = prepare_jit(jit_df, True)
    if not jit.empty:
        summarize(jit['y'], 'JIT y')
        frames.append(jit)

    sand_class = prepare_sandwich_classical(classical_sand, True)
    sand_jit = prepare_sandwich_jit(jit_sand, True)
    if not sand_class.empty:
        summarize(sand_class['y'], 'Classical Sandwich y')
        frames.append(sand_class)
    if not sand_jit.empty:
        summarize(sand_jit['y'], 'JIT-Sandwich y')
        frames.append(sand_jit)

    backrun = prepare_backrun(br_df, True)
    if not backrun.empty:
        summarize(backrun['y'], 'Back-run Arbitrage y')
        frames.append(backrun)

    if len(frames) == 0:
        raise SystemExit('No data to plot. Ensure tidy CSVs exist and contain the requested patterns.')

    df_all = pd.concat(frames, ignore_index=True)

    # Separate top-N by group and take the union for coloring
    counts_jit = df_all.loc[df_all['label'] == 'JIT', 'origin'].value_counts()
    counts_sand = df_all.loc[df_all['label'] != 'JIT', 'origin'].value_counts() # Catches sandwiches AND back-runs

    n_fallback = int(args.color_top_origins) if args.color_top_origins is not None else 0
    n_jit = int(args.color_top_origins_jit) if args.color_top_origins_jit is not None else n_fallback
    n_sand = int(args.color_top_origins_sand) if args.color_top_origins_sand is not None else n_fallback

    n_jit = max(0, n_jit)
    n_sand = max(0, n_sand)

    top_jit = list(counts_jit.head(n_jit).index)
    top_sand = list(counts_sand.head(n_sand).index)

    highlighted = pd.Index(top_jit + top_sand).unique().tolist()
    color_map = build_origin_color_map(pd.Series(highlighted))
    other_color = '#d3d3d3'  # lightgray

    # Figure
    fig = go.Figure()
    width, height = 1150, 800

    # Hover template (adds origin as requested)
    hovertpl = (
        "Origin: %{customdata[0]}<br>"
        "Pattern: %{customdata[1]}<br>"
        "|σ|: %{x:.6g}<br>"
        "Profit: %{y:.6g}<extra></extra>"
    )

    # We’ll show a legend entry only once per highlighted origin.
    legend_shown_for_origin = set()
    max_leg = args.legend_max_origins if args.legend_max_origins is not None else len(highlighted)
    legend_list = highlighted[:max(0, int(max_leg))]

    # Add highlighted origins — split by label, also split under-threshold 'X' points
    for origin in highlighted:
        g_origin = df_all[df_all['origin'] == origin]
        c = color_map.get(origin, other_color)
        for label, g_lab in g_origin.groupby('label'):
            # under/over threshold
            g_under = g_lab[g_lab['under_threshold']]
            g_over  = g_lab[~g_lab['under_threshold']]
            
            # normal markers
            if not g_over.empty:
                fig.add_scatter(
                    x=g_over['sigma'],
                    y=g_over['y'],
                    mode='markers',
                    name=origin,
                    legendgroup=f"origin::{origin}",
                    showlegend=(origin in legend_list) and (origin not in legend_shown_for_origin),
                    marker=dict(symbol=PLOTLY_MARKERS.get(label, 'circle'),
                                color=c, size=6, opacity=0.65, line=dict(width=0)),
                    customdata=np.stack([g_over['origin'], g_over['label']], axis=1),
                    hovertemplate=hovertpl
                )
                legend_shown_for_origin.add(origin)

            # under-threshold X markers
            if not g_under.empty:
                fig.add_scatter(
                    x=g_under['sigma'],
                    y=g_under['y'],
                    mode='markers',
                    name=origin,
                    legendgroup=f"origin::{origin}",
                    showlegend=False,
                    marker=dict(symbol='x', color=c, size=8, opacity=0.65, line=dict(width=1)),
                    customdata=np.stack([g_under['origin'], g_under['label']], axis=1),
                    hovertemplate=hovertpl
                )

    # Aggregate all non-highlighted origins into few traces per label to keep figure light (no legend)
    others = df_all[~df_all['origin'].isin(highlighted)]
    if not others.empty:
        for label, g_lab in others.groupby('label'):
            # over/under-threshold markers
            g_under = g_lab[g_lab['under_threshold']]
            g_over  = g_lab[~g_lab['under_threshold']]

            if not g_over.empty:
                fig.add_scatter(
                    x=g_over['sigma'], y=g_over['y'],
                    mode='markers',
                    name='',
                    showlegend=False,
                    marker=dict(symbol=PLOTLY_MARKERS.get(label, 'circle'),
                                color=other_color, size=5, opacity=0.35, line=dict(width=0)),
                    customdata=np.stack([g_over['origin'], g_over['label']], axis=1),
                    hovertemplate=hovertpl
                )
            if not g_under.empty:
                fig.add_scatter(
                    x=g_under['sigma'], y=g_under['y'],
                    mode='markers',
                    name='',
                    showlegend=False,
                    marker=dict(symbol='x', color=other_color, size=7, opacity=0.35, line=dict(width=1)),
                    customdata=np.stack([g_under['origin'], g_under['label']], axis=1),
                    hovertemplate=hovertpl
                )

    # Add a small "Pattern" legend using dummy traces (to match the Matplotlib separate legend concept)
    present_labels = [lb for lb in PLOTLY_MARKERS.keys() if (df_all['label'] == lb).any()]
    for lb in present_labels:
        fig.add_scatter(
            x=[np.nan], y=[np.nan], mode='markers',
            name=f"Pattern: {lb}",
            marker=dict(symbol=PLOTLY_MARKERS[lb], size=9, color='white', line=dict(color='black', width=1.2)),
            showlegend=True,
            legendgroup="__pattern__"
        )

    # If any under-threshold points exist, add legend item
    if (df_all.get('under_threshold') == True).any():
        fig.add_trace(go.Scatter(
            x=[np.nan], y=[np.nan], mode='markers',
            name='σ < σ_min',
            marker=dict(symbol='x', size=9, color='black'),
            showlegend=True,
            legendgroup="__pattern__"
        ))

    # Layout
    fig.update_layout(
        title='Profit vs. σ (top-N per group highlighted; marker=pattern)',
        width=width, height=height,
        legend=dict(bordercolor='rgba(0,0,0,0.2)', borderwidth=1, title='Origin'),
        margin=dict(l=60, r=30, t=60, b=60)
    )
    fig.update_xaxes(
        title='|σ|  (victim size normalized by native base)',
        showgrid=True, gridcolor='rgba(0,0,0,0.25)', gridwidth=0.5, griddash='dash'
    )
    fig.update_yaxes(
        title='Profit (normalized)',
        showgrid=True, gridcolor='rgba(0,0,0,0.25)', gridwidth=0.5, griddash='dash'
    )

    # Save
    out_path = args.out
    out_dir = os.path.dirname(out_path) or '.'
    os.makedirs(out_dir, exist_ok=True)

    ext = os.path.splitext(out_path)[1].lower()
    if ext in ('.html', ''):
        if ext == '':
            out_path = out_path + '.html'
        fig.write_html(out_path, include_plotlyjs='cdn', full_html=True)
        print(f"Saved interactive HTML → {out_path}")
        if args.show:
            import webbrowser
            webbrowser.open('file://' + os.path.realpath(out_path))
    elif ext in ('.png', '.pdf', '.svg', '.jpg', '.jpeg', '.webp'):
        try:
            fig.write_image(out_path, scale=2)  # requires kaleido
            print(f"Saved static image → {out_path}")
        except Exception as e:
            fallback = os.path.splitext(out_path)[0] + '.html'
            fig.write_html(fallback, include_plotlyjs='cdn', full_html=True)
            print(f"[warn] Static export failed ({e}). Wrote interactive HTML instead → {fallback}")
            if args.show:
                import webbrowser
                webbrowser.open('file://' + os.path.realpath(fallback))
    else:
        # default to html if unknown extension
        out_html = out_path + '.html'
        fig.write_html(out_html, include_plotlyjs='cdn', full_html=True)
        print(f"[info] Unknown extension. Wrote interactive HTML → {out_html}")
        if args.show:
            import webbrowser
            webbrowser.open('file://' + os.path.realpath(out_html))


if __name__ == '__main__':
    main()
