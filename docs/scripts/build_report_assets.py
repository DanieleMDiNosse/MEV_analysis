#!/usr/bin/env python3
"""
Build small, GitHub-Pages-friendly report assets from `mev_out/`.

This script is intentionally dependency-light (standard library only) so the
documentation can be regenerated without requiring a full scientific stack.

It scans the "tidy" outputs produced by `scripts/mev_collect.py`:
- `mev_out/jit_cycles_tidy_5.csv` (default), with fallbacks to legacy names
- `mev_out/sandwich_attacks_tidy_5.csv` (default), with fallbacks to legacy names
- `mev_out/reverse_backruns_tidy_5.csv` (default), with fallbacks to legacy names

Outputs (created/overwritten):
- `docs/assets/tables/summary_metrics.csv`
- `docs/assets/tables/top_origins.csv`
- `docs/assets/tables/summary_metrics.md` (a small Markdown snippet for embedding)

Notes
-----
- Missing/blank numeric fields are ignored in summary statistics.
"""

from __future__ import annotations

import csv
import math
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _to_float(x: Any) -> Optional[float]:
    """Parse a float from CSV field, returning None for blanks/unparseable values."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
        return float(x)
    s = str(x).strip()
    if not s or s.lower() in {"none", "nan"}:
        return None
    try:
        v = float(s)
    except Exception:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _to_bool(x: Any) -> Optional[bool]:
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in {"true", "1", "yes"}:
        return True
    if s in {"false", "0", "no"}:
        return False
    return None


def _quantile(sorted_vals: Sequence[float], q: float) -> float:
    """Compute a quantile using linear interpolation on a sorted sequence."""
    if not sorted_vals:
        return float("nan")
    if q <= 0:
        return float(sorted_vals[0])
    if q >= 1:
        return float(sorted_vals[-1])

    n = len(sorted_vals)
    pos = (n - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_vals[lo])
    w = pos - lo
    return float(sorted_vals[lo] * (1.0 - w) + sorted_vals[hi] * w)


def _summarize(values: List[float]) -> Dict[str, float]:
    """
    Summarize a numeric series.

    Returns
    -------
    dict
        Keys: n, mean, p10, p50, p90, p99 (all floats; n is stored as float for CSV uniformity).
    """
    if not values:
        return {"n": 0.0, "mean": float("nan"), "p10": float("nan"), "p50": float("nan"), "p90": float("nan"), "p99": float("nan")}

    vs = sorted(values)
    return {
        "n": float(len(vs)),
        "mean": float(mean(vs)),
        "p10": _quantile(vs, 0.10),
        "p50": _quantile(vs, 0.50),
        "p90": _quantile(vs, 0.90),
        "p99": _quantile(vs, 0.99),
    }


def _read_csv(path: Path) -> Iterable[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def _pick_tidy_path(mev_out: Path, stem: str) -> Path:
    """Prefer current `mev_collect.py` outputs, but keep backward-compatible fallbacks."""
    candidates = [
        mev_out / f"{stem}_5.csv",
        mev_out / f"{stem}_5.0.csv",
        mev_out / f"{stem}.csv",
        mev_out / f"{stem}_None.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def summarize_file(
    name: str,
    path: Path,
    *,
    origin_col: str,
    pattern_col: Optional[str],
    sigma_col: str,
    profit_col: str,
    extra_bool_cols: Sequence[str] = (),
) -> Tuple[Dict[str, Any], Counter[str], Counter[str]]:
    """
    Compute summary metrics and origin counts for a tidy MEV CSV.

    Parameters
    ----------
    name:
        Short label used in outputs.
    path:
        CSV path.
    origin_col:
        Column name for attacker/origin address.
    pattern_col:
        Optional column name for strategy label (e.g., 'pattern_type').
    sigma_col:
        Column name for σ (gross or net).
    profit_col:
        Column name for normalized profit.
    extra_bool_cols:
        Optional boolean columns to summarize as mean rates (True share).

    Returns
    -------
    metrics, origin_counts, pattern_counts
    """
    n_rows = 0
    origins: Counter[str] = Counter()
    patterns: Counter[str] = Counter()
    sigmas: List[float] = []
    profits: List[float] = []
    bool_true_counts: Dict[str, int] = {c: 0 for c in extra_bool_cols}
    bool_seen_counts: Dict[str, int] = {c: 0 for c in extra_bool_cols}

    for row in _read_csv(path):
        n_rows += 1

        origin = (row.get(origin_col) or "").strip()
        if origin:
            origins[origin] += 1

        if pattern_col:
            pat = (row.get(pattern_col) or "").strip()
            if pat:
                patterns[pat] += 1

        s = _to_float(row.get(sigma_col))
        if s is not None:
            sigmas.append(s)

        p = _to_float(row.get(profit_col))
        if p is not None:
            profits.append(p)

        for c in extra_bool_cols:
            b = _to_bool(row.get(c))
            if b is None:
                continue
            bool_seen_counts[c] += 1
            if b:
                bool_true_counts[c] += 1

    sigma_stats = _summarize(sigmas)
    profit_stats = _summarize(profits)

    out: Dict[str, Any] = {
        "dataset": name,
        "path": str(path),
        "n_rows": n_rows,
        "n_unique_origins": len(origins),
        "n_patterns": len(patterns),
        "sigma_col": sigma_col,
        "profit_col": profit_col,
        "sigma_n": int(sigma_stats["n"]),
        "sigma_mean": sigma_stats["mean"],
        "sigma_p50": sigma_stats["p50"],
        "sigma_p90": sigma_stats["p90"],
        "sigma_p99": sigma_stats["p99"],
        "profit_n": int(profit_stats["n"]),
        "profit_mean": profit_stats["mean"],
        "profit_p50": profit_stats["p50"],
        "profit_p90": profit_stats["p90"],
        "profit_p99": profit_stats["p99"],
    }

    for c in extra_bool_cols:
        seen = bool_seen_counts[c]
        out[f"{c}_n"] = seen
        out[f"{c}_true_share"] = (bool_true_counts[c] / seen) if seen else float("nan")

    return out, origins, patterns


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _write_markdown_summary(path: Path, metrics_rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def fmt(x: Any) -> str:
        if x is None:
            return ""
        if isinstance(x, float):
            if math.isnan(x) or math.isinf(x):
                return ""
            return f"{x:.6g}"
        return str(x)

    lines = []
    lines.append("| Dataset | Rows | Unique origins | σ p50 | σ p90 | Profit p50 | Profit p90 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in metrics_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    fmt(r.get("dataset")),
                    fmt(r.get("n_rows")),
                    fmt(r.get("n_unique_origins")),
                    fmt(r.get("sigma_p50")),
                    fmt(r.get("sigma_p90")),
                    fmt(r.get("profit_p50")),
                    fmt(r.get("profit_p90")),
                ]
            )
            + " |"
        )
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    mev_out = repo_root / "mev_out"
    docs_tables = repo_root / "docs" / "assets" / "tables"

    inputs = [
        {
            "name": "JIT cycles",
            "path": _pick_tidy_path(mev_out, "jit_cycles_tidy"),
            "origin_col": "origin",
            "pattern_col": "pattern_type",
            "sigma_col": "sigma_gross",
            "profit_col": "profit_total_per_x0",
            "extra_bool_cols": (),
        },
        {
            "name": "Sandwich attacks",
            "path": _pick_tidy_path(mev_out, "sandwich_attacks_tidy"),
            "origin_col": "origin",
            "pattern_col": "pattern_type",
            "sigma_col": "sigma_gross",
            "profit_col": "profit_per_x0",
            "extra_bool_cols": (),
        },
        {
            "name": "Reverse back-runs",
            "path": _pick_tidy_path(mev_out, "reverse_backruns_tidy"),
            "origin_col": "arb_origin",
            "pattern_col": "reverse_label",
            "sigma_col": "sigma_gross",
            "profit_col": "profit_obs_per_x0",
            "extra_bool_cols": ("viable_sigma", "reverted", "near_target"),
        },
    ]

    metrics_rows: List[Dict[str, Any]] = []
    top_origin_rows: List[Dict[str, Any]] = []

    for cfg in inputs:
        path: Path = cfg["path"]
        if not path.exists():
            print(f"[skip] Missing: {path}")
            continue

        metrics, origins, patterns = summarize_file(
            cfg["name"],
            path,
            origin_col=cfg["origin_col"],
            pattern_col=cfg["pattern_col"],
            sigma_col=cfg["sigma_col"],
            profit_col=cfg["profit_col"],
            extra_bool_cols=cfg["extra_bool_cols"],
        )
        # Keep tables portable: store a repo-relative path in outputs.
        try:
            metrics["path"] = str(path.relative_to(repo_root))
        except Exception:
            metrics["path"] = str(path)
        metrics_rows.append(metrics)

        for pat, cnt in patterns.most_common():
            top_origin_rows.append(
                {
                    "dataset": cfg["name"],
                    "kind": "pattern_count",
                    "key": pat,
                    "count": cnt,
                }
            )

        for origin, cnt in origins.most_common(25):
            top_origin_rows.append(
                {
                    "dataset": cfg["name"],
                    "kind": "top_origin",
                    "key": origin,
                    "count": cnt,
                }
            )

    if metrics_rows:
        # Keep a stable column order for diffs.
        fieldnames = [
            "dataset",
            "path",
            "n_rows",
            "n_unique_origins",
            "n_patterns",
            "sigma_col",
            "profit_col",
            "sigma_n",
            "sigma_mean",
            "sigma_p50",
            "sigma_p90",
            "sigma_p99",
            "profit_n",
            "profit_mean",
            "profit_p50",
            "profit_p90",
            "profit_p99",
            "viable_sigma_n",
            "viable_sigma_true_share",
            "reverted_n",
            "reverted_true_share",
            "near_target_n",
            "near_target_true_share",
        ]
        _write_csv(docs_tables / "summary_metrics.csv", metrics_rows, fieldnames=fieldnames)
        _write_markdown_summary(docs_tables / "summary_metrics.md", metrics_rows)

    if top_origin_rows:
        _write_csv(
            docs_tables / "top_origins.csv",
            top_origin_rows,
            fieldnames=["dataset", "kind", "key", "count"],
        )


if __name__ == "__main__":
    main()
