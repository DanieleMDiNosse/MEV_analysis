"""
JIT + Sandwich verification (both directions)

We test the identity:
    q_BR_hat = -(q_FR - alpha * v)

All amounts are measured in the victim’s *output* token.
Mapping by direction:
  - If victim_dir = x->y (token0 -> token1):
        q_FR     = - front_a1
        q_BR_obs = - back_a1
        v        = - S_net_token1
  - If victim_dir = y->x (token1 -> token0):
        q_FR     = - front_a0
        q_BR_obs = - back_a0
        v        = - S_net_token0

Alpha:
  alpha = attacker_liq_share
  (fallback if NaN/missing): alpha ≈ mint_amount / (L0 + mint_amount)

Outputs:
  - Overall fit stats + per-direction stats
  - Scatter: observed vs predicted BR
  - Scatter: (alpha * v) vs (|FR| - |BR|)
  - CSV with per-cycle results
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- 1) Load data -------------------------------------------------------------
# Change this path if needed
PATH = Path("/home/danielemdn/Documents/repositories/ABM_Uni_v3/mev_out/jit_sandwich_tidy.csv")
df = pd.read_csv(PATH)

# --- 2) Normalize the direction field ----------------------------------------
# We standardize victim_dir to either "x_to_y" or "y_to_x".
DIR_MAP = {
    "x->y": "x_to_y", "x_to_y": "x_to_y", "x2y": "x_to_y", "0->1": "x_to_y", "0to1": "x_to_y", "0-1": "x_to_y", "01": "x_to_y",
    "y->x": "y_to_x", "y_to_x": "y_to_x", "y2x": "y_to_x", "1->0": "y_to_x", "1to0": "y_to_x", "1-0": "y_to_x", "10": "y_to_x",
}
vdir_raw = df["victim_dir"].astype(str).str.strip().str.lower()
vdir = vdir_raw.map(DIR_MAP)
is_xy = (vdir == "x_to_y")   # victim’s output token is token1
is_yx = (vdir == "y_to_x")   # victim’s output token is token0

# --- 3) alpha (JIT share during the victim) --------------------------
alpha = pd.to_numeric(df["attacker_liq_share"], errors="coerce")

# --- 4) Build q_FR, q_BR_obs, v in the victim’s *output* token ---------------
# Pool-delta -> trader perspective: multiply by -1 (buy output token > 0)
f = 0.0005
r = 1 - f
front_a1 = pd.to_numeric(df.get("front_a1", np.nan), errors="coerce")
front_a0 = pd.to_numeric(df.get("front_a0", np.nan), errors="coerce")
back_a1  = pd.to_numeric(df.get("back_a1",  np.nan), errors="coerce")
back_a0  = pd.to_numeric(df.get("back_a0",  np.nan), errors="coerce")
S1       = pd.to_numeric(df.get("S_net_token1", np.nan), errors="coerce")
S0       = pd.to_numeric(df.get("S_net_token0", np.nan), errors="coerce")

# Use vectorized selection per direction
q_FR     = -np.where(is_xy, front_a1, front_a0)
q_BR_obs = - np.where(is_xy, back_a1/r,  back_a0/r)
v        = -np.where(is_xy, S1,       S0)

# --- 5) Predicted back-run and tidy results ----------------------------------
q_BR_hat = -(q_FR - alpha * v)
res = pd.DataFrame({
    "victim_dir": vdir,
    "q_FR": q_FR,
    "v": v,
    "alpha": alpha,
    "q_BR_obs": q_BR_obs,
    "q_BR_hat": q_BR_hat
}).replace([np.inf, -np.inf], np.nan).dropna()

# --- 6) Metrics ---------------------------------------------------------------
def fit_stats(df_in: pd.DataFrame) -> dict:
    """Return basic fit metrics for q_BR_obs vs q_BR_hat."""
    if df_in.empty:
        return {"n": 0}
    y = df_in["q_BR_obs"].to_numpy()
    yhat = df_in["q_BR_hat"].to_numpy()

    # Correlation
    corr = np.corrcoef(y, yhat)[0, 1] if np.std(yhat) > 0 else np.nan

    # RMSE
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))

    # MAPE (guard against division by ~0)
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = np.median(np.abs((y - yhat) / yhat))
    mape = float(mape) if np.isfinite(mape) else np.nan

    # % of cycles with |BR| < |FR|
    share_smaller = float(np.mean(np.abs(df_in["q_BR_obs"]) < np.abs(df_in["q_FR"])))

    # Relationship alpha*v vs (|FR| - |BR|)
    av  = (df_in["alpha"] * df_in["v"]).to_numpy()
    gap = (np.abs(df_in["q_FR"]) - np.abs(df_in["q_BR_obs"])).to_numpy()
    av_gap_corr = np.corrcoef(av, gap)[0, 1] if (np.std(av) > 0 and np.std(gap) > 0) else np.nan

    # OLS: y ≈ a + b yhat  (we want a≈0, b≈1)
    X = np.column_stack([np.ones(len(yhat)), yhat])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    intercept, slope = map(float, beta)

    return {
        "n": int(len(df_in)),
        "corr_obs_hat": float(corr) if np.isfinite(corr) else np.nan,
        "rmse": rmse,
        "median_abs_pct_error": mape,
        "share_|BR|<|FR|": share_smaller,
        "corr(alpha*v, |FR|-|BR|)": float(av_gap_corr) if np.isfinite(av_gap_corr) else np.nan,
        "ols_intercept": intercept,
        "ols_slope": slope,
    }

overall = fit_stats(res)
by_dir = res.groupby("victim_dir", dropna=False).apply(fit_stats)

print("=== Overall ===")
for k, v_ in overall.items():
    print(f"{k}: {v_}")

print("\n=== By victim direction ===")
for d, stats in by_dir.items():
    print(f"[{d}]")
    for k, v_ in stats.items():
        print(f"  {k}: {v_}")

# --- 7) Plots ----------------------------------------------------------------
if not res.empty:
    # Observed vs Predicted BR (with 45-degree line)
    plt.figure(figsize=(7, 6))
    plt.scatter(res["q_BR_hat"], res["q_BR_obs"], alpha=0.6)
    lo = float(np.nanmin([res["q_BR_hat"].min(), res["q_BR_obs"].min()]))
    hi = float(np.nanmax([res["q_BR_hat"].max(), res["q_BR_obs"].max()]))
    if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
        plt.plot([lo, hi], [lo, hi])
    plt.xlabel("Predicted back-run (q_BR_hat)")
    plt.ylabel("Observed back-run (q_BR_obs)")
    plt.title("Observed vs Predicted Back-run — both directions\nq_BR_hat = -(q_FR - alpha * v)")
    plt.show()

    # alpha*v vs (|FR| - |BR|) — larger alpha*v ⇒ smaller BR needed
    plt.figure(figsize=(7, 6))
    plt.scatter(res["alpha"] * res["v"], np.abs(res["q_FR"]) - np.abs(res["q_BR_obs"]), alpha=0.6)
    plt.xlabel("alpha × v")
    plt.ylabel("|q_FR| - |q_BR_obs|")
    plt.title("alpha × v vs |FR| - |BR|  (higher alpha×v ⇒ smaller BR)")
    plt.show()

