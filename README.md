# MEV_analysis
Maximal Extractable Value theoretical and empirical analysis (Uniswap v3 focus).

Technical report (GitHub Pages entrypoint):
- `docs/index.md`

Main scripts:
- `scripts/data_fetch.py` (fetch pool events + running state)
- `scripts/mev_collect.py` (detect MEV patterns + compute theory fields)
- `scripts/section3_empirical_simple.py` / `scripts/section3_empirical.py` (plots)

## Data fetching architecture

`scripts/data_fetch.py` uses a **subgraph-first** strategy:

1. The Uniswap v3 subgraph provides the canonical event stream (Swap/Mint/Burn),
   including timestamps, amounts, ticks, sqrtPriceX96, and `origin`.
2. RPC calls (`eth_getLogs`) are used **only** for `Swap.liquidity` (active
   liquidity after each swap), which the subgraph does not reliably expose.
3. Gas fields (`gasUsed`, `gasPrice`, `effectiveGasPrice`) are left empty by
   default; enrich later with `scripts/add_gas.py`.

Reliability features:
- **Quarantine-aware RPC** (`scripts/quarantined_rpc.py`): endpoints that return
  HTTP 429 / `Retry-After` are quarantined and skipped until the cooldown expires.
  Rate-limit errors are distinguished from range-too-large errors so that
  range-splitting is only applied when useful.
- **Checkpoint-based resume**: progress is saved atomically to a JSON checkpoint
  after each flush; re-running the same command resumes from where it left off.
- **Progress output**: all stages (subgraph init, RPC fetches, streaming, flushing)
  print progress with `flush=True` so output is visible even in buffered environments.

Configuration (no secrets in repo):
- `UNIV3_GRAPH_URL` or `--graph-url`: subgraph GraphQL endpoint
- `MEV_RPC_URLS` or `--rpc-urls`: space-separated RPC endpoints (for failover)
- `WEB3_PROVIDER_URI`: single RPC endpoint fallback

## Default pipeline (fee tier = 5 bps)

```bash
conda run -n main python scripts/data_fetch.py
conda run -n main python scripts/mev_collect.py
conda run -n main python scripts/section3_empirical_simple.py
conda run -n main python scripts/section3_empirical.py
conda run -n main python docs/scripts/build_report_assets.py
```

- `mev_collect.py` auto-picks input data from `data/processed/` (fallback: `data/raw/`)
- writes `mev_out/jit_cycles_tidy_5.csv`, `mev_out/sandwich_attacks_tidy_5.csv`, `mev_out/reverse_backruns_tidy_5.csv`

## Repository structure (research-friendly defaults)

- `data/`: local datasets + checkpoints (ignored by git; see `data/README.md`)
- `mev_out/`: generated results (ignored by git; curated snapshots under `docs/assets/`)
- `notebooks/`: Jupyter notebooks
- `scripts/`: runnable analysis/ETL scripts
