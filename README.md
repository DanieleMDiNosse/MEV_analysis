# MEV_analysis
Maximal Extractable Value theoretical and empirical analysis (Uniswap v3 focus).

Technical report (GitHub Pages entrypoint):
- `docs/index.md`

Main scripts:
- `scripts/data_fetch.py` (fetch pool events + running state)
- `scripts/mev_collect.py` (detect MEV patterns + compute theory fields)
- `scripts/section3_empirical_simple.py` / `scripts/section3_empirical.py` (plots)

Default pipeline (fee tier = 5 bps):
- `conda run -n main python scripts/mev_collect.py`
  - auto-picks input data from `data/processed/` (fallback: `data/raw/`)
  - writes `mev_out/jit_cycles_tidy_5.csv`, `mev_out/sandwich_attacks_tidy_5.csv`, `mev_out/reverse_backruns_tidy_5.csv`
- `conda run -n main python scripts/section3_empirical_simple.py`
- `conda run -n main python scripts/section3_empirical.py`
- `conda run -n main python docs/scripts/build_report_assets.py`

Repository structure (research-friendly defaults):
- `data/`: local datasets + checkpoints (ignored by git; see `data/README.md`)
- `mev_out/`: generated results (ignored by git; curated snapshots under `docs/assets/`)
- `notebooks/`: Jupyter notebooks
- `scripts/`: runnable analysis/ETL scripts
