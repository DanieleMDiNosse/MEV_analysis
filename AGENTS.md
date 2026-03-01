# Repository Guidelines

## Project Structure & Module Organization
- `scripts/`: runnable ETL/analysis scripts (data fetch, MEV detection, plotting).
- `notebooks/`: exploratory work (e.g., `notebooks/mev_profit_analysis.ipynb`).
- `data/`: local datasets + checkpoints (git-ignored; see `data/README.md`).
  - `data/raw/`, `data/interim/`, `data/processed/`, `data/checkpoints/`
- `mev_out/`: generated results (CSVs/plots; git-ignored). Curated snapshots for the report live in `docs/assets/`.
- `docs/`: GitHub Pages report (`docs/index.md`) + asset builder (`docs/scripts/build_report_assets.py`).
- `paper/`: LaTeX sources; build artifacts are ignored.

## Build, Test, and Development Commands
This repo is script-first (not a packaged library). Prefer the conda env `main`:

```bash
# quick syntax/smoke check
conda run -n main python -m py_compile scripts/*.py docs/scripts/build_report_assets.py

# regenerate small Pages-friendly tables under docs/assets/
conda run -n main python docs/scripts/build_report_assets.py
```

Typical workflow:

```bash
conda run -n main python scripts/data_fetch.py
conda run -n main python scripts/mev_collect.py --in data/raw/univ3_<POOL>.csv --outdir mev_out --fee_bps 5
conda run -n main python scripts/section3_empirical_simple.py --in-jit mev_out/jit_cycles_tidy_5.csv --in-sand mev_out/sandwich_attacks_tidy_5.csv
```

## Coding Style & Naming Conventions
- Python: 4-space indentation, PEP8-ish, optimize for clarity/reproducibility.
- Add type hints for new/modified functions; add docstrings for “public” helpers (Summary/Parameters/Returns/Notes).
- Use `pathlib.Path` + repo-relative paths (avoid `/home/...` hard-coding).
- Avoid adding new dependencies unless necessary; document why in the PR.

## Testing Guidelines
- No formal test suite yet; `py_compile` and `--help` runs are the minimum checks.
- If adding non-trivial logic, add `pytest` tests under `tests/` with small fixture data (never commit multi-GB datasets).

## Commit & Pull Request Guidelines
- Git history is minimal (“Initial commit”, “Add files via upload”), so no strict convention exists.
- Recommended commit format: `scope: imperative summary` (e.g., `scripts: make outputs repo-relative`).
- PRs should include: goal + scientific intent, exact commands to reproduce, any new dependencies, and confirmation that `data/` and `mev_out/` are not committed.

## Security & Configuration Tips
- Do not commit RPC keys/secrets. Prefer environment variables (e.g., `MEV_RPC_URLS` for space-separated RPC endpoints) or local, untracked config.
