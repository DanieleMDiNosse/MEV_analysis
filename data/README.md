# Data directory (not versioned)

This repository follows a research-friendly layout where **code is tracked** and
**large datasets/checkpoints are not**.

By default, everything under `data/` is ignored by git (see `.gitignore`), except
for this README and optional `.gitkeep` placeholder files.

Suggested subfolders:

- `data/raw/`: immutable/raw exports (e.g., pool event CSVs).
- `data/interim/`: intermediate artifacts produced during cleaning/enrichment.
- `data/processed/`: analysis-ready datasets used by scripts/notebooks.
- `data/checkpoints/`: resume checkpoints and caches (safe to delete/recompute).

If you need to share datasets, prefer one of:
- a small `data/` sample file committed under a dedicated `data_sample/` folder
- Git LFS / DVC / an object store (S3/GCS) with documented download steps

