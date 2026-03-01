"""
univ3_checkpoint.py — Minimal, resume-safe JSON checkpoints (atomic writes).

This module provides a tiny helper layer around JSON checkpoints used by
long-running harvesters. It is intentionally dependency-free (stdlib only).
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, Optional


def load_checkpoint(path: str) -> Optional[Dict[str, Any]]:
    """
    Load a JSON checkpoint from disk.

    Parameters
    ----------
    path:
        Path to a JSON file.

    Returns
    -------
    dict | None
        Parsed checkpoint dict, or None if the file does not exist.

    Notes
    -----
    - This function does not validate the schema; callers should validate
      version/required keys.
    - If the file exists but is invalid JSON, this raises `json.JSONDecodeError`.

    Examples
    --------
    >>> ckpt = load_checkpoint("nonexistent.json")
    >>> ckpt is None
    True
    """
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_checkpoint_atomic(path: str, payload: Dict[str, Any]) -> None:
    """
    Atomically save a checkpoint JSON file.

    Parameters
    ----------
    path:
        Target JSON path.
    payload:
        JSON-serializable dictionary to write.

    Returns
    -------
    None

    Notes
    -----
    - Writes to a temp file in the same directory and then `os.replace()`s it,
      making the operation atomic on POSIX filesystems.
    - Uses compact JSON and sorts keys for stable diffs/debugging.

    Examples
    --------
    >>> save_checkpoint_atomic("/tmp/example_ckpt.json", {"a": 1})
    """
    target_dir = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(target_dir, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(prefix=".ckpt_", dir=target_dir, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, separators=(",", ":"), sort_keys=True)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise
