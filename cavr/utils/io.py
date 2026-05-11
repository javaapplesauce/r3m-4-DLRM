"""Atomic IO helpers shared by training, evaluation, and the Colab notebook.

These are deliberately tiny and dependency-light so they're safe to import
before the heavy ML stack (torch, robosuite, sam2) is initialised.
"""
from __future__ import annotations

import csv
import json
import os
from glob import glob
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def atomic_json_write(path: str | os.PathLike, data: Any) -> None:
    """Write *data* as JSON to *path*, surviving mid-write crashes.

    Writes to a sibling .tmp file, fsyncs, then renames over the target.
    A reader will only ever see the previous version or the new version,
    never a half-written file.
    """
    path = str(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_csv_row(
    path: str | os.PathLike,
    row: Dict[str, Any],
    header: Optional[Iterable[str]] = None,
) -> None:
    """Append a single row to a CSV, creating the file (with header) if needed.

    If the file exists, its existing header is used and any row keys not in
    that header are silently dropped. Schema drift won't corrupt earlier rows.
    """
    path = str(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    if file_exists:
        with open(path, "r", newline="") as f:
            reader = csv.reader(f)
            existing_header = next(reader, None)
        cols = existing_header if existing_header else list(row.keys())
    else:
        cols = list(header) if header is not None else list(row.keys())

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def load_run_jsons(glob_pattern: str | os.PathLike):
    """Read every run JSON matching *glob_pattern* into a pandas DataFrame.

    Robust to partially-written or corrupt files: failures land in a
    `__error` column instead of raising.
    """
    import pandas as pd

    rows = []
    for p in sorted(glob(str(glob_pattern))):
        try:
            with open(p) as f:
                d = json.load(f)
            d["__path"] = p
            rows.append(d)
        except Exception as e:
            rows.append({"__path": p, "__error": repr(e)})
    return pd.DataFrame(rows)


def safe_read_json(path: str | os.PathLike) -> Optional[Dict[str, Any]]:
    """Return parsed JSON or None if missing/corrupt."""
    path = str(path)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def git_short_hash(cwd: Optional[str] = None) -> Optional[str]:
    """Best-effort `git rev-parse --short HEAD`. Returns None on failure."""
    import subprocess
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd, stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return None


def gpu_name() -> Optional[str]:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return None
