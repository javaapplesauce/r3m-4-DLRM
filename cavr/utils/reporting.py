"""Reporting helpers for the Colab notebook: resume table + mask spotcheck."""
from __future__ import annotations

import json
import math
import os
from glob import glob
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


def _parse_run_id(run_id: str) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """Best-effort split of '<model>_<task>_seed<N>' into (model, task, seed).

    Returns (None, None, None) for run ids that don't match (e.g. ablation
    variant names like 'cavr_vitl_masked_Lift_seed0').
    """
    if "_seed" not in run_id:
        return None, None, None
    base, _, seed_str = run_id.rpartition("_seed")
    try:
        seed = int(seed_str)
    except ValueError:
        return None, None, None
    # base is "<model>_<task>" — task is the last segment.
    if "_" not in base:
        return base, None, seed
    model, _, task = base.rpartition("_")
    return model, task, seed


def print_resume_status(
    runs_dir: str | os.PathLike,
    expected: Iterable,
) -> None:
    """Print a status table for the runs *expected* to land in *runs_dir*.

    *expected* items may be:
      - a `(model, task, seed)` triple → run_id = f"{model}_{task}_seed{seed}"
      - a `(variant, seed)` 2-tuple    → run_id = f"{variant}_seed{seed}"
      - a plain `run_id` string        → used verbatim

    Cross-references against `outputs/runs/*.json`. Status comes from the
    record's `status` field (`DONE` / `CRASHED`), or is inferred as `CRASHED`
    if `success_rate` is missing/NaN.
    """
    expected_list = list(expected)
    runs_dir = Path(runs_dir)

    done_map = {}
    for path in sorted(glob(str(runs_dir / "*.json"))):
        try:
            with open(path) as f:
                d = json.load(f)
        except Exception as e:
            print(f"  [WARN] could not parse {path}: {e}")
            continue
        run_id = d.get("run_id") or Path(path).stem
        sr = d.get("success_rate")
        status = d.get("status") or ("DONE" if sr is not None and not _isnan(sr) else "CRASHED")
        done_map[run_id] = (status, sr)

    print(f"{'run_id':<40} {'status':<8} {'success_rate':>12}")
    print("-" * 64)

    done = 0
    missing: List[str] = []
    for item in expected_list:
        if isinstance(item, str):
            run_id = item
        elif len(item) == 3:
            model, task, seed = item
            run_id = f"{model}_{task}_seed{seed}"
        elif len(item) == 2:
            variant, seed = item
            run_id = f"{variant}_seed{seed}"
        else:
            raise ValueError(f"Unsupported expected entry: {item!r}")

        info = done_map.get(run_id)
        if info is None:
            print(f"{run_id:<40} {'MISSING':<8} {'-':>12}")
            missing.append(run_id)
            continue
        status, sr = info
        sr_str = f"{sr:.3f}" if isinstance(sr, (int, float)) and not _isnan(sr) else "NaN"
        print(f"{run_id:<40} {status:<8} {sr_str:>12}")
        if status == "DONE":
            done += 1

    print("-" * 64)
    print(f"Completed {done}/{len(expected_list)}.")
    if missing:
        print(f"Remaining ({len(missing)}): {', '.join(missing)}")
    else:
        print("All expected runs are present.")


def _isnan(x) -> bool:
    try:
        return math.isnan(float(x))
    except Exception:
        return False


def render_spotcheck_panel(
    demos_h5: str | os.PathLike,
    masks_h5: Optional[str | os.PathLike],
    task: str,
    n: int = 8,
    out_png: Optional[str | os.PathLike] = None,
    seed: int = 0,
):
    """Render an N×4 panel of (RGB, RGB+box, mask, RGB+overlay) for *task*.

    Reads images and (optional) precomputed masks from HDF5. The bounding
    box and overlay are derived from upsampling the cached mask back to
    image resolution — we don't re-run Grounding DINO here.

    Returns the matplotlib Figure for inline display.
    """
    import h5py
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(seed)

    with h5py.File(str(demos_h5), "r") as df:
        keys = sorted(df.keys())
        if not keys:
            raise RuntimeError(f"{demos_h5} contains no demos")

        # Build a flat (demo_key, t) sample pool, pick n at random.
        pool = []
        for k in keys:
            T = df[k]["images"].shape[0]
            for t in range(T):
                pool.append((k, t))
        if len(pool) < n:
            picks = pool
        else:
            idxs = rng.choice(len(pool), size=n, replace=False)
            picks = [pool[i] for i in idxs]

        rgbs = []
        for k, t in picks:
            img = df[k]["images"][t]  # (3, H, W) uint8
            if img.shape[0] == 3 and img.ndim == 3:
                img = img.transpose(1, 2, 0)
            rgbs.append(img.astype(np.uint8))

    masks_full = [None] * len(picks)
    if masks_h5 is not None and Path(str(masks_h5)).exists():
        with h5py.File(str(masks_h5), "r") as mf:
            for i, (k, t) in enumerate(picks):
                if k in mf:
                    m = np.asarray(mf[k]["masks"][t], dtype=np.uint8)  # (h, w)
                    masks_full[i] = m

    fig, axes = plt.subplots(len(picks), 4, figsize=(12, 3 * len(picks)))
    if len(picks) == 1:
        axes = axes.reshape(1, -1)

    for i, (rgb, mask) in enumerate(zip(rgbs, masks_full)):
        H, W = rgb.shape[:2]

        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title(f"RGB (demo={picks[i][0]}, t={picks[i][1]})" if i == 0 else "")
        axes[i, 0].axis("off")

        if mask is not None and mask.any():
            mask_up = _resize_nearest(mask, (H, W))
            ys, xs = np.where(mask_up > 0)
            x0, x1 = xs.min(), xs.max()
            y0, y1 = ys.min(), ys.max()
            axes[i, 1].imshow(rgb)
            from matplotlib.patches import Rectangle
            rect = Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=2,
                             edgecolor="lime", facecolor="none")
            axes[i, 1].add_patch(rect)
            axes[i, 1].set_title("RGB + box-from-mask" if i == 0 else "")
            axes[i, 1].axis("off")

            axes[i, 2].imshow(mask_up, cmap="gray", vmin=0, vmax=1)
            axes[i, 2].set_title("SAM mask (upsampled)" if i == 0 else "")
            axes[i, 2].axis("off")

            overlay = rgb.astype(np.float32).copy()
            tint = np.array([0.0, 255.0, 0.0])
            sel = mask_up > 0
            overlay[sel] = 0.5 * overlay[sel] + 0.5 * tint
            axes[i, 3].imshow(overlay.clip(0, 255).astype(np.uint8))
            axes[i, 3].set_title("RGB + mask overlay" if i == 0 else "")
            axes[i, 3].axis("off")
        else:
            for c in range(1, 4):
                axes[i, c].imshow(rgb)
                axes[i, c].set_title("(no mask)" if i == 0 and c == 1 else "")
                axes[i, c].axis("off")

    fig.suptitle(f"Mask spot-check: {task}", fontsize=14)
    fig.tight_layout()

    if out_png is not None:
        out_png = str(out_png)
        os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
        fig.savefig(out_png, dpi=120, bbox_inches="tight")

    return fig


def _resize_nearest(mask, target_hw):
    """Nearest-neighbour upsample of a 2-D mask. Avoids pulling in scipy."""
    import numpy as np
    th, tw = target_hw
    h, w = mask.shape
    if (h, w) == (th, tw):
        return mask
    ys = (np.arange(th) * (h / th)).astype(np.int64).clip(0, h - 1)
    xs = (np.arange(tw) * (w / tw)).astype(np.int64).clip(0, w - 1)
    return mask[ys][:, xs]


def mask_coverage_stats(masks_h5: str | os.PathLike):
    """Compute (mean_pixel_fraction, num_frames) over all demos in masks_h5."""
    import h5py
    import numpy as np

    total_pix = 0
    total_on = 0
    n_frames = 0
    with h5py.File(str(masks_h5), "r") as mf:
        for k in mf.keys():
            m = mf[k]["masks"][:]  # (T, h, w) uint8
            total_on += int((m > 0).sum())
            total_pix += int(m.size)
            n_frames += m.shape[0]
    if total_pix == 0:
        return 0.0, n_frames
    return total_on / total_pix, n_frames
