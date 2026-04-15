"""Precompute Grounding DINO + SAM2 concept masks for a demo HDF5.

Per-step masking at training time is prohibitively slow (~1.5s per image on
A100 → ~25s per batch of 16). But masks depend only on (image, task_text),
both of which are fixed across epochs. This script computes the mask for
every demo frame once and stores them in a sidecar HDF5 file that
DemoDataset auto-loads.

Output layout (matches DemoDataset expectations):
    <data_dir>/masks.hdf5
        /demo_i/masks   (T, feature_h, feature_w) uint8
        attrs: feature_h, feature_w, task_description, backbone, image_size

Usage:
    python scripts/precompute_masks.py \
        --config cavr/configs/default.yaml \
        --data-dir data/demos_lift \
        --env Lift

With cached masks present, `python scripts/train.py` automatically uses the
fast path (no live Grounding DINO / SAM2 calls during training).
"""
import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from tqdm import tqdm

from cavr.models.concept_mask import ConceptMasker
from cavr.models.encoder import DINOv2Encoder
from cavr.envs.robosuite_envs import get_task_description


def _feature_grid(image_size, backbone):
    """Infer the DINOv2 feature-map resolution for (image_size, backbone)."""
    patch = DINOv2Encoder.BACKBONES[backbone]["patch"]
    if image_size % patch != 0:
        raise ValueError(
            f"image_size {image_size} is not divisible by patch {patch} "
            f"for backbone {backbone}"
        )
    fh = fw = image_size // patch
    return fh, fw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="cavr/configs/default.yaml")
    parser.add_argument("--data-dir", default=None,
                        help="Directory containing demos.hdf5. Defaults to cfg.data.save_dir.")
    parser.add_argument("--env", default=None,
                        help="Robosuite env name — controls the task description used "
                             "for Grounding DINO. Defaults to cfg.env.name.")
    parser.add_argument("--task-description", default=None,
                        help="Override the text query. Takes precedence over --env.")
    parser.add_argument("--output", default=None,
                        help="Output mask HDF5 path. Defaults to <data-dir>/masks.hdf5.")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Images processed per ConceptMasker call. Currently "
                             "the masker iterates per-image internally, so this "
                             "mostly controls tqdm granularity.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing mask file instead of appending "
                             "only missing demos.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(args.data_dir or cfg["data"]["save_dir"])
    demo_path = data_dir / "demos.hdf5"
    if not demo_path.exists():
        raise FileNotFoundError(f"No demo file at {demo_path}")

    env_name = args.env or cfg["env"]["name"]
    task_desc = args.task_description or get_task_description(env_name)
    backbone = cfg["encoder"]["backbone"]
    image_size = cfg["env"]["camera_height"]
    fh, fw = _feature_grid(image_size, backbone)

    out_path = Path(args.output) if args.output else data_dir / "masks.hdf5"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    masker = ConceptMasker(threshold=cfg["masking"]["threshold"], device=str(device))
    masker._lazy_init(device=device)
    if masker._grounding_model is None or masker._sam_predictor is None:
        raise RuntimeError(
            "Grounding DINO or SAM2 not available. Install transformers and "
            "sam2, and download sam2_hiera_large.pt, before precomputing masks. "
            "(Without these, there's nothing to cache — the runtime falls back "
            "to all-ones masks which you can emulate with --no-masking on train.)"
        )

    print(f"[precompute] demos       = {demo_path}")
    print(f"[precompute] output      = {out_path}")
    print(f"[precompute] env         = {env_name}")
    print(f"[precompute] task text   = {task_desc!r}")
    print(f"[precompute] backbone    = {backbone}")
    print(f"[precompute] image size  = {image_size}  → feature grid {fh}x{fw}")
    print(f"[precompute] device      = {device}")

    mode = "w" if args.force or not out_path.exists() else "a"
    existing_keys = set()
    if mode == "a":
        with h5py.File(out_path, "r") as mf:
            existing_keys = set(mf.keys())
            prev_text = mf.attrs.get("task_description", None)
            prev_fh = int(mf.attrs.get("feature_h", 0))
            prev_fw = int(mf.attrs.get("feature_w", 0))
        if (prev_text != task_desc) or (prev_fh, prev_fw) != (fh, fw):
            raise RuntimeError(
                f"Existing {out_path} was built with task={prev_text!r}, "
                f"grid={prev_fh}x{prev_fw} — incompatible with the current "
                f"request (task={task_desc!r}, grid={fh}x{fw}). Re-run with "
                f"--force to overwrite, or delete the file."
            )

    with h5py.File(demo_path, "r") as demo_f, h5py.File(out_path, mode) as mask_f:
        mask_f.attrs["feature_h"] = fh
        mask_f.attrs["feature_w"] = fw
        mask_f.attrs["task_description"] = task_desc
        mask_f.attrs["backbone"] = backbone
        mask_f.attrs["image_size"] = image_size

        demo_keys = sorted(demo_f.keys())
        total_frames = sum(demo_f[k]["images"].shape[0] for k in demo_keys
                           if k not in existing_keys)
        pbar = tqdm(total=total_frames, desc="precomputing masks")

        for key in demo_keys:
            if key in existing_keys:
                continue
            images = demo_f[key]["images"][:]  # (T, 3, H, W) uint8
            T = images.shape[0]
            out = np.zeros((T, fh, fw), dtype=np.uint8)

            for t in range(T):
                img_t = torch.from_numpy(images[t][None]).to(device).float()
                mask = masker(img_t, task_desc, fh, fw)
                out[t] = (mask[0, ..., 0].cpu().numpy() >
                          cfg["masking"]["threshold"]).astype(np.uint8)
                pbar.update(1)

            g = mask_f.create_group(key)
            g.create_dataset("masks", data=out, compression="gzip", compression_opts=4)
            mask_f.flush()

        pbar.close()

    print(f"[precompute] wrote {out_path}")


if __name__ == "__main__":
    sys.exit(main())
