"""End-to-end (train + eval + record) orchestrator used by the Colab notebook.

`train_and_eval(model, task, seed, ...)` is the single entry point. It:
  1. Skips the run if a complete JSON record already exists on disk.
  2. Builds a per-run config from `cavr/configs/default.yaml` plus overrides.
  3. Sets seeds, constructs model / dataset / trainer / evaluator.
  4. Atomically writes `<runs_dir>/<run_id>.json` and appends one CSV row.
  5. Returns the result dict (also useful for in-notebook reporting).

The function intentionally never raises on training/eval errors — it writes
a `status=CRASHED` record and returns it. Callers decide whether to retry.
"""
from __future__ import annotations

import copy
import math
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from cavr.utils.io import (
    append_csv_row,
    atomic_json_write,
    git_short_hash,
    gpu_name,
    safe_read_json,
)


CSV_HEADER = [
    "run_id", "model", "task", "seed", "variant",
    "success_rate", "mean_return", "std_return", "mean_length",
    "best_val_loss", "num_eval_episodes",
    "elapsed_s", "train_s",
    "git_hash", "gpu", "timestamp",
    "status",
]


def _has_valid_success_rate(d: Dict[str, Any]) -> bool:
    sr = d.get("success_rate")
    if sr is None:
        return False
    try:
        return not math.isnan(float(sr))
    except Exception:
        return False


def _build_cfg(
    config_path: str,
    task: str,
    seed: int,
    *,
    data_root: str,
    ckpt_root: str,
    run_id: str,
    epochs: Optional[int],
    eval_freq: Optional[int],
    eval_episodes: Optional[int],
    batch_size: Optional[int],
    masking_override: Optional[bool],
    backbone_override: Optional[str],
) -> Dict[str, Any]:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)

    cfg["env"]["name"] = task
    cfg["data"]["save_dir"] = str(Path(data_root) / f"demos_{task}")

    cfg["training"]["seed"] = int(seed)
    cfg["training"]["checkpoint_dir"] = str(Path(ckpt_root) / run_id)
    if epochs is not None:
        cfg["training"]["num_epochs"] = int(epochs)
    if eval_freq is not None:
        cfg["training"]["eval_freq"] = int(eval_freq)
    if batch_size is not None:
        cfg["training"]["batch_size"] = int(batch_size)

    if eval_episodes is not None:
        cfg["evaluation"]["num_episodes"] = int(eval_episodes)

    if masking_override is not None:
        cfg["masking"]["enabled"] = bool(masking_override)
    if backbone_override is not None:
        cfg["encoder"]["backbone"] = str(backbone_override)

    return cfg


def _build_model(model_name: str, cfg: Dict[str, Any], device: str):
    if model_name in ("cavr", "cavr_vitl_masked", "cavr_vitl_no_mask",
                      "cavr_vitb_masked", "cavr_vitb_no_mask"):
        from cavr.models.pipeline import CAVR
        return CAVR(cfg)
    if model_name == "r3m":
        from cavr.models.baselines import R3MBaseline
        return R3MBaseline(cfg["policy"], device=device)
    if model_name == "vc1":
        from cavr.models.baselines import VC1Baseline
        return VC1Baseline(cfg["policy"], device=device)
    raise ValueError(f"Unknown model: {model_name}")


def train_and_eval(
    model: str,
    task: str,
    seed: int,
    *,
    runs_dir: str = "outputs/runs",
    csv_path: Optional[str] = "outputs/baseline_results.csv",
    data_root: str = "data",
    ckpt_root: str = "checkpoints",
    config_path: str = "cavr/configs/default.yaml",
    epochs: Optional[int] = None,
    eval_freq: Optional[int] = None,
    eval_episodes: Optional[int] = None,
    batch_size: Optional[int] = None,
    masking_override: Optional[bool] = None,
    backbone_override: Optional[str] = None,
    variant: Optional[str] = None,
    device: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train, evaluate, and persist one (model, task, seed) run.

    Idempotent: re-running with a completed JSON record on disk is a no-op
    (returns the cached record). Crashes write a status=CRASHED record so
    re-runs can decide whether to retry.
    """
    run_id = variant or f"{model}_{task}_seed{seed}"
    json_path = Path(runs_dir) / f"{run_id}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)

    cached = safe_read_json(json_path)
    if cached and _has_valid_success_rate(cached):
        if verbose:
            print(f"[skip] {run_id} sr={cached['success_rate']:.3f}")
        return cached

    import torch
    from cavr.data.dataset import DemoDataset
    from cavr.evaluation.evaluator import PolicyEvaluator
    from cavr.training.bc_trainer import BCTrainer, set_global_seed
    from cavr.envs.robosuite_envs import get_task_description

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = _build_cfg(
        config_path, task, seed,
        data_root=data_root, ckpt_root=ckpt_root, run_id=run_id,
        epochs=epochs, eval_freq=eval_freq, eval_episodes=eval_episodes,
        batch_size=batch_size,
        masking_override=masking_override, backbone_override=backbone_override,
    )

    # Baselines don't accept cached masks (different feature grid + ignored).
    # Skip mask loading for them to keep memory low and avoid confusion.
    dataset_kwargs = {}
    if model in ("r3m", "vc1") or cfg.get("masking", {}).get("enabled") is False:
        dataset_kwargs["mask_filename"] = None

    t_start = time.time()
    record: Dict[str, Any] = {
        "run_id": run_id,
        "model": model,
        "task": task,
        "seed": int(seed),
        "variant": variant,
        "status": "RUNNING",
        "git_hash": git_short_hash(),
        "gpu": gpu_name(),
        "device": device,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {
            "encoder_backbone": cfg["encoder"]["backbone"],
            "masking_enabled": cfg["masking"]["enabled"],
            "num_epochs": cfg["training"]["num_epochs"],
            "batch_size": cfg["training"]["batch_size"],
            "eval_freq": cfg["training"]["eval_freq"],
            "lr": cfg["training"]["lr"],
            "num_episodes": cfg["evaluation"]["num_episodes"],
            "image_size": cfg["env"]["camera_height"],
        },
    }
    atomic_json_write(json_path, record)

    try:
        set_global_seed(int(seed))
        net = _build_model(model, cfg, device)
        task_desc = get_task_description(task)

        dataset = DemoDataset(cfg["data"]["save_dir"], **dataset_kwargs)
        if len(dataset) == 0:
            raise RuntimeError(
                f"DemoDataset at {cfg['data']['save_dir']} has 0 timesteps. "
                f"Did demo collection succeed for task {task}?"
            )

        trainer = BCTrainer(net, cfg, device=device)
        best_val = trainer.train(dataset, task_description=task_desc)

        evaluator = PolicyEvaluator(net, cfg, device=device)
        eval_results = evaluator.evaluate()

        elapsed = time.time() - t_start
        record.update({
            "status": "DONE",
            "success_rate": float(eval_results["success_rate"]),
            "mean_return": float(eval_results["mean_return"]),
            "std_return": float(eval_results["std_return"]),
            "mean_length": float(eval_results["mean_length"]),
            "num_eval_episodes": int(eval_results["num_episodes"]),
            "best_val_loss": float(best_val) if best_val is not None else None,
            "train_losses": list(trainer.train_losses),
            "val_losses": list(trainer.val_losses),
            "elapsed_s": float(elapsed),
            "train_s": float(trainer.train_seconds),
        })
        atomic_json_write(json_path, record)

        if csv_path:
            append_csv_row(csv_path, {k: record.get(k) for k in CSV_HEADER}, header=CSV_HEADER)

        if verbose:
            print(f"[done] {run_id} sr={record['success_rate']:.3f} "
                  f"elapsed={record['elapsed_s']:.0f}s")
        return record

    except Exception as e:
        elapsed = time.time() - t_start
        record.update({
            "status": "CRASHED",
            "success_rate": float("nan"),
            "error": repr(e),
            "traceback": traceback.format_exc(),
            "elapsed_s": float(elapsed),
        })
        atomic_json_write(json_path, record)

        if csv_path:
            append_csv_row(csv_path, {k: record.get(k) for k in CSV_HEADER}, header=CSV_HEADER)

        if verbose:
            print(f"[FAIL] {run_id}: {e}")
        return record
