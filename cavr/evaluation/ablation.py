import copy
import json
import os

import torch

from cavr.models.pipeline import CAVR
from cavr.data.dataset import DemoDataset
from cavr.training.bc_trainer import BCTrainer
from cavr.evaluation.evaluator import PolicyEvaluator
from cavr.envs.robosuite_envs import get_task_description


ABLATION_CONFIGS = {
    "cavr_vitl_masked": {
        "description": "Full CAVR: DINOv2 ViT-L/14 + SAM masking",
        "encoder.backbone": "dinov2_vitl14",
        "masking.enabled": True,
    },
    "cavr_vitl_no_mask": {
        "description": "DINOv2 ViT-L/14 without concept masking",
        "encoder.backbone": "dinov2_vitl14",
        "masking.enabled": False,
    },
    "cavr_vitb_masked": {
        "description": "DINOv2 ViT-B/14 + SAM masking",
        "encoder.backbone": "dinov2_vitb14",
        "masking.enabled": True,
    },
    "cavr_vitb_no_mask": {
        "description": "DINOv2 ViT-B/14 without concept masking",
        "encoder.backbone": "dinov2_vitb14",
        "masking.enabled": False,
    },
}


def _apply_overrides(cfg, overrides):
    cfg = copy.deepcopy(cfg)
    for key, value in overrides.items():
        if key == "description":
            continue
        parts = key.split(".")
        d = cfg
        for p in parts[:-1]:
            d = d[p]
        d[parts[-1]] = value
    return cfg


def run_ablation(base_cfg, device="cpu"):
    """Run ablation studies comparing different CAVR configurations.

    Trains and evaluates each ablation variant, then saves a comparison table.
    """
    data_dir = base_cfg["data"]["save_dir"]
    task_desc = get_task_description(base_cfg["env"]["name"])
    results = {}

    for name, overrides in ABLATION_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Ablation: {name}")
        print(f"  {overrides['description']}")
        print(f"{'='*60}\n")

        cfg = _apply_overrides(base_cfg, overrides)
        cfg["training"]["checkpoint_dir"] = os.path.join("checkpoints", name)

        model = CAVR(cfg)
        dataset = DemoDataset(data_dir)

        trainer = BCTrainer(model, cfg, device=device)
        best_val_loss = trainer.train(dataset, task_description=task_desc)

        evaluator = PolicyEvaluator(model, cfg, device=device)
        eval_results = evaluator.evaluate()

        results[name] = {
            "description": overrides["description"],
            "val_loss": best_val_loss,
            **eval_results,
        }

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "ablation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    _print_comparison(results)
    print(f"\nResults saved to {results_path}")
    return results


def _print_comparison(results):
    print(f"\n{'='*80}")
    print(f"{'Variant':<25} {'SR':>8} {'Return':>10} {'Val Loss':>10}")
    print(f"{'-'*80}")
    for name, r in results.items():
        print(
            f"{name:<25} {r['success_rate']:>7.1%} "
            f"{r['mean_return']:>10.2f} {r['val_loss']:>10.6f}"
        )
    print(f"{'='*80}")
