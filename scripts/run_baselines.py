"""Train and evaluate all baselines for comparison against CAVR."""
import argparse
import json
import os
import yaml
import torch

from cavr.models.pipeline import CAVR
from cavr.models.baselines import R3MBaseline, VC1Baseline
from cavr.data.dataset import DemoDataset
from cavr.training.bc_trainer import BCTrainer
from cavr.evaluation.evaluator import PolicyEvaluator
from cavr.envs.robosuite_envs import get_task_description


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="cavr/configs/default.yaml")
    parser.add_argument("--env", default=None)
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.env:
        cfg["env"]["name"] = args.env
    if args.data_dir:
        cfg["data"]["save_dir"] = args.data_dir

    device = "cuda" if torch.cuda.is_available() else "cpu"
    task_desc = get_task_description(cfg["env"]["name"])
    dataset = DemoDataset(cfg["data"]["save_dir"])

    models = {
        "cavr": CAVR(cfg),
        "r3m": R3MBaseline(cfg["policy"], device=device),
        "vc1": VC1Baseline(cfg["policy"], device=device),
    }

    all_results = {}

    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Training: {name}")
        print(f"{'='*60}\n")

        run_cfg = cfg.copy()
        run_cfg["training"] = cfg["training"].copy()
        run_cfg["training"]["checkpoint_dir"] = f"checkpoints/{name}_{cfg['env']['name']}"

        trainer = BCTrainer(model, run_cfg, device=device)
        val_loss = trainer.train(dataset, task_description=task_desc)

        evaluator = PolicyEvaluator(model, run_cfg, device=device)
        eval_results = evaluator.evaluate()

        all_results[name] = {"val_loss": val_loss, **eval_results}

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/baseline_comparison.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"{'Model':<12} {'SR':>8} {'Return':>10} {'Val Loss':>10}")
    print(f"{'-'*70}")
    for name, r in all_results.items():
        print(
            f"{name:<12} {r['success_rate']:>7.1%} "
            f"{r['mean_return']:>10.2f} {r['val_loss']:>10.6f}"
        )
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
