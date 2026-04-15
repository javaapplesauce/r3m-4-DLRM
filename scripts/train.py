"""Train a CAVR or baseline policy via behavioral cloning."""
import argparse
import yaml
import torch

from cavr.models.pipeline import CAVR
from cavr.models.baselines import R3MBaseline, VC1Baseline
from cavr.data.dataset import DemoDataset
from cavr.training.bc_trainer import BCTrainer
from cavr.envs.robosuite_envs import get_task_description


MODEL_BUILDERS = {
    "cavr": lambda cfg, device: CAVR(cfg),
    "r3m": lambda cfg, device: R3MBaseline(cfg["policy"], device=device),
    "vc1": lambda cfg, device: VC1Baseline(cfg["policy"], device=device),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="cavr/configs/default.yaml")
    parser.add_argument("--model", default="cavr", choices=MODEL_BUILDERS.keys())
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--env", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.env:
        cfg["env"]["name"] = args.env
    if args.data_dir:
        cfg["data"]["save_dir"] = args.data_dir
    if args.epochs:
        cfg["training"]["num_epochs"] = args.epochs
    if args.lr:
        cfg["training"]["lr"] = args.lr
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    if args.seed:
        cfg["training"]["seed"] = args.seed

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg["training"]["seed"])

    model = MODEL_BUILDERS[args.model](cfg, device)
    dataset = DemoDataset(cfg["data"]["save_dir"])
    if len(dataset) == 0:
        raise RuntimeError(
            f"Dataset at {cfg['data']['save_dir']} has 0 timesteps. "
            f"Did demo collection succeed, and does --data-dir match the "
            f"directory you wrote to in scripts/collect_demos.py?"
        )
    task_desc = get_task_description(cfg["env"]["name"])

    cfg["training"]["checkpoint_dir"] = f"checkpoints/{args.model}_{cfg['env']['name']}"

    trainer = BCTrainer(model, cfg, device=device)
    if args.wandb:
        trainer.setup_wandb(
            cfg["training"]["wandb_project"],
            f"{args.model}_{cfg['env']['name']}",
        )

    print(f"Model: {args.model}")
    print(f"Task:  {cfg['env']['name']} ({task_desc})")
    print(f"Data:  {len(dataset)} timesteps")
    print(f"Device: {device}")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Params: {trainable:,} trainable / {frozen:,} frozen\n")

    trainer.train(dataset, task_description=task_desc)


if __name__ == "__main__":
    main()
