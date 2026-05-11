"""Train a CAVR or baseline policy via behavioral cloning.

Two entry modes:

  --run-id RUN_ID  → invokes the cavr.utils.runs.train_and_eval orchestrator
                     (writes outputs/runs/<run_id>.json, prints
                     "[done] <run_id> sr=... elapsed=..." that the Colab
                     notebook parses). This is the path used by the sweep.

  (default)        → classic in-place training, no JSON record. Kept for
                     manual smoke tests and `python scripts/train.py
                     --model cavr --env Lift --epochs 2` style debugging.
"""
import argparse
import time
import yaml
import torch

from cavr.models.pipeline import CAVR
from cavr.models.baselines import R3MBaseline, VC1Baseline
from cavr.data.dataset import DemoDataset
from cavr.training.bc_trainer import BCTrainer, set_global_seed
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
    parser.add_argument(
        "--no-masking",
        action="store_true",
        help="Disable concept masking (overrides cfg.masking.enabled). "
             "Useful for smoke tests — skips Grounding DINO + SAM2 entirely.",
    )
    parser.add_argument(
        "--mask-file",
        default=None,
        help="Path to a precomputed masks.hdf5 file. If omitted, the dataset "
             "auto-detects masks.hdf5 alongside demos.hdf5. Ignored when "
             "--no-masking is set.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="If set, dispatch to cavr.utils.runs.train_and_eval which also "
             "evaluates and writes a JSON record. Use this from automated "
             "sweeps. Without it, this script just trains.",
    )
    parser.add_argument("--runs-dir", default="outputs/runs",
                        help="Directory for per-run JSON records (with --run-id).")
    parser.add_argument("--csv-path", default=None,
                        help="Optional CSV to append a summary row to (with --run-id).")
    parser.add_argument("--num-eval-episodes", type=int, default=None,
                        help="Override evaluation.num_episodes.")
    args = parser.parse_args()

    if args.run_id is not None:
        from cavr.utils.runs import train_and_eval
        train_and_eval(
            args.model,
            args.env or "Lift",
            args.seed if args.seed is not None else 0,
            runs_dir=args.runs_dir,
            csv_path=args.csv_path,
            config_path=args.config,
            epochs=args.epochs,
            batch_size=args.batch_size,
            eval_episodes=args.num_eval_episodes,
            masking_override=False if args.no_masking else None,
            variant=args.run_id,
        )
        return

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
    if args.seed is not None:
        cfg["training"]["seed"] = args.seed
    if args.no_masking:
        cfg["masking"]["enabled"] = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_global_seed(int(cfg["training"]["seed"]))

    model = MODEL_BUILDERS[args.model](cfg, device)

    dataset_kwargs = {}
    if args.no_masking:
        dataset_kwargs["mask_filename"] = None
    elif args.mask_file:
        dataset_kwargs["mask_path"] = args.mask_file
    dataset = DemoDataset(cfg["data"]["save_dir"], **dataset_kwargs)
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

    mask_status = (
        "disabled"
        if not cfg.get("masking", {}).get("enabled", False)
        else f"live (slow)" if not getattr(dataset, "has_masks", False)
        else f"cached ({dataset.mask_path})"
    )
    print(f"Model: {args.model}")
    print(f"Task:  {cfg['env']['name']} ({task_desc})")
    print(f"Data:  {len(dataset)} timesteps")
    print(f"Masks: {mask_status}")
    print(f"Device: {device}")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Params: {trainable:,} trainable / {frozen:,} frozen\n")

    t0 = time.time()
    best_val = trainer.train(dataset, task_description=task_desc)
    elapsed = time.time() - t0
    run_id = f"{args.model}_{cfg['env']['name']}_seed{cfg['training']['seed']}"
    print(f"[done] {run_id} sr=nan elapsed={elapsed:.0f}s best_val={best_val:.6f}")


if __name__ == "__main__":
    main()
