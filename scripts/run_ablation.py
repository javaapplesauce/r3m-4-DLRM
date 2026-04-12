"""Run ablation studies comparing CAVR variants."""
import argparse
import yaml
import torch

from cavr.evaluation.ablation import run_ablation


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
    run_ablation(cfg, device=device)


if __name__ == "__main__":
    main()
