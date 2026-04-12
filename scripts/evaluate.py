"""Evaluate a trained policy in simulation."""
import argparse
import json
import yaml
import torch

from cavr.models.pipeline import CAVR
from cavr.models.baselines import R3MBaseline, VC1Baseline
from cavr.evaluation.evaluator import PolicyEvaluator


MODEL_BUILDERS = {
    "cavr": lambda cfg, device: CAVR(cfg),
    "r3m": lambda cfg, device: R3MBaseline(cfg["policy"], device=device),
    "vc1": lambda cfg, device: VC1Baseline(cfg["policy"], device=device),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="cavr/configs/default.yaml")
    parser.add_argument("--model", default="cavr", choices=MODEL_BUILDERS.keys())
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--env", default=None)
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.env:
        cfg["env"]["name"] = args.env
    if args.num_episodes:
        cfg["evaluation"]["num_episodes"] = args.num_episodes

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MODEL_BUILDERS[args.model](cfg, device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)

    evaluator = PolicyEvaluator(model, cfg, device=device)
    results = evaluator.evaluate()

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
