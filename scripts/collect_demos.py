"""Collect expert demonstrations from robosuite environments."""
import argparse
import yaml

from cavr.data.collector import collect_scripted_demos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="cavr/configs/default.yaml")
    parser.add_argument("--env", default=None, help="Override env name")
    parser.add_argument("--num-demos", type=int, default=None)
    parser.add_argument("--save-dir", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.env:
        cfg["env"]["name"] = args.env
    if args.num_demos:
        cfg["data"]["num_demos"] = args.num_demos
    if args.save_dir:
        cfg["data"]["save_dir"] = args.save_dir

    collect_scripted_demos(cfg)


if __name__ == "__main__":
    main()
