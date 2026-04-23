#!/usr/bin/env python
"""Train deterministic UNet with frozen emulator Minkowski loss."""

import argparse
import optuna
import yaml
import os

from src.utils import load_config, set_seed
from src.trainers.unet_emulator import run_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument(
        "--params_path",
        type=str,
        required=True,
        help="YAML with lr/weight_decay for the UNet",
    )
    parser.add_argument("--data_percentage", type=float, default=100.0)
    parser.add_argument(
        "--weight_geom",
        type=float,
        default=None,
        help="Max geometric loss weight (overrides config)",
    )
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(args.seed)

    if args.tune:
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda t: run_training(config, args, trial=t),
            n_trials=args.n_trials,
        )
        save_path = "training_params/unet_emulator_params.yaml"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            yaml.dump(study.best_params, f)
        print(f"Best params: {study.best_params}")
    else:
        run_training(config, args)


if __name__ == "__main__":
    main()
