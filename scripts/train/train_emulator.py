#!/usr/bin/env python
"""Train the Minkowski functional emulator."""

import argparse
import optuna
import yaml
import os

from src.utils import load_config, set_seed
from src.trainers.emulator import run_training


def main():
    parser = argparse.ArgumentParser(description="Train Minkowski emulator")
    parser.add_argument("config", type=str, help="Path to config.yaml")
    parser.add_argument("--arch", type=str, default="Constrained",
                        choices=["Baseline", "Lipschitz", "Constrained"])
    parser.add_argument("--optimize", action="store_true",
                        help="Run Optuna hyperparameter search")
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--data_fraction", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=7.75e-5)
    parser.add_argument("--wd", type=float, default=2.49e-6)
    parser.add_argument("--load_params", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(args.seed)

    if args.optimize:
        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(),
        )
        study.optimize(
            lambda t: run_training(config, args, trial=t),
            n_trials=args.n_trials,
        )
        print(f"Best params: {study.best_trial.params}")
        save_path = f"training_params/best_params_{args.arch}.yaml"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            yaml.dump(study.best_trial.params, f)

        # Final training with best params
        args.lr = study.best_trial.params["lr"]
        args.wd = study.best_trial.params["weight_decay"]
        run_training(config, args)
    else:
        run_training(config, args)


if __name__ == "__main__":
    main()
