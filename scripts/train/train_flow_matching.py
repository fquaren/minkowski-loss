#!/usr/bin/env python
"""Train the conditional flow-matching residual downscaler (Phase 2)."""

import argparse
import os

import yaml

from src.utils import load_config, set_seed
from src.trainers.flow_matching import run_training

try:
    import optuna
except Exception:
    optuna = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--backbone", type=str, default=None,
                        help="frozen stage-1 backbone checkpoint (overrides BACKBONE_CHECKPOINT)")
    parser.add_argument("--data_percentage", type=float, default=100.0)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_amp", action="store_true",
                        help="disable fp16 autocast/GradScaler (fp32 probe for NaN)")
    parser.add_argument("--max_batches", type=int, default=None,
                        help="cap train/val batches per epoch for fast probing")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(args.seed)

    if args.tune and optuna is not None:
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda t: run_training(config, args, trial=t), n_trials=args.n_trials)
        os.makedirs("training_params", exist_ok=True)
        with open("training_params/flow_matching_params.yaml", "w") as f:
            yaml.dump(study.best_params, f)
    else:
        run_training(config, args)


if __name__ == "__main__":
    main()