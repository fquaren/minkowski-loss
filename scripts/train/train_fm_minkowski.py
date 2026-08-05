#!/usr/bin/env python
"""Launch Phase-3 reward fine-tuning: Minkowski loss coupled to flow matching (DRaFT-K).

Starts from a trained clean flow-matching checkpoint and fine-tunes it against the
structural reward, anchored to the frozen original. Training aborts automatically if the
held-out anisotropy or peak-ratio guards trip, which is what a degenerate
filamentary solution looks like.

Usage:
  python scripts/train/train_fm_minkowski.py config.yaml \
      --fm_checkpoint runs/sr_flow_matching/<run>/fm_best.pth \
      --backbone runs/sr_analytical/<run>/unet_best.pth \
      --w_reward 1e-3

Quick probe before committing a full run (a few optimiser steps only):
  ... --max_steps 5
"""

import argparse

from src.utils import load_config
from src.trainers.reward_finetune import run_reward_finetune


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--fm_checkpoint", required=True,
                    help="clean flow-matching checkpoint to fine-tune from")
    ap.add_argument("--backbone", default=None, help="overrides BACKBONE_CHECKPOINT")
    ap.add_argument("--w_reward", type=float, default=None,
                    help="overrides REWARD_WEIGHT in config")
    ap.add_argument("--max_steps", type=int, default=None,
                    help="cap optimiser steps per epoch (fast probe)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    config = load_config(args.config)
    run_reward_finetune(config, args)


if __name__ == "__main__":
    main()