#!/usr/bin/env python
"""Apply offline mixup augmentation to training data."""

import argparse
import multiprocessing as mp

from src.data.augmentation import run_mixup_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_mixup_pipeline(args.config, seed=args.seed)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
