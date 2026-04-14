#!/usr/bin/env python
"""Consolidate and split daily metadata using native system utilities."""

import os
import sys
import glob
import argparse
import subprocess
import shutil
from tqdm import tqdm

from src.utils import load_config

def get_shuf_command():
    """Detect platform and return the appropriate shuffle command."""
    if sys.platform == "darwin":
        if shutil.which("gshuf"):
            return "gshuf"
        else:
            raise EnvironmentError("macOS requires 'gshuf'. Install via: brew install coreutils")
    if shutil.which("shuf"):
        return "shuf"
    raise EnvironmentError("No 'shuf' utility found on the system.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()
    config = load_config(args.config)

    shuf_cmd = get_shuf_command()
    if not shutil.which("wc"):
        raise EnvironmentError("'wc' utility missing from system PATH.")

    meta_dir = config["METADATA_DIR"]
    split_ratios = config["SPLIT_RATIOS"]
    daily_meta_dir = os.path.join(meta_dir, "temp_metadata")

    consolidated_file = os.path.join(meta_dir, "all_patches.tmp")
    shuffled_file = os.path.join(meta_dir, "all_patches_shuffled.tmp")

    daily_files = sorted(glob.glob(os.path.join(daily_meta_dir, "*.txt")))
    
    with open(consolidated_file, "wb") as outfile:
        for filename in tqdm(daily_files, desc="Concatenating daily metadata"):
            with open(filename, "rb") as infile:
                shutil.copyfileobj(infile, outfile)

    subprocess.run([shuf_cmd, consolidated_file, "-o", shuffled_file], check=True)

    result = subprocess.run(["wc", "-l", shuffled_file], capture_output=True, text=True, check=True)
    total_patches = int(result.stdout.split()[0])

    train_end = int(total_patches * split_ratios["train"])
    val_end = train_end + int(total_patches * split_ratios["validation"])

    paths = {
        "train": config.get("TRAIN_METADATA_FILE", os.path.join(meta_dir, "train_patches_metadata.txt")),
        "val": config.get("VAL_METADATA_FILE", os.path.join(meta_dir, "val_patches_metadata.txt")),
        "test": config.get("TEST_METADATA_FILE", os.path.join(meta_dir, "test_patches_metadata.txt")),
    }

    with open(shuffled_file, "r") as f_in, \
         open(paths["train"], "w") as f_train, \
         open(paths["val"], "w") as f_val, \
         open(paths["test"], "w") as f_test:
        
        for i, line in tqdm(enumerate(f_in), total=total_patches, desc="Writing splits"):
            if i < train_end:
                f_train.write(line)
            elif i < val_end:
                f_val.write(line)
            else:
                f_test.write(line)

    os.remove(consolidated_file)
    os.remove(shuffled_file)
    shutil.rmtree(daily_meta_dir)


if __name__ == "__main__":
    main()