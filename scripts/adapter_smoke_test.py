#!/usr/bin/env python3
# pyre-unsafe

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def get_args():
    p = argparse.ArgumentParser(
        description="Run full adapter smoke pipeline: train(base+adapter), eval, visualize."
    )
    p.add_argument("--output-dir", type=Path, default=Path("outputs/adapter_smoke/full"))
    p.add_argument("--steps", type=int, default=30)
    return p.parse_args()


def main():
    args = get_args()
    out = args.output_dir
    base_dir = out / "base"
    ad_dir = out / "adapter"

    run(
        [
            sys.executable,
            "scripts/adapters/train_adapter_smoke.py",
            "--output-dir",
            str(base_dir),
            "--steps",
            str(args.steps),
        ]
    )
    run(
        [
            sys.executable,
            "scripts/adapters/train_adapter_smoke.py",
            "--output-dir",
            str(ad_dir),
            "--use-adapter",
            "--freeze-base",
            "--save-adapter-only",
            "--steps",
            str(args.steps),
        ]
    )
    run(
        [
            sys.executable,
            "scripts/adapters/eval_adapter_smoke.py",
            "--base-ckpt",
            str(base_dir / "base_or_full.pt"),
            "--output",
            str(base_dir / "metrics.json"),
        ]
    )
    run(
        [
            sys.executable,
            "scripts/adapters/eval_adapter_smoke.py",
            "--base-ckpt",
            str(ad_dir / "base_or_full.pt"),
            "--adapter-ckpt",
            str(ad_dir / "adapter_only.pt"),
            "--use-adapter",
            "--output",
            str(ad_dir / "metrics.json"),
        ]
    )
    run(
        [
            sys.executable,
            "scripts/adapters/visualize_adapter_compare.py",
            "--base-ckpt",
            str(base_dir / "base_or_full.pt"),
            "--adapter-base-ckpt",
            str(ad_dir / "base_or_full.pt"),
            "--adapter-ckpt",
            str(ad_dir / "adapter_only.pt"),
            "--output-image",
            str(out / "compare_base_vs_adapter.png"),
        ]
    )

    print("adapter smoke test passed")


if __name__ == "__main__":
    main()
