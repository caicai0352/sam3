#!/usr/bin/env python3
# pyre-unsafe

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from scripts.adapters.common import TinyRegModel, build_synth_data, import_adapters_module


def get_args():
    p = argparse.ArgumentParser(description="Evaluate tiny model and write metrics.json")
    p.add_argument("--base-ckpt", type=Path, required=True)
    p.add_argument("--adapter-ckpt", type=Path, default=None)
    p.add_argument("--output", type=Path, default=Path("outputs/adapter_smoke/metrics.json"))
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--use-adapter", action="store_true")
    return p.parse_args()


def main():
    args = get_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    model = TinyRegModel().to(args.device)
    adapters = import_adapters_module()

    if args.use_adapter:
        adapters.inject_adapters(
            model,
            {
                "enabled": True,
                "target_patterns": ["net.0", "net.2", "net.4"],
                "target_types": ["Linear"],
                "bottleneck_dim": 8,
            },
        )

    base_ckpt = torch.load(args.base_ckpt, map_location="cpu")
    base_sd = base_ckpt["model"] if isinstance(base_ckpt, dict) and "model" in base_ckpt else base_ckpt
    model.load_state_dict(base_sd, strict=False)

    if args.adapter_ckpt is not None:
        ad = torch.load(args.adapter_ckpt, map_location="cpu")
        ad_sd = ad["model"] if isinstance(ad, dict) and "model" in ad else ad
        adapters.load_adapter_state_dict(model, ad_sd, strict=False)

    model.eval()
    x, y = build_synth_data(n=128, seed=args.seed, device=args.device)
    with torch.no_grad():
        pred = model(x)
    mse = F.mse_loss(pred, y).item()
    mae = torch.mean(torch.abs(pred - y)).item()

    metrics = {
        "mse": mse,
        "mae": mae,
        "base_ckpt": str(args.base_ckpt),
        "adapter_ckpt": str(args.adapter_ckpt) if args.adapter_ckpt else None,
        "use_adapter": args.use_adapter,
    }
    args.output.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
