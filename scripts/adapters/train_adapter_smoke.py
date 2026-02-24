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
    p = argparse.ArgumentParser(description="Train tiny regression model with optional adapters.")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/adapter_smoke"))
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--use-adapter", action="store_true", help="Inject adapters before training.")
    p.add_argument("--freeze-base", action="store_true", help="Freeze non-adapter params when --use-adapter.")
    p.add_argument("--save-adapter-only", action="store_true", help="Save adapter-only checkpoint when --use-adapter.")
    return p.parse_args()


def main():
    args = get_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    model = TinyRegModel().to(args.device)

    adapters = import_adapters_module()
    inject_cfg = {
        "enabled": args.use_adapter,
        "target_patterns": ["net.0", "net.2", "net.4"],
        "target_types": ["Linear"],
        "bottleneck_dim": 8,
    }
    wrapped = adapters.inject_adapters(model, inject_cfg)
    if args.use_adapter and args.freeze_base:
        adapters.freeze_except_adapters(model)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.Adam(trainable, lr=args.lr)

    x, y = build_synth_data(n=256, seed=args.seed, device=args.device)
    losses = []
    for i in range(args.steps):
        idx = torch.randint(0, x.shape[0], (args.batch_size,), device=args.device)
        pred = model(x[idx])
        loss = F.mse_loss(pred, y[idx])
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        losses.append(float(loss.item()))

    ckpt = {
        "model": model.state_dict(),
        "use_adapter": args.use_adapter,
        "wrapped_modules": wrapped,
    }
    torch.save(ckpt, args.output_dir / "base_or_full.pt")

    adapter_ckpt_path = None
    if args.use_adapter and args.save_adapter_only:
        adapter_sd = adapters.get_adapter_state_dict(model)
        adapter_ckpt_path = args.output_dir / "adapter_only.pt"
        torch.save({"model": adapter_sd, "adapter_only": True}, adapter_ckpt_path)

    report = {
        "steps": args.steps,
        "final_loss": losses[-1],
        "use_adapter": args.use_adapter,
        "freeze_base": args.freeze_base,
        "num_wrapped": len(wrapped),
        "checkpoint": str(args.output_dir / "base_or_full.pt"),
        "adapter_checkpoint": str(adapter_ckpt_path) if adapter_ckpt_path else None,
    }
    (args.output_dir / "train_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
