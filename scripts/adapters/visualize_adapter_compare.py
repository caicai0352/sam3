#!/usr/bin/env python3
# pyre-unsafe

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image, ImageDraw

from scripts.adapters.common import TinyRegModel, build_synth_data, import_adapters_module


def load_model(base_ckpt, use_adapter=False, adapter_ckpt=None, device="cpu"):
    model = TinyRegModel().to(device)
    adapters = import_adapters_module()
    if use_adapter:
        adapters.inject_adapters(
            model,
            {
                "enabled": True,
                "target_patterns": ["net.0", "net.2", "net.4"],
                "target_types": ["Linear"],
                "bottleneck_dim": 8,
            },
        )

    ckpt = torch.load(base_ckpt, map_location="cpu")
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(sd, strict=False)

    if adapter_ckpt is not None:
        ad = torch.load(adapter_ckpt, map_location="cpu")
        ad_sd = ad["model"] if isinstance(ad, dict) and "model" in ad else ad
        adapters.load_adapter_state_dict(model, ad_sd, strict=False)

    model.eval()
    return model


def _to_plot_points(gt, pred, w=480, h=360, pad=20):
    gt = gt.astype(np.float32)
    pred = pred.astype(np.float32)
    lo = min(gt.min(), pred.min())
    hi = max(gt.max(), pred.max())
    den = max(hi - lo, 1e-6)
    x = pad + (gt - lo) / den * (w - 2 * pad)
    y = h - pad - (pred - lo) / den * (h - 2 * pad)
    return x, y


def get_args():
    p = argparse.ArgumentParser(description="Visualize base vs adapter prediction comparison.")
    p.add_argument("--base-ckpt", type=Path, required=True)
    p.add_argument("--adapter-base-ckpt", type=Path, default=None, help="full checkpoint for adapter model")
    p.add_argument("--adapter-ckpt", type=Path, default=None, help="adapter-only checkpoint")
    p.add_argument("--output-image", type=Path, default=Path("outputs/adapter_smoke/compare.png"))
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def main():
    args = get_args()
    args.output_image.parent.mkdir(parents=True, exist_ok=True)

    x, y = build_synth_data(n=128, seed=args.seed, device=args.device)
    base_model = load_model(args.base_ckpt, use_adapter=False, device=args.device)
    with torch.no_grad():
        p_base = base_model(x).cpu().numpy()

    p_ad = None
    if args.adapter_base_ckpt is not None:
        adapter_model = load_model(
            args.adapter_base_ckpt,
            use_adapter=True,
            adapter_ckpt=args.adapter_ckpt,
            device=args.device,
        )
        with torch.no_grad():
            p_ad = adapter_model(x).cpu().numpy()

    y = y.cpu().numpy()
    w, h = 960, 360
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, w - 1, h - 1), outline="black", width=1)
    draw.text((10, 8), "Base vs Adapter prediction (dim0)", fill="black")

    x_base, y_base = _to_plot_points(y[:, 0], p_base[:, 0], w=w // 2, h=h)
    for xi, yi in zip(x_base, y_base):
        draw.ellipse((xi - 1, yi - 1, xi + 1, yi + 1), fill=(30, 144, 255))

    draw.line((w // 2, 0, w // 2, h), fill="gray", width=1)
    draw.text((w // 2 + 10, 8), "Abs error histogram", fill="black")

    base_err = np.abs(p_base - y).mean(axis=1)
    bins = np.linspace(0, max(base_err.max(), (np.abs(p_ad - y).mean(axis=1).max() if p_ad is not None else 0.0)) + 1e-6, 20)
    hist_b, _ = np.histogram(base_err, bins=bins)
    hist_a = None
    if p_ad is not None:
        ad_err = np.abs(p_ad - y).mean(axis=1)
        hist_a, _ = np.histogram(ad_err, bins=bins)

    max_h = max(hist_b.max(), hist_a.max() if hist_a is not None else 1)
    x0 = w // 2 + 20
    plot_w = w // 2 - 40
    plot_h = h - 50
    for i in range(len(hist_b)):
        bx0 = x0 + int(i * plot_w / len(hist_b))
        bx1 = x0 + int((i + 1) * plot_w / len(hist_b)) - 1
        bh = int((hist_b[i] / max_h) * plot_h)
        draw.rectangle((bx0, h - 20 - bh, bx1, h - 20), fill=(30, 144, 255))
        if hist_a is not None:
            ah = int((hist_a[i] / max_h) * plot_h)
            draw.rectangle((bx0, h - 20 - ah, bx1, h - 20 - ah + 2), fill=(220, 20, 60))

    draw.text((10, h - 18), "blue=base scatter/hist, red=adapter hist", fill="black")
    img.save(args.output_image)
    print(f"saved: {args.output_image}")


if __name__ == "__main__":
    main()
