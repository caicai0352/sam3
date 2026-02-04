#!/usr/bin/env python3
"""Export SAM3 shared vision backbone to ONNX.

This exports the shared ViT image encoder so it can be reused across experts.
"""
from __future__ import annotations

import argparse
import os
from typing import Tuple

import torch

from sam3.model_builder import build_sam3_image_model


class BackboneOnnxWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.backbone = model.backbone

    def forward(self, image: torch.Tensor):  # type: ignore[override]
        backbone_out = self.backbone.forward_image(image)
        backbone_fpn = backbone_out.get("backbone_fpn", [])
        if not backbone_fpn:
            raise ValueError("backbone_fpn is empty; cannot export.")
        return tuple(backbone_fpn)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export SAM3 shared vision backbone to ONNX."
    )
    parser.add_argument("--checkpoint", required=True, help="SAM3 checkpoint path.")
    parser.add_argument("--output", required=True, help="Output ONNX file path.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=1008,
        help="Input resolution for export (square).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device used for export.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    model = build_sam3_image_model(
        device=args.device,
        eval_mode=True,
        checkpoint_path=args.checkpoint,
        load_from_HF=False,
    )
    wrapper = BackboneOnnxWrapper(model).to(args.device).eval()

    dummy = torch.randn(1, 3, args.resolution, args.resolution, device=args.device)
    torch.onnx.export(
        wrapper,
        dummy,
        args.output,
        input_names=["image"],
        output_names=["fpn_s0", "fpn_s1", "fpn_s2", "fpn_s3"],
        opset_version=17,
        dynamic_axes={"image": {0: "batch"}},
    )

    print(f"Saved ONNX backbone to {args.output}")


if __name__ == "__main__":
    main()
