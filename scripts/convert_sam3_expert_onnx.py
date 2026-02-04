#!/usr/bin/env python3
"""Export SAM3 expert head to ONNX (excluding shared ViT)."""
from __future__ import annotations

import argparse
import os
from typing import Tuple

import torch

from sam3.model.data_misc import FindStage
from sam3.model_builder import build_sam3_image_model


class ExpertOnnxWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.find_stage = FindStage(
            img_ids=torch.tensor([0], dtype=torch.long),
            text_ids=torch.tensor([0], dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )
        self.geometric_prompt = model._get_dummy_prompt()

    def forward(  # type: ignore[override]
        self,
        fpn_s0: torch.Tensor,
        fpn_s1: torch.Tensor,
        fpn_s2: torch.Tensor,
        fpn_s3: torch.Tensor,
        text_attention_mask: torch.Tensor,
        text_memory: torch.Tensor,
        text_embeds: torch.Tensor,
    ):
        backbone_out = {
            "backbone_fpn": [fpn_s0, fpn_s1, fpn_s2, fpn_s3],
            "language_mask": text_attention_mask,
            "language_features": text_memory,
            "language_embeds": text_embeds,
        }
        outputs = self.model.forward_grounding(
            backbone_out=backbone_out,
            find_input=self.find_stage,
            geometric_prompt=self.geometric_prompt,
            find_target=None,
        )
        return outputs["pred_boxes"], outputs["pred_masks"], outputs["pred_logits"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export SAM3 expert head to ONNX.")
    parser.add_argument("--checkpoint", required=True, help="Expert checkpoint path.")
    parser.add_argument("--output", required=True, help="Output ONNX file path.")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device used for export.",
    )
    parser.add_argument(
        "--fpn-shapes",
        default="1x256x126x126,1x256x63x63,1x256x32x32,1x256x16x16",
        help="Comma-separated FPN shapes for dummy export tensors.",
    )
    parser.add_argument(
        "--text-shapes",
        default="1x77,77x1x256,77x1x1024",
        help="Comma-separated shapes for text_attention_mask, text_memory, text_embeds.",
    )
    return parser.parse_args()


def _parse_shape(shape: str) -> Tuple[int, ...]:
    return tuple(int(dim) for dim in shape.split("x"))


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
    wrapper = ExpertOnnxWrapper(model).to(args.device).eval()

    fpn_shapes = [_parse_shape(s) for s in args.fpn_shapes.split(",")]
    text_shapes = [_parse_shape(s) for s in args.text_shapes.split(",")]

    dummy_inputs = [
        torch.randn(shape, device=args.device)
        for shape in fpn_shapes
    ]
    text_attention_mask = torch.zeros(text_shapes[0], device=args.device, dtype=torch.bool)
    text_memory = torch.randn(text_shapes[1], device=args.device)
    text_embeds = torch.randn(text_shapes[2], device=args.device)

    torch.onnx.export(
        wrapper,
        (*dummy_inputs, text_attention_mask, text_memory, text_embeds),
        args.output,
        input_names=[
            "fpn_s0",
            "fpn_s1",
            "fpn_s2",
            "fpn_s3",
            "text_attention_mask",
            "text_memory",
            "text_embeds",
        ],
        output_names=["pred_boxes", "pred_masks", "pred_logits"],
        opset_version=17,
    )

    print(f"Saved expert ONNX to {args.output}")


if __name__ == "__main__":
    main()
