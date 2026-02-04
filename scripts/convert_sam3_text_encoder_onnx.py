#!/usr/bin/env python3
"""Export SAM3 text encoder to ONNX for reuse across experts."""
from __future__ import annotations

import argparse
import os

import torch

from sam3.model_builder import build_sam3_image_model


class TextEncoderOnnxWrapper(torch.nn.Module):
    def __init__(self, text_encoder: torch.nn.Module):
        super().__init__()
        self.text_encoder = text_encoder

    def forward(self, input_ids: torch.Tensor):  # type: ignore[override]
        text_attention_mask = (input_ids != 0).bool()
        inputs_embeds = self.text_encoder.encoder.token_embedding(input_ids)
        _, text_memory = self.text_encoder.encoder(input_ids)
        text_attention_mask = text_attention_mask.ne(1)
        text_memory = text_memory.transpose(0, 1)
        text_memory_resized = self.text_encoder.resizer(text_memory)
        return text_attention_mask, text_memory_resized, inputs_embeds.transpose(0, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export SAM3 text encoder to ONNX.")
    parser.add_argument("--checkpoint", required=True, help="SAM3 checkpoint path.")
    parser.add_argument("--output", required=True, help="Output ONNX file path.")
    parser.add_argument(
        "--context-length",
        type=int,
        default=77,
        help="Token context length for export.",
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
    text_encoder = model.backbone.language_backbone
    wrapper = TextEncoderOnnxWrapper(text_encoder).to(args.device).eval()

    dummy = torch.zeros((1, args.context_length), dtype=torch.long, device=args.device)
    torch.onnx.export(
        wrapper,
        dummy,
        args.output,
        input_names=["input_ids"],
        output_names=["text_attention_mask", "text_memory", "text_embeds"],
        opset_version=17,
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}},
    )

    print(f"Saved text encoder ONNX to {args.output}")


if __name__ == "__main__":
    main()
