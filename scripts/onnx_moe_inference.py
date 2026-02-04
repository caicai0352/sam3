#!/usr/bin/env python3
"""MoE inference using split ONNX weights (backbone + text + expert heads)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np
import onnxruntime as ort

from sam3.model.tokenizer_ve import SimpleTokenizer

sys.path.append(os.path.dirname(__file__))
from onnx_prepost import PreprocessConfig, preprocess_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MoE inference with split ONNX weights (backbone/text/experts)."
    )
    parser.add_argument("--image", action="append", required=True)
    parser.add_argument("--prompt", action="append", required=True)
    parser.add_argument("--category", action="append", required=True)
    parser.add_argument("--backbone-onnx", required=True)
    parser.add_argument("--text-onnx", required=True)
    parser.add_argument("--experts-json", required=True, help="Map category->expert onnx path.")
    parser.add_argument("--bpe-path", required=True, help="BPE vocab path for tokenizer.")
    parser.add_argument("--config", required=True, help="JSON with preprocessing settings.")
    parser.add_argument("--output", default="onnx_moe_output.json")
    return parser.parse_args()


def load_json(path: str) -> Dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing JSON: {path}")
    with open(path, "r") as handle:
        return json.load(handle)


def validate_prompt_category_pairs(prompts: Sequence[str], categories: Sequence[str]) -> None:
    if len(prompts) != len(categories):
        raise ValueError(
            "Each --prompt must have a matching --category. "
            f"Got {len(prompts)} prompts and {len(categories)} categories."
        )


def run_text_onnx(
    session: ort.InferenceSession, input_ids: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    input_name = session.get_inputs()[0].name
    output_names = [out.name for out in session.get_outputs()]
    outputs = session.run(output_names, {input_name: input_ids})
    out_map = {name: value for name, value in zip(output_names, outputs)}
    return (
        out_map["text_attention_mask"],
        out_map["text_memory"],
        out_map["text_embeds"],
    )


def main() -> None:
    args = parse_args()
    validate_prompt_category_pairs(args.prompt, args.category)

    cfg = load_json(args.config)
    preprocess_cfg = PreprocessConfig(
        resolution=cfg.get("resolution", 1008),
        mean=tuple(cfg.get("mean", [0.5, 0.5, 0.5])),
        std=tuple(cfg.get("std", [0.5, 0.5, 0.5])),
    )

    expert_map = load_json(args.experts_json)

    backbone_sess = ort.InferenceSession(args.backbone_onnx, providers=["CPUExecutionProvider"])
    text_sess = ort.InferenceSession(args.text_onnx, providers=["CPUExecutionProvider"])
    expert_sessions: Dict[str, ort.InferenceSession] = {}
    for key, path in expert_map.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expert ONNX not found: {path}")
        expert_sessions[key] = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    tokenizer = SimpleTokenizer(args.bpe_path)

    results: List[Dict] = []
    for image_path in args.image:
        prep = preprocess_image(image_path, preprocess_cfg)
        image_tensor = prep["image"].astype(np.float32)

        backbone_input = backbone_sess.get_inputs()[0].name
        backbone_outputs = [out.name for out in backbone_sess.get_outputs()]
        fpn_outputs = backbone_sess.run(backbone_outputs, {backbone_input: image_tensor})
        fpn_map = {name: value for name, value in zip(backbone_outputs, fpn_outputs)}

        image_results = []
        for prompt, category in zip(args.prompt, args.category):
            if category not in expert_sessions:
                raise ValueError(f"Unknown category '{category}' in experts JSON.")
            input_ids = tokenizer([prompt]).cpu().numpy().astype(np.int64)
            text_attention_mask, text_memory, text_embeds = run_text_onnx(
                text_sess, input_ids
            )

            expert_sess = expert_sessions[category]
            expert_inputs = {
                "fpn_s0": fpn_map["fpn_s0"],
                "fpn_s1": fpn_map["fpn_s1"],
                "fpn_s2": fpn_map["fpn_s2"],
                "fpn_s3": fpn_map["fpn_s3"],
                "text_attention_mask": text_attention_mask,
                "text_memory": text_memory,
                "text_embeds": text_embeds,
            }
            output_names = [out.name for out in expert_sess.get_outputs()]
            pred_boxes, pred_masks, pred_logits = expert_sess.run(
                output_names, expert_inputs
            )

            image_results.append(
                {
                    "prompt": prompt,
                    "category": category,
                    "pred_boxes": pred_boxes.tolist(),
                    "pred_masks": pred_masks.tolist(),
                    "pred_logits": pred_logits.tolist(),
                }
            )

        results.append({"image": image_path, "results": image_results})

    with open(args.output, "w") as handle:
        json.dump({"images": results}, handle, indent=2)

    print(f"Saved outputs to {args.output}")


if __name__ == "__main__":
    main()
