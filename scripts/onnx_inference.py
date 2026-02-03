#!/usr/bin/env python3
"""Run ONNX inference with SAM3-style preprocessing and postprocessing."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict

import numpy as np
import onnxruntime as ort

sys.path.append(os.path.dirname(__file__))
from onnx_prepost import PreprocessConfig, postprocess_outputs, preprocess_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ONNX inference with SAM3 preprocessing and postprocessing."
    )
    parser.add_argument("--onnx", required=True, help="Path to ONNX model.")
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument(
        "--config",
        required=True,
        help=(
            "JSON config with preprocess and output mapping. Example:\n"
            "{\n"
            "  \"resolution\": 1008,\n"
            "  \"mean\": [0.5, 0.5, 0.5],\n"
            "  \"std\": [0.5, 0.5, 0.5],\n"
            "  \"output_names\": {\"boxes\": \"boxes\", \"scores\": \"scores\", \"masks\": \"masks\"},\n"
            "  \"mask_threshold\": 0.5\n"
            "}\n"
        ),
    )
    parser.add_argument(
        "--output",
        default="onnx_inference_output.json",
        help="Output JSON path.",
    )
    return parser.parse_args()


def load_config(path: str) -> Dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r") as handle:
        return json.load(handle)


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.onnx):
        raise FileNotFoundError(f"ONNX model not found: {args.onnx}")

    cfg = load_config(args.config)
    preprocess_cfg = PreprocessConfig(
        resolution=cfg.get("resolution", 1008),
        mean=tuple(cfg.get("mean", [0.5, 0.5, 0.5])),
        std=tuple(cfg.get("std", [0.5, 0.5, 0.5])),
    )

    prep = preprocess_image(args.image, preprocess_cfg)
    image_tensor = prep["image"].astype(np.float32)

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output_names = [out.name for out in sess.get_outputs()]

    raw_outputs = sess.run(output_names, {input_name: image_tensor})
    outputs = {name: value for name, value in zip(output_names, raw_outputs)}

    result = postprocess_outputs(
        outputs=outputs,
        orig_size=prep["orig_size"],
        mask_threshold=cfg.get("mask_threshold", 0.5),
        output_names=cfg.get("output_names"),
    )
    result["onnx_outputs"] = list(outputs.keys())

    with open(args.output, "w") as handle:
        json.dump(result, handle, indent=2)

    print(f"Saved outputs to {args.output}")


if __name__ == "__main__":
    main()
