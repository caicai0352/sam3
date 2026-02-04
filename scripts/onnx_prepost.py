#!/usr/bin/env python3
"""Pre/post-processing utilities for SAM3 ONNX inference."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


@dataclass
class PreprocessConfig:
    resolution: int = 1008
    mean: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    std: Tuple[float, float, float] = (0.5, 0.5, 0.5)


def preprocess_image(image_path: str, config: PreprocessConfig) -> Dict[str, Any]:
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    image = image.resize((config.resolution, config.resolution))
    image_np = np.asarray(image).astype(np.float32) / 255.0
    image_np = (image_np - np.array(config.mean)) / np.array(config.std)
    image_np = np.transpose(image_np, (2, 0, 1))[None, ...]
    return {
        "image": image_np,
        "orig_size": (orig_h, orig_w),
    }


def postprocess_outputs(
    outputs: Dict[str, np.ndarray],
    orig_size: Tuple[int, int],
    mask_threshold: float = 0.5,
    output_names: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    output_names = output_names or {}
    boxes_key = output_names.get("boxes", "boxes")
    scores_key = output_names.get("scores", "scores")
    masks_key = output_names.get("masks", "masks")

    result: Dict[str, Any] = {
        "orig_img_h": int(orig_size[0]),
        "orig_img_w": int(orig_size[1]),
    }

    if boxes_key in outputs:
        result["pred_boxes"] = outputs[boxes_key].tolist()
    if scores_key in outputs:
        result["pred_scores"] = outputs[scores_key].tolist()
    if masks_key in outputs:
        masks = outputs[masks_key]
        result["pred_masks"] = (masks > mask_threshold).astype(np.uint8).tolist()

    if not result.get("pred_boxes") and not result.get("pred_masks"):
        result["raw_outputs"] = {
            name: value.tolist() for name, value in outputs.items()
        }

    return result
