#!/usr/bin/env python3
"""SAM3 MoE continual-learning inference script with a shared ViT backbone.

This script assumes the ViT (vision encoder) is frozen and identical across
experts. It computes the image backbone features once, then routes each request
(prompt/category) to a single expert model for text encoding + DETR + head.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
from PIL import Image

from sam3.agent.viz import visualize
from sam3.model import box_ops
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.train.masks_ops import rle_encode


@dataclass
class ExpertSpec:
    name: str
    checkpoint_path: str


class ExpertRouter:
    def __init__(self, mapping: Dict[str, str], fallback: str):
        self.mapping = mapping
        self.fallback = fallback

    def route(self, category: str) -> str:
        return self.mapping.get(category, self.fallback)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SAM3 MoE inference with shared ViT backbone and expert routing.",
        epilog=(
            "Example:\n"
            "  python scripts/moe_continual_inference.py \\\n"
            "    --image assets/demo.jpg \\\n"
            "    --prompt \"red mug\" \\\n"
            "    --category kitchen \\\n"
            "    --prompt \"blue bowl\" \\\n"
            "    --category kitchen \\\n"
            "    --routing-json routing.json \\\n"
            "    --shared-backbone-ckpt /path/shared_backbone.pt \\\n"
            "    --sam3-ckpt /path/sam3_expert.pt \\\n"
            "    --business-ckpt /path/business_expert.pt \\\n"
            "    --longtext-ckpt /path/longtext_expert.pt \\\n"
            "    --largedata-ckpt /path/largedata_expert.pt\n"
            "\n"
            "routing.json:\n"
            "  {\n"
            "    \"kitchen\": \"business\",\n"
            "    \"documents\": \"longtext\",\n"
            "    \"openworld\": \"sam3\"\n"
            "  }\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--image",
        action="append",
        required=True,
        help="Path to input image. Repeat to batch multiple images.",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        required=True,
        help="Text prompt. Repeat to send multiple prompts per image.",
    )
    parser.add_argument(
        "--category",
        action="append",
        required=True,
        help="Category key for expert routing (repeat to align with --prompt).",
    )
    parser.add_argument(
        "--shared-backbone-ckpt",
        required=True,
        help="Checkpoint used to compute the shared ViT backbone output.",
    )
    parser.add_argument("--sam3-ckpt", required=True, help="SAM3 expert checkpoint.")
    parser.add_argument(
        "--business-ckpt", required=True, help="Business expert checkpoint."
    )
    parser.add_argument(
        "--longtext-ckpt", required=True, help="Long-text expert checkpoint."
    )
    parser.add_argument(
        "--largedata-ckpt", required=True, help="Large-data expert checkpoint."
    )
    parser.add_argument(
        "--routing-json",
        default=None,
        help="Optional JSON mapping category -> expert name.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for mask filtering.",
    )
    parser.add_argument(
        "--output",
        default="moe_inference_output.json",
        help="Path to save JSON outputs.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save visualization overlays for each image/prompt result.",
    )
    parser.add_argument(
        "--viz-output-dir",
        default="moe_viz_outputs",
        help="Directory to store visualization images.",
    )
    parser.add_argument(
        "--report-overhead",
        action="store_true",
        help="Report per-expert parameter/storage/compute overhead stats.",
    )
    return parser.parse_args()


def build_expert_specs(args: argparse.Namespace) -> Dict[str, ExpertSpec]:
    return {
        "sam3": ExpertSpec("sam3", args.sam3_ckpt),
        "business": ExpertSpec("business", args.business_ckpt),
        "longtext": ExpertSpec("longtext", args.longtext_ckpt),
        "largedata": ExpertSpec("largedata", args.largedata_ckpt),
    }


def load_image(path: str) -> Image.Image:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


def load_model(checkpoint_path: str, device: str) -> torch.nn.Module:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return build_sam3_image_model(
        device=device,
        eval_mode=True,
        checkpoint_path=checkpoint_path,
        load_from_HF=False,
    )


def _count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _estimate_storage_mb(param_count: int, dtype_bytes: int = 4) -> float:
    return param_count * dtype_bytes / (1024 * 1024)


def _estimate_expert_overhead(
    expert_model: torch.nn.Module, shared_backbone: torch.nn.Module
) -> Dict:
    shared_params = _count_params(shared_backbone)
    expert_total = _count_params(expert_model)
    extra_params = max(expert_total - shared_params, 0)
    return {
        "shared_backbone_params": shared_params,
        "expert_total_params": expert_total,
        "expert_extra_params": extra_params,
        "shared_backbone_storage_mb": _estimate_storage_mb(shared_params),
        "expert_total_storage_mb": _estimate_storage_mb(expert_total),
        "expert_extra_storage_mb": _estimate_storage_mb(extra_params),
        "compute_note": (
            "Extra compute scales with expert modules (text/encoder/decoder/head); "
            "shared backbone compute is amortized per image."
        ),
    }


def prepare_shared_state(
    shared_model: torch.nn.Module, image: Image.Image, device: str
) -> Dict:
    processor = Sam3Processor(shared_model, device=device)
    state = processor.set_image(image)
    return state


def run_expert_inference(
    shared_state: Dict,
    expert_model: torch.nn.Module,
    prompt: str,
    confidence_threshold: float,
    image_path: str,
) -> Dict:
    device = next(expert_model.parameters()).device
    processor = Sam3Processor(
        expert_model, device=str(device), confidence_threshold=confidence_threshold
    )

    state = {
        "original_height": shared_state["original_height"],
        "original_width": shared_state["original_width"],
        "backbone_out": copy.deepcopy(shared_state["backbone_out"]),
        "geometric_prompt": expert_model._get_dummy_prompt(),
    }

    text_outputs = expert_model.backbone.forward_text([prompt], device=device)
    state["backbone_out"].update(text_outputs)

    state = processor._forward_grounding(state)

    pred_boxes_xyxy = state["boxes"]
    pred_boxes_xyxy_norm = torch.stack(
        [
            pred_boxes_xyxy[:, 0] / state["original_width"],
            pred_boxes_xyxy[:, 1] / state["original_height"],
            pred_boxes_xyxy[:, 2] / state["original_width"],
            pred_boxes_xyxy[:, 3] / state["original_height"],
        ],
        dim=-1,
    )
    pred_boxes_xywh = box_ops.box_xyxy_to_xywh(pred_boxes_xyxy_norm).tolist()

    pred_masks = rle_encode(state["masks"].squeeze(1))
    pred_masks = [mask["counts"] for mask in pred_masks]

    outputs = {
        "original_image_path": image_path,
        "orig_img_h": state["original_height"],
        "orig_img_w": state["original_width"],
        "pred_boxes": pred_boxes_xywh,
        "pred_boxes_xyxy": pred_boxes_xyxy.tolist(),
        "pred_masks": pred_masks,
        "pred_scores": state["scores"].tolist(),
    }
    return outputs


def _sanitize_filename(value: str) -> str:
    return (
        value.replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )


def validate_prompt_category_pairs(
    prompts: Sequence[str], categories: Sequence[str]
) -> None:
    if len(prompts) != len(categories):
        raise ValueError(
            "Each --prompt must have a matching --category. "
            f"Got {len(prompts)} prompts and {len(categories)} categories."
        )


def main() -> None:
    args = parse_args()

    expert_specs = build_expert_specs(args)
    routing_map = {key: key for key in expert_specs}
    if args.routing_json:
        with open(args.routing_json, "r") as handle:
            routing_map.update(json.load(handle))

    router = ExpertRouter(mapping=routing_map, fallback="sam3")
    validate_prompt_category_pairs(args.prompt, args.category)

    shared_model = load_model(args.shared_backbone_ckpt, args.device)

    expert_models: Dict[str, torch.nn.Module] = {}
    for expert_key, spec in expert_specs.items():
        expert_models[expert_key] = load_model(spec.checkpoint_path, args.device)

    results: List[Dict] = []
    overhead_summary: Dict[str, Dict] = {}
    if args.report_overhead:
        for expert_key, model in expert_models.items():
            overhead_summary[expert_key] = _estimate_expert_overhead(
                model, shared_model.backbone
            )
    for image_path in args.image:
        image = load_image(image_path)
        shared_state = prepare_shared_state(shared_model, image, args.device)

        image_results = []
        for prompt, category in zip(args.prompt, args.category):
            expert_key = router.route(category)
            if expert_key not in expert_models:
                raise ValueError(
                    f"Unknown expert key '{expert_key}'. Available: {expert_specs}"
                )
            outputs = run_expert_inference(
                shared_state=shared_state,
                expert_model=expert_models[expert_key],
                prompt=prompt,
                confidence_threshold=args.confidence_threshold,
                image_path=image_path,
            )
            viz_path = None
            if args.visualize:
                os.makedirs(args.viz_output_dir, exist_ok=True)
                image_stem = _sanitize_filename(os.path.basename(image_path))
                prompt_stem = _sanitize_filename(prompt)
                category_stem = _sanitize_filename(category)
                viz_filename = (
                    f"{image_stem}_{category_stem}_{prompt_stem}_viz.png"
                )
                viz_path = os.path.join(args.viz_output_dir, viz_filename)
                viz_image = visualize(outputs)
                viz_image.save(viz_path)
            image_results.append(
                {
                    "prompt": prompt,
                    "category": category,
                    "expert": expert_key,
                    "outputs": outputs,
                    "viz_path": viz_path,
                }
            )

        results.append({"image": image_path, "results": image_results})

    payload = {"images": results}
    if args.report_overhead:
        payload["overhead"] = overhead_summary
    with open(args.output, "w") as handle:
        json.dump(payload, handle, indent=2)

    print(f"Saved outputs to {args.output}")


if __name__ == "__main__":
    main()
