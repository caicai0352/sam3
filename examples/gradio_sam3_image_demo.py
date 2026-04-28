#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Gradio demo: baseline SAM3 vs MoE-style dual-model SAM3 comparison."""

import argparse
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import torch
from PIL import Image

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


@dataclass(frozen=True)
class Detection:
    prompt: str
    score: float
    box_xyxy: List[float]
    mask_area: int


def _parse_prompt_list(text: str) -> List[str]:
    prompts = [p.strip() for p in text.split(",")]
    return [p for p in prompts if p]


def _render_overlay(
    image: np.ndarray,
    masks: List[np.ndarray],
    boxes: List[np.ndarray],
    labels: List[str],
) -> np.ndarray:
    vis = image.astype(np.float32).copy()
    rng = np.random.default_rng(seed=42)

    for mask, box, label in zip(masks, boxes, labels):
        color = rng.integers(0, 255, size=(3,), dtype=np.uint8).astype(np.float32)
        mask_bool = mask.astype(bool)
        vis[mask_bool] = vis[mask_bool] * 0.6 + color * 0.4

        x0, y0, x1, y1 = box.astype(int).tolist()
        x0 = max(0, min(x0, vis.shape[1] - 1))
        y0 = max(0, min(y0, vis.shape[0] - 1))
        x1 = max(0, min(x1, vis.shape[1] - 1))
        y1 = max(0, min(y1, vis.shape[0] - 1))

        vis[y0 : y0 + 2, x0:x1] = color
        vis[y1 - 2 : y1, x0:x1] = color
        vis[y0:y1, x0 : x0 + 2] = color
        vis[y0:y1, x1 - 2 : x1] = color

        # tiny label patch for readability
        label_width = min(12 * max(len(label), 1), vis.shape[1] - x0)
        label_height = min(18, vis.shape[0] - y0)
        vis[y0 : y0 + label_height, x0 : x0 + label_width] = (
            vis[y0 : y0 + label_height, x0 : x0 + label_width] * 0.4 + color * 0.6
        )

    return np.clip(vis, 0, 255).astype(np.uint8)


@lru_cache(maxsize=8)
def _load_processor(
    checkpoint_path: str,
    device: str,
    confidence_threshold: float,
) -> Sam3Processor:
    model = build_sam3_image_model(
        checkpoint_path=checkpoint_path,
        load_from_HF=False,
        device=device,
        eval_mode=True,
    )
    return Sam3Processor(model=model, device=device, confidence_threshold=confidence_threshold)


def _run_model_for_prompts(
    processor: Sam3Processor,
    image: Image.Image,
    prompts: List[str],
) -> Tuple[List[np.ndarray], List[np.ndarray], List[Detection]]:
    state = processor.set_image(image)
    all_masks: List[np.ndarray] = []
    all_boxes: List[np.ndarray] = []
    detections: List[Detection] = []

    for prompt in prompts:
        state = processor.set_text_prompt(prompt=prompt, state=state)
        masks = state.get("masks")
        boxes = state.get("boxes")
        scores = state.get("scores")

        if masks is None or masks.numel() == 0:
            processor.reset_all_prompts(state)
            continue

        masks_np = masks.detach().cpu().numpy().squeeze(1)
        boxes_np = boxes.detach().cpu().numpy()
        scores_np = scores.detach().cpu().numpy()

        for i in range(len(scores_np)):
            all_masks.append(masks_np[i])
            all_boxes.append(boxes_np[i])
            detections.append(
                Detection(
                    prompt=prompt,
                    score=float(scores_np[i]),
                    box_xyxy=[float(v) for v in boxes_np[i].tolist()],
                    mask_area=int(masks_np[i].sum()),
                )
            )

        processor.reset_all_prompts(state)

    return all_masks, all_boxes, detections


def _predict(
    image: Optional[Image.Image],
    prompts_text: str,
    moe_prompts_text: str,
    checkpoint_path_1: str,
    checkpoint_path_2: str,
    device: str,
    confidence_threshold: float,
):
    if image is None:
        return None, None, [], "请先上传图像。", ""

    prompts = _parse_prompt_list(prompts_text)
    if not prompts:
        return None, None, [], "请输入至少一个类别（逗号分隔）。", ""

    moe_prompts = set(_parse_prompt_list(moe_prompts_text))

    processor_1 = _load_processor(checkpoint_path_1, device, confidence_threshold)
    processor_2 = _load_processor(checkpoint_path_2, device, confidence_threshold)

    with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=device == "cuda"):
        t0 = time.perf_counter()
        masks_base, boxes_base, det_base = _run_model_for_prompts(processor_1, image, prompts)
        t_base = time.perf_counter() - t0

        t1 = time.perf_counter()
        state1 = processor_1.set_image(image)
        state2 = processor_2.set_image(image)
        masks_moe: List[np.ndarray] = []
        boxes_moe: List[np.ndarray] = []
        det_moe: List[Detection] = []

        for prompt in prompts:
            if prompt in moe_prompts:
                state2 = processor_2.set_text_prompt(prompt=prompt, state=state2)
                cur_state = state2
                cur_processor = processor_2
                source_model = "model2"
            else:
                state1 = processor_1.set_text_prompt(prompt=prompt, state=state1)
                cur_state = state1
                cur_processor = processor_1
                source_model = "model1"

            masks = cur_state.get("masks")
            boxes = cur_state.get("boxes")
            scores = cur_state.get("scores")
            if masks is not None and masks.numel() > 0:
                masks_np = masks.detach().cpu().numpy().squeeze(1)
                boxes_np = boxes.detach().cpu().numpy()
                scores_np = scores.detach().cpu().numpy()
                for i in range(len(scores_np)):
                    masks_moe.append(masks_np[i])
                    boxes_moe.append(boxes_np[i])
                    det_moe.append(
                        Detection(
                            prompt=f"{prompt} ({source_model})",
                            score=float(scores_np[i]),
                            box_xyxy=[float(v) for v in boxes_np[i].tolist()],
                            mask_area=int(masks_np[i].sum()),
                        )
                    )

            cur_processor.reset_all_prompts(cur_state)

        t_moe = time.perf_counter() - t1

    image_np = np.array(image.convert("RGB"))
    vis_base = (
        Image.fromarray(_render_overlay(image_np, masks_base, boxes_base, [d.prompt for d in det_base]))
        if det_base
        else image
    )
    vis_moe = (
        Image.fromarray(_render_overlay(image_np, masks_moe, boxes_moe, [d.prompt for d in det_moe]))
        if det_moe
        else image
    )

    rows = [
        {
            "mode": "baseline(model1)",
            "prompt": d.prompt,
            "score": round(d.score, 4),
            "mask_area": d.mask_area,
            "box_xyxy": d.box_xyxy,
        }
        for d in det_base
    ]
    rows.extend(
        {
            "mode": "moe(model1+model2)",
            "prompt": d.prompt,
            "score": round(d.score, 4),
            "mask_area": d.mask_area,
            "box_xyxy": d.box_xyxy,
        }
        for d in det_moe
    )

    time_text = (
        f"Baseline(model1) 推理耗时: {t_base * 1000:.1f} ms\n"
        f"MoE(model1+model2) 推理耗时: {t_moe * 1000:.1f} ms"
    )
    summary = (
        f"Baseline 检测数: {len(det_base)} | "
        f"MoE 检测数: {len(det_moe)} | "
        f"MoE 路由到 model2 的类别: {', '.join(sorted(moe_prompts)) if moe_prompts else '无'}"
    )

    return vis_base, vis_moe, rows, time_text, summary


def build_demo(
    checkpoint_path_1: str,
    checkpoint_path_2: str,
    device: str,
    confidence_threshold: float,
) -> gr.Blocks:
    with gr.Blocks(title="SAM3 双模型对比（Baseline vs MoE）") as demo:
        gr.Markdown(
            "# SAM3 双模型可视化对比\n"
            "- 左侧: 原模型(model1)对所有类别推理\n"
            "- 右侧: MoE 路由(model1+model2)，你指定的类别走 model2，其余走 model1"
        )
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(label="上传图像", type="pil")
                prompts_input = gr.Textbox(
                    label="检测类别（逗号分隔）",
                    value="a person who is fishing,tree,boat",
                )
                moe_prompts_input = gr.Textbox(
                    label="走 model2 的类别（逗号分隔）",
                    value="a person who is fishing",
                )
                run_button = gr.Button("运行对比推理", variant="primary")
            with gr.Column(scale=1):
                time_output = gr.Textbox(label="推理耗时", interactive=False)
                summary_output = gr.Textbox(label="结果摘要", interactive=False)

        with gr.Row():
            baseline_image = gr.Image(label="Baseline: model1")
            moe_image = gr.Image(label="MoE: model1 + model2")

        table_output = gr.Dataframe(
            headers=["mode", "prompt", "score", "mask_area", "box_xyxy"],
            datatype=["str", "str", "number", "number", "str"],
            label="检测结果明细",
        )

        run_button.click(
            fn=lambda image, prompts, moe_prompts: _predict(
                image=image,
                prompts_text=prompts,
                moe_prompts_text=moe_prompts,
                checkpoint_path_1=checkpoint_path_1,
                checkpoint_path_2=checkpoint_path_2,
                device=device,
                confidence_threshold=confidence_threshold,
            ),
            inputs=[image_input, prompts_input, moe_prompts_input],
            outputs=[baseline_image, moe_image, table_output, time_output, summary_output],
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM3 Gradio dual-model comparison demo")
    parser.add_argument("--host", default="0.0.0.0", help="Gradio host")
    parser.add_argument("--port", type=int, default=7860, help="Gradio port")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Model running device",
    )
    parser.add_argument("--checkpoint-path-1", required=True, help="Baseline model checkpoint")
    parser.add_argument("--checkpoint-path-2", required=True, help="MoE secondary model checkpoint")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Detection confidence threshold",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    demo = build_demo(
        checkpoint_path_1=args.checkpoint_path_1,
        checkpoint_path_2=args.checkpoint_path_2,
        device=args.device,
        confidence_threshold=args.confidence_threshold,
    )
    demo.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
