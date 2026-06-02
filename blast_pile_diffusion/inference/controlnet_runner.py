"""单样本 ControlNet sim2real 推理 — 对应技术路线 §4.4.3 / §7.2.4。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from blast_pile_diffusion.data.sample_bundle import SampleBundle
from blast_pile_diffusion.inference.prompt_bank import get_negative_prompt, random_prompt
from blast_pile_diffusion.utils.vis import make_comparison_grid


def numpy_to_pil(arr: np.ndarray) -> Image.Image:
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        arr_float = arr.astype(np.float32)
        finite = np.isfinite(arr_float)
        if finite.any() and arr_float[finite].min() >= 0 and arr_float[finite].max() <= 1:
            arr_float = arr_float * 255
        arr = np.nan_to_num(arr_float, nan=0.0, posinf=255.0, neginf=0.0)
        arr = arr.clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def run_single(
    pipe: Any,
    bundle: SampleBundle,
    config: dict,
    seed: int = 42,
    prompt: str | None = None,
    negative_prompt: str | None = None,
) -> np.ndarray:
    """
    对单个预处理完毕的 SampleBundle 执行 ControlNet 推理。

    Args:
        pipe: build_pipeline() 返回的 pipeline
        bundle: 已预处理的 SampleBundle（只要求当前配置用到的 ControlNet 输入已填充）
        config: 推理配置 dict
        seed: 随机种子
        prompt: 可选固定 prompt；默认按 sample_key 和 seed 从 prompt_bank 稳定生成。
        negative_prompt: 可选固定 negative prompt；默认读取配置或 prompt_bank。

    Returns:
        (H, W, 3) uint8 RGB 生成图
    """
    cn_configs = config["controlnets"]
    validate_control_inputs(bundle, cn_configs)
    control_images = []
    conditioning_scales = []

    for cn_cfg in cn_configs:
        name = cn_cfg["name"]
        if name == "depth":
            control_images.append(numpy_to_pil(bundle.depth_cn))
        elif name == "canny":
            canny_3ch = np.stack([bundle.canny] * 3, axis=-1)
            control_images.append(numpy_to_pil(canny_3ch))
        elif name == "normal":
            if bundle.normal_cn is None:
                raise ValueError("配置了 normal ControlNet 但 bundle 没有 normal_cn")
            control_images.append(numpy_to_pil(bundle.normal_cn))
        else:
            raise ValueError(f"未知 ControlNet 类型: {name}")
        conditioning_scales.append(cn_cfg["scale"])

    inf_cfg = config["inference"]
    prompt_text = prompt or config.get("prompt") or random_prompt(seed=seed, salt=bundle.sample_key)
    negative_prompt_text = negative_prompt or config.get("negative_prompt", get_negative_prompt())

    result = pipe(
        prompt=prompt_text,
        negative_prompt=negative_prompt_text,
        image=numpy_to_pil(bundle.rgb),
        control_image=control_images,
        controlnet_conditioning_scale=conditioning_scales,
        strength=inf_cfg["strength"],
        num_inference_steps=inf_cfg["num_inference_steps"],
        guidance_scale=inf_cfg["guidance_scale"],
        generator=torch.Generator(get_generator_device(pipe)).manual_seed(seed),
    ).images[0]

    generated = ensure_rgb_uint8(np.array(result))
    anomaly = detect_image_anomaly(generated, config)
    if anomaly:
        raise ValueError(f"生成图异常: {anomaly}")
    return generated


def validate_control_inputs(bundle: SampleBundle, cn_configs: list[dict[str, Any]]) -> None:
    if not cn_configs:
        raise ValueError("至少需要配置一路 ControlNet")

    for cn_cfg in cn_configs:
        name = cn_cfg["name"]
        if name == "depth" and bundle.depth_cn is None:
            raise ValueError(f"SampleBundle {bundle.sample_key} 缺少 depth_cn")
        if name == "canny" and bundle.canny is None:
            raise ValueError(f"SampleBundle {bundle.sample_key} 缺少 canny")
        if name == "normal" and bundle.normal_cn is None:
            raise ValueError(f"SampleBundle {bundle.sample_key} 缺少 normal_cn")


def get_generator_device(pipe: Any) -> str:
    execution_device = getattr(pipe, "_execution_device", None)
    if isinstance(execution_device, torch.device):
        if execution_device.type == "cuda":
            return str(execution_device)
        return execution_device.type
    return "cuda" if torch.cuda.is_available() else "cpu"


def ensure_rgb_uint8(image: np.ndarray) -> np.ndarray:
    """Normalize pipeline output to an RGB uint8 array."""
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"Expected RGB-like output, got shape {image.shape}")
    image = image[..., :3]
    if image.dtype != np.uint8:
        image = np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=255.0, neginf=0.0)
        if image.max(initial=0) <= 1.0 and image.min(initial=0) >= 0.0:
            image = image * 255.0
        image = image.clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def detect_image_anomaly(image: np.ndarray, config: dict | None = None) -> str | None:
    """
    Return a human-readable anomaly reason for all-black/all-white/near-solid images.

    The thresholds are intentionally conservative: this catches failed VAE/model calls
    without judging normal low-contrast blast-pile images.
    """
    image = ensure_rgb_uint8(image)
    validation_cfg = (
        (config or {}).get("inference", {}).get("image_validation", {})
        if isinstance(config, dict)
        else {}
    )
    if validation_cfg.get("enabled", True) is False:
        return None

    black_threshold = int(validation_cfg.get("black_threshold", 2))
    white_threshold = int(validation_cfg.get("white_threshold", 253))
    min_std = float(validation_cfg.get("min_std", 1.0))

    if np.all(image <= black_threshold):
        return "all_black"
    if np.all(image >= white_threshold):
        return "all_white"
    if float(image.std()) < min_std:
        return f"near_solid_std_lt_{min_std:g}"
    return None


def save_single_result(
    result_dir: Path,
    bundle: SampleBundle,
    generated_rgb: np.ndarray,
    config: dict,
    seed: int,
    prompt: str | None = None,
    negative_prompt: str | None = None,
    elapsed_seconds: float | None = None,
) -> dict[str, Any]:
    """
    Save one D1/D3 result directory using the project output contract.

    Files:
      - generated.png
      - comparison.png (Unity RGB | depth | canny | generated)
      - meta.json
    """
    generated_rgb = ensure_rgb_uint8(generated_rgb)
    anomaly = detect_image_anomaly(generated_rgb, config)
    if anomaly:
        raise ValueError(f"生成图异常: {anomaly}")

    result_dir.mkdir(parents=True, exist_ok=True)
    generated_path = result_dir / "generated.png"
    cv2.imwrite(str(generated_path), cv2.cvtColor(generated_rgb, cv2.COLOR_RGB2BGR))

    output_cfg = config.get("inference", {}).get("output", {})
    save_comparison = output_cfg.get("save_comparison", True)
    comparison_path = result_dir / "comparison.png"
    if save_comparison:
        comparison = make_four_panel_comparison(
            bundle,
            generated_rgb,
            max_width=int(output_cfg.get("comparison_width", 384)),
        )
        cv2.imwrite(str(comparison_path), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))

    prompt_text = prompt or config.get("prompt") or random_prompt(seed=seed, salt=bundle.sample_key)
    negative_prompt_text = negative_prompt or config.get("negative_prompt", get_negative_prompt())
    meta = build_result_metadata(
        bundle=bundle,
        config=config,
        seed=seed,
        prompt=prompt_text,
        negative_prompt=negative_prompt_text,
        elapsed_seconds=elapsed_seconds,
        files={
            "generated": generated_path.name,
            "comparison": comparison_path.name if save_comparison else None,
        },
    )
    with open(result_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta


def make_four_panel_comparison(
    bundle: SampleBundle,
    generated_rgb: np.ndarray,
    max_width: int = 384,
) -> np.ndarray:
    """Build Unity RGB | depth ControlNet | Canny edges | generated RGB comparison."""
    depth_vis = bundle.depth_cn
    if depth_vis is None:
        depth_vis = _depth_to_uint8_rgb(bundle.depth)
    canny_vis = bundle.canny if bundle.canny is not None else np.zeros(bundle.rgb.shape[:2], np.uint8)

    return make_comparison_grid(
        [
            ensure_rgb_uint8(bundle.rgb),
            ensure_rgb_uint8(depth_vis),
            canny_vis,
            ensure_rgb_uint8(generated_rgb),
        ],
        labels=["Unity RGB", "Depth", "Canny", "Generated"],
        max_width=max_width,
    )


def build_result_metadata(
    bundle: SampleBundle,
    config: dict,
    seed: int,
    prompt: str,
    negative_prompt: str,
    elapsed_seconds: float | None,
    files: dict[str, str | None],
) -> dict[str, Any]:
    """Create stable metadata for one generated sample."""
    return {
        "sample_key": bundle.sample_key,
        "scene_id": bundle.scene_id,
        "cam_id": bundle.cam_id,
        "seed": seed,
        "width": int(bundle.width),
        "height": int(bundle.height),
        "num_instances": bundle.num_instances,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "controlnets": [
            {"name": cn_cfg["name"], "scale": cn_cfg.get("scale")}
            for cn_cfg in config.get("controlnets", [])
        ],
        "inference": {
            key: config.get("inference", {}).get(key)
            for key in ("strength", "num_inference_steps", "guidance_scale")
        },
        "elapsed_seconds": elapsed_seconds,
        "files": files,
    }


def _depth_to_uint8_rgb(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(depth)
    if not finite.any():
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    d_min = float(depth[finite].min())
    d_max = float(depth[finite].max())
    if d_max <= d_min:
        gray = np.zeros(depth.shape, dtype=np.uint8)
    else:
        normalized = (depth - d_min) / (d_max - d_min)
        gray = ((1.0 - normalized).clip(0, 1) * 255).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=-1)
