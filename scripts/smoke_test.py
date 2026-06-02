#!/usr/bin/env python3
"""A3 SDXL + single ControlNet-Depth smoke test.

This script validates the local GPU diffusion path with synthetic 1024x1024 inputs.
It intentionally skips LoRA and loads only the depth ControlNet from the standard
2cn_depth_canny inference config.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from blast_pile_diffusion.data.sample_bundle import SampleBundle  # noqa: E402
from blast_pile_diffusion.inference.controlnet_runner import run_single  # noqa: E402
from blast_pile_diffusion.inference.pipeline_builder import build_pipeline, release_pipeline  # noqa: E402


DEFAULT_BASE_CONFIG = PROJECT_ROOT / "configs" / "base.yaml"
DEFAULT_INFERENCE_CONFIG = PROJECT_ROOT / "configs" / "inference" / "2cn_depth_canny.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "generated" / "smoke_test.png"
IMAGE_SIZE = 1024
BYTES_PER_GB = 1024**3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the A3 SDXL + ControlNet-Depth environment smoke test."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_INFERENCE_CONFIG,
        help=f"Inference config path. Default: {DEFAULT_INFERENCE_CONFIG}",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=DEFAULT_BASE_CONFIG,
        help=f"Base config path. Default: {DEFAULT_BASE_CONFIG}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Generated image path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument("--device", default="cuda", help="Torch device. Default: cuda")
    parser.add_argument("--seed", type=int, default=42, help="Generation seed. Default: 42")
    parser.add_argument(
        "--steps",
        type=int,
        default=4,
        help="Smoke-test inference steps. Default: 4",
    )
    parser.add_argument(
        "--max-vram-gb",
        type=float,
        default=24.0,
        help="Maximum allowed peak CUDA reserved/allocated memory. Default: 24",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=60.0,
        help="Maximum allowed inference time in seconds. Default: 60",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_cuda_device(args.device)

    print("== A3 SDXL + ControlNet-Depth smoke test ==")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Config       : {args.config}")
    print(f"Base config  : {args.base_config}")
    print(f"Output       : {args.output}")
    print(f"Device       : {args.device}")
    print(f"Steps        : {args.steps}")
    print("LoRA         : skipped")
    print("ControlNet   : depth only")
    print()

    bundle = make_synthetic_bundle(size=IMAGE_SIZE)

    reset_cuda_peak(args.device)
    pipe = None
    try:
        load_start = time.perf_counter()
        pipe, cfg = build_pipeline(
            inference_config_path=args.config,
            base_config_path=args.base_config,
            device=args.device,
            controlnet_names=("depth",),
            skip_lora=True,
        )
        load_seconds = time.perf_counter() - load_start
        cfg = apply_smoke_overrides(cfg, steps=args.steps)

        infer_start = time.perf_counter()
        output = run_single(pipe=pipe, bundle=bundle, config=cfg, seed=args.seed)
        synchronize_cuda(args.device)
        inference_seconds = time.perf_counter() - infer_start
        peak_vram_gb = peak_cuda_gb(args.device)
    finally:
        release_pipeline(pipe)

    output = ensure_rgb_uint8(output)
    verify_output(
        output=output,
        size=IMAGE_SIZE,
        peak_vram_gb=peak_vram_gb,
        max_vram_gb=args.max_vram_gb,
        inference_seconds=inference_seconds,
        max_seconds=args.max_seconds,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output).save(args.output)

    print("Smoke test passed.")
    print(f"Pipeline load : {load_seconds:.2f}s")
    print(f"Inference     : {inference_seconds:.2f}s")
    print(f"Peak VRAM     : {peak_vram_gb:.2f} GB")
    print(f"Saved image   : {args.output}")
    return 0


def make_synthetic_bundle(size: int) -> SampleBundle:
    rgb = make_gradient_rgb(size)
    depth = make_linear_depth(size)

    normal = np.zeros((size, size, 3), dtype=np.float32)
    normal[..., 2] = 1.0
    mask = np.zeros((size, size), dtype=np.int32)

    return SampleBundle(
        scene_id="smoke",
        cam_id="synthetic",
        rgb=rgb,
        depth=depth,
        normal=normal,
        mask=mask,
        meta={"purpose": "a3_sdxl_controlnet_depth_smoke_test"},
        depth_cn=encode_depth_for_controlnet(depth),
    )


def make_gradient_rgb(size: int) -> np.ndarray:
    axis = np.linspace(0, 255, size, dtype=np.float32)
    x, y = np.meshgrid(axis, axis)
    rgb = np.stack(
        [
            x,
            y,
            0.5 * x + 0.5 * y,
        ],
        axis=-1,
    )
    return rgb.round().clip(0, 255).astype(np.uint8)


def make_linear_depth(size: int, near: float = 0.5, far: float = 100.0) -> np.ndarray:
    depth_row = np.linspace(near, far, size, dtype=np.float32)
    return np.repeat(depth_row[None, :], size, axis=0)


def encode_depth_for_controlnet(depth: np.ndarray) -> np.ndarray:
    finite = np.isfinite(depth)
    if not finite.any():
        raise ValueError("Synthetic depth contains no finite values")

    d_min = float(depth[finite].min())
    d_max = float(depth[finite].max())
    if d_max <= d_min:
        raise ValueError("Synthetic depth must have non-zero range")

    normalized = (depth - d_min) / (d_max - d_min)
    near_bright = (1.0 - normalized).clip(0.0, 1.0)
    gray = (near_bright * 255).round().astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=-1)


def apply_smoke_overrides(cfg: dict, steps: int) -> dict:
    if steps <= 0:
        raise ValueError("--steps must be positive")

    cfg = {**cfg}
    cfg["inference"] = {**cfg["inference"], "num_inference_steps": steps}
    return cfg


def ensure_rgb_uint8(output: np.ndarray) -> np.ndarray:
    if output.ndim != 3 or output.shape[2] < 3:
        raise ValueError(f"Expected RGB-like output, got shape {output.shape}")
    output = output[..., :3]
    if output.dtype != np.uint8:
        output = np.clip(output, 0, 255).astype(np.uint8)
    return output


def verify_output(
    output: np.ndarray,
    size: int,
    peak_vram_gb: float,
    max_vram_gb: float,
    inference_seconds: float,
    max_seconds: float,
) -> None:
    expected_shape = (size, size, 3)
    if output.shape != expected_shape:
        raise AssertionError(f"Output shape {output.shape} != {expected_shape}")
    if np.all(output <= 2):
        raise AssertionError("Output image is all-black or nearly all-black")
    if np.all(output >= 253):
        raise AssertionError("Output image is all-white or nearly all-white")
    if float(output.std()) < 1.0:
        raise AssertionError("Output image appears nearly solid")
    if peak_vram_gb > max_vram_gb:
        raise AssertionError(f"Peak VRAM {peak_vram_gb:.2f} GB > {max_vram_gb:.2f} GB")
    if inference_seconds > max_seconds:
        raise AssertionError(f"Inference time {inference_seconds:.2f}s > {max_seconds:.2f}s")


def ensure_cuda_device(device: str) -> None:
    if not device.startswith("cuda"):
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; A3 is a GPU pipeline validation script.")
    cuda_device = torch.device(device)
    if cuda_device.index is not None:
        torch.cuda.set_device(cuda_device)


def reset_cuda_peak(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(torch.device(device))


def synchronize_cuda(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(device))


def peak_cuda_gb(device: str) -> float:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return 0.0
    cuda_device = torch.device(device)
    allocated = torch.cuda.max_memory_allocated(cuda_device)
    reserved = torch.cuda.max_memory_reserved(cuda_device)
    return max(allocated, reserved) / BYTES_PER_GB


if __name__ == "__main__":
    raise SystemExit(main())
