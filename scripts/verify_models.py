#!/usr/bin/env python3
"""Download and verify pretrained diffusion models in the Hugging Face cache.

This script implements task A2 from docs/task_decomposition.md:

1. Read model IDs from configs/base.yaml and configs/inference/2cn_depth_canny.yaml.
2. Download SDXL base, VAE, ControlNet-depth, and ControlNet-canny via diffusers
   from_pretrained(..., torch_dtype=torch.float16).
3. Delete each loaded object immediately to release memory/VRAM.
4. Re-load each model with local_files_only=True to verify cache integrity.
5. Print download/verification status and on-disk cache size for each model.

HF_ENDPOINT is intentionally not overridden here. If set in the environment,
huggingface_hub/diffusers will use it for mirror sites, e.g.:

    HF_ENDPOINT=https://hf-mirror.com python scripts/verify_models.py
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import yaml
from diffusers import AutoencoderKL, ControlNetModel, StableDiffusionXLPipeline
from huggingface_hub.constants import HF_HUB_CACHE


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_CONFIG = PROJECT_ROOT / "configs" / "base.yaml"
DEFAULT_INFERENCE_CONFIG = PROJECT_ROOT / "configs" / "inference" / "2cn_depth_canny.yaml"


@dataclass(frozen=True)
class ModelSpec:
    label: str
    model_id: str
    loader: Callable[..., Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download SDXL/VAE/ControlNet models and verify local HF cache."
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=DEFAULT_BASE_CONFIG,
        help=f"Path to base config. Default: {DEFAULT_BASE_CONFIG}",
    )
    parser.add_argument(
        "--inference-config",
        type=Path,
        default=DEFAULT_INFERENCE_CONFIG,
        help=f"Path to inference config. Default: {DEFAULT_INFERENCE_CONFIG}",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face cache directory. Defaults to HF_HUB_CACHE.",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Skip download step and only verify existing local cache.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cache_dir = args.cache_dir.expanduser() if args.cache_dir else Path(HF_HUB_CACHE).expanduser()
    specs = read_model_specs(args.base_config, args.inference_config)

    print("== Hugging Face model cache verification ==")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"HF_ENDPOINT  : {os.environ.get('HF_ENDPOINT', '(default Hugging Face endpoint)')}")
    print(f"Cache dir    : {cache_dir}")
    print(f"Model count  : {len(specs)}")
    print()

    try:
        for spec in specs:
            print(f"--- {spec.label}: {spec.model_id} ---")
            if args.local_only:
                print("Download     : skipped (--local-only)")
            else:
                load_model(spec, cache_dir=cache_dir, local_files_only=False)
                print("Download     : ok")

            load_model(spec, cache_dir=cache_dir, local_files_only=True)
            print("Local verify : ok")
            print(f"Disk size    : {format_bytes(model_cache_size(cache_dir, spec.model_id))}")
            print()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Partial downloads remain in the Hugging Face cache.")
        return 130
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        print(
            "Hint: check network/HF_TOKEN/HF_ENDPOINT, then re-run this script; "
            "Hugging Face downloads resume from the cache.",
            file=sys.stderr,
        )
        return 1

    print("All required models are present and loadable from the local Hugging Face cache.")
    return 0


def read_model_specs(base_config: Path, inference_config: Path) -> list[ModelSpec]:
    base = load_yaml(base_config)
    inference = load_yaml(inference_config)

    try:
        sdxl_base = base["sdxl"]["base_model"]
        vae = base["sdxl"]["vae"]
        controlnets = inference["controlnets"]
    except KeyError as exc:
        raise KeyError(f"Missing required config key: {exc}") from exc

    specs = [
        ModelSpec("sdxl-base", str(sdxl_base), StableDiffusionXLPipeline),
        ModelSpec("vae", str(vae), AutoencoderKL),
    ]

    for item in controlnets:
        name = item.get("name")
        model_id = item.get("model")
        if not name or not model_id:
            raise ValueError(f"Invalid controlnet config item: {item}")
        if name not in {"depth", "canny"}:
            continue
        specs.append(ModelSpec(f"controlnet-{name}", str(model_id), ControlNetModel))

    required_labels = {"sdxl-base", "vae", "controlnet-depth", "controlnet-canny"}
    found_labels = {spec.label for spec in specs}
    missing = sorted(required_labels - found_labels)
    if missing:
        raise ValueError(f"Missing required model specs: {', '.join(missing)}")

    return specs


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping: {path}")
    return data


def load_model(spec: ModelSpec, cache_dir: Path, local_files_only: bool) -> None:
    model = None
    try:
        model = spec.loader.from_pretrained(
            spec.model_id,
            torch_dtype=torch.float16,
            cache_dir=str(cache_dir),
            local_files_only=local_files_only,
            low_cpu_mem_usage=True,
        )
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def model_cache_size(cache_dir: Path, model_id: str) -> int:
    repo_cache_dir = cache_dir / f"models--{model_id.replace('/', '--')}"
    if not repo_cache_dir.exists():
        return 0
    return directory_size(repo_cache_dir)


def directory_size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        if item.is_symlink():
            continue
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                pass
    return total


def format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{num_bytes} B"


if __name__ == "__main__":
    raise SystemExit(main())
