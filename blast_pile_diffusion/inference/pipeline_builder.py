"""构建 SDXL + 多路 ControlNet + LoRA 推理 pipeline，统一管理模型加载和显存。"""

from __future__ import annotations

import gc
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import yaml

try:
    from diffusers import AutoencoderKL, ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline
except ImportError:  # pragma: no cover - exercised only in minimal test environments.
    AutoencoderKL = None
    ControlNetModel = None
    StableDiffusionXLControlNetImg2ImgPipeline = None

SUPPORTED_CONTROLNETS = {"depth", "canny", "normal"}


def load_config(config_path: Path) -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_project_path(path: str | Path, project_root: Path) -> Path:
    """Resolve config paths relative to the project root."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return project_root / candidate


def resolve_hf_cache_dir(base_cfg: dict[str, Any], base_config_path: Path) -> Path | None:
    """Return the local Hugging Face cache if this deployment bundle includes one."""
    project_root = base_config_path.resolve().parent.parent
    paths_cfg = base_cfg.get("paths", {}) if isinstance(base_cfg.get("paths"), dict) else {}
    candidates: list[Path] = []
    configured_cache = paths_cfg.get("hf_cache")
    if configured_cache:
        candidates.append(resolve_project_path(configured_cache, project_root))
    candidates.append(project_root / "weights" / "huggingface" / "hub")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def pretrained_load_kwargs(base_cfg: dict[str, Any], base_config_path: Path) -> dict[str, str]:
    """Common kwargs for diffusers from_pretrained calls."""
    cache_dir = resolve_hf_cache_dir(base_cfg, base_config_path)
    return {"cache_dir": str(cache_dir)} if cache_dir is not None else {}


def build_pipeline(
    inference_config_path: Path,
    base_config_path: Path,
    device: str = "cuda",
    controlnet_names: Sequence[str] | None = None,
    skip_lora: bool = False,
) -> tuple[StableDiffusionXLControlNetImg2ImgPipeline, dict]:
    """
    根据配置构建完整的推理 pipeline。

    Args:
        inference_config_path: 推理配置路径。
        base_config_path: 全局配置路径。
        device: 目标设备。
        controlnet_names: 可选，仅加载指定名称的 ControlNet；默认加载配置中的全部。
        skip_lora: 为 True 时跳过 LoRA 加载，用于环境冒烟测试等场景。

    Returns:
        (pipeline, config_dict)
    """
    cfg = load_config(inference_config_path)
    base_cfg = load_config(base_config_path)
    validate_inference_config(cfg)

    project_root = base_config_path.resolve().parent.parent
    dtype = get_torch_dtype(base_cfg.get("dtype", "float16"))
    cfg = select_controlnet_configs(cfg, controlnet_names)

    lora_cfg = {} if skip_lora else cfg.get("lora", {})
    lora_path = None
    if lora_cfg.get("path"):
        lora_path = resolve_project_path(lora_cfg["path"], project_root)
    if lora_path and lora_cfg.get("required", False) and not lora_path.exists():
        raise FileNotFoundError(f"LoRA 权重不存在: {lora_path}")

    ensure_diffusers_available()
    load_kwargs = pretrained_load_kwargs(base_cfg, base_config_path)

    controlnets = []
    for cn_cfg in cfg["controlnets"]:
        cn = ControlNetModel.from_pretrained(cn_cfg["model"], torch_dtype=dtype, **load_kwargs)
        controlnets.append(cn)

    vae = AutoencoderKL.from_pretrained(base_cfg["sdxl"]["vae"], torch_dtype=dtype, **load_kwargs)

    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        base_cfg["sdxl"]["base_model"],
        controlnet=controlnets,
        vae=vae,
        torch_dtype=dtype,
        **load_kwargs,
    )

    if lora_path and lora_path.exists():
        pipe.load_lora_weights(str(lora_path), adapter_name="mine_style")
        pipe.set_adapters(["mine_style"], adapter_weights=[lora_cfg.get("weight", 1.0)])

    configure_memory_strategy(pipe, cfg, device)

    return pipe, cfg


def validate_inference_config(cfg: dict[str, Any]) -> None:
    """Validate the subset of inference config needed before loading models."""
    controlnets = cfg.get("controlnets")
    if not isinstance(controlnets, list) or not controlnets:
        raise ValueError("inference config 必须包含至少一路 controlnets")

    names: list[str] = []
    for index, cn_cfg in enumerate(controlnets):
        if not isinstance(cn_cfg, dict):
            raise ValueError(f"controlnets[{index}] 必须是 dict")
        name = cn_cfg.get("name")
        if name not in SUPPORTED_CONTROLNETS:
            raise ValueError(f"未知 ControlNet 类型: {name}")
        if not cn_cfg.get("model"):
            raise ValueError(f"controlnets[{index}] 缺少 model")
        scale = cn_cfg.get("scale")
        if not isinstance(scale, (int, float)):
            raise ValueError(f"controlnets[{index}] 缺少数值型 scale")
        names.append(name)

    if len(names) != len(set(names)):
        raise ValueError(f"ControlNet 名称重复: {names}")

    inference = cfg.get("inference")
    if not isinstance(inference, dict):
        raise ValueError("inference config 缺少 inference 段")
    for key in ("strength", "num_inference_steps", "guidance_scale"):
        if key not in inference:
            raise ValueError(f"inference config 缺少 inference.{key}")


def select_controlnet_configs(
    cfg: dict[str, Any],
    controlnet_names: Sequence[str] | None,
) -> dict[str, Any]:
    """Return a config copy with only selected ControlNets."""
    if controlnet_names is None:
        return cfg

    selected_names = set(controlnet_names)
    configured_names = {cn_cfg["name"] for cn_cfg in cfg["controlnets"]}
    unknown_names = selected_names - configured_names
    if unknown_names:
        raise ValueError(f"配置中不存在 ControlNet: {sorted(unknown_names)}")

    selected_controlnets = [
        cn_cfg for cn_cfg in cfg["controlnets"] if cn_cfg["name"] in selected_names
    ]
    if not selected_controlnets:
        raise ValueError("至少需要加载一路 ControlNet")
    return {**cfg, "controlnets": selected_controlnets}


def get_torch_dtype(dtype_name: str) -> torch.dtype:
    dtype = getattr(torch, dtype_name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"不支持的 torch dtype: {dtype_name}")
    return dtype


def ensure_diffusers_available() -> None:
    if (
        AutoencoderKL is None
        or ControlNetModel is None
        or StableDiffusionXLControlNetImg2ImgPipeline is None
    ):
        raise ImportError(
            "diffusers is required to build the SDXL ControlNet pipeline. "
            "Install project dependencies before running real inference."
        )


def configure_memory_strategy(pipe: Any, cfg: dict[str, Any], device: str) -> None:
    """
    Configure where model modules live.

    On CUDA, CPU offload is the default to reduce OOM risk during D3 batch runs.
    On CPU, or when explicitly disabled, the pipeline is moved to the requested device.
    """
    memory_cfg = cfg.get("inference", {}).get("memory", {})
    enable_offload = memory_cfg.get("enable_model_cpu_offload", True)
    if enable_offload and device.startswith("cuda") and hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
        return
    if hasattr(pipe, "to"):
        pipe.to(device)


def release_pipeline(pipe: Any | None) -> None:
    """Best-effort cleanup helper for long-running scripts/tests."""
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass
