"""预处理 Pipeline 入口：对 SampleBundle 执行全部预处理，一步到位。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from blast_pile_diffusion.data.sample_bundle import SampleBundle
from blast_pile_diffusion.preprocessing.canny_from_mask import mask_to_canny
from blast_pile_diffusion.preprocessing.depth_processor import (
    DepthProcessingConfig,
    depth_config_from_mapping,
    process_depth,
)
from blast_pile_diffusion.preprocessing.normal_processor import process_normal


REQUIRED_PREPROCESSED_FILES = {
    "rgb.png",
    "depth_cn.png",
    "canny_from_mask.png",
    "mask_instance.png",
    "meta.json",
}


def _validate_raw_bundle(bundle: SampleBundle) -> None:
    h, w = bundle.rgb.shape[:2]
    if bundle.rgb.ndim != 3 or bundle.rgb.shape[2] != 3:
        raise ValueError(f"{bundle.sample_key}: rgb 必须是 (H,W,3)")
    if bundle.rgb.dtype != np.uint8:
        raise ValueError(f"{bundle.sample_key}: rgb dtype 必须是 uint8")
    if bundle.depth.shape[:2] != (h, w):
        raise ValueError(f"{bundle.sample_key}: depth 尺寸与 rgb 不一致")
    if bundle.mask.shape[:2] != (h, w):
        raise ValueError(f"{bundle.sample_key}: mask 尺寸与 rgb 不一致")
    if bundle.normal.shape[:2] != (h, w) or bundle.normal.ndim != 3 or bundle.normal.shape[2] != 3:
        raise ValueError(f"{bundle.sample_key}: normal 必须是 (H,W,3)")


def _canny_config_from_mapping(cfg: Mapping[str, Any] | None) -> dict[str, int]:
    cfg = cfg or {}
    canny_cfg: Mapping[str, Any] = {}
    preprocessing = cfg.get("preprocessing") if isinstance(cfg, Mapping) else None
    if isinstance(preprocessing, Mapping) and isinstance(preprocessing.get("canny"), Mapping):
        canny_cfg = preprocessing["canny"]
    elif isinstance(cfg.get("canny"), Mapping):
        canny_cfg = cfg["canny"]
    return {
        "canny_low": int(canny_cfg.get("low", canny_cfg.get("canny_low", 50))),
        "canny_high": int(canny_cfg.get("high", canny_cfg.get("canny_high", 150))),
        "canny_dilate": int(canny_cfg.get("dilate_kernel", canny_cfg.get("canny_dilate", 2))),
    }


def preprocess_bundle(
    bundle: SampleBundle,
    depth_near_clip: float | None = None,
    depth_far_clip: float | None = None,
    canny_low: int = 50,
    canny_high: int = 150,
    canny_dilate: int = 2,
    normal_from_depth_if_missing: bool = True,
    depth_config: DepthProcessingConfig | Mapping[str, Any] | None = None,
) -> SampleBundle:
    """对 SampleBundle 执行全部预处理，填充 ControlNet 输入字段。"""
    _validate_raw_bundle(bundle)
    if isinstance(depth_config, DepthProcessingConfig):
        resolved_depth = depth_config
    elif isinstance(depth_config, Mapping):
        resolved_depth = depth_config_from_mapping(depth_config)
    else:
        resolved_depth = DepthProcessingConfig()
    effective_near = resolved_depth.near_clip if depth_near_clip is None else depth_near_clip
    effective_far = resolved_depth.far_clip if depth_far_clip is None else depth_far_clip

    bundle.depth_cn = process_depth(
        bundle.depth,
        near_clip=depth_near_clip,
        far_clip=depth_far_clip,
        config=depth_config,
    )

    bundle.canny = mask_to_canny(
        bundle.mask,
        canny_low=canny_low,
        canny_high=canny_high,
        dilate_kernel=canny_dilate,
    )

    if normal_from_depth_if_missing and _normal_is_missing(bundle.normal):
        bundle.normal_cn = process_normal(None, from_depth=True, depth=bundle.depth)
        normal_source = "depth_fallback"
    else:
        bundle.normal_cn = process_normal(bundle.normal)
        normal_source = "normal"
    bundle.meta = {
        **bundle.meta,
        "preprocessing": {
            "depth_near_clip": effective_near,
            "depth_far_clip": effective_far,
            "canny_low": canny_low,
            "canny_high": canny_high,
            "canny_dilate": canny_dilate,
            "normal_source": normal_source,
        },
    }

    return bundle


def _normal_is_missing(normal: np.ndarray) -> bool:
    """Unity reader uses zero normal arrays when the modality is unavailable."""
    if normal is None:
        return True
    normal_arr = np.asarray(normal)
    if normal_arr.size == 0 or not np.isfinite(normal_arr).any():
        return True
    finite = np.nan_to_num(normal_arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return bool(np.allclose(finite, 0.0))


def preprocess_bundle_from_config(
    bundle: SampleBundle,
    config: Mapping[str, Any],
) -> SampleBundle:
    """按配置 dict 预处理，供脚本和测试复用。"""
    depth_cfg = DepthProcessingConfig()
    preprocessing = config.get("preprocessing") if isinstance(config, Mapping) else None
    if isinstance(preprocessing, Mapping) and isinstance(preprocessing.get("depth"), Mapping):
        depth_values = preprocessing["depth"]
        depth_cfg = DepthProcessingConfig(
            near_clip=float(depth_values.get("near_clip", depth_cfg.near_clip)),
            far_clip=float(depth_values.get("far_clip", depth_cfg.far_clip)),
            fill_holes=bool(depth_values.get("fill_holes", depth_cfg.fill_holes)),
            hole_smooth_diameter=int(
                depth_values.get("hole_smooth_diameter", depth_cfg.hole_smooth_diameter)
            ),
            auto_far_percentile=float(
                depth_values.get("auto_far_percentile", depth_cfg.auto_far_percentile)
            ),
        )
    canny_cfg = _canny_config_from_mapping(config)
    return preprocess_bundle(
        bundle,
        depth_near_clip=depth_cfg.near_clip,
        depth_far_clip=depth_cfg.far_clip,
        depth_config=depth_cfg,
        **canny_cfg,
    )


def preprocess_and_save(
    bundle: SampleBundle,
    output_dir: Path,
    **kwargs,
) -> Path:
    """预处理并保存到磁盘。返回保存路径。"""
    bundle = preprocess_bundle(bundle, **kwargs)
    save_dir = output_dir / bundle.sample_key
    bundle.save(save_dir)
    missing = sorted(name for name in REQUIRED_PREPROCESSED_FILES if not (save_dir / name).exists())
    if missing:
        raise RuntimeError(f"{bundle.sample_key}: 预处理输出不完整，缺少 {missing}")
    return save_dir
