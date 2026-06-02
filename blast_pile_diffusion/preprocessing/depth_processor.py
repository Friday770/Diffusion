"""深度图预处理：孔洞填充 → 归一化 → 输出 ControlNet-Depth 格式。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from blast_pile_diffusion.utils.depth_utils import (
    fill_depth_holes,
    normalize_depth_for_controlnet,
    sanitize_depth,
)


@dataclass(frozen=True)
class DepthProcessingConfig:
    near_clip: float = 0.5
    far_clip: float = 100.0
    fill_holes: bool = True
    hole_smooth_diameter: int = 5
    auto_far_percentile: float = 99.0


def load_depth_config(config_path: Path | str = Path("configs/base.yaml")) -> DepthProcessingConfig:
    """从 YAML 配置读取深度预处理参数，缺失时使用安全默认值。"""
    path = Path(config_path)
    if not path.exists():
        return DepthProcessingConfig()
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return depth_config_from_mapping(cfg)


def depth_config_from_mapping(cfg: Mapping[str, Any] | None) -> DepthProcessingConfig:
    """从 dict 中提取 depth 配置。

    支持 ``preprocessing.depth``、顶层 ``depth`` 两种布局，便于兼容不同
    agent 后续可能添加的配置文件结构。
    """
    cfg = cfg or {}
    depth_cfg: Mapping[str, Any] = {}
    preprocessing = cfg.get("preprocessing") if isinstance(cfg, Mapping) else None
    if isinstance(preprocessing, Mapping) and isinstance(preprocessing.get("depth"), Mapping):
        depth_cfg = preprocessing["depth"]
    elif isinstance(cfg.get("depth"), Mapping):
        depth_cfg = cfg["depth"]

    return DepthProcessingConfig(
        near_clip=float(depth_cfg.get("near_clip", DepthProcessingConfig.near_clip)),
        far_clip=float(depth_cfg.get("far_clip", DepthProcessingConfig.far_clip)),
        fill_holes=bool(depth_cfg.get("fill_holes", DepthProcessingConfig.fill_holes)),
        hole_smooth_diameter=int(
            depth_cfg.get("hole_smooth_diameter", DepthProcessingConfig.hole_smooth_diameter)
        ),
        auto_far_percentile=float(
            depth_cfg.get("auto_far_percentile", DepthProcessingConfig.auto_far_percentile)
        ),
    )


def _resolve_config(
    near_clip: float | None,
    far_clip: float | None,
    fill_holes: bool | None,
    config: DepthProcessingConfig | Mapping[str, Any] | None,
    config_path: Path | str | None,
) -> DepthProcessingConfig:
    if config_path is not None:
        resolved = load_depth_config(config_path)
    elif isinstance(config, DepthProcessingConfig):
        resolved = config
    elif isinstance(config, Mapping):
        resolved = depth_config_from_mapping(config)
    else:
        resolved = DepthProcessingConfig()

    return DepthProcessingConfig(
        near_clip=resolved.near_clip if near_clip is None else float(near_clip),
        far_clip=resolved.far_clip if far_clip is None else float(far_clip),
        fill_holes=resolved.fill_holes if fill_holes is None else bool(fill_holes),
        hole_smooth_diameter=resolved.hole_smooth_diameter,
        auto_far_percentile=resolved.auto_far_percentile,
    )


def _auto_far_clip(depth: np.ndarray, percentile: float) -> float:
    valid = depth[np.isfinite(depth) & (depth > 0)]
    if valid.size == 0:
        return DepthProcessingConfig.far_clip
    return float(np.percentile(valid, np.clip(percentile, 1.0, 100.0)))


def process_depth(
    depth: np.ndarray,
    near_clip: float | None = None,
    far_clip: float | None = None,
    fill_holes: bool | None = None,
    config: DepthProcessingConfig | Mapping[str, Any] | None = None,
    config_path: Path | str | None = None,
) -> np.ndarray:
    """
    将 Unity 原始深度图处理为 ControlNet-Depth 输入。

    Returns:
        (H, W, 3) uint8，近处白远处黑的三通道灰度图
    """
    cfg = _resolve_config(near_clip, far_clip, fill_holes, config, config_path)
    depth = sanitize_depth(depth)

    if cfg.fill_holes:
        depth = fill_depth_holes(depth, smooth_diameter=cfg.hole_smooth_diameter)

    far_clip_value = cfg.far_clip
    if far_clip_value <= 0:
        far_clip_value = _auto_far_clip(depth, cfg.auto_far_percentile)
    if far_clip_value <= cfg.near_clip:
        far_clip_value = max(cfg.near_clip + 1e-3, _auto_far_clip(depth, cfg.auto_far_percentile))

    depth_cn = normalize_depth_for_controlnet(
        depth,
        near_clip=cfg.near_clip,
        far_clip=far_clip_value,
    )
    return np.nan_to_num(depth_cn, nan=0, posinf=255, neginf=0).astype(np.uint8)
