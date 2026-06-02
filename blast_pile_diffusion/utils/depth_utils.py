"""深度图处理工具：归一化、孔洞填充、视差转换。"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage


def normalize_depth_for_controlnet(
    depth: np.ndarray,
    near_clip: float = 0.5,
    far_clip: float = 100.0,
) -> np.ndarray:
    """
    将米单位深度图归一化为 ControlNet-Depth 输入格式。

    ControlNet-depth-sdxl 期望灰度图，近处白、远处黑。
    返回 (H, W, 3) uint8 三通道灰度图。
    """
    depth_clipped = np.clip(depth, near_clip, far_clip)
    normalized = (depth_clipped - near_clip) / (far_clip - near_clip + 1e-8)
    inverted = 1.0 - normalized
    gray = (inverted * 255).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def sanitize_depth(depth: np.ndarray) -> np.ndarray:
    """返回 float32 深度图，并把 NaN/Inf/负值清成 0 孔洞。"""
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    cleaned = depth.astype(np.float32, copy=True)
    invalid = ~np.isfinite(cleaned) | (cleaned < 0)
    cleaned[invalid] = 0.0
    return cleaned


def fill_depth_holes(
    depth: np.ndarray,
    smooth_diameter: int = 5,
    *,
    max_hole_size: int | None = None,
) -> np.ndarray:
    """用最近有效深度填充 0/NaN/Inf 孔洞，并只在孔洞区域轻微平滑。

    Args:
        depth: (H, W) 或 (H, W, 1) 深度图，任意 dtype。
        smooth_diameter: 仅在被填充孔洞像素上应用的双边滤波直径；<= 1 关闭平滑。
        max_hole_size: 已废弃，保留为 ``smooth_diameter`` 的 keyword-only 别名以向后兼容。

    Returns:
        (H, W) float32 已填充孔洞的深度图。
    """
    # 向后兼容：仅当 smooth_diameter 仍为默认值时才使用 max_hole_size 覆盖。
    if max_hole_size is not None and smooth_diameter == 5:
        smooth_diameter = int(max_hole_size)

    filled = sanitize_depth(depth)
    invalid = ~np.isfinite(filled) | (filled <= 0)
    if not invalid.any():
        return filled

    if not (~invalid).any():
        return np.zeros_like(filled, dtype=np.float32)

    nearest_indices = ndimage.distance_transform_edt(
        invalid,
        return_distances=False,
        return_indices=True,
    )
    filled[invalid] = filled[tuple(nearest_indices[:, invalid])]

    if smooth_diameter > 1:
        diameter = max(3, int(smooth_diameter) | 1)
        smoothed = cv2.bilateralFilter(
            filled.astype(np.float32),
            d=diameter,
            sigmaColor=0.5,
            sigmaSpace=max(3, diameter),
        )
        filled[invalid] = smoothed[invalid]

    return np.nan_to_num(filled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def depth_to_disparity(depth: np.ndarray, baseline: float = 0.12, focal_px: float = 1000.0) -> np.ndarray:
    """米制深度图 → 视差图（像素单位），用于某些需要视差输入的模型。"""
    safe_depth = np.where(depth > 0.01, depth, 0.01)
    return (baseline * focal_px) / safe_depth


def compute_normals_from_depth(
    depth: np.ndarray,
    pixel_size: float = 1.0,
    gradient_scale: float = 1.0,
) -> np.ndarray:
    """从深度图计算表面法线（相机空间），返回 (H, W, 3) float32 单位法向量。

    ``pixel_size`` 和 ``gradient_scale`` 用于把深度梯度调到与像素尺度相近的
    无量纲范围，避免真实米制深度变化被固定 Z=1 过度压扁。
    """
    if pixel_size <= 0:
        raise ValueError("pixel_size 必须 > 0")
    if gradient_scale <= 0:
        raise ValueError("gradient_scale 必须 > 0")

    depth = np.asarray(depth, dtype=np.float32)
    safe_depth = fill_depth_holes(depth, smooth_diameter=0)
    dz_dx = cv2.Sobel(safe_depth, cv2.CV_32F, 1, 0, ksize=3) / (8.0 * pixel_size)
    dz_dy = cv2.Sobel(safe_depth, cv2.CV_32F, 0, 1, ksize=3) / (8.0 * pixel_size)

    normals = np.zeros((*depth.shape, 3), dtype=np.float32)
    normals[:, :, 0] = -dz_dx * gradient_scale
    normals[:, :, 1] = -dz_dy * gradient_scale
    normals[:, :, 2] = 1.0

    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / (norms + 1e-8)

    return normals
