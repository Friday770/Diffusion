"""统一的多格式图像读写接口，支持 EXR / 16bit PNG / 标准 PNG。"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np


def read_depth(path: Path) -> np.ndarray:
    """读取深度图，返回 float32 (H, W)。支持 EXR、16bit PNG、32bit TIFF。"""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".exr":
        img = cv2.imread(str(path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if img is None:
            raise FileNotFoundError(f"无法读取 EXR: {path}")
        if img.ndim == 3:
            return img[:, :, 0].astype(np.float32)
        return img.astype(np.float32)

    if suffix in (".tiff", ".tif"):
        img = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH)
        if img is None:
            raise FileNotFoundError(f"无法读取 TIFF: {path}")
        return img.astype(np.float32)

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"无法读取: {path}")
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 1000.0
    return img.astype(np.float32)


def read_rgb(path: Path) -> np.ndarray:
    """读取 RGB 图像，返回 (H, W, 3) uint8 RGB 顺序。"""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"无法读取: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_mask(path: Path) -> np.ndarray:
    """读取实例掩码，返回 (H, W) int32。"""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"无法读取: {path}")
    if img.ndim == 3:
        return img[:, :, 0].astype(np.int32)
    return img.astype(np.int32)


def save_depth_vis(depth: np.ndarray, path: Path) -> None:
    """将深度图保存为伪彩色可视化。"""
    valid = depth[depth > 0]
    if len(valid) == 0:
        cv2.imwrite(str(path), np.zeros((*depth.shape, 3), dtype=np.uint8))
        return
    d_min, d_max = np.percentile(valid, [1, 99])
    normalized = np.clip((depth - d_min) / (d_max - d_min + 1e-8), 0, 1)
    colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    cv2.imwrite(str(path), colored)


def read_normal_unity(path: Path) -> np.ndarray:
    """读取 Unity Perception 导出的法线图，自动识别编码并返回 (H,W,3) float32 ∈ [-1, 1]。

    支持以下输入：
      - ``.npy`` 直接 ``np.load``
      - 三通道图像（PNG / EXR / TIFF），自动从 BGR 转为 RGB

    根据数值范围自动反映射：
      - 浮点 [-1, 1]：原样
      - 浮点 [0, 1]：``x * 2 - 1``
      - uint8 [0, 255]：``x / 127.5 - 1``
      - uint16 [0, 65535]：``x / 32767.5 - 1``

    Unity 约定：R=Nx，G=Ny，B=Nz；返回值同样是 RGB 顺序。
    """
    path = Path(path)
    if path.suffix.lower() == ".npy":
        normal = np.load(path).astype(np.float32)
    else:
        img = cv2.imread(str(path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if img is None:
            raise FileNotFoundError(f"无法读取 Normal: {path}")
        if img.ndim != 3 or img.shape[2] < 3:
            raise ValueError(f"Normal 必须是三通道图像: {path}")
        img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
        normal = img.astype(np.float32)

    if normal.ndim != 3 or normal.shape[2] < 3:
        raise ValueError(f"Normal 必须是 (H,W,3): {path}")
    normal = normal[:, :, :3].astype(np.float32, copy=False)
    finite = normal[np.isfinite(normal)]
    if finite.size == 0:
        return np.zeros_like(normal, dtype=np.float32)

    min_value = float(finite.min())
    max_value = float(finite.max())
    if min_value >= 0.0 and max_value <= 1.0:
        normal = normal * 2.0 - 1.0
    elif min_value >= 0.0 and max_value <= 255.0:
        normal = normal / 127.5 - 1.0
    elif min_value >= 0.0 and max_value <= 65535.0:
        normal = normal / 32767.5 - 1.0

    normal = np.nan_to_num(normal, nan=0.0, posinf=1.0, neginf=-1.0)
    return np.clip(normal, -1.0, 1.0).astype(np.float32)


def decode_instance_mask_rgb(
    mask_rgb: np.ndarray,
    colormap: dict[tuple[int, int, int], int],
) -> np.ndarray:
    """将 Unity Perception 的 RGB 颜色编码实例掩码解码为整数实例 ID。

    Args:
        mask_rgb: (H, W, 3) uint8 RGB 顺序（应已从 BGR 转换）；接受单通道时直接返回。
        colormap: {(R, G, B): instance_id} 字典，从 Unity 元数据中提取。

    Returns:
        (H, W) int32 实例 ID 图，背景 (0,0,0) → 0。
        未在 colormap 中的非黑颜色会被自动分配新 instance_id（容错降级）。
    """
    if mask_rgb.ndim == 2:
        return mask_rgb.astype(np.int32)
    if mask_rgb.ndim != 3 or mask_rgb.shape[2] < 3:
        raise ValueError("InstanceSegmentation mask 必须是单通道或 RGB 图像")

    mask_rgb = mask_rgb[:, :, :3]
    decoded = np.zeros(mask_rgb.shape[:2], dtype=np.int32)
    assigned_colors: set[tuple[int, int, int]] = set()
    for color, instance_id in colormap.items():
        if color == (0, 0, 0):
            continue
        matched = np.all(mask_rgb == np.asarray(color, dtype=mask_rgb.dtype), axis=-1)
        if matched.any():
            decoded[matched] = int(instance_id)
            assigned_colors.add(color)

    next_instance_id = int(decoded.max()) + 1
    unique_colors = np.unique(mask_rgb.reshape(-1, 3), axis=0)
    for color_arr in unique_colors:
        color = tuple(int(v) for v in color_arr)
        if color == (0, 0, 0) or color in assigned_colors:
            continue
        matched = np.all(mask_rgb == color_arr, axis=-1)
        decoded[matched] = next_instance_id
        next_instance_id += 1

    return decoded
