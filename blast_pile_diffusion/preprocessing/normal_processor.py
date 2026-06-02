"""法线图格式转换：Unity/深度法线 → ControlNet-Normal 输入格式。

ControlNet-Normal 期望 RGB 编码的相机空间单位法线：
R=X, G=Y, B=Z，[-1, 1] 线性映射到 [0, 255]，Z 朝向相机/外法线时偏蓝。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from blast_pile_diffusion.utils.depth_utils import compute_normals_from_depth


def _decode_normal(normal: np.ndarray) -> np.ndarray:
    """接受 [-1,1]、[0,1] float 或 [0,255] uint 法线图，统一转为 float32 [-1,1]。"""
    arr = np.asarray(normal)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"normal 必须是 (H, W, 3)，实际 shape={arr.shape}")

    if np.issubdtype(arr.dtype, np.integer):
        return (arr.astype(np.float32) / 255.0) * 2.0 - 1.0

    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=1.0, neginf=-1.0)
    if arr.size == 0:
        return arr

    min_value = float(arr.min())
    max_value = float(arr.max())
    if min_value == 0.0 and max_value == 0.0:
        return arr
    if min_value >= 0.0 and max_value <= 1.0:
        encoded = arr * 2.0 - 1.0
        raw_error = _median_unit_length_error(arr)
        encoded_error = _median_unit_length_error(encoded)
        return encoded if encoded_error + 1e-4 < raw_error else arr

    return arr


def _median_unit_length_error(normal: np.ndarray) -> float:
    norms = np.linalg.norm(normal, axis=-1)
    valid = norms > 1e-8
    if not np.any(valid):
        return 0.0
    return float(np.median(np.abs(norms[valid] - 1.0)))


def _axis_indices(axis_order: str | Sequence[int]) -> tuple[int, int, int]:
    if isinstance(axis_order, str):
        normalized = axis_order.lower()
        if sorted(normalized) != ["x", "y", "z"]:
            raise ValueError("axis_order 字符串必须是 xyz 的排列，例如 'xyz' 或 'xzy'")
        lookup = {"x": 0, "y": 1, "z": 2}
        return tuple(lookup[axis] for axis in normalized)

    indices = tuple(int(axis) for axis in axis_order)
    if sorted(indices) != [0, 1, 2]:
        raise ValueError("axis_order 序列必须是 0,1,2 的排列")
    return indices


def _normalize_vectors(normal: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(normal, axis=-1, keepdims=True)
    valid = norms[..., 0] > 1e-8
    normalized = np.zeros_like(normal, dtype=np.float32)
    normalized[valid] = normal[valid] / norms[valid]
    return normalized


def process_normal(
    normal: np.ndarray | None,
    from_depth: bool = False,
    depth: np.ndarray | None = None,
    axis_order: str | Sequence[int] = "xyz",
    flip_x: bool = False,
    flip_y: bool = False,
    flip_z: bool = False,
    normalize: bool = True,
) -> np.ndarray:
    """
    将法线图转换为 ControlNet-Normal 输入格式。

    Args:
        normal: (H, W, 3) 法线图；支持 float [-1,1]、float [0,1] 或 uint8 [0,255]
        from_depth: 若为 True，忽略 normal 参数，从 depth 派生法线
        depth: 深度图，仅当 from_depth=True 时需要
        axis_order: 输入法线轴到输出 RGB 轴的排列，如 "xyz"、"xzy" 或 (0, 2, 1)
        flip_x: 输出前翻转 X 分量
        flip_y: 输出前翻转 Y 分量
        flip_z: 输出前翻转 Z 分量
        normalize: 是否重新单位化法线向量

    Returns:
        (H, W, 3) uint8，RGB 编码的法线图
    """
    if from_depth:
        if depth is None:
            raise ValueError("from_depth=True 时必须提供 depth")
        depth_arr = np.asarray(depth, dtype=np.float32)
        if depth_arr.ndim != 2:
            raise ValueError(f"depth 必须是 (H, W)，实际 shape={depth_arr.shape}")
        normal_arr = compute_normals_from_depth(depth_arr)
    else:
        if normal is None:
            raise ValueError("from_depth=False 时必须提供 normal")
        normal_arr = _decode_normal(normal)

    normal_arr = normal_arr[:, :, _axis_indices(axis_order)].astype(np.float32, copy=False)

    flips = np.array(
        [-1.0 if flip_x else 1.0, -1.0 if flip_y else 1.0, -1.0 if flip_z else 1.0],
        dtype=np.float32,
    )
    normal_arr = normal_arr * flips

    if normalize:
        normal_arr = _normalize_vectors(normal_arr)

    normal_clipped = np.clip(normal_arr, -1.0, 1.0)
    normal_uint8 = np.rint((normal_clipped + 1.0) * 127.5).astype(np.uint8)
    return normal_uint8
