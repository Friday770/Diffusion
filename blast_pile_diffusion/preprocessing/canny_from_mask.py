"""从实例掩码派生 Canny 边缘图。

关键约束（技术路线 §4.4.4）：
Canny 边缘必须从实例掩码 M 派生，绝不从 Unity 渲染的 RGB 上提取。
RGB 上存在渲染阴影边，会导致边缘与实例边界不对齐。
掩码边界才是真实的石块边界，用它做 ControlNet-Canny 条件
可保证标签天然对齐。
"""

from __future__ import annotations

import cv2
import numpy as np


def mask_to_canny(
    mask: np.ndarray,
    canny_low: int = 50,
    canny_high: int = 150,
    dilate_kernel: int = 2,
) -> np.ndarray:
    """
    逐实例从离散 ID 掩码提取边界，合并为统一边缘图。

    这里刻意不从 RGB 或灰度渲染图跑 Canny。对二值实例掩码而言，
    形态学梯度能给出确定、连续的实例轮廓，
    并避免纹理/阴影引入内部杂边。
    canny_low/canny_high 保留为兼容配置入口；边界是否存在只由 mask ID 决定。

    Args:
        mask: (H, W) int32, 背景=0, 每个实例一个唯一 ID
        canny_low: 兼容旧配置，当前 mask 派生路径不使用
        canny_high: 兼容旧配置，当前 mask 派生路径不使用
        dilate_kernel: 边缘膨胀核大小，0 表示不膨胀

    Returns:
        (H, W) uint8 边缘图，255=边缘，0=非边缘
    """
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(
            f"mask_to_canny 需要 (H, W) 单通道实例掩码，实际 shape={mask.shape}"
        )
    if dilate_kernel < 0:
        raise ValueError("dilate_kernel 必须 >= 0")

    _ = (canny_low, canny_high)

    edges = np.zeros(mask.shape[:2], dtype=np.uint8)
    instance_ids = np.unique(mask[mask > 0])
    if len(instance_ids) == 0:
        return edges

    boundary_kernel = np.ones((3, 3), np.uint8)
    for instance_id in instance_ids:
        binary = (mask == instance_id).astype(np.uint8)
        boundary = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, boundary_kernel)
        edges[boundary > 0] = 255

    if dilate_kernel > 0:
        kernel = np.ones((dilate_kernel, dilate_kernel), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

    return edges
