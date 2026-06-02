"""边缘对齐 QC — 对应技术路线 §4.4.6。

用距离变换衡量生成图的 Canny 边缘与实例掩码边缘之间的偏移。
这是整套方案的命门：如果生成图的石块边界与掩码不对齐，
标签就是错的，训出来的分割模型会学到错误的边界。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from blast_pile_diffusion.preprocessing.canny_from_mask import mask_to_canny


@dataclass
class QCResult:
    passed: bool
    mean_offset: float
    p99_offset: float
    gen_edge_count: int
    mask_edge_count: int
    median_offset: float = float("inf")
    p95_offset: float = float("inf")
    max_offset: float = float("inf")
    image_shape: tuple[int, int] | None = None
    thresholds: dict[str, float | int] = field(default_factory=dict)
    failure_reasons: list[str] = field(default_factory=list)
    edge_distance_direction: str = "generated_edge_to_mask_edge"

    def to_dict(self) -> dict:
        def _finite_or_none(value: float, digits: int = 2) -> float | None:
            if np.isfinite(value):
                return round(float(value), digits)
            return None

        return {
            "passed": self.passed,
            "mean_offset_px": _finite_or_none(self.mean_offset),
            "median_offset_px": _finite_or_none(self.median_offset),
            "p95_offset_px": _finite_or_none(self.p95_offset),
            "p99_offset_px": _finite_or_none(self.p99_offset),
            "max_offset_px": _finite_or_none(self.max_offset),
            "gen_edge_count": self.gen_edge_count,
            "mask_edge_count": self.mask_edge_count,
            "image_shape": list(self.image_shape) if self.image_shape else None,
            "thresholds": dict(self.thresholds),
            "edge_distance_direction": self.edge_distance_direction,
            "failure_reasons": list(self.failure_reasons),
        }


def compute_generated_to_mask_offsets(
    generated_edges: np.ndarray,
    mask_edges: np.ndarray,
) -> np.ndarray:
    """Return distances from each generated edge pixel to the nearest mask edge pixel."""
    if generated_edges.ndim != 2 or mask_edges.ndim != 2:
        raise ValueError("generated_edges and mask_edges must both be 2D arrays")
    if generated_edges.shape != mask_edges.shape:
        raise ValueError(
            "generated_edges and mask_edges must have the same shape: "
            f"{generated_edges.shape} != {mask_edges.shape}"
        )

    gen_pts = np.argwhere(generated_edges > 0)
    if len(gen_pts) == 0:
        return np.empty((0,), dtype=np.float64)

    # distance_transform_edt returns the distance to the nearest zero-valued pixel.
    # Passing mask_edges == 0 makes mask-edge pixels the zero set, so indexing this
    # transform at generated-edge pixels gives gen-edge -> mask-edge offsets.
    dt_to_mask = distance_transform_edt(mask_edges == 0)
    return dt_to_mask[gen_pts[:, 0], gen_pts[:, 1]].astype(np.float64, copy=False)


def _normalize_rgb_image(generated_rgb: np.ndarray) -> np.ndarray:
    if generated_rgb.ndim == 2:
        return cv2.cvtColor(generated_rgb, cv2.COLOR_GRAY2RGB)
    if generated_rgb.ndim != 3 or generated_rgb.shape[2] not in (3, 4):
        raise ValueError(
            "generated_rgb must have shape (H, W), (H, W, 3), or (H, W, 4); "
            f"got {generated_rgb.shape}"
        )
    if generated_rgb.shape[2] == 4:
        generated_rgb = generated_rgb[:, :, :3]
    return generated_rgb


def check_edge_alignment(
    generated_rgb: np.ndarray,
    mask: np.ndarray,
    max_mean_offset: float = 4.0,
    max_p99_offset: float = 12.0,
    canny_low: int = 50,
    canny_high: int = 150,
    mask_dilate_kernel: int = 0,
    min_gen_edge_pixels: int = 1,
    min_mask_edge_pixels: int = 1,
) -> QCResult:
    """
    检查生成图的边缘是否与实例掩码边缘对齐。

    Args:
        generated_rgb: (H, W, 3) uint8 生成图（RGB）
        mask: (H, W) int32 实例掩码
        max_mean_offset: 平均偏移阈值（像素）
        max_p99_offset: 99th percentile 偏移阈值（像素）
        canny_low: 生成图与掩码边缘提取使用的 Canny 低阈值
        canny_high: 生成图与掩码边缘提取使用的 Canny 高阈值
        mask_dilate_kernel: 掩码边缘膨胀核大小，0 表示不膨胀

    Returns:
        QCResult
    """
    generated_rgb = _normalize_rgb_image(generated_rgb)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if generated_rgb.shape[:2] != mask.shape[:2]:
        raise ValueError(
            "generated_rgb and mask must have the same spatial shape: "
            f"{generated_rgb.shape[:2]} != {mask.shape[:2]}"
        )

    thresholds: dict[str, float | int] = {
        "max_mean_offset_px": float(max_mean_offset),
        "max_p99_offset_px": float(max_p99_offset),
        "canny_low": int(canny_low),
        "canny_high": int(canny_high),
        "mask_dilate_kernel": int(mask_dilate_kernel),
        "min_gen_edge_pixels": int(min_gen_edge_pixels),
        "min_mask_edge_pixels": int(min_mask_edge_pixels),
    }

    gen_gray = cv2.cvtColor(generated_rgb, cv2.COLOR_RGB2GRAY)
    gen_canny = cv2.Canny(gen_gray, canny_low, canny_high)

    mask_canny = mask_to_canny(
        mask,
        canny_low=canny_low,
        canny_high=canny_high,
        dilate_kernel=mask_dilate_kernel,
    )
    gen_edge_count = int((gen_canny > 0).sum())
    mask_edge_count = int((mask_canny > 0).sum())

    if mask_edge_count < min_mask_edge_pixels:
        return QCResult(
            passed=False,
            mean_offset=float("inf"),
            p99_offset=float("inf"),
            gen_edge_count=gen_edge_count,
            mask_edge_count=mask_edge_count,
            image_shape=mask.shape[:2],
            thresholds=thresholds,
            failure_reasons=["no_mask_edges"],
        )

    if gen_edge_count < min_gen_edge_pixels:
        return QCResult(
            passed=False,
            mean_offset=float("inf"),
            p99_offset=float("inf"),
            gen_edge_count=gen_edge_count,
            mask_edge_count=mask_edge_count,
            image_shape=mask.shape[:2],
            thresholds=thresholds,
            failure_reasons=["no_generated_edges"],
        )

    offsets = compute_generated_to_mask_offsets(gen_canny, mask_canny)
    mean_off = float(offsets.mean())
    median_off = float(np.median(offsets))
    p95_off = float(np.percentile(offsets, 95))
    p99_off = float(np.percentile(offsets, 99))
    max_off = float(offsets.max())

    failure_reasons = []
    if mean_off > max_mean_offset:
        failure_reasons.append("mean_offset_above_threshold")
    if p99_off > max_p99_offset:
        failure_reasons.append("p99_offset_above_threshold")

    passed = not failure_reasons

    return QCResult(
        passed=passed,
        mean_offset=mean_off,
        p99_offset=p99_off,
        gen_edge_count=gen_edge_count,
        mask_edge_count=mask_edge_count,
        median_offset=median_off,
        p95_offset=p95_off,
        max_offset=max_off,
        image_shape=mask.shape[:2],
        thresholds=thresholds,
        failure_reasons=failure_reasons,
    )
