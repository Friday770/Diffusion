"""Debug 可视化：生成对比图用于人工审查。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _as_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"Expected a 2D or RGB/RGBA image, got shape {image.shape}")
    if image.shape[2] == 4:
        return image[:, :, :3]
    return image


def overlay_edges_on_image(
    image: np.ndarray,
    edges: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.7,
) -> np.ndarray:
    """将边缘叠加到图像上。"""
    image = _as_rgb(image)
    if image.shape[:2] != edges.shape[:2]:
        raise ValueError(f"Image and edge shapes differ: {image.shape[:2]} != {edges.shape[:2]}")

    vis = image.copy()
    edge_mask = edges > 0
    vis[edge_mask] = (
        np.array(color, dtype=np.float32) * alpha
        + vis[edge_mask].astype(np.float32) * (1 - alpha)
    ).astype(np.uint8)
    return vis


def overlay_qc_edges(
    image: np.ndarray,
    mask_edges: np.ndarray,
    gen_edges: np.ndarray,
    alpha: float = 0.75,
) -> np.ndarray:
    """合并显示 QC 边缘：绿=mask 边缘，红=生成图边缘，黄=重叠。"""
    vis = overlay_edges_on_image(image, mask_edges, color=(0, 255, 0), alpha=alpha)
    vis = overlay_edges_on_image(vis, gen_edges, color=(255, 0, 0), alpha=alpha)
    overlap = (mask_edges > 0) & (gen_edges > 0)
    vis[overlap] = np.array([255, 255, 0], dtype=np.uint8)
    return vis


def make_comparison_grid(
    images: list[np.ndarray],
    labels: list[str] | None = None,
    max_width: int = 512,
) -> np.ndarray:
    """将多张图拼成一行对比图。"""
    resized = []
    for img in images:
        img = _as_rgb(img)
        h, w = img.shape[:2]
        scale = max_width / w
        resized.append(cv2.resize(img, (max_width, int(h * scale))))

    target_h = max(r.shape[0] for r in resized)
    padded = []
    for img in resized:
        if img.shape[0] < target_h:
            pad = np.zeros((target_h - img.shape[0], img.shape[1], 3), dtype=np.uint8)
            img = np.vstack([img, pad])
        padded.append(img)

    grid = np.hstack(padded)

    if labels:
        for i, label in enumerate(labels):
            x = i * max_width + 10
            cv2.rectangle(grid, (x - 6, 7), (x + max(160, len(label) * 14), 38), (0, 0, 0), -1)
            cv2.putText(
                grid,
                label,
                (x, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
            )

    return grid


def save_qc_debug_image(
    unity_rgb: np.ndarray,
    generated_rgb: np.ndarray,
    mask_canny: np.ndarray,
    gen_canny: np.ndarray,
    output_path: Path,
) -> None:
    """保存 QC debug 对比图：Unity | Gen+Mask | Gen+Gen | Gen+Both。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vis1 = overlay_edges_on_image(generated_rgb, mask_canny, color=(0, 255, 0))
    vis2 = overlay_edges_on_image(generated_rgb, gen_canny, color=(255, 0, 0))
    vis3 = overlay_qc_edges(generated_rgb, mask_canny, gen_canny)
    grid = make_comparison_grid(
        [unity_rgb, vis1, vis2, vis3],
        labels=["Unity RGB", "Gen + Mask Edge", "Gen + Gen Edge", "Combined QC"],
    )
    cv2.imwrite(str(output_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
