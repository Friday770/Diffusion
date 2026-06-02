"""生成图多样性检测 — 防止 mode collapse。

如果同一 Unity 场景的多个 seed 生成了几乎一样的图，说明 prompt 随机化不够
或者 ControlNet 约束太强压制了 SDXL 的多样性。
"""

from __future__ import annotations

from itertools import combinations

import cv2
import numpy as np


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """计算两张图的结构相似度（简化版，不依赖 skimage）。"""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    return float(ssim_map.mean())


def check_diversity(
    images: list[np.ndarray],
    max_ssim: float = 0.85,
) -> dict:
    """
    检查一组生成图的多样性。

    Args:
        images: 同一 Unity 场景不同 seed 的生成图列表，(H, W, 3) uint8
        max_ssim: SSIM 上限，超过此值认为两张图太相似

    Returns:
        {"passed": bool, "mean_ssim": float, "max_ssim": float, "num_similar_pairs": int}
    """
    if len(images) < 2:
        return {
            "passed": True,
            "mean_ssim": 0.0,
            "max_ssim": 0.0,
            "max_ssim_threshold": max_ssim,
            "num_similar_pairs": 0,
            "pair_count": 0,
        }

    gray_images = []
    for img in images:
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        target_size = (256, 256)
        img = cv2.resize(img, target_size)
        gray_images.append(img)

    ssim_values = []
    for i, j in combinations(range(len(gray_images)), 2):
        ssim_values.append(compute_ssim(gray_images[i], gray_images[j]))

    mean_ssim = float(np.mean(ssim_values))
    max_ssim_val = float(np.max(ssim_values))
    num_similar = sum(1 for s in ssim_values if s > max_ssim)

    return {
        "passed": num_similar == 0,
        "mean_ssim": round(mean_ssim, 4),
        "max_ssim": round(max_ssim_val, 4),
        "max_ssim_threshold": max_ssim,
        "num_similar_pairs": num_similar,
        "pair_count": len(ssim_values),
    }
