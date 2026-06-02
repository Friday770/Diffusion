"""测试 mask_to_canny：确保 Canny 边缘从实例掩码正确派生。"""

import cv2
import numpy as np
import pytest

from blast_pile_diffusion.preprocessing.canny_from_mask import mask_to_canny


def _make_simple_mask(h: int = 256, w: int = 256) -> np.ndarray:
    """创建一个包含 3 个矩形实例的简单掩码。"""
    mask = np.zeros((h, w), dtype=np.int32)
    mask[30:80, 30:100] = 1
    mask[100:180, 50:150] = 2
    mask[60:120, 160:230] = 3
    return mask


def _foreground_boundary(mask: np.ndarray, inst_id: int) -> np.ndarray:
    """用邻域 ID 变化定义实例内侧边界，独立于实现细节。"""
    instance = mask == inst_id
    padded = np.pad(mask, 1, mode="edge")
    center = padded[1:-1, 1:-1]
    boundary = np.zeros(mask.shape, dtype=bool)

    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            neighbor = padded[1 + dy : 1 + dy + mask.shape[0], 1 + dx : 1 + dx + mask.shape[1]]
            boundary |= instance & (neighbor != center)

    return boundary


def test_basic_shape():
    mask = _make_simple_mask()
    edges = mask_to_canny(mask)
    assert edges.shape == mask.shape
    assert edges.dtype == np.uint8


def test_background_has_no_edges():
    mask = np.zeros((100, 100), dtype=np.int32)
    edges = mask_to_canny(mask)
    assert edges.max() == 0


def test_edges_on_boundaries():
    mask = _make_simple_mask()
    edges = mask_to_canny(mask, dilate_kernel=0)
    assert edges.max() == 255
    assert (edges > 0).sum() > 0
    for inst_id in [1, 2, 3]:
        boundary = _foreground_boundary(mask, inst_id)
        assert np.all(edges[boundary] == 255), f"Instance {inst_id} boundary has gaps"


def test_no_spurious_edges_inside_instances():
    mask = _make_simple_mask()
    edges = mask_to_canny(mask, dilate_kernel=0)
    kernel = np.ones((9, 9), np.uint8)

    for inst_id in [1, 2, 3]:
        binary = (mask == inst_id).astype(np.uint8)
        deep_interior = cv2.erode(binary, kernel, iterations=1) > 0
        assert not np.any(edges[deep_interior] > 0), f"Instance {inst_id} interior has stray edges"


def test_dilate_expands_edges():
    mask = _make_simple_mask()
    edges_no_dilate = mask_to_canny(mask, dilate_kernel=0)
    edges_dilated = mask_to_canny(mask, dilate_kernel=3)
    assert (edges_dilated > 0).sum() >= (edges_no_dilate > 0).sum()


def test_multiple_instances_all_have_edges():
    mask = _make_simple_mask()
    edges = mask_to_canny(mask, dilate_kernel=0)
    for inst_id in [1, 2, 3]:
        boundary = _foreground_boundary(mask, inst_id)
        overlap = ((edges > 0) & boundary).sum()
        assert overlap > 0, f'Instance {inst_id} has no edge pixels'


def test_touching_instances_have_seam_edges():
    mask = np.zeros((100, 100), dtype=np.int32)
    mask[20:80, 10:50] = 1
    mask[20:80, 50:90] = 2

    edges = mask_to_canny(mask, dilate_kernel=0)

    assert np.all(edges[20:80, 49] == 255)
    assert np.all(edges[20:80, 50] == 255)


def test_invalid_mask_shape_raises():
    mask = np.zeros((32, 32, 3), dtype=np.int32)
    with pytest.raises(ValueError, match="单通道实例掩码"):
        mask_to_canny(mask)
