"""测试 QC 边缘对齐检查。"""

import cv2
import numpy as np

from blast_pile_diffusion.qc.edge_alignment import (
    check_edge_alignment,
    compute_generated_to_mask_offsets,
)


def _make_mask_and_matching_rgb():
    """创建一个掩码和一张边缘完美对齐的 RGB。"""
    mask = np.zeros((256, 256), dtype=np.int32)
    mask[50:150, 50:200] = 1
    mask[160:230, 30:120] = 2

    rgb = np.full((256, 256, 3), 180, dtype=np.uint8)
    rgb[50:150, 50:200] = [100, 80, 60]
    rgb[160:230, 30:120] = [120, 100, 80]

    return mask, rgb


def test_perfect_alignment_passes():
    mask, rgb = _make_mask_and_matching_rgb()
    result = check_edge_alignment(rgb, mask, max_mean_offset=10.0, max_p99_offset=20.0)
    assert result.passed is True
    assert result.mean_offset < 10.0


def test_heavily_shifted_fails():
    mask = np.zeros((256, 256), dtype=np.int32)
    mask[50:150, 50:200] = 1

    rgb = np.full((256, 256, 3), 180, dtype=np.uint8)
    rgb[80:180, 80:230] = [100, 80, 60]

    result = check_edge_alignment(rgb, mask, max_mean_offset=4.0, max_p99_offset=12.0)
    assert result.mean_offset > 4.0
    assert result.passed is False
    assert "mean_offset_above_threshold" in result.failure_reasons


def test_blank_image_fails():
    mask = np.zeros((256, 256), dtype=np.int32)
    mask[50:150, 50:200] = 1

    rgb = np.full((256, 256, 3), 128, dtype=np.uint8)

    result = check_edge_alignment(rgb, mask, max_mean_offset=4.0, max_p99_offset=12.0)
    assert result.passed is False


def test_result_dict():
    mask, rgb = _make_mask_and_matching_rgb()
    result = check_edge_alignment(rgb, mask)
    d = result.to_dict()
    assert "passed" in d
    assert "mean_offset_px" in d
    assert "median_offset_px" in d
    assert "p95_offset_px" in d
    assert "p99_offset_px" in d
    assert "thresholds" in d
    assert d["edge_distance_direction"] == "generated_edge_to_mask_edge"
    assert isinstance(d["mean_offset_px"], float)


def test_distance_transform_direction_is_generated_to_mask():
    """Asymmetric edge sets catch accidental mask->generated averaging."""
    mask_edges = np.zeros((32, 64), dtype=np.uint8)
    gen_edges = np.zeros((32, 64), dtype=np.uint8)

    mask_edges[16, 4] = 255
    mask_edges[16, 60] = 255
    gen_edges[16, 10] = 255

    offsets = compute_generated_to_mask_offsets(gen_edges, mask_edges)
    assert offsets.shape == (1,)
    assert offsets[0] == 6.0
