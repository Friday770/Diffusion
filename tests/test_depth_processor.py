"""测试深度图处理模块。"""

import numpy as np

from blast_pile_diffusion.preprocessing.depth_processor import (
    fill_depth_holes as fill_depth_holes_for_preprocessing,
    load_depth_config,
    process_depth,
)
from blast_pile_diffusion.utils.depth_utils import (
    compute_normals_from_depth,
    fill_depth_holes,
    normalize_depth_for_controlnet,
)


def test_normalize_output_shape():
    depth = np.random.uniform(1.0, 50.0, (256, 256)).astype(np.float32)
    result = normalize_depth_for_controlnet(depth, near_clip=0.5, far_clip=50.0)
    assert result.shape == (256, 256, 3)
    assert result.dtype == np.uint8


def test_normalize_near_is_bright():
    depth = np.ones((100, 100), dtype=np.float32)
    depth[:50] = 1.0
    depth[50:] = 50.0
    result = normalize_depth_for_controlnet(depth, near_clip=0.5, far_clip=50.0)
    near_brightness = result[:50, :, 0].mean()
    far_brightness = result[50:, :, 0].mean()
    assert near_brightness > far_brightness


def test_fill_depth_holes():
    depth = np.random.uniform(5.0, 20.0, (100, 100)).astype(np.float32)
    depth[40:60, 40:60] = 0
    filled = fill_depth_holes(depth)
    assert (filled[40:60, 40:60] > 0).all()


def test_compute_normals_shape():
    depth = np.random.uniform(1.0, 10.0, (128, 128)).astype(np.float32)
    normals = compute_normals_from_depth(depth)
    assert normals.shape == (128, 128, 3)
    norms = np.linalg.norm(normals, axis=-1)
    np.testing.assert_allclose(norms, 1.0, atol=0.01)


def test_process_depth_end_to_end():
    depth = np.random.uniform(2.0, 30.0, (256, 256)).astype(np.float32)
    depth[100:120, 100:120] = 0
    result = process_depth(depth, near_clip=0.5, far_clip=50.0, fill_holes=True)
    assert result.shape == (256, 256, 3)
    assert result.dtype == np.uint8
    assert result.min() >= 0
    assert result.max() <= 255


def test_preprocessing_hole_fill_uses_local_valid_depth():
    depth = np.full((64, 64), 8.0, dtype=np.float32)
    depth[:, 48:] = 40.0
    depth[20:30, 20:30] = 0.0
    depth[35:40, 35:40] = np.nan

    filled = fill_depth_holes_for_preprocessing(depth)

    assert np.isfinite(filled).all()
    assert (filled > 0).all()
    assert filled[20:30, 20:30].mean() < 15.0
    assert filled[35:40, 35:40].mean() < 20.0


def test_process_depth_handles_nan_inf_and_auto_far_clip():
    depth = np.linspace(1.0, 20.0, 100, dtype=np.float32).reshape(10, 10)
    depth[2, 2] = np.nan
    depth[3, 3] = np.inf
    depth[4, 4] = 0.0

    result = process_depth(depth, near_clip=1.0, far_clip=0.0, fill_holes=True)

    assert result.shape == (10, 10, 3)
    assert result.dtype == np.uint8
    assert np.isfinite(result).all()
    assert result[0, 0, 0] > result[-1, -1, 0]


def test_load_depth_config_from_base_yaml_shape(tmp_path):
    config_path = tmp_path / "base.yaml"
    config_path.write_text(
        """
preprocessing:
  depth:
    near_clip: 0.25
    far_clip: 42.0
    fill_holes: false
    hole_smooth_diameter: 7
    auto_far_percentile: 95.0
"""
    )

    cfg = load_depth_config(config_path)

    assert cfg.near_clip == 0.25
    assert cfg.far_clip == 42.0
    assert cfg.fill_holes is False
    assert cfg.hole_smooth_diameter == 7
    assert cfg.auto_far_percentile == 95.0


def test_process_depth_uses_config_path_when_not_overridden(tmp_path):
    config_path = tmp_path / "base.yaml"
    config_path.write_text(
        """
preprocessing:
  depth:
    near_clip: 2.0
    far_clip: 4.0
    fill_holes: true
"""
    )
    depth = np.array([[2.0, 4.0]], dtype=np.float32)

    result = process_depth(depth, config_path=config_path)

    assert result[0, 0, 0] == 255
    assert result[0, 1, 0] == 0
