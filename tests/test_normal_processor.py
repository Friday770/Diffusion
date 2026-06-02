"""测试 ControlNet-Normal 法线预处理。"""

import numpy as np
import pytest

from blast_pile_diffusion.preprocessing.normal_processor import process_normal
from blast_pile_diffusion.preprocessing.preprocessor import preprocess_bundle
from blast_pile_diffusion.data.sample_bundle import SampleBundle


def test_process_normal_outputs_uint8_rgb():
    normal = np.zeros((16, 16, 3), dtype=np.float32)
    normal[..., 2] = 1.0

    result = process_normal(normal)

    assert result.shape == (16, 16, 3)
    assert result.dtype == np.uint8
    assert result.min() >= 0
    assert result.max() <= 255
    assert result[..., 2].mean() == 255
    assert 127 <= result[..., 0].mean() <= 128
    assert 127 <= result[..., 1].mean() <= 128


def test_uint8_encoded_normal_round_trips_to_controlnet_encoding():
    normal = np.zeros((8, 8, 3), dtype=np.uint8)
    normal[..., 0] = 128
    normal[..., 1] = 128
    normal[..., 2] = 255

    result = process_normal(normal)

    assert np.all(result[..., 2] == 255)
    assert np.all((127 <= result[..., 0]) & (result[..., 0] <= 129))
    assert np.all((127 <= result[..., 1]) & (result[..., 1] <= 129))


def test_float_zero_to_one_encoded_normal_is_decoded():
    normal = np.zeros((8, 8, 3), dtype=np.float32)
    normal[..., 0] = 0.5
    normal[..., 1] = 0.5
    normal[..., 2] = 1.0

    result = process_normal(normal)

    assert np.all(result[..., 2] == 255)
    assert np.all((127 <= result[..., 0]) & (result[..., 0] <= 128))
    assert np.all((127 <= result[..., 1]) & (result[..., 1] <= 128))


def test_axis_order_and_flip_support_unity_convention_adjustments():
    normal = np.zeros((4, 4, 3), dtype=np.float32)
    normal[..., 1] = 1.0

    result = process_normal(normal, axis_order="xzy", flip_z=True)

    assert np.all(result[..., 0] == 128)
    assert np.all(result[..., 1] == 128)
    assert np.all(result[..., 2] == 0)


def test_non_unit_vectors_are_normalized_before_encoding():
    normal = np.zeros((8, 8, 3), dtype=np.float32)
    normal[..., 2] = 5.0

    result = process_normal(normal)

    assert np.all(result[..., 2] == 255)


def test_zero_normal_encodes_to_neutral_gray():
    normal = np.zeros((8, 8, 3), dtype=np.float32)

    result = process_normal(normal)

    assert np.all(result == 128)


def test_from_depth_fallback_produces_blue_flat_surface():
    depth = np.full((16, 16), 10.0, dtype=np.float32)

    result = process_normal(None, from_depth=True, depth=depth)

    assert result.shape == (16, 16, 3)
    assert result.dtype == np.uint8
    assert np.all(result[..., 2] == 255)
    assert np.all((127 <= result[..., 0]) & (result[..., 0] <= 128))
    assert np.all((127 <= result[..., 1]) & (result[..., 1] <= 128))


def test_from_depth_requires_depth():
    with pytest.raises(ValueError, match="必须提供 depth"):
        process_normal(None, from_depth=True)


def test_invalid_normal_shape_raises():
    with pytest.raises(ValueError, match="normal 必须是"):
        process_normal(np.zeros((8, 8), dtype=np.float32))


def test_preprocess_bundle_uses_depth_fallback_for_missing_normal():
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    depth = np.full((16, 16), 10.0, dtype=np.float32)
    normal = np.zeros((16, 16, 3), dtype=np.float32)
    mask = np.zeros((16, 16), dtype=np.int32)
    mask[4:12, 4:12] = 1
    bundle = SampleBundle(
        scene_id="scene",
        cam_id="cam",
        rgb=rgb,
        depth=depth,
        normal=normal,
        mask=mask,
    )

    processed = preprocess_bundle(bundle)

    assert processed.normal_cn is not None
    assert processed.meta["preprocessing"]["normal_source"] == "depth_fallback"
    assert np.all(processed.normal_cn[..., 2] == 255)
