"""测试 Unity Perception reader，使用合成 fixture 代替真实 Unity 导出。"""

import json
from pathlib import Path

import cv2
import numpy as np

from blast_pile_diffusion.data.sample_bundle import SampleBundle
from blast_pile_diffusion.data.unity_reader import find_unity_scenes, iter_bundles
from blast_pile_diffusion.preprocessing.preprocessor import preprocess_and_save


def _write_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def _make_unity_scene(root: Path, with_colormap: bool = True) -> Path:
    scene = root / "scene_001"
    rgb_path = scene / "RGB" / "rgb_0001.png"
    depth_path = scene / "Depth" / "depth_0001.png"
    normal_path = scene / "Normal" / "normal_0001.png"
    mask_path = scene / "InstanceSegmentation" / "instance_0001.png"
    captures_path = scene / "captures" / "captures_000.json"
    captures_path.parent.mkdir(parents=True, exist_ok=True)

    rgb = np.zeros((8, 10, 3), dtype=np.uint8)
    rgb[:, :, 0] = 80
    rgb[:, :, 1] = np.arange(10, dtype=np.uint8)[None, :] * 10
    rgb[:, :, 2] = 200
    _write_rgb(rgb_path, rgb)

    depth_m = np.linspace(0.5, 4.0, 80, dtype=np.float32).reshape(8, 10)
    depth_mm = (depth_m * 1000).astype(np.uint16)
    depth_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(depth_path), depth_mm)

    normal_rgb = np.zeros((8, 10, 3), dtype=np.uint8)
    normal_rgb[:, :, :] = [128, 128, 255]
    _write_rgb(normal_path, normal_rgb)

    mask_rgb = np.zeros((8, 10, 3), dtype=np.uint8)
    mask_rgb[1:4, 1:5] = [10, 20, 30]
    mask_rgb[4:7, 5:9] = [200, 50, 60]
    _write_rgb(mask_path, mask_rgb)

    values = []
    if with_colormap:
        values = [
            {"instance_id": 101, "color": [10, 20, 30]},
            {"instance_id": 202, "pixel_value": [200, 50, 60]},
        ]
    metadata = {
        "captures": [
            {
                "step": 1,
                "sensor": {"sensor_id": "cam-main"},
                "filename": "RGB/rgb_0001.png",
                "annotations": [
                    {
                        "annotation_definition": "Depth",
                        "filename": "Depth/depth_0001.png",
                    },
                    {
                        "annotation_definition": "Normal",
                        "filename": "Normal/normal_0001.png",
                    },
                    {
                        "annotation_definition": "InstanceSegmentation",
                        "filename": "InstanceSegmentation/instance_0001.png",
                        "values": values,
                    },
                ],
            }
        ]
    }
    captures_path.write_text(json.dumps(metadata))
    return scene


def test_find_unity_scenes_detects_capture_metadata(tmp_path: Path):
    scene = _make_unity_scene(tmp_path)
    assert find_unity_scenes(tmp_path) == [scene]


def test_iter_bundles_decodes_unity_perception_export(tmp_path: Path):
    scene = _make_unity_scene(tmp_path)
    bundles = list(iter_bundles(scene))
    assert len(bundles) == 1
    bundle = bundles[0]

    assert bundle.scene_id == "scene_001"
    assert bundle.cam_id == "cam-main_f000001"
    assert bundle.rgb.shape == (8, 10, 3)
    assert bundle.rgb.dtype == np.uint8
    assert bundle.depth.shape == (8, 10)
    assert bundle.depth.dtype == np.float32
    assert np.isclose(bundle.depth.min(), 0.5)
    assert np.isclose(bundle.depth.max(), 4.0)
    assert bundle.normal.shape == (8, 10, 3)
    assert bundle.normal.dtype == np.float32
    assert bundle.normal[:, :, 2].mean() > 0.9
    assert bundle.mask.shape == (8, 10)
    assert bundle.mask.dtype == np.int32
    assert set(np.unique(bundle.mask)) == {0, 101, 202}
    assert bundle.num_instances == 2


def test_iter_bundles_assigns_instance_ids_without_colormap(tmp_path: Path):
    scene = _make_unity_scene(tmp_path, with_colormap=False)
    bundle = next(iter_bundles(scene))
    ids = set(np.unique(bundle.mask))
    assert 0 in ids
    assert len(ids - {0}) == 2
    assert bundle.num_instances == 2


def test_preprocess_and_save_outputs_required_b5_files(tmp_path: Path):
    scene = _make_unity_scene(tmp_path)
    bundle = next(iter_bundles(scene))
    out_dir = tmp_path / "preprocessed"

    save_dir = preprocess_and_save(
        bundle,
        out_dir,
        depth_near_clip=0.5,
        depth_far_clip=5.0,
    )

    for name in ("rgb.png", "depth_cn.png", "canny_from_mask.png", "mask_instance.png", "meta.json"):
        assert (save_dir / name).exists()
    loaded = SampleBundle.load(save_dir)
    assert loaded.is_preprocessed()
    assert loaded.depth_cn.shape == (8, 10, 3)
    assert loaded.canny.shape == (8, 10)
    assert loaded.num_instances == 2
