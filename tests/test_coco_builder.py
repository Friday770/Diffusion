"""测试 COCO 数据集构建模块。"""

import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from pycocotools.coco import COCO

from blast_pile_diffusion.data.coco_builder import build_coco_dataset, mask_to_polygon, mask_to_rle


REPO_ROOT = Path(__file__).resolve().parent.parent


def load_build_script():
    script_path = REPO_ROOT / "scripts" / "06_build_coco_dataset.py"
    spec = importlib.util.spec_from_file_location("build_coco_dataset_script", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_mask_to_rle_basic():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:60, 30:70] = 1
    rle = mask_to_rle(mask)
    assert "counts" in rle
    assert "size" in rle
    assert rle["size"] == [100, 100]


def test_mask_to_polygon_basic():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:60, 30:70] = 1
    polys = mask_to_polygon(mask)
    assert len(polys) >= 1
    assert len(polys[0]) >= 6
    for p in polys:
        assert all(isinstance(v, float) for v in p)


def test_mask_to_polygon_empty():
    mask = np.zeros((100, 100), dtype=np.uint8)
    polys = mask_to_polygon(mask)
    assert polys == []


def write_rgb(path: Path, shape: tuple[int, int] = (32, 32)) -> None:
    h, w = shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = 80
    img[:, :, 1] = 120
    img[:, :, 2] = 160
    cv2.imwrite(str(path), img)


def test_build_coco_dataset_loads_with_pycocotools(tmp_path: Path):
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()

    write_rgb(image_dir / "sample_a.png")
    mask_a = np.zeros((32, 32), dtype=np.uint16)
    mask_a[2:10, 3:12] = 1
    mask_a[14:25, 18:30] = 2
    cv2.imwrite(str(mask_dir / "sample_a.png"), mask_a)

    write_rgb(image_dir / "sample_b.png")
    mask_b = np.zeros((32, 32), dtype=np.uint16)
    mask_b[5:20, 6:18] = 7
    cv2.imwrite(str(mask_dir / "sample_b.png"), mask_b)

    output_json = tmp_path / "annotations.json"
    dataset = build_coco_dataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        output_json=output_json,
        require_all_images=True,
    )

    assert len(dataset["images"]) == 2
    assert len(dataset["annotations"]) == 3
    assert len({image["id"] for image in dataset["images"]}) == 2
    assert len({ann["id"] for ann in dataset["annotations"]}) == 3

    coco = COCO(str(output_json))
    assert len(coco.getImgIds()) == 2
    assert len(coco.getAnnIds()) == 3

    for ann in coco.loadAnns(coco.getAnnIds()):
        decoded = coco.annToMask(ann)
        assert int(decoded.sum()) == ann["area"]
        ys, xs = np.where(decoded > 0)
        x, y, w, h = ann["bbox"]
        assert x == int(xs.min())
        assert y == int(ys.min())
        assert w == int(xs.max() - xs.min() + 1)
        assert h == int(ys.max() - ys.min() + 1)


def test_build_coco_dataset_rejects_extra_unpaired_images(tmp_path: Path):
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()

    write_rgb(image_dir / "sample_a.png")
    write_rgb(image_dir / "orphan.png")
    mask = np.zeros((32, 32), dtype=np.uint16)
    mask[2:10, 3:12] = 1
    cv2.imwrite(str(mask_dir / "sample_a.png"), mask)

    try:
        build_coco_dataset(
            image_dir=image_dir,
            mask_dir=mask_dir,
            output_json=tmp_path / "annotations.json",
            require_all_images=True,
        )
    except ValueError as exc:
        assert "没有对应 mask" in str(exc)
    else:
        raise AssertionError("extra unpaired image should fail strict COCO assembly")


def test_collect_passed_samples_only_uses_qc_true(tmp_path: Path):
    script = load_build_script()
    generated_dir = tmp_path / "generated"
    preprocessed_dir = tmp_path / "preprocessed"
    output_images = tmp_path / "final" / "images"
    output_masks = tmp_path / "final" / "_masks"

    for sample_name, passed in [
        ("scene--cam0--s42", True),
        ("scene--cam0--s43", False),
    ]:
        sample_dir = generated_dir / sample_name
        sample_dir.mkdir(parents=True)
        write_rgb(sample_dir / "generated.png")
        with open(sample_dir / "qc.json", "w") as f:
            json.dump({"passed": passed}, f)
        with open(sample_dir / "meta.json", "w") as f:
            json.dump({"sample_key": "scene--cam0"}, f)

    mask_dir = preprocessed_dir / "scene--cam0"
    mask_dir.mkdir(parents=True)
    mask = np.zeros((32, 32), dtype=np.uint16)
    mask[4:16, 5:18] = 1
    cv2.imwrite(str(mask_dir / "mask_instance.png"), mask)

    count = script.collect_passed_samples(
        generated_dir,
        preprocessed_dir,
        output_images,
        output_masks,
    )

    assert count == 1
    assert sorted(path.name for path in output_images.glob("*.png")) == ["scene--cam0--s42.png"]
    assert sorted(path.name for path in output_masks.glob("*.png")) == ["scene--cam0--s42.png"]
