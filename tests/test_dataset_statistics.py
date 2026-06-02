"""测试 COCO 数据集统计脚本。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np

from blast_pile_diffusion.data.coco_builder import build_coco_dataset


REPO_ROOT = Path(__file__).resolve().parent.parent


def load_statistics_script():
    script_path = REPO_ROOT / "scripts" / "dataset_statistics.py"
    spec = importlib.util.spec_from_file_location("dataset_statistics_script", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_rgb(path: Path, shape: tuple[int, int] = (80, 80)) -> None:
    h, w = shape
    image = np.zeros((h, w, 3), dtype=np.uint8)
    image[:, :, 0] = 70
    image[:, :, 1] = 110
    image[:, :, 2] = 150
    cv2.imwrite(str(path), image)


def make_test_dataset(tmp_path: Path) -> tuple[Path, Path]:
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()

    write_rgb(image_dir / "a.png")
    mask_a = np.zeros((80, 80), dtype=np.uint16)
    mask_a[0:3, 0:3] = 1
    mask_a[10:15, 10:15] = 2
    cv2.imwrite(str(mask_dir / "a.png"), mask_a)

    write_rgb(image_dir / "b.png")
    mask_b = np.zeros((80, 80), dtype=np.uint16)
    mask_b[20:40, 20:40] = 3
    cv2.imwrite(str(mask_dir / "b.png"), mask_b)

    write_rgb(image_dir / "c.png")
    mask_c = np.zeros((80, 80), dtype=np.uint16)
    mask_c[5:55, 5:55] = 4
    cv2.imwrite(str(mask_dir / "c.png"), mask_c)

    annotation_path = tmp_path / "annotations.json"
    build_coco_dataset(image_dir, mask_dir, annotation_path, require_all_images=True)
    return annotation_path, image_dir


def test_compute_statistics_area_classes(tmp_path: Path):
    stats_script = load_statistics_script()
    annotation_path, _ = make_test_dataset(tmp_path)
    dataset = stats_script.load_coco_json(annotation_path)

    stats = stats_script.compute_statistics(
        dataset,
        bins=stats_script.AreaBins(small_max=10, medium_max=100, large_max=1000),
        target_gradation={"small": 25, "medium": 25, "large": 25, "boulder": 25},
    )

    assert stats["summary"]["total_images"] == 3
    assert stats["summary"]["total_annotations"] == 4
    assert stats["area_class_counts"] == {
        "small": 1,
        "medium": 1,
        "large": 1,
        "boulder": 1,
    }
    assert stats["area_class_percentages"] == {
        "small": 25.0,
        "medium": 25.0,
        "large": 25.0,
        "boulder": 25.0,
    }
    assert stats["target_gradation_comparison"]["small"]["delta_percentage_points"] == 0.0
    assert stats["resolution_distribution"] == {"80x80": 3}


def test_spotcheck_overlay_outputs_png(tmp_path: Path):
    stats_script = load_statistics_script()
    annotation_path, image_dir = make_test_dataset(tmp_path)
    output_dir = tmp_path / "samples"

    overlays = stats_script.save_spotcheck_overlays(
        annotation_path,
        image_dir,
        output_dir,
        sample_count=2,
        seed=7,
    )

    assert len(overlays) == 2
    for path in overlays:
        assert path.exists()
        assert cv2.imread(str(path)) is not None


def test_statistics_reports_are_written(tmp_path: Path):
    stats_script = load_statistics_script()
    annotation_path, _ = make_test_dataset(tmp_path)
    dataset = stats_script.load_coco_json(annotation_path)
    stats = stats_script.compute_statistics(dataset)

    json_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"
    stats_script.write_json_report(stats, json_path)
    stats_script.write_markdown_report(stats, md_path)

    assert json_path.exists()
    assert md_path.exists()
    assert "Dataset Statistics" in md_path.read_text()
