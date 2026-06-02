"""将 QC 通过的生成样本组装为 COCO 实例分割格式数据集。

输出格式符合 COCO annotations JSON schema：
{
  "images": [...],
  "annotations": [...],
  "categories": [{"id": 1, "name": "rock"}]
}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


COCO_CATEGORIES = [{"id": 1, "name": "rock", "supercategory": "fragment"}]
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


def mask_to_rle(binary_mask: np.ndarray) -> dict:
    """将二值掩码转为 COCO RLE 编码。"""
    from pycocotools import mask as mask_utils

    fortran_mask = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = mask_utils.encode(fortran_mask)
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def mask_to_polygon(binary_mask: np.ndarray, tolerance: float = 1.0) -> list[list[float]]:
    """将二值掩码转为多边形坐标列表。"""
    contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1
    )
    polygons = []
    for contour in contours:
        if contour.shape[0] < 3:
            continue
        approx = cv2.approxPolyDP(contour, tolerance, True)
        polygon = approx.astype(np.float32).flatten().tolist()
        if len(polygon) >= 6:
            polygons.append(polygon)
    return polygons


def _read_image_shape(image_path: Path) -> tuple[int, int]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    h, w = image.shape[:2]
    return h, w


def _read_instance_mask(mask_path: Path) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"无法读取实例掩码: {mask_path}")
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask.astype(np.int32)


def _candidate_image_paths(image_dir: Path, mask_path: Path) -> list[Path]:
    candidates = [image_dir / mask_path.name]
    candidates.extend(image_dir / f"{mask_path.stem}{suffix}" for suffix in IMAGE_SUFFIXES)

    if "mask_instance" in mask_path.stem:
        generated_stem = mask_path.stem.replace("mask_instance", "generated")
        candidates.extend(image_dir / f"{generated_stem}{suffix}" for suffix in IMAGE_SUFFIXES)

    unique_candidates = []
    seen = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        unique_candidates.append(path)
    return unique_candidates


def find_image_for_mask(image_dir: Path, mask_path: Path) -> Path:
    """按 COCO 组装约定寻找与 mask 一一对应的 generated 图。"""
    exact_path = image_dir / mask_path.name
    if exact_path.exists():
        return exact_path

    matches = [path for path in _candidate_image_paths(image_dir, mask_path) if path.exists()]
    if not matches:
        raise FileNotFoundError(f"找不到与掩码对应的图像: {mask_path.name}")
    if len(matches) > 1:
        match_names = ", ".join(path.name for path in matches)
        raise ValueError(f"掩码 {mask_path.name} 对应多个候选图像: {match_names}")
    return matches[0]


def _list_image_files(image_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def validate_coco_dataset(dataset: dict) -> None:
    """做轻量 schema/唯一性检查；完整解码验证交给 pycocotools。"""
    image_ids = [image["id"] for image in dataset.get("images", [])]
    annotation_ids = [ann["id"] for ann in dataset.get("annotations", [])]

    if len(image_ids) != len(set(image_ids)):
        raise ValueError("COCO images 中存在重复 image_id")
    if len(annotation_ids) != len(set(annotation_ids)):
        raise ValueError("COCO annotations 中存在重复 annotation_id")

    image_by_id = {image["id"]: image for image in dataset.get("images", [])}
    for ann in dataset.get("annotations", []):
        if ann["image_id"] not in image_by_id:
            raise ValueError(f"annotation {ann['id']} 引用了不存在的 image_id")
        if ann.get("category_id") != 1:
            raise ValueError(f"annotation {ann['id']} 的 category_id 不是 rock=1")
        bbox = ann.get("bbox")
        if not bbox or len(bbox) != 4:
            raise ValueError(f"annotation {ann['id']} 缺少合法 bbox")
        x, y, w, h = bbox
        image = image_by_id[ann["image_id"]]
        if w <= 0 or h <= 0:
            raise ValueError(f"annotation {ann['id']} 的 bbox 尺寸非法")
        if x < 0 or y < 0 or x + w > image["width"] or y + h > image["height"]:
            raise ValueError(f"annotation {ann['id']} 的 bbox 超出图像范围")
        if ann.get("area", 0) <= 0:
            raise ValueError(f"annotation {ann['id']} 的 area 非法")
        if "segmentation" not in ann:
            raise ValueError(f"annotation {ann['id']} 缺少 segmentation")

    categories = dataset.get("categories", [])
    if categories != COCO_CATEGORIES:
        raise ValueError("COCO categories 必须只包含 rock 类别")


def build_coco_dataset(
    image_dir: Path,
    mask_dir: Path,
    output_json: Path,
    annotation_format: str = "rle",
    image_ids_start: int = 1,
    annotation_ids_start: int = 1,
    existing_json: Optional[Path] = None,
    min_area: int = 1,
    require_all_images: bool = False,
) -> dict:
    """
    扫描 image_dir 和 mask_dir，组装 COCO JSON。

    mask_dir 下每个文件应为实例掩码（uint16 PNG），像素值=实例ID，0=背景。
    image_dir 下文件名与 mask_dir 一一对应。
    """
    if annotation_format not in {"rle", "polygon"}:
        raise ValueError("annotation_format 必须是 'rle' 或 'polygon'")
    if min_area < 1:
        raise ValueError("min_area 必须 >= 1")

    images: list[dict] = []
    annotations: list[dict] = []
    img_id = image_ids_start
    ann_id = annotation_ids_start

    if existing_json and existing_json.exists():
        with open(existing_json) as f:
            existing = json.load(f)
        images = existing.get("images", [])
        annotations = existing.get("annotations", [])
        if images:
            img_id = max(i["id"] for i in images) + 1
        if annotations:
            ann_id = max(a["id"] for a in annotations) + 1

    mask_paths = sorted(mask_dir.glob("*.png"))
    matched_images = set()

    for mask_path in mask_paths:
        image_path = find_image_for_mask(image_dir, mask_path)
        matched_images.add(image_path.resolve())

        h, w = _read_image_shape(image_path)
        mask = _read_instance_mask(mask_path)
        if mask.shape[:2] != (h, w):
            raise ValueError(
                f"图像与掩码尺寸不一致: {image_path.name} "
                f"image={(h, w)} mask={mask.shape[:2]}"
            )

        images.append(
            {"id": img_id, "file_name": image_path.name, "width": w, "height": h}
        )

        for instance_id in np.unique(mask):
            if instance_id == 0:
                continue

            binary = (mask == instance_id).astype(np.uint8)
            area = int(binary.sum())
            if area < min_area:
                continue

            ys, xs = np.where(binary)
            bbox = [
                int(xs.min()),
                int(ys.min()),
                int(xs.max() - xs.min() + 1),
                int(ys.max() - ys.min() + 1),
            ]

            ann = {
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": bbox,
                "area": area,
                "iscrowd": 0,
            }

            if annotation_format == "rle":
                ann["segmentation"] = mask_to_rle(binary)
            else:
                ann["segmentation"] = mask_to_polygon(binary)

            annotations.append(ann)
            ann_id += 1

        img_id += 1

    if require_all_images:
        extra_images = [
            path.name
            for path in _list_image_files(image_dir)
            if path.resolve() not in matched_images
        ]
        if extra_images:
            raise ValueError(f"存在没有对应 mask 的图像: {', '.join(extra_images)}")

    dataset = {
        "images": images,
        "annotations": annotations,
        "categories": COCO_CATEGORIES,
    }
    validate_coco_dataset(dataset)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(dataset, f, indent=2)

    return dataset
