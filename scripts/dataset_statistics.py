#!/usr/bin/env python3
"""COCO 数据集统计与 spot-check overlay 生成脚本。"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from blast_pile_diffusion.data.coco_builder import validate_coco_dataset


AREA_CLASS_ORDER = ("small", "medium", "large", "boulder")


@dataclass(frozen=True)
class AreaBins:
    """面积分档阈值，单位为像素面积。"""

    small_max: int = 32 * 32
    medium_max: int = 96 * 96
    large_max: int = 256 * 256


def load_coco_json(annotation_path: Path) -> dict[str, Any]:
    with open(annotation_path) as f:
        dataset = json.load(f)
    validate_coco_dataset(dataset)
    return dataset


def classify_area(area: float, bins: AreaBins) -> str:
    if area < bins.small_max:
        return "small"
    if area < bins.medium_max:
        return "medium"
    if area < bins.large_max:
        return "large"
    return "boulder"


def _safe_percent(count: int, total: int) -> float:
    return round(count / total * 100, 2) if total else 0.0


def _numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None, "p90": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
    }


def compute_statistics(
    dataset: dict[str, Any],
    bins: AreaBins = AreaBins(),
    target_gradation: dict[str, float] | None = None,
) -> dict[str, Any]:
    """计算面积、实例数、分辨率和目标级配偏差。"""
    images = dataset.get("images", [])
    annotations = dataset.get("annotations", [])
    image_ids = [image["id"] for image in images]
    instances_per_image = {image_id: 0 for image_id in image_ids}

    areas: list[float] = []
    class_counts: Counter[str] = Counter({name: 0 for name in AREA_CLASS_ORDER})
    for ann in annotations:
        instances_per_image[ann["image_id"]] = instances_per_image.get(ann["image_id"], 0) + 1
        area = float(ann["area"])
        areas.append(area)
        class_counts[classify_area(area, bins)] += 1

    resolution_counts = Counter(f"{image['width']}x{image['height']}" for image in images)
    instance_counts = list(instances_per_image.values())
    total_annotations = len(annotations)
    class_percentages = {
        name: _safe_percent(class_counts[name], total_annotations) for name in AREA_CLASS_ORDER
    }

    target_comparison = None
    if target_gradation:
        target_comparison = {}
        for name in AREA_CLASS_ORDER:
            target = float(target_gradation.get(name, 0.0))
            observed = class_percentages[name]
            target_comparison[name] = {
                "observed_percent": observed,
                "target_percent": target,
                "delta_percentage_points": round(observed - target, 2),
            }

    return {
        "summary": {
            "total_images": len(images),
            "total_annotations": total_annotations,
            "avg_instances_per_image": round(
                total_annotations / max(len(images), 1), 4
            ),
        },
        "area_bins": asdict(bins),
        "area_summary": _numeric_summary(areas),
        "instances_per_image_summary": _numeric_summary([float(v) for v in instance_counts]),
        "area_class_counts": {name: int(class_counts[name]) for name in AREA_CLASS_ORDER},
        "area_class_percentages": class_percentages,
        "target_gradation_comparison": target_comparison,
        "instances_per_image": {
            str(image_id): int(count) for image_id, count in sorted(instances_per_image.items())
        },
        "resolution_distribution": dict(sorted(resolution_counts.items())),
        "areas": areas,
    }


def load_target_gradation(path: Path | None) -> dict[str, float] | None:
    if path is None:
        return None
    with open(path) as f:
        data = json.load(f)
    return {name: float(data[name]) for name in AREA_CLASS_ORDER if name in data}


def write_json_report(stats: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)


def write_markdown_report(stats: dict[str, Any], output_path: Path) -> None:
    summary = stats["summary"]
    area_summary = stats["area_summary"]
    instance_summary = stats["instances_per_image_summary"]
    lines = [
        "# Dataset Statistics",
        "",
        "This report was generated from the COCO annotations file.",
        "",
        "## Summary",
        "",
        f"- Images: {summary['total_images']}",
        f"- Annotations: {summary['total_annotations']}",
        f"- Average instances per image: {summary['avg_instances_per_image']}",
        "",
        "## Area Distribution",
        "",
        f"- Count: {area_summary['count']}",
        f"- Min area: {area_summary['min']}",
        f"- Median area: {area_summary['median']}",
        f"- Mean area: {area_summary['mean']}",
        f"- P90 area: {area_summary['p90']}",
        f"- Max area: {area_summary['max']}",
        "",
        "## Instance Count Per Image",
        "",
        f"- Min: {instance_summary['min']}",
        f"- Median: {instance_summary['median']}",
        f"- Mean: {instance_summary['mean']}",
        f"- P90: {instance_summary['p90']}",
        f"- Max: {instance_summary['max']}",
        "",
        "## Area Classes",
        "",
    ]
    for name in AREA_CLASS_ORDER:
        count = stats["area_class_counts"][name]
        pct = stats["area_class_percentages"][name]
        lines.append(f"- {name}: {count} ({pct}%)")

    comparison = stats.get("target_gradation_comparison")
    lines.extend(["", "## Target Gradation Comparison", ""])
    if comparison:
        for name in AREA_CLASS_ORDER:
            item = comparison[name]
            lines.append(
                f"- {name}: observed {item['observed_percent']}%, "
                f"target {item['target_percent']}%, "
                f"delta {item['delta_percentage_points']} pp"
            )
    else:
        lines.append(
            "- No target gradation JSON was provided; only observed distribution is reported."
        )

    lines.extend(["", "## Resolution Distribution", ""])
    for resolution, count in stats["resolution_distribution"].items():
        lines.append(f"- {resolution}: {count}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def _new_figure(title: str):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.title(title)
    return plt


def plot_statistics(stats: dict[str, Any], output_dir: Path) -> list[Path]:
    """保存面积、实例数、级配和分辨率图表。"""
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    areas = [area for area in stats["areas"] if area > 0]

    plt = _new_figure("Instance Area Distribution")
    if areas:
        bins = min(50, max(10, int(np.sqrt(len(areas)))))
        plt.hist(areas, bins=bins, color="#4f8ad9", edgecolor="black")
        plt.xscale("log")
        for label, value in stats["area_bins"].items():
            plt.axvline(value, linestyle="--", linewidth=1, label=label)
        plt.legend()
    else:
        plt.text(0.5, 0.5, "No annotations", ha="center", va="center")
    plt.xlabel("Area (pixels, log scale)")
    plt.ylabel("Instance count")
    path = output_dir / "area_distribution.png"
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    written.append(path)

    counts = list(stats["instances_per_image"].values())
    plt = _new_figure("Instances Per Image")
    if counts:
        bins = range(0, max(counts) + 2)
        plt.hist(counts, bins=bins, color="#5aa469", edgecolor="black", align="left")
    else:
        plt.text(0.5, 0.5, "No images", ha="center", va="center")
    plt.xlabel("Instances per image")
    plt.ylabel("Image count")
    path = output_dir / "instances_per_image.png"
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    written.append(path)

    plt = _new_figure("Area Class Distribution")
    labels = list(AREA_CLASS_ORDER)
    values = [stats["area_class_percentages"][label] for label in labels]
    plt.bar(labels, values, color=["#77aadd", "#99dd99", "#eecc66", "#ee8866"])
    plt.ylabel("Annotations (%)")
    path = output_dir / "area_class_distribution.png"
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    written.append(path)

    plt = _new_figure("Resolution Distribution")
    resolutions = list(stats["resolution_distribution"].keys())[:20]
    values = [stats["resolution_distribution"][resolution] for resolution in resolutions]
    if resolutions:
        plt.bar(resolutions, values, color="#8f79bd")
        plt.xticks(rotation=35, ha="right")
    else:
        plt.text(0.5, 0.5, "No images", ha="center", va="center")
    plt.ylabel("Image count")
    path = output_dir / "resolution_distribution.png"
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    written.append(path)

    return written


def _color_for_index(index: int) -> np.ndarray:
    rng = np.random.default_rng(index + 17)
    return rng.integers(40, 230, size=3, dtype=np.uint8)


def make_overlay(image_rgb: np.ndarray, masks: list[np.ndarray], alpha: float = 0.45) -> np.ndarray:
    overlay = image_rgb.copy()
    for idx, mask in enumerate(masks):
        mask_bool = mask.astype(bool)
        if not mask_bool.any():
            continue
        color = _color_for_index(idx)
        overlay[mask_bool] = (
            overlay[mask_bool].astype(np.float32) * (1 - alpha)
            + color.astype(np.float32) * alpha
        ).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, color.tolist(), 1)
    return overlay


def save_spotcheck_overlays(
    annotation_path: Path,
    image_dir: Path,
    output_dir: Path,
    sample_count: int = 20,
    seed: int = 42,
) -> list[Path]:
    """随机抽样图像并保存 pycocotools 解码后的实例 overlay。"""
    from pycocotools.coco import COCO

    output_dir.mkdir(parents=True, exist_ok=True)
    coco = COCO(str(annotation_path))
    image_ids = sorted(coco.getImgIds())
    if not image_ids or sample_count <= 0:
        return []

    rng = random.Random(seed)
    chosen_ids = rng.sample(image_ids, k=min(sample_count, len(image_ids)))
    written: list[Path] = []

    for image_id in chosen_ids:
        image_info = coco.loadImgs([image_id])[0]
        image_path = image_dir / image_info["file_name"]
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            print(f"[WARN] spot-check 图像不存在，跳过: {image_path}")
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        ann_ids = coco.getAnnIds(imgIds=[image_id])
        anns = coco.loadAnns(ann_ids)
        masks = [coco.annToMask(ann) for ann in anns]
        overlay = make_overlay(image_rgb, masks)

        stem = Path(image_info["file_name"]).stem
        output_path = output_dir / f"spotcheck_{image_id}_{stem}.png"
        cv2.imwrite(str(output_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        written.append(output_path)

    return written


def default_image_dir(annotation_path: Path) -> Path:
    return annotation_path.parent / "images"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="统计 COCO 数据集并生成 spot-check overlay"
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/final_dataset/train/annotations.json"),
        help="COCO annotations.json 路径",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="图像目录，默认与标注同级 images/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/dataset_samples"),
        help="统计图、报告和 spot-check overlay 输出目录",
    )
    parser.add_argument("--sample-count", type=int, default=20, help="spot-check 图像数量")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--small-max-area", type=int, default=AreaBins.small_max)
    parser.add_argument("--medium-max-area", type=int, default=AreaBins.medium_max)
    parser.add_argument("--large-max-area", type=int, default=AreaBins.large_max)
    parser.add_argument(
        "--target-gradation-json",
        type=Path,
        default=None,
        help="可选 JSON，键为 small/medium/large/boulder，值为目标百分比",
    )
    parser.add_argument("--no-plots", action="store_true", help="不生成统计图")
    parser.add_argument("--no-overlays", action="store_true", help="不生成 spot-check overlay")
    args = parser.parse_args()

    image_dir = args.image_dir or default_image_dir(args.annotations)
    output_dir = args.output_dir
    bins = AreaBins(args.small_max_area, args.medium_max_area, args.large_max_area)

    dataset = load_coco_json(args.annotations)
    target_gradation = load_target_gradation(args.target_gradation_json)
    stats = compute_statistics(dataset, bins=bins, target_gradation=target_gradation)

    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_plots:
        try:
            stats["plot_files"] = [str(path) for path in plot_statistics(stats, output_dir)]
        except ImportError as exc:
            print(f"[WARN] matplotlib 不可用，跳过统计图生成: {exc}")
            stats["plot_files"] = []
    else:
        stats["plot_files"] = []

    if not args.no_overlays:
        stats["spotcheck_overlays"] = [
            str(path)
            for path in save_spotcheck_overlays(
                args.annotations,
                image_dir,
                output_dir,
                sample_count=args.sample_count,
                seed=args.seed,
            )
        ]
    else:
        stats["spotcheck_overlays"] = []

    json_report = output_dir / "dataset_statistics.json"
    md_report = output_dir / "dataset_statistics.md"
    write_json_report(stats, json_report)
    write_markdown_report(stats, md_report)

    summary = stats["summary"]
    print("=== Dataset Statistics ===")
    print(f"images: {summary['total_images']}")
    print(f"annotations: {summary['total_annotations']}")
    print(f"avg instances/image: {summary['avg_instances_per_image']}")
    print(f"report: {json_report}")
    print(f"spot-check overlays: {len(stats['spotcheck_overlays'])}")


if __name__ == "__main__":
    main()
