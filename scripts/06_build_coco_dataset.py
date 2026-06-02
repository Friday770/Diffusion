#!/usr/bin/env python3
"""Step 6: 组装最终 COCO 格式数据集。

收集所有 QC 通过的生成图 + 对应的实例掩码，
组装为 COCO instance segmentation JSON 格式，
写入 data/final_dataset/。
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from blast_pile_diffusion.data.coco_builder import build_coco_dataset, validate_coco_dataset


SEED_SUFFIX_RE = re.compile(r"^(?P<sample_key>.+?)(?:--s|_s)(?P<seed>\d+)$")


@dataclass(frozen=True)
class PassedSample:
    sample_dir: Path
    sample_key: str
    generated_path: Path
    mask_path: Path
    output_name: str


def infer_sample_key(sample_dir: Path) -> str:
    """从 meta.json 或目录名恢复预处理样本 key。"""
    meta_path = sample_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        sample_key = meta.get("sample_key")
        if isinstance(sample_key, str) and sample_key:
            return sample_key

    match = SEED_SUFFIX_RE.match(sample_dir.name)
    if not match:
        raise ValueError(f"无法从生成目录名解析 sample_key: {sample_dir.name}")
    return match.group("sample_key")


def discover_passed_samples(
    generated_dir: Path,
    preprocessed_dir: Path,
    allow_missing_masks: bool = False,
) -> list[PassedSample]:
    """扫描 generated_dir，返回 qc.json 中 passed=true 的样本。"""
    if not generated_dir.exists():
        return []

    samples: list[PassedSample] = []
    missing_masks: list[str] = []
    output_names: set[str] = set()

    for sample_dir in sorted(path for path in generated_dir.iterdir() if path.is_dir()):
        qc_path = sample_dir / "qc.json"
        gen_path = sample_dir / "generated.png"
        if not qc_path.exists() or not gen_path.exists():
            continue

        with open(qc_path) as f:
            qc = json.load(f)
        if qc.get("passed") is not True:
            continue

        sample_key = infer_sample_key(sample_dir)
        mask_path = preprocessed_dir / sample_key / "mask_instance.png"
        if not mask_path.exists():
            missing_masks.append(f"{sample_dir.name} -> {mask_path}")
            if allow_missing_masks:
                continue
            continue

        output_name = f"{sample_dir.name}.png"
        if output_name in output_names:
            raise ValueError(f"输出文件名重复: {output_name}")
        output_names.add(output_name)

        samples.append(
            PassedSample(
                sample_dir=sample_dir,
                sample_key=sample_key,
                generated_path=gen_path,
                mask_path=mask_path,
                output_name=output_name,
            )
        )

    if missing_masks and not allow_missing_masks:
        details = "\n".join(f"  - {item}" for item in missing_masks[:20])
        raise FileNotFoundError(f"QC 通过样本缺少原始 mask:\n{details}")

    return samples


def replace_dir(path: Path) -> None:
    """清理并重建输出子目录，避免旧样本污染本次构建。"""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def collect_passed_samples(
    generated_dir: Path,
    preprocessed_dir: Path,
    output_image_dir: Path,
    output_mask_dir: Path,
    allow_missing_masks: bool = False,
) -> int:
    """收集 QC 通过的样本，复制到输出目录。"""
    samples = discover_passed_samples(
        generated_dir, preprocessed_dir, allow_missing_masks=allow_missing_masks
    )
    if not samples:
        return 0

    replace_dir(output_image_dir)
    replace_dir(output_mask_dir)

    for sample in samples:
        shutil.copy2(sample.generated_path, output_image_dir / sample.output_name)
        shutil.copy2(sample.mask_path, output_mask_dir / sample.output_name)

    return len(samples)


def directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def read_qc_report_passed(generated_dir: Path) -> int | None:
    report_path = generated_dir / "qc_report.json"
    if not report_path.exists():
        return None
    with open(report_path) as f:
        report = json.load(f)
    passed = report.get("summary", {}).get("passed")
    return int(passed) if passed is not None else None


def validate_with_pycocotools(annotation_path: Path) -> None:
    from pycocotools.coco import COCO

    COCO(str(annotation_path))


def main():
    parser = argparse.ArgumentParser(description="组装 COCO 数据集")
    parser.add_argument("--generated-dir", type=Path, default=Path("data/generated"))
    parser.add_argument("--preprocessed-dir", type=Path, default=Path("data/preprocessed"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/final_dataset/train"))
    parser.add_argument(
        "--allow-missing-masks",
        action="store_true",
        help="跳过缺少 mask 的 passed 样本；默认严格报错",
    )
    parser.add_argument(
        "--keep-masks",
        action="store_true",
        help="保留组装过程中的 _masks 目录，便于调试",
    )
    args = parser.parse_args()

    tmp_images = args.output_dir / "images"
    tmp_masks = args.output_dir / "_masks"

    print("=== Step 6a: 收集 QC 通过的样本 ===")
    count = collect_passed_samples(
        args.generated_dir,
        args.preprocessed_dir,
        tmp_images,
        tmp_masks,
        allow_missing_masks=args.allow_missing_masks,
    )
    print(f"收集 {count} 个通过 QC 的样本")

    if count == 0:
        print("[WARN] 没有通过 QC 的样本，请先运行 scripts/05_run_qc.py")
        return

    report_passed = read_qc_report_passed(args.generated_dir)
    if report_passed is not None and report_passed != count:
        print(f"[WARN] qc_report.json passed={report_passed}，本次实际收集={count}")

    print("\n=== Step 6b: 构建 COCO JSON ===")
    annotation_path = args.output_dir / "annotations.json"
    dataset = build_coco_dataset(
        image_dir=tmp_images,
        mask_dir=tmp_masks,
        output_json=annotation_path,
        require_all_images=True,
    )
    validate_coco_dataset(dataset)
    validate_with_pycocotools(annotation_path)

    n_images = len(dataset["images"])
    n_anns = len(dataset["annotations"])
    avg_instances = n_anns / max(n_images, 1)

    if not args.keep_masks and tmp_masks.exists():
        shutil.rmtree(tmp_masks)

    disk_bytes = directory_size_bytes(args.output_dir)

    print(f"COCO 数据集: {n_images} 张图, {n_anns} 个标注实例")
    print(f"平均每图实例数: {avg_instances:.2f}")
    print(f"磁盘占用: {disk_bytes / (1024 ** 2):.2f} MiB")
    print(f"标注文件 → {annotation_path}")

    summary_path = args.output_dir / "dataset_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "images": n_images,
                "annotations": n_anns,
                "avg_instances_per_image": avg_instances,
                "disk_bytes": disk_bytes,
                "annotation_path": str(annotation_path),
            },
            f,
            indent=2,
        )
    print(f"摘要文件 → {summary_path}")

    print(f"\n数据集构建完成 → {args.output_dir}")


if __name__ == "__main__":
    main()
