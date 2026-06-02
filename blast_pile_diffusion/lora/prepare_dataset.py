"""构建 kohya_ss / sd-scripts 兼容的 LoRA 训练目录结构。

sd-scripts 期望的目录格式：
  train_data/
    {repeat}_{concept}/
      image1.png
      image1.txt       (caption)
      image2.png
      image2.txt
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from blast_pile_diffusion.data.caption_generator import (
    IMAGE_EXTENSIONS,
    TRIGGER_WORD,
    ensure_trigger_word,
)


@dataclass(frozen=True)
class KohyaDatasetReport:
    concept_dir: Path
    image_count: int
    caption_count: int
    missing_captions: list[str]
    captions_without_trigger: list[str]

    @property
    def valid(self) -> bool:
        return (
            self.image_count > 0
            and self.image_count == self.caption_count
            and not self.missing_captions
            and not self.captions_without_trigger
        )


def iter_image_files(image_dir: Path) -> list[Path]:
    """Return supported image files sorted by stable filename order."""
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    if not image_dir.is_dir():
        raise NotADirectoryError(f"Image path is not a directory: {image_dir}")
    return sorted(
        (p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda p: p.name.lower(),
    )


def _concept_dir(output_dir: Path, repeats: int, concept_name: str) -> Path:
    if repeats <= 0:
        raise ValueError("repeats must be a positive integer")
    if not concept_name:
        raise ValueError("concept_name must not be empty")
    return output_dir / f"{repeats}_{concept_name}"


def build_kohya_dataset(
    image_dir: Path,
    caption_dir: Path,
    output_dir: Path,
    concept_name: str = "mine_blast_pile",
    repeats: int = 10,
    trigger_word: str = TRIGGER_WORD,
    strict_captions: bool = True,
    clean: bool = False,
) -> Path:
    """
    将图像+caption 组装为 sd-scripts 训练目录。

    Args:
        image_dir: 原始图片目录
        caption_dir: caption .txt 文件目录（文件名须与图片对应）
        output_dir: 输出的 sd-scripts 训练根目录
        concept_name: 概念名称
        repeats: 每张图每 epoch 重复次数
        trigger_word: LoRA 触发词，默认 <mine_blast_pile>
        strict_captions: True 时缺失 caption 直接报错
        clean: True 时先清空目标 concept 子目录，避免旧文件残留

    Returns:
        训练数据子目录路径
    """
    images = iter_image_files(image_dir)
    if not images:
        raise ValueError(f"No supported images found in {image_dir}")

    concept_dir = _concept_dir(output_dir, repeats=repeats, concept_name=concept_name)
    if clean and concept_dir.exists():
        shutil.rmtree(concept_dir)
    concept_dir.mkdir(parents=True, exist_ok=True)

    missing: list[Path] = []
    for img_path in images:
        caption_path = caption_dir / f"{img_path.stem}.txt"
        if not caption_path.exists():
            missing.append(caption_path)

    if missing and strict_captions:
        missing_list = ", ".join(str(path) for path in missing[:5])
        if len(missing) > 5:
            missing_list += f", ... ({len(missing)} total)"
        raise FileNotFoundError(f"Missing caption file(s): {missing_list}")

    for img_path in images:
        shutil.copy2(img_path, concept_dir / img_path.name)

        caption_path = caption_dir / f"{img_path.stem}.txt"
        target_caption = concept_dir / f"{img_path.stem}.txt"
        if caption_path.exists():
            caption = caption_path.read_text(encoding="utf-8").strip()
        else:
            caption = "open-pit mine muck pile, fresh angular rock fragments, photorealistic"
        target_caption.write_text(
            ensure_trigger_word(caption, trigger_word=trigger_word),
            encoding="utf-8",
        )

    report = validate_kohya_dataset(concept_dir, trigger_word=trigger_word)
    if not report.valid:
        raise ValueError(
            "Built kohya dataset failed validation: "
            f"images={report.image_count}, captions={report.caption_count}, "
            f"missing={len(report.missing_captions)}, "
            f"no_trigger={len(report.captions_without_trigger)}"
        )

    print(f"构建完成：{report.image_count} 对图像+caption -> {concept_dir}")
    return concept_dir


def validate_kohya_dataset(
    concept_dir: Path,
    trigger_word: str = TRIGGER_WORD,
) -> KohyaDatasetReport:
    """Validate that a kohya concept directory has paired images/captions."""
    if not concept_dir.exists():
        raise FileNotFoundError(f"Kohya concept directory does not exist: {concept_dir}")
    if not concept_dir.is_dir():
        raise NotADirectoryError(f"Kohya concept path is not a directory: {concept_dir}")

    images = sorted(
        (p for p in concept_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda p: p.name.lower(),
    )
    captions = sorted(concept_dir.glob("*.txt"), key=lambda p: p.name.lower())
    caption_by_stem = {path.stem: path for path in captions}

    missing_captions = [
        str(image_path.with_suffix(".txt"))
        for image_path in images
        if image_path.stem not in caption_by_stem
    ]
    captions_without_trigger = []
    for caption_path in captions:
        text = caption_path.read_text(encoding="utf-8")
        if trigger_word not in text:
            captions_without_trigger.append(str(caption_path))

    return KohyaDatasetReport(
        concept_dir=concept_dir,
        image_count=len(images),
        caption_count=len(captions),
        missing_captions=missing_captions,
        captions_without_trigger=captions_without_trigger,
    )
