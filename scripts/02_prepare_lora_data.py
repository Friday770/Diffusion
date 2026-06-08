#!/usr/bin/env python3
"""Step 2: 准备 LoRA 训练数据。

为 data/lora_real/images/ 下的真实爆堆图片生成 caption，
然后组装为 kohya_ss / sd-scripts 兼容的训练目录结构。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from blast_pile_diffusion.data.caption_generator import (  # noqa: E402
    TRIGGER_WORD,
    generate_captions_for_directory,
)
from blast_pile_diffusion.lora.prepare_dataset import (  # noqa: E402
    build_kohya_dataset,
    validate_kohya_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="准备 LoRA 训练数据")
    parser.add_argument("--image-dir", type=Path, default=Path("PhotoForLoRA"))
    parser.add_argument("--caption-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("data/lora_real/kohya_dataset"))
    parser.add_argument("--concept-name", default="mine_blast_pile")
    parser.add_argument("--trigger-word", default=TRIGGER_WORD)
    parser.add_argument(
        "--use-blip2",
        action="store_true",
        help="使用 BLIP-2 自动生成 caption",
    )
    parser.add_argument(
        "--generate-captions",
        action="store_true",
        help="Generate captions instead of using existing .txt files next to the images.",
    )
    parser.add_argument(
        "--skip-caption-generation",
        action="store_true",
    )
    parser.add_argument("--no-overwrite-captions", action="store_true")
    parser.add_argument("--allow-missing-captions", action="store_true")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="清空目标 concept 子目录后重新构建",
    )
    parser.add_argument("--repeats", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    caption_dir = args.caption_dir or args.image_dir
    should_generate_captions = args.generate_captions and not args.skip_caption_generation

    if should_generate_captions:
        print("=== Step 2a: 生成 Caption ===")
        count = generate_captions_for_directory(
            args.image_dir,
            caption_dir,
            use_blip2=args.use_blip2,
            trigger_word=args.trigger_word,
            overwrite=not args.no_overwrite_captions,
        )
        print(f"生成/确认 {count} 个 caption -> {caption_dir}")
    else:
        print(f"=== Step 2a: 使用已有 Caption -> {caption_dir} ===")

    print("\n=== Step 2b: 构建 kohya_ss 训练目录 ===")
    concept_dir = build_kohya_dataset(
        args.image_dir,
        caption_dir,
        args.output_dir,
        concept_name=args.concept_name,
        repeats=args.repeats,
        trigger_word=args.trigger_word,
        strict_captions=not args.allow_missing_captions,
        clean=args.clean,
    )

    report = validate_kohya_dataset(concept_dir, trigger_word=args.trigger_word)
    print("\n=== Dataset validation ===")
    print(f"Concept dir:       {report.concept_dir}")
    print(f"Image count:       {report.image_count}")
    print(f"Caption count:     {report.caption_count}")
    print(f"Missing captions:  {len(report.missing_captions)}")
    print(f"No trigger word:   {len(report.captions_without_trigger)}")

    if not report.valid:
        print("LoRA 训练数据验证失败", file=sys.stderr)
        return 1

    print(f"\nLoRA 训练数据准备完成 -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
