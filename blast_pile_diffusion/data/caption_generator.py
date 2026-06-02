"""为真实爆堆图片生成 LoRA 训练用 caption。

支持两种模式：
1. BLIP-2 自动 captioning + 注入触发词
2. 模板化稳定描述（不依赖 BLIP-2 模型）
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from pathlib import Path

TRIGGER_WORD = "<mine_blast_pile>"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

LIGHTING = [
    "sunny noon",
    "overcast morning",
    "late afternoon golden hour",
    "dawn soft light",
    "harsh midday sun",
]
WEATHER = ["clear", "slight haze", "dusty", "after rain wet rocks", "windy dust"]
TIME = ["dawn", "early morning", "midday", "afternoon", "late afternoon"]
COMPOSITION = [
    "wide open-pit mine view",
    "close-up texture study",
    "low camera angle near the haul road",
    "telephoto view of the blasted bench",
    "ground-level view across the muck pile",
]
ROCK_FEATURES = [
    "gray-brown angular fragments",
    "fresh fractured faces and sharp edges",
    "mixed boulder sizes with fine crushed material",
    "dust-coated rock surfaces",
    "irregular fractured blast rock",
]
EXTRA = [
    "drilling residue visible",
    "fresh blast fragments",
    "machinery in background",
    "open-pit bench face behind",
    "tire tracks on haul road",
]


def _stable_indices(seed_source: str, count: int) -> list[int]:
    digest = hashlib.sha256(seed_source.encode("utf-8")).digest()
    return [digest[i] for i in range(count)]


def _iter_images(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    if not image_dir.is_dir():
        raise NotADirectoryError(f"Image path is not a directory: {image_dir}")
    return sorted(
        (p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda p: p.name.lower(),
    )


def ensure_trigger_word(caption: str, trigger_word: str = TRIGGER_WORD) -> str:
    """Return a normalized caption with a single leading trigger word."""
    parts = [part.strip() for part in caption.split(",") if part.strip()]
    parts = [part for part in parts if part != trigger_word]
    return ", ".join([trigger_word, *parts])


def template_caption(seed_source: str, trigger_word: str = TRIGGER_WORD) -> str:
    """生成一条基于文件名稳定变化的爆堆描述 caption。"""
    i0, i1, i2, i3, i4, i5 = _stable_indices(seed_source, 6)
    caption = (
        f"{trigger_word}, open-pit mine muck pile after blasting, "
        f"{COMPOSITION[i0 % len(COMPOSITION)]}, "
        f"{ROCK_FEATURES[i1 % len(ROCK_FEATURES)]}, "
        f"{LIGHTING[i2 % len(LIGHTING)]}, {WEATHER[i3 % len(WEATHER)]}, "
        f"{TIME[i4 % len(TIME)]}, {EXTRA[i5 % len(EXTRA)]}, "
        "photorealistic, detailed surface texture, sharp edges"
    )
    return ensure_trigger_word(caption, trigger_word=trigger_word)


def random_caption(seed: int | str | None = None) -> str:
    """Backward-compatible wrapper for deterministic template captions."""
    seed_source = "mine_blast_pile" if seed is None else str(seed)
    return template_caption(seed_source)


def _clean_blip_caption(caption: str) -> str:
    caption = re.sub(r"\s+", " ", caption.strip())
    return caption.rstrip(" .")


def blip2_caption(image_path: Path, trigger_word: str = TRIGGER_WORD, device: str = "cuda") -> str:
    """用 BLIP-2 生成 caption 并注入触发词。需要安装 salesforce-lavis。"""
    from PIL import Image

    try:
        from lavis.models import load_model_and_preprocess
    except ImportError:
        raise ImportError("请安装 salesforce-lavis: pip install salesforce-lavis")

    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_opt", model_type="caption_coco_opt2.7b", is_eval=True, device=device
    )
    image = Image.open(image_path).convert("RGB")
    image_tensor = vis_processors["eval"](image).unsqueeze(0).to(device)
    caption = _clean_blip_caption(model.generate({"image": image_tensor})[0])

    return ensure_trigger_word(caption, trigger_word=trigger_word)


def generate_captions_for_directory(
    image_dir: Path,
    output_dir: Path,
    use_blip2: bool = False,
    trigger_word: str = TRIGGER_WORD,
    overwrite: bool = True,
    caption_fn: Callable[[Path], str] | None = None,
) -> int:
    """为目录下所有图片生成 caption .txt 文件。返回处理数量。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for img_path in _iter_images(image_dir):
        caption_path = output_dir / f"{img_path.stem}.txt"
        if caption_path.exists() and not overwrite:
            count += 1
            continue

        if caption_fn is not None:
            caption = caption_fn(img_path)
        elif use_blip2:
            caption = blip2_caption(img_path, trigger_word=trigger_word)
        else:
            caption = template_caption(img_path.name, trigger_word=trigger_word)

        caption_path.write_text(
            ensure_trigger_word(caption, trigger_word=trigger_word),
            encoding="utf-8",
        )
        count += 1

    return count
