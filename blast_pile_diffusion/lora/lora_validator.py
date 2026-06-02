"""LoRA 质量快速验证：用固定 prompt 生成几张图，目视检查风格是否正确。"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageDraw

VALIDATION_PROMPTS = [
    "<mine_blast_pile>, open-pit mine muck pile, sunny noon, fresh angular rocks",
    "<mine_blast_pile>, blasted rock fragments, overcast, dusty atmosphere",
    "<mine_blast_pile>, mine bench face, golden hour, sharp shadows on rocks",
    "<mine_blast_pile>, muck pile close-up, wet rocks after rain, detailed texture",
    "<mine_blast_pile>, wide quarry bench, gray-brown angular blast pile, slight haze",
    "<mine_blast_pile>, low angle haul road view, dusty fractured boulders",
    "<mine_blast_pile>, telephoto mine face, irregular rock fragments, harsh sun",
    "<mine_blast_pile>, wet muck pile after rain, realistic mineral texture",
]

DEFAULT_NEGATIVE_PROMPT = "cartoon, smooth, cobblestone, beach, text, watermark, people"


def build_validation_prompts(num_samples: int) -> list[str]:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    prompts: list[str] = []
    while len(prompts) < num_samples:
        prompts.extend(VALIDATION_PROMPTS)
    return prompts[:num_samples]


def _write_manifest(
    output_dir: Path,
    lora_path: Path,
    prompts: Sequence[str],
    paths: Sequence[Path],
    mode: str,
) -> None:
    manifest = {
        "mode": mode,
        "lora_path": str(lora_path),
        "num_samples": len(prompts),
        "samples": [
            {"prompt": prompt, "output_path": str(path)} for prompt, path in zip(prompts, paths)
        ],
    }
    (output_dir / "validation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _mock_validation_image(prompt: str, index: int, size: int) -> Image.Image:
    width = height = size
    base = Image.new("RGB", (width, height), (116 + index * 7 % 40, 105, 86))
    draw = ImageDraw.Draw(base)

    for row in range(0, height, max(16, size // 24)):
        shade = 70 + (row * 47 + index * 19) % 110
        draw.polygon(
            [
                (0, min(height, row + size // 10)),
                (width // 3, row),
                (width, min(height, row + size // 8)),
                (width, min(height, row + size // 5)),
                (width // 4, min(height, row + size // 7)),
            ],
            fill=(shade, max(60, shade - 18), max(45, shade - 34)),
        )

    label = f"MOCK LoRA validation {index + 1}\n{prompt[:84]}"
    draw.rectangle((16, 16, width - 16, 76), fill=(245, 241, 230))
    draw.text((26, 24), label, fill=(38, 35, 31))
    return base


def _load_pipeline(base_model: str, torch_dtype: Any, device: str) -> Any:
    from diffusers import StableDiffusionXLPipeline

    pipe = StableDiffusionXLPipeline.from_pretrained(base_model, torch_dtype=torch_dtype)
    return pipe.to(device)


def _load_lora_weights(pipe: Any, lora_path: Path) -> None:
    try:
        pipe.load_lora_weights(str(lora_path), adapter_name="mine_style")
        if hasattr(pipe, "set_adapters"):
            pipe.set_adapters(["mine_style"], adapter_weights=[1.0])
    except TypeError:
        pipe.load_lora_weights(str(lora_path))


def validate_lora(
    lora_path: Path,
    output_dir: Path,
    num_samples: int = 4,
    base_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.0,
    seed: int = 42,
    device: str | None = None,
    mock: bool = False,
    dry_run: bool = False,
    image_size: int = 1024,
    pipeline_factory: Callable[[str, Any, str], Any] | None = None,
) -> list[Path]:
    """用训好的 LoRA 生成验证图片。

    mock=True 会创建轻量占位图，适合测试接口，不会加载 SDXL。
    dry_run=True 只写 validation_manifest.json，不生成图片。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts = build_validation_prompts(num_samples)
    paths = [output_dir / f"lora_val_{i:02d}.png" for i in range(num_samples)]

    if dry_run:
        _write_manifest(output_dir, lora_path, prompts, paths, mode="dry-run")
        print(f"[dry-run] planned {num_samples} LoRA validation sample(s) -> {output_dir}")
        return paths

    if mock:
        for i, (prompt, out_path) in enumerate(zip(prompts, paths)):
            image = _mock_validation_image(prompt, i, image_size)
            image.save(out_path)
            print(f"  [{i + 1}/{num_samples}] mock saved: {out_path}")
        _write_manifest(output_dir, lora_path, prompts, paths, mode="mock")
        return paths

    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA weight does not exist: {lora_path}")

    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    factory = pipeline_factory or _load_pipeline
    pipe = factory(base_model, torch_dtype, device)
    _load_lora_weights(pipe, lora_path)

    try:
        for i, (prompt, out_path) in enumerate(zip(prompts, paths)):
            generator = torch.Generator(device).manual_seed(seed + i)
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images[0]

            result.save(str(out_path))
            print(f"  [{i + 1}/{num_samples}] saved: {out_path}")
    finally:
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _write_manifest(output_dir, lora_path, prompts, paths, mode="real")
    return paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LoRA validation samples.")
    parser.add_argument(
        "--lora-path",
        type=Path,
        default=Path("lora_weights/mine_blast_pile.safetensors"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/generated/lora_validation"))
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--base-model", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Create lightweight placeholder images.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only write a validation manifest.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_lora(
        lora_path=args.lora_path,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        base_model=args.base_model,
        device=args.device,
        seed=args.seed,
        mock=args.mock,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
