#!/usr/bin/env python3
"""Inspect and optionally upscale real LoRA training images."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageOps

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

try:
    LANCZOS = Image.Resampling.LANCZOS
except AttributeError:  # Pillow < 9.1
    LANCZOS = Image.LANCZOS


@dataclass(frozen=True)
class LoraImageInfo:
    path: str
    width: int
    height: int
    image_format: str
    mode: str
    undersized: bool

    @property
    def resolution(self) -> tuple[int, int]:
        return self.width, self.height


@dataclass(frozen=True)
class LoraImageReport:
    image_dir: str
    min_size: int
    min_count: int
    total: int
    min_resolution: tuple[int, int] | None
    max_resolution: tuple[int, int] | None
    average_resolution: tuple[float, float] | None
    format_distribution: dict[str, int]
    undersized: list[str]
    unreadable: dict[str, str]

    @property
    def passed_min_size(self) -> bool:
        return (
            self.total >= self.min_count
            and self.total > 0
            and not self.undersized
            and not self.unreadable
        )

    def to_dict(self) -> dict:
        return asdict(self)


def iter_image_paths(image_dir: Path) -> list[Path]:
    """Return supported image files sorted by filename."""
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    if not image_dir.is_dir():
        raise NotADirectoryError(f"Image path is not a directory: {image_dir}")
    return sorted(
        (p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda p: p.name.lower(),
    )


def inspect_image(path: Path, min_size: int = 1024) -> LoraImageInfo:
    """Read image metadata without modifying the file."""
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        width, height = image.size
        image_format = image.format or path.suffix.lstrip(".").upper()
        mode = image.mode
    return LoraImageInfo(
        path=str(path),
        width=width,
        height=height,
        image_format=image_format.upper(),
        mode=mode,
        undersized=width < min_size or height < min_size,
    )


def scan_lora_images(
    image_dir: Path,
    min_size: int = 1024,
    min_count: int = 0,
) -> LoraImageReport:
    """Scan a LoRA image directory and summarize resolution/format coverage."""
    infos: list[LoraImageInfo] = []
    unreadable: dict[str, str] = {}

    for image_path in iter_image_paths(image_dir):
        try:
            infos.append(inspect_image(image_path, min_size=min_size))
        except Exception as exc:  # pragma: no cover - exact PIL errors vary
            unreadable[str(image_path)] = str(exc)

    if infos:
        min_info = min(infos, key=lambda item: item.width * item.height)
        max_info = max(infos, key=lambda item: item.width * item.height)
        avg_width = sum(item.width for item in infos) / len(infos)
        avg_height = sum(item.height for item in infos) / len(infos)
        min_resolution: tuple[int, int] | None = min_info.resolution
        max_resolution: tuple[int, int] | None = max_info.resolution
        average_resolution: tuple[float, float] | None = (round(avg_width, 1), round(avg_height, 1))
    else:
        min_resolution = None
        max_resolution = None
        average_resolution = None

    formats = Counter(item.image_format for item in infos)
    undersized = [item.path for item in infos if item.undersized]

    return LoraImageReport(
        image_dir=str(image_dir),
        min_size=min_size,
        min_count=min_count,
        total=len(infos),
        min_resolution=min_resolution,
        max_resolution=max_resolution,
        average_resolution=average_resolution,
        format_distribution=dict(sorted(formats.items())),
        undersized=undersized,
        unreadable=unreadable,
    )


def _save_image(image: Image.Image, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = destination.suffix.lower()
    save_kwargs: dict[str, object] = {}
    if suffix in {".jpg", ".jpeg"}:
        save_kwargs = {"quality": 95, "subsampling": 0}
    image.save(destination, **save_kwargs)


def upscale_image(path: Path, min_size: int = 1024, output_dir: Path | None = None) -> Path:
    """Upscale one image with LANCZOS so both dimensions are at least min_size."""
    destination = output_dir / path.name if output_dir is not None else path

    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        width, height = image.size
        scale = max(min_size / width, min_size / height)

        if scale <= 1.0:
            if output_dir is not None and destination != path:
                _save_image(image, destination)
            return destination

        new_size = (
            max(min_size, int(math.ceil(width * scale))),
            max(min_size, int(math.ceil(height * scale))),
        )
        upscaled = image.resize(new_size, LANCZOS)
        _save_image(upscaled, destination)
    return destination


def upscale_undersized_images(
    image_dir: Path,
    min_size: int = 1024,
    output_dir: Path | None = None,
) -> list[Path]:
    """Upscale every image below the required resolution."""
    report = scan_lora_images(image_dir, min_size=min_size)
    upscaled: list[Path] = []
    for image_path in report.undersized:
        upscaled.append(upscale_image(Path(image_path), min_size=min_size, output_dir=output_dir))
    return upscaled


def copy_or_upscale_all_images(
    image_dir: Path,
    min_size: int = 1024,
    output_dir: Path | None = None,
) -> list[Path]:
    """Copy every image to output_dir, upscaling only the undersized ones."""
    if output_dir is None:
        return upscale_undersized_images(image_dir, min_size=min_size, output_dir=None)
    output_paths: list[Path] = []
    for image_path in iter_image_paths(image_dir):
        output_paths.append(upscale_image(image_path, min_size=min_size, output_dir=output_dir))
    return output_paths


def print_report(report: LoraImageReport) -> None:
    print("=== LoRA image check ===")
    print(f"Image directory: {report.image_dir}")
    print(f"Total images:    {report.total}")
    print(f"Min count:       {report.min_count}")
    print(f"Min size:        {report.min_size}x{report.min_size}")
    print(f"Min resolution:  {report.min_resolution}")
    print(f"Max resolution:  {report.max_resolution}")
    print(f"Avg resolution:  {report.average_resolution}")
    print(f"Formats:         {report.format_distribution}")

    if report.total == 0:
        print("\nNo supported image files found.")
    elif report.total < report.min_count:
        print(f"\nImage count below requirement: {report.total} < {report.min_count}")
    elif report.undersized:
        print("\nImages below minimum resolution:")
        for path in report.undersized:
            print(f"  - {path}")
    else:
        print("\nAll readable images meet the minimum resolution.")

    if report.unreadable:
        print("\nUnreadable files:")
        for path, error in report.unreadable.items():
            print(f"  - {path}: {error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check LoRA image resolution and formats.")
    parser.add_argument("--image-dir", type=Path, default=Path("data/lora_real/images"))
    parser.add_argument("--min-size", type=int, default=1024)
    parser.add_argument(
        "--min-count",
        type=int,
        default=0,
        help="Minimum required readable image count. Use 50 for C1 acceptance.",
    )
    parser.add_argument(
        "--upscale",
        action="store_true",
        help="Upscale undersized images in place, or into --output-dir if provided.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for upscaled copies. Default is in-place replacement.",
    )
    parser.add_argument("--json", type=Path, default=None, help="Write the final report as JSON.")
    parser.add_argument(
        "--fail-under-min",
        action="store_true",
        help="Exit with code 1 if any readable image is still below --min-size.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scan_dir = args.output_dir if args.output_dir is not None and args.upscale else args.image_dir

    if args.upscale:
        before = scan_lora_images(args.image_dir, min_size=args.min_size)
        if before.undersized:
            print(f"Upscaling {len(before.undersized)} undersized image(s) with LANCZOS...")
            if args.output_dir is None:
                upscale_undersized_images(args.image_dir, min_size=args.min_size, output_dir=None)
            else:
                copy_or_upscale_all_images(
                    args.image_dir, min_size=args.min_size, output_dir=args.output_dir
                )
        else:
            print("No undersized images found; skipping upscale.")
            if args.output_dir is not None:
                copy_or_upscale_all_images(
                    args.image_dir, min_size=args.min_size, output_dir=args.output_dir
                )

    report = scan_lora_images(scan_dir, min_size=args.min_size, min_count=args.min_count)
    print_report(report)

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    if report.unreadable:
        return 1
    if args.fail_under_min and not report.passed_min_size:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
