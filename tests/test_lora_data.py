"""Tests for LoRA image checks, kohya data prep, dry-run training, and validation."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

from blast_pile_diffusion.data.caption_generator import TRIGGER_WORD
from blast_pile_diffusion.lora.lora_validator import validate_lora
from blast_pile_diffusion.lora.prepare_dataset import build_kohya_dataset, validate_kohya_dataset
from blast_pile_diffusion.lora.train_launcher import build_train_command, launch_training


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_check_lora_images_module():
    module_path = PROJECT_ROOT / "scripts" / "check_lora_images.py"
    spec = importlib.util.spec_from_file_location("check_lora_images", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_image(
    path: Path,
    size: tuple[int, int],
    color: tuple[int, int, int] = (110, 95, 75),
) -> None:
    Image.new("RGB", size, color).save(path)


def test_check_lora_images_reports_and_upscales(tmp_path):
    checker = _load_check_lora_images_module()
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    small = image_dir / "small.jpg"
    large = image_dir / "large.png"
    _write_image(small, (512, 768))
    _write_image(large, (1200, 1024))

    before = checker.scan_lora_images(image_dir, min_size=1024)
    assert before.total == 2
    assert before.undersized == [str(small)]

    checker.upscale_undersized_images(image_dir, min_size=1024)

    after = checker.scan_lora_images(image_dir, min_size=1024)
    assert after.undersized == []
    with Image.open(small) as image:
        assert image.width >= 1024
        assert image.height >= 1024


def test_check_lora_images_min_count_marks_empty_dir_incomplete(tmp_path):
    checker = _load_check_lora_images_module()
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    report = checker.scan_lora_images(image_dir, min_size=1024, min_count=50)

    assert report.total == 0
    assert report.passed_min_size is False


def test_build_kohya_dataset_pairs_images_and_normalizes_captions(tmp_path):
    image_dir = tmp_path / "images"
    caption_dir = tmp_path / "captions"
    output_dir = tmp_path / "kohya"
    image_dir.mkdir()
    caption_dir.mkdir()

    _write_image(image_dir / "a.jpg", (1024, 1024))
    _write_image(image_dir / "b.png", (1024, 1200))
    (caption_dir / "a.txt").write_text("dusty mine pile, angular fragments", encoding="utf-8")
    (caption_dir / "b.txt").write_text(
        f"{TRIGGER_WORD}, wet blasted rocks, overcast",
        encoding="utf-8",
    )

    concept_dir = build_kohya_dataset(
        image_dir,
        caption_dir,
        output_dir,
        repeats=10,
        clean=True,
    )
    report = validate_kohya_dataset(concept_dir)

    assert concept_dir.name == "10_mine_blast_pile"
    assert report.valid
    assert (concept_dir / "a.txt").read_text(encoding="utf-8").startswith(TRIGGER_WORD)


def test_build_kohya_dataset_strict_missing_caption_fails(tmp_path):
    image_dir = tmp_path / "images"
    caption_dir = tmp_path / "captions"
    output_dir = tmp_path / "kohya"
    image_dir.mkdir()
    caption_dir.mkdir()
    _write_image(image_dir / "a.jpg", (1024, 1024))

    with pytest.raises(FileNotFoundError):
        build_kohya_dataset(image_dir, caption_dir, output_dir)


def test_train_command_is_dry_run_compatible_and_uses_txt_captions(tmp_path):
    config_path = PROJECT_ROOT / "configs" / "lora" / "sdxl_rank32.yaml"
    base_config_path = PROJECT_ROOT / "configs" / "base.yaml"
    train_data_dir = tmp_path / "missing_kohya_dataset"
    output_dir = tmp_path / "lora_weights"

    cmd = build_train_command(config_path, base_config_path, train_data_dir, output_dir)
    assert "sdxl_train_network.py" in cmd
    assert "--caption_extension" in cmd
    assert ".txt" in cmd

    exit_code = launch_training(
        config_path,
        base_config_path,
        train_data_dir,
        output_dir,
        dry_run=True,
        sd_scripts_dir=tmp_path / "missing_sd_scripts",
    )
    assert exit_code == 0


def test_5090_train_command_uses_high_vram_profile(tmp_path):
    config_path = PROJECT_ROOT / "configs" / "lora" / "sdxl_5090_32gb.yaml"
    base_config_path = PROJECT_ROOT / "configs" / "base.yaml"
    train_data_dir = tmp_path / "kohya_dataset"
    output_dir = tmp_path / "lora_weights"

    cmd = build_train_command(config_path, base_config_path, train_data_dir, output_dir)

    assert "--train_batch_size" in cmd
    assert cmd[cmd.index("--train_batch_size") + 1] == "2"
    assert "--gradient_accumulation_steps" in cmd
    assert cmd[cmd.index("--gradient_accumulation_steps") + 1] == "2"
    assert "--sdpa" in cmd
    assert "--gradient_checkpointing" in cmd
    assert "--mixed_precision" in cmd
    assert "bf16" in cmd


def test_lora_validator_mock_creates_images_and_manifest(tmp_path):
    output_dir = tmp_path / "validation"
    paths = validate_lora(
        lora_path=tmp_path / "missing.safetensors",
        output_dir=output_dir,
        num_samples=2,
        mock=True,
        image_size=128,
    )

    assert len(paths) == 2
    assert all(path.exists() for path in paths)
    with Image.open(paths[0]) as image:
        assert image.size == (128, 128)

    manifest = json.loads((output_dir / "validation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["mode"] == "mock"
    assert manifest["num_samples"] == 2
