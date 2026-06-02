"""LoRA caption generation tests."""

from pathlib import Path

from PIL import Image

from blast_pile_diffusion.data.caption_generator import (
    TRIGGER_WORD,
    ensure_trigger_word,
    generate_captions_for_directory,
)


def _write_image(path: Path, size: tuple[int, int] = (64, 64)) -> None:
    Image.new("RGB", size, (120, 100, 80)).save(path)


def test_ensure_trigger_word_normalizes_to_single_leading_token():
    caption = ensure_trigger_word(f"sunny mine pile, {TRIGGER_WORD}, dusty rocks")
    assert caption.startswith(TRIGGER_WORD)
    assert caption.count(TRIGGER_WORD) == 1


def test_template_caption_generation_has_trigger_and_variation(tmp_path):
    image_dir = tmp_path / "images"
    caption_dir = tmp_path / "captions"
    image_dir.mkdir()

    for name in ["blast_a.jpg", "blast_b.png", "blast_c.webp"]:
        _write_image(image_dir / name)
    (image_dir / "notes.txt").write_text("ignore me", encoding="utf-8")

    count = generate_captions_for_directory(image_dir, caption_dir)

    captions = sorted(path.read_text(encoding="utf-8") for path in caption_dir.glob("*.txt"))
    assert count == 3
    assert len(captions) == 3
    assert all(TRIGGER_WORD in caption for caption in captions)
    assert len(set(captions)) > 1
