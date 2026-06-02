import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from blast_pile_diffusion.data.sample_bundle import SampleBundle
from blast_pile_diffusion.inference.batch_processor import (
    parse_result_dir_name,
    process_batch,
    result_dir_name,
    scan_generated_outputs,
)


class MockPipeline:
    _execution_device = torch.device("cpu")

    def __init__(self, output_mode: str = "normal") -> None:
        self.output_mode = output_mode
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        image = np.array(kwargs["image"], dtype=np.uint8)
        if self.output_mode == "black":
            generated = np.zeros_like(image)
        else:
            generated = np.roll(image, shift=1, axis=0)
            generated[..., 0] = np.clip(generated[..., 0] + 17, 0, 255)
        return type("Result", (), {"images": [Image.fromarray(generated)]})()


def _config() -> dict:
    return {
        "controlnets": [
            {"name": "depth", "model": "mock-depth", "scale": 0.7},
            {"name": "canny", "model": "mock-canny", "scale": 0.6},
        ],
        "inference": {
            "strength": 0.45,
            "num_inference_steps": 2,
            "guidance_scale": 6.5,
            "seeds_per_sample": 2,
            "output": {"save_comparison": True, "comparison_width": 32},
            "image_validation": {
                "enabled": True,
                "black_threshold": 2,
                "white_threshold": 253,
                "min_std": 1.0,
            },
            "memory": {"empty_cache_interval": 1},
        },
        "negative_prompt": "mock negative",
    }


def _bundle() -> SampleBundle:
    h, w = 32, 40
    x = np.linspace(0, 255, w, dtype=np.uint8)
    y = np.linspace(0, 255, h, dtype=np.uint8)
    xx, yy = np.meshgrid(x, y)
    rgb = np.stack([xx, yy, ((xx.astype(np.uint16) + yy) // 2).astype(np.uint8)], axis=-1)
    depth = np.linspace(1.0, 20.0, h * w, dtype=np.float32).reshape(h, w)
    normal = np.dstack(
        [
            np.zeros((h, w), dtype=np.float32),
            np.zeros((h, w), dtype=np.float32),
            np.ones((h, w), dtype=np.float32),
        ]
    )
    mask = np.zeros((h, w), dtype=np.int32)
    mask[6:24, 8:30] = 1
    depth_cn = np.repeat(np.linspace(255, 0, w, dtype=np.uint8)[None, :, None], h, axis=0)
    depth_cn = np.repeat(depth_cn, 3, axis=2)
    canny = np.zeros((h, w), dtype=np.uint8)
    canny[6:24, 8] = 255
    canny[6:24, 29] = 255
    canny[6, 8:30] = 255
    canny[23, 8:30] = 255
    return SampleBundle(
        scene_id="sceneA",
        cam_id="cam0",
        rgb=rgb,
        depth=depth,
        normal=normal,
        mask=mask,
        depth_cn=depth_cn,
        canny=canny,
    )


def _save_bundle(tmp_path: Path) -> Path:
    bundle_dir = tmp_path / "preprocessed" / "sceneA--cam0"
    _bundle().save(bundle_dir)
    return bundle_dir


def test_result_dir_name_contract() -> None:
    assert result_dir_name("sceneA--cam0", 42) == "sceneA--cam0_s42"
    assert parse_result_dir_name("sceneA--cam0_s42") == ("sceneA--cam0", 42)
    assert parse_result_dir_name("sceneA--cam0--s42") is None


def test_process_batch_saves_outputs_and_resumes(tmp_path: Path) -> None:
    bundle_dir = _save_bundle(tmp_path)
    output_dir = tmp_path / "generated"
    pipe = MockPipeline()

    stats = process_batch(
        pipe,
        [bundle_dir],
        _config(),
        output_dir=output_dir,
        seeds_per_sample=2,
        base_seed=42,
        empty_cache_interval=1,
    )

    assert stats["total"] == 2
    assert stats["generated"] == 2
    assert stats["skipped"] == 0
    assert stats["failed"] == 0
    assert stats["anomalous"] == 0
    assert len(pipe.calls) == 2

    result_dir = output_dir / "sceneA--cam0_s42"
    assert (result_dir / "generated.png").exists()
    assert (result_dir / "comparison.png").exists()
    assert (result_dir / "meta.json").exists()
    with open(result_dir / "meta.json") as f:
        meta = json.load(f)
    assert meta["sample_key"] == "sceneA--cam0"
    assert meta["seed"] == 42
    assert meta["files"]["generated"] == "generated.png"

    second_pipe = MockPipeline()
    resumed = process_batch(
        second_pipe,
        [bundle_dir],
        _config(),
        output_dir=output_dir,
        seeds_per_sample=2,
        base_seed=42,
    )
    assert resumed["generated"] == 0
    assert resumed["skipped"] == 2
    assert len(second_pipe.calls) == 0


def test_process_batch_records_black_image_failure(tmp_path: Path) -> None:
    bundle_dir = _save_bundle(tmp_path)
    output_dir = tmp_path / "generated"

    stats = process_batch(
        MockPipeline(output_mode="black"),
        [bundle_dir],
        _config(),
        output_dir=output_dir,
        seeds_per_sample=1,
        base_seed=7,
        write_report=False,
    )

    assert stats["generated"] == 0
    assert stats["failed"] == 1
    assert "all_black" in stats["failures"][0]["error"]
    assert not (output_dir / "sceneA--cam0_s7" / "generated.png").exists()


def test_scan_generated_outputs_finds_all_white(tmp_path: Path) -> None:
    output_dir = tmp_path / "generated"
    result_dir = output_dir / "sceneA--cam0_s42"
    result_dir.mkdir(parents=True)
    cv2.imwrite(str(result_dir / "generated.png"), np.full((8, 8, 3), 255, dtype=np.uint8))
    (result_dir / "meta.json").write_text("{}", encoding="utf-8")

    anomalies = scan_generated_outputs(output_dir, _config())

    assert len(anomalies) == 1
    assert anomalies[0]["result_dir"] == "sceneA--cam0_s42"
    assert anomalies[0]["reason"] == "all_white"
