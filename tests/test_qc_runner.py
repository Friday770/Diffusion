"""测试批量 QC runner 的落盘报告与失败样本 debug 输出。"""

import json
from pathlib import Path

import cv2
import numpy as np

from blast_pile_diffusion.qc.qc_runner import load_qc_config, run_qc_batch


def _write_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def _write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask.astype(np.uint16))


def _make_mask() -> np.ndarray:
    mask = np.zeros((128, 128), dtype=np.int32)
    mask[30:90, 25:100] = 1
    return mask


def _make_generated_rgb(shift: int = 0) -> np.ndarray:
    rgb = np.full((128, 128, 3), 180, dtype=np.uint8)
    rgb[30 + shift : 90 + shift, 25 + shift : 100 + shift] = [90, 70, 50]
    return rgb


def _write_generated_sample(
    generated_dir: Path,
    sample_key: str,
    seed: int,
    rgb: np.ndarray,
) -> Path:
    sample_dir = generated_dir / f"{sample_key}--s{seed}"
    _write_rgb(sample_dir / "generated.png", rgb)
    with open(sample_dir / "meta.json", "w") as f:
        json.dump({"sample_key": sample_key, "seed": seed}, f)
    return sample_dir


def _write_qc_config(path: Path) -> None:
    path.write_text(
        """
edge_alignment:
  max_mean_offset_px: 4.0
  max_p99_offset_px: 12.0
  canny_low: 50
  canny_high: 150
  mask_dilate_kernel: 0
reporting:
  low_scene_pass_rate_pct: 20.0
""".strip()
    )


def test_load_qc_config_applies_defaults(tmp_path):
    cfg_path = tmp_path / "thresholds.yaml"
    cfg_path.write_text(
        """
edge_alignment:
  max_mean_offset_px: 5
  max_p99_offset_px: 15
  canny_low: 30
  canny_high: 120
""".strip()
    )

    cfg = load_qc_config(cfg_path)
    assert cfg["edge_alignment"]["mask_dilate_kernel"] == 0
    assert cfg["edge_alignment"]["min_gen_edge_pixels"] == 1
    assert cfg["reporting"]["low_scene_pass_rate_pct"] == 20.0


def test_run_qc_batch_writes_sample_and_summary_reports(tmp_path):
    generated_dir = tmp_path / "generated"
    preprocessed_dir = tmp_path / "preprocessed"
    cfg_path = tmp_path / "thresholds.yaml"
    _write_qc_config(cfg_path)

    sample_key = "sceneA--cam0"
    mask = _make_mask()
    _write_mask(preprocessed_dir / sample_key / "mask_instance.png", mask)
    _write_rgb(preprocessed_dir / sample_key / "rgb.png", _make_generated_rgb())

    passed_dir = _write_generated_sample(
        generated_dir,
        sample_key,
        seed=42,
        rgb=_make_generated_rgb(),
    )
    failed_dir = _write_generated_sample(
        generated_dir,
        sample_key,
        seed=43,
        rgb=_make_generated_rgb(shift=20),
    )
    missing_mask_dir = _write_generated_sample(
        generated_dir,
        "sceneB--cam0",
        seed=42,
        rgb=_make_generated_rgb(),
    )

    report = run_qc_batch(generated_dir, preprocessed_dir, cfg_path, save_debug_images=True)

    assert (generated_dir / "qc_report.json").exists()
    assert (passed_dir / "qc.json").exists()
    assert (failed_dir / "qc.json").exists()
    assert (missing_mask_dir / "qc.json").exists()

    with open(passed_dir / "qc.json") as f:
        passed_qc = json.load(f)
    with open(failed_dir / "qc.json") as f:
        failed_qc = json.load(f)
    with open(missing_mask_dir / "qc.json") as f:
        missing_qc = json.load(f)

    assert passed_qc["passed"] is True
    assert failed_qc["passed"] is False
    assert missing_qc["status"] == "error"
    assert missing_qc["failure_reasons"] == ["missing_mask"]

    assert not (passed_dir / "debug_overlay.png").exists()
    assert (failed_dir / "debug_overlay.png").exists()

    assert report["summary"]["total"] == 3
    assert report["summary"]["passed"] == 1
    assert report["summary"]["failed"] == 2
    assert report["summary"]["errors"] == 1
    assert report["summary"]["debug_images_written"] == 1

    assert report["per_scene"]["sceneA"]["total"] == 2
    assert report["per_scene"]["sceneA"]["passed"] == 1
    assert report["per_scene"]["sceneA"]["failed"] == 1
    assert report["per_scene"]["sceneA"]["pass_rate_pct"] == 50.0
    assert report["per_scene"]["sceneB"]["errors"] == 1
    assert "sceneB" in report["low_pass_rate_scenes"]
