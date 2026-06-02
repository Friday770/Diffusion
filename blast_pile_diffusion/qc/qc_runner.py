"""批量 QC 运行器：扫描 generated/ 目录，逐样本执行 QC，输出统计报告。"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from blast_pile_diffusion.qc.edge_alignment import QCResult, check_edge_alignment
from blast_pile_diffusion.utils.image_io import read_mask, read_rgb
from blast_pile_diffusion.utils.vis import save_qc_debug_image
from blast_pile_diffusion.preprocessing.canny_from_mask import mask_to_canny


DEFAULT_QC_CONFIG: dict[str, Any] = {
    "edge_alignment": {
        "max_mean_offset_px": 4.0,
        "max_p99_offset_px": 12.0,
        "canny_low": 50,
        "canny_high": 150,
        "mask_dilate_kernel": 0,
        "min_gen_edge_pixels": 1,
        "min_mask_edge_pixels": 1,
    },
    "diversity": {
        "max_ssim_similarity": 0.85,
    },
    "reporting": {
        "low_scene_pass_rate_pct": 20.0,
    },
}


def _deep_merge(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(defaults)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _validate_edge_config(edge_cfg: dict[str, Any]) -> None:
    required = ("max_mean_offset_px", "max_p99_offset_px", "canny_low", "canny_high")
    missing = [key for key in required if key not in edge_cfg]
    if missing:
        raise ValueError(f"QC edge_alignment config missing keys: {missing}")

    for key in ("max_mean_offset_px", "max_p99_offset_px"):
        if float(edge_cfg[key]) <= 0:
            raise ValueError(f"edge_alignment.{key} must be > 0")

    canny_low = int(edge_cfg["canny_low"])
    canny_high = int(edge_cfg["canny_high"])
    if canny_low < 0 or canny_high <= canny_low:
        raise ValueError("edge_alignment.canny_high must be greater than canny_low >= 0")

    for key in ("mask_dilate_kernel", "min_gen_edge_pixels", "min_mask_edge_pixels"):
        if int(edge_cfg.get(key, 0)) < 0:
            raise ValueError(f"edge_alignment.{key} must be >= 0")


def load_qc_config(config_path: Path) -> dict:
    """Load QC thresholds, apply defaults, and validate the expected schema."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"QC config not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}

    cfg = _deep_merge(DEFAULT_QC_CONFIG, loaded)
    loaded_edge_cfg = loaded.get("edge_alignment", {})
    if "dilate_kernel" in loaded_edge_cfg and "mask_dilate_kernel" not in loaded_edge_cfg:
        cfg["edge_alignment"]["mask_dilate_kernel"] = loaded_edge_cfg["dilate_kernel"]
    _validate_edge_config(cfg["edge_alignment"])
    return cfg


def _round_or_none(value: float, digits: int = 2) -> float | None:
    if np.isfinite(value):
        return round(float(value), digits)
    return None


def _read_generated_sample_metadata(sample_dir: Path) -> dict[str, Any]:
    meta_path = sample_dir / "meta.json"
    meta: dict[str, Any] = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    sample_key = meta.get("sample_key")
    seed = meta.get("seed")
    if not sample_key:
        sample_key, parsed_seed = _parse_sample_dir_name(sample_dir.name)
        if seed is None:
            seed = parsed_seed

    scene_id = meta.get("scene_id") or str(sample_key).split("--", 1)[0]
    return {
        "sample_key": sample_key,
        "scene_id": scene_id,
        "seed": seed,
        "sample_dir_name": sample_dir.name,
    }


def _parse_sample_dir_name(name: str) -> tuple[str, int | str | None]:
    for separator in ("_s", "--s"):
        if separator not in name:
            continue
        sample_key, seed_text = name.rsplit(separator, 1)
        if not sample_key or not seed_text:
            continue
        try:
            seed: int | str | None = int(seed_text)
        except ValueError:
            seed = seed_text
        return sample_key, seed
    return name, None


def _new_stats() -> dict[str, Any]:
    return {"total": 0, "passed": 0, "failed": 0, "errors": 0}


def _finalize_stats(stats: dict[str, Any], offsets: list[float]) -> dict[str, Any]:
    total = int(stats.get("total", 0))
    passed = int(stats.get("passed", 0))
    stats["pass_rate_pct"] = round(passed / max(total, 1) * 100, 1)
    stats["pass_rate"] = stats["pass_rate_pct"]
    if offsets:
        arr = np.asarray(offsets, dtype=np.float64)
        stats["mean_offset_all_px"] = round(float(np.mean(arr)), 2)
        stats["median_offset_all_px"] = round(float(np.median(arr)), 2)
        stats["p95_offset_all_px"] = round(float(np.percentile(arr, 95)), 2)
        stats["p99_offset_all_px"] = round(float(np.percentile(arr, 99)), 2)

        # Backward-compatible aliases used by the original script output.
        stats["mean_offset_all"] = stats["mean_offset_all_px"]
        stats["median_offset_all"] = stats["median_offset_all_px"]
    return stats


def _sample_qc_record(
    metadata: dict[str, Any],
    result: QCResult,
    mask_path: Path,
    generated_path: Path,
) -> dict[str, Any]:
    record = {
        "sample_key": metadata["sample_key"],
        "scene_id": metadata["scene_id"],
        "seed": metadata["seed"],
        "generated_path": str(generated_path),
        "mask_path": str(mask_path),
        "status": "passed" if result.passed else "failed",
    }
    record.update(result.to_dict())
    return record


def _error_qc_record(
    metadata: dict[str, Any],
    generated_path: Path,
    mask_path: Path,
    reason: str,
    message: str,
) -> dict[str, Any]:
    return {
        "sample_key": metadata["sample_key"],
        "scene_id": metadata["scene_id"],
        "seed": metadata["seed"],
        "generated_path": str(generated_path),
        "mask_path": str(mask_path),
        "status": "error",
        "passed": False,
        "mean_offset_px": None,
        "median_offset_px": None,
        "p95_offset_px": None,
        "p99_offset_px": None,
        "max_offset_px": None,
        "gen_edge_count": 0,
        "mask_edge_count": 0,
        "edge_distance_direction": "generated_edge_to_mask_edge",
        "failure_reasons": [reason],
        "error_message": message,
    }


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, allow_nan=False)


def run_qc_batch(
    generated_dir: Path,
    preprocessed_dir: Path,
    qc_config_path: Path,
    save_debug_images: bool = True,
    report_path: Path | None = None,
) -> dict:
    """
    对 generated/ 下所有样本执行 QC。

    Args:
        generated_dir: data/generated/ 路径
        preprocessed_dir: data/preprocessed/ 路径（用于读取掩码）
        qc_config_path: configs/qc/thresholds.yaml 路径
        save_debug_images: 是否保存 debug 对比图
        report_path: qc_report.json 输出路径；None 时写到 generated_dir/qc_report.json

    Returns:
        统计信息 dict
    """
    generated_dir = Path(generated_dir)
    preprocessed_dir = Path(preprocessed_dir)
    report_path = generated_dir / "qc_report.json" if report_path is None else Path(report_path)

    cfg = load_qc_config(qc_config_path)
    edge_cfg = cfg["edge_alignment"]
    low_scene_pass_rate = float(cfg.get("reporting", {}).get("low_scene_pass_rate_pct", 20.0))

    stats = _new_stats()
    stats["debug_images_written"] = 0
    stats["debug_image_errors"] = 0
    offset_list = []
    per_scene = defaultdict(_new_stats)
    per_scene_offsets: dict[str, list[float]] = defaultdict(list)
    failed_samples = []

    sample_dirs = []
    if generated_dir.exists():
        sample_dirs = sorted(
            d for d in generated_dir.iterdir() if d.is_dir() and (d / "generated.png").exists()
        )

    for sample_dir in sample_dirs:
        gen_path = sample_dir / "generated.png"
        metadata = _read_generated_sample_metadata(sample_dir)
        sample_key = metadata["sample_key"]
        scene_id = metadata["scene_id"]
        mask_path = preprocessed_dir / str(sample_key) / "mask_instance.png"

        stats["total"] += 1
        per_scene[scene_id]["total"] += 1

        if sample_key is None or not mask_path.exists():
            reason = "missing_mask" if sample_key is not None else "missing_sample_key"
            message = f"Mask not found: {mask_path}" if sample_key is not None else "No sample_key"
            qc_record = _error_qc_record(metadata, gen_path, mask_path, reason, message)
            _write_json(sample_dir / "qc.json", qc_record)
            stats["failed"] += 1
            stats["errors"] += 1
            per_scene[scene_id]["failed"] += 1
            per_scene[scene_id]["errors"] += 1
            failed_samples.append(
                {
                    "sample_dir": sample_dir.name,
                    "sample_key": sample_key,
                    "scene_id": scene_id,
                    "seed": metadata["seed"],
                    "status": "error",
                    "failure_reasons": [reason],
                }
            )
            continue

        try:
            gen_rgb = read_rgb(gen_path)
            mask = read_mask(mask_path)

            result: QCResult = check_edge_alignment(
                gen_rgb,
                mask,
                max_mean_offset=edge_cfg["max_mean_offset_px"],
                max_p99_offset=edge_cfg["max_p99_offset_px"],
                canny_low=edge_cfg.get("canny_low", 50),
                canny_high=edge_cfg.get("canny_high", 150),
                mask_dilate_kernel=edge_cfg.get("mask_dilate_kernel", edge_cfg.get("dilate_kernel", 0)),
                min_gen_edge_pixels=edge_cfg.get("min_gen_edge_pixels", 1),
                min_mask_edge_pixels=edge_cfg.get("min_mask_edge_pixels", 1),
            )

            qc_record = _sample_qc_record(metadata, result, mask_path, gen_path)
            _write_json(sample_dir / "qc.json", qc_record)

            if result.passed:
                stats["passed"] += 1
                per_scene[scene_id]["passed"] += 1
            else:
                stats["failed"] += 1
                per_scene[scene_id]["failed"] += 1
                failed_samples.append(
                    {
                        "sample_dir": sample_dir.name,
                        "sample_key": sample_key,
                        "scene_id": scene_id,
                        "seed": metadata["seed"],
                        "status": "failed",
                        "mean_offset_px": _round_or_none(result.mean_offset),
                        "p99_offset_px": _round_or_none(result.p99_offset),
                        "failure_reasons": list(result.failure_reasons),
                    }
                )

            if np.isfinite(result.mean_offset):
                offset_list.append(result.mean_offset)
                per_scene_offsets[scene_id].append(result.mean_offset)

            if save_debug_images and not result.passed:
                try:
                    mask_canny = mask_to_canny(
                        mask,
                        canny_low=edge_cfg.get("canny_low", 50),
                        canny_high=edge_cfg.get("canny_high", 150),
                        dilate_kernel=edge_cfg.get(
                            "mask_dilate_kernel",
                            edge_cfg.get("dilate_kernel", 0),
                        ),
                    )
                    gen_gray = cv2.cvtColor(gen_rgb, cv2.COLOR_RGB2GRAY)
                    gen_canny = cv2.Canny(
                        gen_gray,
                        edge_cfg.get("canny_low", 50),
                        edge_cfg.get("canny_high", 150),
                    )
                    unity_rgb_path = preprocessed_dir / str(sample_key) / "rgb.png"
                    unity_rgb = read_rgb(unity_rgb_path) if unity_rgb_path.exists() else gen_rgb
                    save_qc_debug_image(
                        unity_rgb,
                        gen_rgb,
                        mask_canny,
                        gen_canny,
                        sample_dir / "debug_overlay.png",
                    )
                    stats["debug_images_written"] += 1
                except Exception as debug_error:
                    stats["debug_image_errors"] += 1
                    print(f"[QC DEBUG ERROR] {sample_dir.name}: {debug_error}")

        except Exception as e:
            print(f"[QC ERROR] {sample_dir.name}: {e}")
            qc_record = _error_qc_record(
                metadata,
                gen_path,
                mask_path,
                reason="qc_exception",
                message=str(e),
            )
            _write_json(sample_dir / "qc.json", qc_record)
            stats["failed"] += 1
            stats["errors"] += 1
            per_scene[scene_id]["failed"] += 1
            per_scene[scene_id]["errors"] += 1
            failed_samples.append(
                {
                    "sample_dir": sample_dir.name,
                    "sample_key": sample_key,
                    "scene_id": scene_id,
                    "seed": metadata["seed"],
                    "status": "error",
                    "failure_reasons": ["qc_exception"],
                    "error_message": str(e),
                }
            )

    summary = _finalize_stats(stats, offset_list)

    finalized_per_scene = {}
    low_pass_rate_scenes = []
    for scene_id, scene_stats in sorted(per_scene.items()):
        scene_summary = _finalize_stats(dict(scene_stats), per_scene_offsets[scene_id])
        finalized_per_scene[scene_id] = scene_summary
        if scene_summary["total"] > 0 and scene_summary["pass_rate_pct"] < low_scene_pass_rate:
            low_pass_rate_scenes.append(scene_id)

    report = {
        "summary": summary,
        "per_scene": finalized_per_scene,
        "failed_samples": failed_samples,
        "low_pass_rate_scenes": low_pass_rate_scenes,
        "config": {
            "qc_config_path": str(qc_config_path),
            "edge_alignment": edge_cfg,
            "reporting": cfg.get("reporting", {}),
        },
    }
    _write_json(report_path, report)
    return report
