"""批量 ControlNet 推理，带断点续传和进度追踪。"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any

import cv2
import torch
from tqdm import tqdm

from blast_pile_diffusion.data.sample_bundle import SampleBundle
from blast_pile_diffusion.inference.controlnet_runner import (
    detect_image_anomaly,
    run_single,
    save_single_result,
)
from blast_pile_diffusion.inference.prompt_bank import random_prompt


def result_dir_name(sample_key: str, seed: int) -> str:
    """D1/D3 canonical result directory name: {sample_key}_s{seed}."""
    return f"{sample_key}_s{seed}"


def parse_result_dir_name(name: str) -> tuple[str, int] | None:
    """Parse canonical result directory names."""
    if "_s" not in name:
        return None
    sample_key, seed_text = name.rsplit("_s", 1)
    if not sample_key or not seed_text.isdigit():
        return None
    return sample_key, int(seed_text)


def _legacy_result_dir_name(sample_key: str, seed: int) -> str:
    return f"{sample_key}--s{seed}"


def _processed_result_dir(output_dir: Path, sample_key: str, seed: int) -> Path | None:
    """Return an existing completed result dir, including old --s names for resume."""
    for name in (result_dir_name(sample_key, seed), _legacy_result_dir_name(sample_key, seed)):
        result_dir = output_dir / name
        if (result_dir / "generated.png").exists() and (result_dir / "meta.json").exists():
            return result_dir
    return None


def _is_already_processed(output_dir: Path, sample_key: str, seed: int) -> bool:
    """检查某个 (sample, seed) 组合是否已经处理过。"""
    return _processed_result_dir(output_dir, sample_key, seed) is not None


def process_batch(
    pipe: Any,
    bundle_dirs: list[Path],
    config: dict,
    output_dir: Path,
    seeds_per_sample: int | None = None,
    base_seed: int = 42,
    empty_cache_interval: int | None = None,
    write_report: bool = True,
) -> dict:
    """
    批量执行 ControlNet 推理。

    Args:
        pipe: build_pipeline() 返回的 pipeline
        bundle_dirs: 预处理完毕的 SampleBundle 目录路径列表
        config: 推理配置 dict
        output_dir: 输出目录
        seeds_per_sample: 每个样本生成几个不同 seed 的版本
        base_seed: 起始种子
        empty_cache_interval: 每处理多少个任务清一次 CUDA cache；默认读配置。
        write_report: 是否写 data/generated/batch_report.json

    Returns:
        统计信息 dict
    """
    if seeds_per_sample is None:
        seeds_per_sample = config.get("inference", {}).get("seeds_per_sample", 4)
    if empty_cache_interval is None:
        empty_cache_interval = (
            config.get("inference", {}).get("memory", {}).get("empty_cache_interval", 50)
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    total_tasks = len(bundle_dirs) * seeds_per_sample
    stats: dict[str, Any] = {
        "total": total_tasks,
        "attempted": 0,
        "generated": 0,
        "skipped": 0,
        "failed": 0,
        "anomalous": 0,
        "failures": [],
        "anomalies": [],
        "elapsed_seconds": None,
    }

    batch_start = time.perf_counter()
    pbar = tqdm(total=total_tasks, desc="ControlNet 推理")

    for bundle_dir in bundle_dirs:
        try:
            bundle = SampleBundle.load(bundle_dir)
        except Exception as e:
            error = f"加载 SampleBundle 失败: {e}"
            print(f"\n[ERROR] {bundle_dir}: {error}")
            for i in range(seeds_per_sample):
                stats["failed"] += 1
                stats["failures"].append(
                    {
                        "bundle_dir": str(bundle_dir),
                        "seed": base_seed + i,
                        "error": error,
                    }
                )
                pbar.update(1)
            continue

        for i in range(seeds_per_sample):
            seed = base_seed + i
            stats["attempted"] += 1

            processed_dir = _processed_result_dir(output_dir, bundle.sample_key, seed)
            if processed_dir is not None:
                stats["skipped"] += 1
                pbar.update(1)
                continue

            try:
                prompt = random_prompt(seed=seed, salt=bundle.sample_key)
                task_start = time.perf_counter()
                generated = run_single(pipe, bundle, config, seed=seed, prompt=prompt)
                elapsed_seconds = time.perf_counter() - task_start

                result_dir = output_dir / result_dir_name(bundle.sample_key, seed)
                save_single_result(
                    result_dir=result_dir,
                    bundle=bundle,
                    generated_rgb=generated,
                    config=config,
                    seed=seed,
                    prompt=prompt,
                    elapsed_seconds=elapsed_seconds,
                )

                stats["generated"] += 1

            except (RuntimeError, ValueError, OSError, torch.cuda.OutOfMemoryError) as e:
                print(f"\n[ERROR] {bundle.sample_key} seed={seed}: {e}")
                stats["failed"] += 1
                stats["failures"].append(
                    {"sample_key": bundle.sample_key, "seed": seed, "error": str(e)}
                )
                release_memory(force_cuda=True)

            pbar.update(1)
            if empty_cache_interval and stats["attempted"] % empty_cache_interval == 0:
                release_memory(force_cuda=True)
        del bundle
        release_memory(force_cuda=False)

    pbar.close()

    stats["anomalies"] = scan_generated_outputs(output_dir, config)
    stats["anomalous"] = len(stats["anomalies"])
    stats["elapsed_seconds"] = round(time.perf_counter() - batch_start, 3)
    if write_report:
        with open(output_dir / "batch_report.json", "w") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n批量推理完成: {stats}")
    return stats


def scan_generated_outputs(output_dir: Path, config: dict) -> list[dict[str, Any]]:
    """Scan generated.png files for all-black/all-white/near-solid anomalies."""
    anomalies: list[dict[str, Any]] = []
    if not output_dir.exists():
        return anomalies

    for result_dir in sorted(d for d in output_dir.iterdir() if d.is_dir()):
        generated_path = result_dir / "generated.png"
        if not generated_path.exists():
            continue
        image_bgr = cv2.imread(str(generated_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            anomalies.append(
                {"result_dir": result_dir.name, "path": str(generated_path), "reason": "unreadable"}
            )
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        reason = detect_image_anomaly(image_rgb, config)
        if reason:
            anomalies.append(
                {"result_dir": result_dir.name, "path": str(generated_path), "reason": reason}
            )
    return anomalies


def release_memory(force_cuda: bool = False) -> None:
    """Release Python and CUDA caches between batch tasks."""
    gc.collect()
    if force_cuda and torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass
