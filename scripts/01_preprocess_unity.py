#!/usr/bin/env python3
"""Step 1: 预处理所有 Unity Perception 输出。

将 data/unity_raw/ 下的场景转换为标准化的 SampleBundle，
生成 ControlNet 所需的 depth_cn / canny_from_mask / normal_cn，
写入 data/preprocessed/。
"""

import argparse
import random
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from blast_pile_diffusion.data.sample_bundle import SampleBundle
from blast_pile_diffusion.data.unity_reader import find_unity_scenes, iter_bundles
from blast_pile_diffusion.preprocessing.preprocessor import preprocess_and_save


def load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def preprocessing_kwargs(config: dict, args: argparse.Namespace) -> dict:
    preprocessing = config.get("preprocessing", {})
    depth_cfg = preprocessing.get("depth", {}) if isinstance(preprocessing, dict) else {}
    canny_cfg = preprocessing.get("canny", {}) if isinstance(preprocessing, dict) else {}

    depth_near = args.depth_near
    if depth_near is None:
        depth_near = float(depth_cfg.get("near_clip", 0.5))
    depth_far = args.depth_far
    if depth_far is None:
        depth_far = float(depth_cfg.get("far_clip", 100.0))

    return {
        "depth_near_clip": depth_near,
        "depth_far_clip": depth_far,
        "canny_low": int(args.canny_low if args.canny_low is not None else canny_cfg.get("low", 50)),
        "canny_high": int(
            args.canny_high if args.canny_high is not None else canny_cfg.get("high", 150)
        ),
        "canny_dilate": int(
            args.canny_dilate
            if args.canny_dilate is not None
            else canny_cfg.get("dilate_kernel", 2)
        ),
    }


def spot_check_outputs(bundle_dirs: list[Path], sample_size: int = 5) -> None:
    if not bundle_dirs:
        return
    selected = random.Random(42).sample(bundle_dirs, k=min(sample_size, len(bundle_dirs)))
    for bundle_dir in selected:
        bundle = SampleBundle.load(bundle_dir)
        missing = [
            name
            for name in ("rgb.png", "depth_cn.png", "canny_from_mask.png", "mask_instance.png", "meta.json")
            if not (bundle_dir / name).exists()
        ]
        if missing:
            raise RuntimeError(f"{bundle_dir} 输出不完整，缺少 {missing}")
        if not bundle.is_preprocessed():
            raise RuntimeError(f"{bundle_dir} 无法作为预处理 bundle 加载")


def main():
    parser = argparse.ArgumentParser(description="预处理 Unity 输出")
    parser.add_argument("--unity-dir", type=Path, default=Path("data/unity_raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/preprocessed"))
    parser.add_argument("--base-config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--depth-near", type=float, default=None, help="覆盖 base config 的 near clip")
    parser.add_argument("--depth-far", type=float, default=None, help="覆盖 base config 的 far clip；<=0 表示自动估计")
    parser.add_argument("--canny-low", type=int, default=None)
    parser.add_argument("--canny-high", type=int, default=None)
    parser.add_argument("--canny-dilate", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None, help="只处理前 N 个样本，用于轻量冒烟")
    args = parser.parse_args()

    config = load_config(args.base_config)
    kwargs = preprocessing_kwargs(config, args)
    scenes = find_unity_scenes(args.unity_dir)
    if not scenes:
        print(f"[WARN] 在 {args.unity_dir} 下未找到 Unity Perception 场景")
        return

    print(f"发现 {len(scenes)} 个场景目录")
    print(
        "预处理参数: "
        f"depth_near={kwargs['depth_near_clip']}, "
        f"depth_far={kwargs['depth_far_clip']}, "
        f"canny=({kwargs['canny_low']}, {kwargs['canny_high']}), "
        f"dilate={kwargs['canny_dilate']}"
    )

    total = 0
    saved_dirs: list[Path] = []
    started_at = time.perf_counter()
    for scene_dir in scenes:
        print(f"\n处理场景: {scene_dir.name}")
        for bundle in iter_bundles(scene_dir):
            save_path = preprocess_and_save(
                bundle,
                args.output_dir,
                **kwargs,
            )
            total += 1
            saved_dirs.append(save_path)
            print(f"  [{total}] {bundle.sample_key} → {save_path}")
            if args.max_samples is not None and total >= args.max_samples:
                break
        if args.max_samples is not None and total >= args.max_samples:
            break

    elapsed = time.perf_counter() - started_at
    if total > 0:
        spot_check_outputs(saved_dirs)
        print(f"\n抽检通过: {min(5, len(saved_dirs))} 个 bundle 可加载且文件完整")
    print(
        f"\n预处理完成，共 {total} 个样本 → {args.output_dir} "
        f"(总耗时 {elapsed:.2f}s，平均 {elapsed / total:.3f}s/帧)" if total else
        f"\n预处理完成，共 0 个样本 → {args.output_dir}"
    )


if __name__ == "__main__":
    main()
