#!/usr/bin/env python3
"""Step 4: 批量 ControlNet sim2real 推理。

读取 data/preprocessed/ 下的所有 SampleBundle，
用 SDXL + ControlNet + LoRA 生成真实风格 RGB，
写入 data/generated/。支持断点续传。
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from blast_pile_diffusion.inference.batch_processor import process_batch
from blast_pile_diffusion.inference.batch_processor import scan_generated_outputs
from blast_pile_diffusion.inference.pipeline_builder import build_pipeline, release_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="批量 ControlNet 推理")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/inference/2cn_depth_canny.yaml"),
        help="推理配置文件路径",
    )
    parser.add_argument("--base-config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--input-dir", type=Path, default=Path("data/preprocessed"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/generated"))
    parser.add_argument("--seeds", type=int, default=None, help="每样本生成数量（覆盖配置文件）")
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda", help="Torch device. Default: cuda")
    parser.add_argument(
        "--controlnets",
        nargs="+",
        default=None,
        help="仅加载指定 ControlNet 名称，例如: --controlnets depth canny",
    )
    parser.add_argument("--skip-lora", action="store_true", help="跳过 LoRA 加载")
    parser.add_argument("--max-samples", type=int, default=None, help="最多处理多少个 bundle")
    parser.add_argument(
        "--empty-cache-interval",
        type=int,
        default=None,
        help="每处理 N 个任务清理一次 CUDA cache；默认读配置",
    )
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="只扫描 generated/ 下的全黑/全白/近纯色异常图，不加载模型",
    )
    parser.add_argument(
        "--fail-on-anomaly",
        action="store_true",
        help="扫描发现异常图时以非零退出码结束",
    )
    args = parser.parse_args()

    if args.scan_only:
        from blast_pile_diffusion.inference.pipeline_builder import load_config

        cfg = load_config(args.config)
        anomalies = scan_generated_outputs(args.output_dir, cfg)
        print(f"扫描目录: {args.output_dir}")
        print(f"异常图数量: {len(anomalies)}")
        for item in anomalies:
            print(f"[ANOMALY] {item['result_dir']}: {item['reason']} ({item['path']})")
        return 2 if anomalies and args.fail_on_anomaly else 0

    bundle_dirs = sorted(
        d for d in args.input_dir.iterdir() if d.is_dir() and (d / "rgb.png").exists()
    )
    if args.max_samples is not None:
        bundle_dirs = bundle_dirs[: args.max_samples]

    if not bundle_dirs:
        print(f"[WARN] 在 {args.input_dir} 下未找到预处理完毕的样本")
        return 0

    print(f"发现 {len(bundle_dirs)} 个预处理样本")
    print(f"推理配置: {args.config}")
    print(f"输出目录: {args.output_dir}")
    print(f"设备: {args.device}")
    if args.controlnets:
        print(f"ControlNet 子集: {args.controlnets}")
    if args.skip_lora:
        print("LoRA: skipped")

    print("\n加载 pipeline...")
    pipe = None
    try:
        pipe, cfg = build_pipeline(
            inference_config_path=args.config,
            base_config_path=args.base_config,
            device=args.device,
            controlnet_names=args.controlnets,
            skip_lora=args.skip_lora,
        )

        stats = process_batch(
            pipe,
            bundle_dirs,
            cfg,
            output_dir=args.output_dir,
            seeds_per_sample=args.seeds,
            base_seed=args.base_seed,
            empty_cache_interval=args.empty_cache_interval,
        )
    finally:
        release_pipeline(pipe)

    print(f"\n最终统计: {stats}")
    if args.fail_on_anomaly and stats.get("anomalous", 0):
        return 2
    return 1 if stats.get("failed", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
