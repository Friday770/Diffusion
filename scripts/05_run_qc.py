#!/usr/bin/env python3
"""Step 5: 批量 QC 过滤。

扫描 data/generated/ 下所有生成图，检查边缘对齐质量，
在每个样本目录写入 qc.json，输出总体通过率统计。
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from blast_pile_diffusion.qc.qc_runner import run_qc_batch


def main():
    parser = argparse.ArgumentParser(description="批量 QC 过滤")
    parser.add_argument("--generated-dir", type=Path, default=Path("data/generated"))
    parser.add_argument("--preprocessed-dir", type=Path, default=Path("data/preprocessed"))
    parser.add_argument("--qc-config", type=Path, default=Path("configs/qc/thresholds.yaml"))
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("--no-debug-images", action="store_true")
    args = parser.parse_args()

    print("=== QC 开始 ===")
    print(f"  生成目录:   {args.generated_dir}")
    print(f"  预处理目录: {args.preprocessed_dir}")
    print(f"  QC 配置:    {args.qc_config}")

    report = run_qc_batch(
        args.generated_dir,
        args.preprocessed_dir,
        args.qc_config,
        save_debug_images=not args.no_debug_images,
        report_path=args.report_path,
    )

    summary = report["summary"]
    print("\n========== QC 报告 ==========")
    print(f"  总数:     {summary['total']}")
    print(f"  通过:     {summary['passed']}")
    print(f"  失败:     {summary['failed']}")
    print(f"  错误:     {summary['errors']}")
    print(f"  通过率:   {summary.get('pass_rate', 0)}%")
    print(f"  平均偏移: {summary.get('mean_offset_all', 'N/A')} px")
    print(f"  中位偏移: {summary.get('median_offset_all', 'N/A')} px")
    print(f"  Debug图:  {summary.get('debug_images_written', 0)}")
    print("==============================")

    report_path = args.report_path or args.generated_dir / "qc_report.json"
    print(f"\n详细报告 → {report_path}")

    low_scene_ids = report.get("low_pass_rate_scenes", [])
    if low_scene_ids:
        print("\n[WARN] 以下场景通过率低于配置阈值，建议优先检查 debug_overlay：")
        for scene_id in low_scene_ids[:20]:
            scene = report["per_scene"][scene_id]
            print(f"  - {scene_id}: {scene['pass_rate_pct']}% ({scene['passed']}/{scene['total']})")
        if len(low_scene_ids) > 20:
            print(f"  ... 还有 {len(low_scene_ids) - 20} 个场景")

    if summary.get("pass_rate", 0) < 60:
        print("\n[WARN] 通过率低于 60%，建议排查：")
        print("  1. Canny 边缘是否来自 mask 而非 RGB")
        print("  2. 提高 Canny ControlNet 权重")
        print("  3. 降低 denoising strength")
        print("  4. 检查 QC 阈值是否过严")


if __name__ == "__main__":
    main()
