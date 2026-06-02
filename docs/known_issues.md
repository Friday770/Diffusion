# 已知问题清单

> 更新时间：2026-05-27
> 本文档跟踪当前代码层已识别问题。已修复项仍保留，便于后续真实数据验收时回看。

---

## 已修复（2026-05-27）

| ID | 文件 | 问题 | 状态 |
|----|------|------|------|
| #1 | `coco_builder.py` | NumPy 2.0 `ptp()` 兼容 + bbox off-by-one | 已修复 |
| #2 | `unity_reader.py` | Unity InstanceSegmentation RGB colormap 需要解码为 instance id | 已修复，仍需真实 Unity 导出确认 metadata 格式 |
| #3 | `unity_reader.py` | Unity normal 需要三通道读取并反映射到 `[-1, 1]` | 已修复，仍需真实 Unity normal 目视确认轴约定 |
| #4 | `sample_bundle.py` | `normal_cn` RGB/BGR 通道交换 | 已修复 |
| #5 | `image_io.py` | EXR 读取需要在 `import cv2` 前设置 `OPENCV_IO_ENABLE_OPENEXR=1` | 已修复 |
| #6 | `prompt_bank.py` | `random_prompt` 缺少 sample salt 导致多样性不足 | 已修复 |
| #7 | `batch_processor.py` | 批量推理不应一次性持有全部 `SampleBundle` | 已修复，当前按 bundle 目录逐个加载 |
| #8 | `sample_bundle.py`, `qc_runner.py`, `batch_processor.py` | `sample_key` 分隔符不一致，scene id 含下划线时解析错误 | 已修复，统一使用 `scene--cam` 与 `_s{seed}` |
| #9 | `depth_utils.py`, `depth_processor.py` | 深度孔洞填充不应使用全局常数中值 | 已修复，使用最近有效深度 + 局部平滑 |
| #10 | `qc_runner.py` | `inf` offset 污染统计 | 已修复 |
| #11 | `depth_utils.py` | `compute_normals_from_depth` 梯度尺度不可调 | 已修复，增加 `pixel_size` / `gradient_scale` 参数 |
| #12 | `coco_builder.py` | `mask_to_polygon` 未使用 tolerance / 返回 int | 已修复 |
| #13 | `test_canny_from_mask.py` | 空测试 `pass` | 已修复 |
| #14 | `batch_processor.py` | 批量推理失败统计和异常图扫描不足 | 已修复 |
| #15 | `batch_processor.py` | 缺少周期性 `empty_cache()` | 已修复 |
| #16 | `train_launcher.py` | `subprocess.run()` 缺少 `cwd` 以适配 `sd-scripts` | 已修复 |
| #17 | `preprocessor.py` | 缺失 normal 时主流程未自动使用 depth-derived normal fallback | 已修复 |
| #18 | `check_lora_images.py` | 空 LoRA 图片目录在严格检查下仍可能通过 | 已修复 |

当前轻量测试状态：

```bash
python3 -m pytest tests -q
# 63 passed, 2 warnings

uv run --with pytest --with 'numpy<2' pytest tests -q
# 63 passed, 2 warnings
```

---

## 仍需真实数据或生产环境确认

这些不是当前代码缺口，而是必须在真实数据/GPU 环境中完成的验收项：

- 用真实 Unity Perception 导出确认 `unity_reader.py` 的 metadata 路径解析、colormap 解码、EXR/PNG depth 编码和 normal 轴约定。
- 用真实 `data/lora_real/images/` 确认 `check_lora_images.py --min-count 50 --fail-under-min`、caption 生成和 kohya 数据集结构。
- 在 CUDA 生产环境运行 `scripts/smoke_test.py`，确认 A3 的 1024x1024 输出、非黑白图、VRAM 和耗时。
- 用真实 LoRA 权重运行 D1/D3 推理，再用 E1/E2/E3 标定 QC 阈值。
- 用 QC 通过样本构建最终 COCO 数据集并运行 `scripts/dataset_statistics.py`。
