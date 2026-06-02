# 爆堆扩散模型数据生成 — 子任务拆分文档

> 项目：blast-pile-diffusion（阶段三：ControlNet sim2real 翻译）
> 编制日期：2026-05-27
> 总任务数：25 个子任务，分 7 个阶段

---

## 任务依赖总图

```
Phase A: 基础设施                Phase B: 预处理管线            Phase C: LoRA 训练
┌─────┐   ┌─────┐              ┌─────┐                       ┌─────┐
│ A1  │──►│ A2  │──►┐          │ B1  │──►┐                   │ C1  │──►┐
└─────┘   └─────┘   │          └─────┘   │                   └─────┘   │
                     ▼          ┌─────┐   │  ┌─────┐         ┌─────┐   │  ┌─────┐
                  ┌─────┐       │ B2  │──►├─►│ B5  │         │ C2  │──►├─►│ C4  │
                  │ A3  │       └─────┘   │  └──┬──┘         └─────┘   │  └──┬──┘
                  └──┬──┘       ┌─────┐   │     │            ┌─────┐   │     │
                     │          │ B3  │──►┘     │            │ C3  │──►┘     │
                     │          └─────┘         │            └─────┘         │
                     │          ┌─────┐         │                            │
                     │          │ B4  │──►──────┘                            │
                     │          └─────┘                                      │
                     │                                                       │
                     ▼                                                       ▼
Phase D: ControlNet 推理         Phase E: 质量控制              Phase F: 数据集组装
┌─────┐                         ┌─────┐                       ┌─────┐
│ D1  │──►┐                     │ E1  │──►┐                   │ F1  │──►┐
└─────┘   │  ┌─────┐           └─────┘   │  ┌─────┐         └─────┘   │  ┌─────┐
┌─────┐   ├─►│ D3  │──────────►┌─────┐   ├─►│ E3  │──────────►┌─────┐├─►│ F3  │
│ D2  │──►┘  └─────┘           │ E2  │──►┘  └─────┘         │ F2  │──►┘  └─────┘
└─────┘                        └─────┘                       └─────┘
                                                                          │
                                                                          ▼
                                                              Phase G: 集成交付
                                                              ┌─────┐  ┌─────┐
                                                              │ G1  │─►│ G2  │
                                                              └─────┘  └─────┘
```

**关键路径**：A1 → A2 → A3 → D1 → D2 → D3 → E3 → F2 → F3 → G1 → G2

---

## Phase A：基础设施搭建 / Infrastructure Setup

---

### A1 — Python 环境与依赖安装

| 项目 | 内容 |
|------|------|
| **目标** | 创建可复现的 Python 环境，安装全部推理与训练依赖 |
| **输入** | `pyproject.toml`, `requirements.txt`, `requirements_train.txt` |
| **输出** | 可用的 conda/venv 环境，`pip install -e .` 成功 |
| **边界约束** | Python ≥ 3.10；PyTorch ≥ 2.0 + CUDA ≥ 11.8；不修改项目源码，只管环境 |
| **依赖** | 无 |
| **验收标准** | ① `python -c "import torch; print(torch.cuda.is_available())"` 输出 True ② `python -c "from blast_pile_diffusion.data.sample_bundle import SampleBundle"` 无报错 ③ `pytest tests/ -x` 全部通过 |

**中文提示词：**
> 在项目根目录 /Users/mauschiel/Desktop/Diffusion 下，创建一个 Python 虚拟环境（conda 或 venv 均可），安装 pyproject.toml 中声明的全部依赖以及 requirements_train.txt 中的训练依赖。要求 PyTorch 版本 ≥ 2.0 且支持 CUDA。安装完成后，用 `pip install -e .` 以开发模式安装本项目包。最后运行 `pytest tests/ -x` 确认所有现有测试通过。不要修改任何项目源码文件。

**English Prompt：**
> Set up a Python virtual environment (conda or venv) at the project root /Users/mauschiel/Desktop/Diffusion. Install all dependencies declared in pyproject.toml plus training extras from requirements_train.txt. PyTorch must be ≥ 2.0 with CUDA support. Run `pip install -e .` to install the package in editable mode. Verify by running `pytest tests/ -x` — all existing tests must pass. Do not modify any source code files.

---

### A2 — 预训练模型下载与缓存验证

| 项目 | 内容 |
|------|------|
| **目标** | 将 SDXL、ControlNet、VAE 预训练权重下载到本地 HuggingFace 缓存 |
| **输入** | `configs/base.yaml`（含模型 ID）, `configs/inference/2cn_depth_canny.yaml` |
| **输出** | `~/.cache/huggingface/hub/` 下存在全部所需模型文件 |
| **边界约束** | 仅下载模型权重，不启动推理；不修改配置文件；若网络不稳定可指定镜像 |
| **依赖** | A1 |
| **验收标准** | 写一个 `verify_models.py` 脚本，对每个模型 ID 调用 `from_pretrained(..., local_files_only=True)` 不报错 |

**中文提示词：**
> 读取 configs/base.yaml 和 configs/inference/2cn_depth_canny.yaml 中声明的所有 HuggingFace 模型 ID（SDXL base、VAE、ControlNet-depth、ControlNet-canny），编写一个脚本 scripts/verify_models.py 逐个下载并验证。每个模型用 diffusers 的 from_pretrained() 加载一次（dtype=float16），加载成功后立即 del 释放显存。最终用 local_files_only=True 重新加载确认缓存完整。脚本应打印每个模型的下载状态和磁盘占用大小。如果 HuggingFace 访问不稳定，在脚本中支持通过环境变量 HF_ENDPOINT 设置镜像地址。

**English Prompt：**
> Read all HuggingFace model IDs from configs/base.yaml and configs/inference/2cn_depth_canny.yaml (SDXL base, VAE, ControlNet-depth, ControlNet-canny). Write scripts/verify_models.py that downloads each model via diffusers from_pretrained() with dtype=float16, then immediately deletes it to free VRAM. After all downloads, re-load each with local_files_only=True to confirm the cache is intact. Print download status and disk size for each model. Support HF_ENDPOINT env var for mirror sites.

---

### A3 — SDXL + ControlNet 最小化冒烟测试

| 项目 | 内容 |
|------|------|
| **目标** | 用一张合成测试图跑通 SDXL + 单路 ControlNet-Depth 推理，确认 GPU 管线畅通 |
| **输入** | 一张任意 1024×1024 RGB 图 + 对应深度图（可用纯色/渐变伪造） |
| **输出** | 一张 1024×1024 生成图（不要求质量，只要不报错不黑图） |
| **边界约束** | 不加载 LoRA；仅用单路 ControlNet-Depth；不使用项目的 batch_processor，直接用 pipeline_builder + controlnet_runner 的最短路径 |
| **依赖** | A2 |
| **验收标准** | ① 输出图尺寸 1024×1024 ② 不全黑/全白 ③ GPU 显存峰值 < 24GB ④ 推理耗时 < 60 秒 |

**中文提示词：**
> 编写 scripts/smoke_test.py：用 numpy 创建一张 1024×1024 的渐变色 RGB 测试图和一张从近到远的线性深度图。用 blast_pile_diffusion.inference.pipeline_builder.build_pipeline() 加载 configs/inference/2cn_depth_canny.yaml 配置（但跳过 LoRA 加载），然后调用 blast_pile_diffusion.inference.controlnet_runner.run_single() 做一次推理。将生成图保存到 data/generated/smoke_test.png。验证输出图尺寸为 1024×1024、非全黑/全白、GPU 显存峰值不超过 24GB。此脚本仅用于验证环境，不追求生成质量。

**English Prompt：**
> Write scripts/smoke_test.py: create a synthetic 1024×1024 gradient RGB image and a linear depth map using numpy. Load the pipeline via blast_pile_diffusion.inference.pipeline_builder.build_pipeline() with configs/inference/2cn_depth_canny.yaml (skip LoRA loading). Run one inference via blast_pile_diffusion.inference.controlnet_runner.run_single(). Save output to data/generated/smoke_test.png. Verify: output is 1024×1024, not all-black/white, peak VRAM < 24GB. This is an environment validation script only — generation quality does not matter.

---

## Phase B：数据预处理管线 / Data Preprocessing Pipeline

---

### B1 — Unity Perception 数据解析器适配

| 项目 | 内容 |
|------|------|
| **目标** | 确保 unity_reader.py 能正确解析实际的 Unity Perception 导出目录 |
| **输入** | data/unity_raw/ 下至少 1 个 Unity Perception 导出场景 |
| **输出** | 成功生成 SampleBundle 对象，RGB/depth/normal/mask 维度和值域正确 |
| **边界约束** | 只修改 blast_pile_diffusion/data/unity_reader.py；不改变 SampleBundle 接口；如果 Unity 输出格式与预设不同，适配 reader 而非改 Unity |
| **依赖** | A1；需要至少 1 个真实 Unity 导出场景 |
| **验收标准** | ① `iter_bundles(scene_dir)` 能 yield 出至少 1 个 bundle ② bundle.rgb.shape == (H, W, 3) 且 dtype==uint8 ③ bundle.depth 为 float32 且值域合理（0.5~100m）④ bundle.mask 中 unique ID 数 > 1 ⑤ bundle.num_instances > 0 |

**中文提示词：**
> 将一个真实的 Unity Perception 导出场景放到 data/unity_raw/ 下。检查 blast_pile_diffusion/data/unity_reader.py 的 iter_bundles() 函数能否正确解析该场景。如果 Unity 的实际目录结构（子文件夹命名、文件命名规则、深度图格式）与 reader 预设不匹配，修改 unity_reader.py 来适配。要求：返回的 SampleBundle 中 rgb 为 (H,W,3) uint8，depth 为 (H,W) float32 米单位，mask 为 (H,W) int32 且背景=0。编写 tests/test_unity_reader.py 验证以上条件。不要修改 SampleBundle 的接口定义。

**English Prompt：**
> Place a real Unity Perception export scene under data/unity_raw/. Test whether blast_pile_diffusion.data.unity_reader.iter_bundles() correctly parses it. If the actual directory structure (subfolder naming, file naming, depth format) differs from the reader's assumptions, adapt unity_reader.py — do not change the Unity export. Requirements: returned SampleBundle must have rgb=(H,W,3) uint8, depth=(H,W) float32 in meters, mask=(H,W) int32 with background=0. Write tests/test_unity_reader.py to verify. Do not change the SampleBundle interface.

---

### B2 — 深度图预处理模块验证

| 项目 | 内容 |
|------|------|
| **目标** | 用真实 Unity 深度图验证 depth_processor.py 的归一化和孔洞填充效果 |
| **输入** | B1 产出的 SampleBundle 中的 depth 字段 |
| **输出** | 归一化后的 depth_cn (H,W,3) uint8，近白远黑 |
| **边界约束** | near_clip / far_clip 需根据实际 Unity 场景尺度调整，写入 configs/base.yaml；不改变 ControlNet 期望的输入格式 |
| **依赖** | B1 |
| **验收标准** | ① depth_cn 值域 [0, 255] ② 近处像素亮度 > 远处像素亮度 ③ 无 NaN/Inf ④ 孔洞（depth=0）区域被合理填充 ⑤ 保存可视化对比图（原始深度 vs 归一化深度）到 notebooks/ |

**中文提示词：**
> 用 B1 产出的真实 Unity 深度图测试 blast_pile_diffusion/preprocessing/depth_processor.py。检查 process_depth() 的输出是否满足：(H,W,3) uint8，近处亮远处暗，无 NaN。如果 Unity 场景的深度范围不在默认的 0.5~100m 内，调整 configs/base.yaml 中的参数并在 depth_processor.py 中支持从配置读取。如果原始深度图存在 depth=0 的孔洞，验证 fill_depth_holes() 是否有效。在 notebooks/02_depth_normalization_tuning.ipynb 中画出原始深度直方图、归一化后的伪彩色图、孔洞填充前后对比。

**English Prompt：**
> Test blast_pile_diffusion/preprocessing/depth_processor.py with real Unity depth maps from B1. Verify process_depth() output: (H,W,3) uint8, near=bright far=dark, no NaN. If scene depth range differs from the 0.5–100m default, adjust configs/base.yaml and make depth_processor.py read from config. Verify fill_depth_holes() handles depth=0 holes. Create notebooks/02_depth_normalization_tuning.ipynb with: raw depth histogram, pseudo-colored normalized depth, before/after hole-filling comparison.

---

### B3 — 掩码→Canny 边缘派生验证（关键模块）

| 项目 | 内容 |
|------|------|
| **目标** | 确认 mask_to_canny 输出的边缘图精确对应每个实例的轮廓线 |
| **输入** | B1 产出的 SampleBundle 中的 mask 字段 |
| **输出** | canny 边缘图 (H,W) uint8 |
| **边界约束** | ⚠️ **关键约束：Canny 只能从 instance mask 派生，绝不能从 RGB 提取**。这是整套方案的核心假设，违反等于标签全错 |
| **依赖** | B1 |
| **验收标准** | ① 每个实例的边界上都有边缘像素 ② 实例内部无杂散边缘 ③ 边缘线条连续、无断裂 ④ 将 canny 叠加到 RGB 上目视检查边界对齐 ⑤ tests/test_canny_from_mask.py 全通过 |

**中文提示词：**
> 用 B1 产出的真实 Unity 实例掩码测试 blast_pile_diffusion/preprocessing/canny_from_mask.py 的 mask_to_canny() 函数。这是整个方案最关键的模块——Canny 边缘必须从实例掩码派生，绝不能从 RGB 提取。验证：① 每个实例（mask 中每个唯一 ID）的边界上都存在 Canny 边缘像素 ② 实例内部区域（远离边界）没有杂散边缘 ③ 边缘线条连续不断裂。在 notebooks/03_canny_threshold_sweep.ipynb 中，将不同 canny_low/canny_high 参数下的边缘图叠加到 Unity RGB 上做可视化对比，找到最佳参数写回 configs/qc/thresholds.yaml。运行 pytest tests/test_canny_from_mask.py 确认通过。

**English Prompt：**
> Test blast_pile_diffusion/preprocessing/canny_from_mask.py mask_to_canny() with real Unity instance masks from B1. THIS IS THE MOST CRITICAL MODULE — Canny edges MUST come from the instance mask, NEVER from RGB. Verify: ① every instance boundary has edge pixels ② no spurious edges inside instances ③ edge lines are continuous. In notebooks/03_canny_threshold_sweep.ipynb, overlay edges on Unity RGB at different canny_low/canny_high values, find optimal params, write back to configs/qc/thresholds.yaml. Run pytest tests/test_canny_from_mask.py.

---

### B4 — 法线图格式转换验证

| 项目 | 内容 |
|------|------|
| **目标** | 确认 normal_processor.py 正确将 Unity 法线转为 ControlNet-Normal 输入格式 |
| **输入** | B1 产出的 SampleBundle 中的 normal 字段 |
| **输出** | normal_cn (H,W,3) uint8 RGB 编码法线图 |
| **边界约束** | Unity 输出法线的坐标约定（左/右手、Y-up/Z-up）需与 ControlNet-Normal 期望对齐；若不一致需在 normal_processor.py 中做轴翻转 |
| **依赖** | B1 |
| **验收标准** | ① 值域 [0, 255] ② 朝上平面呈蓝色（Z 分量高）③ 法线图目视上石块几何结构清晰 ④ 若 Unity 无法线输出，验证从 depth 派生法线的备选路径 |

**中文提示词：**
> 用 B1 产出的真实 Unity 法线图测试 blast_pile_diffusion/preprocessing/normal_processor.py 的 process_normal()。检查 Unity 输出法线的坐标系约定（可能是左手 Y-up 或右手 Z-up）是否与 ControlNet-Normal 期望的格式一致。如果不一致，在 process_normal() 中添加轴交换/翻转逻辑。验证输出 normal_cn 为 (H,W,3) uint8，朝上平面呈蓝色（Z 分量高）。如果 Unity 场景没有导出法线图，验证 from_depth=True 备选路径：从深度图派生法线是否在视觉上合理。

**English Prompt：**
> Test blast_pile_diffusion/preprocessing/normal_processor.py process_normal() with real Unity normal maps from B1. Check if Unity's normal coordinate convention (possibly left-hand Y-up or right-hand Z-up) matches ControlNet-Normal expectations. If mismatched, add axis swapping/flipping in process_normal(). Verify output normal_cn is (H,W,3) uint8 with upward-facing surfaces appearing blue (high Z component). If Unity scene has no normal export, verify the from_depth=True fallback path produces visually reasonable normals.

---

### B5 — 预处理全流程端到端测试

| 项目 | 内容 |
|------|------|
| **目标** | 用 Script 01 对全部 Unity 场景执行完整预处理，验证输出完整性 |
| **输入** | data/unity_raw/ 全部场景 |
| **输出** | data/preprocessed/ 下每个 sample_key 目录包含完整 bundle |
| **边界约束** | 每个 bundle 目录必须包含 rgb.png + depth_cn.png + canny_from_mask.png + mask_instance.png + meta.json；normal_cn.png 可选 |
| **依赖** | B2, B3, B4 |
| **验收标准** | ① data/preprocessed/ 下目录数 = Unity 导出帧数 ② 每个目录文件完整 ③ 无报错 ④ 总处理耗时记录 |

**中文提示词：**
> 运行 `python scripts/01_preprocess_unity.py --unity-dir data/unity_raw --output-dir data/preprocessed`，对 data/unity_raw/ 下的所有 Unity 场景执行完整预处理。运行结束后验证：① data/preprocessed/ 下的子目录数等于 Unity 导出的总帧数 ② 每个子目录包含 rgb.png、depth_cn.png、canny_from_mask.png、mask_instance.png、meta.json 五个文件 ③ 随机抽检 5 个 bundle，用 SampleBundle.load() 加载确认数据完整。记录总处理耗时和每帧平均耗时。如果有任何帧处理失败，定位原因并修复，不要跳过。

**English Prompt：**
> Run `python scripts/01_preprocess_unity.py --unity-dir data/unity_raw --output-dir data/preprocessed` on all Unity scenes. Verify: ① number of subdirectories in data/preprocessed/ equals total exported frames ② each subdirectory contains rgb.png, depth_cn.png, canny_from_mask.png, mask_instance.png, meta.json ③ spot-check 5 random bundles with SampleBundle.load(). Record total time and per-frame average. If any frame fails, diagnose and fix — do not skip.

---

## Phase C：LoRA 风格迁移训练 / LoRA Style Training

---

### C1 — 真实爆堆图片采集与整理

| 项目 | 内容 |
|------|------|
| **目标** | 收集 50–200 张真实矿场爆堆照片，整理到标准目录 |
| **输入** | 现场拍摄的原始照片 |
| **输出** | data/lora_real/images/ 下 50–200 张 ≥1024×1024 的图片 |
| **边界约束** | 必须覆盖多种光照（晴/阴/早/午/黄昏）和天气（干/湿/扬尘）；分辨率 ≥ 1024×1024；剔除模糊、遮挡严重、非爆堆主体的图；此任务为人工操作，不写代码 |
| **依赖** | 无 |
| **验收标准** | ① 图片数量 ≥ 50 ② 最小分辨率 ≥ 1024×1024 ③ 至少 3 种光照条件 ④ 图片清晰、爆堆占画面 >50% |

**中文提示词：**
> 将收集到的真实矿场爆堆照片整理到 data/lora_real/images/ 目录下。编写 scripts/check_lora_images.py 脚本，扫描该目录并报告：① 图片总数 ② 最小/最大/平均分辨率 ③ 分辨率低于 1024×1024 的图片列表（需要手动替换或上采样）④ 文件格式分布。对于分辨率不足的图片，用 Pillow 的 LANCZOS 上采样到至少 1024 短边。确保最终所有图片 ≥ 1024×1024。

**English Prompt：**
> Organize collected real mine blast pile photos into data/lora_real/images/. Write scripts/check_lora_images.py to scan the directory and report: ① total count ② min/max/average resolution ③ list of images below 1024×1024 (need replacement or upscaling) ④ format distribution. For undersized images, upscale to at least 1024 on the short side using Pillow LANCZOS. Ensure all final images are ≥ 1024×1024.

---

### C2 — Caption 生成与训练数据构建

| 项目 | 内容 |
|------|------|
| **目标** | 为所有真实图片生成 caption，构建 kohya_ss 兼容的训练目录 |
| **输入** | data/lora_real/images/ 下的图片 |
| **输出** | data/lora_real/kohya_dataset/{repeats}_{concept}/ 目录，包含图片+caption 对 |
| **边界约束** | 每条 caption 必须包含触发词 `<mine_blast_pile>`；caption 不能过于雷同，需描述具体场景特征；repeats 默认 10 |
| **依赖** | C1 |
| **验收标准** | ① 每张图片都有对应 .txt caption ② 每条 caption 包含触发词 ③ caption 间有足够差异（非全部复制粘贴）④ kohya_ss 目录结构正确 |

**中文提示词：**
> 运行 `python scripts/02_prepare_lora_data.py --image-dir data/lora_real/images --caption-dir data/lora_real/captions --output-dir data/lora_real/kohya_dataset`。首先检查生成的 caption 质量：打开 data/lora_real/captions/ 下的 .txt 文件，确认每条都包含触发词 `<mine_blast_pile>` 且描述了不同的场景特征。如果使用模板生成的 caption 过于雷同，改用 --use-blip2 模式重新生成（需要 GPU）。然后检查 kohya_dataset/ 目录结构是否为 `{repeats}_{concept}/` 格式，内含图片和对应 .txt 文件。

**English Prompt：**
> Run `python scripts/02_prepare_lora_data.py --image-dir data/lora_real/images --caption-dir data/lora_real/captions --output-dir data/lora_real/kohya_dataset`. Check generated caption quality: open .txt files in data/lora_real/captions/, verify each contains trigger word `<mine_blast_pile>` and describes distinct scene features. If template captions are too similar, re-run with --use-blip2 (requires GPU). Verify kohya_dataset/ directory structure follows `{repeats}_{concept}/` format with paired images and .txt files.

---

### C3 — LoRA 训练执行

| 项目 | 内容 |
|------|------|
| **目标** | 在 SDXL base 上训练风格 LoRA，产出 mine_blast_pile.safetensors |
| **输入** | C2 产出的 kohya_dataset, configs/lora/sdxl_rank32.yaml |
| **输出** | lora_weights/mine_blast_pile.safetensors |
| **边界约束** | 需要 sd-scripts 仓库已克隆且可运行；GPU 显存 ≥ 24GB；训练步数 2000–3000；不修改 SDXL base 权重本身 |
| **依赖** | A2, C2 |
| **验收标准** | ① safetensors 文件生成 ② 文件大小合理（rank32 约 50–150MB）③ 训练 loss 曲线下降并收敛 ④ 无 NaN loss |

**中文提示词：**
> 首先确保 sd-scripts (kohya_ss) 已克隆到本地。然后运行 `bash scripts/03_train_lora.sh`（先用 --dry-run 检查命令是否正确）。训练过程中监控 GPU 显存使用和 loss 变化。训练完成后确认 lora_weights/mine_blast_pile.safetensors 已生成，文件大小在 50–150MB 之间（rank=32 的合理范围）。如果 sd-scripts 的接口与脚本中的参数名不匹配（版本差异），修改 scripts/03_train_lora.sh 和 blast_pile_diffusion/lora/train_launcher.py 来适配。不要修改 configs/lora/sdxl_rank32.yaml 中的超参数，除非训练明确失败。

**English Prompt：**
> Ensure sd-scripts (kohya_ss) is cloned locally. Run `bash scripts/03_train_lora.sh` (first with --dry-run to verify command). Monitor GPU VRAM and loss during training. After completion, verify lora_weights/mine_blast_pile.safetensors exists and is 50–150MB (reasonable for rank=32). If sd-scripts CLI has interface changes, adapt scripts/03_train_lora.sh and blast_pile_diffusion/lora/train_launcher.py. Do not change hyperparameters in configs/lora/sdxl_rank32.yaml unless training explicitly fails.

---

### C4 — LoRA 质量验证与迭代

| 项目 | 内容 |
|------|------|
| **目标** | 用训好的 LoRA 生成测试图，目视判断风格是否接近真实矿场 |
| **输入** | lora_weights/mine_blast_pile.safetensors |
| **输出** | 4–8 张验证图 + 质量判断报告 |
| **边界约束** | 此阶段只评估 LoRA 风格效果，不涉及 ControlNet 结构约束；如果风格不对需要回到 C3 调整训练参数而非修改推理代码 |
| **依赖** | C3 |
| **验收标准** | ① 生成图的颜色调性、光照质感接近真实爆堆照片 ② 触发词有效（加入触发词 vs 不加有明显差异）③ 不存在严重 artifact（坏手、文字等非自然物体）|

**中文提示词：**
> 运行 blast_pile_diffusion/lora/lora_validator.py 中的 validate_lora() 函数，用训好的 LoRA 生成 4 张验证图到 data/generated/lora_validation/。将这 4 张图与 data/lora_real/images/ 中的真实照片放在一起对比。判断：① 颜色调性是否接近（灰褐色的岩石、黄土色的扬尘）② 光照质感是否真实 ③ 是否有明显 artifact。如果风格差距大，按以下顺序排查：增加真实图到 150 张 → 增加训练步数到 4000 → 提高 LoRA rank 到 64。每次修改后重新训练并验证。

**English Prompt：**
> Run validate_lora() from blast_pile_diffusion/lora/lora_validator.py to generate 4 validation images to data/generated/lora_validation/. Compare side-by-side with real photos from data/lora_real/images/. Judge: ① color tone similarity (gray-brown rocks, dusty atmosphere) ② lighting realism ③ no severe artifacts. If style gap is large, troubleshoot in order: add more real images to 150 → increase training steps to 4000 → increase LoRA rank to 64. Retrain and re-validate after each change.

---

## Phase D：ControlNet 推理管线 / ControlNet Inference Pipeline

---

### D1 — 单样本推理管线调通

| 项目 | 内容 |
|------|------|
| **目标** | 用一个真实的预处理 bundle + LoRA，生成一张完整的 sim2real 翻译图 |
| **输入** | data/preprocessed/ 下任一 bundle, lora_weights/mine_blast_pile.safetensors |
| **输出** | 一张 1024×1024 真实风格 RGB 图 |
| **边界约束** | 使用 2cn_depth_canny.yaml 配置；输出图需保存到 data/generated/ 下规范目录 |
| **依赖** | B5, C4 |
| **验收标准** | ① 输出图 1024×1024 非全黑 ② 视觉上像真实爆堆照片 ③ 石块大致结构可辨（没有完全融化/扭曲）④ 推理耗时 < 60s |

**中文提示词：**
> 用一个真实的预处理 bundle 跑通完整的 sim2real 推理管线。步骤：① SampleBundle.load() 加载 data/preprocessed/ 下任一目录 ② build_pipeline() 加载带 LoRA 的完整管线 ③ run_single() 执行推理 ④ 保存结果到 data/generated/{sample_key}_s42/generated.png。将 Unity 原始 RGB、深度图、Canny 边缘图、生成图这四张排成一行对比图保存。检查生成图是否看起来像真实爆堆照片、石块结构是否大致保持。如果出现全黑图，检查 VAE 是否用了 fp16-fix 版本。

**English Prompt：**
> Run the full sim2real inference pipeline with one real preprocessed bundle. Steps: ① SampleBundle.load() from any data/preprocessed/ directory ② build_pipeline() with LoRA ③ run_single() for inference ④ save to data/generated/{sample_key}_s42/generated.png. Create a 4-panel comparison: Unity RGB, depth map, Canny edges, generated image. Check if generated image looks like a real blast pile photo with preserved rock structure. If output is all-black, verify VAE uses the fp16-fix version.

---

### D2 — 超参数搜索

| 项目 | 内容 |
|------|------|
| **目标** | 找到 ControlNet 权重、denoising strength、guidance scale 的最优组合 |
| **输入** | 3–5 个 bundle + D1 已调通的管线 |
| **输出** | 最优参数组合，写回 configs/inference/2cn_depth_canny.yaml |
| **边界约束** | 搜索范围：depth_scale [0.5, 0.7, 0.9], canny_scale [0.4, 0.6, 0.8], strength [0.30, 0.40, 0.50, 0.60], guidance [5.0, 6.5, 8.0]；每组合只需跑 1 个 seed；用目视+QC 分数联合评判 |
| **依赖** | D1, E1 |
| **验收标准** | ① 搜索覆盖至少 36 种参数组合 ② 对每种组合记录 QC mean_offset 和目视评分 ③ 最优组合的 QC pass rate > 70% ④ 参数写回配置文件 |

**中文提示词：**
> 在 notebooks/04_inference_param_sweep.ipynb 中实现超参数搜索。选取 3–5 个代表性 bundle（含不同视角和石块密度）。对以下参数做网格搜索：depth ControlNet scale [0.5, 0.7, 0.9]、canny ControlNet scale [0.4, 0.6, 0.8]、denoising strength [0.30, 0.40, 0.50, 0.60]、guidance scale [5.0, 6.5, 8.0]。每种组合用 1 个 seed 生成一张图，调用 check_edge_alignment() 计算 QC 分数，同时保存缩略图以便目视。最终用 QC mean_offset 排序找到最优组合，将最优参数写回 configs/inference/2cn_depth_canny.yaml。如果所有组合的 pass rate 都 < 50%，按技术路线 §4.6 的排查流程处理。

**English Prompt：**
> In notebooks/04_inference_param_sweep.ipynb, implement hyperparameter search. Select 3–5 representative bundles (varying viewpoints and rock density). Grid search: depth ControlNet scale [0.5, 0.7, 0.9], canny scale [0.4, 0.6, 0.8], denoising strength [0.30, 0.40, 0.50, 0.60], guidance scale [5.0, 6.5, 8.0]. For each combo, generate 1 image per seed, run check_edge_alignment() for QC score, save thumbnails for visual review. Rank by QC mean_offset, write best params to configs/inference/2cn_depth_canny.yaml. If all combos have pass rate < 50%, follow troubleshooting from technical roadmap §4.6.

---

### D3 — 批量推理执行

| 项目 | 内容 |
|------|------|
| **目标** | 对全部预处理样本执行批量 ControlNet 推理 |
| **输入** | data/preprocessed/ 全部 bundle, D2 确定的最优配置 |
| **输出** | data/generated/ 下每个 {sample_key}_s{seed}/ 目录包含 generated.png + meta.json |
| **边界约束** | 每样本 4 个 seed（seeds_per_sample=4）；断点续传（已处理的跳过）；GPU 显存不能 OOM |
| **依赖** | D2 |
| **验收标准** | ① 生成总数 = 预处理样本数 × 4 ② 中途无崩溃（或崩溃后可续传恢复）③ 单张推理耗时稳定 ④ 无全黑/全白异常图 |

**中文提示词：**
> 运行 `python scripts/04_run_inference.py --config configs/inference/2cn_depth_canny.yaml --seeds 4`。监控进度和 GPU 使用。预计总量 = data/preprocessed/ 下的目录数 × 4。如果中途 OOM 崩溃，检查是否开启了 pipe.enable_model_cpu_offload()；如果仍然 OOM，在 batch_processor.py 每处理 50 张后手动 torch.cuda.empty_cache()。运行结束后统计：生成总数、跳过数（断点续传）、失败数。检查 data/generated/ 下无全黑/全白的异常图（可用脚本扫描像素均值）。

**English Prompt：**
> Run `python scripts/04_run_inference.py --config configs/inference/2cn_depth_canny.yaml --seeds 4`. Monitor progress and GPU usage. Expected total = preprocessed directories × 4. If OOM occurs, verify pipe.enable_model_cpu_offload() is active; if still OOM, add torch.cuda.empty_cache() every 50 images in batch_processor.py. After completion, report: total generated, skipped (resume), failed. Scan data/generated/ for anomalous all-black/white images (check pixel mean).

---

## Phase E：质量控制系统 / Quality Control System

---

### E1 — 边缘对齐 QC 模块调试

| 项目 | 内容 |
|------|------|
| **目标** | 用真实生成图验证 edge_alignment.py 的 QC 判断逻辑正确 |
| **输入** | D1 产出的生成图 + 对应掩码 |
| **输出** | QCResult 对象，含 mean_offset / p99_offset / passed |
| **边界约束** | QC 只检查边缘偏移，不判断风格质量；distance_transform_edt 方向是从生成图边缘到掩码边缘 |
| **依赖** | D1 |
| **验收标准** | ① 好样本（肉眼边界对齐）的 mean_offset < 4px ② 差样本（肉眼明显错位）的 mean_offset > 8px ③ QC 判断与人工判断一致率 > 80% ④ debug_overlay 图清晰可读 |

**中文提示词：**
> 用 D1 产出的生成图和对应掩码测试 blast_pile_diffusion/qc/edge_alignment.py 的 check_edge_alignment()。准备两组样本：① 人工判断边界对齐较好的 2–3 张 ② 人工判断明显错位的 2–3 张。分别运行 QC，检查 mean_offset 是否能区分好坏样本（好样本 < 4px，差样本 > 8px）。用 vis.py 的 save_qc_debug_image() 生成 debug 叠加图，目视确认边缘检测合理。如果 QC 无法区分好坏，排查 Canny 参数或 distance_transform 方向是否有误。运行 pytest tests/test_edge_alignment_qc.py 确认通过。

**English Prompt：**
> Test blast_pile_diffusion/qc/edge_alignment.py check_edge_alignment() with real generated images from D1. Prepare two groups: ① 2–3 images with visually good alignment ② 2–3 with obvious misalignment. Run QC on both, check if mean_offset distinguishes them (good < 4px, bad > 8px). Use save_qc_debug_image() for visual verification. If QC can't distinguish, investigate Canny params or distance_transform direction. Run pytest tests/test_edge_alignment_qc.py.

---

### E2 — QC 阈值标定

| 项目 | 内容 |
|------|------|
| **目标** | 基于实际生成图的偏移分布，确定最终 QC 阈值 |
| **输入** | D3 产出的部分生成图（至少 50 张）+ 掩码 |
| **输出** | 更新后的 configs/qc/thresholds.yaml |
| **边界约束** | 阈值太严（pass rate < 50%）数据量不够；太松（pass rate > 95%）质量没保障；目标 pass rate 60–85% |
| **依赖** | E1, D3（部分输出即可） |
| **验收标准** | ① 画出 mean_offset 直方图 ② 标注阈值切点 ③ 目标 pass rate 在 60–85% 区间 ④ 人工抽查 passed 样本确认质量可接受 |

**中文提示词：**
> 在 notebooks/05_qc_threshold_calibration.ipynb 中，对 data/generated/ 下的前 50–100 张生成图批量运行 check_edge_alignment()（使用宽松默认阈值）。画出 mean_offset 和 p99_offset 的直方图和累积分布曲线。在图上标注不同阈值对应的 pass rate。目标：找到一组阈值使 pass rate 在 60–85% 之间。然后人工抽查 10 张刚好在阈值附近（pass/fail 边界）的样本，确认阈值合理。将最终阈值写回 configs/qc/thresholds.yaml。

**English Prompt：**
> In notebooks/05_qc_threshold_calibration.ipynb, run check_edge_alignment() on the first 50–100 images from data/generated/ with permissive thresholds. Plot histograms and CDFs of mean_offset and p99_offset. Mark pass rates at different threshold values. Target: find thresholds giving 60–85% pass rate. Manually inspect 10 borderline samples (near pass/fail boundary) to confirm the threshold is reasonable. Write final thresholds to configs/qc/thresholds.yaml.

---

### E3 — 批量 QC 执行与报告分析

| 项目 | 内容 |
|------|------|
| **目标** | 对全部生成图执行 QC，产出统计报告，识别系统性问题 |
| **输入** | data/generated/ 全部样本, E2 确定的阈值 |
| **输出** | 每个样本目录下 qc.json + data/generated/qc_report.json |
| **边界约束** | QC 不删除任何文件，只写标记；debug 图仅对 failed 样本生成（节省磁盘） |
| **依赖** | E2, D3 |
| **验收标准** | ① 所有样本都有 qc.json ② pass rate 在预期范围 ③ 按场景的 pass rate 无极端偏差（某场景 pass rate < 20% 需排查）④ 报告文件完整 |

**中文提示词：**
> 运行 `python scripts/05_run_qc.py --qc-config configs/qc/thresholds.yaml`。检查输出报告中的 pass rate 是否在 E2 标定的预期范围内。分析 per_scene 统计：如果某个场景的 pass rate 显著低于平均（< 20%），查看该场景的 debug_overlay 图，判断是场景本身有问题（如极端视角、遮挡严重）还是 ControlNet 推理参数需要针对性调整。输出一份简要分析到 docs/qc_analysis.md，包含：总体 pass rate、按场景分布、失败样本的主要原因分类（边界错位 / 生成图模糊 / 严重 artifact）。

**English Prompt：**
> Run `python scripts/05_run_qc.py --qc-config configs/qc/thresholds.yaml`. Check if pass rate matches E2's expected range. Analyze per_scene stats: if any scene has pass rate significantly below average (< 20%), inspect its debug_overlay images — determine if the issue is scene-specific (extreme viewpoint, heavy occlusion) or requires inference parameter tuning. Write a brief analysis to docs/qc_analysis.md covering: overall pass rate, per-scene distribution, main failure categories (edge misalignment / blurry generation / severe artifacts).

---

## Phase F：数据集组装与验证 / Dataset Assembly & Validation

---

### F1 — COCO JSON 构建器调试

| 项目 | 内容 |
|------|------|
| **目标** | 确认 coco_builder.py 的输出符合 COCO instance segmentation JSON schema |
| **输入** | 5–10 张 QC 通过的生成图 + 掩码 |
| **输出** | 一份小规模测试 COCO JSON |
| **边界约束** | JSON 必须可被 pycocotools 的 COCO() 正确加载；annotation 的 bbox/area/segmentation 数值正确；只有一个 category: rock |
| **依赖** | E1 |
| **验收标准** | ① `COCO(json_path)` 不报错 ② images 和 annotations 数量正确 ③ 用 pycocotools 的 showAnns() 可视化掩码，与原始掩码吻合 ④ bbox 包围掩码区域 |

**中文提示词：**
> 用 5–10 张 QC 通过的样本测试 blast_pile_diffusion/data/coco_builder.py 的 build_coco_dataset()。生成测试用 COCO JSON 后，用 pycocotools 加载验证：`from pycocotools.coco import COCO; coco = COCO(json_path)` 不报错。检查：① images 条目数 = 输入图片数 ② annotations 条目数 = 所有图片的实例总数 ③ 每个 annotation 的 bbox 正确包围掩码区域 ④ segmentation 字段（RLE 或 polygon）可被 coco.annToMask() 正确解码回二值掩码。用 matplotlib 画出 3 张图的掩码叠加可视化。运行 pytest tests/test_coco_builder.py。

**English Prompt：**
> Test blast_pile_diffusion/data/coco_builder.py build_coco_dataset() with 5–10 QC-passed samples. Load the COCO JSON via pycocotools: `COCO(json_path)` must not error. Verify: ① image count matches input ② annotation count matches total instances ③ each bbox correctly encloses its mask ④ segmentation (RLE or polygon) decodes back to correct binary mask via coco.annToMask(). Visualize 3 images with mask overlay using matplotlib. Run pytest tests/test_coco_builder.py.

---

### F2 — 最终数据集组装

| 项目 | 内容 |
|------|------|
| **目标** | 收集全部 QC 通过的样本，组装为完整的 COCO 训练集 |
| **输入** | data/generated/ 全部 QC 通过样本, data/preprocessed/ 对应掩码 |
| **输出** | data/final_dataset/train/images/ + annotations.json |
| **边界约束** | 只收集 qc.json 中 passed=true 的样本；图片与掩码必须一一对应，无遗漏无多余；annotation_id 全局唯一 |
| **依赖** | E3, F1 |
| **验收标准** | ① 图片数 = QC 通过数 ② annotations.json 可被 pycocotools 加载 ③ 无重复 image_id 或 annotation_id ④ 磁盘占用记录 |

**中文提示词：**
> 运行 `python scripts/06_build_coco_dataset.py`。运行后检查 data/final_dataset/train/ 下的 images/ 目录图片数是否等于 QC 通过数（qc_report.json 中的 passed 字段）。用 pycocotools 加载 annotations.json，验证 image_id 和 annotation_id 无重复。统计并记录：总图片数、总标注实例数、平均每张图的实例数、数据集磁盘占用。如果图片数显著少于预期（< 5000），回溯检查是 QC 阈值过严还是推理本身质量有问题。

**English Prompt：**
> Run `python scripts/06_build_coco_dataset.py`. Verify data/final_dataset/train/images/ count equals QC passed count from qc_report.json. Load annotations.json with pycocotools, check no duplicate image_id or annotation_id. Record: total images, total annotation instances, average instances per image, dataset disk size. If image count is significantly below expected (< 5000), trace back whether QC thresholds are too strict or inference quality is the issue.

---

### F3 — 数据集统计与质量验证

| 项目 | 内容 |
|------|------|
| **目标** | 生成数据集统计报告，验证分布合理性 |
| **输入** | data/final_dataset/train/annotations.json |
| **输出** | 统计报告（图表 + 数字） |
| **边界约束** | 统计不修改数据集本身；重点关注实例尺寸分布是否与目标级配匹配 |
| **依赖** | F2 |
| **验收标准** | ① 实例面积分布直方图 ② 每图实例数分布 ③ 小/中/大石块占比 ④ 与 Unity 原始分布对比无显著偏移 ⑤ 随机抽查 20 张图目视质量可接受 |

**中文提示词：**
> 编写 scripts/dataset_statistics.py 或在 notebook 中，对 data/final_dataset/train/annotations.json 做全面统计分析。输出以下图表和数字：① 实例面积分布直方图（像素面积，分 small/medium/large 三档标注）② 每张图的实例数分布（直方图）③ 按面积的 small/medium/large/boulder 四档占比（与技术路线 §3.3 的目标级配对比）④ 图像分辨率分布 ⑤ 随机抽查 20 张图，用 pycocotools 画出实例掩码叠加，保存到 docs/dataset_samples/ 目录下。如果尺寸分布与目标级配差异 > 15%，回溯 Unity 采样策略。

**English Prompt：**
> Write scripts/dataset_statistics.py to analyze data/final_dataset/train/annotations.json. Produce: ① instance area histogram (mark small/medium/large thresholds) ② instances-per-image histogram ③ small/medium/large/boulder percentage (compare with target gradation from tech roadmap §3.3) ④ image resolution distribution ⑤ random 20-image visual spot-check with mask overlays saved to docs/dataset_samples/. If size distribution deviates > 15% from target gradation, trace back to Unity sampling strategy.

---

## Phase G：集成优化与交付 / Integration & Delivery

---

### G1 — 全流程端到端集成测试

| 项目 | 内容 |
|------|------|
| **目标** | 用 10 个样本走通从 Unity 原始输出到 COCO 数据集的完整管线 |
| **输入** | data/unity_raw/ 下至少 10 帧 |
| **输出** | data/final_dataset/ 下的小规模完整数据集 |
| **边界约束** | 按 01→02→03→04→05→06 顺序串行执行全部脚本，中间不做手动干预 |
| **依赖** | F2 |
| **验收标准** | ① 6 个脚本依次执行无报错 ② 最终数据集含 > 0 张图片和标注 ③ 全流程耗时记录 ④ 任何一步失败都有清晰的错误信息 |

**中文提示词：**
> 做一次全流程端到端集成测试。用 data/unity_raw/ 下 10 个样本，按顺序运行全部 6 个脚本（01_preprocess → 02_prepare_lora → 跳过 03_train_lora 用已有权重 → 04_run_inference → 05_run_qc → 06_build_coco_dataset）。每步记录耗时和输出文件数。最终检查 data/final_dataset/train/ 下有有效的 images/ 和 annotations.json。如果任何一步失败，定位是脚本 bug 还是数据问题，修复后重新从该步开始执行。将全流程执行日志保存到 docs/integration_test_log.md。

**English Prompt：**
> Run a full end-to-end integration test with 10 samples from data/unity_raw/. Execute all 6 scripts in order (01_preprocess → 02_prepare_lora → skip 03_train_lora, use existing weights → 04_run_inference → 05_run_qc → 06_build_coco_dataset). Record time and output file count for each step. Verify data/final_dataset/train/ has valid images/ and annotations.json. If any step fails, diagnose whether it's a script bug or data issue, fix, and re-run from that step. Save execution log to docs/integration_test_log.md.

---

### G2 — 大规模批量生成与最终交付

| 项目 | 内容 |
|------|------|
| **目标** | 生成完整的 ~8000 张合成训练数据集并交付给阶段四（分割模型训练） |
| **输入** | 全部 Unity 场景 + 调优后的全套配置 |
| **输出** | data/final_dataset/train/ 完整数据集（~8000 张图 + COCO JSON） |
| **边界约束** | 批量生成可能需要数天 GPU 时间；必须有断点续传；最终数据集必须与阶段四的数据加载器兼容 |
| **依赖** | G1 |
| **验收标准** | ① 图片数 ≥ 5000（考虑 QC 淘汰后）② COCO JSON 完整可加载 ③ 数据集统计报告无异常 ④ 用阶段四的分割模型代码能成功加载数据集并开始训练（跑 1 个 epoch 不报错）|

**中文提示词：**
> 执行最终的大规模批量生成。在全部 Unity 场景上运行 scripts/04_run_inference.py（seeds_per_sample=4），预计总生成量 = 场景数 × 相机数 × 4。这可能需要数天 GPU 时间，确保断点续传正常工作。生成完毕后运行 QC 和数据集组装。最终验证：① 数据集图片数 ≥ 5000 ② 运行 F3 的统计脚本确认分布合理 ③ 用阶段四的分割模型训练框架（Detectron2 或 MMDetection）尝试加载此数据集并跑 1 个 epoch，确认数据接口兼容。将最终数据集的统计摘要写入 docs/dataset_delivery_report.md。

**English Prompt：**
> Execute final large-scale batch generation. Run scripts/04_run_inference.py on all Unity scenes (seeds_per_sample=4). Expected total = scenes × cameras × 4. This may take days of GPU time — ensure resume capability works. After generation, run QC and dataset assembly. Final verification: ① dataset has ≥ 5000 images ② F3 statistics script confirms reasonable distribution ③ stage-4 segmentation framework (Detectron2 or MMDetection) can load the dataset and train for 1 epoch without errors. Write dataset summary to docs/dataset_delivery_report.md.

---

## 附录：任务总览表

| ID | 阶段 | 任务名称 | 预计耗时 | 依赖 |
|----|------|---------|---------|------|
| A1 | 基础设施 | 环境搭建与依赖安装 | 2h | — |
| A2 | 基础设施 | 预训练模型下载验证 | 1–3h | A1 |
| A3 | 基础设施 | 最小化冒烟测试 | 1h | A2 |
| B1 | 预处理 | Unity 数据解析器适配 | 3–6h | A1 |
| B2 | 预处理 | 深度图预处理验证 | 2–4h | B1 |
| B3 | 预处理 | 掩码→Canny 验证 ⚠️关键 | 2–4h | B1 |
| B4 | 预处理 | 法线图转换验证 | 1–2h | B1 |
| B5 | 预处理 | 预处理全流程端到端 | 1–2h | B2,B3,B4 |
| C1 | LoRA | 真实图片采集整理 | 1–3天 | — |
| C2 | LoRA | Caption 生成与数据构建 | 2–4h | C1 |
| C3 | LoRA | LoRA 训练执行 | 4–8h | A2,C2 |
| C4 | LoRA | LoRA 质量验证迭代 | 2–4h | C3 |
| D1 | 推理 | 单样本推理调通 | 2–4h | B5,C4 |
| D2 | 推理 | 超参数搜索 | 4–8h | D1,E1 |
| D3 | 推理 | 批量推理执行 | 1–3天 | D2 |
| E1 | QC | QC 模块调试 | 2–3h | D1 |
| E2 | QC | QC 阈值标定 | 2–4h | E1,D3(部分) |
| E3 | QC | 批量 QC 执行 | 1–2h | E2,D3 |
| F1 | 数据集 | COCO 构建器调试 | 2–3h | E1 |
| F2 | 数据集 | 最终数据集组装 | 1–2h | E3,F1 |
| F3 | 数据集 | 数据集统计验证 | 2–3h | F2 |
| G1 | 集成 | 端到端集成测试 | 4–6h | F2 |
| G2 | 集成 | 大规模生成与交付 | 3–7天 | G1 |

**总预计工时**：约 4–6 周（单人，含等待 GPU 时间）

**并行策略**：Phase B（预处理）和 Phase C（LoRA）可并行推进，两者汇合后进入 Phase D。
