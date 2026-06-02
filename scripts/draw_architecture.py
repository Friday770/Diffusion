#!/usr/bin/env python3
"""生成 sim2real 翻译管线的论文级架构图。

设计原则：
  1. 严格 4 列对齐 — 输入 / 预处理 / 条件 / ControlNet 在同一列上下贯通
  2. 所有箭头只走横/竖（曼哈顿路径），无对角线，无交叉
  3. RGB 在最左侧通道独立向下，Mask 在最右侧独立通道向下（不与中间列冲突）
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch
from matplotlib.path import Path as MplPath

# ---- 全局样式 ----
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "font.size": 9.5,
    "mathtext.fontset": "stix",
    "mathtext.default": "regular",
    "axes.linewidth": 0.6,
})

C = {
    "input_fill": "#E8F1F8",  "input_edge": "#2C5F8C",
    "geom_fill":  "#FFF1E0",  "geom_edge":  "#C77B30",
    "cond_fill":  "#FAFAFA",  "cond_edge":  "#666666",
    "cn_fill":    "#EFE6FA",  "cn_edge":    "#6E3DBE",
    "sdxl_fill":  "#FFE0E0",  "sdxl_edge":  "#B53030",
    "lora_fill":  "#FFD6E0",  "lora_edge":  "#A02060",
    "prompt_fill":"#F7F7F7",  "prompt_edge":"#888888",
    "out_fill":   "#DEF4DE",  "out_edge":   "#2E8C42",
    "qc_fill":    "#FFF6BC",  "qc_edge":    "#B58B00",
    "data_fill":  "#EAEAEA",  "data_edge":  "#555555",
}

ARROW_SOLID = dict(arrowstyle="-|>", mutation_scale=14, lw=1.3, color="#222")
ARROW_DASH  = dict(arrowstyle="-|>", mutation_scale=13, lw=1.1,
                   color="#888", linestyle=(0, (4, 3)))
ARROW_INJ   = dict(arrowstyle="-|>", mutation_scale=13, lw=1.1,
                   color="#6E3DBE")

fig, ax = plt.subplots(figsize=(15, 13), dpi=300)
ax.set_xlim(0, 150)
ax.set_ylim(-20, 130)
ax.set_aspect("equal")
ax.axis("off")


def box(x, y, w, h, fill, edge, text, fs=9.5, weight="normal",
        italic=False, lw=1.0, radius=0.7):
    bb = FancyBboxPatch((x, y), w, h,
                        boxstyle=f"round,pad=0.05,rounding_size={radius}",
                        linewidth=lw, facecolor=fill, edgecolor=edge)
    ax.add_patch(bb)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, weight=weight,
            style="italic" if italic else "normal", color="#1a1a1a")


def varrow(x, y_from, y_to, style=None):
    s = style if style is not None else ARROW_SOLID
    ax.add_patch(FancyArrowPatch((x, y_from), (x, y_to), **s))


def harrow(y, x_from, x_to, style=None):
    s = style if style is not None else ARROW_SOLID
    ax.add_patch(FancyArrowPatch((x_from, y), (x_to, y), **s))


def manhattan(points, style=None, arrow=True):
    """曼哈顿折线（只走横/竖），最后一段加箭头头。"""
    s = style if style is not None else ARROW_SOLID
    verts = list(points)
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(verts) - 1)
    pp = PathPatch(MplPath(verts, codes), fill=False,
                   lw=s["lw"], edgecolor=s["color"],
                   linestyle=s.get("linestyle", "-"))
    ax.add_patch(pp)
    if arrow:
        p_last, p_end = verts[-2], verts[-1]
        ax.add_patch(FancyArrowPatch(p_last, p_end,
                                     arrowstyle="-|>",
                                     mutation_scale=s["mutation_scale"],
                                     lw=0, color=s["color"]))


def label(x, y, text, fs=8, color="#444", italic=True, ha="center"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fs,
            style="italic" if italic else "normal", color=color)


def section(x, y, text, color):
    ax.text(x, y, text, ha="left", va="bottom",
            fontsize=11, weight="bold", color=color)


def panel(x, y, w, h, color, alpha=0.07):
    rect = mpatches.Rectangle((x, y), w, h, facecolor=color, alpha=alpha,
                              edgecolor=color, linewidth=0.5,
                              linestyle=(0, (1, 2)))
    ax.add_patch(rect)


# ============================================================================
# 列定义（5 条独立通道，绝不交叉）
#   关键：Mask 输入框直接放在 Canny 预处理模块正上方，避免水平走廊横跨其他列
# ============================================================================
COL_RGB_RAIL = 8       # 最左：RGB → img2img init 专用通道
COL_DEPTH    = 35
COL_CANNY    = 65      # 中央主轴（Mask 输入也在此列）
COL_NORMAL   = 95
COL_LABEL    = 130     # 最右：label pass-through 专用通道

# 行定义
Y_INPUT_T  = 117       # Stage A 顶
Y_INPUT_B  = 110       # Stage A 底
Y_PREP_T   = 100       # Stage B 顶
Y_PREP_B   = 90        # Stage B 底
Y_COND_T   = 83        # Stage C 顶
Y_COND_B   = 78        # Stage C 底
Y_CN_T     = 70        # Stage D 顶
Y_CN_B     = 60        # Stage D 底
Y_SDXL_T   = 50        # Stage E 顶
Y_SDXL_B   = 39        # Stage E 底
Y_GEN_T    = 30        # 生成图 顶
Y_GEN_B    = 23        # 生成图 底
Y_QC_T     = 17        # QC 顶
Y_QC_B     = 10        # QC 底
Y_DATA_T   = 4         # 数据集 顶
Y_DATA_B   = -2        # 数据集 底

# ============================================================================
# 标题
# ============================================================================
ax.text(70, 127,
        "Multi-Conditional ControlNet Sim2Real Translation Pipeline",
        ha="center", va="center", fontsize=15, weight="bold")
ax.text(70, 123.5,
        "for Open-Pit Mine Blast-Pile Instance Segmentation Data Synthesis",
        ha="center", va="center", fontsize=10.5, style="italic", color="#555")

# ============================================================================
# Stage A : 输入
# ============================================================================
panel(2, 109, 140, 11, "#2C5F8C", alpha=0.04)
section(3.5, 118.5, "Stage A   Unity Perception Synthetic Inputs", C["input_edge"])

inputs = [
    (COL_RGB_RAIL,  "Unity RGB",      r"$\mathbf{I}_u \in \mathbb{R}^{H\times W\times 3}$"),
    (COL_DEPTH,     "Depth Map",      r"$\mathbf{D} \in \mathbb{R}^{H\times W}$"),
    (COL_CANNY,     "Instance Mask",  r"$\mathbf{M} \in \mathbb{Z}^{H\times W}$"),
    (COL_NORMAL,    "Surface Normal", r"$\mathbf{N} \in \mathbb{R}^{H\times W\times 3}$"),
]
for cx, name, math in inputs:
    box(cx - 8, Y_INPUT_B, 16, 7, C["input_fill"], C["input_edge"],
        f"{name}\n{math}", fs=9)

# ============================================================================
# Stage B : 预处理
# ============================================================================
panel(2, 88, 140, 14, "#C77B30", alpha=0.06)
section(3.5, 100.5, "Stage B   Geometry-Aware Condition Preprocessing", C["geom_edge"])

# Depth Norm
box(COL_DEPTH - 13, Y_PREP_B, 26, 10, C["geom_fill"], C["geom_edge"],
    "Depth Normalization\n+ Hole Filling\n"
    r"$\phi_D : \mathbf{D} \rightarrow \widehat{\mathbf{D}}$",
    fs=9, weight="bold")

# Mask → Canny  ★
box(COL_CANNY - 16, Y_PREP_B, 32, 10, C["geom_fill"], C["geom_edge"],
    r"Mask $\rightarrow$ Canny  $\star$" "\n"
    r"$\mathbf{E} = \bigcup_{i} \mathrm{Canny}(\,\mathbf{1}[\mathbf{M}=i]\,)$" "\n"
    r"edges from labels, not RGB",
    fs=9, weight="bold", lw=1.5)

# Normal Re-encoding
box(COL_NORMAL - 13, Y_PREP_B, 26, 10, C["geom_fill"], C["geom_edge"],
    "Normal Re-encoding\n"
    r"(camera $\rightarrow$ RGB-coded)" "\n"
    r"$\phi_N : \mathbf{N} \rightarrow \widehat{\mathbf{N}}$",
    fs=9, weight="bold")

# 输入 → 预处理（全部纯垂直，无任何水平走廊）
varrow(COL_DEPTH,  Y_INPUT_B, Y_PREP_T)
varrow(COL_CANNY,  Y_INPUT_B, Y_PREP_T)    # Mask → Canny 模块
varrow(COL_NORMAL, Y_INPUT_B, Y_PREP_T)

# Mask → label pass-through 轨道（从 Mask 框右侧引出 → 右行 → 沿 COL_LABEL 下到底部）
# 这一段在最上方独立走，不与任何其他箭头共享 y 区间
Y_LABEL_OUT = Y_INPUT_B + 3.5   # Mask 框右侧出口位置（框中线）
manhattan([
    (COL_CANNY + 8, Y_LABEL_OUT),
    (COL_LABEL, Y_LABEL_OUT),
    (COL_LABEL, Y_DATA_B + 3),
    (COL_CANNY + 20, Y_DATA_B + 3),
], style=ARROW_DASH)
label(COL_LABEL + 1, (Y_LABEL_OUT + Y_DATA_B) / 2 + 10,
      r"label  $\mathbf{M}$" "\n" "pass-through",
      fs=8.5, color="#555", ha="left")

# ============================================================================
# Stage C : 条件张量
# ============================================================================
box(COL_DEPTH  - 13, Y_COND_B, 26, 5, C["cond_fill"], C["cond_edge"],
    r"$\widehat{\mathbf{D}}$   depth condition", fs=9, italic=True)
box(COL_CANNY  - 16, Y_COND_B, 32, 5, C["cond_fill"], C["cond_edge"],
    r"$\mathbf{E}$   canny condition", fs=9, italic=True)
box(COL_NORMAL - 13, Y_COND_B, 26, 5, C["cond_fill"], C["cond_edge"],
    r"$\widehat{\mathbf{N}}$   normal condition", fs=9, italic=True)

varrow(COL_DEPTH,  Y_PREP_B, Y_COND_T)
varrow(COL_CANNY,  Y_PREP_B, Y_COND_T)
varrow(COL_NORMAL, Y_PREP_B, Y_COND_T)

# ============================================================================
# Stage D : 三路 ControlNet
# ============================================================================
panel(2, 58, 140, 16, "#6E3DBE", alpha=0.05)
section(3.5, 72.5, "Stage D   Multi-Path ControlNet Conditioning", C["cn_edge"])

box(COL_DEPTH  - 13, Y_CN_B, 26, 10, C["cn_fill"], C["cn_edge"],
    "ControlNet\nDepth (SDXL)\n" r"$\lambda_D = 0.8$", fs=9, weight="bold")
box(COL_CANNY  - 16, Y_CN_B, 32, 10, C["cn_fill"], C["cn_edge"],
    "ControlNet\nCanny (SDXL)\n" r"$\lambda_E = 0.6$",
    fs=9, weight="bold", lw=1.5)
box(COL_NORMAL - 13, Y_CN_B, 26, 10, C["cn_fill"], C["cn_edge"],
    "ControlNet\nNormal (SDXL)\n" r"$\lambda_N = 0.4$", fs=9, weight="bold")

varrow(COL_DEPTH,  Y_COND_B, Y_CN_T)
varrow(COL_CANNY,  Y_COND_B, Y_CN_T)
varrow(COL_NORMAL, Y_COND_B, Y_CN_T)

# ============================================================================
# Stage E : SDXL backbone
# ============================================================================
panel(2, 37, 140, 16, "#B53030", alpha=0.04)
section(3.5, 51.5, "Stage E   SDXL Diffusion Backbone with Style LoRA", C["sdxl_edge"])

# 主块
sx, sy, sw, sh = COL_CANNY - 24, Y_SDXL_B, 48, 11
box(sx, sy, sw, sh, C["sdxl_fill"], C["sdxl_edge"],
    "SDXL Base U-Net  +  VAE\n"
    r"$\epsilon_{\theta}(\,\mathbf{z}_t,\; c_{\mathrm{text}},\;\{\Phi_k(c_k)\}_{k=1}^{3}\,)$" "\n"
    r"img2img,  $s = 0.45$,  $T = 30$",
    fs=10, weight="bold", lw=1.4)

# LoRA（左侧附挂）
lora_w, lora_h = 26, 5
lx, ly = sx - lora_w - 3, sy + (sh - lora_h) / 2
box(lx, ly, lora_w, lora_h, C["lora_fill"], C["lora_edge"],
    r"Style LoRA  $\langle$mine_blast_pile$\rangle$" "\n"
    r"rank $=32$,  $\alpha=32$",
    fs=8.5, weight="bold")
harrow(ly + lora_h / 2, lx + lora_w, sx,
       style=dict(arrowstyle="-|>", mutation_scale=11, lw=1.0,
                  color=C["lora_edge"]))

# Text prompt（右侧附挂）
tx, ty_ = sx + sw + 3, sy + (sh - lora_h) / 2
box(tx, ty_, lora_w, lora_h, C["prompt_fill"], C["prompt_edge"],
    r"Text Prompt  $c_{\mathrm{text}}$" "\n"
    r"$\langle$mine_blast_pile$\rangle$, ...",
    fs=8.5, italic=True)
harrow(ty_ + lora_h / 2, tx, sx + sw,
       style=dict(arrowstyle="-|>", mutation_scale=11, lw=1.0, color="#888"))

# 3 路 ControlNet → SDXL（注入）
# 中路直接下
varrow(COL_CANNY, Y_CN_B, sy + sh, style=ARROW_INJ)
# 左路：CN-D 底部 → 水平走廊 y=55 → SDXL 顶部左侧
Y_INJ_CORRIDOR = 55
manhattan([
    (COL_DEPTH, Y_CN_B),
    (COL_DEPTH, Y_INJ_CORRIDOR),
    (sx + 6, Y_INJ_CORRIDOR),
    (sx + 6, sy + sh),
], style=ARROW_INJ)
# 右路：CN-N 底部 → 水平走廊 y=55 → SDXL 顶部右侧
manhattan([
    (COL_NORMAL, Y_CN_B),
    (COL_NORMAL, Y_INJ_CORRIDOR),
    (sx + sw - 6, Y_INJ_CORRIDOR),
    (sx + sw - 6, sy + sh),
], style=ARROW_INJ)

label(COL_CANNY, Y_INJ_CORRIDOR + 1.5,
      "zero-conv injection  " r"$\{\Phi_k(c_k)\}$",
      fs=8.5, color="#6E3DBE")

# RGB → SDXL  (img2img init)：最左通道独立竖线，曼哈顿到 SDXL 左下角
manhattan([
    (COL_RGB_RAIL, Y_INPUT_B),
    (COL_RGB_RAIL, sy + 3),
    (sx, sy + 3),
], style=ARROW_DASH)
label(COL_RGB_RAIL, (Y_INPUT_B + sy) / 2 + 5,
      r"img2img init  $\mathbf{I}_u$", fs=8.5, color="#555")

# ============================================================================
# Stage F : 生成图 + QC + 数据集
# ============================================================================
panel(2, -3, 140, 36, "#2E8C42", alpha=0.04)
section(3.5, 32, "Stage F   Generated Image  +  Edge-Alignment QC  +  Dataset Assembly",
        C["out_edge"])

# 生成图
box(COL_CANNY - 18, Y_GEN_B, 36, 7, C["out_fill"], C["out_edge"],
    r"Generated RGB  $\widehat{\mathbf{I}}$   (photorealistic, label-aligned)",
    fs=10, weight="bold")
varrow(COL_CANNY, sy, Y_GEN_T)
label(COL_CANNY + 12, (sy + Y_GEN_T) / 2,
      r"denoise  $T$  steps", fs=8, color="#444")

# QC
box(COL_CANNY - 24, Y_QC_B, 48, 7, C["qc_fill"], C["qc_edge"],
    "Automatic QC :  Edge-Alignment\n"
    r"$\bar{\delta} = \mathbb{E}_p[\,\mathrm{DT}(\mathbf{E})(p)\;|\;\mathrm{Canny}(\widehat{\mathbf{I}})(p)>0\,] < 4\,\mathrm{px}$",
    fs=9, weight="bold")
varrow(COL_CANNY, Y_GEN_B, Y_QC_T)

# 数据集
box(COL_CANNY - 20, Y_DATA_B, 40, 6, C["data_fill"], C["data_edge"],
    "COCO Instance Seg. Dataset   "
    r"$\{\widehat{\mathbf{I}},\,\mathbf{M}\}_{\mathrm{accepted}}$",
    fs=9.5, weight="bold")
varrow(COL_CANNY, Y_QC_B, Y_DATA_T)

# 淘汰分支（左侧虚线）
manhattan([
    (COL_CANNY - 24, (Y_QC_T + Y_QC_B) / 2),
    (COL_CANNY - 32, (Y_QC_T + Y_QC_B) / 2),
    (COL_CANNY - 32, Y_DATA_B - 4),
], style=ARROW_DASH)
label(COL_CANNY - 32, Y_DATA_B - 5.5, "reject", fs=8.5, color="#888")

# （Mask 直传通道已在 Stage A 输入下方统一定义，此处无需重复）

# ============================================================================
# 图例 + 关键贡献注脚
# ============================================================================
lgx, lgy = 3, -10
ax.add_patch(FancyArrowPatch((lgx, lgy), (lgx + 5, lgy), **ARROW_SOLID))
ax.text(lgx + 5.5, lgy, "  data flow", fontsize=9, va="center", color="#333")

ax.add_patch(FancyArrowPatch((lgx + 28, lgy), (lgx + 33, lgy), **ARROW_DASH))
ax.text(lgx + 33.5, lgy, "  auxiliary  (init / pass-through / reject)",
        fontsize=9, va="center", color="#333")

ax.add_patch(FancyArrowPatch((lgx + 90, lgy), (lgx + 95, lgy), **ARROW_INJ))
ax.text(lgx + 95.5, lgy, "  ControlNet zero-conv injection",
        fontsize=9, va="center", color="#333")

ax.text(70, -15,
        r"$\star$  Key contribution :  The Canny condition $\mathbf{E}$ is derived directly from instance masks $\mathbf{M}$, "
        r"not from the RGB image —",
        ha="center", fontsize=9.5, color="#B53030", weight="bold")
ax.text(70, -17.5,
        "this guarantees pixel-perfect alignment between generated edges and ground-truth segmentation boundaries.",
        ha="center", fontsize=9.5, color="#B53030")

# ============================================================================
# 保存
# ============================================================================
out_dir = Path("docs/figures")
out_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(out_dir / "architecture.pdf", bbox_inches="tight", pad_inches=0.25)
fig.savefig(out_dir / "architecture.png", bbox_inches="tight", pad_inches=0.25,
            dpi=300)
print(f"saved: {out_dir / 'architecture.pdf'}")
print(f"saved: {out_dir / 'architecture.png'}")
