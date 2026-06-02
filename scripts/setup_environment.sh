#!/usr/bin/env bash
# scripts/setup_environment.sh —— 一键完整安装脚本
#
# 这个脚本完成以下事情：
#   1. 安装 Python 依赖（推理 + 训练）
#   2. git clone sd-scripts（kohya_ss，LoRA 训练工具）
#   3. 下载全部预训练权重到 HuggingFace 缓存（~/.cache/huggingface/hub/）
#      - 必需：SDXL Base / VAE fp16-fix / ControlNet-Depth / ControlNet-Canny  (~12 GB)
#      - 可选：ControlNet-Normal / BLIP-2 / IP-Adapter（按 --with-optional 开关）
#   4. 跑 verify_models.py --local-only 验证下载完整
#
# 用法：
#   bash scripts/setup_environment.sh                          # 仅必需
#   bash scripts/setup_environment.sh --with-optional          # 必需 + 可选全装
#   bash scripts/setup_environment.sh --with-normal            # 必需 + ControlNet-Normal
#   bash scripts/setup_environment.sh --with-blip2             # 必需 + BLIP-2 (~15 GB)
#   bash scripts/setup_environment.sh --with-ip-adapter        # 必需 + IP-Adapter
#   bash scripts/setup_environment.sh --skip-deps              # 跳过 pip install
#   bash scripts/setup_environment.sh --skip-sd-scripts        # 跳过 sd-scripts 克隆
#   bash scripts/setup_environment.sh --skip-models            # 只装 deps 和 sd-scripts
#
# 国内镜像加速：
#   HF_ENDPOINT=https://hf-mirror.com bash scripts/setup_environment.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# ---- 配色 ----
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[ OK ]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[FAIL]${NC}  $*"; }
step()  { echo -e "\n${BLUE}════════ $* ════════${NC}"; }

# ---- 参数解析 ----
WITH_NORMAL=false
WITH_BLIP2=false
WITH_IP_ADAPTER=false
SKIP_DEPS=false
SKIP_SD_SCRIPTS=false
SKIP_MODELS=false

for arg in "$@"; do
    case "$arg" in
        --with-optional)
            WITH_NORMAL=true
            WITH_BLIP2=true
            WITH_IP_ADAPTER=true
            ;;
        --with-normal)      WITH_NORMAL=true ;;
        --with-blip2)       WITH_BLIP2=true ;;
        --with-ip-adapter)  WITH_IP_ADAPTER=true ;;
        --skip-deps)        SKIP_DEPS=true ;;
        --skip-sd-scripts)  SKIP_SD_SCRIPTS=true ;;
        --skip-models)      SKIP_MODELS=true ;;
        -h|--help)
            grep '^#' "$0" | head -25
            exit 0
            ;;
        *)
            err "未知参数: $arg"
            exit 1
            ;;
    esac
done

# ---- 信息汇总 ----
step "Setup configuration"
info "Project root:    $PROJECT_ROOT"
info "HF_ENDPOINT:     ${HF_ENDPOINT:-https://huggingface.co (default)}"
info "Install deps:    $([ "$SKIP_DEPS" = true ] && echo no || echo yes)"
info "Clone sd-scripts: $([ "$SKIP_SD_SCRIPTS" = true ] && echo no || echo yes)"
info "Download models:  $([ "$SKIP_MODELS" = true ] && echo no || echo yes)"
info "Optional models:"
info "  ControlNet-Normal: $([ "$WITH_NORMAL" = true ] && echo yes || echo no)"
info "  BLIP-2 (~15 GB):   $([ "$WITH_BLIP2" = true ] && echo yes || echo no)"
info "  IP-Adapter:        $([ "$WITH_IP_ADAPTER" = true ] && echo yes || echo no)"
echo ""

# ---- 磁盘空间检查 ----
REQUIRED_GB=12
[ "$WITH_NORMAL" = true ] && REQUIRED_GB=$((REQUIRED_GB + 3))
[ "$WITH_BLIP2" = true ] && REQUIRED_GB=$((REQUIRED_GB + 15))
[ "$WITH_IP_ADAPTER" = true ] && REQUIRED_GB=$((REQUIRED_GB + 1))
AVAILABLE_GB=$(df -g "$HOME" 2>/dev/null | awk 'NR==2 {print $4}' || echo 999)
info "Predicted download size:  ~${REQUIRED_GB} GB"
info "Available space in \$HOME: ~${AVAILABLE_GB} GB"
if [ "${AVAILABLE_GB}" -lt "${REQUIRED_GB}" ]; then
    warn "磁盘空间可能不足；继续之前请确认"
    sleep 3
fi

# =============================================================================
# Step 1: Python 依赖
# =============================================================================
if [ "$SKIP_DEPS" = false ]; then
    step "Step 1/4  Install Python dependencies"
    info "pip install -e ."
    pip install -e . 2>&1 | tail -5
    ok "Base dependencies installed"

    if [ -f "requirements_train.txt" ]; then
        info "pip install -r requirements_train.txt"
        pip install -r requirements_train.txt 2>&1 | tail -5
        ok "Training dependencies installed"
    fi

    info "Verifying torch + CUDA..."
    python3 -c "
import torch
print(f'  torch version: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA device count: {torch.cuda.device_count()}')
    print(f'  Device 0: {torch.cuda.get_device_name(0)}')
"
else
    step "Step 1/4  Skipped (Python dependencies)"
fi

# =============================================================================
# Step 2: sd-scripts (kohya_ss)
# =============================================================================
if [ "$SKIP_SD_SCRIPTS" = false ]; then
    step "Step 2/4  Clone sd-scripts (kohya_ss)"
    if [ -d "$PROJECT_ROOT/sd-scripts/.git" ]; then
        ok "sd-scripts already cloned at $PROJECT_ROOT/sd-scripts (skipping)"
    else
        SDSCRIPTS_URL="https://github.com/kohya-ss/sd-scripts.git"
        info "Cloning $SDSCRIPTS_URL → $PROJECT_ROOT/sd-scripts"
        git clone "$SDSCRIPTS_URL" "$PROJECT_ROOT/sd-scripts"
        ok "sd-scripts cloned"
        info "Installing sd-scripts requirements..."
        (cd "$PROJECT_ROOT/sd-scripts" && pip install -r requirements.txt 2>&1 | tail -5)
        ok "sd-scripts requirements installed"
    fi
else
    step "Step 2/4  Skipped (sd-scripts)"
fi

# =============================================================================
# Step 3: 下载预训练权重
# =============================================================================
if [ "$SKIP_MODELS" = false ]; then
    step "Step 3/4  Download pretrained weights"

    # 用 huggingface-cli 而不是 diffusers，更稳定 + 进度条好看
    if ! command -v huggingface-cli >/dev/null 2>&1; then
        info "Installing huggingface_hub CLI..."
        pip install -q "huggingface_hub[cli]"
    fi

    download_model() {
        local repo_id="$1"
        local label="$2"
        info "Downloading [$label]: $repo_id"
        # --resume-download 让中断后能续传；--exclude 跳过冗余 fp32 权重节省空间
        huggingface-cli download "$repo_id" \
            --exclude "*.fp32.safetensors" "*.bin" "*.msgpack" "*.onnx" "*.ot" \
            --quiet || {
                err "下载失败: $repo_id"
                return 1
            }
        ok "Downloaded [$label]"
    }

    # ---- 必需 4 个 ----
    info "Required models (4 models, ~12 GB)..."
    download_model "stabilityai/stable-diffusion-xl-base-1.0" "SDXL-Base"
    download_model "madebyollin/sdxl-vae-fp16-fix" "SDXL-VAE-fp16-fix"
    download_model "diffusers/controlnet-depth-sdxl-1.0" "ControlNet-Depth-SDXL"
    download_model "diffusers/controlnet-canny-sdxl-1.0" "ControlNet-Canny-SDXL"

    # ---- 可选 ----
    if [ "$WITH_NORMAL" = true ]; then
        info "Optional: ControlNet-Normal (~2.5 GB)..."
        # 社区里 SDXL normal 几个候选源，按可用性顺序尝试
        download_model "diffusers/controlnet-normal-sdxl-1.0" "ControlNet-Normal-SDXL" \
            || warn "ControlNet-Normal-SDXL 不可下载，已跳过（可手动找替代）"
    fi
    if [ "$WITH_BLIP2" = true ]; then
        info "Optional: BLIP-2 (~15 GB)..."
        download_model "Salesforce/blip2-opt-2.7b" "BLIP-2"
    fi
    if [ "$WITH_IP_ADAPTER" = true ]; then
        info "Optional: IP-Adapter for SDXL (~700 MB)..."
        download_model "h94/IP-Adapter" "IP-Adapter"
    fi
else
    step "Step 3/4  Skipped (models)"
fi

# =============================================================================
# Step 4: 验证
# =============================================================================
step "Step 4/4  Verify local model cache"
if [ "$SKIP_MODELS" = false ]; then
    python3 scripts/verify_models.py --local-only || {
        err "verify_models.py 验证失败 — 请检查上面输出"
        exit 1
    }
    ok "All required models verified locally"
fi

# ---- pytest sanity check ----
info "Running pytest sanity check..."
python3 -m pytest tests/ -q --no-header 2>&1 | tail -3 || warn "Some tests failed — environment may be incomplete"

# =============================================================================
# 完成
# =============================================================================
step "Setup complete!"
echo ""
ok "你现在可以："
echo ""
echo "  1. 跑冒烟测试验证 GPU pipeline:"
echo "     python scripts/smoke_test.py"
echo ""
echo "  2. 把 Unity 导出放到 data/unity_raw/ 然后跑预处理:"
echo "     python scripts/01_preprocess_unity.py"
echo ""
echo "  3. 真实矿场图准备好后训练 LoRA:"
echo "     python scripts/02_prepare_lora_data.py"
echo "     bash scripts/03_train_lora.sh"
echo ""
echo "  4. 推理 + QC + 组装数据集:"
echo "     python scripts/04_run_inference.py"
echo "     python scripts/05_run_qc.py"
echo "     python scripts/06_build_coco_dataset.py"
echo ""
ok "所有模型缓存路径：${HF_HOME:-~/.cache/huggingface/hub/}"
ok "sd-scripts 路径：$PROJECT_ROOT/sd-scripts"
