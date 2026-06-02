#!/usr/bin/env bash
# Step 3: 启动 LoRA 训练
#
# 前置条件：
#   1. 已安装 sd-scripts: git clone https://github.com/kohya-ss/sd-scripts
#   2. 已运行 scripts/02_prepare_lora_data.py 准备训练数据
#   3. 已下载 SDXL base model
#
# 用法：
#   bash scripts/03_train_lora.sh --dry-run
#   SD_SCRIPTS_DIR=/path/to/sd-scripts bash scripts/03_train_lora.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

python3 -m blast_pile_diffusion.lora.train_launcher "$@"
