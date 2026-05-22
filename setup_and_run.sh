#!/bin/bash
# ============================================================
# MB-Defense 一键环境配置 + 验证脚本
# AutoDL 上 clone 后直接执行: bash setup_and_run.sh
# ============================================================

set -e  # 遇到错误立即停止

echo "============================================================"
echo "MB-Defense: 一键环境配置"
echo "============================================================"

# 获取脚本所在目录（即项目根目录）
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"
echo "Project directory: $PROJECT_DIR"

# ============================================================
# Step 1: 安装 Python 依赖
# ============================================================
echo ""
echo "[Step 1/5] Installing Python dependencies..."
pip install -q transformers>=4.36.0 accelerate>=0.25.0 sentencepiece protobuf
pip install -q sentence-transformers>=2.2.0
pip install -q numpy>=1.24.0 pandas>=2.0.0 scikit-learn>=1.3.0
pip install -q tqdm pyyaml
pip install -q jailbreakbench>=0.1.0
echo "  ✓ Dependencies installed"

# ============================================================
# Step 2: 下载/定位模型
# ============================================================
echo ""
echo "[Step 2/5] Locating Llama-2-7B-Chat model..."

MODEL_PATH=""

# 检查常见的 AutoDL 公共模型路径
POSSIBLE_PATHS=(
    "/root/share/new_models/meta-llama/Llama-2-7b-chat-hf"
    "/root/share/model_repos/meta-llama/Llama-2-7b-chat-hf"
    "/root/autodl-tmp/models/llama2-7b-chat"
    "/root/models/llama2-7b-chat"
    "/root/autodl-tmp/Llama-2-7b-chat-hf"
)

for path in "${POSSIBLE_PATHS[@]}"; do
    if [ -d "$path" ] && [ -f "$path/config.json" ]; then
        MODEL_PATH="$path"
        echo "  ✓ Found model at: $MODEL_PATH"
        break
    fi
done

# 如果没找到，下载模型
if [ -z "$MODEL_PATH" ]; then
    echo "  Model not found in common paths. Downloading..."
    MODEL_PATH="/root/autodl-tmp/llama2-7b-chat"
    
    export HF_ENDPOINT=https://hf-mirror.com
    
    if command -v huggingface-cli &> /dev/null; then
        huggingface-cli download meta-llama/Llama-2-7b-chat-hf --local-dir "$MODEL_PATH"
    else
        pip install -q huggingface_hub
        huggingface-cli download meta-llama/Llama-2-7b-chat-hf --local-dir "$MODEL_PATH"
    fi
    
    echo "  ✓ Model downloaded to: $MODEL_PATH"
fi

# 保存模型路径到配置文件，方便后续使用
echo "$MODEL_PATH" > "$PROJECT_DIR/.model_path"
echo "  Model path saved to .model_path"

# ============================================================
# Step 3: 准备数据
# ============================================================
echo ""
echo "[Step 3/5] Preparing datasets..."
python "$PROJECT_DIR/scripts/prepare_data.py"
echo "  ✓ Data prepared"

# ============================================================
# Step 4: 验证 pipeline
# ============================================================
echo ""
echo "[Step 4/5] Running quick pipeline test..."
python "$PROJECT_DIR/scripts/quick_test.py" --model_path "$MODEL_PATH" --skip_mb
echo "  ✓ Basic pipeline test passed"

# ============================================================
# Step 5: 打印后续操作指南
# ============================================================
echo ""
echo "============================================================"
echo "✓ SETUP COMPLETE!"
echo "============================================================"
echo ""
echo "Model path: $MODEL_PATH"
echo ""
echo "接下来可以执行:"
echo ""
echo "  # 完整 pipeline 测试 (含 MB-Defense，约5分钟)"
echo "  python scripts/quick_test.py --model_path $MODEL_PATH"
echo ""
echo "  # 跑原版 Backtranslation baseline (GCG, 50样本)"
echo "  python scripts/run_backtranslation.py --model_path $MODEL_PATH --attack gcg --run_benign"
echo ""
echo "  # 跑 MB-Defense (GCG, 50样本)"
echo "  python scripts/run_mb_defense.py --model_path $MODEL_PATH --attack gcg --run_benign"
echo ""
echo "  # 跑全部实验 + 消融"
echo "  bash scripts/run_all.sh"
echo ""
echo "============================================================"
