#!/bin/bash
# ============================================================
# 完整实验脚本: baseline + MB-Defense + 消融 + FPR
# 用法: bash scripts/run_all.sh
# ============================================================

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# 读取模型路径
if [ -f "$PROJECT_DIR/.model_path" ]; then
    MODEL_PATH=$(cat "$PROJECT_DIR/.model_path")
else
    MODEL_PATH="${1:-/root/autodl-tmp/llama2-7b-chat}"
fi

echo "============================================================"
echo "MB-Defense Full Experiment Suite"
echo "Model: $MODEL_PATH"
echo "Start time: $(date)"
echo "============================================================"

# ============================================================
# Part 1: Baseline - 原版 Backtranslation
# ============================================================
echo ""
echo ">>> [Part 1/4] Running Backtranslation Baseline..."
echo ""

for ATTACK in gcg pair; do
    echo "--- Backtranslation vs $ATTACK ---"
    python scripts/run_backtranslation.py \
        --model_path "$MODEL_PATH" \
        --attack "$ATTACK" \
        --num_samples 50 \
        --run_benign \
        --output_dir results/backtranslation/
done

# ============================================================
# Part 2: MB-Defense 完整版 (N=3, filter=on)
# ============================================================
echo ""
echo ">>> [Part 2/4] Running MB-Defense (N=3, filter=on)..."
echo ""

for ATTACK in gcg pair; do
    echo "--- MB-Defense vs $ATTACK ---"
    python scripts/run_mb_defense.py \
        --model_path "$MODEL_PATH" \
        --attack "$ATTACK" \
        --num_perspectives 3 \
        --num_samples 50 \
        --run_benign \
        --output_dir results/mb_defense/
done

# ============================================================
# Part 3: 消融实验
# ============================================================
echo ""
echo ">>> [Part 3/4] Running Ablation Studies (GCG only)..."
echo ""

ATTACK="gcg"

# 消融1: 无预筛选
echo "--- Ablation: no filter ---"
python scripts/run_mb_defense.py \
    --model_path "$MODEL_PATH" \
    --attack "$ATTACK" \
    --num_perspectives 3 \
    --no_filter \
    --num_samples 50 \
    --output_dir results/mb_defense/

# 消融2: N=1 (单视角)
echo "--- Ablation: N=1 ---"
python scripts/run_mb_defense.py \
    --model_path "$MODEL_PATH" \
    --attack "$ATTACK" \
    --num_perspectives 1 \
    --num_samples 50 \
    --output_dir results/mb_defense/

# 消融3: N=2
echo "--- Ablation: N=2 ---"
python scripts/run_mb_defense.py \
    --model_path "$MODEL_PATH" \
    --attack "$ATTACK" \
    --num_perspectives 2 \
    --num_samples 50 \
    --output_dir results/mb_defense/

# 消融4: N=5
echo "--- Ablation: N=5 ---"
python scripts/run_mb_defense.py \
    --model_path "$MODEL_PATH" \
    --attack "$ATTACK" \
    --num_perspectives 5 \
    --num_samples 50 \
    --output_dir results/mb_defense/

# ============================================================
# Part 4: 汇总结果
# ============================================================
echo ""
echo ">>> [Part 4/4] Evaluating all results..."
echo ""
python scripts/evaluate.py --results_dir results/

echo ""
echo "============================================================"
echo "ALL EXPERIMENTS COMPLETE!"
echo "End time: $(date)"
echo "Results saved in: results/"
echo ""
echo "下一步: git add results/ && git commit -m 'exp: full results' && git push"
echo "============================================================"
