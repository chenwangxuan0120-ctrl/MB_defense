#!/bin/bash
# 消融实验脚本
# 用法: bash scripts/run_ablation.sh /root/models/llama2-7b-chat

MODEL_PATH=${1:-"/root/models/llama2-7b-chat"}
ATTACK="gcg"  # 先用GCG做消融，确认有效后再跑全量

echo "=========================================="
echo "Ablation Study for MB-Defense"
echo "Model: $MODEL_PATH"
echo "Attack: $ATTACK"
echo "=========================================="

# 1. 原版 Backtranslation (baseline)
echo "[1/6] Running original Backtranslation..."
python scripts/run_backtranslation.py --model_path $MODEL_PATH --attack $ATTACK --output_dir results/backtranslation/

# 2. MB-Defense 完整版 (N=3, with filter)
echo "[2/6] Running MB-Defense full (N=3, filter=on)..."
python scripts/run_mb_defense.py --model_path $MODEL_PATH --attack $ATTACK --num_perspectives 3 --output_dir results/mb_defense/

# 3. MB-Defense 无预筛选 (N=3, no filter)
echo "[3/6] Running MB-Defense without filter..."
python scripts/run_mb_defense.py --model_path $MODEL_PATH --attack $ATTACK --num_perspectives 3 --no_filter --output_dir results/mb_defense/

# 4. MB-Defense N=1 (单视角，验证多视角的贡献)
echo "[4/6] Running MB-Defense N=1..."
python scripts/run_mb_defense.py --model_path $MODEL_PATH --attack $ATTACK --num_perspectives 1 --output_dir results/mb_defense/

# 5. MB-Defense N=2
echo "[5/6] Running MB-Defense N=2..."
python scripts/run_mb_defense.py --model_path $MODEL_PATH --attack $ATTACK --num_perspectives 2 --output_dir results/mb_defense/

# 6. MB-Defense N=5
echo "[6/6] Running MB-Defense N=5..."
python scripts/run_mb_defense.py --model_path $MODEL_PATH --attack $ATTACK --num_perspectives 5 --output_dir results/mb_defense/

# 汇总结果
echo ""
echo "=========================================="
echo "Evaluating all results..."
echo "=========================================="
python scripts/evaluate.py --results_dir results/

echo "Done!"
