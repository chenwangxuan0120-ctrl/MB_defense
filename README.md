# MB-Defense: Multi-perspective Backtranslation Defense

Enhanced Backtranslation Defense against LLM Jailbreaking via Multi-Perspective Intent Inference and Semantic Divergence Verification.

## 环境配置

```bash
pip install -r requirements.txt
```

## 模型准备

```bash
export HF_ENDPOINT=https://hf-mirror.com

# Llama-2-7B-Chat (主要目标模型)
huggingface-cli download meta-llama/Llama-2-7b-chat-hf --local-dir /root/models/llama2-7b-chat

# Vicuna-7B-v1.5 (第二目标模型)
huggingface-cli download lmsys/vicuna-7b-v1.5 --local-dir /root/models/vicuna-7b-v1.5
```

## 运行步骤

### Step 1: 复现原版 Backtranslation
```bash
python scripts/run_backtranslation.py --model_path /root/models/llama2-7b-chat --attack gcg
```

### Step 2: 运行 MB-Defense
```bash
python scripts/run_mb_defense.py --model_path /root/models/llama2-7b-chat --attack gcg
```

### Step 3: 评估
```bash
python scripts/evaluate.py --results_dir results/
```

## 项目结构

```
MB-Defense/
├── src/                    # 核心代码
│   ├── backtranslation.py          # 原版BT逻辑
│   ├── multi_perspective_bt.py     # 多视角BT (Module 2)
│   ├── semantic_divergence.py      # 语义偏离计算 (Module 3)
│   └── lightweight_filter.py       # 轻量预筛选 (Module 1)
├── scripts/                # 运行脚本
│   ├── run_backtranslation.py      # 复现原版
│   ├── run_mb_defense.py           # 我们的方法
│   └── evaluate.py                 # 评估
├── data/                   # 攻击数据集
├── configs/                # 配置文件
└── results/                # 实验结果
```
