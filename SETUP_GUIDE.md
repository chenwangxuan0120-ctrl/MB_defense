# AutoDL 环境配置指南

## 1. 租机器

- 镜像: PyTorch 2.1.0 / Python 3.10 (ubuntu22.04) / CUDA 12.1
- GPU: A100 80G 或 A800 80G
- 数据盘: 50GB+

## 2. 开机后执行

```bash
# 拉代码
git clone https://github.com/<你的用户名>/MB-Defense.git
cd MB-Defense

# 装依赖
pip install -r requirements.txt

# 准备数据
python scripts/prepare_data.py

# 下载模型 (如果公共目录没有)
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download meta-llama/Llama-2-7b-chat-hf --local-dir /root/models/llama2-7b-chat

# 验证模型能加载
python -c "from transformers import AutoTokenizer; t = AutoTokenizer.from_pretrained('/root/models/llama2-7b-chat'); print('OK')"
```

## 3. 快速验证 pipeline 能跑通

```bash
# 用5个样本快速测试
python scripts/run_backtranslation.py --model_path /root/models/llama2-7b-chat --attack gcg --num_samples 5
```

## 4. 正式跑实验

```bash
# 跑原版baseline
python scripts/run_backtranslation.py --model_path /root/models/llama2-7b-chat --attack all --num_samples 50

# 跑MB-Defense
python scripts/run_mb_defense.py --model_path /root/models/llama2-7b-chat --attack all --num_samples 50

# 消融实验
bash scripts/run_ablation.sh /root/models/llama2-7b-chat
```

## 5. 结果回传

```bash
git add results/
git commit -m "exp: add results for [attack_name]"
git push
```
