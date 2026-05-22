# AutoDL 使用指南

## 一键启动（clone 后只需执行这一条）

```bash
git clone https://github.com/chenwangxuan0120-ctrl/MB_defense.git
cd MB_defense
bash setup_and_run.sh
```

这个脚本会自动完成:
1. 安装所有 Python 依赖
2. 查找/下载 Llama-2-7B-Chat 模型
3. 下载实验数据集 (AdvBench, HarmBench, JailbreakBench)
4. 运行 pipeline 验证

## 跑实验

```bash
# 跑全部实验（baseline + MB-Defense + 消融），约 20-30 小时
bash scripts/run_all.sh

# 或者分步跑:

# 只跑 GCG baseline
python scripts/run_backtranslation.py --model_path $(cat .model_path) --attack gcg --run_benign

# 只跑 MB-Defense vs GCG
python scripts/run_mb_defense.py --model_path $(cat .model_path) --attack gcg --run_benign
```

## 结果回传

```bash
git add results/
git commit -m "exp: results"
git push
```
