# 数据准备说明

## 1. AdvBench (有害行为数据集)

下载 harmful_behaviors.csv:
```bash
wget https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv
```

## 2. 生成各攻击方法的 Jailbreak Prompts

每种攻击方法需要先用其开源代码生成对应的jailbreak prompts，然后保存为JSON格式。

### GCG Attack
```bash
# Clone GCG代码
git clone https://github.com/llm-attacks/llm-attacks.git
# 按其README运行，生成攻击prompt
# 将结果保存为 data/gcg_prompts.json (List[str]格式)
```

### PAIR Attack
```bash
git clone https://github.com/patrickrchao/JailbreakingLLMs.git
# 运行PAIR攻击，保存结果为 data/pair_prompts.json
```

### AutoDAN Attack
```bash
git clone https://github.com/SheltonLiu-N/AutoDAN.git
# 运行AutoDAN，保存结果为 data/autodan_prompts.json
```

### DeepInception Attack
```bash
git clone https://github.com/tmlr-group/DeepInception.git
# 运行DeepInception，保存结果为 data/deepinception_prompts.json
```

## 3. 正常Prompt数据 (测试误拒率)

从Alpaca数据集中采样:
```bash
# 下载alpaca_eval数据
wget https://raw.githubusercontent.com/tatsu-lab/alpaca_eval/main/src/alpaca_eval/evaluators_configs/alpaca_eval/alpaca_eval.json
# 提取instruction字段，保存为 data/benign_prompts.json
```

## 文件格式

所有prompt文件均为JSON格式的字符串列表:
```json
[
    "prompt 1",
    "prompt 2",
    ...
]
```
