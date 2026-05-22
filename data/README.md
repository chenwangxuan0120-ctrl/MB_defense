# 数据准备说明

## 快速开始

```bash
# 自动下载 AdvBench + HarmBench + 验证 JailbreakBench
python scripts/prepare_data.py
```

## 数据来源

### 自动获取（无需手动操作）

| 数据集 | 攻击类型 | 说明 |
|--------|----------|------|
| JailbreakBench | GCG, PAIR | `pip install jailbreakbench` 后自动加载预生成的 jailbreak prompts |
| AdvBench | Direct Attack | 520 harmful behaviors，自动下载 |
| HarmBench | Direct Attack | 标准化测试集，自动下载 |

### 需要手动准备（可选）

| 文件 | 攻击类型 | 来源 |
|------|----------|------|
| `autodan_prompts.json` | AutoDAN | https://github.com/SheltonLiu-N/AutoDAN |
| `deepinception_prompts.json` | DeepInception | https://github.com/tmlr-group/DeepInception |

手动准备的文件格式为 JSON 字符串列表:
```json
["jailbreak prompt 1", "jailbreak prompt 2", ...]
```

## 推荐实验策略

对于 PRICAI 投稿，建议优先使用:
1. **GCG** (JailbreakBench) — 最经典的优化型攻击
2. **PAIR** (JailbreakBench) — 最经典的生成型攻击  
3. **Direct Attack** (HarmBench/AdvBench) — 无 jailbreak 包装的直接有害请求

如果时间充裕再加:
4. **AutoDAN** — 需要自己跑攻击代码生成 prompts
5. **DeepInception** — 需要自己跑攻击代码生成 prompts
