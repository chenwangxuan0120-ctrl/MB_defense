"""
验证所有模块导入是否正常（不需要GPU，纯CPU检查）
用法: python verify_imports.py
"""

import sys
import os

# 确保项目根目录在 path 中
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

print("Verifying imports...")

errors = []

# 1. 检查 src 模块
try:
    from src.backtranslation import BacktranslationDefense
    print("  ✓ src.backtranslation")
except Exception as e:
    errors.append(f"  ✗ src.backtranslation: {e}")

try:
    from src.lightweight_filter import LightweightFilter
    print("  ✓ src.lightweight_filter")
except Exception as e:
    errors.append(f"  ✗ src.lightweight_filter: {e}")

try:
    from src.multi_perspective_bt import MultiPerspectiveBT
    print("  ✓ src.multi_perspective_bt")
except Exception as e:
    errors.append(f"  ✗ src.multi_perspective_bt: {e}")

try:
    from src.semantic_divergence import SemanticDivergenceVerifier
    print("  ✓ src.semantic_divergence")
except Exception as e:
    errors.append(f"  ✗ src.semantic_divergence: {e}")

try:
    from src.mb_defense import MBDefense
    print("  ✓ src.mb_defense")
except Exception as e:
    errors.append(f"  ✗ src.mb_defense: {e}")

# 2. 检查 data 模块
try:
    from data.load_data import load_attack_data, load_benign_data
    print("  ✓ data.load_data")
except Exception as e:
    errors.append(f"  ✗ data.load_data: {e}")

# 3. 检查第三方依赖
deps = [
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("accelerate", "accelerate"),
    ("sentence_transformers", "sentence-transformers"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("sklearn", "scikit-learn"),
    ("yaml", "pyyaml"),
    ("tqdm", "tqdm"),
]

print("\nChecking dependencies...")
for module_name, pip_name in deps:
    try:
        __import__(module_name)
        print(f"  ✓ {pip_name}")
    except ImportError:
        errors.append(f"  ✗ {pip_name} not installed (pip install {pip_name})")

# 检查 jailbreakbench（可选）
try:
    import jailbreakbench
    print("  ✓ jailbreakbench")
except ImportError:
    print("  ⚠ jailbreakbench not installed (optional, pip install jailbreakbench)")

# 4. 检查配置文件
print("\nChecking config files...")
config_path = os.path.join(PROJECT_DIR, "configs", "default.yaml")
if os.path.exists(config_path):
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    print(f"  ✓ configs/default.yaml loaded")
else:
    errors.append(f"  ✗ configs/default.yaml not found")

# 结果
print("\n" + "=" * 50)
if errors:
    print("ERRORS FOUND:")
    for e in errors:
        print(e)
    print("\nPlease fix the above errors before running experiments.")
    sys.exit(1)
else:
    print("✓ All imports OK! Ready to run experiments.")
    sys.exit(0)
