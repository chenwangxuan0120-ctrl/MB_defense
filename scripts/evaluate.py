"""
评估脚本: 汇总所有实验结果，生成对比表格

用法:
    python scripts/evaluate.py --results_dir results/
"""

import os
import json
import argparse
from collections import defaultdict


def load_results(results_dir: str) -> dict:
    """加载所有结果文件"""
    all_results = {}
    for root, dirs, files in os.walk(results_dir):
        for f in files:
            if f.endswith("_results.json"):
                filepath = os.path.join(root, f)
                with open(filepath, "r") as fp:
                    data = json.load(fp)
                method = os.path.basename(root)  # backtranslation / mb_defense
                key = f"{method}/{f}"
                all_results[key] = data
    return all_results


def print_comparison_table(all_results: dict):
    """打印对比表格"""
    # 按方法和攻击类型组织
    table = defaultdict(dict)

    for key, data in all_results.items():
        method = key.split("/")[0]
        attack = data["attack"]
        defense_rate = data["summary"]["defense_rate"]
        table[method][attack] = defense_rate

    # 打印表格
    attacks = sorted(set(a for m in table.values() for a in m.keys()))
    methods = sorted(table.keys())

    print("\n" + "=" * 80)
    print("Defense Success Rate (%) Comparison")
    print("=" * 80)

    # Header
    header = f"{'Method':<25}" + "".join(f"{a:<15}" for a in attacks) + f"{'Avg':<10}"
    print(header)
    print("-" * len(header))

    # Rows
    for method in methods:
        row = f"{method:<25}"
        rates = []
        for attack in attacks:
            rate = table[method].get(attack, -1)
            if rate >= 0:
                row += f"{rate:<15.1f}"
                rates.append(rate)
            else:
                row += f"{'N/A':<15}"
        avg = sum(rates) / len(rates) if rates else 0
        row += f"{avg:<10.1f}"
        print(row)

    print("=" * 80)


def print_ablation_summary(results_dir: str):
    """打印消融实验摘要"""
    mb_dir = os.path.join(results_dir, "mb_defense")
    if not os.path.exists(mb_dir):
        return

    print("\n" + "=" * 80)
    print("Ablation Study Summary")
    print("=" * 80)

    for f in sorted(os.listdir(mb_dir)):
        if f.endswith("_results.json"):
            filepath = os.path.join(mb_dir, f)
            with open(filepath, "r") as fp:
                data = json.load(fp)
            print(f"  {f}: Defense Rate = {data['summary']['defense_rate']:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare results")
    parser.add_argument("--results_dir", type=str, default="results/", help="Results directory")
    args = parser.parse_args()

    if not os.path.exists(args.results_dir):
        print(f"Results directory not found: {args.results_dir}")
        return

    all_results = load_results(args.results_dir)

    if not all_results:
        print("No results found.")
        return

    print(f"Found {len(all_results)} result files.")
    print_comparison_table(all_results)
    print_ablation_summary(args.results_dir)


if __name__ == "__main__":
    main()
