"""
运行 MB-Defense 完整防御流程

用法:
    python scripts/run_mb_defense.py --model_path /root/models/llama2-7b-chat --attack gcg
    python scripts/run_mb_defense.py --model_path /root/models/llama2-7b-chat --attack all --no_filter
"""

import os
import sys
import json
import time
import yaml
import argparse
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.mb_defense import MBDefense
from data.load_data import load_attack_data, load_benign_data


def main():
    parser = argparse.ArgumentParser(description="Run MB-Defense")
    parser.add_argument("--model_path", type=str, required=True, help="Path to target model")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config file path")
    parser.add_argument("--attack", type=str, default="gcg",
                        choices=["gcg", "pair", "autodan", "deepinception", "direct", "all"])
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples")
    parser.add_argument("--num_perspectives", type=int, default=3, help="Number of BT perspectives (1-5)")
    parser.add_argument("--no_filter", action="store_true", help="Disable lightweight filter (for ablation)")
    parser.add_argument("--run_benign", action="store_true", help="Also run on benign prompts for FPR")
    parser.add_argument("--output_dir", type=str, default="results/mb_defense/")
    args = parser.parse_args()

    # 加载配置
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 覆盖配置
    config["model"]["target_model_path"] = args.model_path
    config["multi_perspective"]["num_perspectives"] = args.num_perspectives
    if args.no_filter:
        config["lightweight_filter"]["enabled"] = False

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 初始化MB-Defense
    defense = MBDefense(config)

    # 加载攻击数据
    attacks = [args.attack] if args.attack != "all" else ["gcg", "pair", "autodan", "deepinception"]

    for attack_name in attacks:
        print(f"\n{'='*60}")
        print(f"[MB-Defense] vs {attack_name}")
        print(f"  Perspectives: N={args.num_perspectives}")
        print(f"  Filter enabled: {config['lightweight_filter']['enabled']}")
        print(f"{'='*60}")

        prompts = load_attack_data(attack_name, num_samples=args.num_samples)
        if not prompts:
            print(f"  No data for {attack_name}, skipping.")
            continue

        results = []
        latencies = []

        for i, prompt in enumerate(tqdm(prompts, desc=f"[{attack_name}]")):
            start_time = time.time()
            result = defense.defend(prompt)
            elapsed = time.time() - start_time

            result["attack_type"] = attack_name
            result["sample_id"] = i
            result["latency_seconds"] = elapsed
            results.append(result)
            latencies.append(elapsed)

        # 统计
        total = len(results)
        rejected = sum(1 for r in results if r["decision"] == "reject")
        filter_passed = sum(1 for r in results if r.get("stage_reached") == "filter_pass")
        model_refused = sum(1 for r in results if r.get("stage_reached") == "model_self_refused")
        full_pipeline = sum(1 for r in results if r.get("stage_reached") == "full_pipeline")
        defense_rate = rejected / total * 100 if total > 0 else 0
        avg_latency = sum(latencies) / len(latencies) if latencies else 0

        print(f"\nResults for {attack_name}:")
        print(f"  Total samples: {total}")
        print(f"  Filter passed (low risk): {filter_passed}")
        print(f"  Model self-refused: {model_refused}")
        print(f"  Full pipeline: {full_pipeline}")
        print(f"  Rejected (defended): {rejected}")
        print(f"  Defense Success Rate: {defense_rate:.1f}%")
        print(f"  Avg Latency: {avg_latency:.2f}s per sample")

        # 保存结果
        suffix = f"_N{args.num_perspectives}"
        if args.no_filter:
            suffix += "_nofilter"
        output_file = os.path.join(args.output_dir, f"{attack_name}{suffix}_results.json")

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({
                "method": "mb_defense",
                "attack": attack_name,
                "model": args.model_path,
                "num_perspectives": args.num_perspectives,
                "filter_enabled": config["lightweight_filter"]["enabled"],
                "config": config,
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "total": total,
                    "rejected": rejected,
                    "defense_rate": defense_rate,
                    "avg_latency": avg_latency,
                    "filter_passed": filter_passed,
                    "model_refused": model_refused,
                    "full_pipeline": full_pipeline,
                },
                "details": results,
            }, f, indent=2, ensure_ascii=False)

        print(f"  Results saved to: {output_file}")

    # 测试误拒率 (FPR)
    if args.run_benign:
        print(f"\n{'='*60}")
        print(f"[MB-Defense] Testing False Positive Rate (benign prompts)")
        print(f"{'='*60}")

        benign_prompts = load_benign_data(num_samples=args.num_samples)
        results = []
        latencies = []

        for i, prompt in enumerate(tqdm(benign_prompts, desc="[benign]")):
            start_time = time.time()
            result = defense.defend(prompt)
            elapsed = time.time() - start_time

            result["sample_id"] = i
            result["latency_seconds"] = elapsed
            results.append(result)
            latencies.append(elapsed)

        total = len(results)
        false_positives = sum(1 for r in results if r["decision"] == "reject")
        fpr = false_positives / total * 100 if total > 0 else 0
        avg_latency = sum(latencies) / len(latencies) if latencies else 0

        print(f"\nBenign Results:")
        print(f"  Total: {total}")
        print(f"  False Positives: {false_positives}")
        print(f"  FPR: {fpr:.1f}%")
        print(f"  Avg Latency: {avg_latency:.2f}s")

        suffix = f"_N{args.num_perspectives}"
        if args.no_filter:
            suffix += "_nofilter"
        output_file = os.path.join(args.output_dir, f"benign{suffix}_results.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({
                "method": "mb_defense",
                "type": "benign",
                "model": args.model_path,
                "num_perspectives": args.num_perspectives,
                "filter_enabled": config["lightweight_filter"]["enabled"],
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "total": total,
                    "false_positives": false_positives,
                    "fpr": fpr,
                    "avg_latency": avg_latency,
                },
                "details": results,
            }, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
