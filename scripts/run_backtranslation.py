"""
复现原版 Backtranslation 防御

用法:
    python scripts/run_backtranslation.py --model_path /root/models/llama2-7b-chat --attack gcg
    python scripts/run_backtranslation.py --model_path /root/models/llama2-7b-chat --attack all --num_samples 100
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.backtranslation import BacktranslationDefense
from data.load_data import load_attack_data, load_benign_data


def main():
    parser = argparse.ArgumentParser(description="Run original Backtranslation defense")
    parser.add_argument("--model_path", type=str, required=True, help="Path to target model")
    parser.add_argument("--attack", type=str, default="gcg",
                        choices=["gcg", "pair", "autodan", "deepinception", "direct", "all"])
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples to evaluate")
    parser.add_argument("--run_benign", action="store_true", help="Also run on benign prompts for FPR")
    parser.add_argument("--output_dir", type=str, default="results/backtranslation/")
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 初始化防御
    defense = BacktranslationDefense(model_path=args.model_path)

    # 加载攻击数据
    attacks = [args.attack] if args.attack != "all" else ["gcg", "pair", "autodan", "deepinception"]

    for attack_name in attacks:
        print(f"\n{'='*60}")
        print(f"[Backtranslation] Evaluating against: {attack_name}")
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
        defense_rate = rejected / total * 100 if total > 0 else 0
        avg_latency = sum(latencies) / len(latencies) if latencies else 0

        print(f"\nResults for {attack_name}:")
        print(f"  Total samples: {total}")
        print(f"  Rejected (defended): {rejected}")
        print(f"  Defense Success Rate: {defense_rate:.1f}%")
        print(f"  Avg Latency: {avg_latency:.2f}s per sample")

        # 保存结果
        output_file = os.path.join(args.output_dir, f"{attack_name}_results.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({
                "method": "backtranslation",
                "attack": attack_name,
                "model": args.model_path,
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "total": total,
                    "rejected": rejected,
                    "defense_rate": defense_rate,
                    "avg_latency": avg_latency,
                },
                "details": results,
            }, f, indent=2, ensure_ascii=False)

        print(f"  Results saved to: {output_file}")

    # 测试误拒率 (FPR)
    if args.run_benign:
        print(f"\n{'='*60}")
        print(f"[Backtranslation] Testing False Positive Rate (benign prompts)")
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

        output_file = os.path.join(args.output_dir, "benign_results.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({
                "method": "backtranslation",
                "type": "benign",
                "model": args.model_path,
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
