"""
复现原版 Backtranslation 防御

用法:
    python scripts/run_backtranslation.py --model_path /root/models/llama2-7b-chat --attack gcg
"""

import os
import sys
import json
import argparse
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.backtranslation import BacktranslationDefense
from data.load_data import load_attack_data


def main():
    parser = argparse.ArgumentParser(description="Run original Backtranslation defense")
    parser.add_argument("--model_path", type=str, required=True, help="Path to target model")
    parser.add_argument("--attack", type=str, default="gcg", choices=["gcg", "pair", "autodan", "deepinception", "all"])
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples to evaluate")
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
        print(f"Evaluating against: {attack_name}")
        print(f"{'='*60}")

        prompts = load_attack_data(attack_name, num_samples=args.num_samples)
        results = []

        for i, prompt in enumerate(tqdm(prompts, desc=f"[{attack_name}]")):
            result = defense.defend(prompt)
            result["attack_type"] = attack_name
            result["sample_id"] = i
            results.append(result)

        # 统计
        total = len(results)
        rejected = sum(1 for r in results if r["decision"] == "reject")
        defense_rate = rejected / total * 100 if total > 0 else 0

        print(f"\nResults for {attack_name}:")
        print(f"  Total samples: {total}")
        print(f"  Rejected (defended): {rejected}")
        print(f"  Defense Success Rate: {defense_rate:.1f}%")

        # 保存结果
        output_file = os.path.join(args.output_dir, f"{attack_name}_results.json")
        with open(output_file, "w") as f:
            json.dump({
                "attack": attack_name,
                "model": args.model_path,
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "total": total,
                    "rejected": rejected,
                    "defense_rate": defense_rate,
                },
                "details": results,
            }, f, indent=2, ensure_ascii=False)

        print(f"  Results saved to: {output_file}")


if __name__ == "__main__":
    main()
