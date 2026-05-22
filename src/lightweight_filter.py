"""
Module 1: Adaptive Invocation Strategy

目的: 根据 prompt 的 ambiguity 程度决定验证强度
- 高 ambiguity → 使用完整的 multi-perspective BT (N=3 或 N=5)
- 低 ambiguity → 使用轻量 single BT (N=1)

注意: 不会完全绕过防御，只是调整验证强度。
这避免了 "catastrophic false negative"（危险 prompt 被直接放行）的问题。

特征:
- Prompt 长度
- 特殊字符比例
- 角色扮演/指令嵌套标记
- Token-level entropy (可选)
"""

import re
import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class LightweightFilter:
    """Adaptive Invocation Strategy 模块"""

    # Ambiguity 标记词（常见于 jailbreak prompt 的结构性特征）
    AMBIGUITY_MARKERS = [
        # 角色扮演
        "pretend you are",
        "act as",
        "you are now",
        "roleplay",
        "imagine you",
        "in a fictional",
        "hypothetical",
        # 指令嵌套/覆盖
        "ignore previous",
        "ignore all",
        "disregard",
        "override",
        "bypass",
        "no restrictions",
        "without limitations",
        # 模式切换
        "DAN",
        "do anything now",
        "developer mode",
        "jailbreak",
        # 间接请求
        "for educational purposes",
        "for research",
        "in a story",
        "as a character",
    ]

    def __init__(
        self,
        model=None,
        tokenizer=None,
        ppl_threshold: float = 200.0,
        length_threshold: int = 500,
        special_char_ratio: float = 0.3,
    ):
        """
        Args:
            model: 用于计算 PPL 的模型（可复用目标模型）
            tokenizer: tokenizer
            ppl_threshold: PPL 阈值
            length_threshold: prompt 长度阈值
            special_char_ratio: 特殊字符比例阈值
        """
        self.model = model
        self.tokenizer = tokenizer
        self.ppl_threshold = ppl_threshold
        self.length_threshold = length_threshold
        self.special_char_ratio_threshold = special_char_ratio

    def compute_perplexity(self, text: str) -> float:
        """计算文本的 perplexity"""
        if self.model is None or self.tokenizer is None:
            return 0.0

        try:
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            )
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss

            return math.exp(loss.item())
        except Exception:
            return 0.0

    def compute_special_char_ratio(self, text: str) -> float:
        """计算特殊字符比例"""
        if len(text) == 0:
            return 0.0
        special_chars = re.findall(r'[^\w\s.,!?;:\'"()\-]', text)
        return len(special_chars) / len(text)

    def count_ambiguity_markers(self, text: str) -> int:
        """计算 ambiguity 标记词数量"""
        text_lower = text.lower()
        count = 0
        for marker in self.AMBIGUITY_MARKERS:
            if marker.lower() in text_lower:
                count += 1
        return count

    def filter(self, prompt: str) -> dict:
        """
        Adaptive Invocation Strategy

        根据 prompt 的 ambiguity 程度决定验证强度:
        - high ambiguity → full multi-perspective BT
        - low ambiguity → lightweight single BT (仍然有防御，不会直接放行)

        Args:
            prompt: 用户输入

        Returns:
            dict: {
                "ambiguity_level": "high" or "low",
                "recommended_perspectives": int,  # 推荐的 BT 视角数
                "pass_through": False,  # 永远不直接放行
                "features": {...}
            }
        """
        # 提取特征
        ppl = self.compute_perplexity(prompt)
        length = len(prompt)
        special_ratio = self.compute_special_char_ratio(prompt)
        marker_count = self.count_ambiguity_markers(prompt)

        features = {
            "perplexity": ppl,
            "length": length,
            "special_char_ratio": special_ratio,
            "ambiguity_marker_count": marker_count,
        }

        # 判定 ambiguity 程度
        # 任一高风险特征触发 → high ambiguity → full verification
        is_high_ambiguity = (
            ppl > self.ppl_threshold
            or length > self.length_threshold
            or special_ratio > self.special_char_ratio_threshold
            or marker_count >= 2
        )

        # 推荐的验证强度
        if is_high_ambiguity:
            recommended_n = 3  # 完整多视角
        else:
            recommended_n = 1  # 轻量单视角（仍然做 BT，不放行）

        return {
            "ambiguity_level": "high" if is_high_ambiguity else "low",
            "recommended_perspectives": recommended_n,
            "pass_through": False,  # 关键: 永远不直接放行
            "features": features,
        }
