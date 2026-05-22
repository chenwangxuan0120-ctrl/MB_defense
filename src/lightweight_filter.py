"""
Module 1: 轻量预筛选 (Lightweight Pre-filter)

目的: 减少不必要的backtranslation调用，降低平均延迟
方法: 
- 计算输入prompt的perplexity
- 提取简单特征: prompt长度、特殊字符比例、角色扮演关键词
- 低风险直接放行，高风险进入完整流程
"""

import re
import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class LightweightFilter:
    """轻量预筛选模块"""

    # 可疑关键词（常见于jailbreak prompt）
    SUSPICIOUS_KEYWORDS = [
        "ignore previous",
        "ignore all",
        "disregard",
        "pretend you are",
        "act as",
        "you are now",
        "roleplay",
        "jailbreak",
        "DAN",
        "do anything now",
        "developer mode",
        "hypothetical",
        "imagine you",
        "in a fictional",
        "for educational purposes",
        "bypass",
        "override",
        "no restrictions",
        "without limitations",
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
            model: 用于计算PPL的模型（可复用目标模型）
            tokenizer: tokenizer
            ppl_threshold: PPL阈值，高于此值视为高风险
            length_threshold: prompt长度阈值
            special_char_ratio: 特殊字符比例阈值
        """
        self.model = model
        self.tokenizer = tokenizer
        self.ppl_threshold = ppl_threshold
        self.length_threshold = length_threshold
        self.special_char_ratio_threshold = special_char_ratio

    def compute_perplexity(self, text: str) -> float:
        """计算文本的perplexity"""
        if self.model is None or self.tokenizer is None:
            return 0.0

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

        return math.exp(loss.item())

    def compute_special_char_ratio(self, text: str) -> float:
        """计算特殊字符比例"""
        if len(text) == 0:
            return 0.0
        # 非字母、数字、空格、常见标点的字符
        special_chars = re.findall(r'[^\w\s.,!?;:\'"()\-]', text)
        return len(special_chars) / len(text)

    def has_suspicious_keywords(self, text: str) -> bool:
        """检查是否包含可疑关键词"""
        text_lower = text.lower()
        for keyword in self.SUSPICIOUS_KEYWORDS:
            if keyword.lower() in text_lower:
                return True
        return False

    def filter(self, prompt: str) -> dict:
        """
        对输入prompt进行预筛选

        Args:
            prompt: 用户输入

        Returns:
            dict: {
                "risk_level": "low" or "high",
                "pass_through": bool,  # True表示直接放行，False表示需要进入完整流程
                "features": {
                    "perplexity": float,
                    "length": int,
                    "special_char_ratio": float,
                    "has_suspicious_keywords": bool,
                }
            }
        """
        # 提取特征
        ppl = self.compute_perplexity(prompt)
        length = len(prompt)
        special_ratio = self.compute_special_char_ratio(prompt)
        has_keywords = self.has_suspicious_keywords(prompt)

        features = {
            "perplexity": ppl,
            "length": length,
            "special_char_ratio": special_ratio,
            "has_suspicious_keywords": has_keywords,
        }

        # 判定逻辑：任一高风险特征触发则进入完整流程
        is_high_risk = (
            ppl > self.ppl_threshold
            or length > self.length_threshold
            or special_ratio > self.special_char_ratio_threshold
            or has_keywords
        )

        return {
            "risk_level": "high" if is_high_risk else "low",
            "pass_through": not is_high_risk,
            "features": features,
        }
