"""
Module 3: 语义偏离校验 + 投票判定 (Semantic Divergence Verification)

目的: 量化攻击伪装程度，提升判定准确性
方法:
1. 用sentence embedding计算原始prompt与backtranslated prompts的语义偏离度
2. 结合拒绝投票做最终判定
"""

import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer


class SemanticDivergenceVerifier:
    """语义偏离校验 + 投票判定模块"""

    # 拒绝关键词（与backtranslation.py一致）
    REFUSAL_KEYWORDS = [
        "I cannot",
        "I can't",
        "I'm not able to",
        "I am not able to",
        "I will not",
        "I won't",
        "I'm sorry, but I cannot",
        "As an AI",
        "As a responsible AI",
        "I must refuse",
        "not appropriate",
        "against my programming",
        "harmful",
        "unethical",
        "illegal",
    ]

    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        alpha: float = 0.6,
        beta: float = 0.4,
        threshold: float = 0.5,
    ):
        """
        Args:
            embedding_model_name: sentence embedding模型名
            alpha: 拒绝比例权重
            beta: 语义偏离度权重
            threshold: 最终判定阈值
        """
        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold

        print(f"Loading embedding model: {embedding_model_name}...")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        print("Embedding model loaded.")

    def compute_divergence(self, original_prompt: str, backtranslated_prompts: List[str]) -> List[float]:
        """
        计算原始prompt与每个backtranslated prompt的语义偏离度

        Args:
            original_prompt: 原始输入prompt
            backtranslated_prompts: 多视角推断出的prompts

        Returns:
            List[float]: 每个BT prompt的偏离度 (0-1, 越高越偏离)
        """
        # 编码
        all_texts = [original_prompt] + backtranslated_prompts
        embeddings = self.embedding_model.encode(all_texts, normalize_embeddings=True)

        original_emb = embeddings[0]
        divergences = []

        for i in range(1, len(embeddings)):
            # 余弦相似度（已归一化，直接点积）
            cosine_sim = np.dot(original_emb, embeddings[i])
            # 偏离度 = 1 - 相似度
            divergence = 1.0 - cosine_sim
            # clip到[0, 1]
            divergence = max(0.0, min(1.0, divergence))
            divergences.append(divergence)

        return divergences

    def is_refusal(self, response: str) -> bool:
        """判断回复是否为拒绝"""
        response_lower = response.lower()
        for keyword in self.REFUSAL_KEYWORDS:
            if keyword.lower() in response_lower:
                return True
        return False

    def verify(
        self,
        original_prompt: str,
        backtranslated_prompts: List[str],
        bt_responses: List[str],
    ) -> dict:
        """
        综合判定

        Score = α × (拒绝比例) + β × (平均语义偏离度)
        if Score > threshold: Reject
        else: Accept

        Args:
            original_prompt: 原始输入prompt
            backtranslated_prompts: 多视角推断出的prompts
            bt_responses: 对BT prompts的回复

        Returns:
            dict: {
                "decision": "accept" or "reject",
                "score": float,
                "refusal_ratio": float,
                "avg_divergence": float,
                "divergences": List[float],
                "refusals": List[bool],
            }
        """
        # 计算语义偏离度
        divergences = self.compute_divergence(original_prompt, backtranslated_prompts)
        avg_divergence = np.mean(divergences) if divergences else 0.0

        # 统计拒绝
        refusals = [self.is_refusal(resp) for resp in bt_responses]
        refusal_ratio = sum(refusals) / len(refusals) if refusals else 0.0

        # 综合评分
        score = self.alpha * refusal_ratio + self.beta * avg_divergence

        # 判定
        decision = "reject" if score > self.threshold else "accept"

        return {
            "decision": decision,
            "score": score,
            "refusal_ratio": refusal_ratio,
            "avg_divergence": avg_divergence,
            "divergences": divergences,
            "refusals": refusals,
        }
