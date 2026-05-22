"""
Module 3: Intent Consistency Verification

核心思想:
- Benign prompt: 多个 BT 推断出的意图高度一致 (低 entropy)
- Jailbreak prompt: 多个 BT 推断出的意图分散不一致 (高 entropy)

检测信号:
1. Intent Entropy: 多个 BT 结果之间的 pairwise similarity 的 variance
2. Surface-Intent Gap: 原始 prompt 与 BT prompts 的语义距离
3. Refusal Consistency: 模型对 BT prompts 的拒绝一致性

最终判定:
Score = α × refusal_ratio + β × intent_entropy + γ × surface_intent_gap
"""

import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer


class IntentConsistencyVerifier:
    """Intent Consistency Verification 模块"""

    # 拒绝关键词
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
        "I'm not going to",
        "I apologize, but",
        "I'm unable to",
        "violates my guidelines",
        "I don't think I should",
    ]

    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        alpha: float = 0.4,
        beta: float = 0.35,
        gamma: float = 0.25,
        threshold: float = 0.45,
    ):
        """
        Args:
            embedding_model_name: sentence embedding 模型名
            alpha: 拒绝比例权重
            beta: intent entropy 权重
            gamma: surface-intent gap 权重
            threshold: 最终判定阈值
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.threshold = threshold

        print(f"Loading embedding model: {embedding_model_name}...")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        print("Embedding model loaded.")

    def compute_intent_entropy(self, backtranslated_prompts: List[str]) -> float:
        """
        计算 Intent Entropy: 多个 BT 结果之间的一致性

        方法: 计算所有 BT prompt 之间的 pairwise cosine similarity，
        然后取 variance。variance 越大说明推断结果越不一致。

        Benign: BT 结果一致 → pairwise sim 都高 → variance 低 → entropy 低
        Jailbreak: BT 结果分散 → pairwise sim 参差不齐 → variance 高 → entropy 高

        Args:
            backtranslated_prompts: 多视角推断出的 prompts

        Returns:
            float: intent entropy (0-1, 归一化后)
        """
        if len(backtranslated_prompts) < 2:
            return 0.0

        # 编码所有 BT prompts
        embeddings = self.embedding_model.encode(
            backtranslated_prompts, normalize_embeddings=True
        )

        # 计算 pairwise cosine similarity
        n = len(embeddings)
        pairwise_sims = []
        for i in range(n):
            for j in range(i + 1, n):
                sim = np.dot(embeddings[i], embeddings[j])
                pairwise_sims.append(sim)

        if not pairwise_sims:
            return 0.0

        # Intent entropy = 1 - mean_similarity + std_similarity
        # 高 entropy = BT 结果之间不一致
        mean_sim = np.mean(pairwise_sims)
        std_sim = np.std(pairwise_sims)

        # 归一化到 [0, 1]
        # mean_sim 低 → 不一致 → entropy 高
        # std_sim 高 → 不稳定 → entropy 高
        entropy = (1.0 - mean_sim) * 0.7 + std_sim * 0.3
        entropy = max(0.0, min(1.0, entropy))

        return entropy

    def compute_surface_intent_gap(
        self, original_prompt: str, backtranslated_prompts: List[str]
    ) -> float:
        """
        计算 Surface-Intent Gap: 原始 prompt 与推断意图之间的距离

        攻击 prompt 经过伪装，表面语义与真实意图差距大 → gap 大
        正常 prompt 表面就是真实意图 → gap 小

        Args:
            original_prompt: 原始输入 prompt
            backtranslated_prompts: 多视角推断出的 prompts

        Returns:
            float: surface-intent gap (0-1)
        """
        all_texts = [original_prompt] + backtranslated_prompts
        embeddings = self.embedding_model.encode(all_texts, normalize_embeddings=True)

        original_emb = embeddings[0]
        gaps = []

        for i in range(1, len(embeddings)):
            sim = np.dot(original_emb, embeddings[i])
            gap = 1.0 - sim
            gaps.append(max(0.0, min(1.0, gap)))

        return np.mean(gaps) if gaps else 0.0

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
        Intent Consistency Verification

        Score = α × refusal_ratio + β × intent_entropy + γ × surface_intent_gap
        if Score > threshold: Reject (intent inconsistency detected)
        else: Accept

        Args:
            original_prompt: 原始输入 prompt
            backtranslated_prompts: 多视角推断出的 prompts
            bt_responses: 对 BT prompts 的回复

        Returns:
            dict: 完整的验证结果
        """
        # 1. Intent Entropy
        intent_entropy = self.compute_intent_entropy(backtranslated_prompts)

        # 2. Surface-Intent Gap
        surface_intent_gap = self.compute_surface_intent_gap(
            original_prompt, backtranslated_prompts
        )

        # 3. Refusal Ratio
        refusals = [self.is_refusal(resp) for resp in bt_responses]
        refusal_ratio = sum(refusals) / len(refusals) if refusals else 0.0

        # 综合评分
        score = (
            self.alpha * refusal_ratio
            + self.beta * intent_entropy
            + self.gamma * surface_intent_gap
        )

        # 判定
        decision = "reject" if score > self.threshold else "accept"

        return {
            "decision": decision,
            "score": score,
            "refusal_ratio": refusal_ratio,
            "intent_entropy": intent_entropy,
            "surface_intent_gap": surface_intent_gap,
            "refusals": refusals,
            "threshold": self.threshold,
            "weights": {"alpha": self.alpha, "beta": self.beta, "gamma": self.gamma},
        }


# 保持向后兼容
SemanticDivergenceVerifier = IntentConsistencyVerifier
