"""
Module 2: 多视角 Backtranslation (Multi-Perspective Intent Inference)

目的: 提升意图还原的准确性和鲁棒性
方法: 设计N个不同角度的backtranslation prompt模板，从不同视角推断原始意图
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List


class MultiPerspectiveBT:
    """多视角Backtranslation模块"""

    # N个不同角度的backtranslation模板
    TEMPLATES = [
        # 模板1: 直接推断
        (
            "Based on the following response from an AI assistant, "
            "what was the user's original question or request? "
            "Only output the inferred question.\n\n"
            "Response: {response}\n\n"
            "Inferred question:"
        ),
        # 模板2: 意图聚焦
        (
            "What is the core intent or goal behind a query that would produce "
            "the following answer from an AI assistant? "
            "State the intent as a direct question or request.\n\n"
            "Answer: {response}\n\n"
            "Core intent:"
        ),
        # 模板3: 安全视角
        (
            "If someone received this response from an AI assistant, "
            "what potentially harmful or sensitive request might they have made? "
            "If the request seems benign, state it as-is.\n\n"
            "Response: {response}\n\n"
            "Possible original request:"
        ),
    ]

    def __init__(
        self,
        model=None,
        tokenizer=None,
        device: str = "cuda",
        num_perspectives: int = 3,
        temperature: float = 0.7,
        max_new_tokens: int = 256,
    ):
        """
        Args:
            model: LLM模型（可复用目标模型）
            tokenizer: tokenizer
            device: 设备
            num_perspectives: 使用的视角数量（1-3）
            temperature: 生成温度
            max_new_tokens: 最大生成token数
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.num_perspectives = min(num_perspectives, len(self.TEMPLATES))
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def generate(self, prompt: str) -> str:
        """生成回复"""
        formatted = f"[INST] {prompt} [/INST]"
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True if self.temperature > 0 else False,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return response

    def multi_backtranslate(self, response: str) -> List[str]:
        """
        从response用多个视角反推原始prompt

        Args:
            response: 目标模型对原始prompt的回复

        Returns:
            List[str]: N个backtranslated prompts
        """
        backtranslated_prompts = []

        for i in range(self.num_perspectives):
            template = self.TEMPLATES[i]
            bt_prompt = template.format(response=response)
            inferred = self.generate(bt_prompt)
            backtranslated_prompts.append(inferred)

        return backtranslated_prompts

    def get_responses(self, backtranslated_prompts: List[str]) -> List[str]:
        """
        对每个backtranslated prompt生成回复

        Args:
            backtranslated_prompts: 多视角推断出的prompts

        Returns:
            List[str]: 对应的回复列表
        """
        responses = []
        for bt_prompt in backtranslated_prompts:
            resp = self.generate(bt_prompt)
            responses.append(resp)
        return responses
