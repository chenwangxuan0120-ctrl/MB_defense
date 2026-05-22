"""
原版 Backtranslation 防御实现
参考: "Defending LLMs against Jailbreaking Attacks via Backtranslation" (ACL 2024 Findings)

流程:
1. 目标LLM对输入prompt生成初始response
2. 用LLM从response反推backtranslated prompt
3. 目标LLM对backtranslated prompt生成回复
4. 如果模型拒绝了backtranslated prompt，则判定原始prompt有害
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class BacktranslationDefense:
    """原版Backtranslation防御"""

    # Llama-2-Chat 的拒绝关键词
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
        "I don't think I should",
        "harmful",
        "unethical",
        "illegal",
    ]

    BACKTRANSLATION_TEMPLATE = (
        "You are a helpful assistant. Based on the following response from an AI assistant, "
        "please infer what the user's original question or request was. "
        "Only output the inferred question, nothing else.\n\n"
        "AI Response: {response}\n\n"
        "Inferred user question:"
    )

    def __init__(self, model_path: str, device: str = "cuda", max_new_tokens: int = 256):
        """
        Args:
            model_path: 模型路径 (Llama-2-7B-Chat)
            device: 设备
            max_new_tokens: 最大生成token数
        """
        self.max_new_tokens = max_new_tokens

        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device
        print(f"Model loaded on {self.device}.")

    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        """生成回复"""
        # Llama-2-Chat 格式
        formatted = f"[INST] {prompt} [/INST]"
        inputs = self.tokenizer(formatted, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # 只取生成的部分
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return response

    def is_refusal(self, response: str) -> bool:
        """判断回复是否为拒绝"""
        response_lower = response.lower()
        for keyword in self.REFUSAL_KEYWORDS:
            if keyword.lower() in response_lower:
                return True
        return False

    def backtranslate(self, response: str) -> str:
        """从response反推原始prompt"""
        bt_prompt = self.BACKTRANSLATION_TEMPLATE.format(response=response)
        return self.generate(bt_prompt)

    def defend(self, input_prompt: str) -> dict:
        """
        执行完整的Backtranslation防御流程

        Args:
            input_prompt: 用户输入的prompt

        Returns:
            dict: {
                "decision": "accept" or "reject",
                "initial_response": str,
                "backtranslated_prompt": str,
                "bt_response": str,
                "bt_refused": bool,
            }
        """
        # Step 1: 生成初始回复
        initial_response = self.generate(input_prompt)

        # 如果初始回复就是拒绝，直接放行（模型本身就拒绝了，不需要防御）
        if self.is_refusal(initial_response):
            return {
                "decision": "accept",  # 模型已经自己拒绝了有害请求
                "initial_response": initial_response,
                "backtranslated_prompt": None,
                "bt_response": None,
                "bt_refused": True,
                "note": "Model already refused the original prompt",
            }

        # Step 2: Backtranslation - 从response反推prompt
        backtranslated_prompt = self.backtranslate(initial_response)

        # Step 3: 对backtranslated prompt生成回复
        bt_response = self.generate(backtranslated_prompt)

        # Step 4: 判断是否拒绝
        bt_refused = self.is_refusal(bt_response)

        # 如果模型拒绝了backtranslated prompt，说明原始prompt有害
        decision = "reject" if bt_refused else "accept"

        return {
            "decision": decision,
            "initial_response": initial_response,
            "backtranslated_prompt": backtranslated_prompt,
            "bt_response": bt_response,
            "bt_refused": bt_refused,
        }
