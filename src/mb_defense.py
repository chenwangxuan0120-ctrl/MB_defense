"""
MB-Defense: 完整的多视角Backtranslation防御流程

整合三个模块:
- Module 1: 轻量预筛选
- Module 2: 多视角Backtranslation
- Module 3: 语义偏离校验 + 投票判定
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.lightweight_filter import LightweightFilter
from src.multi_perspective_bt import MultiPerspectiveBT
from src.semantic_divergence import SemanticDivergenceVerifier


class MBDefense:
    """MB-Defense 完整防御系统"""

    def __init__(self, config: dict):
        """
        Args:
            config: 配置字典，参考 configs/default.yaml
        """
        self.config = config
        model_path = config["model"]["target_model_path"]
        device = config["model"]["device"]

        # 加载目标模型
        print(f"Loading target model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model.eval()
        # device_map="auto" 时，模型自动分配设备，用 model.device 获取实际设备
        self.device = next(self.model.parameters()).device
        print(f"Target model loaded on {self.device}.")

        # 初始化 Module 1: 轻量预筛选
        filter_cfg = config["lightweight_filter"]
        self.lightweight_filter = LightweightFilter(
            model=self.model,
            tokenizer=self.tokenizer,
            ppl_threshold=filter_cfg["ppl_threshold"],
            length_threshold=filter_cfg["length_threshold"],
            special_char_ratio=filter_cfg["special_char_ratio"],
        )

        # 初始化 Module 2: 多视角BT
        mp_cfg = config["multi_perspective"]
        self.multi_bt = MultiPerspectiveBT(
            model=self.model,
            tokenizer=self.tokenizer,
            device=device,
            num_perspectives=mp_cfg["num_perspectives"],
            temperature=mp_cfg["temperature"],
            max_new_tokens=mp_cfg["max_new_tokens"],
        )

        # 初始化 Module 3: 语义偏离校验
        sd_cfg = config["semantic_divergence"]
        self.verifier = SemanticDivergenceVerifier(
            embedding_model_name=config["model"]["embedding_model"],
            alpha=sd_cfg["alpha"],
            beta=sd_cfg["beta"],
            threshold=sd_cfg["threshold"],
        )

    def generate(self, prompt: str) -> str:
        """生成回复"""
        formatted = f"[INST] {prompt} [/INST]"
        inputs = self.tokenizer(formatted, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config["model"]["max_new_tokens"],
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def defend(self, input_prompt: str) -> dict:
        """
        执行完整的MB-Defense防御流程

        Args:
            input_prompt: 用户输入的prompt

        Returns:
            dict: 包含决策和各模块中间结果
        """
        result = {
            "input_prompt": input_prompt,
            "decision": None,
            "stage_reached": None,
            "filter_result": None,
            "initial_response": None,
            "backtranslated_prompts": None,
            "bt_responses": None,
            "verification_result": None,
        }

        # ===== Stage 1: 轻量预筛选 =====
        if self.config["lightweight_filter"]["enabled"]:
            filter_result = self.lightweight_filter.filter(input_prompt)
            result["filter_result"] = filter_result

            if filter_result["pass_through"]:
                # 低风险，直接放行
                result["decision"] = "accept"
                result["stage_reached"] = "filter_pass"
                return result

        # ===== Stage 2: 多视角 Backtranslation =====
        result["stage_reached"] = "full_pipeline"

        # 先生成初始回复
        initial_response = self.generate(input_prompt)
        result["initial_response"] = initial_response

        # 如果模型本身就拒绝了，不需要防御
        if self.verifier.is_refusal(initial_response):
            result["decision"] = "accept"
            result["stage_reached"] = "model_self_refused"
            return result

        # 多视角backtranslation
        backtranslated_prompts = self.multi_bt.multi_backtranslate(initial_response)
        result["backtranslated_prompts"] = backtranslated_prompts

        # 对每个BT prompt生成回复
        bt_responses = self.multi_bt.get_responses(backtranslated_prompts)
        result["bt_responses"] = bt_responses

        # ===== Stage 3: 语义偏离校验 + 投票判定 =====
        verification = self.verifier.verify(
            original_prompt=input_prompt,
            backtranslated_prompts=backtranslated_prompts,
            bt_responses=bt_responses,
        )
        result["verification_result"] = verification
        result["decision"] = verification["decision"]

        return result
