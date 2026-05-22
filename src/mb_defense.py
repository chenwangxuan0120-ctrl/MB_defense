"""
MB-Defense: Multi-Perspective Backtranslation with Intent Consistency Verification

整合三个模块:
- Module 1: Adaptive Invocation Strategy (根据 ambiguity 调整验证强度)
- Module 2: Multi-Perspective Backtranslation (多视角意图推断)
- Module 3: Intent Consistency Verification (意图一致性校验)

核心思想:
- Jailbreak prompt 经过伪装，多视角 BT 会推断出不一致的意图 (高 entropy)
- Benign prompt 意图明确，多视角 BT 推断结果一致 (低 entropy)
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.lightweight_filter import LightweightFilter
from src.multi_perspective_bt import MultiPerspectiveBT
from src.semantic_divergence import IntentConsistencyVerifier


class MBDefense:
    """MB-Defense 完整防御系统"""

    def __init__(self, config: dict):
        """
        Args:
            config: 配置字典，参考 configs/default.yaml
        """
        self.config = config
        model_path = config["model"]["target_model_path"]

        # 加载目标模型
        print(f"Loading target model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device
        print(f"Target model loaded on {self.device}.")

        # 初始化 Module 1: Adaptive Invocation Strategy
        filter_cfg = config["lightweight_filter"]
        self.adaptive_filter = LightweightFilter(
            model=self.model,
            tokenizer=self.tokenizer,
            ppl_threshold=filter_cfg["ppl_threshold"],
            length_threshold=filter_cfg["length_threshold"],
            special_char_ratio=filter_cfg["special_char_ratio"],
        )

        # 初始化 Module 2: Multi-Perspective BT
        mp_cfg = config["multi_perspective"]
        self.multi_bt = MultiPerspectiveBT(
            model=self.model,
            tokenizer=self.tokenizer,
            device=str(self.device),
            num_perspectives=mp_cfg["num_perspectives"],
            temperature=mp_cfg["temperature"],
            max_new_tokens=mp_cfg["max_new_tokens"],
        )

        # 初始化 Module 3: Intent Consistency Verification
        sd_cfg = config["semantic_divergence"]
        self.verifier = IntentConsistencyVerifier(
            embedding_model_name=config["model"]["embedding_model"],
            alpha=sd_cfg["alpha"],
            beta=sd_cfg.get("beta", 0.35),
            gamma=sd_cfg.get("gamma", 0.25),
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
        执行 MB-Defense 防御流程

        流程:
        1. Adaptive Invocation: 判断 ambiguity → 决定用 N=1 还是 N=3
        2. 生成初始 response
        3. Multi-Perspective BT: 用 N 个视角推断意图
        4. Intent Consistency Verification: 检测意图一致性

        Args:
            input_prompt: 用户输入的 prompt

        Returns:
            dict: 包含决策和各模块中间结果
        """
        result = {
            "input_prompt": input_prompt,
            "decision": None,
            "stage_reached": None,
            "num_perspectives_used": None,
            "filter_result": None,
            "initial_response": None,
            "backtranslated_prompts": None,
            "bt_responses": None,
            "verification_result": None,
        }

        # ===== Stage 1: Adaptive Invocation Strategy =====
        if self.config["lightweight_filter"]["enabled"]:
            filter_result = self.adaptive_filter.filter(input_prompt)
            result["filter_result"] = filter_result
            # 根据 ambiguity 调整视角数（不直接放行）
            num_perspectives = filter_result["recommended_perspectives"]
        else:
            # filter 禁用时使用配置中的默认值
            num_perspectives = self.config["multi_perspective"]["num_perspectives"]

        result["num_perspectives_used"] = num_perspectives

        # ===== Stage 2: 生成初始回复 =====
        initial_response = self.generate(input_prompt)
        result["initial_response"] = initial_response

        # 如果模型本身就拒绝了，说明安全对齐生效，不需要额外防御
        if self.verifier.is_refusal(initial_response):
            result["decision"] = "accept"
            result["stage_reached"] = "model_self_refused"
            return result

        # ===== Stage 3: Multi-Perspective Backtranslation =====
        result["stage_reached"] = f"bt_N{num_perspectives}"

        # 临时调整视角数
        original_n = self.multi_bt.num_perspectives
        self.multi_bt.num_perspectives = min(num_perspectives, len(self.multi_bt.TEMPLATES))

        backtranslated_prompts = self.multi_bt.multi_backtranslate(initial_response)
        result["backtranslated_prompts"] = backtranslated_prompts

        # 对每个 BT prompt 生成回复
        bt_responses = self.multi_bt.get_responses(backtranslated_prompts)
        result["bt_responses"] = bt_responses

        # 恢复原始设置
        self.multi_bt.num_perspectives = original_n

        # ===== Stage 4: Intent Consistency Verification =====
        verification = self.verifier.verify(
            original_prompt=input_prompt,
            backtranslated_prompts=backtranslated_prompts,
            bt_responses=bt_responses,
        )
        result["verification_result"] = verification
        result["decision"] = verification["decision"]

        return result
