"""ModelSelector — Lựa chọn mô hình và Cấu hình LoRA/QLoRA."""

import logging
from typing import Optional

import torch
from peft import LoraConfig, PeftModel, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)

# Danh sách mô hình tiếng Việt phổ biến
POPULAR_VIETNAMESE_MODELS = [
    "Viet-Mistral/Vistral-7B-Chat",
    "vinai/PhoGPT-4B-Chat",
    "google/gemma-2b",
    "meta-llama/Llama-2-7b-chat-hf",
]


class ModelSelector:
    """Lựa chọn và cấu hình mô hình LLM cho Instruction Tuning."""

    def load_base_model(
        self,
        model_name: str,
        quantization: Optional[str] = "4bit",
    ) -> AutoModelForCausalLM:
        """Tải mô hình nền với quantization tùy chọn (4-bit/8-bit).

        Args:
            model_name: Tên mô hình trên HuggingFace Hub hoặc đường dẫn cục bộ.
            quantization: "4bit", "8bit", hoặc None.

        Returns:
            AutoModelForCausalLM đã tải.

        Raises:
            ValueError: Nếu quantization không hợp lệ.
            OSError: Nếu mô hình không tồn tại.
        """
        if quantization not in ("4bit", "8bit", None):
            raise ValueError(
                f"quantization chỉ nhận '4bit', '8bit', hoặc None, "
                f"nhận được: {quantization!r}"
            )

        kwargs = {"device_map": "auto", "trust_remote_code": True}

        if quantization == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        elif quantization == "8bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            kwargs["torch_dtype"] = torch.float16

        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        except OSError as e:
            raise OSError(
                f"Không thể tải mô hình '{model_name}'. "
                f"Vui lòng kiểm tra tên mô hình hoặc thử các mô hình tiếng Việt phổ biến: "
                f"{', '.join(POPULAR_VIETNAMESE_MODELS)}"
            ) from e

        model.config.use_cache = False
        logger.info(f"Đã tải mô hình: {model_name} (quantization={quantization})")
        return model

    def load_tokenizer(self, model_name: str) -> AutoTokenizer:
        """Tải tokenizer tương ứng với mô hình.

        Tự động gán pad_token = eos_token nếu thiếu.
        """
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "right"
        logger.info(f"Đã tải tokenizer: {model_name}")
        return tokenizer

    def create_lora_config(
        self,
        r: int = 16,
        alpha: int = 32,
        dropout: float = 0.05,
        target_modules: Optional[list[str]] = None,
    ) -> LoraConfig:
        """Tạo cấu hình LoRA/QLoRA.

        Args:
            r: Rank của LoRA adapter.
            alpha: Scaling factor.
            dropout: Dropout rate.
            target_modules: Danh sách module cần áp dụng LoRA.

        Returns:
            LoraConfig đã cấu hình.
        """
        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

        config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        logger.info(
            f"Đã tạo LoRA config: r={r}, alpha={alpha}, "
            f"dropout={dropout}, modules={target_modules}"
        )
        return config

    def apply_lora(
        self,
        model: AutoModelForCausalLM,
        lora_config: LoraConfig,
    ) -> PeftModel:
        """Áp dụng LoRA adapter lên mô hình nền."""
        model = get_peft_model(model, lora_config)
        trainable, total = self.get_trainable_params_info(model)
        ratio = trainable / total if total > 0 else 0
        logger.info(
            f"Đã áp dụng LoRA: {trainable:,} / {total:,} params "
            f"({ratio:.4%} trainable)"
        )
        return model

    def get_trainable_params_info(self, model) -> tuple[int, int]:
        """Trả về (trainable_params, total_params)."""
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        return trainable, total

    def get_trainable_params_ratio(self, model) -> float:
        """Tính tỷ lệ tham số trainable / tổng tham số.

        Returns:
            Tỷ lệ trainable params (0.0 - 1.0).
        """
        trainable, total = self.get_trainable_params_info(model)
        if total == 0:
            return 0.0
        ratio = trainable / total
        assert ratio < 0.02, (
            f"Tỷ lệ trainable params ({ratio:.4%}) vượt quá 2%. "
            f"Kiểm tra lại cấu hình LoRA."
        )
        return ratio
