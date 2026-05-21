from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ProjectConfig:
    """Cấu hình toàn bộ dự án."""

    # Mô hình
    base_model_name: str = "Viet-Mistral/Vistral-7B-Chat"
    tokenizer_name: Optional[str] = None

    # Dữ liệu
    dataset_names: list[str] = field(default_factory=lambda: ["uitnlp/ViQuAD"])
    dataset_paths: Optional[list[str]] = None
    max_seq_length: int = 2048
    val_split_ratio: float = 0.1
    test_split_ratio: float = 0.1
    use_word_segmentation: bool = False
    instruction_format: str = "alpaca"  # "alpaca" hoặc "chatml"

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = None

    # Huấn luyện
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    fp16: bool = True
    bf16: bool = False
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100

    # Quantization
    quantization: Optional[str] = "4bit"  # "4bit", "8bit", None

    # FPT Cloud API (Baseline & Inference)
    fpt_api_base_url: str = "https://mkp-api.fptcloud.com"
    fpt_api_models: list[str] = field(default_factory=lambda: [
        "Llama-3.3-70B-Instruct",
        "gemma-3-27b-it",
        "Qwen3-32B",
    ])
    fpt_api_max_tokens: int = 512
    fpt_api_temperature: float = 0.1
    enable_api_baseline: bool = True

    # Đường dẫn
    output_dir: str = "./outputs"
    logging_dir: str = "./logs"
    eda_report_dir: str = "./reports/eda"
    eval_report_dir: str = "./reports/eval"

    def validate_config(self) -> None:
        """Xác thực tất cả giá trị cấu hình. Raise ValueError nếu không hợp lệ."""
        # lora_r ∈ [4, 64]
        if not isinstance(self.lora_r, int) or not (4 <= self.lora_r <= 64):
            raise ValueError(
                f"lora_r phải là số nguyên trong khoảng [4, 64], nhận được: {self.lora_r}"
            )

        # learning_rate ∈ (0, 1)
        if not (0 < self.learning_rate < 1):
            raise ValueError(
                f"learning_rate phải nằm trong khoảng (0, 1), nhận được: {self.learning_rate}"
            )

        # quantization ∈ {"4bit", "8bit", None}
        if self.quantization not in ("4bit", "8bit", None):
            raise ValueError(
                f"quantization chỉ nhận giá trị '4bit', '8bit', hoặc None, "
                f"nhận được: {self.quantization!r}"
            )

        # instruction_format ∈ {"alpaca", "chatml"}
        if self.instruction_format not in ("alpaca", "chatml"):
            raise ValueError(
                f"instruction_format chỉ nhận giá trị 'alpaca' hoặc 'chatml', "
                f"nhận được: {self.instruction_format!r}"
            )

        # fpt_api_temperature ∈ [0, 2]
        if not (0 <= self.fpt_api_temperature <= 2):
            raise ValueError(
                f"fpt_api_temperature phải nằm trong khoảng [0, 2], "
                f"nhận được: {self.fpt_api_temperature}"
            )

        # val_split_ratio + test_split_ratio < 1.0
        if self.val_split_ratio + self.test_split_ratio >= 1.0:
            raise ValueError(
                f"val_split_ratio + test_split_ratio phải < 1.0, "
                f"nhận được: {self.val_split_ratio} + {self.test_split_ratio} = "
                f"{self.val_split_ratio + self.test_split_ratio}"
            )
