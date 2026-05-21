"""InstructionTuner — Huấn luyện với Instruction Tuning + Fine-tuning."""

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from datasets import Dataset
from peft import PeftModel
from transformers import AutoTokenizer, TrainingArguments
from trl import SFTTrainer

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Cấu hình huấn luyện."""

    output_dir: str = "./outputs"
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_seq_length: int = 2048
    warmup_ratio: float = 0.1
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    fp16: bool = True
    bf16: bool = False
    weight_decay: float = 0.01
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: Optional[list[str]] = None
    instruction_format: str = "alpaca"
    save_total_limit: int = 3


class InstructionTuner:
    """Instruction Tuning + Fine-tuning LLM cho QA tiếng Việt."""

    def create_training_args(self, config: TrainingConfig) -> TrainingArguments:
        """Tạo cấu hình huấn luyện từ TrainingConfig."""
        return TrainingArguments(
            output_dir=config.output_dir,
            num_train_epochs=config.num_epochs,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            warmup_ratio=config.warmup_ratio,
            weight_decay=config.weight_decay,
            logging_steps=config.logging_steps,
            save_steps=config.save_steps,
            eval_steps=config.eval_steps,
            eval_strategy="steps",
            save_total_limit=config.save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            fp16=config.fp16,
            bf16=config.bf16,
            report_to="tensorboard",
            logging_dir=os.path.join(config.output_dir, "logs"),
            remove_unused_columns=False,
        )

    def train(
        self,
        model: PeftModel,
        tokenizer: AutoTokenizer,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        config: TrainingConfig,
    ) -> SFTTrainer:
        """Thực hiện Instruction Tuning với SFTTrainer.

        Xử lý CUDA OOM bằng cách ghi log cảnh báo và gợi ý giảm batch_size.
        """
        training_args = self.create_training_args(config)

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            dataset_text_field="instruction_text",
            max_seq_length=config.max_seq_length,
        )

        try:
            trainer.train()
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "CUDA" in str(e):
                logger.error(
                    "CUDA OOM! Gợi ý: giảm batch_size hoặc tăng "
                    "gradient_accumulation_steps. "
                    f"Hiện tại: batch_size={config.batch_size}, "
                    f"grad_accum={config.gradient_accumulation_steps}"
                )
            raise

        logger.info("Huấn luyện hoàn tất.")
        return trainer

    def resume_training(self, trainer: SFTTrainer, checkpoint_path: str) -> None:
        """Tiếp tục huấn luyện từ checkpoint."""
        if not os.path.isdir(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint không tồn tại: {checkpoint_path}"
            )
        logger.info(f"Tiếp tục huấn luyện từ: {checkpoint_path}")
        trainer.train(resume_from_checkpoint=checkpoint_path)

    def save_model(
        self,
        model: PeftModel,
        tokenizer: AutoTokenizer,
        output_dir: str,
    ) -> None:
        """Lưu adapter weights và tokenizer."""
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Đã lưu mô hình và tokenizer tại: {output_dir}")
