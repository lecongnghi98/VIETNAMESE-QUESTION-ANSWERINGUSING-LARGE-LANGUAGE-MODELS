"""Tiền xử lý và định dạng dữ liệu QA tiếng Việt theo chuẩn Instruction."""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

from datasets import Dataset

from src.templates import ALPACA_TRAIN_TEMPLATE, CHATML_TRAIN_TEMPLATE


@dataclass
class QASample:
    """Một mẫu hỏi-đáp đã chuẩn hóa."""

    question: str
    context: str
    answer: str
    answer_start: Optional[int] = None
    metadata: Optional[dict] = field(default_factory=dict)


class DataPreprocessor:
    """Tiền xử lý và định dạng dữ liệu QA tiếng Việt theo chuẩn Instruction."""

    def normalize_vietnamese(self, text: str) -> str:
        """Chuẩn hóa Unicode NFC và loại bỏ khoảng trắng thừa."""
        text = unicodedata.normalize("NFC", text)
        text = " ".join(text.split())
        return text.strip()

    def remove_noise(self, text: str) -> str:
        """Loại bỏ HTML tags và ký tự điều khiển."""
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        return text

    def segment_words(self, text: str, use_segmenter: bool = True) -> str:
        """Tách từ tiếng Việt sử dụng underthesea (tùy chọn)."""
        if not use_segmenter:
            return text
        from underthesea import word_tokenize

        return word_tokenize(text, format="text")

    def format_alpaca(self, sample: QASample) -> str:
        """Chuyển đổi mẫu QA sang định dạng Alpaca instruction."""
        return ALPACA_TRAIN_TEMPLATE.format(
            context=sample.context,
            question=sample.question,
            answer=sample.answer,
        )

    def format_chatml(self, sample: QASample) -> str:
        """Chuyển đổi mẫu QA sang định dạng ChatML."""
        return CHATML_TRAIN_TEMPLATE.format(
            context=sample.context,
            question=sample.question,
            answer=sample.answer,
        )

    def prepare_instruction_dataset(
        self, dataset: Dataset, template_format: str = "alpaca"
    ) -> Dataset:
        """Chuyển đổi toàn bộ dataset sang định dạng instruction.

        Lọc mẫu rỗng và thêm trường instruction_text.
        Raise ValueError nếu template_format không hợp lệ.
        """
        if template_format not in ("alpaca", "chatml"):
            raise ValueError(
                f"instruction_format chỉ nhận giá trị 'alpaca' hoặc 'chatml', "
                f"nhận được: {template_format!r}"
            )

        # Lọc mẫu có trường rỗng
        filtered = dataset.filter(
            lambda x: bool(x.get("context", "").strip())
            and bool(x.get("question", "").strip())
            and bool(x.get("answer", "").strip())
        )

        def add_instruction(sample: dict) -> dict:
            qa = QASample(
                question=sample["question"],
                context=sample["context"],
                answer=sample["answer"],
            )
            if template_format == "alpaca":
                sample["instruction_text"] = self.format_alpaca(qa)
            else:
                sample["instruction_text"] = self.format_chatml(qa)
            return sample

        return filtered.map(add_instruction)

    def create_splits(
        self,
        dataset: Dataset,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ) -> tuple[Dataset, Dataset, Dataset]:
        """Chia dataset thành train/val/test.

        Validate val_ratio + test_ratio < 1.0.
        Sử dụng seed=42 cho reproducibility.
        """
        if val_ratio + test_ratio >= 1.0:
            raise ValueError(
                f"val_ratio + test_ratio phải < 1.0, "
                f"nhận được: {val_ratio} + {test_ratio} = {val_ratio + test_ratio}"
            )

        # Chia test trước
        split1 = dataset.train_test_split(test_size=test_ratio, seed=42)
        test_dataset = split1["test"]

        # Chia val từ phần còn lại
        remaining_val_ratio = val_ratio / (1 - test_ratio)
        split2 = split1["train"].train_test_split(
            test_size=remaining_val_ratio, seed=42
        )
        train_dataset = split2["train"]
        val_dataset = split2["test"]

        return train_dataset, val_dataset, test_dataset
