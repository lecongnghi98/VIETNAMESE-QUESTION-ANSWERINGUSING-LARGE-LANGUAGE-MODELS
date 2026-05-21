"""Tests cho DataPreprocessor và QASample."""

import pytest
from datasets import Dataset

from src.data_preprocessor import DataPreprocessor, QASample


@pytest.fixture
def preprocessor():
    return DataPreprocessor()


@pytest.fixture
def sample_dataset():
    """Dataset mẫu với dữ liệu QA tiếng Việt."""
    return Dataset.from_dict(
        {
            "context": [
                "Hà Nội là thủ đô của Việt Nam.",
                "Python là ngôn ngữ lập trình phổ biến.",
                "Trái đất quay quanh mặt trời.",
                "Đây là context hợp lệ.",
                "Một context khác nữa.",
            ],
            "question": [
                "Thủ đô của Việt Nam là gì?",
                "Python là gì?",
                "Trái đất quay quanh gì?",
                "Câu hỏi gì?",
                "Hỏi gì nữa?",
            ],
            "answer": [
                "Hà Nội",
                "Ngôn ngữ lập trình",
                "Mặt trời",
                "Trả lời",
                "Đáp án",
            ],
        }
    )


@pytest.fixture
def sample_dataset_with_empty():
    """Dataset mẫu có mẫu rỗng."""
    return Dataset.from_dict(
        {
            "context": ["Context hợp lệ.", "", "Context khác."],
            "question": ["Câu hỏi?", "Câu hỏi 2?", ""],
            "answer": ["Trả lời", "Trả lời 2", "Trả lời 3"],
        }
    )


class TestQASample:
    def test_dataclass_fields(self):
        sample = QASample(
            question="Câu hỏi?",
            context="Ngữ cảnh",
            answer="Trả lời",
            answer_start=0,
            metadata={"source": "test"},
        )
        assert sample.question == "Câu hỏi?"
        assert sample.context == "Ngữ cảnh"
        assert sample.answer == "Trả lời"
        assert sample.answer_start == 0
        assert sample.metadata == {"source": "test"}

    def test_defaults(self):
        sample = QASample(question="Q", context="C", answer="A")
        assert sample.answer_start is None
        assert sample.metadata == {}


class TestNormalizeVietnamese:
    def test_nfc_normalization(self, preprocessor):
        # Dạng NFD (tổ hợp) -> NFC (hợp nhất)
        nfd_text = "Việt Nam"  # có thể ở dạng NFD
        result = preprocessor.normalize_vietnamese(nfd_text)
        import unicodedata
        assert unicodedata.is_normalized("NFC", result)

    def test_remove_extra_whitespace(self, preprocessor):
        text = "  Hà   Nội   là   thủ   đô  "
        result = preprocessor.normalize_vietnamese(text)
        assert result == "Hà Nội là thủ đô"

    def test_idempotent(self, preprocessor):
        text = "Việt Nam là đất nước xinh đẹp"
        once = preprocessor.normalize_vietnamese(text)
        twice = preprocessor.normalize_vietnamese(once)
        assert once == twice

    def test_empty_string(self, preprocessor):
        assert preprocessor.normalize_vietnamese("") == ""
        assert preprocessor.normalize_vietnamese("   ") == ""


class TestRemoveNoise:
    def test_remove_html_tags(self, preprocessor):
        text = "<p>Xin chào</p> <b>Việt Nam</b>"
        result = preprocessor.remove_noise(text)
        assert "<p>" not in result
        assert "<b>" not in result
        assert "Xin chào" in result
        assert "Việt Nam" in result

    def test_remove_control_characters(self, preprocessor):
        text = "Hello\x00World\x07Test\x1f"
        result = preprocessor.remove_noise(text)
        assert "\x00" not in result
        assert "\x07" not in result
        assert "\x1f" not in result
        assert "HelloWorldTest" in result

    def test_preserves_normal_text(self, preprocessor):
        text = "Đây là văn bản bình thường tiếng Việt."
        assert preprocessor.remove_noise(text) == text

    def test_preserves_newlines_and_tabs(self, preprocessor):
        text = "Dòng 1\nDòng 2\tTab"
        result = preprocessor.remove_noise(text)
        assert "\n" in result
        assert "\t" in result


class TestSegmentWords:
    def test_no_segmentation(self, preprocessor):
        text = "Hà Nội là thủ đô"
        result = preprocessor.segment_words(text, use_segmenter=False)
        assert result == text

    def test_with_segmentation(self, preprocessor):
        text = "Hà Nội là thủ đô của Việt Nam"
        result = preprocessor.segment_words(text, use_segmenter=True)
        assert isinstance(result, str)
        assert len(result) > 0


class TestFormatAlpaca:
    def test_contains_markers(self, preprocessor):
        sample = QASample(question="Câu hỏi?", context="Ngữ cảnh", answer="Trả lời")
        result = preprocessor.format_alpaca(sample)
        assert "### Instruction:" in result
        assert "### Input:" in result
        assert "### Response:" in result

    def test_contains_content(self, preprocessor):
        sample = QASample(
            question="Python là gì?",
            context="Python là ngôn ngữ lập trình.",
            answer="Ngôn ngữ lập trình",
        )
        result = preprocessor.format_alpaca(sample)
        assert "Python là gì?" in result
        assert "Python là ngôn ngữ lập trình." in result
        assert "Ngôn ngữ lập trình" in result


class TestFormatChatML:
    def test_contains_tokens(self, preprocessor):
        sample = QASample(question="Câu hỏi?", context="Ngữ cảnh", answer="Trả lời")
        result = preprocessor.format_chatml(sample)
        assert "<|im_start|>system" in result
        assert "<|im_start|>user" in result
        assert "<|im_start|>assistant" in result

    def test_contains_content(self, preprocessor):
        sample = QASample(
            question="Thủ đô?",
            context="Hà Nội là thủ đô.",
            answer="Hà Nội",
        )
        result = preprocessor.format_chatml(sample)
        assert "Thủ đô?" in result
        assert "Hà Nội là thủ đô." in result
        assert "Hà Nội" in result


class TestPrepareInstructionDataset:
    def test_alpaca_format(self, preprocessor, sample_dataset):
        result = preprocessor.prepare_instruction_dataset(sample_dataset, "alpaca")
        assert "instruction_text" in result.column_names
        assert len(result) == len(sample_dataset)
        assert "### Instruction:" in result[0]["instruction_text"]

    def test_chatml_format(self, preprocessor, sample_dataset):
        result = preprocessor.prepare_instruction_dataset(sample_dataset, "chatml")
        assert "instruction_text" in result.column_names
        assert "<|im_start|>system" in result[0]["instruction_text"]

    def test_invalid_format_raises(self, preprocessor, sample_dataset):
        with pytest.raises(ValueError, match="instruction_format"):
            preprocessor.prepare_instruction_dataset(sample_dataset, "invalid")

    def test_filters_empty_samples(self, preprocessor, sample_dataset_with_empty):
        result = preprocessor.prepare_instruction_dataset(
            sample_dataset_with_empty, "alpaca"
        )
        # Only the first sample has all non-empty fields
        assert len(result) == 1
        assert result[0]["context"] == "Context hợp lệ."


class TestCreateSplits:
    def test_basic_split(self, preprocessor, sample_dataset):
        train, val, test = preprocessor.create_splits(sample_dataset, 0.2, 0.2)
        total = len(train) + len(val) + len(test)
        assert total == len(sample_dataset)

    def test_no_overlap(self, preprocessor, sample_dataset):
        train, val, test = preprocessor.create_splits(sample_dataset, 0.2, 0.2)
        train_ctx = set(train["context"])
        val_ctx = set(val["context"])
        test_ctx = set(test["context"])
        # No overlap between splits
        assert len(train_ctx & val_ctx) == 0
        assert len(train_ctx & test_ctx) == 0
        assert len(val_ctx & test_ctx) == 0

    def test_invalid_ratios_raises(self, preprocessor, sample_dataset):
        with pytest.raises(ValueError, match="val_ratio \\+ test_ratio"):
            preprocessor.create_splits(sample_dataset, 0.5, 0.5)
        with pytest.raises(ValueError, match="val_ratio \\+ test_ratio"):
            preprocessor.create_splits(sample_dataset, 0.6, 0.5)

    def test_reproducibility(self, preprocessor, sample_dataset):
        train1, val1, test1 = preprocessor.create_splits(sample_dataset, 0.2, 0.2)
        train2, val2, test2 = preprocessor.create_splits(sample_dataset, 0.2, 0.2)
        assert train1["context"] == train2["context"]
        assert val1["context"] == val2["context"]
        assert test1["context"] == test2["context"]
