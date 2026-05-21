"""Tests cho DataExplorer và EDAReport."""

import json
import os
import tempfile

import pytest
from datasets import Dataset

from src.data_explorer import DataExplorer, EDAReport


@pytest.fixture
def explorer():
    return DataExplorer()


@pytest.fixture
def sample_dataset():
    """Dataset mẫu với dữ liệu QA tiếng Việt."""
    return Dataset.from_dict(
        {
            "context": [
                "Hà Nội là thủ đô của Việt Nam.",
                "Python là ngôn ngữ lập trình phổ biến.",
                "Trái đất quay quanh mặt trời.",
            ],
            "question": [
                "Thủ đô của Việt Nam là gì?",
                "Python là gì?",
                "Trái đất quay quanh gì?",
            ],
            "answer": [
                "Hà Nội",
                "Ngôn ngữ lập trình",
                "Mặt trời",
            ],
        }
    )


@pytest.fixture
def sample_dataset_with_issues():
    """Dataset mẫu có vấn đề chất lượng."""
    return Dataset.from_dict(
        {
            "context": [
                "Đây là context hợp lệ.",
                "",
                "Context khác.",
                "Context trùng câu hỏi.",
            ],
            "question": [
                "Câu hỏi 1?",
                "Câu hỏi 2?",
                "",
                "Câu hỏi 1?",  # duplicate
            ],
            "answer": [
                "Trả lời 1",
                "Trả lời 2",
                "Trả lời 3",
                "",
            ],
        }
    )


class TestEDAReport:
    def test_dataclass_fields(self):
        report = EDAReport(
            total_samples=100,
            avg_context_length=50.0,
            avg_question_length=10.0,
            avg_answer_length=5.0,
            context_length_distribution={"min": 1, "max": 200, "median": 50.0, "std": 30.0},
            question_length_distribution={"min": 2, "max": 30, "median": 10.0, "std": 5.0},
            answer_length_distribution={"min": 1, "max": 50, "median": 5.0, "std": 3.0},
            missing_fields_count={"context": 0, "question": 0, "answer": 0},
            duplicate_count=5,
            language_quality_issues=2,
            dataset_source="test",
        )
        assert report.total_samples == 100
        assert report.dataset_source == "test"
        assert report.duplicate_count == 5


class TestLoadDataset:
    def test_load_csv(self, explorer, tmp_path):
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(
            "context,question,answer\n"
            "ctx1,q1,a1\n"
            "ctx2,q2,a2\n",
            encoding="utf-8",
        )
        ds = explorer.load_dataset(str(csv_path))
        assert len(ds) == 2
        assert ds[0]["context"] == "ctx1"

    def test_load_json_list(self, explorer, tmp_path):
        json_path = tmp_path / "test.json"
        data = [
            {"context": "ctx1", "question": "q1", "answer": "a1"},
            {"context": "ctx2", "question": "q2", "answer": "a2"},
        ]
        json_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        ds = explorer.load_dataset(str(json_path))
        assert len(ds) == 2

    def test_load_unsupported_format(self, explorer, tmp_path):
        txt_path = tmp_path / "test.txt"
        txt_path.write_text("hello")
        with pytest.raises(ValueError, match="Định dạng file không được hỗ trợ"):
            explorer.load_dataset(str(txt_path))


class TestComputeStatistics:
    def test_basic_statistics(self, explorer, sample_dataset):
        report = explorer.compute_statistics(sample_dataset, source="test")
        assert report.total_samples == 3
        assert report.avg_context_length > 0
        assert report.avg_question_length > 0
        assert report.avg_answer_length > 0
        assert report.dataset_source == "test"

    def test_distribution_min_le_max(self, explorer, sample_dataset):
        report = explorer.compute_statistics(sample_dataset)
        for dist in [
            report.context_length_distribution,
            report.question_length_distribution,
            report.answer_length_distribution,
        ]:
            assert dist["min"] <= dist["max"]
            assert dist["min"] <= dist["median"] <= dist["max"]
            assert dist["std"] >= 0


class TestCheckDataQuality:
    def test_no_issues(self, explorer, sample_dataset):
        quality = explorer.check_data_quality(sample_dataset)
        assert quality["missing_fields_count"]["context"] == 0
        assert quality["missing_fields_count"]["question"] == 0
        assert quality["missing_fields_count"]["answer"] == 0
        assert quality["duplicate_count"] == 0

    def test_detects_missing_and_duplicates(self, explorer, sample_dataset_with_issues):
        quality = explorer.check_data_quality(sample_dataset_with_issues)
        assert quality["missing_fields_count"]["context"] == 1  # empty context
        assert quality["missing_fields_count"]["question"] == 1  # empty question
        assert quality["missing_fields_count"]["answer"] == 1  # empty answer
        assert quality["duplicate_count"] == 1  # "Câu hỏi 1?" appears twice


class TestCompareDatasets:
    def test_compare_two_datasets(self, explorer, sample_dataset):
        ds2 = Dataset.from_dict(
            {
                "context": ["Ngắn."],
                "question": ["Gì?"],
                "answer": ["Đáp."],
            }
        )
        result = explorer.compare_datasets(sample_dataset, ds2)
        assert "dataset1" in result
        assert "dataset2" in result
        assert "comparison" in result
        assert result["dataset1"]["total_samples"] == 3
        assert result["dataset2"]["total_samples"] == 1


class TestPlotDistributions:
    def test_creates_plot_files(self, explorer, sample_dataset, tmp_path):
        output_dir = str(tmp_path / "plots")
        explorer.plot_distributions(sample_dataset, output_dir)
        assert os.path.isfile(os.path.join(output_dir, "context_length_distribution.png"))
        assert os.path.isfile(os.path.join(output_dir, "question_length_distribution.png"))
        assert os.path.isfile(os.path.join(output_dir, "answer_length_distribution.png"))


class TestGenerateEDAReport:
    def test_generates_report(self, explorer, sample_dataset, tmp_path):
        output_path = str(tmp_path / "eda_output")
        datasets_dict = {"test_ds": sample_dataset}
        reports = explorer.generate_eda_report(datasets_dict, output_path)
        assert len(reports) == 1
        assert reports[0].dataset_source == "test_ds"

        # Kiểm tra file JSON báo cáo
        report_file = os.path.join(output_path, "eda_report.json")
        assert os.path.isfile(report_file)
        with open(report_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["dataset_source"] == "test_ds"

        # Kiểm tra thư mục plots
        plot_dir = os.path.join(output_path, "test_ds_plots")
        assert os.path.isdir(plot_dir)
