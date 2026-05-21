"""DataExplorer — Thu thập và Phân tích dữ liệu (EDA) cho QA tiếng Việt."""

import json
import os
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class EDAReport:
    """Báo cáo phân tích khám phá dữ liệu."""

    total_samples: int
    avg_context_length: float
    avg_question_length: float
    avg_answer_length: float
    context_length_distribution: dict  # {min, max, median, std}
    question_length_distribution: dict
    answer_length_distribution: dict
    missing_fields_count: dict  # {"context": n, "question": n, "answer": n}
    duplicate_count: int
    language_quality_issues: int  # Số mẫu có vấn đề Unicode
    dataset_source: str


class DataExplorer:
    """Thu thập và phân tích khám phá dữ liệu QA tiếng Việt."""

    def load_dataset(self, source: str, split: str = "train") -> "Dataset":
        """Tải dữ liệu từ nguồn (HuggingFace Hub hoặc file cục bộ CSV/JSON).

        Args:
            source: Tên dataset trên HuggingFace Hub hoặc đường dẫn file cục bộ.
            split: Split cần tải (mặc định "train").

        Returns:
            datasets.Dataset chứa dữ liệu đã tải.
        """
        from datasets import Dataset as HFDataset
        from datasets import load_dataset as hf_load_dataset

        # Kiểm tra nếu là file cục bộ
        if os.path.isfile(source):
            ext = os.path.splitext(source)[1].lower()
            if ext == ".csv":
                df = pd.read_csv(source)
                return HFDataset.from_pandas(df)
            elif ext == ".json":
                with open(source, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return HFDataset.from_list(data)
                return HFDataset.from_dict(data)
            else:
                raise ValueError(
                    f"Định dạng file không được hỗ trợ: {ext}. "
                    f"Chỉ hỗ trợ CSV và JSON."
                )

        # Tải từ HuggingFace Hub
        ds = hf_load_dataset(source, split=split)
        return ds

    def _compute_length_distribution(self, lengths: list[int]) -> dict:
        """Tính phân phối độ dài cho một danh sách giá trị."""
        if not lengths:
            return {"min": 0, "max": 0, "median": 0.0, "std": 0.0}
        arr = np.array(lengths, dtype=float)
        return {
            "min": int(np.min(arr)),
            "max": int(np.max(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
        }

    def compute_statistics(self, dataset, source: str = "unknown") -> EDAReport:
        """Tính toán thống kê mô tả cho dataset.

        Args:
            dataset: datasets.Dataset chứa các trường context, question, answer.
            source: Tên nguồn dữ liệu.

        Returns:
            EDAReport chứa đầy đủ thống kê mô tả.
        """
        total = len(dataset)

        contexts = [str(row.get("context", "") or "") for row in dataset]
        questions = [str(row.get("question", "") or "") for row in dataset]
        answers = [str(row.get("answer", "") or "") for row in dataset]

        context_lengths = [len(c.split()) for c in contexts]
        question_lengths = [len(q.split()) for q in questions]
        answer_lengths = [len(a.split()) for a in answers]

        # Kiểm tra chất lượng dữ liệu
        quality = self.check_data_quality(dataset)

        return EDAReport(
            total_samples=total,
            avg_context_length=float(np.mean(context_lengths)) if context_lengths else 0.0,
            avg_question_length=float(np.mean(question_lengths)) if question_lengths else 0.0,
            avg_answer_length=float(np.mean(answer_lengths)) if answer_lengths else 0.0,
            context_length_distribution=self._compute_length_distribution(context_lengths),
            question_length_distribution=self._compute_length_distribution(question_lengths),
            answer_length_distribution=self._compute_length_distribution(answer_lengths),
            missing_fields_count=quality["missing_fields_count"],
            duplicate_count=quality["duplicate_count"],
            language_quality_issues=quality["language_quality_issues"],
            dataset_source=source,
        )

    def check_data_quality(self, dataset) -> dict:
        """Kiểm tra chất lượng dữ liệu: missing values, duplicates, Unicode issues.

        Args:
            dataset: datasets.Dataset chứa các trường context, question, answer.

        Returns:
            dict chứa missing_fields_count, duplicate_count, language_quality_issues.
        """
        contexts = [str(row.get("context", "") or "") for row in dataset]
        questions = [str(row.get("question", "") or "") for row in dataset]
        answers = [str(row.get("answer", "") or "") for row in dataset]

        # Missing values
        missing_context = sum(1 for c in contexts if not c.strip())
        missing_question = sum(1 for q in questions if not q.strip())
        missing_answer = sum(1 for a in answers if not a.strip())

        # Duplicates (dựa trên question)
        unique_questions = set(questions)
        duplicate_count = len(questions) - len(unique_questions)

        # Vấn đề Unicode tiếng Việt
        unicode_issues = 0
        for text in contexts + questions + answers:
            if text and unicodedata.normalize("NFC", text) != text:
                unicode_issues += 1

        return {
            "missing_fields_count": {
                "context": missing_context,
                "question": missing_question,
                "answer": missing_answer,
            },
            "duplicate_count": duplicate_count,
            "language_quality_issues": unicode_issues,
        }

    def compare_datasets(self, dataset1, dataset2) -> dict:
        """So sánh thống kê giữa 2 dataset.

        Args:
            dataset1: Dataset thứ nhất.
            dataset2: Dataset thứ hai.

        Returns:
            dict chứa thống kê so sánh giữa 2 dataset.
        """
        report1 = self.compute_statistics(dataset1, source="dataset1")
        report2 = self.compute_statistics(dataset2, source="dataset2")

        return {
            "dataset1": {
                "source": report1.dataset_source,
                "total_samples": report1.total_samples,
                "avg_context_length": report1.avg_context_length,
                "avg_question_length": report1.avg_question_length,
                "avg_answer_length": report1.avg_answer_length,
                "context_length_distribution": report1.context_length_distribution,
                "question_length_distribution": report1.question_length_distribution,
                "answer_length_distribution": report1.answer_length_distribution,
                "missing_fields_count": report1.missing_fields_count,
                "duplicate_count": report1.duplicate_count,
                "language_quality_issues": report1.language_quality_issues,
            },
            "dataset2": {
                "source": report2.dataset_source,
                "total_samples": report2.total_samples,
                "avg_context_length": report2.avg_context_length,
                "avg_question_length": report2.avg_question_length,
                "avg_answer_length": report2.avg_answer_length,
                "context_length_distribution": report2.context_length_distribution,
                "question_length_distribution": report2.question_length_distribution,
                "answer_length_distribution": report2.answer_length_distribution,
                "missing_fields_count": report2.missing_fields_count,
                "duplicate_count": report2.duplicate_count,
                "language_quality_issues": report2.language_quality_issues,
            },
            "comparison": {
                "total_samples_diff": report1.total_samples - report2.total_samples,
                "avg_context_length_diff": report1.avg_context_length - report2.avg_context_length,
                "avg_question_length_diff": report1.avg_question_length - report2.avg_question_length,
                "avg_answer_length_diff": report1.avg_answer_length - report2.avg_answer_length,
            },
        }

    def plot_distributions(self, dataset, output_dir: str) -> None:
        """Vẽ biểu đồ phân phối độ dài context, question, answer.

        Args:
            dataset: datasets.Dataset chứa các trường context, question, answer.
            output_dir: Thư mục lưu biểu đồ.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(output_dir, exist_ok=True)

        contexts = [str(row.get("context", "") or "") for row in dataset]
        questions = [str(row.get("question", "") or "") for row in dataset]
        answers = [str(row.get("answer", "") or "") for row in dataset]

        fields = {
            "context": [len(c.split()) for c in contexts],
            "question": [len(q.split()) for q in questions],
            "answer": [len(a.split()) for a in answers],
        }

        for name, lengths in fields.items():
            if not lengths:
                continue
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(lengths, bins=50, edgecolor="black", alpha=0.7)
            ax.set_title(f"Phân phối độ dài {name} (số từ)")
            ax.set_xlabel("Số từ")
            ax.set_ylabel("Số mẫu")
            ax.axvline(
                np.mean(lengths),
                color="red",
                linestyle="--",
                label=f"Trung bình: {np.mean(lengths):.1f}",
            )
            ax.axvline(
                np.median(lengths),
                color="green",
                linestyle="--",
                label=f"Trung vị: {np.median(lengths):.1f}",
            )
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, f"{name}_length_distribution.png"), dpi=150)
            plt.close(fig)

    def generate_eda_report(
        self, datasets: dict, output_path: str
    ) -> list[EDAReport]:
        """Tạo báo cáo EDA tổng hợp cho tất cả dataset.

        Args:
            datasets: dict mapping tên dataset -> Dataset object.
            output_path: Đường dẫn thư mục lưu báo cáo.

        Returns:
            Danh sách EDAReport cho mỗi dataset.
        """
        os.makedirs(output_path, exist_ok=True)
        reports = []

        for name, dataset in datasets.items():
            report = self.compute_statistics(dataset, source=name)
            reports.append(report)

            # Vẽ biểu đồ cho từng dataset
            plot_dir = os.path.join(output_path, f"{name}_plots")
            self.plot_distributions(dataset, plot_dir)

        # Lưu báo cáo tổng hợp dạng JSON
        report_data = []
        for r in reports:
            report_data.append({
                "dataset_source": r.dataset_source,
                "total_samples": r.total_samples,
                "avg_context_length": r.avg_context_length,
                "avg_question_length": r.avg_question_length,
                "avg_answer_length": r.avg_answer_length,
                "context_length_distribution": r.context_length_distribution,
                "question_length_distribution": r.question_length_distribution,
                "answer_length_distribution": r.answer_length_distribution,
                "missing_fields_count": r.missing_fields_count,
                "duplicate_count": r.duplicate_count,
                "language_quality_issues": r.language_quality_issues,
            })

        report_file = os.path.join(output_path, "eda_report.json")
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)

        return reports
