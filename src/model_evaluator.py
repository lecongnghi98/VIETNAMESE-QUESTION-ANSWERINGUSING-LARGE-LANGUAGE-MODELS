"""ModelEvaluator — Đánh giá và So sánh kết quả mô hình QA tiếng Việt."""

import json
import logging
import os
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Kết quả đánh giá trên một dataset."""

    dataset_name: str
    exact_match: float
    f1_score: float
    rouge_l: float
    bert_score: float
    num_samples: int
    model_source: str = "local"  # "local" hoặc "api"
    model_name: Optional[str] = None
    per_sample_results: Optional[list[dict]] = None


@dataclass
class ComparisonReport:
    """Báo cáo so sánh kết quả giữa các dataset và mô hình."""

    results: list[EvalResult]
    summary: dict


class ModelEvaluator:
    """Đánh giá và so sánh kết quả mô hình QA tiếng Việt."""

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Chuẩn hóa text trước khi so sánh."""
        text = unicodedata.normalize("NFC", text)
        return text.strip().lower()

    def compute_exact_match(
        self, predictions: list[str], references: list[str]
    ) -> float:
        """Tính Exact Match score."""
        assert len(predictions) == len(references), (
            f"predictions ({len(predictions)}) và references ({len(references)}) "
            f"phải có cùng độ dài"
        )
        if not predictions:
            return 0.0
        matches = sum(
            1
            for p, r in zip(predictions, references)
            if self._normalize_text(p) == self._normalize_text(r)
        )
        return matches / len(predictions)

    def compute_f1(
        self, predictions: list[str], references: list[str]
    ) -> float:
        """Tính F1 score dựa trên token overlap."""
        assert len(predictions) == len(references), (
            f"predictions ({len(predictions)}) và references ({len(references)}) "
            f"phải có cùng độ dài"
        )
        if not predictions:
            return 0.0

        f1_scores = []
        for pred, ref in zip(predictions, references):
            pred_tokens = self._normalize_text(pred).split()
            ref_tokens = self._normalize_text(ref).split()

            if not ref_tokens and not pred_tokens:
                f1_scores.append(1.0)
                continue
            if not ref_tokens or not pred_tokens:
                f1_scores.append(0.0)
                continue

            common = Counter(pred_tokens) & Counter(ref_tokens)
            num_common = sum(common.values())

            if num_common == 0:
                f1_scores.append(0.0)
                continue

            precision = num_common / len(pred_tokens)
            recall = num_common / len(ref_tokens)
            f1 = 2 * precision * recall / (precision + recall)
            f1_scores.append(f1)

        return float(np.mean(f1_scores))

    def compute_rouge_l(
        self, predictions: list[str], references: list[str]
    ) -> float:
        """Tính ROUGE-L score dựa trên LCS."""
        assert len(predictions) == len(references), (
            f"predictions ({len(predictions)}) và references ({len(references)}) "
            f"phải có cùng độ dài"
        )
        if not predictions:
            return 0.0

        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        scores = []
        for pred, ref in zip(predictions, references):
            result = scorer.score(
                self._normalize_text(ref), self._normalize_text(pred)
            )
            scores.append(result["rougeL"].fmeasure)

        return float(np.mean(scores))

    def compute_bert_score(
        self, predictions: list[str], references: list[str]
    ) -> float:
        """Tính BERTScore F1 với mô hình multilingual.

        Trả về -1 nếu BERTScore model không khả dụng.
        """
        assert len(predictions) == len(references), (
            f"predictions ({len(predictions)}) và references ({len(references)}) "
            f"phải có cùng độ dài"
        )
        if not predictions:
            return 0.0

        try:
            from bert_score import score as bert_score_fn

            P, R, F1 = bert_score_fn(
                predictions,
                references,
                model_type="bert-base-multilingual-cased",
                lang="vi",
                verbose=False,
            )
            return float(F1.mean().item())
        except Exception as e:
            logger.warning(f"BERTScore không khả dụng: {e}. Trả về -1.")
            return -1.0

    def evaluate_on_dataset(
        self,
        model: PeftModel,
        tokenizer: AutoTokenizer,
        test_dataset: Dataset,
        dataset_name: str,
        max_new_tokens: int = 256,
    ) -> EvalResult:
        """Đánh giá mô hình local trên một tập test cụ thể."""
        from src.templates import ALPACA_INFERENCE_TEMPLATE

        predictions = []
        references = []
        per_sample = []

        model.eval()
        device = next(model.parameters()).device

        for sample in test_dataset:
            context = sample.get("context", "")
            question = sample.get("question", "")
            reference = sample.get("answer", "")

            prompt = ALPACA_INFERENCE_TEMPLATE.format(
                context=context, question=question
            )
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                )

            generated = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()

            # Chuẩn hóa Unicode NFC
            generated = unicodedata.normalize("NFC", generated)
            reference = unicodedata.normalize("NFC", reference)

            predictions.append(generated)
            references.append(reference)
            per_sample.append({
                "question": question,
                "reference": reference,
                "prediction": generated,
            })

        em = self.compute_exact_match(predictions, references)
        f1 = self.compute_f1(predictions, references)
        rouge_l = self.compute_rouge_l(predictions, references)
        bert_s = self.compute_bert_score(predictions, references)

        return EvalResult(
            dataset_name=dataset_name,
            exact_match=em,
            f1_score=f1,
            rouge_l=rouge_l,
            bert_score=bert_s,
            num_samples=len(predictions),
            model_source="local",
            per_sample_results=per_sample,
        )

    def evaluate_api_model(
        self,
        api_client,
        test_dataset: Dataset,
        dataset_name: str,
        max_tokens: int = 512,
    ) -> EvalResult:
        """Đánh giá mô hình FPT Cloud API trên tập test (baseline)."""
        samples = [
            {"context": s["context"], "question": s["question"]}
            for s in test_dataset
        ]
        references = [s["answer"] for s in test_dataset]

        results = api_client.batch_inference(samples, max_tokens=max_tokens)
        predictions = [r.predicted_answer for r in results]

        # Chuẩn hóa Unicode NFC
        predictions = [unicodedata.normalize("NFC", p) for p in predictions]
        references = [unicodedata.normalize("NFC", r) for r in references]

        em = self.compute_exact_match(predictions, references)
        f1 = self.compute_f1(predictions, references)
        rouge_l = self.compute_rouge_l(predictions, references)
        bert_s = self.compute_bert_score(predictions, references)

        per_sample = [
            {
                "question": s["question"],
                "reference": ref,
                "prediction": pred,
            }
            for s, ref, pred in zip(samples, references, predictions)
        ]

        return EvalResult(
            dataset_name=dataset_name,
            exact_match=em,
            f1_score=f1,
            rouge_l=rouge_l,
            bert_score=bert_s,
            num_samples=len(predictions),
            model_source="api",
            model_name=api_client.model_name,
            per_sample_results=per_sample,
        )

    def compare_results(self, results: list[EvalResult]) -> ComparisonReport:
        """So sánh kết quả đánh giá giữa các dataset và mô hình."""
        if not results:
            return ComparisonReport(results=[], summary={})

        local_results = [r for r in results if r.model_source == "local"]
        api_results = [r for r in results if r.model_source == "api"]

        summary = {
            "total_evaluations": len(results),
            "avg_em": float(np.mean([r.exact_match for r in results])),
            "avg_f1": float(np.mean([r.f1_score for r in results])),
            "avg_rouge_l": float(np.mean([r.rouge_l for r in results])),
            "avg_bert_score": float(
                np.mean([r.bert_score for r in results if r.bert_score >= 0])
            ) if any(r.bert_score >= 0 for r in results) else -1.0,
        }

        if local_results:
            summary["local_avg_f1"] = float(
                np.mean([r.f1_score for r in local_results])
            )
            summary["local_avg_em"] = float(
                np.mean([r.exact_match for r in local_results])
            )

        if api_results:
            summary["api_avg_f1"] = float(
                np.mean([r.f1_score for r in api_results])
            )
            summary["api_avg_em"] = float(
                np.mean([r.exact_match for r in api_results])
            )

        return ComparisonReport(results=results, summary=summary)

    def generate_report(
        self, comparison: ComparisonReport, output_path: str
    ) -> None:
        """Xuất báo cáo đánh giá và so sánh chi tiết dạng JSON."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        report_data = {
            "summary": comparison.summary,
            "results": [],
        }

        for r in comparison.results:
            entry = {
                "dataset_name": r.dataset_name,
                "model_source": r.model_source,
                "model_name": r.model_name,
                "num_samples": r.num_samples,
                "exact_match": r.exact_match,
                "f1_score": r.f1_score,
                "rouge_l": r.rouge_l,
                "bert_score": r.bert_score,
            }
            report_data["results"].append(entry)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Đã xuất báo cáo đánh giá tại: {output_path}")
