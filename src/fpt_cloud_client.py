"""FPTCloudClient — Tích hợp FPT Cloud API cho Inference & Baseline."""

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

FPT_AVAILABLE_MODELS = [
    "Llama-3.3-70B-Instruct",
    "gemma-3-27b-it",
    "llama-3-70b-instruct",
    "Qwen3-32B",
]

FPT_API_BASE_URL = "https://mkp-api.fptcloud.com"


@dataclass
class APIInferenceResult:
    """Kết quả inference từ FPT Cloud API."""

    model_name: str
    question: str
    context: str
    predicted_answer: str
    usage: Optional[dict] = None


class FPTCloudClient:
    """Client tương tác với FPT Cloud API (OpenAI-compatible) cho LLM inference."""

    def __init__(
        self,
        model_name: str = "Llama-3.3-70B-Instruct",
        max_retries: int = 3,
    ):
        """Khởi tạo client với API key từ biến môi trường.

        Raises:
            ValueError: Nếu FPT_API_KEY chưa được thiết lập hoặc model_name không hợp lệ.
        """
        api_key = os.environ.get("FPT_API_KEY")
        if not api_key:
            raise ValueError(
                "FPT_API_KEY chưa được thiết lập. "
                "Vui lòng đặt biến môi trường: export FPT_API_KEY='your-api-key'"
            )
        if model_name not in FPT_AVAILABLE_MODELS:
            raise ValueError(
                f"model_name '{model_name}' không hợp lệ. "
                f"Các mô hình khả dụng: {FPT_AVAILABLE_MODELS}"
            )
        self.client = OpenAI(api_key=api_key, base_url=FPT_API_BASE_URL)
        self.model_name = model_name
        self.max_retries = max_retries

    def _call_api(self, messages: list[dict], max_tokens: int, temperature: float) -> dict:
        """Gọi API với retry logic và xử lý rate limit."""
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return response
            except Exception as e:
                error_str = str(e)
                # Rate limit (429)
                if "429" in error_str or "rate" in error_str.lower():
                    wait = 2 ** (attempt + 1)
                    logger.warning(
                        f"Rate limit, chờ {wait}s trước khi thử lại "
                        f"(lần {attempt + 1}/{self.max_retries})"
                    )
                    time.sleep(wait)
                    continue
                # Lỗi kết nối/timeout
                if attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(
                        f"Lỗi API: {e}. Thử lại sau {wait}s "
                        f"(lần {attempt + 1}/{self.max_retries})"
                    )
                    time.sleep(wait)
                    continue
                raise
        raise RuntimeError(f"Đã thử {self.max_retries} lần nhưng không thành công.")

    def inference(
        self,
        context: str,
        question: str,
        max_tokens: int = 512,
        temperature: float = 0.1,
    ) -> APIInferenceResult:
        """Gửi câu hỏi QA đến FPT Cloud API và nhận câu trả lời."""
        system_prompt = (
            "Bạn là trợ lý AI chuyên trả lời câu hỏi bằng tiếng Việt. "
            "Hãy trả lời dựa trên ngữ cảnh được cung cấp một cách chính xác và ngắn gọn."
        )
        user_prompt = f"Ngữ cảnh: {context}\n\nCâu hỏi: {question}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = self._call_api(messages, max_tokens, temperature)

        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return APIInferenceResult(
            model_name=self.model_name,
            question=question,
            context=context,
            predicted_answer=response.choices[0].message.content.strip(),
            usage=usage,
        )

    def batch_inference(
        self,
        samples: list[dict],
        max_tokens: int = 512,
        temperature: float = 0.1,
    ) -> list[APIInferenceResult]:
        """Thực hiện inference hàng loạt trên danh sách mẫu QA.

        Đảm bảo len(results) == len(samples).
        """
        results = []
        for i, sample in enumerate(samples):
            try:
                result = self.inference(
                    context=sample["context"],
                    question=sample["question"],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Lỗi inference mẫu {i}: {e}")
                results.append(
                    APIInferenceResult(
                        model_name=self.model_name,
                        question=sample.get("question", ""),
                        context=sample.get("context", ""),
                        predicted_answer="",
                        usage=None,
                    )
                )
        assert len(results) == len(samples)
        return results

    def generate_training_data(
        self,
        contexts: list[str],
        num_qa_per_context: int = 3,
        max_tokens: int = 1024,
    ) -> list[dict]:
        """Sử dụng API để sinh dữ liệu huấn luyện bổ sung (data augmentation).

        Bỏ qua context rỗng, tiếp tục xử lý các context còn lại.
        """
        generated_data = []

        for i, context in enumerate(contexts):
            if not context or not context.strip():
                logger.warning(f"Bỏ qua context rỗng tại vị trí {i}")
                continue

            prompt = (
                f"Dựa vào đoạn văn sau, hãy tạo {num_qa_per_context} cặp câu hỏi-trả lời "
                f"bằng tiếng Việt. Trả về dưới dạng JSON array với các trường 'question' và 'answer'.\n\n"
                f"Đoạn văn: {context}"
            )

            try:
                response = self._call_api(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.7,
                )
                raw_output = response.choices[0].message.content.strip()

                # Thử parse JSON
                parsed = None
                try:
                    parsed = json.loads(raw_output)
                except json.JSONDecodeError:
                    # Thử tìm JSON array trong output
                    start = raw_output.find("[")
                    end = raw_output.rfind("]") + 1
                    if start != -1 and end > start:
                        try:
                            parsed = json.loads(raw_output[start:end])
                        except json.JSONDecodeError:
                            pass

                generated_data.append({
                    "context": context,
                    "raw_generated": raw_output,
                    "parsed_qa": parsed,
                })
            except Exception as e:
                logger.error(f"Lỗi sinh dữ liệu cho context {i}: {e}")
                generated_data.append({
                    "context": context,
                    "raw_generated": "",
                    "parsed_qa": None,
                })

        return generated_data
