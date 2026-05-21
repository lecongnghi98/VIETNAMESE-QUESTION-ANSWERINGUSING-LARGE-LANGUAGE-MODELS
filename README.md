# 🇻🇳 Prompt Engineering cho Hỏi–Đáp tiếng Việt trên LLM

Đánh giá hiệu quả của 8 chiến lược prompt engineering trên các mô hình ngôn ngữ lớn (Llama-3.3-70B, gpt-oss-120b) cho bài toán extractive QA tiếng Việt.

## Kết quả chính

| Mô hình | Chiến lược | Dataset | F1 |
|---|---|---|---|
| Llama-3.3-70B | P1 Extractive | ViNewsQA (1.992 mẫu) | **0.8205** |
| Llama-3.3-70B | P4 CoT | ViSpanExtractQA (2.000 mẫu) | **0.7697** |
| gpt-oss-120b | P4 CoT | ViSpanExtractQA | 0.5897 |

Prompt engineering cải thiện F1 từ +84% đến +297% so với zero-shot baseline.

## Cấu trúc project

```
├── src/                          # Source code modules
│   ├── config.py                 # Cấu hình dự án
│   ├── data_explorer.py          # EDA
│   ├── data_preprocessor.py      # Tiền xử lý + Instruction format
│   ├── model_selector.py         # Chọn mô hình + LoRA
│   ├── instruction_tuner.py      # Huấn luyện (SFTTrainer)
│   ├── model_evaluator.py        # Đánh giá (EM, F1, ROUGE-L)
│   ├── fpt_cloud_client.py       # FPT Cloud API client
│   └── templates.py              # Instruction templates
├── tests/                        # Unit tests
├── dataset/                      # Dữ liệu (ViNewsQA, ViCoV19QA, ViSpanExtractQA)
├── reports/
│   ├── eda/                      # Báo cáo EDA + biểu đồ
│   └── eval/                     # Kết quả đánh giá
├── paper/                        # Báo cáo đồ án
├── run_eda.py                    # Chạy phân tích dữ liệu
├── run_api_baseline.py           # Đánh giá baseline (P0)
├── run_prompt_engineering.py     # Đánh giá prompt engineering (P1-P7)
├── colab_pipeline.py             # Pipeline fine-tuning (self-contained cho Colab)
└── requirements.txt
```

## Cài đặt

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Chạy

```bash
# 1. EDA
python run_eda.py --with-hf

# 2. Prompt Engineering (cần FPT Cloud API key)
export FPT_API_KEY='your-key'
python run_prompt_engineering.py

# 3. Fine-tuning (trên Google Colab)
# Upload colab_pipeline.py → Colab → chọn T4 GPU → chạy
```

## Công nghệ

- Python 3.10+
- FPT Cloud API (OpenAI-compatible)
- HuggingFace Transformers, PEFT, TRL
- QLoRA 4-bit (bitsandbytes)
