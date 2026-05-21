# -*- coding: utf-8 -*-
"""
 Tinh chỉnh LLM cho Hỏi–Đáp tiếng Việt — Google Colab Pipeline

Pipeline 5 giai đoạn:
1. Thu thập & Phân tích dữ liệu (EDA)
2. Tiền xử lý & Định dạng Instruction
3. Lựa chọn mô hình & Cấu hình LoRA/QLoRA
4. Huấn luyện (Instruction Tuning + Fine-tuning)
5. Đánh giá & So sánh kết quả (EM, F1, ROUGE-L, BERTScore)

Hướng dẫn sử dụng trên Google Colab:
1. Mở Google Colab: https://colab.research.google.com
2. File > Upload notebook > chọn file này
3. Runtime > Change runtime type > T4 GPU
4. Chạy từng cell theo thứ tự

Hoặc copy từng phần vào các cell riêng biệt trong Colab.
"""

# ============================================================
# CELL 1: Cài đặt Dependencies & Kiểm tra GPU
# ============================================================
import subprocess
subprocess.run(["pip", "install", "-q", "bitsandbytes>=0.46.1", "accelerate", "peft", "trl", "datasets"], check=True)

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================================
# CELL 2: Cấu hình
# ============================================================
import os
import logging

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- CẤU HÌNH — TỐI ƯU CHO KAGGLE T4x2 (2×15GB VRAM, 30GB RAM) ---
BASE_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # 1.5B — tận dụng tốt T4x2
DATASET_NAMES = ["ntphuc149/ViSpanExtractQA"]  # Dataset từ HuggingFace Hub
DATASET_LOCAL_PATHS = []  # Kaggle không dùng Drive
INSTRUCTION_FORMAT = "alpaca"  # "alpaca" hoặc "chatml"
QUANTIZATION = "4bit"  # QLoRA 4-bit
NUM_EPOCHS = 1
BATCH_SIZE = 4  # Batch lớn hơn nhờ 2 GPU
GRADIENT_ACCUMULATION = 4  # Effective batch = 4 × 4 = 16
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 512  # Ngắn hơn để tiết kiệm VRAM
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
OUTPUT_DIR = "./outputs"

# FPT Cloud API (tùy chọn — để so sánh baseline)
# os.environ['FPT_API_KEY'] = 'your-api-key-here'
ENABLE_API_BASELINE = 'FPT_API_KEY' in os.environ
FPT_API_MODELS = ["Llama-3.3-70B-Instruct", "gemma-3-27b-it", "Qwen3-32B"]

print(f"Mô hình: {BASE_MODEL_NAME}")
print(f"Dataset: {DATASET_NAMES}")
print(f"Quantization: {QUANTIZATION}")
print(f"FPT API baseline: {'Bật' if ENABLE_API_BASELINE else 'Tắt'}")

# ============================================================
# CELL 3: Instruction Templates
# ============================================================
ALPACA_TRAIN_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Dựa vào ngữ cảnh được cung cấp, hãy trả lời câu hỏi một cách chính xác và ngắn gọn bằng tiếng Việt.

### Input:
Ngữ cảnh: {context}

Câu hỏi: {question}

### Response:
{answer}"""

ALPACA_INFERENCE_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Dựa vào ngữ cảnh được cung cấp, hãy trả lời câu hỏi một cách chính xác và ngắn gọn bằng tiếng Việt.

### Input:
Ngữ cảnh: {context}

Câu hỏi: {question}

### Response:
"""

CHATML_TRAIN_TEMPLATE = """<|im_start|>system
Bạn là trợ lý AI chuyên trả lời câu hỏi bằng tiếng Việt. Hãy trả lời dựa trên ngữ cảnh được cung cấp một cách chính xác và ngắn gọn.<|im_end|>
<|im_start|>user
Ngữ cảnh: {context}

Câu hỏi: {question}<|im_end|>
<|im_start|>assistant
{answer}<|im_end|>"""

CHATML_INFERENCE_TEMPLATE = """<|im_start|>system
Bạn là trợ lý AI chuyên trả lời câu hỏi bằng tiếng Việt. Hãy trả lời dựa trên ngữ cảnh được cung cấp một cách chính xác và ngắn gọn.<|im_end|>
<|im_start|>user
Ngữ cảnh: {context}

Câu hỏi: {question}<|im_end|>
<|im_start|>assistant
"""

TRAIN_TEMPLATE = ALPACA_TRAIN_TEMPLATE if INSTRUCTION_FORMAT == "alpaca" else CHATML_TRAIN_TEMPLATE
INFERENCE_TEMPLATE = ALPACA_INFERENCE_TEMPLATE if INSTRUCTION_FORMAT == "alpaca" else CHATML_INFERENCE_TEMPLATE
print(f"Instruction format: {INSTRUCTION_FORMAT}")

# ============================================================
# CELL 4: GIAI ĐOẠN 1 — Thu thập & Phân tích dữ liệu (EDA)
# ============================================================
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import Dataset, load_dataset, concatenate_datasets

@dataclass
class EDAReport:
    total_samples: int
    avg_context_length: float
    avg_question_length: float
    avg_answer_length: float
    context_length_distribution: dict
    question_length_distribution: dict
    answer_length_distribution: dict
    missing_fields_count: dict
    duplicate_count: int
    language_quality_issues: int
    dataset_source: str

def compute_length_dist(lengths):
    if not lengths:
        return {"min": 0, "max": 0, "median": 0.0, "std": 0.0}
    arr = np.array(lengths, dtype=float)
    return {"min": int(np.min(arr)), "max": int(np.max(arr)),
            "median": float(np.median(arr)), "std": float(np.std(arr))}

def load_qa_dataset(source, split="train"):
    """Tải dataset từ HuggingFace Hub hoặc file cục bộ."""
    if os.path.isfile(source):
        ext = os.path.splitext(source)[1].lower()
        if ext == ".csv":
            return Dataset.from_pandas(pd.read_csv(source))
        elif ext == ".json":
            with open(source, "r", encoding="utf-8") as f:
                data = json.load(f)
            return Dataset.from_list(data) if isinstance(data, list) else Dataset.from_dict(data)
    return load_dataset(source, split=split)

def run_eda(dataset, source_name):
    """Chạy EDA trên một dataset."""
    contexts = [str(row.get("context", "") or "") for row in dataset]
    questions = [str(row.get("question", "") or "") for row in dataset]
    answers = [str(row.get("answer", "") or "") for row in dataset]

    ctx_len = [len(c.split()) for c in contexts]
    q_len = [len(q.split()) for q in questions]
    a_len = [len(a.split()) for a in answers]

    missing_ctx = sum(1 for c in contexts if not c.strip())
    missing_q = sum(1 for q in questions if not q.strip())
    missing_a = sum(1 for a in answers if not a.strip())
    dupes = len(questions) - len(set(questions))
    unicode_issues = sum(1 for t in contexts + questions + answers
                         if t and unicodedata.normalize("NFC", t) != t)

    report = EDAReport(
        total_samples=len(dataset), dataset_source=source_name,
        avg_context_length=float(np.mean(ctx_len)) if ctx_len else 0,
        avg_question_length=float(np.mean(q_len)) if q_len else 0,
        avg_answer_length=float(np.mean(a_len)) if a_len else 0,
        context_length_distribution=compute_length_dist(ctx_len),
        question_length_distribution=compute_length_dist(q_len),
        answer_length_distribution=compute_length_dist(a_len),
        missing_fields_count={"context": missing_ctx, "question": missing_q, "answer": missing_a},
        duplicate_count=dupes, language_quality_issues=unicode_issues,
    )

    # Vẽ biểu đồ
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (name, lengths) in zip(axes, [("Context", ctx_len), ("Question", q_len), ("Answer", a_len)]):
        ax.hist(lengths, bins=50, edgecolor="black", alpha=0.7)
        ax.set_title(f"{source_name} — Phân phối độ dài {name}")
        ax.set_xlabel("Số từ")
        ax.axvline(np.mean(lengths), color="red", linestyle="--", label=f"TB: {np.mean(lengths):.0f}")
        ax.legend()
    plt.tight_layout()
    plt.show()

    print(f"\n EDA Report — {source_name}")
    print(f"  Tổng mẫu: {report.total_samples}")
    print(f"  Avg context/question/answer: {report.avg_context_length:.0f} / {report.avg_question_length:.0f} / {report.avg_answer_length:.0f} từ")
    print(f"  Missing: {report.missing_fields_count}")
    print(f"  Duplicates: {report.duplicate_count}, Unicode issues: {report.language_quality_issues}")
    return report

# --- Tải và phân tích dataset ---
datasets_dict = {}
eda_reports = []

# Tải từ HuggingFace Hub
for i, source in enumerate(DATASET_NAMES):
    name = source.split("/")[-1] if "/" in source else f"dataset_{i+1}"
    try:
        ds = load_qa_dataset(source)
        datasets_dict[name] = ds
        report = run_eda(ds, name)
        eda_reports.append(report)
    except Exception as e:
        logger.error(f"Không thể tải {source}: {e}")

# Tải ViNewsQA từ Google Drive (nếu có)
VINEWSQA_DIR = "/content/drive/MyDrive/Fine-Tune/dataset/ViNewsQA"
if os.path.isdir(VINEWSQA_DIR):
    print(f" Tải ViNewsQA từ Drive: {VINEWSQA_DIR}")
    vinews_samples = []
    for split in ["Train", "Dev", "Test"]:
        split_dir = os.path.join(VINEWSQA_DIR, split)
        if not os.path.isdir(split_dir):
            continue
        for fname in sorted(os.listdir(split_dir)):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(split_dir, fname), "r", encoding="utf-8") as f:
                data = json.load(f)
            for para in data.get("paragraphs", []):
                context = para.get("context", "")
                for qa in para.get("qas", []):
                    question = qa.get("question", "")
                    answers = qa.get("answers", [])
                    if answers:
                        vinews_samples.append({
                            "context": context,
                            "question": question,
                            "answer": answers[0].get("text", ""),
                        })
    if vinews_samples:
        ds = Dataset.from_list(vinews_samples)
        datasets_dict["ViNewsQA"] = ds
        report = run_eda(ds, "ViNewsQA")
        eda_reports.append(report)
        print(f"   ViNewsQA: {len(vinews_samples)} mẫu")

print(f"\n Đã tải {len(datasets_dict)} dataset")

# ============================================================
# CELL 5: GIAI ĐOẠN 2 — Tiền xử lý & Định dạng Instruction
# ============================================================

def normalize_vietnamese(text):
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.split()).strip()

def remove_noise(text):
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text

def sanitize_input(text):
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text.strip()

def preprocess_sample(sample):
    for field in ["context", "question", "answer"]:
        val = sample.get(field, "")
        sample[field] = sanitize_input(remove_noise(normalize_vietnamese(val)))
    return sample

def add_instruction(sample):
    sample["text"] = TRAIN_TEMPLATE.format(
        context=sample["context"], question=sample["question"], answer=sample["answer"]
    )
    return sample

# --- Tiền xử lý và chia splits ---
processed_datasets = {}
for name, ds in datasets_dict.items():
    # Tiền xử lý
    ds = ds.map(preprocess_sample)

    # Lọc mẫu rỗng
    ds = ds.filter(lambda x: bool(x.get("context", "").strip())
                   and bool(x.get("question", "").strip())
                   and bool(x.get("answer", "").strip()))

    # Thêm instruction text
    ds = ds.map(add_instruction)

    # Chia splits
    split1 = ds.train_test_split(test_size=0.1, seed=42)
    test_ds = split1["test"]
    split2 = split1["train"].train_test_split(test_size=0.1/0.9, seed=42)
    train_ds, val_ds = split2["train"], split2["test"]

    processed_datasets[name] = {"train": train_ds, "val": val_ds, "test": test_ds}
    print(f" {name}: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

# Gộp datasets
all_train = concatenate_datasets([v["train"] for v in processed_datasets.values()])
all_val = concatenate_datasets([v["val"] for v in processed_datasets.values()])
all_test = {name: splits["test"] for name, splits in processed_datasets.items()}

# Giới hạn dataset cho T4 (train nhanh hơn)
MAX_TRAIN_SAMPLES = 2000
if len(all_train) > MAX_TRAIN_SAMPLES:
    all_train = all_train.select(range(MAX_TRAIN_SAMPLES))
    print(f" Giới hạn train: {MAX_TRAIN_SAMPLES} mẫu (tiết kiệm thời gian trên T4)")

print(f"\n Tổng: train={len(all_train)}, val={len(all_val)}")
print(f"\n Ví dụ instruction text:\n{all_train[0]['text'][:500]}...")

# ============================================================
# CELL 6: GIAI ĐOẠN 3 — Lựa chọn mô hình & Cấu hình LoRA/QLoRA
# ============================================================
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType

# Cấu hình quantization
quant_config = None
model_kwargs = {"device_map": "auto", "trust_remote_code": True}

if QUANTIZATION == "4bit":
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model_kwargs["quantization_config"] = quant_config
elif QUANTIZATION == "8bit":
    quant_config = BitsAndBytesConfig(load_in_8bit=True)
    model_kwargs["quantization_config"] = quant_config
else:
    model_kwargs["torch_dtype"] = torch.float16

# Tải model
print(f" Đang tải mô hình: {BASE_MODEL_NAME}...")
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, **model_kwargs)
model.config.use_cache = False

# Tải tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"

# Cấu hình LoRA
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# Áp dụng LoRA
model = get_peft_model(model, lora_config)

# Thống kê params
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
ratio = trainable / total

print(f" Đã tải mô hình: {BASE_MODEL_NAME}")
print(f"   Trainable params: {trainable:,} / {total:,} ({ratio:.4%})")
print(f"   LoRA: r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")

# ============================================================
# CELL 7: GIAI ĐOẠN 4 — Huấn luyện (Instruction Tuning + Fine-tuning)
# ============================================================
from transformers import TrainingArguments
from trl import SFTTrainer

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    learning_rate=LEARNING_RATE,
    warmup_steps=50,
    weight_decay=0.01,
    logging_steps=10,
    save_steps=100,
    eval_steps=100,
    eval_strategy="steps",
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    fp16=True,
    bf16=False,
    gradient_checkpointing=True,
    report_to="none",
    remove_unused_columns=False,
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=all_train,
    eval_dataset=all_val,
    args=training_args,
)

print(f" Bắt đầu huấn luyện: {NUM_EPOCHS} epochs, batch_size={BATCH_SIZE}...")
try:
    trainer.train()
    print(" Huấn luyện hoàn tất!")
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        print(" CUDA OOM! Thử giảm BATCH_SIZE hoặc tăng GRADIENT_ACCUMULATION")
    raise

# Lưu mô hình
final_model_dir = os.path.join(OUTPUT_DIR, "final_model")
os.makedirs(final_model_dir, exist_ok=True)
model.save_pretrained(final_model_dir)
tokenizer.save_pretrained(final_model_dir)
print(f" Đã lưu mô hình tại: {final_model_dir}")

# ============================================================
# CELL 8: GIAI ĐOẠN 5 — Đánh giá & So sánh kết quả
# ============================================================

def normalize_text(text):
    return unicodedata.normalize("NFC", text).strip().lower()

def compute_exact_match(predictions, references):
    if not predictions:
        return 0.0
    return sum(1 for p, r in zip(predictions, references)
               if normalize_text(p) == normalize_text(r)) / len(predictions)

def compute_f1(predictions, references):
    if not predictions:
        return 0.0
    scores = []
    for pred, ref in zip(predictions, references):
        p_tok = normalize_text(pred).split()
        r_tok = normalize_text(ref).split()
        if not r_tok and not p_tok:
            scores.append(1.0); continue
        if not r_tok or not p_tok:
            scores.append(0.0); continue
        common = Counter(p_tok) & Counter(r_tok)
        nc = sum(common.values())
        if nc == 0:
            scores.append(0.0); continue
        prec, rec = nc / len(p_tok), nc / len(r_tok)
        scores.append(2 * prec * rec / (prec + rec))
    return float(np.mean(scores))

def compute_rouge_l(predictions, references):
    if not predictions:
        return 0.0
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    return float(np.mean([
        scorer.score(normalize_text(r), normalize_text(p))["rougeL"].fmeasure
        for p, r in zip(predictions, references)
    ]))

def compute_bert_score_metric(predictions, references):
    if not predictions:
        return 0.0
    try:
        from bert_score import score as bert_score_fn
        P, R, F1 = bert_score_fn(predictions, references,
                                   model_type="bert-base-multilingual-cased",
                                   lang="vi", verbose=False)
        return float(F1.mean().item())
    except Exception as e:
        print(f" BERTScore không khả dụng: {e}")
        return -1.0

# --- Đánh giá mô hình local ---
print(" Đánh giá mô hình fine-tuned trên tập test...")
model.eval()
device = next(model.parameters()).device
all_results = []

for ds_name, test_ds in all_test.items():
    predictions, references = [], []
    for sample in test_ds:
        prompt = INFERENCE_TEMPLATE.format(context=sample["context"], question=sample["question"])
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        predictions.append(unicodedata.normalize("NFC", generated))
        references.append(unicodedata.normalize("NFC", sample["answer"]))

    em = compute_exact_match(predictions, references)
    f1 = compute_f1(predictions, references)
    rouge_l = compute_rouge_l(predictions, references)
    bert_s = compute_bert_score_metric(predictions, references)

    result = {"dataset": ds_name, "source": "local", "model": BASE_MODEL_NAME,
              "n": len(predictions), "EM": em, "F1": f1, "ROUGE-L": rouge_l, "BERTScore": bert_s}
    all_results.append(result)
    print(f"   [Local] {ds_name}: EM={em:.4f}, F1={f1:.4f}, ROUGE-L={rouge_l:.4f}, BERTScore={bert_s:.4f}")

# --- Đánh giá API baseline (tùy chọn) ---
if ENABLE_API_BASELINE:
    import time
    from openai import OpenAI

    FPT_API_BASE_URL = "https://mkp-api.fptcloud.com"
    api_key = os.environ.get("FPT_API_KEY")
    client = OpenAI(api_key=api_key, base_url=FPT_API_BASE_URL)

    for api_model in FPT_API_MODELS:
        print(f"\n Đánh giá API baseline: {api_model}...")
        for ds_name, test_ds in all_test.items():
            predictions, references = [], []
            for sample in test_ds:
                try:
                    response = client.chat.completions.create(
                        model=api_model,
                        messages=[
                            {"role": "system", "content": "Bạn là trợ lý AI chuyên trả lời câu hỏi bằng tiếng Việt. Hãy trả lời dựa trên ngữ cảnh được cung cấp một cách chính xác và ngắn gọn."},
                            {"role": "user", "content": f"Ngữ cảnh: {sample['context']}\n\nCâu hỏi: {sample['question']}"},
                        ],
                        max_tokens=512, temperature=0.1,
                    )
                    pred = response.choices[0].message.content.strip()
                except Exception as e:
                    pred = ""
                predictions.append(unicodedata.normalize("NFC", pred))
                references.append(unicodedata.normalize("NFC", sample["answer"]))
                time.sleep(0.1)  # Rate limit

            em = compute_exact_match(predictions, references)
            f1 = compute_f1(predictions, references)
            rouge_l = compute_rouge_l(predictions, references)
            bert_s = compute_bert_score_metric(predictions, references)

            result = {"dataset": ds_name, "source": "api", "model": api_model,
                      "n": len(predictions), "EM": em, "F1": f1, "ROUGE-L": rouge_l, "BERTScore": bert_s}
            all_results.append(result)
            print(f"   [API:{api_model}] {ds_name}: EM={em:.4f}, F1={f1:.4f}, ROUGE-L={rouge_l:.4f}, BERTScore={bert_s:.4f}")

# ============================================================
# CELL 9: Báo cáo tổng hợp
# ============================================================
print("\n" + "="*80)
print(" BÁO CÁO ĐÁNH GIÁ TỔNG HỢP")
print("="*80)

results_df = pd.DataFrame(all_results)
print(results_df.to_string(index=False))

# Lưu báo cáo
os.makedirs("./reports/eval", exist_ok=True)
report_path = "./reports/eval/evaluation_report.json"
with open(report_path, "w", encoding="utf-8") as f:
    json.dump({"results": all_results}, f, ensure_ascii=False, indent=2)
print(f"\n Báo cáo đã lưu tại: {report_path}")

# So sánh local vs API
local_results = [r for r in all_results if r["source"] == "local"]
api_results = [r for r in all_results if r["source"] == "api"]

if local_results:
    avg_f1_local = np.mean([r["F1"] for r in local_results])
    print(f"\n Local fine-tuned — Avg F1: {avg_f1_local:.4f}")
if api_results:
    avg_f1_api = np.mean([r["F1"] for r in api_results])
    print(f"  API baseline    — Avg F1: {avg_f1_api:.4f}")

print("\n PIPELINE HOÀN TẤT!")
print(f"   Mô hình đã lưu: {final_model_dir}")
print(f"   Báo cáo: {report_path}")

# ============================================================
# CELL 10 (Tùy chọn): Tải mô hình về máy local
# ============================================================
# from google.colab import files
# !zip -r final_model.zip {final_model_dir}
# files.download("final_model.zip")
