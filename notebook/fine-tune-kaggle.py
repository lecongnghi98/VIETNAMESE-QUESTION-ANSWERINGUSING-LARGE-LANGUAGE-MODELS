# -*- coding: utf-8 -*-
"""
Fine-tune LLM cho Hoi-Dap tieng Viet - Kaggle T4x2
Model: Qwen2.5-1.5B-Instruct + QLoRA 4-bit
Dataset: ViSpanExtractQA + ViNewsQA
Thoi gian uoc tinh: 1-2 gio tren T4x2
"""

# ============================================================
# CELL 1: Cai dat va kiem tra GPU
# ============================================================
import subprocess
subprocess.run(["pip", "install", "-q", "bitsandbytes>=0.46.1", "accelerate", "peft", "trl>=0.15", "datasets", "rouge-score"], check=True)

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
print(f"So GPU: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)} - {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")

# ============================================================
# CELL 2: Cau hinh
# ============================================================
import os
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- CAU HINH TOI UU CHO KAGGLE T4x2 ---
BASE_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
QUANTIZATION = None  # Khong quantize - chia model len 2 GPU bang device_map
NUM_EPOCHS = 3
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 4   # Effective batch = 4 * 4 = 16
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 512
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
OUTPUT_DIR = "./outputs"
MAX_TRAIN_SAMPLES = 5000    # Gioi han de train trong 1-2h

print(f"Model: {BASE_MODEL_NAME}")
print(f"Quantization: {QUANTIZATION}")
print(f"Epochs: {NUM_EPOCHS}, Batch: {BATCH_SIZE}, Grad Accum: {GRADIENT_ACCUMULATION}")
print(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION}")

# ============================================================
# CELL 3: Instruction Template
# ============================================================
ALPACA_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Dựa vào ngữ cảnh được cung cấp, hãy trả lời câu hỏi một cách chính xác và ngắn gọn bằng tiếng Việt. Chỉ trích xuất thông tin từ ngữ cảnh, không thêm thông tin ngoài.

### Input:
Ngữ cảnh: {context}

Câu hỏi: {question}

### Response:
{answer}"""

ALPACA_INFERENCE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Dựa vào ngữ cảnh được cung cấp, hãy trả lời câu hỏi một cách chính xác và ngắn gọn bằng tiếng Việt. Chỉ trích xuất thông tin từ ngữ cảnh, không thêm thông tin ngoài.

### Input:
Ngữ cảnh: {context}

Câu hỏi: {question}

### Response:
"""

# ============================================================
# CELL 4: Tai va tien xu ly du lieu
# ============================================================
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset, concatenate_datasets

def normalize_vietnamese(text):
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return " ".join(text.split()).strip()

def add_text_field(sample):
    sample["text"] = ALPACA_TEMPLATE.format(
        context=normalize_vietnamese(sample["context"]),
        question=normalize_vietnamese(sample["question"]),
        answer=normalize_vietnamese(sample["answer"])
    )
    return sample

# Tai dataset tu Kaggle input
KAGGLE_DATASET_PATH = "/kaggle/input/datasets/lcngngh/uat-finetune/dataset"

print("Dang tai dataset...")

# Tai ViNewsQA tu Kaggle
vinewsqa_dir = os.path.join(KAGGLE_DATASET_PATH, "ViNewsQA")
vinews_samples = []
if os.path.isdir(vinewsqa_dir):
    for split in ["Train", "Dev", "Test"]:
        split_dir = os.path.join(vinewsqa_dir, split)
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
    print(f"  ViNewsQA: {len(vinews_samples)} mau")

# Tai ViSpanExtractQA tu Kaggle input (da co san)
vispan_dir = os.path.join(KAGGLE_DATASET_PATH, "ViSpanExtractQA")
vispan_samples = []
if os.path.isdir(vispan_dir):
    from datasets import load_from_disk
    # Thu load tu arrow format
    for split_name in ["train", "test", "validation"]:
        split_path = os.path.join(vispan_dir, split_name)
        if os.path.isdir(split_path):
            try:
                split_ds = load_from_disk(split_path)
                for row in split_ds:
                    ctx = str(row.get("context", "")).strip()
                    q = str(row.get("question", "")).strip()
                    a = str(row.get("answer_text", row.get("answer", ""))).strip()
                    if ctx and q and a:
                        vispan_samples.append({"context": ctx, "question": q, "answer": a})
            except Exception:
                pass
    # Neu khong co arrow, thu CSV
    if not vispan_samples:
        for fname in os.listdir(vispan_dir):
            if fname.endswith(".csv"):
                df = pd.read_csv(os.path.join(vispan_dir, fname))
                for _, row in df.iterrows():
                    ctx = str(row.get("context", "")).strip()
                    q = str(row.get("question", "")).strip()
                    a = str(row.get("answer_text", row.get("answer", ""))).strip()
                    if ctx and q and a:
                        vispan_samples.append({"context": ctx, "question": q, "answer": a})
    print(f"  ViSpanExtractQA: {len(vispan_samples)} mau")
else:
    # Fallback: tai tu HuggingFace
    print("  ViSpanExtractQA khong co tren Kaggle, tai tu HuggingFace...")
    vispan_ds = load_dataset("ntphuc149/ViSpanExtractQA", split="train")
    for row in vispan_ds:
        ctx = str(row.get("context", "")).strip()
        q = str(row.get("question", "")).strip()
        a = str(row.get("answer_text", "")).strip()
        if ctx and q and a:
            vispan_samples.append({"context": ctx, "question": q, "answer": a})
    print(f"  ViSpanExtractQA: {len(vispan_samples)} mau")

# Gop datasets
all_samples = vinews_samples + vispan_samples
print(f"  Tong: {len(all_samples)} mau")

# Tao HuggingFace Dataset
ds = Dataset.from_list(all_samples)

# Them text field
ds = ds.map(add_text_field)

# Loc mau qua dai
ds = ds.filter(lambda x: len(x["text"].split()) <= MAX_SEQ_LENGTH)
print(f"  Sau loc (max {MAX_SEQ_LENGTH} tu): {len(ds)} mau")

# Chia train/val/test
split1 = ds.train_test_split(test_size=0.1, seed=42)
test_ds = split1["test"]
split2 = split1["train"].train_test_split(test_size=0.1, seed=42)
train_ds = split2["train"]
val_ds = split2["test"]

# Gioi han train samples
if len(train_ds) > MAX_TRAIN_SAMPLES:
    train_ds = train_ds.select(range(MAX_TRAIN_SAMPLES))

print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
print(f"  Vi du:\n{train_ds[0]['text'][:300]}...")

# ============================================================
# CELL 5: Tai model + LoRA
# ============================================================
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# Quantization config (khong dung - de model chia len 2 GPU)
print(f"Dang tai model: {BASE_MODEL_NAME}...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16,
)
model.config.use_cache = False

# Tai tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"

# LoRA config
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# Ap dung LoRA
model = get_peft_model(model, lora_config)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Da tai model: {BASE_MODEL_NAME}")
print(f"  Trainable: {trainable:,} / {total:,} ({trainable/total:.4%})")
print(f"  LoRA: r={LORA_R}, alpha={LORA_ALPHA}")

# ============================================================
# CELL 5.5: Danh gia BASE MODEL (truoc fine-tune)
# ============================================================

def normalize_text(text):
    return unicodedata.normalize("NFC", text).strip().lower()

def compute_em(preds, refs):
    return sum(1 for p, r in zip(preds, refs) if normalize_text(p) == normalize_text(r)) / len(preds)

def compute_f1(preds, refs):
    scores = []
    for pred, ref in zip(preds, refs):
        p_tok = normalize_text(pred).split()
        r_tok = normalize_text(ref).split()
        if not r_tok and not p_tok: scores.append(1.0); continue
        if not r_tok or not p_tok: scores.append(0.0); continue
        common = Counter(p_tok) & Counter(r_tok)
        nc = sum(common.values())
        if nc == 0: scores.append(0.0); continue
        prec, rec = nc / len(p_tok), nc / len(r_tok)
        scores.append(2 * prec * rec / (prec + rec))
    return float(np.mean(scores))

def compute_rouge_l(preds, refs):
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    return float(np.mean([
        scorer.score(normalize_text(r), normalize_text(p))["rougeL"].fmeasure
        for p, r in zip(preds, refs)
    ]))

print("Danh gia base model (truoc fine-tune)...")
model.eval()
device = next(model.parameters()).device

eval_samples = min(200, len(test_ds))
base_preds, base_refs = [], []

for i in range(eval_samples):
    sample = test_ds[i]
    prompt = ALPACA_INFERENCE.format(context=sample["context"], question=sample["question"])
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    base_preds.append(unicodedata.normalize("NFC", generated))
    base_refs.append(unicodedata.normalize("NFC", sample["answer"]))
    if (i + 1) % 50 == 0:
        print(f"  [{i+1}/{eval_samples}] F1={compute_f1(base_preds, base_refs):.4f}")

base_em = compute_em(base_preds, base_refs)
base_f1 = compute_f1(base_preds, base_refs)
base_rl = compute_rouge_l(base_preds, base_refs)
print(f"\n  BASE MODEL: EM={base_em:.4f}, F1={base_f1:.4f}, ROUGE-L={base_rl:.4f}")

# ============================================================
# CELL 6: Huan luyen
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
    warmup_steps=100,
    weight_decay=0.01,
    logging_steps=20,
    save_steps=200,
    eval_steps=200,
    eval_strategy="steps",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    fp16=True,
    bf16=False,
    gradient_checkpointing=True,
    ddp_find_unused_parameters=False,
    report_to="none",
    remove_unused_columns=False,
    dataloader_num_workers=2,
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    args=training_args,
)

print(f"Bat dau huan luyen: {NUM_EPOCHS} epochs...")
print(f"  Total steps: {len(train_ds) * NUM_EPOCHS // (BATCH_SIZE * GRADIENT_ACCUMULATION)}")

# Resume tu checkpoint neu co
checkpoint = None
if os.path.isdir(OUTPUT_DIR):
    checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
    if checkpoints:
        latest = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
        checkpoint = os.path.join(OUTPUT_DIR, latest)
        print(f"  Tim thay checkpoint: {checkpoint} - se resume tu day")

trainer.train(resume_from_checkpoint=checkpoint)
print("Huan luyen hoan tat!")

# Luu model
final_dir = os.path.join(OUTPUT_DIR, "final_model")
os.makedirs(final_dir, exist_ok=True)
model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)
print(f"Da luu model tai: {final_dir}")

# ============================================================
# CELL 7: Danh gia
# ============================================================

def normalize_text(text):
    return unicodedata.normalize("NFC", text).strip().lower()

def compute_em(preds, refs):
    return sum(1 for p, r in zip(preds, refs) if normalize_text(p) == normalize_text(r)) / len(preds)

def compute_f1(preds, refs):
    scores = []
    for pred, ref in zip(preds, refs):
        p_tok = normalize_text(pred).split()
        r_tok = normalize_text(ref).split()
        if not r_tok and not p_tok: scores.append(1.0); continue
        if not r_tok or not p_tok: scores.append(0.0); continue
        common = Counter(p_tok) & Counter(r_tok)
        nc = sum(common.values())
        if nc == 0: scores.append(0.0); continue
        prec, rec = nc / len(p_tok), nc / len(r_tok)
        scores.append(2 * prec * rec / (prec + rec))
    return float(np.mean(scores))

def compute_rouge_l(preds, refs):
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    return float(np.mean([
        scorer.score(normalize_text(r), normalize_text(p))["rougeL"].fmeasure
        for p, r in zip(preds, refs)
    ]))

# Danh gia tren TUNG dataset rieng biet
print("Dang danh gia model tren tung dataset...")
model.eval()
device = next(model.parameters()).device

# Tach test set theo dataset
vinews_test = [s for s in test_ds if len(s["answer"].split()) > 8]  # ViNewsQA: answer dai hon
vispan_test = [s for s in test_ds if len(s["answer"].split()) <= 8]  # ViSpanExtractQA: answer ngan

datasets_eval = {
    "Toan bo": list(test_ds),
    "ViNewsQA (answer > 8 tu)": vinews_test,
    "ViSpanExtractQA (answer <= 8 tu)": vispan_test,
}

all_results = {}

for ds_name, ds_samples in datasets_eval.items():
    eval_samples = min(200, len(ds_samples))
    if eval_samples == 0:
        continue
    predictions, references = [], []

    for i in range(eval_samples):
        sample = ds_samples[i]
        prompt = ALPACA_INFERENCE.format(context=sample["context"], question=sample["question"])
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH).to(device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)

        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        predictions.append(unicodedata.normalize("NFC", generated))
        references.append(unicodedata.normalize("NFC", sample["answer"]))

        if (i + 1) % 50 == 0:
            print(f"  [{ds_name}] [{i+1}/{eval_samples}] F1={compute_f1(predictions, references):.4f}")

    em = compute_em(predictions, references)
    f1 = compute_f1(predictions, references)
    rl = compute_rouge_l(predictions, references)
    all_results[ds_name] = {"EM": em, "F1": f1, "ROUGE-L": rl, "n": eval_samples}

# Ket qua
print(f"\n{'='*60}")
print(f"KET QUA DANH GIA FINE-TUNED MODEL")
print(f"{'='*60}")
print(f"  {'Dataset':<35} {'N':<6} {'EM':<8} {'F1':<8} {'ROUGE-L'}")
print(f"  {'-'*35} {'-'*6} {'-'*8} {'-'*8} {'-'*8}")
for ds_name, r in all_results.items():
    print(f"  {ds_name:<35} {r['n']:<6} {r['EM']:<8.4f} {r['F1']:<8.4f} {r['ROUGE-L']:.4f}")
print(f"{'='*60}")

# Vi du predictions
print("\nVi du predictions:")
samples_show = list(test_ds)[:5]
preds_show = []
for sample in samples_show:
    prompt = ALPACA_INFERENCE.format(context=sample["context"], question=sample["question"])
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    preds_show.append(generated)

for i in range(len(samples_show)):
    print(f"\n  Q: {samples_show[i]['question'][:80]}")
    print(f"  Ref: {samples_show[i]['answer'][:80]}")
    print(f"  Pred: {preds_show[i][:80]}")

# So sanh base vs fine-tuned
total_r = all_results.get("Toan bo", {})
print(f"\n{'='*60}")
print(f"SO SANH: BASE vs FINE-TUNED (toan bo test set)")
print(f"{'='*60}")
print(f"  {'Metric':<10} {'Base':<10} {'Fine-tuned':<12} {'Cai thien'}")
print(f"  {'EM':<10} {base_em:<10.4f} {total_r.get('EM',0):<12.4f} +{(total_r.get('EM',0)-base_em)/max(base_em,0.001)*100:.1f}%")
print(f"  {'F1':<10} {base_f1:<10.4f} {total_r.get('F1',0):<12.4f} +{(total_r.get('F1',0)-base_f1)/max(base_f1,0.001)*100:.1f}%")
print(f"  {'ROUGE-L':<10} {base_rl:<10.4f} {total_r.get('ROUGE-L',0):<12.4f} +{(total_r.get('ROUGE-L',0)-base_rl)/max(base_rl,0.001)*100:.1f}%")
print(f"{'='*60}")
