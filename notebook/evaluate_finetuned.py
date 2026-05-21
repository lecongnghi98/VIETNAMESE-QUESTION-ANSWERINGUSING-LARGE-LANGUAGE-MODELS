# -*- coding: utf-8 -*-
"""
Danh gia model fine-tuned tren toan bo 2 dataset: ViNewsQA + ViSpanExtractQA
Chay tren Kaggle sau khi da fine-tune xong.

Model path: /kaggle/working/outputs/final_model
Dataset path: /kaggle/input/datasets/lcngngh/uat-finetune/dataset
"""

import json
import os
import unicodedata
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ============================================================
# CAU HINH
# ============================================================
MODEL_PATH = "outputs/kaggle/working/outputs/final_model"  # Duong dan local
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET_PATH = "dataset"  # Duong dan local
MAX_SEQ_LENGTH = 512
MAX_NEW_TOKENS = 128
BATCH_LOG_EVERY = 50

# Chon device: MPS (Mac Apple Silicon) hoac CPU
import platform
if platform.system() == "Darwin" and torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"

ALPACA_INFERENCE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Dua vao ngu canh duoc cung cap, hay tra loi cau hoi mot cach chinh xac va ngan gon bang tieng Viet. Chi trich xuat thong tin tu ngu canh, khong them thong tin ngoai.

### Input:
Ngu canh: {context}

Cau hoi: {question}

### Response:
"""

# ============================================================
# METRICS
# ============================================================
def normalize_text(text):
    return unicodedata.normalize("NFC", text).strip().lower()

def compute_em(preds, refs):
    if not preds:
        return 0.0
    return sum(1 for p, r in zip(preds, refs) if normalize_text(p) == normalize_text(r)) / len(preds)

def compute_f1(preds, refs):
    if not preds:
        return 0.0
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
    if not preds:
        return 0.0
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    return float(np.mean([
        scorer.score(normalize_text(r), normalize_text(p))["rougeL"].fmeasure
        for p, r in zip(preds, refs)
    ]))

# ============================================================
# TAI DATASET
# ============================================================
def load_vinewsqa_test(base_dir):
    """Tai ViNewsQA test set."""
    samples = []
    test_dir = os.path.join(base_dir, "ViNewsQA", "Test")
    if not os.path.isdir(test_dir):
        print(f"  Khong tim thay: {test_dir}")
        return samples
    for fname in sorted(os.listdir(test_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(test_dir, fname), "r", encoding="utf-8") as f:
            data = json.load(f)
        for para in data.get("paragraphs", []):
            context = para.get("context", "")
            for qa in para.get("qas", []):
                question = qa.get("question", "")
                answers = qa.get("answers", [])
                if answers:
                    samples.append({
                        "context": context,
                        "question": question,
                        "answer": answers[0].get("text", ""),
                    })
    return samples

def load_vispanextractqa_test(base_dir):
    """Tai ViSpanExtractQA test set (lay 2000 mau cuoi lam test)."""
    samples = []
    vispan_dir = os.path.join(base_dir, "ViSpanExtractQA")
    if not os.path.isdir(vispan_dir):
        print(f"  Khong tim thay: {vispan_dir}")
        return samples
    
    try:
        from datasets import load_from_disk
        for split_name in ["test", "validation", "train"]:
            split_path = os.path.join(vispan_dir, split_name)
            if os.path.isdir(split_path):
                ds = load_from_disk(split_path)
                for row in ds:
                    ctx = str(row.get("context", "")).strip()
                    q = str(row.get("question", "")).strip()
                    a = str(row.get("answer_text", row.get("answer", ""))).strip()
                    if ctx and q and a:
                        samples.append({"context": ctx, "question": q, "answer": a})
                if split_name == "test":
                    break  # Chi lay test set
    except Exception as e:
        print(f"  Loi load arrow: {e}")
        # Fallback: load CSV
        import pandas as pd
        for fname in os.listdir(vispan_dir):
            if fname.endswith(".csv"):
                df = pd.read_csv(os.path.join(vispan_dir, fname))
                for _, row in df.iterrows():
                    ctx = str(row.get("context", "")).strip()
                    q = str(row.get("question", "")).strip()
                    a = str(row.get("answer_text", row.get("answer", ""))).strip()
                    if ctx and q and a:
                        samples.append({"context": ctx, "question": q, "answer": a})
    
    # Neu khong co test split rieng, lay 2000 mau cuoi
    if len(samples) > 2000:
        samples = samples[-2000:]
    return samples

# ============================================================
# TAI MODEL
# ============================================================
print("=" * 60)
print("DANH GIA MODEL FINE-TUNED TREN TOAN BO 2 DATASET")
print("=" * 60)

print(f"\nDang tai model tu: {MODEL_PATH}")
print(f"Base model: {BASE_MODEL}")

# Tai base model
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map={"": DEVICE} if DEVICE != "cpu" else None,
    torch_dtype=torch.float32,  # float32 cho CPU/MPS
    trust_remote_code=True,
)

# Tai LoRA adapter
model = PeftModel.from_pretrained(model, MODEL_PATH)
model.eval()
if DEVICE == "cpu":
    pass  # Already on CPU
elif DEVICE == "mps":
    model = model.to("mps")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

device = DEVICE
print(f"Model loaded on: {device}")

# ============================================================
# TAI DATASETS
# ============================================================
print(f"\nDang tai datasets tu: {DATASET_PATH}")

vinews_test = load_vinewsqa_test(DATASET_PATH)
print(f"  ViNewsQA Test: {len(vinews_test)} mau")

vispan_test = load_vispanextractqa_test(DATASET_PATH)
print(f"  ViSpanExtractQA Test: {len(vispan_test)} mau")

# ============================================================
# DANH GIA
# ============================================================
def evaluate_dataset(samples, dataset_name, max_samples=None):
    """Danh gia model tren 1 dataset."""
    if max_samples and len(samples) > max_samples:
        samples = samples[:max_samples]
    
    print(f"\n{'='*60}")
    print(f"Dang danh gia: {dataset_name} ({len(samples)} mau)")
    print(f"{'='*60}")
    
    predictions, references = [], []
    
    for i, sample in enumerate(samples):
        prompt = ALPACA_INFERENCE.format(
            context=sample["context"][:1500],  # Truncate context dai
            question=sample["question"]
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH).to(device)        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        
        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        predictions.append(unicodedata.normalize("NFC", generated))
        references.append(unicodedata.normalize("NFC", sample["answer"]))
        
        if (i + 1) % BATCH_LOG_EVERY == 0:
            cur_f1 = compute_f1(predictions, references)
            print(f"  [{i+1}/{len(samples)}] F1={cur_f1:.4f}")
    
    # Tinh metrics
    em = compute_em(predictions, references)
    f1 = compute_f1(predictions, references)
    rl = compute_rouge_l(predictions, references)
    
    print(f"\n  KET QUA {dataset_name}:")
    print(f"    EM:      {em:.4f}")
    print(f"    F1:      {f1:.4f}")
    print(f"    ROUGE-L: {rl:.4f}")
    print(f"    Samples: {len(samples)}")
    
    # In vi du
    print(f"\n  Vi du (5 mau dau):")
    for j in range(min(5, len(predictions))):
        print(f"    Q: {samples[j]['question'][:60]}")
        print(f"    Ref: {references[j][:60]}")
        print(f"    Pred: {predictions[j][:60]}")
        print()
    
    return {"dataset": dataset_name, "EM": em, "F1": f1, "ROUGE-L": rl, "n": len(samples)}

# Danh gia tren tung dataset
results = []

if vinews_test:
    r = evaluate_dataset(vinews_test, "ViNewsQA")
    results.append(r)

if vispan_test:
    r = evaluate_dataset(vispan_test, "ViSpanExtractQA")
    results.append(r)

# ============================================================
# TONG HOP
# ============================================================
print(f"\n{'='*60}")
print(f"TONG HOP KET QUA DANH GIA")
print(f"{'='*60}")
print(f"  {'Dataset':<25} {'N':<8} {'EM':<8} {'F1':<8} {'ROUGE-L'}")
print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
for r in results:
    print(f"  {r['dataset']:<25} {r['n']:<8} {r['EM']:<8.4f} {r['F1']:<8.4f} {r['ROUGE-L']:.4f}")

if len(results) == 2:
    avg_em = np.mean([r['EM'] for r in results])
    avg_f1 = np.mean([r['F1'] for r in results])
    avg_rl = np.mean([r['ROUGE-L'] for r in results])
    print(f"  {'TRUNG BINH':<25} {'-':<8} {avg_em:<8.4f} {avg_f1:<8.4f} {avg_rl:.4f}")

print(f"{'='*60}")

# Luu ket qua
output_file = "reports/eval/finetune_eval_results.json"
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\nDa luu ket qua tai: {output_file}")
