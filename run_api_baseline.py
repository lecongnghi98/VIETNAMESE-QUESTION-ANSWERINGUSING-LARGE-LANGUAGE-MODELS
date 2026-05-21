"""
Đánh giá baseline LLM FPT Cloud trên dataset QA tiếng Việt.
Chạy: python3 run_api_baseline.py

Yêu cầu: export FPT_API_KEY='your-api-key'
"""

import json
import os
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI

# ============================================================
# CẤU HÌNH
# ============================================================
FPT_API_BASE_URL = "https://mkp-api.fptcloud.com"
FPT_MODELS = [ "Llama-3.3-70B-Instruct","gpt-oss-120b"]
MAX_TOKENS = 512
TEMPERATURE = 0.1
MAX_SAMPLES = 50  # Giới hạn số mẫu test (để tiết kiệm API calls). Đặt None để chạy hết.

# ============================================================
# Kiểm tra API key
# ============================================================
api_key = os.environ.get("FPT_API_KEY")
if not api_key:
    print("❌ Chưa thiết lập FPT_API_KEY!")
    print("   Chạy: export FPT_API_KEY='your-api-key'")
    sys.exit(1)

client = OpenAI(api_key=api_key, base_url=FPT_API_BASE_URL)
print(f"✅ FPT Cloud API đã kết nối")

# ============================================================
# Tải Dataset
# ============================================================

def load_vinewsqa(split_dir):
    """Tải ViNewsQA từ thư mục JSON files."""
    samples = []
    json_dir = Path(split_dir)
    for f in sorted(json_dir.glob("*.json")):
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        for para in data.get("paragraphs", []):
            context = para.get("context", "")
            for qa in para.get("qas", []):
                question = qa.get("question", "")
                answers = qa.get("answers", [])
                if answers:
                    answer = answers[0].get("text", "")
                    samples.append({"context": context, "question": question, "answer": answer, "source": "ViNewsQA"})
    return samples


def load_vispanextractqa(dataset_dir):
    """Tải ViSpanExtractQA từ HuggingFace arrow format."""
    from datasets import load_from_disk
    ds = load_from_disk(dataset_dir)
    samples = []
    for row in ds:
        ctx = str(row.get("context", "")).strip()
        q = str(row.get("question", "")).strip()
        a = str(row.get("answer_text", "")).strip()
        if ctx and q and a:
            samples.append({"context": ctx, "question": q, "answer": a, "source": "ViSpanExtractQA"})
    return samples


print("\n📦 Đang tải datasets...")

datasets = {}

# ViNewsQA Test
vinewsqa_test_dir = "dataset/ViNewsQA/Test"
if Path(vinewsqa_test_dir).exists():
    samples = load_vinewsqa(vinewsqa_test_dir)
    datasets["ViNewsQA"] = samples
    print(f"  ViNewsQA Test: {len(samples)} mẫu")

# ViSpanExtractQA Test
vispanextract_test = "dataset/ViSpanExtractQA/test"
if Path(vispanextract_test).exists():
    samples = load_vispanextractqa(vispanextract_test)
    datasets["ViSpanExtractQA"] = samples
    print(f"  ViSpanExtractQA Test: {len(samples)} mẫu")

if not datasets:
    print("❌ Không tìm thấy dataset nào!")
    sys.exit(1)

# ============================================================
# Metrics
# ============================================================

def normalize_text(text):
    return unicodedata.normalize("NFC", text).strip().lower()


def exact_match(pred, ref):
    return 1.0 if normalize_text(pred) == normalize_text(ref) else 0.0


def f1_score(pred, ref):
    p_tok = normalize_text(pred).split()
    r_tok = normalize_text(ref).split()
    if not r_tok and not p_tok:
        return 1.0
    if not r_tok or not p_tok:
        return 0.0
    common = Counter(p_tok) & Counter(r_tok)
    nc = sum(common.values())
    if nc == 0:
        return 0.0
    prec, rec = nc / len(p_tok), nc / len(r_tok)
    return 2 * prec * rec / (prec + rec)


def rouge_l_score(pred, ref):
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    return scorer.score(normalize_text(ref), normalize_text(pred))["rougeL"].fmeasure


# ============================================================
# Gọi API & Đánh giá
# ============================================================

def call_fpt_api(client, model_name, context, question, max_retries=3):
    """Gọi FPT Cloud API với retry."""
    if context:
        system_msg = "Bạn là trợ lý AI chuyên trả lời câu hỏi bằng tiếng Việt. Hãy trả lời dựa trên ngữ cảnh được cung cấp một cách chính xác và ngắn gọn."
        user_msg = f"Ngữ cảnh: {context}\n\nCâu hỏi: {question}"
    else:
        system_msg = "Bạn là trợ lý AI chuyên trả lời câu hỏi bằng tiếng Việt. Hãy trả lời một cách chính xác và ngắn gọn."
        user_msg = question

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
            content = response.choices[0].message.content
            return content.strip() if content else ""
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"    ⚠️ Lỗi API: {e}. Thử lại sau {wait}s...")
                time.sleep(wait)
            else:
                print(f"    ❌ Lỗi API sau {max_retries} lần: {e}")
                return ""

# ============================================================
# Chạy đánh giá
# ============================================================

all_results = []

for model_name in FPT_MODELS:
    print(f"\n{'='*60}")
    print(f"🤖 Mô hình: {model_name}")
    print(f"{'='*60}")

    for ds_name, samples in datasets.items():
        test_samples = samples[:MAX_SAMPLES] if MAX_SAMPLES else samples
        print(f"\n  📊 Dataset: {ds_name} ({len(test_samples)} mẫu)")

        em_scores, f1_scores, rouge_scores = [], [], []
        per_sample = []

        for i, sample in enumerate(test_samples):
            pred = call_fpt_api(client, model_name, sample["context"], sample["question"])
            ref = sample["answer"]

            em = exact_match(pred, ref)
            f1 = f1_score(pred, ref)
            rl = rouge_l_score(pred, ref)

            em_scores.append(em)
            f1_scores.append(f1)
            rouge_scores.append(rl)

            per_sample.append({
                "question": sample["question"],
                "reference": ref[:100],
                "prediction": pred[:100],
                "EM": em, "F1": round(f1, 4), "ROUGE-L": round(rl, 4),
            })

            if (i + 1) % 10 == 0:
                print(f"    Đã xử lý {i+1}/{len(test_samples)}...")

            time.sleep(0.2)  # Rate limit

        avg_em = float(np.mean(em_scores))
        avg_f1 = float(np.mean(f1_scores))
        avg_rl = float(np.mean(rouge_scores))

        result = {
            "model": model_name,
            "dataset": ds_name,
            "num_samples": len(test_samples),
            "EM": round(avg_em, 4),
            "F1": round(avg_f1, 4),
            "ROUGE-L": round(avg_rl, 4),
        }
        all_results.append(result)

        print(f"    ✅ EM={avg_em:.4f}, F1={avg_f1:.4f}, ROUGE-L={avg_rl:.4f}")

        # Lưu kết quả chi tiết
        os.makedirs("reports/eval", exist_ok=True)
        detail_path = f"reports/eval/{model_name}_{ds_name}_details.json"
        with open(detail_path, "w", encoding="utf-8") as f:
            json.dump(per_sample, f, ensure_ascii=False, indent=2)
        print(f"    💾 Chi tiết: {detail_path}")

# ============================================================
# Báo cáo tổng hợp
# ============================================================
print(f"\n{'='*60}")
print("📋 BÁO CÁO TỔNG HỢP")
print(f"{'='*60}")

df = pd.DataFrame(all_results)
print(df.to_string(index=False))

# Lưu báo cáo
report_path = "reports/eval/api_baseline_report.json"
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)
print(f"\n💾 Báo cáo: {report_path}")

# Hiển thị vài ví dụ
print(f"\n📝 Ví dụ kết quả:")
for ds_name in datasets:
    detail_path = f"reports/eval/{FPT_MODELS[0]}_{ds_name}_details.json"
    if os.path.exists(detail_path):
        with open(detail_path, "r", encoding="utf-8") as f:
            details = json.load(f)
        print(f"\n  --- {ds_name} ---")
        for d in details[:3]:
            print(f"  Q: {d['question'][:80]}")
            print(f"  Ref: {d['reference'][:80]}")
            print(f"  Pred: {d['prediction'][:80]}")
            print(f"  EM={d['EM']}, F1={d['F1']}, ROUGE-L={d['ROUGE-L']}")
            print()
