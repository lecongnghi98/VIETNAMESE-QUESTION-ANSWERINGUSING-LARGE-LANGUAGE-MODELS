"""
Chạy EDA (Exploratory Data Analysis) trên các dataset QA tiếng Việt.

Tạo báo cáo thống kê + biểu đồ phân phối cho:
  1. ViNewsQA (extractive QA từ bài báo)
  2. UIT-ViCoV19QA (generative QA về COVID-19)
  3. ViSpanExtractQA (extractive QA từ HuggingFace)

Output: reports/eda/
"""

import json
import os
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Cấu hình font cho tiếng Việt
rcParams['font.family'] = 'DejaVu Sans'

# ============================================================
# HÀM TẢI DỮ LIỆU
# ============================================================

def load_vinewsqa(base_dir="dataset/ViNewsQA"):
    """Tải ViNewsQA từ các file JSON annotated."""
    samples = []
    for split in ["Train", "Dev", "Test"]:
        split_dir = os.path.join(base_dir, split)
        if not os.path.isdir(split_dir):
            continue
        for fname in sorted(os.listdir(split_dir)):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(split_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            for para in data.get("paragraphs", []):
                context = para.get("context", "")
                for qa in para.get("qas", []):
                    question = qa.get("question", "")
                    answers = qa.get("answers", [])
                    answer = answers[0]["text"] if answers else ""
                    samples.append({
                        "context": context,
                        "question": question,
                        "answer": answer,
                        "split": split,
                        "source": "ViNewsQA",
                    })
    return samples


def load_vicov19qa(base_dir="dataset/UIT-ViCoV19QA"):
    """Tải UIT-ViCoV19QA từ các file CSV."""
    samples = []
    # Dùng thư mục 1_ans (1 câu trả lời)
    ans_dir = os.path.join(base_dir, "1_ans")
    if not os.path.isdir(ans_dir):
        return samples

    for fname in ["train.csv", "dev.csv", "test.csv"]:
        fpath = os.path.join(ans_dir, fname)
        if not os.path.isfile(fpath):
            continue
        df = pd.read_csv(fpath)
        split = fname.replace(".csv", "").capitalize()
        for _, row in df.iterrows():
            question = str(row.get("Question", ""))
            answer = str(row.get("Answer_1", row.get("Answer", "")))
            samples.append({
                "context": "",  # ViCoV19QA không có context riêng
                "question": question,
                "answer": answer,
                "split": split,
                "source": "UIT-ViCoV19QA",
            })
    return samples


def load_vispanextractqa():
    """Tải ViSpanExtractQA từ HuggingFace Hub."""
    try:
        from datasets import load_dataset
        ds = load_dataset("ntphuc149/ViSpanExtractQA", split="train")
        samples = []
        for row in ds:
            samples.append({
                "context": str(row.get("context", "")),
                "question": str(row.get("question", "")),
                "answer": str(row.get("answer_text", row.get("answer", ""))),
                "split": "train",
                "source": "ViSpanExtractQA",
            })
        return samples
    except Exception as e:
        print(f"⚠️ Không thể tải ViSpanExtractQA từ HuggingFace: {e}")
        return []


# ============================================================
# PHÂN TÍCH EDA
# ============================================================

def compute_stats(lengths):
    """Tính thống kê cho một danh sách độ dài."""
    if not lengths:
        return {"count": 0, "min": 0, "max": 0, "mean": 0, "median": 0, "std": 0}
    arr = np.array(lengths, dtype=float)
    return {
        "count": len(arr),
        "min": int(np.min(arr)),
        "max": int(np.max(arr)),
        "mean": round(float(np.mean(arr)), 2),
        "median": round(float(np.median(arr)), 2),
        "std": round(float(np.std(arr)), 2),
        "q25": round(float(np.percentile(arr, 25)), 2),
        "q75": round(float(np.percentile(arr, 75)), 2),
    }


def check_quality(samples):
    """Kiểm tra chất lượng dữ liệu."""
    missing_context = sum(1 for s in samples if not s["context"].strip())
    missing_question = sum(1 for s in samples if not s["question"].strip())
    missing_answer = sum(1 for s in samples if not s["answer"].strip())

    # Duplicates (dựa trên question)
    questions = [s["question"] for s in samples]
    duplicate_count = len(questions) - len(set(questions))

    # Unicode issues
    unicode_issues = 0
    for s in samples:
        for field in ["context", "question", "answer"]:
            text = s[field]
            if text and unicodedata.normalize("NFC", text) != text:
                unicode_issues += 1
                break

    return {
        "missing_context": missing_context,
        "missing_question": missing_question,
        "missing_answer": missing_answer,
        "duplicate_questions": duplicate_count,
        "unicode_issues": unicode_issues,
    }


def run_eda_for_dataset(samples, dataset_name, output_dir):
    """Chạy EDA đầy đủ cho một dataset."""
    print(f"\n{'='*60}")
    print(f"📊 EDA: {dataset_name} ({len(samples)} mẫu)")
    print(f"{'='*60}")

    # Tính độ dài (số từ)
    context_lengths = [len(s["context"].split()) for s in samples]
    question_lengths = [len(s["question"].split()) for s in samples]
    answer_lengths = [len(s["answer"].split()) for s in samples]

    # Tính độ dài (số ký tự)
    context_char_lengths = [len(s["context"]) for s in samples]
    question_char_lengths = [len(s["question"]) for s in samples]
    answer_char_lengths = [len(s["answer"]) for s in samples]

    # Thống kê
    stats = {
        "dataset_name": dataset_name,
        "total_samples": len(samples),
        "splits": {},
        "word_length": {
            "context": compute_stats(context_lengths),
            "question": compute_stats(question_lengths),
            "answer": compute_stats(answer_lengths),
        },
        "char_length": {
            "context": compute_stats(context_char_lengths),
            "question": compute_stats(question_char_lengths),
            "answer": compute_stats(answer_char_lengths),
        },
        "quality": check_quality(samples),
    }

    # Thống kê theo split
    splits = set(s["split"] for s in samples)
    for split in sorted(splits):
        split_samples = [s for s in samples if s["split"] == split]
        stats["splits"][split] = len(split_samples)

    # In kết quả
    print(f"\n  Tổng mẫu: {stats['total_samples']}")
    print(f"  Splits: {stats['splits']}")
    print(f"\n  Độ dài (số từ):")
    for field in ["context", "question", "answer"]:
        s = stats["word_length"][field]
        print(f"    {field:10s}: mean={s['mean']:6.1f}, median={s['median']:6.1f}, "
              f"min={s['min']:4d}, max={s['max']:5d}, std={s['std']:6.1f}")
    print(f"\n  Chất lượng dữ liệu:")
    q = stats["quality"]
    print(f"    Missing context:  {q['missing_context']}")
    print(f"    Missing question: {q['missing_question']}")
    print(f"    Missing answer:   {q['missing_answer']}")
    print(f"    Duplicate questions: {q['duplicate_questions']}")
    print(f"    Unicode issues:   {q['unicode_issues']}")

    # Vẽ biểu đồ
    ds_output_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(ds_output_dir, exist_ok=True)

    # Biểu đồ phân phối độ dài (số từ)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{dataset_name} — Phan phoi do dai (so tu)", fontsize=14)

    fields_data = [
        ("Context", context_lengths, "steelblue"),
        ("Question", question_lengths, "darkorange"),
        ("Answer", answer_lengths, "forestgreen"),
    ]

    for ax, (name, lengths, color) in zip(axes, fields_data):
        if not lengths or max(lengths) == 0:
            ax.set_title(f"{name} (khong co du lieu)")
            continue
        # Loại bỏ outliers cho visualization
        q99 = np.percentile(lengths, 99) if lengths else 1
        filtered = [l for l in lengths if l <= q99]

        ax.hist(filtered, bins=50, edgecolor="black", alpha=0.7, color=color)
        ax.set_title(f"{name} (n={len(lengths)})")
        ax.set_xlabel("So tu")
        ax.set_ylabel("So mau")
        ax.axvline(np.mean(lengths), color="red", linestyle="--",
                   label=f"Mean: {np.mean(lengths):.1f}")
        ax.axvline(np.median(lengths), color="green", linestyle="-.",
                   label=f"Median: {np.median(lengths):.1f}")
        ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(os.path.join(ds_output_dir, "word_length_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  ✅ Biểu đồ: {ds_output_dir}/word_length_distribution.png")

    # Biểu đồ boxplot
    fig, ax = plt.subplots(figsize=(10, 6))
    data_to_plot = []
    labels = []
    for name, lengths in [("Context", context_lengths), ("Question", question_lengths), ("Answer", answer_lengths)]:
        if lengths:
            data_to_plot.append(lengths)
            labels.append(f"{name}\n(n={len(lengths)})")

    if data_to_plot:
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                        showfliers=False)  # Ẩn outliers cho dễ đọc
        colors = ["steelblue", "darkorange", "forestgreen"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_title(f"{dataset_name} — Boxplot do dai (so tu)")
        ax.set_ylabel("So tu")
        ax.grid(axis="y", alpha=0.3)

    fig.savefig(os.path.join(ds_output_dir, "boxplot_word_length.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Biểu đồ: {ds_output_dir}/boxplot_word_length.png")

    # Lưu stats JSON
    stats_file = os.path.join(ds_output_dir, "eda_stats.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"  ✅ Thống kê: {stats_file}")

    return stats


def plot_comparison(all_stats, output_dir):
    """Vẽ biểu đồ so sánh giữa các dataset."""
    if len(all_stats) < 2:
        return

    print(f"\n{'='*60}")
    print(f"📊 SO SÁNH GIỮA CÁC DATASET")
    print(f"{'='*60}")

    # So sánh mean word length
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("So sanh do dai trung binh giua cac dataset (so tu)", fontsize=13)

    fields = ["context", "question", "answer"]
    colors = ["steelblue", "darkorange", "forestgreen"]

    for ax, field, color in zip(axes, fields, colors):
        names = []
        means = []
        stds = []
        for stats in all_stats:
            names.append(stats["dataset_name"])
            means.append(stats["word_length"][field]["mean"])
            stds.append(stats["word_length"][field]["std"])

        bars = ax.bar(names, means, color=color, alpha=0.7, edgecolor="black")
        ax.errorbar(names, means, yerr=stds, fmt="none", color="black", capsize=5)
        ax.set_title(f"{field.capitalize()}")
        ax.set_ylabel("So tu (trung binh)")
        ax.tick_params(axis='x', rotation=15)

        # Thêm giá trị lên bar
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                    f"{mean:.1f}", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "comparison_mean_length.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Biểu đồ so sánh: {output_dir}/comparison_mean_length.png")

    # Bảng so sánh tổng hợp
    print(f"\n  {'Dataset':<20s} {'Samples':>8s} {'Ctx(từ)':>10s} {'Q(từ)':>10s} {'A(từ)':>10s} {'Missing':>8s} {'Dupes':>8s}")
    print(f"  {'-'*20} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
    for stats in all_stats:
        q = stats["quality"]
        total_missing = q["missing_context"] + q["missing_question"] + q["missing_answer"]
        print(f"  {stats['dataset_name']:<20s} "
              f"{stats['total_samples']:>8d} "
              f"{stats['word_length']['context']['mean']:>10.1f} "
              f"{stats['word_length']['question']['mean']:>10.1f} "
              f"{stats['word_length']['answer']['mean']:>10.1f} "
              f"{total_missing:>8d} "
              f"{q['duplicate_questions']:>8d}")

    # Lưu comparison JSON
    comparison_file = os.path.join(output_dir, "comparison_summary.json")
    comparison_data = {
        "datasets": [
            {
                "name": s["dataset_name"],
                "total_samples": s["total_samples"],
                "splits": s["splits"],
                "avg_context_words": s["word_length"]["context"]["mean"],
                "avg_question_words": s["word_length"]["question"]["mean"],
                "avg_answer_words": s["word_length"]["answer"]["mean"],
                "quality": s["quality"],
            }
            for s in all_stats
        ]
    }
    with open(comparison_file, "w", encoding="utf-8") as f:
        json.dump(comparison_data, f, ensure_ascii=False, indent=2)
    print(f"\n  ✅ So sánh tổng hợp: {comparison_file}")


# ============================================================
# MAIN
# ============================================================

def main():
    output_dir = "reports/eda"
    os.makedirs(output_dir, exist_ok=True)

    all_stats = []

    # 1. ViNewsQA
    print("\n⏳ Đang tải ViNewsQA...")
    vinewsqa_samples = load_vinewsqa()
    if vinewsqa_samples:
        stats = run_eda_for_dataset(vinewsqa_samples, "ViNewsQA", output_dir)
        all_stats.append(stats)
    else:
        print("  ⚠️ Không tìm thấy dữ liệu ViNewsQA")

    # 2. UIT-ViCoV19QA
    print("\n⏳ Đang tải UIT-ViCoV19QA...")
    vicov19_samples = load_vicov19qa()
    if vicov19_samples:
        stats = run_eda_for_dataset(vicov19_samples, "UIT-ViCoV19QA", output_dir)
        all_stats.append(stats)
    else:
        print("  ⚠️ Không tìm thấy dữ liệu UIT-ViCoV19QA")

    # 3. ViSpanExtractQA (từ HuggingFace — tùy chọn)
    if "--with-hf" in sys.argv:
        print("\n⏳ Đang tải ViSpanExtractQA từ HuggingFace...")
        vispan_samples = load_vispanextractqa()
        if vispan_samples:
            stats = run_eda_for_dataset(vispan_samples, "ViSpanExtractQA", output_dir)
            all_stats.append(stats)
        else:
            print("  ⚠️ Không thể tải ViSpanExtractQA")

    # So sánh
    if all_stats:
        plot_comparison(all_stats, output_dir)

    # Lưu báo cáo tổng hợp
    full_report_file = os.path.join(output_dir, "eda_report.json")
    with open(full_report_file, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"✅ EDA HOÀN TẤT!")
    print(f"   Output: {output_dir}/")
    print(f"   Báo cáo: {full_report_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
