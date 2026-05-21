"""
Prompt Engineering — So sánh các chiến lược prompt trên FPT Cloud API.
Chạy: python3 run_prompt_engineering.py

Yêu cầu: export FPT_API_KEY='your-api-key'

Thiết kế prompt theo framework: Template + Strategy + Format
  - Template: Role definition + persona phù hợp task
  - Strategy: Kỹ thuật prompting (extractive, few-shot, CoT, adaptive...)
  - Format: Output constraints + structure rõ ràng

Chiến lược prompt:
  P0: Baseline (từ run_api_baseline.py) — đã có kết quả, không chạy lại
  P1: Extractive — trích xuất nguyên văn từ context (tối ưu cho QA có context)
  P2: Short answer — trả lời cực ngắn, format nghiêm ngặt
  P3: Few-shot — cho ví dụ mẫu phù hợp từng loại dataset
  P4: Chain-of-thought — suy luận rồi trích xuất
  P5: Adaptive — tự động điều chỉnh độ dài/phong cách theo loại câu hỏi
  P6: Role-play Expert — nhập vai chuyên gia lĩnh vực, trả lời có chiều sâu
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
MODELS = ["Llama-3.3-70B-Instruct","gpt-oss-120b"]
MAX_TOKENS_MAP = {"P1": 128, "P2": 64, "P3": 128, "P4": 256, "P5": 512, "P6": 512, "P7": 512}
TEMPERATURE = 0.1
MAX_SAMPLES = None
MAX_SAMPLES_PER_DATASET = {"ViSpanExtractQA": 2000}  # Giới hạn riêng cho từng dataset

api_key = os.environ.get("FPT_API_KEY")
if not api_key:
    print("❌ Chưa thiết lập FPT_API_KEY!")
    sys.exit(1)

client = OpenAI(api_key=api_key, base_url=FPT_API_BASE_URL)
print("✅ FPT Cloud API đã kết nối")
print(f"📋 Models: {', '.join(MODELS)}")

# ============================================================
# Tải Dataset
# ============================================================

def load_vinewsqa(split_dir):
    samples = []
    for f in sorted(Path(split_dir).glob("*.json")):
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        for para in data.get("paragraphs", []):
            context = para.get("context", "")
            for qa in para.get("qas", []):
                question = qa.get("question", "")
                answers = qa.get("answers", [])
                if answers:
                    samples.append({"context": context, "question": question,
                                    "answer": answers[0].get("text", "")})
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
            samples.append({"context": ctx, "question": q, "answer": a})
    return samples


print("\n📦 Đang tải datasets...")
datasets = {}

if Path("dataset/ViNewsQA/Test").exists():
    datasets["ViNewsQA"] = load_vinewsqa("dataset/ViNewsQA/Test")
    print(f"  ViNewsQA Test: {len(datasets['ViNewsQA'])} mẫu")

if Path("dataset/ViSpanExtractQA/test").exists():
    datasets["ViSpanExtractQA"] = load_vispanextractqa("dataset/ViSpanExtractQA/test")
    print(f"  ViSpanExtractQA Test: {len(datasets['ViSpanExtractQA'])} mẫu")

if not datasets:
    print("❌ Không tìm thấy dataset!")
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
# Định nghĩa các chiến lược Prompt
# ============================================================

# ============================================================
# TEMPLATE: Role definition rõ ràng cho từng persona
# STRATEGY: Kỹ thuật prompting phù hợp từng loại task
# FORMAT: Output constraints + structure
# ============================================================

# --- P1: Extractive — trích xuất nguyên văn (tối ưu cho QA có context) ---
# Template: Chuyên gia trích xuất | Strategy: Span extraction | Format: Cụm từ nguyên văn
P1_SYSTEM_CTX = (
    "Vai trò: Bạn là hệ thống trích xuất câu trả lời (extractive QA) cho tiếng Việt.\n\n"
    "Chiến lược:\n"
    "1. Đọc kỹ ngữ cảnh và xác định câu chứa thông tin trả lời.\n"
    "2. Trích xuất cụm từ ngắn nhất từ ngữ cảnh mà trả lời đúng câu hỏi.\n\n"
    "Quy tắc output:\n"
    "- CHỈ trả lời bằng cụm từ copy nguyên văn từ ngữ cảnh.\n"
    "- KHÔNG thêm từ nào ngoài ngữ cảnh.\n"
    "- KHÔNG viết thành câu hoàn chỉnh.\n"
    "- Nếu câu trả lời là số liệu, trích cả đơn vị kèm theo."
)
P1_USER_CTX = (
    "Ngữ cảnh:\n\"\"\"\n{context}\n\"\"\"\n\n"
    "Câu hỏi: {question}\n\n"
    "Trích xuất đáp án từ ngữ cảnh:"
)

# Khi không có context → chuyển sang generative ngắn gọn
P1_SYSTEM_NO_CTX = (
    "Vai trò: Bạn là hệ thống trả lời câu hỏi tiếng Việt.\n\n"
    "Chiến lược: Trả lời trực tiếp, đầy đủ thông tin cần thiết.\n\n"
    "Quy tắc output:\n"
    "- Trả lời đúng trọng tâm câu hỏi.\n"
    "- Nếu câu hỏi yêu cầu giải thích, hãy giải thích ngắn gọn.\n"
    "- Nếu câu hỏi yêu cầu liệt kê, hãy liệt kê đầy đủ.\n"
    "- Không lặp lại câu hỏi. Không thêm lời chào."
)
P1_USER_NO_CTX = "Câu hỏi: {question}\n\nTrả lời:"

# --- P2: Short answer — cực ngắn, format nghiêm ngặt ---
# Template: Hệ thống đáp án ngắn | Strategy: Constrained generation | Format: <10 từ
P2_SYSTEM_CTX = (
    "Vai trò: Hệ thống trả lời câu hỏi dạng ngắn (short-answer QA).\n\n"
    "Chiến lược: Trích xuất trực tiếp từ ngữ cảnh, ưu tiên cụm danh từ/số liệu.\n\n"
    "Quy tắc output:\n"
    "- Trả lời DƯỚI 10 từ.\n"
    "- Trích xuất trực tiếp từ ngữ cảnh.\n"
    "- Không bắt đầu bằng \"Theo...\", \"Dựa vào...\", hay lặp lại câu hỏi.\n"
    "- Không thêm giải thích."
)
P2_USER_CTX = (
    "Ngữ cảnh:\n\"\"\"\n{context}\n\"\"\"\n\n"
    "Câu hỏi: {question}\n\n"
    "Đáp án ngắn:"
)

P2_SYSTEM_NO_CTX = (
    "Vai trò: Hệ thống trả lời câu hỏi tiếng Việt dạng ngắn.\n\n"
    "Chiến lược: Đưa ra đáp án cốt lõi, đủ ý nhưng không dài dòng.\n\n"
    "Quy tắc output:\n"
    "- Trả lời đúng trọng tâm, tối đa 2-3 câu.\n"
    "- Đi thẳng vào đáp án.\n"
    "- Không lặp lại câu hỏi.\n"
    "- Không thêm lời chào hay giải thích thừa."
)
P2_USER_NO_CTX = "Câu hỏi: {question}\n\nĐáp án:"

# --- P3: Few-shot — cho ví dụ mẫu phù hợp từng loại dataset ---
# Template: Hệ thống QA | Strategy: In-context learning | Format: Theo mẫu
P3_SYSTEM_CTX = (
    "Vai trò: Hệ thống trả lời câu hỏi tiếng Việt dựa trên ngữ cảnh.\n\n"
    "Chiến lược: Học từ các ví dụ mẫu bên dưới, trả lời theo đúng phong cách.\n"
    "- Trích xuất thông tin chính xác từ ngữ cảnh.\n"
    "- Giữ đáp án ngắn gọn, đúng trọng tâm."
)
P3_FEWSHOT_CTX = [
    {"role": "user", "content": (
        "Ngữ cảnh:\n\"\"\"\nBệnh viện Bạch Mai được thành lập năm 1911, là bệnh viện đa khoa "
        "hạng đặc biệt tuyến cuối của miền Bắc với quy mô 3.000 giường bệnh.\n\"\"\"\n\n"
        "Câu hỏi: Bệnh viện Bạch Mai được thành lập năm nào?\n\nĐáp án:"
    )},
    {"role": "assistant", "content": "năm 1911"},
    {"role": "user", "content": (
        "Ngữ cảnh:\n\"\"\"\nTheo thống kê, Việt Nam có khoảng 96 triệu dân, trong đó 54 dân tộc "
        "cùng sinh sống trên lãnh thổ.\n\"\"\"\n\n"
        "Câu hỏi: Việt Nam có bao nhiêu dân tộc?\n\nĐáp án:"
    )},
    {"role": "assistant", "content": "54 dân tộc"},
]

P3_SYSTEM_NO_CTX = (
    "Vai trò: Hệ thống trả lời câu hỏi y tế và sức khỏe bằng tiếng Việt.\n\n"
    "Chiến lược: Học từ các ví dụ mẫu bên dưới, trả lời theo đúng phong cách.\n"
    "- Trả lời đầy đủ, chính xác theo kiến thức y khoa.\n"
    "- Nếu câu hỏi yêu cầu hướng dẫn, liệt kê các bước/điểm chính.\n"
    "- Giữ ngôn ngữ chuyên môn nhưng dễ hiểu."
)
P3_FEWSHOT_NO_CTX = [
    {"role": "user", "content": (
        "Câu hỏi: Virus SARS-CoV-2 lây truyền chủ yếu qua đường nào?\n\nĐáp án:"
    )},
    {"role": "assistant", "content": (
        "Virus SARS-CoV-2 lây truyền chủ yếu qua đường hô hấp thông qua các giọt bắn "
        "(droplet) khi người bệnh ho, hắt hơi, nói chuyện. Ngoài ra còn lây qua tiếp xúc "
        "với bề mặt nhiễm virus rồi đưa tay lên mắt, mũi, miệng."
    )},
    {"role": "user", "content": (
        "Câu hỏi: Triệu chứng phổ biến của COVID-19 là gì?\n\nĐáp án:"
    )},
    {"role": "assistant", "content": (
        "Triệu chứng phổ biến bao gồm: sốt, ho khan, mệt mỏi, đau họng, đau đầu, "
        "đau nhức cơ thể, mất vị giác hoặc khứu giác. Trường hợp nặng có thể khó thở, "
        "đau tức ngực, viêm phổi."
    )},
]

# --- P4: Chain-of-thought extractive ---
# Template: Hệ thống suy luận | Strategy: CoT + extraction | Format: Suy luận → Đáp án
P4_SYSTEM_CTX = (
    "Vai trò: Hệ thống trả lời câu hỏi với khả năng suy luận.\n\n"
    "Chiến lược: Thực hiện 2 bước:\n"
    "  Bước 1 — Xác định: Tìm câu/đoạn trong ngữ cảnh chứa thông tin trả lời.\n"
    "  Bước 2 — Trích xuất: Lấy cụm từ ngắn nhất trả lời chính xác câu hỏi.\n\n"
    "Format output (bắt buộc):\n"
    "Suy luận: <trích dẫn câu chứa đáp án từ ngữ cảnh>\n"
    "Đáp án: <cụm từ ngắn nhất>"
)
P4_USER_CTX = (
    "Ngữ cảnh:\n\"\"\"\n{context}\n\"\"\"\n\n"
    "Câu hỏi: {question}"
)

P4_SYSTEM_NO_CTX = (
    "Vai trò: Hệ thống trả lời câu hỏi tiếng Việt với khả năng suy luận.\n\n"
    "Chiến lược: Thực hiện 2 bước:\n"
    "  Bước 1 — Phân tích: Xác định loại câu hỏi và thông tin cần trả lời.\n"
    "  Bước 2 — Trả lời: Đưa ra đáp án đầy đủ, chính xác.\n\n"
    "Format output (bắt buộc):\n"
    "Suy luận: <phân tích ngắn gọn>\n"
    "Đáp án: <câu trả lời đầy đủ>"
)
P4_USER_NO_CTX = "Câu hỏi: {question}"

# --- P5: Adaptive — tự điều chỉnh theo loại câu hỏi ---
# Template: Trợ lý thông minh | Strategy: Task classification + adaptive response | Format: Tự động
P5_SYSTEM_CTX = (
    "Vai trò: Bạn là trợ lý AI thông minh chuyên trả lời câu hỏi tiếng Việt.\n\n"
    "Chiến lược thích ứng — phân loại câu hỏi rồi điều chỉnh cách trả lời:\n"
    "- Câu hỏi THỰC THỂ (ai, ở đâu, năm nào, bao nhiêu): → Trích xuất cụm từ ngắn từ ngữ cảnh.\n"
    "- Câu hỏi GIẢI THÍCH (tại sao, như thế nào, cách nào): → Trả lời 1-2 câu dựa trên ngữ cảnh.\n"
    "- Câu hỏi LIỆT KÊ (những gì, các loại nào): → Liệt kê ngắn gọn từ ngữ cảnh.\n\n"
    "Quy tắc output:\n"
    "- Mọi thông tin phải có trong ngữ cảnh.\n"
    "- Không thêm kiến thức bên ngoài.\n"
    "- Không lặp lại câu hỏi."
)
P5_USER_CTX = (
    "Ngữ cảnh:\n\"\"\"\n{context}\n\"\"\"\n\n"
    "Câu hỏi: {question}\n\n"
    "Trả lời:"
)

P5_SYSTEM_NO_CTX = (
    "Vai trò: Bạn là trợ lý AI thông minh chuyên trả lời câu hỏi tiếng Việt.\n\n"
    "Chiến lược thích ứng — phân loại câu hỏi rồi điều chỉnh cách trả lời:\n"
    "- Câu hỏi THỰC THỂ (ai, ở đâu, năm nào, bao nhiêu): → Đáp án ngắn, chính xác.\n"
    "- Câu hỏi GIẢI THÍCH (tại sao, như thế nào, cách nào): → Giải thích rõ ràng, đầy đủ.\n"
    "- Câu hỏi HƯỚNG DẪN (cách điều trị, phòng ngừa, xử lý): → Liệt kê các bước/điểm chính.\n"
    "- Câu hỏi ĐỊNH NGHĨA (là gì, nghĩa là gì): → Định nghĩa ngắn gọn + bổ sung nếu cần.\n\n"
    "Quy tắc output:\n"
    "- Trả lời đầy đủ, đúng trọng tâm.\n"
    "- Dùng ngôn ngữ chuyên môn nhưng dễ hiểu.\n"
    "- Không lặp lại câu hỏi. Không thêm lời chào."
)
P5_USER_NO_CTX = "Câu hỏi: {question}\n\nTrả lời:"

# --- P6: Role-play Expert — nhập vai chuyên gia lĩnh vực ---
# Template: Chuyên gia lĩnh vực | Strategy: Persona-based generation | Format: Chuyên sâu
P6_SYSTEM_CTX = (
    "Vai trò: Bạn là chuyên gia phân tích văn bản tiếng Việt với nhiều năm kinh nghiệm.\n\n"
    "Chiến lược:\n"
    "1. Đọc kỹ ngữ cảnh, xác định các thông tin quan trọng liên quan đến câu hỏi.\n"
    "2. Tổng hợp và trả lời chính xác dựa trên ngữ cảnh.\n"
    "3. Nếu ngữ cảnh chứa số liệu, trích dẫn chính xác.\n\n"
    "Quy tắc output:\n"
    "- Trả lời dựa hoàn toàn trên ngữ cảnh.\n"
    "- Ngắn gọn nhưng đầy đủ thông tin.\n"
    "- Không suy diễn ngoài ngữ cảnh."
)
P6_USER_CTX = (
    "Ngữ cảnh:\n\"\"\"\n{context}\n\"\"\"\n\n"
    "Câu hỏi: {question}\n\n"
    "Phân tích và trả lời:"
)

P6_SYSTEM_NO_CTX = (
    "Vai trò: Bạn là bác sĩ chuyên khoa với kiến thức sâu rộng về y tế và dịch bệnh, "
    "đặc biệt về COVID-19 và các bệnh truyền nhiễm.\n\n"
    "Chiến lược:\n"
    "1. Phân tích câu hỏi để xác định lĩnh vực chuyên môn cần trả lời.\n"
    "2. Trả lời với độ chi tiết phù hợp:\n"
    "   - Câu hỏi về triệu chứng/chẩn đoán: mô tả đầy đủ các dấu hiệu.\n"
    "   - Câu hỏi về điều trị/thuốc: liệt kê phác đồ, liều lượng nếu biết.\n"
    "   - Câu hỏi về phòng ngừa: hướng dẫn cụ thể, thực tế.\n"
    "3. Sử dụng thuật ngữ y khoa chính xác kèm giải thích dễ hiểu.\n\n"
    "Quy tắc output:\n"
    "- Trả lời đầy đủ, có cấu trúc.\n"
    "- Nếu cần liệt kê, dùng gạch đầu dòng.\n"
    "- Không lặp lại câu hỏi. Đi thẳng vào nội dung."
)
P6_USER_NO_CTX = "Câu hỏi: {question}\n\nTrả lời chuyên môn:"

# --- P7: Plain Text Consultant — tối ưu cho model hay format markdown ---
# Template: Tư vấn viên y tế | Strategy: Anti-markdown + conversational style | Format: Text thuần
P7_SYSTEM_CTX = (
    "Bạn là trợ lý trả lời câu hỏi dựa trên ngữ cảnh. "
    "Trả lời ngắn gọn bằng text thuần. "
    "KHÔNG dùng markdown, bold, bảng, hay bullet points. "
    "Chỉ trích xuất thông tin từ ngữ cảnh."
)
P7_USER_CTX = (
    "Ngữ cảnh:\n\"\"\"\n{context}\n\"\"\"\n\n"
    "Câu hỏi: {question}\n\n"
    "Trả lời:"
)

P7_SYSTEM_NO_CTX = (
    "Bạn là tư vấn viên y tế thân thiện, trả lời câu hỏi về COVID-19 và sức khỏe bằng tiếng Việt. "
    "Trả lời như đang tư vấn trực tiếp cho bệnh nhân. "
    "Bắt đầu bằng lời chào ngắn (Chào anh/chị). "
    "Trả lời đầy đủ, dễ hiểu, dùng ngôn ngữ đời thường. "
    "TUYỆT ĐỐI KHÔNG dùng markdown, bold (**), bảng, hay ký hiệu đặc biệt. "
    "Chỉ viết text thuần túy."
)
P7_USER_NO_CTX = "Câu hỏi: {question}\n\nTrả lời:"


# ============================================================
# Hàm gọi API cho từng chiến lược
# ============================================================

def call_api(messages, max_tokens, model, max_retries=3):
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages,
                max_tokens=max_tokens, temperature=TEMPERATURE,
            )
            content = resp.choices[0].message.content
            return content.strip() if content else ""
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))
            else:
                print(f"    ❌ API error: {e}")
                return ""


def extract_answer_p4(text):
    """Trích xuất phần 'Đáp án:' từ output P4 (chain-of-thought)."""
    for line in text.split("\n"):
        line = line.strip()
        if line.lower().startswith("đáp án:"):
            return line.split(":", 1)[1].strip()
    # Fallback: lấy dòng cuối
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return lines[-1] if lines else text


def build_messages(strategy, context, question):
    """Tạo messages cho từng chiến lược prompt.
    
    Thiết kế theo framework Template + Strategy + Format:
    - Template: Role definition trong system message
    - Strategy: Kỹ thuật prompting (extractive, few-shot, CoT, adaptive, role-play)
    - Format: Output constraints trong system + user message
    """
    has_ctx = bool(context)

    if strategy == "P1":
        sys_msg = P1_SYSTEM_CTX if has_ctx else P1_SYSTEM_NO_CTX
        usr_tpl = P1_USER_CTX if has_ctx else P1_USER_NO_CTX
        usr_msg = usr_tpl.format(context=context, question=question)
        return [{"role": "system", "content": sys_msg},
                {"role": "user", "content": usr_msg}]

    elif strategy == "P2":
        sys_msg = P2_SYSTEM_CTX if has_ctx else P2_SYSTEM_NO_CTX
        usr_tpl = P2_USER_CTX if has_ctx else P2_USER_NO_CTX
        usr_msg = usr_tpl.format(context=context, question=question)
        return [{"role": "system", "content": sys_msg},
                {"role": "user", "content": usr_msg}]

    elif strategy == "P3":
        sys_msg = P3_SYSTEM_CTX if has_ctx else P3_SYSTEM_NO_CTX
        fewshot = P3_FEWSHOT_CTX if has_ctx else P3_FEWSHOT_NO_CTX
        if has_ctx:
            usr_msg = (
                f"Ngữ cảnh:\n\"\"\"\n{context}\n\"\"\"\n\n"
                f"Câu hỏi: {question}\n\nĐáp án:"
            )
        else:
            usr_msg = f"Câu hỏi: {question}\n\nĐáp án:"
        return [{"role": "system", "content": sys_msg}] + fewshot + \
               [{"role": "user", "content": usr_msg}]

    elif strategy == "P4":
        sys_msg = P4_SYSTEM_CTX if has_ctx else P4_SYSTEM_NO_CTX
        usr_tpl = P4_USER_CTX if has_ctx else P4_USER_NO_CTX
        usr_msg = usr_tpl.format(context=context, question=question)
        return [{"role": "system", "content": sys_msg},
                {"role": "user", "content": usr_msg}]

    elif strategy == "P5":
        sys_msg = P5_SYSTEM_CTX if has_ctx else P5_SYSTEM_NO_CTX
        usr_tpl = P5_USER_CTX if has_ctx else P5_USER_NO_CTX
        usr_msg = usr_tpl.format(context=context, question=question)
        return [{"role": "system", "content": sys_msg},
                {"role": "user", "content": usr_msg}]

    elif strategy == "P6":
        sys_msg = P6_SYSTEM_CTX if has_ctx else P6_SYSTEM_NO_CTX
        usr_tpl = P6_USER_CTX if has_ctx else P6_USER_NO_CTX
        usr_msg = usr_tpl.format(context=context, question=question)
        return [{"role": "system", "content": sys_msg},
                {"role": "user", "content": usr_msg}]

    elif strategy == "P7":
        sys_msg = P7_SYSTEM_CTX if has_ctx else P7_SYSTEM_NO_CTX
        usr_tpl = P7_USER_CTX if has_ctx else P7_USER_NO_CTX
        usr_msg = usr_tpl.format(context=context, question=question)
        return [{"role": "system", "content": sys_msg},
                {"role": "user", "content": usr_msg}]

    raise ValueError(f"Unknown strategy: {strategy}")


# ============================================================
# Chạy đánh giá
# ============================================================

STRATEGIES = ["P1", "P4"]
STRATEGY_NAMES = {
    "P1": "Extractive",
    "P2": "Short Answer",
    "P3": "Few-shot",
    "P4": "Chain-of-Thought",
    "P5": "Adaptive",
    "P6": "Role-play Expert",
    "P7": "Plain Text Consultant",
}

all_results = []

for current_model in MODELS:
    print(f"\n{'#'*70}")
    print(f"🤖 MODEL: {current_model}")
    print(f"{'#'*70}")

    for strategy in STRATEGIES:
        print(f"\n{'='*60}")
        print(f"🧪 [{current_model}] Chiến lược: {strategy} — {STRATEGY_NAMES[strategy]}")
        print(f"{'='*60}")

        max_tok = MAX_TOKENS_MAP[strategy]

        for ds_name, samples in datasets.items():
            test_samples = samples[:MAX_SAMPLES] if MAX_SAMPLES else samples
            # Áp dụng giới hạn riêng cho từng dataset
            ds_limit = MAX_SAMPLES_PER_DATASET.get(ds_name)
            if ds_limit and len(test_samples) > ds_limit:
                test_samples = test_samples[:ds_limit]

            # === RESUME: Đọc kết quả đã lưu và tiếp tục từ chỗ dừng ===
            detail_path = f"reports/eval/prompt_eng/{strategy}_{current_model}_{ds_name}_details.json"
            per_sample = []
            resume_from = 0
            if os.path.exists(detail_path):
                with open(detail_path, "r", encoding="utf-8") as f:
                    per_sample = json.load(f)
                # Chỉ resume nếu chưa hoàn thành
                if len(per_sample) < len(test_samples):
                    resume_from = len(per_sample)
                    print(f"\n  📊 {ds_name} — RESUME từ mẫu {resume_from}/{len(test_samples)}")
                else:
                    print(f"\n  📊 {ds_name} — ĐÃ HOÀN THÀNH ({len(per_sample)} mẫu), bỏ qua.")
                    # Tính lại metrics từ kết quả đã lưu
                    em_scores = [s["EM"] for s in per_sample]
                    f1_scores = [s["F1"] for s in per_sample]
                    rouge_scores = [s["ROUGE-L"] for s in per_sample]
                    avg_em = float(np.mean(em_scores))
                    avg_f1 = float(np.mean(f1_scores))
                    avg_rl = float(np.mean(rouge_scores))
                    result = {
                        "strategy": strategy, "strategy_name": STRATEGY_NAMES[strategy],
                        "model": current_model, "dataset": ds_name,
                        "num_samples": len(per_sample),
                        "EM": round(avg_em, 4), "F1": round(avg_f1, 4), "ROUGE-L": round(avg_rl, 4),
                    }
                    all_results.append(result)
                    print(f"    ✅ EM={avg_em:.4f}, F1={avg_f1:.4f}, ROUGE-L={avg_rl:.4f}")
                    continue
            else:
                print(f"\n  📊 {ds_name} ({len(test_samples)} mẫu, max_tokens={max_tok})")

            em_scores = [s["EM"] for s in per_sample]
            f1_scores = [s["F1"] for s in per_sample]
            rouge_scores = [s["ROUGE-L"] for s in per_sample]

            for i in range(resume_from, len(test_samples)):
                sample = test_samples[i]
                msgs = build_messages(strategy, sample["context"], sample["question"])
                raw_pred = call_api(msgs, max_tok, current_model)

                # P4: trích xuất phần đáp án
                pred = extract_answer_p4(raw_pred) if strategy == "P4" else raw_pred
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
                    "raw_prediction": raw_pred[:200],
                    "prediction": pred[:100],
                    "EM": em, "F1": round(f1, 4), "ROUGE-L": round(rl, 4),
                })

                if (i + 1) % 10 == 0:
                    cur_f1 = np.mean(f1_scores)
                    print(f"    [{i+1}/{len(test_samples)}] F1={cur_f1:.4f}")

                # Lưu checkpoint mỗi 50 mẫu
                if (i + 1) % 50 == 0:
                    os.makedirs("reports/eval/prompt_eng", exist_ok=True)
                    with open(detail_path, "w", encoding="utf-8") as f:
                        json.dump(per_sample, f, ensure_ascii=False, indent=2)

                time.sleep(1)

            avg_em = float(np.mean(em_scores))
            avg_f1 = float(np.mean(f1_scores))
            avg_rl = float(np.mean(rouge_scores))

            result = {
                "strategy": strategy,
                "strategy_name": STRATEGY_NAMES[strategy],
                "model": current_model,
                "dataset": ds_name,
                "num_samples": len(test_samples),
                "EM": round(avg_em, 4),
                "F1": round(avg_f1, 4),
                "ROUGE-L": round(avg_rl, 4),
            }
            all_results.append(result)
            print(f"    ✅ EM={avg_em:.4f}, F1={avg_f1:.4f}, ROUGE-L={avg_rl:.4f}")

            # Lưu chi tiết
            os.makedirs("reports/eval/prompt_eng", exist_ok=True)
            with open(detail_path, "w", encoding="utf-8") as f:
                json.dump(per_sample, f, ensure_ascii=False, indent=2)
            print(f"    💾 {detail_path}")


# ============================================================
# Báo cáo tổng hợp + so sánh với Baseline
# ============================================================

# Thêm baseline P0 vào bảng so sánh
baseline_path = "reports/eval/api_baseline_report.json"
if Path(baseline_path).exists():
    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline = json.load(f)
    for b in baseline:
        # Chỉ thêm baseline cho model đã có trong kết quả
        if b["model"] in MODELS:
            all_results.insert(0, {
                "strategy": "P0",
                "strategy_name": "Baseline",
                "model": b["model"],
                "dataset": b["dataset"],
                "num_samples": b["num_samples"],
                "EM": b["EM"],
                "F1": b["F1"],
                "ROUGE-L": b["ROUGE-L"],
            })

# Lưu báo cáo
report_path = "reports/eval/prompt_eng/comparison_report.json"
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

# Hiển thị bảng so sánh
df = pd.DataFrame(all_results)

for current_model in MODELS:
    model_df = df[df["model"] == current_model]
    if model_df.empty:
        continue

    print(f"\n{'='*70}")
    print(f"📋 SO SÁNH CÁC CHIẾN LƯỢC PROMPT — {current_model}")
    print(f"{'='*70}")

    for ds_name in datasets:
        ds_df = model_df[model_df["dataset"] == ds_name].copy()
        if ds_df.empty:
            continue
        ds_df = ds_df[["strategy", "strategy_name", "EM", "F1", "ROUGE-L"]]
        print(f"\n--- {ds_name} ---")
        print(ds_df.to_string(index=False))

        best_f1 = ds_df.loc[ds_df["F1"].idxmax()]
        best_rl = ds_df.loc[ds_df["ROUGE-L"].idxmax()]
        print(f"  🏆 Best F1:      {best_f1['strategy']} ({best_f1['strategy_name']}) = {best_f1['F1']}")
        print(f"  🏆 Best ROUGE-L: {best_rl['strategy']} ({best_rl['strategy_name']}) = {best_rl['ROUGE-L']}")

# So sánh giữa các model
if len(MODELS) > 1:
    print(f"\n{'='*70}")
    print("📊 SO SÁNH GIỮA CÁC MODEL (Best strategy mỗi dataset)")
    print(f"{'='*70}")

    for ds_name in datasets:
        print(f"\n--- {ds_name} ---")
        ds_df = df[df["dataset"] == ds_name]
        for m in MODELS:
            m_df = ds_df[ds_df["model"] == m]
            if m_df.empty:
                continue
            best = m_df.loc[m_df["F1"].idxmax()]
            print(f"  {m:30s}: {best['strategy']} ({best['strategy_name']:15s}) "
                  f"F1={best['F1']:.4f}, ROUGE-L={best['ROUGE-L']:.4f}")

# Tính % cải thiện so với baseline
print(f"\n{'='*70}")
print("📈 CẢI THIỆN SO VỚI BASELINE (P0)")
print(f"{'='*70}")

for current_model in MODELS:
    model_df = df[df["model"] == current_model]
    if model_df.empty:
        continue

    print(f"\n🤖 {current_model}")
    for ds_name in datasets:
        ds_df = model_df[model_df["dataset"] == ds_name]
        p0_rows = ds_df[ds_df["strategy"] == "P0"]
        if p0_rows.empty:
            continue
        p0 = p0_rows.iloc[0]
        print(f"\n  --- {ds_name} (Baseline: F1={p0['F1']}, ROUGE-L={p0['ROUGE-L']}) ---")
        for _, row in ds_df[ds_df["strategy"] != "P0"].iterrows():
            f1_delta = row["F1"] - p0["F1"]
            rl_delta = row["ROUGE-L"] - p0["ROUGE-L"]
            f1_pct = (f1_delta / p0["F1"] * 100) if p0["F1"] > 0 else 0
            rl_pct = (rl_delta / p0["ROUGE-L"] * 100) if p0["ROUGE-L"] > 0 else 0
            arrow_f1 = "↑" if f1_delta > 0 else "↓"
            arrow_rl = "↑" if rl_delta > 0 else "↓"
            print(f"    {row['strategy']} ({row['strategy_name']:15s}): "
                  f"F1 {arrow_f1}{abs(f1_pct):+.1f}% ({row['F1']:.4f}), "
                  f"ROUGE-L {arrow_rl}{abs(rl_pct):+.1f}% ({row['ROUGE-L']:.4f})")

print(f"\n💾 Báo cáo: {report_path}")
print("✅ Hoàn tất!")
