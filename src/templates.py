# ===== ĐỊNH DẠNG ALPACA =====
# Template Alpaca cho QA tiếng Việt (dùng khi huấn luyện)
ALPACA_TRAIN_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Dựa vào ngữ cảnh được cung cấp, hãy trả lời câu hỏi một cách chính xác và ngắn gọn bằng tiếng Việt.

### Input:
Ngữ cảnh: {context}

Câu hỏi: {question}

### Response:
{answer}"""

# Template Alpaca cho inference (không có answer)
ALPACA_INFERENCE_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Dựa vào ngữ cảnh được cung cấp, hãy trả lời câu hỏi một cách chính xác và ngắn gọn bằng tiếng Việt.

### Input:
Ngữ cảnh: {context}

Câu hỏi: {question}

### Response:
"""

# ===== ĐỊNH DẠNG CHATML =====
# Template ChatML cho QA tiếng Việt (dùng khi huấn luyện)
CHATML_TRAIN_TEMPLATE = """<|im_start|>system
Bạn là trợ lý AI chuyên trả lời câu hỏi bằng tiếng Việt. Hãy trả lời dựa trên ngữ cảnh được cung cấp một cách chính xác và ngắn gọn.<|im_end|>
<|im_start|>user
Ngữ cảnh: {context}

Câu hỏi: {question}<|im_end|>
<|im_start|>assistant
{answer}<|im_end|>"""

# Template ChatML cho inference
CHATML_INFERENCE_TEMPLATE = """<|im_start|>system
Bạn là trợ lý AI chuyên trả lời câu hỏi bằng tiếng Việt. Hãy trả lời dựa trên ngữ cảnh được cung cấp một cách chính xác và ngắn gọn.<|im_end|>
<|im_start|>user
Ngữ cảnh: {context}

Câu hỏi: {question}<|im_end|>
<|im_start|>assistant
"""

# ===== CHAT-BASED MODELS (dùng với tokenizer.apply_chat_template) =====
# Template cơ bản cho QA
CHAT_QA_MESSAGES = [
    {
        "role": "system",
        "content": "Bạn là trợ lý AI chuyên trả lời câu hỏi bằng tiếng Việt. "
                   "Hãy trả lời dựa trên ngữ cảnh được cung cấp một cách chính xác và ngắn gọn."
    },
    {
        "role": "user",
        "content": "Ngữ cảnh: {context}\n\nCâu hỏi: {question}"
    },
    {
        "role": "assistant",
        "content": "{answer}"
    }
]

# Template nâng cao: Adaptive QA (tự điều chỉnh theo loại câu hỏi)
CHAT_ADAPTIVE_QA_MESSAGES = [
    {
        "role": "system",
        "content": (
            "Vai trò: Bạn là trợ lý AI thông minh chuyên trả lời câu hỏi tiếng Việt.\n\n"
            "Chiến lược thích ứng — điều chỉnh cách trả lời theo loại câu hỏi:\n"
            "- Câu hỏi THỰC THỂ (ai, ở đâu, năm nào): → Trích xuất cụm từ ngắn.\n"
            "- Câu hỏi GIẢI THÍCH (tại sao, như thế nào): → Trả lời 1-2 câu.\n"
            "- Câu hỏi LIỆT KÊ (những gì, các loại nào): → Liệt kê ngắn gọn.\n\n"
            "Quy tắc: Trả lời dựa trên ngữ cảnh, chính xác, không thêm thông tin ngoài."
        )
    },
    {
        "role": "user",
        "content": "Ngữ cảnh:\n\"\"\"\n{context}\n\"\"\"\n\nCâu hỏi: {question}"
    },
    {
        "role": "assistant",
        "content": "{answer}"
    }
]
