"""Main Pipeline — Kết nối toàn bộ 5 giai đoạn hệ thống."""

import logging
import os
import re
import unicodedata

from src.config import ProjectConfig
from src.data_explorer import DataExplorer
from src.data_preprocessor import DataPreprocessor, QASample
from src.fpt_cloud_client import FPTCloudClient
from src.instruction_tuner import InstructionTuner, TrainingConfig
from src.model_evaluator import ModelEvaluator
from src.model_selector import ModelSelector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def sanitize_input(text: str) -> str:
    """Validate và sanitize input trước khi đưa vào template hoặc API.

    - Chuẩn hóa Unicode NFC
    - Loại bỏ ký tự điều khiển
    - Cảnh báo nếu phát hiện PII patterns
    """
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Cảnh báo PII patterns
    pii_patterns = [
        (r"\b\d{9,12}\b", "số điện thoại/CMND"),
        (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "email"),
    ]
    for pattern, pii_type in pii_patterns:
        if re.search(pattern, text):
            logger.warning(
                f"Phát hiện khả năng PII ({pii_type}) trong dữ liệu. "
                f"Cân nhắc loại bỏ trước khi gửi qua API."
            )

    return text.strip()


def run_pipeline(config: ProjectConfig) -> None:
    """Thực thi pipeline 5 giai đoạn tuần tự.

    Args:
        config: Cấu hình dự án đã validate.
    """
    # Validate config
    config.validate_config()
    logger.info("=== BẮT ĐẦU PIPELINE TINH CHỈNH LLM CHO QA TIẾNG VIỆT ===")

    # ========== GIAI ĐOẠN 1: Thu thập & Phân tích dữ liệu (EDA) ==========
    logger.info("--- Giai đoạn 1: Thu thập & Phân tích dữ liệu (EDA) ---")
    explorer = DataExplorer()

    datasets_dict = {}
    for i, source in enumerate(config.dataset_names):
        name = f"dataset_{i+1}"
        try:
            ds = explorer.load_dataset(source)
            datasets_dict[name] = ds
            logger.info(f"Đã tải {name}: {len(ds)} mẫu từ {source}")
        except Exception as e:
            logger.error(f"Không thể tải dataset {source}: {e}")
            # Thử tải từ đường dẫn cục bộ
            if config.dataset_paths and i < len(config.dataset_paths):
                local_path = config.dataset_paths[i]
                ds = explorer.load_dataset(local_path)
                datasets_dict[name] = ds
                logger.info(f"Đã tải {name} từ file cục bộ: {local_path}")

    if not datasets_dict:
        raise RuntimeError("Không thể tải bất kỳ dataset nào.")

    reports = explorer.generate_eda_report(datasets_dict, config.eda_report_dir)
    logger.info(f"Đã tạo báo cáo EDA cho {len(reports)} dataset")

    # ========== GIAI ĐOẠN 2: Tiền xử lý & Định dạng Instruction ==========
    logger.info("--- Giai đoạn 2: Tiền xử lý & Định dạng Instruction ---")
    preprocessor = DataPreprocessor()

    processed_datasets = {}
    for name, ds in datasets_dict.items():
        # Tiền xử lý: chuẩn hóa Unicode, loại bỏ nhiễu
        def preprocess_sample(sample):
            sample["context"] = sanitize_input(
                preprocessor.remove_noise(
                    preprocessor.normalize_vietnamese(sample.get("context", ""))
                )
            )
            sample["question"] = sanitize_input(
                preprocessor.remove_noise(
                    preprocessor.normalize_vietnamese(sample.get("question", ""))
                )
            )
            sample["answer"] = sanitize_input(
                preprocessor.remove_noise(
                    preprocessor.normalize_vietnamese(sample.get("answer", ""))
                )
            )
            return sample

        ds = ds.map(preprocess_sample)

        # Định dạng instruction
        ds = preprocessor.prepare_instruction_dataset(
            ds, template_format=config.instruction_format
        )

        # Chia splits
        train_ds, val_ds, test_ds = preprocessor.create_splits(
            ds,
            val_ratio=config.val_split_ratio,
            test_ratio=config.test_split_ratio,
        )
        processed_datasets[name] = {
            "train": train_ds,
            "val": val_ds,
            "test": test_ds,
        }
        logger.info(
            f"{name}: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}"
        )

    # Gộp tất cả train/val sets
    all_train = None
    all_val = None
    all_test = {}
    from datasets import concatenate_datasets

    train_list = [v["train"] for v in processed_datasets.values()]
    val_list = [v["val"] for v in processed_datasets.values()]
    all_train = concatenate_datasets(train_list)
    all_val = concatenate_datasets(val_list)
    for name, splits in processed_datasets.items():
        all_test[name] = splits["test"]

    logger.info(f"Tổng train: {len(all_train)}, val: {len(all_val)}")

    # ========== GIAI ĐOẠN 3: Lựa chọn mô hình & Cấu hình ==========
    logger.info("--- Giai đoạn 3: Lựa chọn mô hình & Cấu hình LoRA/QLoRA ---")
    selector = ModelSelector()

    model = selector.load_base_model(
        config.base_model_name, quantization=config.quantization
    )
    tokenizer = selector.load_tokenizer(
        config.tokenizer_name or config.base_model_name
    )

    lora_config = selector.create_lora_config(
        r=config.lora_r,
        alpha=config.lora_alpha,
        dropout=config.lora_dropout,
        target_modules=config.target_modules,
    )
    model = selector.apply_lora(model, lora_config)
    ratio = selector.get_trainable_params_ratio(model)
    logger.info(f"Tỷ lệ trainable params: {ratio:.4%}")

    # ========== GIAI ĐOẠN 4: Huấn luyện ==========
    logger.info("--- Giai đoạn 4: Huấn luyện (Instruction Tuning + Fine-tuning) ---")
    tuner = InstructionTuner()

    training_config = TrainingConfig(
        output_dir=config.output_dir,
        num_epochs=config.num_epochs,
        batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        max_seq_length=config.max_seq_length,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        fp16=config.fp16,
        bf16=config.bf16,
        weight_decay=config.weight_decay,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        instruction_format=config.instruction_format,
    )

    trainer = tuner.train(model, tokenizer, all_train, all_val, training_config)

    # Lưu mô hình
    final_model_dir = os.path.join(config.output_dir, "final_model")
    tuner.save_model(model, tokenizer, final_model_dir)

    # ========== GIAI ĐOẠN 5: Đánh giá & So sánh ==========
    logger.info("--- Giai đoạn 5: Đánh giá & So sánh kết quả ---")
    evaluator = ModelEvaluator()
    all_results = []

    # Đánh giá mô hình local trên từng test set
    for name, test_ds in all_test.items():
        result = evaluator.evaluate_on_dataset(
            model, tokenizer, test_ds, dataset_name=name
        )
        all_results.append(result)
        logger.info(
            f"[Local] {name}: EM={result.exact_match:.4f}, "
            f"F1={result.f1_score:.4f}, ROUGE-L={result.rouge_l:.4f}, "
            f"BERTScore={result.bert_score:.4f}"
        )

    # Đánh giá API baseline (nếu bật)
    if config.enable_api_baseline:
        api_key = os.environ.get("FPT_API_KEY")
        if api_key:
            for api_model_name in config.fpt_api_models:
                try:
                    api_client = FPTCloudClient(model_name=api_model_name)
                    for name, test_ds in all_test.items():
                        result = evaluator.evaluate_api_model(
                            api_client,
                            test_ds,
                            dataset_name=f"{name}_api_{api_model_name}",
                            max_tokens=config.fpt_api_max_tokens,
                        )
                        result.model_name = api_model_name
                        all_results.append(result)
                        logger.info(
                            f"[API:{api_model_name}] {name}: "
                            f"EM={result.exact_match:.4f}, "
                            f"F1={result.f1_score:.4f}, "
                            f"ROUGE-L={result.rouge_l:.4f}, "
                            f"BERTScore={result.bert_score:.4f}"
                        )
                except Exception as e:
                    logger.warning(
                        f"Không thể đánh giá API model {api_model_name}: {e}"
                    )
        else:
            logger.warning(
                "FPT_API_KEY chưa được thiết lập. Bỏ qua đánh giá API baseline."
            )

    # So sánh và xuất báo cáo
    comparison = evaluator.compare_results(all_results)
    report_path = os.path.join(config.eval_report_dir, "evaluation_report.json")
    evaluator.generate_report(comparison, report_path)

    logger.info("=== PIPELINE HOÀN TẤT ===")
    logger.info(f"Báo cáo đánh giá: {report_path}")
    logger.info(f"Mô hình đã lưu: {final_model_dir}")


if __name__ == "__main__":
    config = ProjectConfig()
    run_pipeline(config)
