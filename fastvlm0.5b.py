# fastvlm_lora_tuning.py
import torch
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForVision2Seq, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

# ---------------------------
# 1. 모델 및 프로세서 로드
# ---------------------------
model_id = "apple/FastVLM-0.5B"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

# ---------------------------
# 2. LoRA 설정 (LLM 부분만)
# ---------------------------
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM",
)
model = get_peft_model(model, config)

# ---------------------------
# 3. 데이터셋 로드 (이미지 포함)
# ---------------------------
dataset = load_dataset("json", data_files="dataset/food_instruction.jsonl")

# ---------------------------
# 4. 전처리 함수
# ---------------------------
def preprocess(examples):
    images = [x for x in examples["image"]]
    texts = [f"{inst}\n답변:" for inst in examples["instruction"]]
    inputs = processor(
        images=images,
        text=texts,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    labels = processor.tokenizer(
        examples["response"],
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).input_ids

    inputs["labels"] = labels
    return inputs

tokenized = dataset["train"].map(preprocess, batched=True, remove_columns=dataset["train"].column_names)

# ---------------------------
# 5. 학습 설정
# ---------------------------
training_args = TrainingArguments(
    output_dir="./fastvlm_food_lora",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    save_total_limit=2,
    logging_steps=10,
    save_steps=100,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized
)

# ---------------------------
# 6. 학습 수행
# ---------------------------
trainer.train()

# ---------------------------
# 7. 모델 저장
# ---------------------------
model.save_pretrained("./fastvlm_food_lora")
processor.save_pretrained("./fastvlm_food_lora")
