import os
import random
import numpy as np
from collections import defaultdict
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torchvision import transforms
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

# =========================================================
# Config
# =========================================================
IMAGE_TOKEN_INDEX = -200
DEVICE = "cuda"
SEED = 42
DATASET_ROOT = "dataset"
TEST_IMAGE_PATH = "debug_cheesecake.png"

EPOCHS = 20
BATCH_SIZE = 4
LR = 1e-4
GRAD_CLIP = 1.0
VAL_SPLIT = 0.1
LORA_RANK = 32
LORA_ALPHA = 64
SAVE_DIR = "FastFoodVLM-0.5B"
BEST_CKPT_DIR = "checkpoints/best"

# Vision Tower 마지막 N개 블록 unfreeze. 메모리 부족 시 0으로 설정
UNFREEZE_VISION_BLOCKS = 0

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")

# 재현성
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# =========================================================
# (A) Load FastVLM
# =========================================================
model_name = "apple/FastVLM-0.5B"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).to(DEVICE)

vision_processor = model.get_vision_tower().image_processor
vision_tower = model.get_vision_tower()

# Vision Tower 전체 freeze
for p in vision_tower.parameters():
    p.requires_grad = False

print("✅ Model loaded + Vision Tower frozen")

# =========================================================
# (B) Apply LoRA
#     Attention + MLP 모두 포함, rank 32로 확장
# =========================================================
lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # attention
        "gate_proj", "up_proj", "down_proj",       # MLP (기존에 누락됨)
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)

# =========================================================
# (B-1) mm_projector unfreeze (PEFT 적용 후에 해야 함)
#       PEFT가 get_peft_model 시 모든 base model 파라미터를 freeze하므로
#       반드시 이후에 unfreeze 해야 효과가 있음
# =========================================================
projector = None
for attr in ["multi_modal_projector", "model.multi_modal_projector"]:
    try:
        obj = model
        for part in attr.split("."):
            obj = getattr(obj, part)
        projector = obj
        break
    except AttributeError:
        continue

if projector is not None:
    for p in projector.parameters():
        p.requires_grad = True
    print("✅ mm_projector unfrozen")
else:
    print("⚠️  mm_projector를 찾지 못했습니다. 구조를 확인하세요.")

# =========================================================
# (B-2) Vision Tower 부분 unfreeze (선택, PEFT 적용 후)
#       UNFREEZE_VISION_BLOCKS > 0 일 때만 적용
# =========================================================
if UNFREEZE_VISION_BLOCKS > 0:
    try:
        vit_layers = list(vision_tower.vision_model.encoder.layers)
        for layer in vit_layers[-UNFREEZE_VISION_BLOCKS:]:
            for p in layer.parameters():
                p.requires_grad = True
        print(f"✅ Vision Tower 마지막 {UNFREEZE_VISION_BLOCKS}개 블록 unfrozen")
    except Exception as e:
        print(f"⚠️  Vision Tower unfreeze 실패: {e}")

model.print_trainable_parameters()
print("✅ LoRA applied")


# =========================================================
# (C) Dataset
# =========================================================
train_augment = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
])


class FoodImageDataset(Dataset):
    def __init__(self, tokenizer, vision_processor, samples, augment=False):
        self.tokenizer = tokenizer
        self.vision_processor = vision_processor
        self.samples = samples  # list of (img_path, food_name)
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def inject_image_token(self, text, image_token_id=IMAGE_TOKEN_INDEX):
        if "<image>" not in text:
            raise ValueError("Prompt must contain <image> placeholder.")
        pre, post = text.split("<image>", 1)
        pre_ids = self.tokenizer(pre, add_special_tokens=False).input_ids
        post_ids = self.tokenizer(post, add_special_tokens=False).input_ids
        return pre_ids + [image_token_id] + post_ids

    def __getitem__(self, idx):
        img_path, food_name = self.samples[idx]

        # (1) Image + optional augmentation
        pil_img = Image.open(img_path).convert("RGB")
        if self.augment:
            pil_img = train_augment(pil_img)
        px = self.vision_processor(images=pil_img, return_tensors="pt")["pixel_values"][0]
        px = px.to(torch.bfloat16)

        # food_name 정규화: 폴더명 언더스코어/하이픈 → 공백
        food_label = food_name.replace("_", " ").replace("-", " ")

        # (2) Chat template
        messages = [
            {
                "role": "user",
                "content": "<image>\nAnswer ONLY with the food name in one or two English words. No extra text."
            },
            {
                "role": "assistant",
                "content": food_label
            }
        ]
        chat_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # (3) input_ids
        input_ids = self.inject_image_token(chat_text)
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        # (4) Label masking: food_label 토큰만 학습
        answer_char_idx = chat_text.rindex(food_label)
        pre_answer_text = chat_text[:answer_char_idx]
        ans_start = len(self.inject_image_token(pre_answer_text))

        labels = input_ids.clone()
        labels[:ans_start] = -100
        labels[labels == IMAGE_TOKEN_INDEX] = -100

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.ones_like(input_ids),
            "pixel_values": px,
        }


# =========================================================
# (D) collate_fn
# =========================================================
def collate_fn(batch):
    pad_id = tokenizer.eos_token_id
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [b["input_ids"] for b in batch], batch_first=True, padding_value=pad_id)
    labels = torch.nn.utils.rnn.pad_sequence(
        [b["labels"] for b in batch], batch_first=True, padding_value=-100)
    masks = torch.nn.utils.rnn.pad_sequence(
        [b["attention_mask"] for b in batch], batch_first=True, padding_value=0)
    px = torch.stack([b["pixel_values"] for b in batch], dim=0)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": masks, "pixel_values": px}


# =========================================================
# (E) 샘플 수집 + Stratified train/val split
# =========================================================
all_samples = []
for food_name in sorted(os.listdir(DATASET_ROOT)):
    food_dir = os.path.join(DATASET_ROOT, food_name)
    if not os.path.isdir(food_dir):
        continue
    for file in os.listdir(food_dir):
        if file.lower().endswith(IMAGE_EXTENSIONS):
            all_samples.append((os.path.join(food_dir, file), food_name))

# 클래스별로 분리 후 split
class_to_samples = defaultdict(list)
for s in all_samples:
    class_to_samples[s[1]].append(s)

train_samples, val_samples = [], []
for cls, samps in class_to_samples.items():
    random.shuffle(samps)
    n_val = max(1, int(len(samps) * VAL_SPLIT))
    val_samples.extend(samps[:n_val])
    train_samples.extend(samps[n_val:])

# WeightedRandomSampler: 클래스 불균형 완화
class_counts = defaultdict(int)
for _, fname in train_samples:
    class_counts[fname] += 1
sample_weights = [1.0 / class_counts[fname] for _, fname in train_samples]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_samples), replacement=True)

train_dataset = FoodImageDataset(tokenizer, vision_processor, train_samples, augment=True)
val_dataset = FoodImageDataset(tokenizer, vision_processor, val_samples, augment=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

print(f"✅ Dataset: train={len(train_dataset)}, val={len(val_dataset)}, classes={len(class_to_samples)}")


# =========================================================
# (F) Inference 함수 (before/after 공용)
# =========================================================
def run_inference(model, image_path):
    if not os.path.exists(image_path):
        print(f"⚠️  테스트 이미지 없음: {image_path}")
        return None

    test_img = Image.open(image_path).convert("RGB")
    px = vision_processor(images=test_img, return_tensors="pt")["pixel_values"].to(DEVICE, dtype=torch.bfloat16)

    infer_msg = [{"role": "user", "content": "<image>\nAnswer ONLY with the food name in one or two English words. No extra text."}]
    infer_text = tokenizer.apply_chat_template(infer_msg, tokenize=False, add_generation_prompt=True)

    pre, post = infer_text.split("<image>", 1)
    pre_ids = tokenizer(pre, add_special_tokens=False).input_ids
    post_ids = tokenizer(post, add_special_tokens=False).input_ids
    test_ids = torch.tensor([pre_ids + [IMAGE_TOKEN_INDEX] + post_ids]).to(DEVICE)
    mask = torch.ones_like(test_ids)

    model.eval()
    with torch.no_grad():
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        gen = model.generate(
            inputs=test_ids,
            attention_mask=mask,
            images=px,
            max_new_tokens=8,
            do_sample=False,
            eos_token_id=im_end_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    # 새로 생성된 토큰만 디코딩 (입력 제외)
    new_tokens = gen[0, test_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


print("\n===== Before Fine-Tuning =====")
result_before = run_inference(model, TEST_IMAGE_PATH)
if result_before is not None:
    print(result_before)


# =========================================================
# (G) Training
# =========================================================
total_steps = EPOCHS * len(train_loader)
warmup_steps = len(train_loader)  # 1 epoch warmup

optimizer = AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LR,
    weight_decay=0.01,
)
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)

os.makedirs(BEST_CKPT_DIR, exist_ok=True)
best_val_loss = float("inf")

for epoch in range(EPOCHS):
    model.train()
    epoch_loss_sum = 0.0
    epoch_steps = 0

    for step, batch in enumerate(train_loader):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            images=batch["pixel_values"],
            labels=batch["labels"],
        )
        loss = out.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        epoch_loss_sum += loss.item()
        epoch_steps += 1

        if step % 10 == 0:
            print(f"[epoch {epoch+1}/{EPOCHS}] step {step}  loss={loss.item():.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

    epoch_avg_loss = epoch_loss_sum / max(1, epoch_steps)

    # Validation
    model.eval()
    val_loss_sum = 0.0
    val_steps = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                images=batch["pixel_values"],
                labels=batch["labels"],
            )
            val_loss_sum += out.loss.item()
            val_steps += 1
    val_avg_loss = val_loss_sum / max(1, val_steps)

    print(f"[epoch {epoch+1}/{EPOCHS}] train_loss={epoch_avg_loss:.4f}  val_loss={val_avg_loss:.4f}")

    # Best checkpoint 저장 (val_loss 기준)
    if val_avg_loss < best_val_loss:
        best_val_loss = val_avg_loss
        model.save_pretrained(BEST_CKPT_DIR)
        tokenizer.save_pretrained(BEST_CKPT_DIR)
        print(f"  ✅ Best model saved (val_loss={best_val_loss:.4f})")

print("✅ Training done!")


# =========================================================
# (H) Inference after fine-tuning
# =========================================================
print("\n===== After Fine-Tuning =====")
result_after = run_inference(model, TEST_IMAGE_PATH)
if result_after is not None:
    print(result_after)


# =========================================================
# (I) Merge & Save (최종 모델)
#     Best checkpoint는 checkpoints/best/ 에 별도 저장되어 있음
# =========================================================
merged = model.merge_and_unload()
merged.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"✅ Merged model saved to ./{SAVE_DIR}/")
print(f"   Best checkpoint (lowest val_loss): ./{BEST_CKPT_DIR}/")
