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
from config.config import load_config

# =========================================================
# Config
# =========================================================
cfg = load_config("configs/first_config.yaml")

# 재현성
random.seed(cfg['base']['seed'])
np.random.seed(cfg['base']['seed'])
torch.manual_seed(cfg['base']['seed'])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(cfg['base']['seed'])

# =========================================================
# (A) Load FastVLM
# =========================================================
tokenizer = AutoTokenizer.from_pretrained(cfg['base']['model_name'], trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    cfg['base']['model_name'],
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).to(cfg['base']['device'])

vision_processor = model.get_vision_tower().image_processor
vision_tower = model.get_vision_tower()

for p in vision_tower.parameters():
    p.requires_grad = False

print("✅ Model loaded + Vision Tower frozen")

# =========================================================
# (B) Apply LoRA
# =========================================================
lora_config = LoraConfig(
    r=cfg['train']['lora_rank'],
    lora_alpha=cfg['train']['lora_alpha'],
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=cfg['train']['lora_dropout'],
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)

# =========================================================
# (B-1) mm_projector unfreeze (PEFT 적용 후에 해야 함)
# =========================================================
PROJECTOR_KEYWORDS = ["mm_projector", "multi_modal_projector", "projector", "connector", "mm_proj"]

def find_and_unfreeze_projector(model):
    unfrozen = []
    seen = set()

    for name, module in model.named_modules():
        name_lower = name.lower()
        if any(kw in name_lower for kw in PROJECTOR_KEYWORDS) and len(name.split(".")) <= 5:
            if any(name.startswith(seen_name + ".") for seen_name in seen):
                continue
            for p in module.parameters():
                p.requires_grad = True
            seen.add(name)
            unfrozen.append(name)

    if unfrozen:
        for n in unfrozen:
            print(f"✅ mm_projector unfrozen: {n}")
    else:
        print("⚠️  projector를 찾지 못했습니다. 아래 모듈 목록을 확인하세요:")
        for name, _ in model.named_modules():
            if len(name.split(".")) <= 3:
                print(f"   {name}")

    return len(unfrozen) > 0

find_and_unfreeze_projector(model)

# =========================================================
# (B-2) Vision Tower 부분 unfreeze (선택, PEFT 적용 후)
# =========================================================
if cfg['train']['unfreeze_vision_blocks'] > 0:
    try:
        vit_layers = list(vision_tower.vision_model.encoder.layers)
        for layer in vit_layers[-cfg['train']['unfreeze_vision_blocks']:]:
            for p in layer.parameters():
                p.requires_grad = True
        print(f"✅ Vision Tower 마지막 {cfg['train']['unfreeze_vision_blocks']}개 블록 unfrozen")
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
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def inject_image_token(self, text, image_token_id=None):
        if image_token_id is None:
            image_token_id = cfg['base']['image_token_index']
        if "<image>" not in text:
            raise ValueError("Prompt must contain <image> placeholder.")
        pre, post = text.split("<image>", 1)
        pre_ids = self.tokenizer(pre, add_special_tokens=False).input_ids
        post_ids = self.tokenizer(post, add_special_tokens=False).input_ids
        return pre_ids + [image_token_id] + post_ids

    def __getitem__(self, idx):
        img_path, food_name = self.samples[idx]

        pil_img = Image.open(img_path).convert("RGB")
        if self.augment:
            pil_img = train_augment(pil_img)
        px = self.vision_processor(images=pil_img, return_tensors="pt")["pixel_values"][0]
        px = px.to(torch.bfloat16)

        food_label = food_name.replace("_", " ").replace("-", " ")

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

        input_ids = self.inject_image_token(chat_text)
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        answer_char_idx = chat_text.rindex(food_label)
        pre_answer_text = chat_text[:answer_char_idx]
        ans_start = len(self.inject_image_token(pre_answer_text))

        labels = input_ids.clone()
        labels[:ans_start] = -100
        labels[labels == cfg['base']['image_token_index']] = -100

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
for food_name in sorted(os.listdir(cfg['base']['dataset_root'])):
    food_dir = os.path.join(cfg['base']['dataset_root'], food_name)
    if not os.path.isdir(food_dir):
        continue
    for file in os.listdir(food_dir):
        if file.lower().endswith(tuple(cfg['base']['image_extensions'])):
            all_samples.append((os.path.join(food_dir, file), food_name))

class_to_samples = defaultdict(list)
for s in all_samples:
    class_to_samples[s[1]].append(s)

train_samples, val_samples = [], []
for cls, samps in class_to_samples.items():
    random.shuffle(samps)
    n_val = max(1, int(len(samps) * cfg['train']['val_split']))
    val_samples.extend(samps[:n_val])
    train_samples.extend(samps[n_val:])

class_counts = defaultdict(int)
for _, fname in train_samples:
    class_counts[fname] += 1
sample_weights = [1.0 / class_counts[fname] for _, fname in train_samples]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_samples), replacement=True)

train_dataset = FoodImageDataset(tokenizer, vision_processor, train_samples, augment=True)
val_dataset = FoodImageDataset(tokenizer, vision_processor, val_samples, augment=False)

train_loader = DataLoader(train_dataset,
                          batch_size=cfg['train']['batch_size'],
                          sampler=sampler,
                          collate_fn=collate_fn,
                          num_workers=8, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=cfg['train']['batch_size'], shuffle=False, collate_fn=collate_fn)

print(f"✅ Dataset: train={len(train_dataset)}, val={len(val_dataset)}, classes={len(class_to_samples)}")


# =========================================================
# (F) Inference 함수 (before/after 공용)
# =========================================================
def run_inference(model, image_path):
    if not os.path.exists(image_path):
        print(f"⚠️  테스트 이미지 없음: {image_path}")
        return None

    test_img = Image.open(image_path).convert("RGB")
    px = vision_processor(images=test_img, return_tensors="pt")["pixel_values"].to(cfg['base']['device'], dtype=torch.bfloat16)

    infer_msg = [{"role": "user", "content": "<image>\nAnswer ONLY with the food name in one or two English words. No extra text."}]
    infer_text = tokenizer.apply_chat_template(infer_msg, tokenize=False, add_generation_prompt=True)

    pre, post = infer_text.split("<image>", 1)
    pre_ids = tokenizer(pre, add_special_tokens=False).input_ids
    post_ids = tokenizer(post, add_special_tokens=False).input_ids
    test_ids = torch.tensor([pre_ids + [cfg['base']['image_token_index']] + post_ids]).to(cfg['base']['device'])
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
    new_tokens = gen[0, test_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


print("\n===== Before Fine-Tuning =====")
result_before = run_inference(model, cfg['base']['test_image_path'])
if result_before is not None:
    print(result_before)


# =========================================================
# (G) Training
# =========================================================
total_steps = cfg['train']['epochs'] * len(train_loader)
warmup_steps = len(train_loader)

optimizer = AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=cfg['train']['lr'],
    weight_decay=0.01,
)
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)

os.makedirs(cfg['base']['best_ckpt_dir'], exist_ok=True)
best_val_loss = float("inf")

for epoch in range(cfg['train']['epochs']):
    model.train()
    epoch_loss_sum = 0.0
    epoch_steps = 0

    for step, batch in enumerate(train_loader):
        batch = {k: v.to(cfg['base']['device'], non_blocking=True) for k, v in batch.items()}

        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            images=batch["pixel_values"],
            labels=batch["labels"],
        )
        loss = out.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['train']['grad_clip'])
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        epoch_loss_sum += loss.item()
        epoch_steps += 1

        if step % 10 == 0:
            print(f"[epoch {epoch+1}/{cfg['train']['epochs']}] step {step}  loss={loss.item():.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

    epoch_avg_loss = epoch_loss_sum / max(1, epoch_steps)

    # Validation
    model.eval()
    val_loss_sum = 0.0
    val_steps = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(cfg['base']['device'], non_blocking=True) for k, v in batch.items()}
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                images=batch["pixel_values"],
                labels=batch["labels"],
            )
            val_loss_sum += out.loss.item()
            val_steps += 1
    val_avg_loss = val_loss_sum / max(1, val_steps)

    print(f"[epoch {epoch+1}/{cfg['train']['epochs']}] train_loss={epoch_avg_loss:.4f}  val_loss={val_avg_loss:.4f}")

    if val_avg_loss < best_val_loss:
        best_val_loss = val_avg_loss
        model.save_pretrained(cfg['base']['best_ckpt_dir'])
        tokenizer.save_pretrained(cfg['base']['best_ckpt_dir'])
        print(f"  ✅ Best model saved (val_loss={best_val_loss:.4f})")

print("✅ Training done!")


# =========================================================
# (H) Inference after fine-tuning
# =========================================================
print("\n===== After Fine-Tuning =====")
result_after = run_inference(model, cfg['base']['test_image_path'])
if result_after is not None:
    print(result_after)


# =========================================================
# (I) Merge & Save (최종 모델)
# =========================================================
merged = model.merge_and_unload()
merged.save_pretrained(cfg['base']['save_dir'])
tokenizer.save_pretrained(cfg['base']['save_dir'])
print(f"✅ Merged model saved to ./{cfg['base']['save_dir']}/")
print(f"   Best checkpoint (lowest val_loss): ./{cfg['base']['best_ckpt_dir']}/")
