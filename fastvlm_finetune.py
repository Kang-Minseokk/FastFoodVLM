import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

# =========================================================
# Constants
# =========================================================
IMAGE_TOKEN_INDEX = -200
DEVICE = "cuda"


# =========================================================
# (A) Load FastVLM
# =========================================================
model_name = "apple/FastVLM-0.5B"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    trust_remote_code=True
).to(DEVICE)

# vision tower freeze
vision_processor = model.get_vision_tower().image_processor
vision_tower = model.get_vision_tower()
for p in vision_tower.parameters():
    p.requires_grad = False
print("✅ Model loaded + Vision Tower frozen")


# =========================================================
# (B) Apply LoRA
# =========================================================
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
print("✅ LoRA applied")


# =========================================================
# (C) Dataset
# =========================================================
class FoodImageDataset(Dataset):
    def __init__(self, root_dir, tokenizer, vision_processor):
        self.samples = []  # (img_path, food_name)

        for food_name in os.listdir(root_dir):
            food_dir = os.path.join(root_dir, food_name, "images")
            if not os.path.isdir(food_dir):
                continue

            for file in os.listdir(food_dir):
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(food_dir, file)
                    self.samples.append((img_path, food_name))

        self.tokenizer = tokenizer
        self.vision_processor = vision_processor

    def __len__(self):
        return len(self.samples)

    def inject_image_token(self, text, image_token_id=IMAGE_TOKEN_INDEX):
        """ "<image>" placeholder → token id 삽입 """
        if "<image>" not in text:
            raise ValueError("Prompt must contain <image> placeholder.")

        pre, post = text.split("<image>", 1)

        pre_ids = self.tokenizer(pre, add_special_tokens=False).input_ids
        if isinstance(pre_ids[0], list):
            pre_ids = pre_ids[0]

        post_ids = self.tokenizer(post, add_special_tokens=False).input_ids
        if isinstance(post_ids[0], list):
            post_ids = post_ids[0]

        merged = pre_ids + [image_token_id] + post_ids
        return merged

    def __getitem__(self, idx):
        img_path, food_name = self.samples[idx]

        # ===== (1) image load
        pil_img = Image.open(img_path).convert("RGB")
        px = self.vision_processor(images=pil_img, return_tensors="pt")["pixel_values"][0]
        px = px.to(torch.float32)

        # ===== (2) messages
        messages = [
            {
                "role": "user",
                "content": "<image>\nAnswer ONLY with the food name in one or two English words. No extra text."
            },
            {
                "role": "assistant",
                "content": food_name
            }
        ]

        # text template 구성
        chat_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        # input_ids 생성 (<image> → IMAGE_TOKEN_INDEX)
        input_ids = self.inject_image_token(chat_text)
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        # ===== (3) 정답 mask
        answer = food_name
        answer_char_idx = chat_text.rindex(answer)
        pre_answer_text = chat_text[:answer_char_idx]

        ans_start = len(self.inject_image_token(pre_answer_text))
        labels = input_ids.clone()
        labels[:ans_start] = -100
        labels[labels == IMAGE_TOKEN_INDEX] = -100

        # attention mask
        attention_mask = torch.ones_like(input_ids)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "pixel_values": px,
        }


# =========================================================
# (D) collate_fn
# =========================================================
def collate_fn(batch):
    input_ids = [b["input_ids"] for b in batch]
    labels = [b["labels"] for b in batch]
    masks = [b["attention_mask"] for b in batch]
    px = torch.stack([b["pixel_values"] for b in batch], dim=0)

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    masks = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=0)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": masks,
        "pixel_values": px
    }


# =========================================================
# (E) DataLoader
# =========================================================
dataset = FoodImageDataset(
    root_dir="dataset",
    tokenizer=tokenizer,
    vision_processor=vision_processor
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn
)

print(f"✅ Dataset size = {len(dataset)}")


# =========================================================
# (F) Training
# =========================================================
optimizer = AdamW(model.parameters(), lr=1e-4)

model.train()
for epoch in range(3):
    for step, batch in enumerate(loader):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            images=batch["pixel_values"],
            labels=batch["labels"],
        )

        loss = out.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 10 == 0:
            print(f"[epoch {epoch}] step {step} loss = {loss.item():.4f}")

print("✅ Training done!")


# =========================================================
# (G) Inference example
# =========================================================
def inject_image_token_simple(text):
    pre, post = text.split("<image>", 1)
    pre_ids = tokenizer(pre, add_special_tokens=False).input_ids
    post_ids = tokenizer(post, add_special_tokens=False).input_ids
    merged = pre_ids + [IMAGE_TOKEN_INDEX] + post_ids
    return torch.tensor([merged])


test_img = Image.open("dataset/apple/apple1.jpg").convert("RGB")
px = vision_processor(images=test_img, return_tensors="pt")["pixel_values"].to(DEVICE)
px = px.to(torch.float32)

infer_msg = [
    {"role": "user",
     "content": "<image>\nAnswer ONLY with the food name in one or two English words. No extra text."}
]

infer_text = tokenizer.apply_chat_template(
    infer_msg,
    tokenize=False,
    add_generation_prompt=True,
)

test_ids = inject_image_token_simple(infer_text).to(DEVICE)
mask = torch.ones_like(test_ids).to(DEVICE)

model.eval()
with torch.no_grad():
    gen = model.generate(
        inputs=test_ids,
        attention_mask=mask,
        images=px,
        max_new_tokens=8,
        do_sample=False
    )

print("\n===== MODEL OUTPUT =====")
print(tokenizer.decode(gen[0], skip_special_tokens=True).strip())
