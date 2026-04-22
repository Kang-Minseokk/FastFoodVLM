import os
import random
import numpy as np
import torch
from torch.optim import AdamW
from peft import get_peft_model, LoraConfig, TaskType
from transformers import get_cosine_schedule_with_warmup
from config.config import load_config
from model_base.build_model import build_model, find_and_unfreeze_projector
from data_utils.build_dataloader import build_dataloader
from evaluate.evaluator import Evaluator

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
model, tokenizer, vision_processor, vision_tower = build_model(cfg)


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
# (B-sub) mm_projector unfreeze (PEFT 적용 후에 해야 함)
# =========================================================
find_and_unfreeze_projector(model)
model.print_trainable_parameters()
print("✅ LoRA applied")


# =========================================================
# (C) Dataset
# =========================================================
train_loader, val_loader, class_to_samples = build_dataloader(cfg, tokenizer, vision_processor)


# =========================================================
# (D) Initialize Evaluator and test "before FT"
# =========================================================
evaluator = Evaluator(model, tokenizer, vision_processor, cfg)
evaluator.test_sample("Before Fine-Tuning")


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

    val_avg_loss = evaluator.evaluate(val_loader)

    print(f"[epoch {epoch+1}/{cfg['train']['epochs']}] train_loss={epoch_avg_loss:.4f}  val_loss={val_avg_loss:.4f}")

    if val_avg_loss < best_val_loss:
        best_val_loss = val_avg_loss
        model.save_pretrained(cfg['base']['best_ckpt_dir'])
        tokenizer.save_pretrained(cfg['base']['best_ckpt_dir'])
        print(f"  ✅ Best model saved (val_loss={best_val_loss:.4f})")

print("✅ Training done!")


# =========================================================
# (H) Inference Test (After)
# =========================================================
evaluator.test_sample("After Fine-Tuning")


# =========================================================
# (I) Merge & Save (최종 모델)
# =========================================================
merged = model.merge_and_unload()
merged.save_pretrained(cfg['base']['save_dir'])
tokenizer.save_pretrained(cfg['base']['save_dir'])
print(f"✅ Merged model saved to ./{cfg['base']['save_dir']}/")
print(f"   Best checkpoint (lowest val_loss): ./{cfg['base']['best_ckpt_dir']}/")
