import os
import random
import numpy as np
import torch
from peft import get_peft_model, LoraConfig, TaskType
from utils.config import load_config
from model_base.build_model import build_model, find_and_unfreeze_projector
from data_utils.build_dataloader import build_dataloader
from utils.evaluator import Evaluator
from utils.trainer import Trainer
from utils.saver import merge_and_save

# =========================================================
# Config
# =========================================================
cfg = load_config("configs/first_config.yaml")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 재현성
random.seed(cfg['base']['seed'])
np.random.seed(cfg['base']['seed'])
torch.manual_seed(cfg['base']['seed'])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(cfg['base']['seed'])


# =========================================================
# (A) Load FastVLM
# =========================================================
model, tokenizer, vision_processor = build_model(cfg)


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
train_loader, val_loader = build_dataloader(cfg, tokenizer, vision_processor)


# =========================================================
# (D) Initialize Evaluator and test "before FT"
# =========================================================
evaluator = Evaluator(model, tokenizer, vision_processor, cfg)
evaluator.test_sample("Before Fine-Tuning")


# =========================================================
# (E) Training
# =========================================================
trainer = Trainer(model, train_loader, cfg)

os.makedirs(cfg['base']['best_ckpt_dir'], exist_ok=True)
best_val_loss = float("inf")

for epoch in range(cfg['train']['epochs']):
    epoch_avg_loss = trainer.train_one_epoch(train_loader, epoch)
    val_avg_loss = evaluator.evaluate(val_loader)

    print(f"[epoch {epoch+1}/{cfg['train']['epochs']}] train_loss={epoch_avg_loss:.4f}  val_loss={val_avg_loss:.4f}")

    if val_avg_loss < best_val_loss:
        best_val_loss = val_avg_loss
        model.save_pretrained(cfg['base']['best_ckpt_dir'])
        tokenizer.save_pretrained(cfg['base']['best_ckpt_dir'])
        print(f"✅ Best model saved (val_loss={best_val_loss:.4f})")

print("✅ Training done!")


# =========================================================
# (F) Inference Test (After)
# =========================================================
evaluator.test_sample("After Fine-Tuning")


# =========================================================
# (G) Merge & Save
# =========================================================
merge_and_save(model, tokenizer, cfg)
