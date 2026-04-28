import os
import random
import numpy as np
import torch
from transformers import AutoProcessor
from utils.config import load_config
from model_base.food_classifier_model import build_food_classifier, save_classifier
from data_utils.build_classifier_dataloader import build_classifier_dataloader
from utils.classifier_trainer import ClassifierTrainer
from utils.classifier_evaluator import ClassifierEvaluator

# =========================================================
# Config
# =========================================================
cfg = load_config("configs/food_classifier_config.yaml")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

random.seed(cfg['base']['seed'])
np.random.seed(cfg['base']['seed'])
torch.manual_seed(cfg['base']['seed'])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(cfg['base']['seed'])


# =========================================================
# (A) Vision Processor
# =========================================================
processor = AutoProcessor.from_pretrained(cfg['base']['model_name'])
vision_processor = processor.image_processor
# LlavaOnevision processor tiles images by default → produces 5D tensors.
# Disable splitting so the vision encoder receives standard (B, C, H, W) input.
if hasattr(vision_processor, 'do_image_splitting'):
    vision_processor.do_image_splitting = False


# =========================================================
# (B) Dataset
# =========================================================
train_loader, val_loader, class_to_idx = build_classifier_dataloader(cfg, vision_processor)
idx_to_class = {v: k for k, v in class_to_idx.items()}
num_classes = len(class_to_idx)


# =========================================================
# (C) Build Model
# =========================================================
model = build_food_classifier(cfg, num_classes)
model.count_trainable_params()


# =========================================================
# (D) Evaluator + baseline sample
# =========================================================
evaluator = ClassifierEvaluator(model, vision_processor, idx_to_class, cfg)
evaluator.test_sample(cfg['base']['test_image_path'], "Before Training")


# =========================================================
# (E) Training
# =========================================================
trainer = ClassifierTrainer(model, train_loader, cfg)

os.makedirs(cfg['base']['best_ckpt_dir'], exist_ok=True)
best_top1 = 0.0

for epoch in range(cfg['train']['epochs']):
    train_loss, train_acc = trainer.train_one_epoch(train_loader, epoch)
    metrics = evaluator.evaluate(val_loader)

    print(
        f"[epoch {epoch+1}/{cfg['train']['epochs']}] "
        f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
        f"val_loss={metrics['val_loss']:.4f}  "
        f"top1={metrics['top1_acc']:.4f}  top5={metrics['top5_acc']:.4f}"
    )

    if metrics['top1_acc'] > best_top1:
        best_top1 = metrics['top1_acc']
        save_classifier(model, cfg, class_to_idx, cfg['base']['best_ckpt_dir'])
        print(f"✅ Best model saved (top1={best_top1:.4f})")

print("✅ Training done!")


# =========================================================
# (F) Final sample test
# =========================================================
evaluator.test_sample(cfg['base']['test_image_path'], "After Training")
