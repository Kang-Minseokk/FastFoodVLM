import torch
import torch.nn.functional as F
from torch.optim import AdamW, SGD, Adam
from transformers import get_cosine_schedule_with_warmup

OPTIMIZER_MAP = {
    "adamw": AdamW,
    "adam": Adam,
    "sgd": SGD,
}


class ClassifierTrainer:
    def __init__(self, model, train_loader, cfg):
        self.model = model
        self.cfg = cfg
        self.device = cfg['base']['device']
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler(len(train_loader))

    def _build_optimizer(self):
        optimizer_cls = OPTIMIZER_MAP.get(self.cfg['train']['optimizer'].lower())
        if optimizer_cls is None:
            raise ValueError(f"Unsupported optimizer: {self.cfg['train']['optimizer']}. Choose from {list(OPTIMIZER_MAP)}")
        return optimizer_cls(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.cfg['train']['lr'],
            weight_decay=self.cfg['train']['weight_decay'],
        )

    def _build_scheduler(self, steps_per_epoch):
        total_steps = self.cfg['train']['epochs'] * steps_per_epoch
        return get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=steps_per_epoch,
            num_training_steps=total_steps,
        )

    def train_one_epoch(self, train_loader, epoch):
        self.model.train()
        loss_sum = 0.0
        correct = 0
        total = 0

        for step, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)

            logits = self.model(pixel_values)
            loss = F.cross_entropy(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg['train']['grad_clip'])
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            loss_sum += loss.item()
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += len(labels)

            if step % 10 == 0:
                acc = correct / total if total > 0 else 0.0
                print(f"[epoch {epoch+1}/{self.cfg['train']['epochs']}] step {step}  loss={loss.item():.4f}  acc={acc:.4f}  lr={self.scheduler.get_last_lr()[0]:.2e}")

        n_steps = max(1, step + 1)
        return loss_sum / n_steps, correct / max(1, total)
