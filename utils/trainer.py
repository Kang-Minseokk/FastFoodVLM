import torch
from torch.optim import AdamW, SGD, Adam
from transformers import get_cosine_schedule_with_warmup

OPTIMIZER_MAP = {
    "adamw": AdamW,
    "adam": Adam,
    "sgd": SGD,
}


class Trainer:
    def __init__(self, model, train_loader, cfg):
        self.model = model
        self.cfg = cfg
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler(len(train_loader))

    def build_optimizer(self):
        optimizer_cls = OPTIMIZER_MAP.get(self.cfg['train']['optimizer'].lower())
        if optimizer_cls is None:
            raise ValueError(f"Unsupported optimizer: {self.cfg['train']['optimizer']}. Choose from {list(OPTIMIZER_MAP.keys())}")
        return optimizer_cls(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.cfg['train']['lr'],
            weight_decay=self.cfg['train']['weight_decay'],
        )

    def build_scheduler(self, steps_per_epoch):
        total_steps = self.cfg['train']['epochs'] * steps_per_epoch
        warmup_steps = steps_per_epoch
        return get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    def train_one_epoch(self, train_loader, epoch):
        self.model.train()
        epoch_loss_sum = 0.0
        epoch_steps = 0

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(self.cfg['base']['device'], non_blocking=True) for k, v in batch.items()}

            out = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                images=batch["pixel_values"],
                labels=batch["labels"],
            )
            loss = out.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg['train']['grad_clip'])
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            epoch_loss_sum += loss.item()
            epoch_steps += 1

            if step % 10 == 0:
                print(f"[epoch {epoch+1}/{self.cfg['train']['epochs']}] step {step}  loss={loss.item():.4f}  lr={self.scheduler.get_last_lr()[0]:.2e}")

        return epoch_loss_sum / max(1, epoch_steps)
