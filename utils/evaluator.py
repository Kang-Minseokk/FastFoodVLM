import torch
from utils.run_infer import run_inference


class Evaluator:
    def __init__(self, model, tokenizer, vision_processor, cfg):
        self.model = model
        self.tokenizer = tokenizer
        self.vision_processor = vision_processor
        self.cfg = cfg

    def test_sample(self, label=""):
        print(f"\n===== {label} =====")
        result = run_inference(self.model, self.tokenizer, self.vision_processor, self.cfg, self.cfg['base']['test_image_path'])
        if result is not None:
            print(result)

    def evaluate(self, val_loader):
        self.model.eval()
        val_loss_sum = 0.0
        val_steps = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.cfg['base']['device'], non_blocking=True) for k, v in batch.items()}
                out = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    images=batch["pixel_values"],
                    labels=batch["labels"],
                )
                val_loss_sum += out.loss.item()
                val_steps += 1
        return val_loss_sum / max(1, val_steps)
