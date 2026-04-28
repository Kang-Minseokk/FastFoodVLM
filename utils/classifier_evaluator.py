import torch
import torch.nn.functional as F
from PIL import Image


class ClassifierEvaluator:
    def __init__(self, model, vision_processor, idx_to_class, cfg):
        self.model = model
        self.vision_processor = vision_processor
        self.idx_to_class = idx_to_class
        self.cfg = cfg
        self.device = cfg['base']['device']

    def evaluate(self, val_loader):
        self.model.eval()
        loss_sum = 0.0
        top1_correct = 0
        top5_correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(self.device, non_blocking=True)
                labels = batch["labels"].to(self.device, non_blocking=True)

                logits = self.model(pixel_values)
                loss_sum += F.cross_entropy(logits, labels).item()

                top1_correct += (logits.argmax(dim=-1) == labels).sum().item()

                k = min(5, logits.size(1))
                top5_indices = logits.topk(k, dim=-1).indices
                top5_correct += sum(labels[i] in top5_indices[i] for i in range(len(labels)))
                total += len(labels)

        return {
            "val_loss": loss_sum / max(1, len(val_loader)),
            "top1_acc": top1_correct / max(1, total),
            "top5_acc": top5_correct / max(1, total),
        }

    def test_sample(self, img_path, label=""):
        print(f"\n===== {label} =====")
        pil_img = Image.open(img_path).convert("RGB")
        px = self.vision_processor(images=pil_img, return_tensors="pt")["pixel_values"]
        px = px.to(self.device, dtype=torch.bfloat16)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(px)

        probs = logits.softmax(dim=-1)[0]
        k = min(5, len(probs))
        top = probs.topk(k)
        for prob, idx in zip(top.values, top.indices):
            print(f"  {self.idx_to_class[idx.item()]}: {prob.item():.4f}")
