import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


train_augment = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
])


class FoodClassifierDataset(Dataset):
    def __init__(self, vision_processor, samples, class_to_idx, augment=False):
        self.vision_processor = vision_processor
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, food_name = self.samples[idx]
        pil_img = Image.open(img_path).convert("RGB")
        if self.augment:
            pil_img = train_augment(pil_img)

        px = self.vision_processor(images=pil_img, return_tensors="pt")["pixel_values"][0]
        label = self.class_to_idx[food_name]
        return {
            "pixel_values": px.to(torch.bfloat16),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def classifier_collate_fn(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }
