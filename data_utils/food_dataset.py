import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


train_augment = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
])


class FoodImageDataset(Dataset):
    def __init__(self, tokenizer, vision_processor, samples, cfg, augment=False):
        self.tokenizer = tokenizer
        self.vision_processor = vision_processor
        self.samples = samples
        self.cfg = cfg
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def inject_image_token(self, text, image_token_id=None):
        if image_token_id is None:
            image_token_id = self.cfg['base']['image_token_index']
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
        labels[labels == self.cfg['base']['image_token_index']] = -100

        return {
            "input_ids": input_ids,
            "labels": labels,
            # Don't confuse with real "attention" in transformer block 😂 
            # This is just for padding processing as variable length!
            "attention_mask": torch.ones_like(input_ids),
            "pixel_values": px,
        }


def collate_fn(batch, pad_id):
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [b["input_ids"] for b in batch], batch_first=True, padding_value=pad_id)
    labels = torch.nn.utils.rnn.pad_sequence(
        [b["labels"] for b in batch], batch_first=True, padding_value=-100)
    masks = torch.nn.utils.rnn.pad_sequence(
        [b["attention_mask"] for b in batch], batch_first=True, padding_value=0)
    px = torch.stack([b["pixel_values"] for b in batch], dim=0)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": masks, "pixel_values": px}
