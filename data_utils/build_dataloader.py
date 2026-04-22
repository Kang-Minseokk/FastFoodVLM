import os
import random
from collections import defaultdict
from functools import partial
from torch.utils.data import DataLoader, WeightedRandomSampler
from data_utils.food_dataset import FoodImageDataset, collate_fn


def build_dataloader(cfg, tokenizer, vision_processor):
    all_samples = []
    for food_name in sorted(os.listdir(cfg['base']['dataset_root'])):
        food_dir = os.path.join(cfg['base']['dataset_root'], food_name)
        if not os.path.isdir(food_dir):
            continue
        for file in os.listdir(food_dir):
            if file.lower().endswith(tuple(cfg['base']['image_extensions'])):
                all_samples.append((os.path.join(food_dir, file), food_name))

    class_to_samples = defaultdict(list)
    for s in all_samples:
        class_to_samples[s[1]].append(s)

    train_samples, val_samples = [], []
    for cls, samps in class_to_samples.items():
        random.shuffle(samps)
        n_val = max(1, int(len(samps) * cfg['train']['val_split']))
        val_samples.extend(samps[:n_val])
        train_samples.extend(samps[n_val:])

    class_counts = defaultdict(int)
    for _, fname in train_samples:
        class_counts[fname] += 1
    sample_weights = [1.0 / class_counts[fname] for _, fname in train_samples]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_samples), replacement=True)

    train_dataset = FoodImageDataset(tokenizer, vision_processor, train_samples, cfg, augment=True)
    val_dataset = FoodImageDataset(tokenizer, vision_processor, val_samples, cfg, augment=False)

    _collate_fn = partial(collate_fn, pad_id=tokenizer.eos_token_id)

    train_loader = DataLoader(train_dataset,
                              batch_size=cfg['train']['batch_size'],
                              sampler=sampler,
                              collate_fn=_collate_fn,
                              num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=cfg['train']['batch_size'],
                            shuffle=False,
                            collate_fn=_collate_fn)

    print(f"✅ Dataset: train={len(train_dataset)}, val={len(val_dataset)}, classes={len(class_counts)}")

    return train_loader, val_loader
