import os
import json
import random
from collections import defaultdict
from torch.utils.data import DataLoader, WeightedRandomSampler
from data_utils.food_classifier_dataset import FoodClassifierDataset, classifier_collate_fn


def build_classifier_dataloader(cfg, vision_processor):
    all_samples = []
    for food_name in sorted(os.listdir(cfg['base']['dataset_root'])):
        food_dir = os.path.join(cfg['base']['dataset_root'], food_name)
        if not os.path.isdir(food_dir):
            continue
        for file in os.listdir(food_dir):
            if file.lower().endswith(tuple(cfg['base']['image_extensions'])):
                all_samples.append((os.path.join(food_dir, file), food_name))

    class_names = sorted({s[1] for s in all_samples})
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    os.makedirs(cfg['base']['save_dir'], exist_ok=True)
    with open(os.path.join(cfg['base']['save_dir'], 'class_to_idx.json'), 'w') as f:
        json.dump(class_to_idx, f, indent=2)

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

    train_dataset = FoodClassifierDataset(vision_processor, train_samples, class_to_idx, augment=True)
    val_dataset = FoodClassifierDataset(vision_processor, val_samples, class_to_idx, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['train']['batch_size'],
        sampler=sampler,
        collate_fn=classifier_collate_fn,
        num_workers=8,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg['train']['batch_size'],
        shuffle=False,
        collate_fn=classifier_collate_fn,
        num_workers=8,
        pin_memory=True,
    )

    print(f"✅ Classifier dataset: train={len(train_dataset)}, val={len(val_dataset)}, classes={len(class_names)}")
    return train_loader, val_loader, class_to_idx
