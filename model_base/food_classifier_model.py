import gc
import json
import os
import torch
import torch.nn as nn


class MLPHead(nn.Module):
    def __init__(self, vision_dim, num_classes, hidden_dims, dropout):
        super().__init__()
        layers = []
        in_dim = vision_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.GELU(), nn.Dropout(dropout)])
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class FoodClassifier(nn.Module):
    def __init__(self, vision_encoder, vision_dim, num_classes, hidden_dims, dropout):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.head = MLPHead(vision_dim, num_classes, hidden_dims, dropout)

    def forward(self, pixel_values):
        outputs = self.vision_encoder(pixel_values=pixel_values)
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            features = outputs.pooler_output
        else:
            features = outputs.last_hidden_state.mean(dim=1)
        return self.head(features)

    def count_trainable_params(self):
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"✅ Trainable parameters: {total:,}")


def build_food_classifier(cfg, num_classes):
    from transformers import AutoModelForVision2Seq

    model_name = cfg['base']['model_name']
    device = cfg['base']['device']

    # Load on CPU first to avoid holding full model on GPU
    full_model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    )

    vision_encoder = full_model.vision_tower
    vision_dim = vision_encoder.config.hidden_size

    # Release the rest of the model before moving to device
    del full_model
    gc.collect()
    torch.cuda.empty_cache()

    vision_encoder = vision_encoder.to(device)

    for p in vision_encoder.parameters():
        p.requires_grad = False

    if cfg['train'].get('unfreeze_vision_blocks', 0) > 0:
        try:
            vit_layers = list(vision_encoder.vision_model.encoder.layers)
            for layer in vit_layers[-cfg['train']['unfreeze_vision_blocks']:]:
                for p in layer.parameters():
                    p.requires_grad = True
            print(f"✅ Vision Tower last {cfg['train']['unfreeze_vision_blocks']} blocks unfrozen")
        except Exception as e:
            print(f"⚠️  Vision Tower unfreeze failed: {e}")

    classifier = FoodClassifier(
        vision_encoder=vision_encoder,
        vision_dim=vision_dim,
        num_classes=num_classes,
        hidden_dims=cfg['model']['hidden_dims'],
        dropout=cfg['model']['dropout'],
    ).to(device)

    print(f"✅ FoodClassifier built (vision_dim={vision_dim}, num_classes={num_classes})")
    return classifier


def save_classifier(model, cfg, class_to_idx, ckpt_dir):
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(model.head.state_dict(), os.path.join(ckpt_dir, 'mlp_head.pt'))
    with open(os.path.join(ckpt_dir, 'classifier_config.json'), 'w') as f:
        json.dump({
            'num_classes': len(class_to_idx),
            'hidden_dims': cfg['model']['hidden_dims'],
            'dropout': cfg['model']['dropout'],
            'vision_dim': model.head.layers[0].in_features,
        }, f, indent=2)
    with open(os.path.join(ckpt_dir, 'class_to_idx.json'), 'w') as f:
        json.dump(class_to_idx, f, indent=2)
