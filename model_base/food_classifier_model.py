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
        outputs = self.vision_encoder(pixel_values)

        if isinstance(outputs, torch.Tensor):
            if outputs.ndim == 3:
                features = outputs.mean(dim=1)
            elif outputs.ndim == 4:
                features = outputs.mean(dim=[2, 3])
            elif outputs.ndim == 2:
                features = outputs
            else:
                raise ValueError(f"Unexpected vision output shape: {tuple(outputs.shape)}")
        elif hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            features = outputs.pooler_output
        elif hasattr(outputs, 'last_hidden_state'):
            features = outputs.last_hidden_state.mean(dim=1)
        else:
            raise ValueError(f"Unsupported vision encoder output: {type(outputs)}")

        features = features.to(self.head.layers[0].weight.dtype)
        return self.head(features)

    def count_trainable_params(self):
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"✅ Trainable parameters: {total:,}")


def _load_siglip_backbone(cfg):
    from transformers import AutoModelForVision2Seq, AutoProcessor

    model_name = cfg['base']['model_name']

    full_model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    )
    vision_encoder = full_model.vision_tower
    vision_dim = vision_encoder.config.hidden_size

    del full_model
    gc.collect()
    torch.cuda.empty_cache()

    processor = AutoProcessor.from_pretrained(model_name)
    vision_processor = processor.image_processor
    if hasattr(vision_processor, 'do_image_splitting'):
        vision_processor.do_image_splitting = False

    return vision_encoder, vision_processor, vision_dim


def _load_fastvlm_backbone(cfg):
    from transformers import AutoModelForCausalLM

    model_name = cfg['base']['model_name']

    full_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    vision_tower = full_model.get_vision_tower()
    vision_processor = vision_tower.image_processor
    vision_dim = vision_tower.hidden_size

    del full_model
    gc.collect()
    torch.cuda.empty_cache()

    return vision_tower, vision_processor, vision_dim


_BACKBONE_LOADERS = {
    "siglip_qwen2": _load_siglip_backbone,
    "fastvlm": _load_fastvlm_backbone,
}


def build_vision_backbone(cfg):
    model_type = cfg['base']['model_type']
    if model_type not in _BACKBONE_LOADERS:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose from {list(_BACKBONE_LOADERS)}")
    encoder, processor, dim = _BACKBONE_LOADERS[model_type](cfg)
    print(f"✅ Vision backbone loaded (type={model_type}, vision_dim={dim})")
    return encoder, processor, dim


def _apply_freeze_policy(cfg, vision_encoder):
    model_type = cfg['base']['model_type']
    n_unfreeze = cfg['train'].get('unfreeze_vision_blocks', 0)

    for p in vision_encoder.parameters():
        p.requires_grad = False

    if n_unfreeze <= 0:
        return

    if model_type == 'fastvlm':
        # MobileCLIPVisionTower wraps forward in torch.no_grad() unless
        # tune_vision_tower=True, so requires_grad alone is insufficient.
        # FastVLM block-level unfreeze is not implemented yet — skip with warning.
        print(f"⚠️  unfreeze_vision_blocks={n_unfreeze} ignored for fastvlm "
              f"(not implemented). Encoder remains frozen.")
        return

    try:
        vit_layers = list(vision_encoder.vision_model.encoder.layers)
        for layer in vit_layers[-n_unfreeze:]:
            for p in layer.parameters():
                p.requires_grad = True
        print(f"✅ Vision Tower last {n_unfreeze} blocks unfrozen")
    except Exception as e:
        print(f"⚠️  Vision Tower unfreeze failed: {e}")


def build_food_classifier(cfg, vision_encoder, vision_dim, num_classes):
    device = cfg['base']['device']
    vision_encoder = vision_encoder.to(device)

    _apply_freeze_policy(cfg, vision_encoder)

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
