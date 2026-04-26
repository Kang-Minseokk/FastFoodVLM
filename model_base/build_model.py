import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _build_fastvlm(cfg):
    model_name = cfg['base']['model_name']
    device = cfg['base']['device']

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)

    vision_tower = model.get_vision_tower()
    vision_processor = vision_tower.image_processor

    for p in vision_tower.parameters():
        p.requires_grad = False

    if cfg['train']['unfreeze_vision_blocks'] > 0:
        try:
            vit_layers = list(vision_tower.vision_model.encoder.layers)
            for layer in vit_layers[-cfg['train']['unfreeze_vision_blocks']:]:
                for p in layer.parameters():
                    p.requires_grad = True
            print(f"✅ Vision Tower last {cfg['train']['unfreeze_vision_blocks']} blocks unfrozen")
        except Exception as e:
            print(f"⚠️  Vision Tower unfreeze failed: {e}")

    return model, tokenizer, vision_processor


def _build_siglip_qwen2(cfg):
    from transformers import AutoModelForVision2Seq, AutoProcessor

    model_name = cfg['base']['model_name']
    device = cfg['base']['device']

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    ).to(device)

    model_img_token = model.config.image_token_index
    if cfg['base']['image_token_index'] != model_img_token:
        print(f"⚠️  image_token_index config={cfg['base']['image_token_index']} "
              f"!= model={model_img_token}. Overriding with model value.")
        cfg['base']['image_token_index'] = model_img_token

    for p in model.vision_tower.parameters():
        p.requires_grad = False

    if cfg['train']['unfreeze_vision_blocks'] > 0:
        try:
            vit_layers = list(model.vision_tower.vision_model.encoder.layers)
            for layer in vit_layers[-cfg['train']['unfreeze_vision_blocks']:]:
                for p in layer.parameters():
                    p.requires_grad = True
            print(f"✅ Vision Tower last {cfg['train']['unfreeze_vision_blocks']} blocks unfrozen")
        except Exception as e:
            print(f"⚠️  Vision Tower unfreeze failed: {e}")

    return model, processor.tokenizer, processor.image_processor


_BUILDERS = {
    "fastvlm": _build_fastvlm,
    "siglip_qwen2": _build_siglip_qwen2,
}


def build_model(cfg):
    model_type = cfg['base']['model_type']
    if model_type not in _BUILDERS:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose from {list(_BUILDERS)}")

    model, tokenizer, vision_processor = _BUILDERS[model_type](cfg)
    print(f"✅ Model loaded (type={model_type})")
    return model, tokenizer, vision_processor


def _unfreeze_projector_fastvlm(model):
    projector = model.base_model.model.model.mm_projector
    for p in projector.parameters():
        p.requires_grad = True
    print("✅ Projector unfrozen: base_model.model.model.mm_projector")


def _unfreeze_projector_siglip_qwen2(model):
    projector = model.base_model.model.multi_modal_projector
    for p in projector.parameters():
        p.requires_grad = True
    print("✅ Projector unfrozen: base_model.model.multi_modal_projector")


_PROJECTOR_UNFREEZERS = {
    "fastvlm": _unfreeze_projector_fastvlm,
    "siglip_qwen2": _unfreeze_projector_siglip_qwen2,
}


def unfreeze_projector(model, cfg):
    model_type = cfg['base']['model_type']
    if model_type not in _PROJECTOR_UNFREEZERS:
        raise ValueError(f"Unknown model_type '{model_type}'")
    _PROJECTOR_UNFREEZERS[model_type](model)


def forward_model(model, batch, cfg):
    model_type = cfg['base']['model_type']
    if model_type == 'fastvlm':
        return model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            images=batch["pixel_values"],
            labels=batch["labels"],
        )
    elif model_type == 'siglip_qwen2':
        return model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
            labels=batch["labels"],
        )
    else:
        raise ValueError(f"Unknown model_type '{model_type}'")
