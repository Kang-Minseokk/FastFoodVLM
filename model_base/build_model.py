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
    raise NotImplementedError("siglip_qwen2 not yet implemented")


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


_PROJECTOR_UNFREEZERS = {
    "fastvlm": _unfreeze_projector_fastvlm,
    "siglip_qwen2": None,
}


def unfreeze_projector(model, cfg):
    model_type = cfg['base']['model_type']
    if model_type not in _PROJECTOR_UNFREEZERS:
        raise ValueError(f"Unknown model_type '{model_type}'")
    fn = _PROJECTOR_UNFREEZERS[model_type]
    if fn is None:
        raise NotImplementedError(f"unfreeze_projector not yet implemented for '{model_type}'")
    fn(model)
