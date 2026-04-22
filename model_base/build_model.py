import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_model(cfg):
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

    print("✅ Model loaded + Vision Tower frozen")

    return model, tokenizer, vision_processor, vision_tower
