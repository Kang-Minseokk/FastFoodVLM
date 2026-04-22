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

    if cfg['train']['unfreeze_vision_blocks'] > 0:
        try:
            vit_layers = list(vision_tower.vision_model.encoder.layers)
            for layer in vit_layers[-cfg['train']['unfreeze_vision_blocks']:]:
                for p in layer.parameters():
                    p.requires_grad = True
            print(f"✅ Vision Tower 마지막 {cfg['train']['unfreeze_vision_blocks']}개 블록 unfrozen")
        except Exception as e:
            print(f"⚠️  Vision Tower unfreeze 실패: {e}")

    print("✅ Model loaded + Vision Tower frozen")

    return model, tokenizer, vision_processor


PROJECTOR_KEYWORDS = ["mm_projector", "multi_modal_projector", "projector", "connector", "mm_proj"]

def find_and_unfreeze_projector(model):
    unfrozen = []
    seen = set()

    for name, module in model.named_modules():
        name_lower = name.lower()
        if any(kw in name_lower for kw in PROJECTOR_KEYWORDS) and len(name.split(".")) <= 5:
            if any(name.startswith(seen_name + ".") for seen_name in seen):
                continue
            for p in module.parameters():
                p.requires_grad = True
            seen.add(name)
            unfrozen.append(name)

    if unfrozen:
        for n in unfrozen:
            print(f"✅ mm_projector unfrozen: {n}")
    else:
        print("⚠️  projector를 찾지 못했습니다. 아래 모듈 목록을 확인하세요:")
        for name, _ in model.named_modules():
            if len(name.split(".")) <= 3:
                print(f"   {name}")

    return len(unfrozen) > 0
