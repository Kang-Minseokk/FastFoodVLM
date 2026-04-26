import os
import torch
from PIL import Image


def run_inference(model, tokenizer, vision_processor, cfg, image_path):
    if not os.path.exists(image_path):
        print(f"⚠️  테스트 이미지 없음: {image_path}")
        return None

    test_img = Image.open(image_path).convert("RGB")
    model_type = cfg['base']['model_type']

    if model_type == 'fastvlm':
        return _infer_fastvlm(model, tokenizer, vision_processor, cfg, test_img)
    elif model_type == 'siglip_qwen2':
        return _infer_siglip_qwen2(model, tokenizer, vision_processor, cfg, test_img)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'")


def _infer_fastvlm(model, tokenizer, vision_processor, cfg, test_img):
    px = vision_processor(images=test_img, return_tensors="pt")["pixel_values"].to(cfg['base']['device'], dtype=torch.bfloat16)

    infer_msg = [{"role": "user", "content": "<image>\nAnswer ONLY with the food name in one or two English words. No extra text."}]
    infer_text = tokenizer.apply_chat_template(infer_msg, tokenize=False, add_generation_prompt=True)

    pre, post = infer_text.split("<image>", 1)
    pre_ids = tokenizer(pre, add_special_tokens=False).input_ids
    post_ids = tokenizer(post, add_special_tokens=False).input_ids
    test_ids = torch.tensor([pre_ids + [cfg['base']['image_token_index']] + post_ids]).to(cfg['base']['device'])
    mask = torch.ones_like(test_ids)

    model.eval()
    with torch.no_grad():
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        gen = model.generate(
            inputs=test_ids,
            attention_mask=mask,
            images=px,
            max_new_tokens=8,
            do_sample=False,
            eos_token_id=im_end_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = gen[0, test_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _infer_siglip_qwen2(model, tokenizer, vision_processor, cfg, test_img):
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(cfg['base']['model_name'])

    infer_msg = [{"role": "user", "content": "<image>\nAnswer ONLY with the food name in one or two English words. No extra text."}]
    infer_text = tokenizer.apply_chat_template(infer_msg, tokenize=False, add_generation_prompt=True)

    inputs = processor(text=infer_text, images=test_img, return_tensors="pt")
    input_ids = inputs.input_ids.to(cfg['base']['device'])
    pixel_values = inputs.pixel_values.to(cfg['base']['device'], dtype=torch.bfloat16)
    mask = inputs.attention_mask.to(cfg['base']['device'])

    model.eval()
    with torch.no_grad():
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        gen = model.generate(
            inputs=input_ids,
            attention_mask=mask,
            pixel_values=pixel_values,
            max_new_tokens=8,
            do_sample=False,
            eos_token_id=im_end_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = gen[0, input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
