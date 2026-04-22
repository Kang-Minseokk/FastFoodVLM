import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM

IMAGE_TOKEN_INDEX = -200
MID = "apple/FastVLM-0.5B"

def generate_tokenizer_and_model():
    tok = AutoTokenizer.from_pretrained(MID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    return tok, model

def generate_input_ids(tokenizer, device='cuda'):
    """
    [설명 나중에 작성 예정]
    """    
    
    messages = [
    {"role": "user", "content": "<image>\nDescribe this image in detail."}
    ]
        
    tok = tokenizer
    rendered = tok.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    ) 
    pre, post = rendered.split("<image>", 1)
    pre_ids  = tok(pre,  return_tensors="pt", add_special_tokens=False).input_ids
    post_ids = tok(post, return_tensors="pt", add_special_tokens=False).input_ids
    
    img_tok = torch.tensor([[IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)
    input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1).to(device=device)
    attention_mask = torch.ones_like(input_ids, device=device)
    
    return input_ids, attention_mask

def convert_px(model):
    img = Image.open("avocado1.jpg").convert("RGB")
    px = model.get_vision_tower().image_processor(images=img, return_tensors="pt")["pixel_values"]
    px = px.to(model.device, dtype=model.dtype)