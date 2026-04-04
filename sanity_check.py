# sanity_infer_fastfoodvlm.py
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM

# ✅ FastVLM은 <image>가 실제 토큰이 아니라 "센티넬 id"로 처리되는 경우가 많음
IMAGE_TOKEN_INDEX = -200

MODEL_DIR = "./FastFoodVLM-0.5B"
IMAGE_PATH = "./debug_cheesecake.png"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

def inject_image_token(tokenizer, text: str, image_token_id=IMAGE_TOKEN_INDEX):
    """'<image>' placeholder를 센티넬 토큰 id로 치환해서 input_ids를 만든다."""
    if "<image>" not in text:
        raise ValueError("Prompt must contain <image> placeholder.")
    pre, post = text.split("<image>", 1)

    pre_ids  = tokenizer(pre,  add_special_tokens=False).input_ids
    post_ids = tokenizer(post, add_special_tokens=False).input_ids

    # tokenizer(...)가 list[list[int]] 형태로 나오는 경우 방어
    if isinstance(pre_ids[0], list):  pre_ids  = pre_ids[0]
    if isinstance(post_ids[0], list): post_ids = post_ids[0]

    merged = pre_ids + [image_token_id] + post_ids
    return torch.tensor([merged], dtype=torch.long)

def get_pixel_values(model, pil_img):
    """
    로컬 모델에서 vision image_processor를 얻어서 pixel_values 생성.
    (AutoProcessor가 안될 수 있으니 model.get_vision_tower().image_processor를 사용)
    """
    vision_processor = model.get_vision_tower().image_processor
    px = vision_processor(images=pil_img, return_tensors="pt")["pixel_values"]
    return px

def run_infer():
    print(f"[INFO] device={DEVICE}, dtype={DTYPE}")
    print(f"[INFO] loading model from: {MODEL_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=DTYPE,
        trust_remote_code=True
    ).to(DEVICE)
    model.eval()

    # --- 이미지 준비 ---
    pil_img = Image.open(IMAGE_PATH).convert("RGB")
    px = get_pixel_values(model, pil_img).to(DEVICE).to(DTYPE)

    # --- 프롬프트(학습과 동일하게) ---
    messages = [
        {
            "role": "user",
            "content": "<image>\nAnswer ONLY with the food name in one or two English words. No extra text."
        }
    ]

    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    input_ids = inject_image_token(tokenizer, chat_text).to(DEVICE)
    attn_mask = torch.ones_like(input_ids, device=DEVICE)

    # --- generate 호출 (모델 구현에 따라 인자명이 다를 수 있어 try 2번) ---
    with torch.no_grad():
        try:
            gen = model.generate(
                inputs=input_ids,
                attention_mask=attn_mask,
                images=px,
                max_new_tokens=16,
                do_sample=False,
            )
        except TypeError:
            # 어떤 구현은 images 대신 pixel_values를 씀
            gen = model.generate(
                inputs=input_ids,
                attention_mask=attn_mask,
                pixel_values=px,
                max_new_tokens=16,
                do_sample=False,
            )

    out = tokenizer.decode(gen[0], skip_special_tokens=True).strip()

    print("\n===== OUTPUT =====")
    print(out)

if __name__ == "__main__":
    run_infer()