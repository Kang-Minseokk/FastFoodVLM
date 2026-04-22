import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "apple/FastVLM-0.5B"
DEVICE = "cuda"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).to(DEVICE)

inputs = tokenizer("Sau OK.", return_tensors="pt").to(DEVICE)

with torch.no_grad():
    out_ids = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
    )
print("token ids:", out_ids[0][:30].tolist())
print("text:", tokenizer.decode(out_ids[0], skip_special_tokens=False))
