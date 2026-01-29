import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.model_fn import generate_tokens, insert_image_token

# 0. Constant 정의 부분
MID = "apple/FastVLM-0.5B"
IMAGE_TOKEN_INDEX = -200  # 이는 PlaceHolder의 역할을 해준다. 값 자체가 큰 의미가 있지는 않다.

# 1. 모델 및 토크나이저 로드
tok = AutoTokenizer.from_pretrained(MID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True,
)

# 2. 챗 템플릿 생성 : 이 템플릿은 FastVLM의 허깅페이스 페이지를 참고하였습니다. 
messages = [
    {"role": "user", "content": "<image>\nDescribe this image in detail."}
]    

# 3. 챗 템플릿 기준으로 들어가게 될 입력을 토큰화해줍니다.
pre_ids, post_ids = generate_tokens(tok, messages) # 토큰화된 pre와 post를 받아옵니다.

img_tok = torch.tensor([[IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)
input_ids, attention_mask = insert_image_token(pre_ids, img_tok, post_ids)

# 4. 실제 넣을 이미지를 넣기 위해 전처리해줍니다. vision_tower의 image 처리기로 합니다.
img = Image.open("avocado1.jpg").convert("RGB")
px = model.get_vision_tower().image_processor(images=img, return_tensors="pt")["pixel_values"]
px = px.to(model.device, dtype=model.dtype)

# 5. 실제 모델에 입력을 주어 연산을 수행합니다.
with torch.no_grad():
    out = model.generate(
        inputs=input_ids,
        attention_mask=attention_mask,
        images=px,
        max_new_tokens=128,
    )

print(tok.decode(out[0], skip_special_tokens=True))

##############################################################################
# 위의 코드는 raw model을 가져와서 실제로 유효한 답변을 주는지 여부를 확인하기 위한 코드입니다.  #
# 아래부터는 실제로 학습하고자 하는 데이터를 사용하여 파인튜닝을 진행하려고 합니다               #
##############################################################################

