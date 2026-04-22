# safetensors 파일의 키 이름을 확인할 때 사용하는 코드입니다.
# 이전에 한참 오류가 발생했었던 부분인데, mm_project라는 키워드가 앞에 붙어있어서 발생하던 오류를 
# 해결하기 위한 코드였습니다. 🥹

from safetensors.torch import load_file

orig = load_file("./fastvithd/model.safetensors")

# key 목록
orig_keys = list(orig.keys())

print("🔢 Number of keys:", len(orig_keys))
print("📌 First 50 keys:")
for k in orig_keys[:50]:
    print(k)

    
print("====================================================")
state = load_file("./fastvithd/model.safetensors")

problem_keys = [k for k in state.keys() if "mm_projector" in k]
print(len(problem_keys))
for k in problem_keys:
    print(k)


# from safetensors.torch import load_file

# state = load_file("./FastFoodVLM-0.5B/model.safetensors")

# print("---- check ----")
# for k in state.keys():
#     if "multi_modal_projector" in k:
#         print("[FOUND]", k)
