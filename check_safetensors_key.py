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
