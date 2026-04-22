from safetensors.torch import load_file, save_file

src_path = "./prefix_model.safetensors"
dst_path = "./FastFoodVLM-0.5B/model_fixed.safetensors"

state = load_file(src_path)

# =====================================================
# 1) projector rename 먼저 (language_model 포함된 원본 우선 처리)
# =====================================================
rename_map = {
    "language_model.multi_modal_projector.0.weight": "multi_modal_projector.linear_0.weight",
    "language_model.multi_modal_projector.0.bias":   "multi_modal_projector.linear_0.bias",
    "language_model.multi_modal_projector.2.weight": "multi_modal_projector.linear_2.weight",
    "language_model.multi_modal_projector.2.bias":   "multi_modal_projector.linear_2.bias",

    # 혹시 strip 이후 형태도 감안해서 포함
    "multi_modal_projector.0.weight": "multi_modal_projector.linear_0.weight",
    "multi_modal_projector.0.bias":   "multi_modal_projector.linear_0.bias",
    "multi_modal_projector.2.weight": "multi_modal_projector.linear_2.weight",
    "multi_modal_projector.2.bias":   "multi_modal_projector.linear_2.bias",
}

tmp_state = {}
for k, v in state.items():
    if k in rename_map:
        new_k = rename_map[k]
        print(f"[RENAME] {k} -> {new_k}")
        tmp_state[new_k] = v
    else:
        tmp_state[k] = v

# =====================================================
# 2) 마지막으로 language_model. prefix 완전 제거 (전체)
# =====================================================
final_state = {}
for k, v in tmp_state.items():
    kk = k.replace("language_model.", "")
    final_state[kk] = v

# =====================================================
# 3) 저장
# =====================================================
save_file(final_state, dst_path)
print(f"✅ Saved → {dst_path}")


# =====================================================
# 3) 삭제
# =====================================================
# state = load_file("./FastFoodVLM-0.5B/model.safetensors")

# clean = {}
# for k, v in state.items():
#     if not k.startswith("multi_modal_projector."):
#         clean[k] = v

# save_file(clean, "./FastFoodVLM-0.5B/model.safetensors")
