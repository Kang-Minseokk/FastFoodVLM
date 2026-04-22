from safetensors.torch import load_file, save_file
import argparse


def remap_key(k):
    # -------------------------------------------------------
    # (1) Vision Tower: model.vision_tower.xxx → vision_tower.xxx
    #     "model." 만 제거. language_model 접두어 붙이면 안 됨.
    # -------------------------------------------------------
    if k.startswith("model.vision_tower."):
        return k[len("model."):]  # "model." 만 제거

    # -------------------------------------------------------
    # (2) Projector: model.mm_projector.0.x → multi_modal_projector.linear_0.x
    #                model.mm_projector.2.x → multi_modal_projector.linear_2.x
    # -------------------------------------------------------
    if k.startswith("model.mm_projector."):
        k = k.replace("model.mm_projector.0.", "multi_modal_projector.linear_0.")
        k = k.replace("model.mm_projector.2.", "multi_modal_projector.linear_2.")
        return k

    # -------------------------------------------------------
    # (3) LLM: model.xxx → language_model.model.xxx
    #          lm_head.xxx → language_model.lm_head.xxx
    # -------------------------------------------------------
    if k.startswith("model."):
        return "language_model." + k  # model.layers... → language_model.model.layers...

    if k.startswith("lm_head."):
        return "language_model." + k  # lm_head.weight → language_model.lm_head.weight

    return k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", type=str, required=True)
    parser.add_argument("--outfile", type=str, required=True)
    args = parser.parse_args()

    print(f"Loading: {args.infile}")
    w = load_file(args.infile)

    new_state = {}
    for k, v in w.items():
        new_k = remap_key(k)
        new_state[new_k] = v

    # 결과 샘플 출력
    print(f"\n총 키 수: {len(new_state)}")
    print("\n--- 변환 샘플 (앞 10개) ---")
    for old, new in zip(list(w.keys())[:10], list(new_state.keys())[:10]):
        print(f"  {old}")
        print(f"  → {new}\n")

    print("--- projector 키 ---")
    for k in new_state.keys():
        if "projector" in k:
            print(f"  {k}")

    print("\n--- vision_tower 키 (앞 3개) ---")
    vt_keys = [k for k in new_state.keys() if "vision_tower" in k]
    for k in vt_keys[:3]:
        print(f"  {k}")

    save_file(new_state, args.outfile)
    print(f"\n✅ 저장 완료 → {args.outfile}")


if __name__ == "__main__":
    main()
