from safetensors.torch import load_file, save_file
import re
import argparse
import torch

def remap_key(k):
    new_k = k

    # (A) LLM backbone prefix mapping
    if new_k.startswith("model."):
        new_k = new_k.replace("model.", "language_model.model.")

    # (B) projector mapping
    # model.mm_projector.linear_0 → multi_modal_projector.linear_0
    if "mm_projector" in new_k:
        new_k = new_k.replace("language_model.model.mm_projector",
                              "multi_modal_projector")

    return new_k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", type=str, required=True)
    parser.add_argument("--outfile", type=str, required=True)
    args = parser.parse_args()

    print(f"✅ Loading weights: {args.infile}")
    w = load_file(args.infile)

    new_state = {}
    skipped = []

    for k, v in w.items():
        new_k = remap_key(k)

        if new_k is None:
            skipped.append(k)
        else:
            new_state[new_k] = v

    print(f"✅ # loaded:   {len(w)}")
    print(f"✅ # remapped: {len(new_state)}")
    print(f"⚠️  # skipped: {len(skipped)}")

    if skipped:
        print("⚠️ Skipped keys:")
        for s in skipped[:20]:
            print("   -", s)
        if len(skipped) > 20:
            print(" ...")

    save_file(new_state, args.outfile)
    print(f"✅ Saved remapped weights → {args.outfile}")


if __name__ == "__main__":
    main()
