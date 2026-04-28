"""
Convert mlp_head.pt (PyTorch) → mlp_head.safetensors (MLX-loadable)

Usage:
    python convert/convert_mlp_to_mlx.py \
        --ckpt_dir checkpoints/classifier_trial1 \
        --out_dir mlx_classifier \
        --image_size 384
"""
import argparse
import json
import os
import torch
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", required=True, help="Directory containing mlp_head.pt and classifier_config.json")
    parser.add_argument("--out_dir", required=True, help="Output directory for MLX weights")
    parser.add_argument("--image_size", type=int, default=384, help="SigLIP input image size (384 for so400m-patch14-384)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load PyTorch weights
    pt_path = os.path.join(args.ckpt_dir, "mlp_head.pt")
    state_dict = torch.load(pt_path, map_location="cpu")

    # Convert to float32 (MLX Swift requires safetensors, not npz)
    float_weights = {k: v.float() for k, v in state_dict.items()}

    out_path = os.path.join(args.out_dir, "mlp_head.safetensors")
    save_file(float_weights, out_path)
    print(f"✅ Saved MLX weights → {out_path}")
    print(f"   Keys: {list(float_weights.keys())}")

    # Copy config (inject image_size) and class mapping alongside weights
    config_src = os.path.join(args.ckpt_dir, "classifier_config.json")
    if os.path.exists(config_src):
        with open(config_src) as f:
            config = json.load(f)
        config["image_size"] = args.image_size
        with open(os.path.join(args.out_dir, "classifier_config.json"), "w") as f:
            json.dump(config, f, indent=2)
        print(f"✅ Saved classifier_config.json (image_size={args.image_size})")

    idx_src = os.path.join(args.ckpt_dir, "class_to_idx.json")
    if os.path.exists(idx_src):
        with open(idx_src) as f:
            data = json.load(f)
        with open(os.path.join(args.out_dir, "class_to_idx.json"), "w") as f:
            json.dump(data, f, indent=2)
        print(f"✅ Copied class_to_idx.json")


if __name__ == "__main__":
    main()
