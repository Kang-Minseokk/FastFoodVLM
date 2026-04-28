"""
Convert mlp_head.pt (PyTorch) → mlp_head.npz (MLX-loadable)

Usage:
    python convert/convert_mlp_to_mlx.py \
        --ckpt_dir checkpoints/classifier_trial1 \
        --out_dir mlx_classifier
"""
import argparse
import json
import os
import torch
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", required=True, help="Directory containing mlp_head.pt and classifier_config.json")
    parser.add_argument("--out_dir", required=True, help="Output directory for MLX weights")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load PyTorch weights
    pt_path = os.path.join(args.ckpt_dir, "mlp_head.pt")
    state_dict = torch.load(pt_path, map_location="cpu")

    # Convert to float32 numpy (MLX doesn't support bfloat16 natively)
    np_weights = {k: v.float().numpy() for k, v in state_dict.items()}

    out_path = os.path.join(args.out_dir, "mlp_head.npz")
    np.savez(out_path, **np_weights)
    print(f"✅ Saved MLX weights → {out_path}")
    print(f"   Keys: {list(np_weights.keys())}")

    # Copy config and class mapping alongside weights
    for fname in ("classifier_config.json", "class_to_idx.json"):
        src = os.path.join(args.ckpt_dir, fname)
        dst = os.path.join(args.out_dir, fname)
        if os.path.exists(src):
            with open(src) as f:
                data = json.load(f)
            with open(dst, "w") as f:
                json.dump(data, f, indent=2)
            print(f"✅ Copied {fname} → {dst}")


if __name__ == "__main__":
    main()
