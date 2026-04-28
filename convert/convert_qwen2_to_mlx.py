"""
Convert Qwen2-0.5B-Instruct (HuggingFace) → MLX format

Requires: pip install mlx-lm

Usage:
    python convert/convert_qwen2_to_mlx.py --out_dir mlx_qwen2_0.5b

Output: mlx_qwen2_0.5b/ directory ready to copy into the app's Documents folder.
Note: 4-bit quantization reduces model size from ~1GB to ~250MB.
"""
import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="mlx_qwen2_0.5b")
    parser.add_argument("--quantize", action="store_true", default=True,
                        help="4-bit quantize to reduce size (~250MB vs ~1GB)")
    args = parser.parse_args()

    cmd = [
        sys.executable, "-m", "mlx_lm.convert",
        "--hf-path", "Qwen/Qwen2-0.5B-Instruct",
        "--mlx-path", args.out_dir,
    ]
    if args.quantize:
        cmd += ["-q"]

    print(f"Converting Qwen2-0.5B-Instruct → {args.out_dir} ...")
    subprocess.run(cmd, check=True)
    print(f"✅ Done → {args.out_dir}")
    print(f"   Copy this directory to your device's Documents/mlx_qwen2_0.5b/")


if __name__ == "__main__":
    main()
