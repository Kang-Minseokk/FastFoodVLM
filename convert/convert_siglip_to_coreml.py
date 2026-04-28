"""
Convert SigLIP vision encoder from llava-onevision to CoreML (.mlpackage)

Run this on Mac (requires coremltools, torch, transformers):
    python convert/convert_siglip_to_coreml.py \
        --model_name llava-hf/llava-onevision-qwen2-0.5b-si-hf \
        --out_dir mlx_classifier

Output: mlx_classifier/siglip_encoder.mlpackage
"""
import argparse
import gc
import torch
import coremltools as ct
from transformers import AutoModelForVision2Seq, AutoProcessor


class SigLIPEncoderWrapper(torch.nn.Module):
    """Wraps just the vision tower and returns pooler_output."""
    def __init__(self, vision_tower):
        super().__init__()
        self.vision_tower = vision_tower

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.vision_tower(pixel_values=pixel_values)
        if outputs.pooler_output is not None:
            return outputs.pooler_output
        return outputs.last_hidden_state.mean(dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="llava-hf/llava-onevision-qwen2-0.5b-si-hf")
    parser.add_argument("--out_dir", default="mlx_classifier")
    args = parser.parse_args()

    import os
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading model: {args.model_name}")
    processor = AutoProcessor.from_pretrained(args.model_name)
    processor.image_processor.do_image_splitting = False

    full_model = AutoModelForVision2Seq.from_pretrained(args.model_name, torch_dtype=torch.float32)
    vision_tower = full_model.vision_tower.eval()

    # Get expected image size from config
    image_size = vision_tower.config.image_size
    print(f"SigLIP image size: {image_size}x{image_size}")

    del full_model
    gc.collect()

    wrapper = SigLIPEncoderWrapper(vision_tower).eval()

    # Trace with example input (float32 required for CoreML conversion)
    example_input = torch.zeros(1, 3, image_size, image_size, dtype=torch.float32)
    print("Tracing model...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example_input)

    print("Converting to CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(
            name="pixel_values",
            shape=(1, 3, image_size, image_size),
            dtype=float,
        )],
        outputs=[ct.TensorType(name="pooler_output")],
        minimum_deployment_target=ct.target.iOS16,
        compute_precision=ct.precision.FLOAT16,
    )

    out_path = os.path.join(args.out_dir, "siglip_encoder.mlpackage")
    mlmodel.save(out_path)
    print(f"✅ Saved CoreML model → {out_path}")


if __name__ == "__main__":
    main()
