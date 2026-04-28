# FastFoodVLM

## Overview

FastFoodVLM supports two training approaches for on-device food recognition:

### Approach 1 — FastVLM (VLM fine-tuning)
Fine-tunes [Apple's FastVLM-0.5B](https://huggingface.co/apple/FastVLM-0.5B) with LoRA to output food names directly from images using a Vision Language Model pipeline.

### Approach 2 — SigLIP + MLP Classifier (recommended for on-device)
Uses a frozen [SigLIP vision encoder](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-si-hf) with a lightweight MLP classification head. Much faster and more efficient for on-device inference. Converts to CoreML (vision encoder) + MLX (MLP head) for deployment.

---

## Approach 1: FastVLM Fine-tuning

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare dataset

```
dataset/
├── pizza/
│   ├── image1.jpg
│   └── image2.jpg
├── sushi/
│   └── ...
```

Each subdirectory name is used as the food label. Supported formats: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.tif`, `.tiff`.

### 3. Configure training

Edit `configs/first_config.yaml`:

```yaml
base:
  model_name: "apple/FastVLM-0.5B"
  device: "cuda"
  dataset_root: "dataset"
  save_dir: "FastFoodVLM-0.5B"
  best_ckpt_dir: "checkpoints/best"

train:
  epochs: 20
  batch_size: 4
  lr: 1.0e-4
  optimizer: "adamw"
  lora_rank: 32
  lora_alpha: 64
  unfreeze_vision_blocks: 0
```

### 4. Run training

```bash
python fastvlm_finetune.py
```

### 5. Outputs

| Path | Description |
|------|-------------|
| `checkpoints/best/` | Best checkpoint by validation loss |
| `FastFoodVLM-0.5B/` | Final merged model ready for inference |

---

## Approach 2: SigLIP + MLP Classifier

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare dataset

Same structure as Approach 1.

### 3. Configure training

Edit `configs/food_classifier_config.yaml`:

```yaml
base:
  model_name: "llava-hf/llava-onevision-qwen2-0.5b-si-hf"
  device: "cuda"
  dataset_root: "dataset"
  save_dir: "FoodClassifier/trial1"
  best_ckpt_dir: "checkpoints/classifier_trial1"

model:
  hidden_dims: [512]
  dropout: 0.1

train:
  epochs: 20
  batch_size: 32
  lr: 1.0e-3
  optimizer: "adamw"
  unfreeze_vision_blocks: 0
```

### 4. Run training (on GPU server)

```bash
python food_classifier_finetune.py
```

Training will:
- Load the SigLIP vision encoder from the pretrained model (frozen)
- Train only the MLP classification head (~650K parameters)
- Save best checkpoint by top-1 accuracy to `checkpoints/classifier_trial1/`
- Output: `mlp_head.pt`, `classifier_config.json`, `class_to_idx.json`

### 5. Convert for on-device deployment (on Mac)

After downloading the checkpoint from the server, run the two conversion scripts:

**Step 1 — Convert MLP weights to MLX format:**
```bash
python convert/convert_mlp_to_mlx.py \
    --ckpt_dir checkpoints/classifier_trial1 \
    --out_dir mlx_classifier \
    --image_size 384
```

**Step 2 — Convert SigLIP vision encoder to CoreML (only needed once):**
```bash
python convert/convert_siglip_to_coreml.py \
    --model_name llava-hf/llava-onevision-qwen2-0.5b-si-hf \
    --out_dir mlx_classifier
```

### 6. Outputs

| Path | Description |
|------|-------------|
| `mlx_classifier/mlp_head.safetensors` | MLP weights in MLX format |
| `mlx_classifier/siglip_encoder.mlpackage` | SigLIP encoder in CoreML format |
| `mlx_classifier/classifier_config.json` | Architecture config (vision_dim, hidden_dims, num_classes, image_size) |
| `mlx_classifier/class_to_idx.json` | Food class name ↔ index mapping |

Copy the `mlx_classifier/` directory into your Xcode project as bundle resources.

### On-device inference pipeline

```
Image → siglip_encoder.mlpackage (CoreML) → 1152-dim vector → MLX MLP → class index → class_to_idx.json → food name
```

---

## Contact

Minseok Kang — pilot920@hanyang.ac.kr
