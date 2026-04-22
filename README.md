# FastFoodVLM

## Overview

FastFoodVLM is a fine-tuned version of [Apple's FastVLM-0.5B](https://huggingface.co/apple/FastVLM-0.5B) trained on a food image dataset.

The project targets two goals:

1. **Instruction Tuning** — The base FastVLM model tends to produce verbose, descriptive responses even for simple questions. We fine-tune it to follow short, direct instructions and respond with only the food name in one or two words.

2. **Food Domain Adaptation** — We train on a food classification dataset to improve the model's ability to correctly identify food categories, going beyond the general visual knowledge of the base model.

We are also planning to extend this work to:
- Larger FastVLM variants (e.g. FastVLM-7B)
- Alternative vision projectors
- Alternative LLM backbones

---

## How to Train

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare your dataset

Organize your food images in the following structure under the `dataset/` directory:

```
dataset/
├── pizza/
│   ├── image1.jpg
│   └── image2.jpg
├── burger/
│   ├── image1.jpg
│   └── ...
└── sushi/
    └── ...
```

Each subdirectory name is used as the food label. Supported formats: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.tif`, `.tiff`.

### 3. Configure training

Edit `configs/first_config.yaml` to adjust training settings:

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
  optimizer: "adamw"   # options: adamw, adam, sgd
  lora_rank: 32
  lora_alpha: 64
  unfreeze_vision_blocks: 0  # set > 0 to unfreeze top N vision transformer blocks
```

### 4. Run training

```bash
python fastvlm_finetune.py
```

Training will:
- Run a before fine-tuning inference test on the sample image
- Train for the configured number of epochs with cosine LR scheduling and 1-epoch warmup
- Save the best checkpoint (by validation loss) to `checkpoints/best/`
- Run an after fine-tuning inference test
- Merge LoRA weights and save the final model to `FastFoodVLM-0.5B/`

### 5. Outputs

| Path | Description |
|------|-------------|
| `checkpoints/best/` | Best checkpoint by validation loss |
| `FastFoodVLM-0.5B/` | Final merged model ready for inference |

---

## Contact

Minseok Kang — pilot920@hanyang.ac.kr
