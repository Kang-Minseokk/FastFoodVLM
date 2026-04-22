from inference.run_infer import run_inference


def test_before_finetune(model, tokenizer, vision_processor, cfg):
    print("\n===== Before Fine-Tuning =====")
    result = run_inference(model, tokenizer, vision_processor, cfg, cfg['base']['test_image_path'])
    if result is not None:
        print(result)


def test_after_finetune(model, tokenizer, vision_processor, cfg):
    print("\n===== After Fine-Tuning =====")
    result = run_inference(model, tokenizer, vision_processor, cfg, cfg['base']['test_image_path'])
    if result is not None:
        print(result)
