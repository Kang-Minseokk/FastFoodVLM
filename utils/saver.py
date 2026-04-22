def merge_and_save(model, tokenizer, cfg):
    merged = model.merge_and_unload()
    merged.save_pretrained(cfg['base']['save_dir'])
    tokenizer.save_pretrained(cfg['base']['save_dir'])
    print(f"✅ Merged model saved to ./{cfg['base']['save_dir']}/")
    print(f"   Best checkpoint (lowest val_loss): ./{cfg['base']['best_ckpt_dir']}/")
