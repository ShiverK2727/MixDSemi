import json
import os
import torch
import torch.nn.functional as F
from open_clip import get_tokenizer, create_model_and_transforms
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS

def create_biomedclip_model_and_preprocess_local(model_path, device):
    # Load the model and config files
    model_name = "biomedclip_local"
    with open(os.path.join(model_path, "open_clip_config.json"), "r") as f:
        config = json.load(f)
        model_cfg = config["model_cfg"]
        preprocess_cfg = config["preprocess_cfg"]

    if (not model_name.startswith(HF_HUB_PREFIX)
        and model_name not in _MODEL_CONFIGS
        and config is not None):
        _MODEL_CONFIGS[model_name] = model_cfg

    tokenizer = get_tokenizer(model_name)

    model, _, preprocess = create_model_and_transforms(
        model_name=model_name,
        pretrained=os.path.join(model_path, "open_clip_pytorch_model.bin"),
        **{f"image_{k}": v for k, v in preprocess_cfg.items()},
        device=device
    )

    return model, preprocess, tokenizer

