"""测试 BiomedCLIP tokenizer"""
import json
import os
from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS

def create_biomedclip_model_and_preprocess_local(model_path: str, device: str):
    model_name = "biomedclip_local"
    config_file = os.path.join(model_path, "open_clip_config.json")
    
    with open(config_file, "r") as f:
        config = json.load(f)
        model_cfg = config["model_cfg"]
        preprocess_cfg = config["preprocess_cfg"]

    if (
        not model_name.startswith(HF_HUB_PREFIX)
        and model_name not in _MODEL_CONFIGS
        and config is not None
    ):
        _MODEL_CONFIGS[model_name] = model_cfg

    tokenizer = get_tokenizer(model_name)
    return tokenizer

if __name__ == "__main__":
    BIOMEDCLIP_MODEL_PATH = "/root/models/BiomedCLIP"
    
    tokenizer = create_biomedclip_model_and_preprocess_local(BIOMEDCLIP_MODEL_PATH, "cpu")
    
    print(f"Tokenizer 类型: {type(tokenizer)}")
    print(f"Tokenizer 属性: {[x for x in dir(tokenizer) if not x.startswith('_')]}")
    
    # 测试 tokenize
    text = "prostate"
    tokens = tokenizer(text)
    print(f"\n文本 '{text}' 的 tokens:")
    print(f"  shape: {tokens.shape}")
    print(f"  values: {tokens}")
    
    # 检查特殊token
    if hasattr(tokenizer, 'tokenizer'):
        print(f"\n内部 tokenizer: {type(tokenizer.tokenizer)}")
        if hasattr(tokenizer.tokenizer, 'cls_token_id'):
            print(f"  CLS token ID: {tokenizer.tokenizer.cls_token_id}")
        if hasattr(tokenizer.tokenizer, 'sep_token_id'):
            print(f"  SEP token ID: {tokenizer.tokenizer.sep_token_id}")
        if hasattr(tokenizer.tokenizer, 'pad_token_id'):
            print(f"  PAD token ID: {tokenizer.tokenizer.pad_token_id}")
    
    # 检查是否有 sot/eot
    if hasattr(tokenizer, 'sot_token_id'):
        print(f"  SOT token ID: {tokenizer.sot_token_id}")
    else:
        print(f"  ✗ 没有 sot_token_id")
    
    if hasattr(tokenizer, 'eot_token_id'):
        print(f"  EOT token ID: {tokenizer.eot_token_id}")
    else:
        print(f"  ✗ 没有 eot_token_id")
