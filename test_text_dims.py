"""检查 BiomedCLIP 文本编码器的维度"""
import json
import os
import torch
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

    model_file = os.path.join(model_path, "open_clip_pytorch_model.bin")
    model, _, preprocess = create_model_and_transforms(
        model_name=model_name,
        pretrained=model_file,
        **{f"image_{k}": v for k, v in preprocess_cfg.items()},
        device=device,
    )

    return model, preprocess, tokenizer

if __name__ == "__main__":
    BIOMEDCLIP_MODEL_PATH = "/root/models/BiomedCLIP"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    model, preprocess, tokenizer = create_biomedclip_model_and_preprocess_local(
        BIOMEDCLIP_MODEL_PATH, DEVICE
    )
    
    print("=== 文本编码器维度信息 ===")
    
    # BERT embeddings
    if hasattr(model.text.transformer, 'embeddings'):
        word_emb = model.text.transformer.embeddings.word_embeddings
        print(f"BERT word_embeddings 维度: {word_emb.embedding_dim}")
        print(f"  权重形状: {word_emb.weight.shape}")
        
        pos_emb = model.text.transformer.embeddings.position_embeddings
        print(f"\nBERT position_embeddings 维度: {pos_emb.embedding_dim}")
        print(f"  权重形状: {pos_emb.weight.shape}")
    
    # Projection layer
    if hasattr(model.text, 'proj'):
        print(f"\n投影层 (proj) 结构:")
        for i, layer in enumerate(model.text.proj):
            print(f"  Layer {i}: {layer}")
            if hasattr(layer, 'in_features'):
                print(f"    输入维度: {layer.in_features}")
                print(f"    输出维度: {layer.out_features}")
    
    # Output dimension
    print(f"\n最终输出维度 (output_dim): {model.text.output_dim}")
    
    print("\n=== 推荐配置 ===")
    print(f"TextPromptConfig.embed_dim 应设置为: 768 (BERT内部维度)")
    print(f"最终输出会通过 proj 投影到: 512 (CLIP空间)")
