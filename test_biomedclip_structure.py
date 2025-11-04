"""
测试 BiomedCLIP 模型结构
"""
import json
import os
import torch
from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS

def create_biomedclip_model_and_preprocess_local(model_path: str, device: str):
    """从本地路径加载 BiomedCLIP 模型及预处理配置."""
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
    if not os.path.exists(model_file):
        model_file_alt = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(model_file_alt):
            model_file = model_file_alt

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
    
    print("加载 BiomedCLIP 模型...")
    model, preprocess, tokenizer = create_biomedclip_model_and_preprocess_local(
        BIOMEDCLIP_MODEL_PATH, DEVICE
    )
    
    print("\n=== 文本编码器结构 ===")
    print(f"model.text 类型: {type(model.text)}")
    print(f"model.text 属性: {dir(model.text)}")
    
    # 检查 text_projection
    if hasattr(model.text, 'text_projection'):
        print(f"\n✓ 找到 text_projection")
        print(f"  类型: {type(model.text.text_projection)}")
        print(f"  形状: {model.text.text_projection.shape}")
    else:
        print("\n✗ 未找到 text_projection")
    
    # 检查 proj
    if hasattr(model.text, 'proj'):
        print(f"\n✓ 找到 proj")
        print(f"  类型: {type(model.text.proj)}")
        if hasattr(model.text.proj, 'shape'):
            print(f"  形状: {model.text.proj.shape}")
        else:
            print(f"  proj 是 Sequential 或其他容器类型")
            if isinstance(model.text.proj, torch.nn.Sequential):
                print(f"  Sequential 包含的层数: {len(model.text.proj)}")
                for i, layer in enumerate(model.text.proj):
                    print(f"    Layer {i}: {type(layer)}")
                    if hasattr(layer, 'weight'):
                        print(f"      权重形状: {layer.weight.shape}")
    else:
        print("\n✗ 未找到 proj")
    
    # 检查 token_embedding
    if hasattr(model.text, 'token_embedding'):
        print(f"\n✓ 找到 token_embedding")
        print(f"  类型: {type(model.text.token_embedding)}")
        print(f"  embedding_dim: {model.text.token_embedding.embedding_dim}")
    
    # 检查 transformer
    if hasattr(model.text, 'transformer'):
        print(f"\n✓ 找到 transformer")
        print(f"  类型: {type(model.text.transformer)}")
        if hasattr(model.text.transformer, 'layers'):
            print(f"  层数: {len(model.text.transformer.layers)}")
        elif hasattr(model.text.transformer, 'resblocks'):
            print(f"  层数: {len(model.text.transformer.resblocks)}")
    
    print("\n=== 视觉编码器结构 ===")
    print(f"model.visual 类型: {type(model.visual)}")
    if hasattr(model.visual, 'trunk'):
        print(f"model.visual.trunk 类型: {type(model.visual.trunk)}")
        if hasattr(model.visual.trunk, 'blocks'):
            print(f"  blocks 数量: {len(model.visual.trunk.blocks)}")
            print(f"  embed_dim: {model.visual.trunk.embed_dim}")
    
    if hasattr(model.visual, 'head'):
        print(f"\nmodel.visual.head 类型: {type(model.visual.head)}")
        if hasattr(model.visual.head, 'proj'):
            print(f"  head.proj 类型: {type(model.visual.head.proj)}")
            if hasattr(model.visual.head.proj, 'shape'):
                print(f"  head.proj 形状: {model.visual.head.proj.shape}")
    
    print("\n=== CLIP 嵌入维度 ===")
    # 正确的获取方式
    if hasattr(model.text, 'text_projection'):
        clip_dim = model.text.text_projection.shape[1]
        print(f"✓ 从 text_projection 获取: {clip_dim}")
    elif hasattr(model.text, 'output_dim'):
        clip_dim = model.text.output_dim
        print(f"✓ 从 output_dim 获取: {clip_dim}")
    
    print("\n完成!")
