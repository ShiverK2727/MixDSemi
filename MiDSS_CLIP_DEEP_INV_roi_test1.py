import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import json
# 导入 Invariant-Only VPT-Deep 版本
from biomedclip_vpt_invariant_only import build_invariant_prompt_image_encoder

# --- 1. 用户配置：请修改以下路径 ---
BIOMEDCLIP_PATH = "/root/models/BiomedCLIP" 
# 确保这个路径指向您 *新* 训练的 Invariant-Only 权重
VPT_WEIGHTS_PATH = "/app/MixDSemi/SynFoCLIP/model/prostate/train_MiDSS_NDC_MARK2_CLIP_DEEP_INV/vpt_basic_test_d1_vpt6/vpt_final_weights.pth"
IMAGE_PATH = "/app/MixDSemi/data/ProstateSlice/UCL/train/image/00_08.png"

# --- 2. 文本锚点配置 (与新训练脚本一致) ---
TEXT_ROOT_PATH = "/app/MixDSemi/SynFoCLIP/code/text"
TEXT_DATASET_NAME = "ProstateSlice"
LLM_MODEL = "GPT5"
DESCRIBE_NUMS = 80 # 保持与训练一致

# --- 3. 模型和VPT配置 (与新训练脚本一致) ---
NUM_PROMPTS = 4 
EMBED_DIM = 768 
ENABLE_PROMPT_SCALE = not True 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------

def setup_model(biomedclip_path, vpt_weights_path, num_prompts, embed_dim, enable_scale, device):
    """
    加载 BiomedCLIP, 注入 Invariant-Only VPT-Deep, 并加载权重。
    """
    print(f"正在从 {biomedclip_path} 加载 BiomedCLIP (Invariant-Only)...")
    model, preprocess, tokenizer = build_invariant_prompt_image_encoder(
        model_path=biomedclip_path,
        device=device,
        num_prompts=num_prompts,
        embed_dim=embed_dim,
        enable_prompt_scale=enable_scale,
        freeze_backbone=True,
    )
    
    print(f"正在从 {vpt_weights_path} 加载 Invariant-Only 权重...")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        model.load_prompts(vpt_weights_path, map_location=device)
        
    model.eval()
    print("模型准备就绪 (eval mode)。")
    return model, preprocess, tokenizer

def load_training_texts(text_root, dataset_name, llm_model, describe_nums):
    """
    加载训练时使用的所有文本描述 (a_global_anchor 的基础)。
    """
    text_file = os.path.join(text_root, f"{dataset_name}.json")
    if not os.path.exists(text_file):
        raise FileNotFoundError(f"找不到文本文件: {text_file}")
        
    print(f"正在从 {text_file} 加载训练文本...")
    with open(text_file, 'r') as f:
        text_descriptions = json.load(f)

    if llm_model not in text_descriptions:
        raise KeyError(f"LLM 模型 '{llm_model}' 不在 {text_file} 中。")
        
    all_texts_by_type = text_descriptions[llm_model]
    
    flat_list = []
    for type_key, texts in all_texts_by_type.items():
        flat_list.extend(texts[:describe_nums]) 

    if len(flat_list) == 0:
        raise ValueError("未能从 JSON 文件中加载任何文本。")
        
    print(f"成功加载 {len(flat_list)} 条训练文本描述。")
    return flat_list


def generate_heatmap_text_patch(model, preprocess, tokenizer, image_pil, text_anchor, device):
    """
    为单个图像和“平均文本锚点”生成热图。
    (已移除 prompt_group)
    """
    
    # --- 步骤 1: 准备图像 ---
    processed_image = preprocess(image_pil).unsqueeze(0).to(device) 

    # --- 步骤 2: 获取VPT调节的Patch特征 ---
    with torch.no_grad():
        outputs = model.forward(
            processed_image, 
            return_tokens=True, 
            normalize=False 
        )

    all_tokens = outputs["tokens"] 
    patch_tokens = all_tokens[:, 1:, :] # [1, N, 768]

    # --- 步骤 3: 特征对齐与相似度计算 ---
    head = model.model.visual.head
    projected_patch_tokens = head(patch_tokens) # [1, N, 512]
    projected_patch_tokens = F.normalize(projected_patch_tokens, dim=-1)
    
    # text_anchor 形状: [1, 512]
    similarity_map = torch.einsum('bpd,bd->bp', projected_patch_tokens, text_anchor)
    similarity_map = similarity_map.squeeze(0).detach() # [N]

    # --- 步骤 4: Reshape 和上采样 ---
    num_patches = similarity_map.shape[0]
    num_patches_side = int(math.sqrt(num_patches) + 1e-9)
    
    if num_patches_side * num_patches_side != num_patches:
        raise ValueError(f"Patch 数量 ({num_patches}) 不是一个完美的平方数，无法 reshape。")
        
    heatmap_low_res = similarity_map.reshape(1, 1, num_patches_side, num_patches_side)
    
    heatmap_high_res = F.interpolate(
        heatmap_low_res,
        size=(image_pil.height, image_pil.width), 
        mode='bilinear',
        align_corners=False
    )
    
    heatmap_numpy = heatmap_high_res.squeeze().cpu().numpy()
    
    min_val = heatmap_numpy.min()
    max_val = heatmap_numpy.max()
    heatmap_normalized = (heatmap_numpy - min_val) / (max_val - min_val + 1e-8)
    
    return heatmap_normalized


def visualize_heatmap(original_image, heatmap_inv, text_count): # 简化
    fig, axes = plt.subplots(1, 2, figsize=(12, 6)) # 1x2 布局
    
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(original_image, cmap='gray')
    axes[1].imshow(heatmap_inv, cmap='jet', alpha=0.5)
    axes[1].set_title("Invariant Heatmap (Text-Patch)")
    axes[1].axis('off')
    
    fig.suptitle(f"Heatmap based on Average of {text_count} Texts (Invariant-Only)", fontsize=16)
    plt.tight_layout()
    
    save_path = "vpt_heatmap_invariant_only.png"
    plt.savefig(save_path)
    print(f"热图已保存到: {save_path}")

if __name__ == "__main__":
    paths_to_check = {
        "BIOMEDCLIP_PATH": BIOMEDCLIP_PATH,
        "VPT_WEIGHTS_PATH": VPT_WEIGHTS_PATH,
        "IMAGE_PATH": IMAGE_PATH,
        "TEXT_ROOT_PATH": TEXT_ROOT_PATH,
    }
    
    all_paths_ok = True
    for name, path in paths_to_check.items():
        if not os.path.exists(path):
            print(f"错误: 找不到路径: {name} = {path}")
            all_paths_ok = False
            
    if all_paths_ok:
        # 1. 加载模型
        model, preprocess, tokenizer = setup_model(
            BIOMEDCLIP_PATH, 
            VPT_WEIGHTS_PATH, 
            NUM_PROMPTS, 
            EMBED_DIM, 
            ENABLE_PROMPT_SCALE, 
            DEVICE
        )
        
        # 2. 加载并计算“平均文本锚点” (a_global_anchor)
        all_training_texts = load_training_texts(
            TEXT_ROOT_PATH, 
            TEXT_DATASET_NAME, 
            LLM_MODEL, 
            DESCRIBE_NUMS
        )
        
        with torch.no_grad():
            text_tokens = tokenizer(all_training_texts).to(DEVICE)
            all_text_features = model.encode_text(text_tokens, normalize=True) # [N, 512]
            text_anchor = torch.mean(all_text_features, dim=0, keepdim=True) # [1, 512]
            text_anchor = F.normalize(text_anchor, dim=-1)
        
        # 3. 加载图像
        image_pil = Image.open(IMAGE_PATH).convert("RGB")

        # 4. 生成“域不变”热图
        print(f"正在生成 'invariant' 热图 (Text-Patch)...")
        heatmap_inv = generate_heatmap_text_patch(
            model, preprocess, tokenizer, image_pil, text_anchor, DEVICE
        )
        
        # 5. 可视化
        print("正在可视化热图...")
        visualize_heatmap(image_pil, heatmap_inv, len(all_training_texts))
        
        print("完成。")
    else:
        print("请检查脚本顶部的路径配置。")
