import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import json # 导入 json
from biomedclip_vpt import build_dual_selective_prompt_image_encoder

# --- 1. 用户配置：请修改以下路径 ---

# 您的 BiomedCLIP 模型文件夹路径
BIOMEDCLIP_PATH = "/root/models/BiomedCLIP" 

# 您训练好的 VPT 权重文件路径
VPT_WEIGHTS_PATH = "/app/MixDSemi/SynFoCLIP/model/prostate/train_MiDSS_NDC_MARK2_CLIP/vpt_basic_test_d1/vpt_final_weights.pth"

# 您想要分析的测试图像
IMAGE_PATH = "/app/MixDSemi/data/ProstateSlice/BIDMC/train/image/00_08.png"

# --- 2. 文本锚点配置 (基于您的训练脚本) ---

# 对应 args.text_root
TEXT_ROOT_PATH = "/app/MixDSemi/SynFoCLIP/code/text"
# 对应 TEXT_DATASET_DEFAULTS (prostate -> ProstateSlice)
TEXT_DATASET_NAME = "ProstateSlice"
# 对应 args.llm_model
LLM_MODEL = "gemini"
# 对应 args.describe_nums
DESCRIBE_NUMS = 40

# --- 3. 模型和VPT配置 (基于您的训练脚本) ---

# 对应 args.biomedclip_num_prompts
NUM_PROMPTS = 4 
# 对应 args.biomedclip_embed_dim
EMBED_DIM = 768 
# 对应 args.biomedclip_disable_scale
ENABLE_PROMPT_SCALE = not True 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------

def setup_model(biomedclip_path, vpt_weights_path, num_prompts, embed_dim, enable_scale, device):
    """
    加载 BiomedCLIP, 注入 VPTs, 并加载训练好的VPT权重。
    """
    print(f"正在从 {biomedclip_path} 加载 BiomedCLIP...")
    model, preprocess, tokenizer = build_dual_selective_prompt_image_encoder(
        model_path=biomedclip_path,
        device=device,
        num_prompts=num_prompts,
        embed_dim=embed_dim,
        enable_prompt_scale=enable_scale,
        freeze_backbone=True,
    )
    
    print(f"正在从 {vpt_weights_path} 加载 VPT 权重...")
    model.load_prompts(vpt_weights_path, map_location=device)
    model.eval()
    print("模型准备就绪 (eval mode)。")
    return model, preprocess, tokenizer

def load_training_texts(text_root, dataset_name, llm_model, describe_nums):
    """
    (已修正) 加载训练时使用的所有文本描述。
    这模拟了 train_MiDSS_NDC_MARK2_CLIP.py 中 TextSampler.load_texts 的逻辑，
    以获取用于计算 a_global_anchor 的 'flat_list'。
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
    # 遍历 JSON 中的所有类别 (e.g., "base", "background", ...)
    for type_key, texts in all_texts_by_type.items():
        # 选取前 describe_nums 个文本
        flat_list.extend(texts[:describe_nums]) 

    if len(flat_list) == 0:
        raise ValueError("未能从 JSON 文件中加载任何文本。")
        
    print(f"成功加载 {len(flat_list)} 条训练文本描述。")
    return flat_list


def generate_heatmap_text_patch(model, preprocess, tokenizer, image_pil, text_anchor, prompt_group, device):
    """
    为单个图像和“平均文本锚点”生成热图。
    
    Args:
        text_anchor (torch.Tensor): [1, 512] 维的平均文本特征
    """
    
    # --- 步骤 1: 准备图像 ---
    processed_image = preprocess(image_pil).unsqueeze(0).to(device) # [1, 3, 224, 224]

    # --- 步骤 2: 获取VPT调节的Patch特征 ---
    # 这是“VPTs作为辅助”的核心：
    # 我们使用 'prompt_group' (e.g., "invariant") 来获取被调节过的 patch tokens
    
    with torch.no_grad():
        outputs = model.forward(
            processed_image, 
            prompt_group=prompt_group, 
            return_tokens=True, 
            normalize=False # 我们在投影后手动归一化
        )

    all_tokens = outputs["tokens"] 
    num_prompts = model.prompt_config.num_prompts
    patch_tokens = all_tokens[:, 1 + num_prompts :, :] # [1, N, 768]

    # --- 步骤 3: 特征对齐与相似度计算 ---
    
    head = model.model.visual.head
    projected_patch_tokens = head(patch_tokens) # [1, N, 512]
    
    # 归一化 patch 特征
    projected_patch_tokens = F.normalize(projected_patch_tokens, dim=-1)
    
    # 计算相似度 (Cosine Similarity)
    # text_anchor 形状: [1, 512]
    # projected_patch_tokens 形状: [1, N, 512]
    
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
    
    # 归一化到 [0, 1]
    min_val = heatmap_numpy.min()
    max_val = heatmap_numpy.max()
    heatmap_normalized = (heatmap_numpy - min_val) / (max_val - min_val + 1e-8)
    
    return heatmap_normalized


def visualize_heatmaps(original_image, heatmap_inv, heatmap_spec, text_count):
    """
    使用 Matplotlib 并排显示两个热图。
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(original_image, cmap='gray')
    axes[1].imshow(heatmap_inv, cmap='jet', alpha=0.5)
    axes[1].set_title("Domain Invariant Heatmap (Text-Patch)")
    axes[1].axis('off')
    
    axes[2].imshow(original_image, cmap='gray')
    axes[2].imshow(heatmap_spec, cmap='jet', alpha=0.5)
    axes[2].set_title("Domain Specific Heatmap (Text-Patch)")
    axes[2].axis('off')
    
    fig.suptitle(f"Heatmap based on Average of {text_count} Training Texts", fontsize=16)
    plt.tight_layout()
    
    save_path = "vpt_heatmap_comparison_text_patch.png"
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
            # 计算平均特征，这对应您训练中的 a_global_anchor
            text_anchor = torch.mean(all_text_features, dim=0, keepdim=True) # [1, 512]
            text_anchor = F.normalize(text_anchor, dim=-1)
        
        # 3. 加载图像
        image_pil = Image.open(IMAGE_PATH).convert("RGB")

        # 4. 生成“域不变”热图 (这就是您的ROI引导)
        print(f"正在生成 'invariant' 热图 (Text-Patch)...")
        heatmap_inv = generate_heatmap_text_patch(
            model, preprocess, tokenizer, image_pil, text_anchor, "invariant", DEVICE
        )
        
        # 5. (可选) 生成“域特定”热图以供对比
        print(f"正在生成 'specific' 热图 (Text-Patch)...")
        heatmap_spec = generate_heatmap_text_patch(
            model, preprocess, tokenizer, image_pil, text_anchor, "specific", DEVICE
        )
        
        # 6. 可视化对比
        print("正在可视化热图...")
        visualize_heatmaps(image_pil, heatmap_inv, heatmap_spec, len(all_training_texts))
        
        print("完成。")
    else:
        print("请检查脚本顶部的路径配置。")
