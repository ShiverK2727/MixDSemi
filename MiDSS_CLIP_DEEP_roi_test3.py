import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import json
# 导入 VPT-Deep 版本
from SynFoCLIP.code.biomedclip_vpt_invariant_only import build_dual_selective_prompt_image_encoder

# --- 1. 用户配置：请修改以下路径 ---
BIOMEDCLIP_PATH = "/root/models/BiomedCLIP" 
# 确保这个路径指向您 *新* 训练的 VPT-Deep 权重
VPT_WEIGHTS_PATH = "/app/MixDSemi/SynFoCLIP/model/prostate/train_MiDSS_NDC_MARK2_CLIP_DEEP/vpt_basic_test_d1/vpt_final_weights.pth"
IMAGE_PATH = "/app/MixDSemi/data/ProstateSlice/UCL/train/image/00_08.png"

# --- 2. 模型和VPT配置 ---
NUM_PROMPTS_PER_LAYER = 4 
EMBED_DIM = 768 
ENABLE_PROMPT_SCALE = not True 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------

def setup_model(biomedclip_path, vpt_weights_path, num_prompts, embed_dim, enable_scale, device):
    # (此函数与方案一相同)
    print(f"正在从 {biomedclip_path} 加载 BiomedCLIP (VPT-Deep)...")
    model, preprocess, tokenizer = build_dual_selective_prompt_image_encoder(
        model_path=biomedclip_path,
        device=device,
        num_prompts=num_prompts,
        embed_dim=embed_dim,
        enable_prompt_scale=enable_scale,
        freeze_backbone=True,
    )
    print(f"正在从 {vpt_weights_path} 加载 VPT-Deep 权重...")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        model.load_prompts(vpt_weights_path, map_location=device)
    model.eval()
    print("模型准备就绪 (eval mode)。")
    return model, preprocess, tokenizer

def get_patch_tokens(model, preprocess, image_pil, prompt_group):
    """
    辅助函数：获取指定 prompt_group 的 patch tokens
    """
    device = next(model.parameters()).device
    processed_image = preprocess(image_pil).unsqueeze(0).to(device) 

    with torch.no_grad():
        outputs = model.forward(
            processed_image, 
            prompt_group=prompt_group, 
            return_tokens=True, 
            normalize=False 
        )
    
    all_tokens = outputs["tokens"] 
    patch_tokens = all_tokens[:, 1:, :] # [1, N, 768]
    return patch_tokens

def generate_heatmap_difference(patch_tokens_inv, patch_tokens_spec, image_pil):
    """
    计算 invariant 和 specific patch tokens 之间的余弦相似度图
    """
    
    # 归一化
    patch_inv_norm = F.normalize(patch_tokens_inv, dim=-1)
    patch_spec_norm = F.normalize(patch_tokens_spec, dim=-1)
    
    # 计算相似度: [1, N]
    # (我们计算相似度，因为我们想找“共识”区域)
    similarity = F.cosine_similarity(patch_inv_norm, patch_spec_norm, dim=-1)
    heatmap_scores = similarity.squeeze(0).detach() # [N]

    # --- Reshape 和上采样 ---
    num_patches = heatmap_scores.shape[0]
    num_patches_side = int(math.sqrt(num_patches) + 1e-9)
    
    heatmap_low_res = heatmap_scores.reshape(1, 1, num_patches_side, num_patches_side)
    
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
    
    # 返回相似度图 (高分 = 共识) 和差异图 (高分 = 差异)
    return heatmap_normalized, 1.0 - heatmap_normalized


def visualize_heatmaps(original_image, heatmap_similarity, heatmap_difference):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(original_image, cmap='gray')
    axes[1].imshow(heatmap_similarity, cmap='jet', alpha=0.5)
    axes[1].set_title("Consensus Heatmap (Inv-Spec Similarity)")
    axes[1].axis('off')
    
    axes[2].imshow(original_image, cmap='gray')
    axes[2].imshow(heatmap_difference, cmap='jet', alpha=0.5)
    axes[2].set_title("Difference Heatmap (1 - Similarity)")
    axes[2].axis('off')
    
    fig.suptitle("VPT-Deep Invariant vs Specific Feature Map", fontsize=16)
    plt.tight_layout()
    
    save_path = "vpt_heatmap_deep_difference.png"
    plt.savefig(save_path)
    print(f"热图已保存到: {save_path}")

if __name__ == "__main__":
    # 1. 加载模型
    model, preprocess, tokenizer = setup_model(
        BIOMEDCLIP_PATH, 
        VPT_WEIGHTS_PATH, 
        NUM_PROMPTS_PER_LAYER, 
        EMBED_DIM, 
        ENABLE_PROMPT_SCALE, 
        DEVICE
    )
    
    # 2. 加载目标图像
    image_pil = Image.open(IMAGE_PATH).convert("RGB")
    
    # 3. 获取两组 patch tokens
    print("正在生成 'invariant' patch tokens...")
    patch_tokens_inv = get_patch_tokens(model, preprocess, image_pil, "invariant")
    
    print("正在生成 'specific' patch tokens...")
    patch_tokens_spec = get_patch_tokens(model, preprocess, image_pil, "specific")
    
    # 4. 生成热图
    print("正在计算差异图...")
    heatmap_sim, heatmap_diff = generate_heatmap_difference(
        patch_tokens_inv, patch_tokens_spec, image_pil
    )
    
    # 5. 可视化
    print("正在可视化热图...")
    visualize_heatmaps(image_pil, heatmap_sim, heatmap_diff)
    
    print("完成。")
