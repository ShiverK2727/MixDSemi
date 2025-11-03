import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import json
from biomedclip_vpt import build_dual_selective_prompt_image_encoder

# --- 1. 用户配置：请修改以下路径 ---
# (这些路径基于您之前的日志)

# 您的 BiomedCLIP 模型文件夹路径
BIOMEDCLIP_PATH = "/root/models/BiomedCLIP" 

# 您训练好的 VPT 权重文件路径
VPT_WEIGHTS_PATH = "/app/MixDSemi/SynFoCLIP/model/prostate/train_MiDSS_NDC_MARK2_CLIP/vpt_basic_test_d1/vpt_final_weights.pth"

# 您想要分析的测试图像
IMAGE_PATH = "/app/MixDSemi/data/ProstateSlice/UCL/train/image/00_08.png"

# --- 2. 模型和VPT配置 (基于您的训练脚本) ---

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
    # 注意：此方案不使用 tokenizer，但我们仍然返回它以保持接口一致
    return model, preprocess, tokenizer


def generate_heatmap_pps(model, preprocess, image_pil, prompt_group, device):
    """
    为单个图像生成热图 (方案 1: Prompt-Patch Similarity, PPS)
    此方法不使用任何文本提示，而是直接比较VPTs和Patch Tokens。
    
    Args:
        model (DualSelectivePromptBiomedCLIP): 已加载VPT权重的模型
        preprocess (callable): 图像预处理函数
        image_pil (PIL.Image): 输入的PIL图像
        prompt_group (str): "invariant" 或 "specific"
        device (str): "cuda" 或 "cpu"

    Returns:
        np.array: 归一化后的热图 (与原图大小相同)
    """
    
    # --- 步骤 1: 准备图像 ---
    
    # 预处理图像 (resize, center_crop, normalize)
    processed_image = preprocess(image_pil).unsqueeze(0).to(device) # [1, 3, 224, 224]

    # --- 步骤 2: 获取VPT调节的Patch特征 和 Prompt特征 ---
    
    # 我们调用 model.forward 来获取 *所有* tokens
    with torch.no_grad():
        outputs = model.forward(
            processed_image, 
            prompt_group=prompt_group, 
            return_tokens=True, 
            normalize=False
        )

    # all_tokens 形状: [B, 1 + num_prompts + num_patches, 768]
    all_tokens = outputs["tokens"] 
    B = all_tokens.shape[0] # Batch size (应为 1)
    
    # --- 方案 1 (PPS) 逻辑 ---
    # 提取 Prompt Tokens 和 Patch Tokens
    P = model.prompt_config.num_prompts         # Prompt 数量 (e.g., 4)
    
    # prompt_tokens 形状: [B, P, 768] (e.g., [1, 4, 768])
    prompt_tokens = all_tokens[:, 1 : 1 + P, :]
    
    # patch_tokens 形状: [B, N, 768] (e.g., [1, 196, 768])
    patch_tokens = all_tokens[:, 1 + P :, :]
    
    # --- 步骤 3: 计算 Prompt-Patch 相似度 ---
    
    # 归一化
    prompt_norm = F.normalize(prompt_tokens, dim=-1)
    patch_norm = F.normalize(patch_tokens, dim=-1)
    
    # 计算所有 prompts vs 所有 patches 的相似度
    # b=batch, p=prompts, n=patches, d=dimension
    # pp_sim 形状: [B, P, N] (e.g., [1, 4, 196])
    pp_sim = torch.einsum("bpd,bnd->bpn", prompt_norm, patch_norm)

    # 聚合 P 个 prompt 的相似度
    # 我们使用 .max() 来查找与 *任何* 一个 prompt 最相似的 patch
    # heatmap_scores 形状: [B, N] (e.g., [1, 196])
    heatmap_scores = pp_sim.max(dim=1).values 
    
    similarity_map = heatmap_scores.squeeze(0) # 形状: [N] (e.g., [196])

    # --- 步骤 4: Reshape 和上采样 ---
    
    # 获取 patch 网格的边长 (e.g., sqrt(196) = 14)
    num_patches = similarity_map.shape[0]
    # 增加一个 epsilon 防止浮点数精度问题
    num_patches_side = int(math.sqrt(num_patches) + 1e-9)
    
    if num_patches_side * num_patches_side != num_patches:
        raise ValueError(f"Patch 数量 ({num_patches}) 不是一个完美的平方数，无法 reshape。")
        
    # Reshape 为 2D 热图: [196] -> [1, 1, 14, 14]
    heatmap_low_res = similarity_map.reshape(1, 1, num_patches_side, num_patches_side)
    
    # 使用双线性插值上采样到原始图像的分辨率
    heatmap_high_res = F.interpolate(
        heatmap_low_res,
        size=(image_pil.height, image_pil.width), # 恢复到原始 PIL 图像大小
        mode='bilinear',
        align_corners=False
    )
    
    heatmap_numpy = heatmap_high_res.squeeze().cpu().numpy() # [H, W]
    
    # 归一化到 [0, 1] 以便可视化
    min_val = heatmap_numpy.min()
    max_val = heatmap_numpy.max()
    heatmap_normalized = (heatmap_numpy - min_val) / (max_val - min_val + 1e-8)
    
    return heatmap_normalized


def visualize_heatmaps(original_image, heatmap_inv, heatmap_spec):
    """
    使用 Matplotlib 并排显示两个热图。
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 原始图像
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # 域不变热图
    axes[1].imshow(original_image, cmap='gray')
    axes[1].imshow(heatmap_inv, cmap='jet', alpha=0.5)
    axes[1].set_title("Domain Invariant Heatmap (PPS)")
    axes[1].axis('off')
    
    # 域特定热图
    axes[2].imshow(original_image, cmap='gray')
    axes[2].imshow(heatmap_spec, cmap='jet', alpha=0.5)
    axes[2].set_title("Domain Specific Heatmap (PPS)")
    axes[2].axis('off')
    
    fig.suptitle(f"Heatmap based on Prompt-Patch Similarity (Scheme 1)", fontsize=16)
    plt.tight_layout()
    
    # 保存或显示
    save_path = "vpt_heatmap_comparison_pps.png"
    plt.savefig(save_path)
    print(f"热图已保存到: {save_path}")
    # plt.show() # 如果在 Jupyter 或类似环境中，取消此行注释

if __name__ == "__main__":
    # 检查所有路径
    paths_to_check = {
        "BIOMEDCLIP_PATH": BIOMEDCLIP_PATH,
        "VPT_WEIGHTS_PATH": VPT_WEIGHTS_PATH,
        "IMAGE_PATH": IMAGE_PATH,
    }
    
    all_paths_ok = True
    for name, path in paths_to_check.items():
        if not os.path.exists(path):
            print(f"错误: 找不到路径: {name} = {path}")
            all_paths_ok = False
            
    if all_paths_ok:
        # 1. 加载模型
        model, preprocess, _ = setup_model(
            BIOMEDCLIP_PATH, 
            VPT_WEIGHTS_PATH, 
            NUM_PROMPTS, 
            EMBED_DIM, 
            ENABLE_PROMPT_SCALE, 
            DEVICE
        )
        
        # 2. 加载图像 (必须转为 RGB)
        image_pil = Image.open(IMAGE_PATH).convert("RGB")

        # 3. 生成域不变热图
        print(f"正在生成 'invariant' 热图 (PPS)...")
        heatmap_inv = generate_heatmap_pps(
            model, preprocess, image_pil, "invariant", DEVICE
        )
        
        # 4. 生成域特定热图
        print(f"正在生成 'specific' 热图 (PPS)...")
        heatmap_spec = generate_heatmap_pps(
            model, preprocess, image_pil, "specific", DEVICE
        )
        
        # 5. 可视化对比
        print("正在可视化热图...")
        visualize_heatmaps(image_pil, heatmap_inv, heatmap_spec)
        
        print("完成。")
    else:
        print("请检查脚本顶部的路径配置。")
