import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from biomedclip_vpt import build_dual_selective_prompt_image_encoder
import torchvision.transforms as T # 导入 torchvision

# --- 1. 用户配置：请修改以下路径 ---
# (这些路径基于您之前的日志)

# 您的 BiomedCLIP 模型文件夹路径
BIOMEDCLIP_PATH = "/root/models/BiomedCLIP" 

# 您训练好的 VPT 权重文件路径
VPT_WEIGHTS_PATH = "/app/MixDSemi/SynFoCLIP/model/prostate/train_MiDSS_NDC_MARK2_CLIP/vpt_basic_test_d1/vpt_final_weights.pth"

# 您想要分析的测试图像
# IMAGE_PATH = "/app/MixDSemi/data/ProstateSlice/BIDMC/train/image/00_08.png"
IMAGE_PATH = "/app/MixDSemi/data/ProstateSlice/UCL/train/image/00_08.png"

# --- 2. 模型和VPT配置 (基于您的训练脚本) ---

# 对应 args.biomedclip_num_prompts
NUM_PROMPTS = 4 
# 对应 args.biomedclip_embed_dim
EMBED_DIM = 768 
# 对应 args.biomedclip_disable_scale
ENABLE_PROMPT_SCALE = not True 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 3. 强增强定义 (基于 train_MiDSS_NDC_MARK2_CLIP.py) ---

# 从您的训练脚本 (prostate) 推断的参数:
# min_v = 0.1, max_v = 2
# patch_size = 384 (我们将使用CLIP的224)
# kernel_size = int(0.1 * patch_size) -> 约 38 (我们将使用 23, 奇数)
# num_channels = 1 (但CLIP需要3)

# 我们使用 torchvision 模拟您的 'strong' 变换
# 这些变换在 'preprocess' (Resize/Crop/Normalize) 之前应用
strong_aug = T.Compose([
    T.ColorJitter(brightness=(0.1, 2.0), contrast=(0.1, 2.0)),
    # 使用一个与 patch_size 10% 相当的模糊核 (224 * 0.1 ≈ 23)
    T.GaussianBlur(kernel_size=23, sigma=(0.1, 3.0)) 
])

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


def generate_heatmap_sw_cam(model, preprocess, strong_aug_transform, image_pil, prompt_group, device):
    """
    为单个图像生成热图 (SW-CAM 方案)
    此方法计算弱增强(原图)和强增强图像的 patch 特征差异。
    
    Args:
        model (DualSelectivePromptBiomedCLIP): 已加载VPT权重的模型
        preprocess (callable): 图像预处理函数
        strong_aug_transform (callable): 强增强变换
        image_pil (PIL.Image): 输入的PIL图像
        prompt_group (str): "invariant" 或 "specific"
        device (str): "cuda" 或 "cpu"

    Returns:
        np.array: 归一化后的热图 (与原图大小相同)
    """
    
    # --- 步骤 1: 准备弱/强增强图像 ---
    
    # 弱增强 = 原始图像
    img_weak_pil = image_pil
    # 强增强 = 应用变换
    img_strong_pil = strong_aug_transform(image_pil)
    
    # 分别进行预处理 (Resize, Crop, Normalize)
    img_weak_tensor = preprocess(img_weak_pil).unsqueeze(0).to(device) # [1, 3, 224, 224]
    img_strong_tensor = preprocess(img_strong_pil).unsqueeze(0).to(device) # [1, 3, 224, 224]

    # --- 步骤 2: 分别获取VPT调节的Patch特征 ---
    
    with torch.no_grad():
        # 1. 弱增强图像
        outputs_weak = model.encode_image_with_prompts(
            img_weak_tensor, 
            prompt_group=prompt_group,
            pooling="cls",
            normalize=True # 与训练时一致
        )
        
        # 2. 强增强图像
        outputs_strong = model.encode_image_with_prompts(
            img_strong_tensor, 
            prompt_group=prompt_group,
            pooling="cls",
            normalize=True # 与训练时一致
        )

    P = model.prompt_config.num_prompts
    
    # 弱增强的 Patch Tokens [B, N, C]
    patch_tokens_weak = outputs_weak[1][:, 1 + P :, :]
    
    # 强增强的 Patch Tokens [B, N, C]
    patch_tokens_strong = outputs_strong[1][:, 1 + P :, :]
    
    
    # --- 步骤 3: SW-CAM 核心逻辑 ---
    
    # 3.1 归一化特征 (在余弦相似度计算前)
    patch_weak_n = F.normalize(patch_tokens_weak, dim=-1)
    patch_strong_n = F.normalize(patch_tokens_strong, dim=-1)
    
    # 3.2 计算特征相似度 (Cosine Similarity)
    # (B, N, C) * (B, N, C) -> (B, N)
    # 逐元素相乘后在 C 维度求和
    similarity = (patch_weak_n * patch_strong_n).sum(dim=-1) 
    
    # 3.3 计算距离 (0=相同, 2=完全相反)
    # distance = 1.0 - similarity 
    # 我们直接用 similarity，因为高相似度 = ROI
    
    similarity_map = similarity.squeeze(0).detach() # [N]
    
    # 此时，高分 (接近1.0) = 稳定 = ROI
    # 低分 (<< 1.0) = 不稳定 = 背景
    
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


def visualize_heatmaps(original_image, heatmap_inv, heatmap_spec):
    """
    使用 Matplotlib 并排显示两个热图。
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(original_image, cmap='gray')
    axes[1].imshow(heatmap_inv, cmap='jet', alpha=0.5)
    axes[1].set_title("Domain Invariant Heatmap (SW-CAM)")
    axes[1].axis('off')
    
    axes[2].imshow(original_image, cmap='gray')
    axes[2].imshow(heatmap_spec, cmap='jet', alpha=0.5)
    axes[2].set_title("Domain Specific Heatmap (SW-CAM)")
    axes[2].axis('off')
    
    fig.suptitle(f"Heatmap based on Strong-Weak Consistency (SW-CAM)", fontsize=16)
    plt.tight_layout()
    
    save_path = "vpt_heatmap_comparison_sw_cam.png"
    plt.savefig(save_path)
    print(f"热图已保存到: {save_path}")

if __name__ == "__main__":
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
        model, preprocess, _ = setup_model(
            BIOMEDCLIP_PATH, 
            VPT_WEIGHTS_PATH, 
            NUM_PROMPTS, 
            EMBED_DIM, 
            ENABLE_PROMPT_SCALE, 
            DEVICE
        )
        
        image_pil = Image.open(IMAGE_PATH).convert("RGB")

        print(f"正在生成 'invariant' 热图 (SW-CAM)...")
        # SW-CAM 在 no_grad() 内部运行
        heatmap_inv = generate_heatmap_sw_cam(
            model, preprocess, strong_aug, image_pil, "invariant", DEVICE
        )
        
        print(f"正在生成 'specific' 热图 (SW-CAM)...")
        heatmap_spec = generate_heatmap_sw_cam(
            model, preprocess, strong_aug, image_pil, "specific", DEVICE
        )
        
        print("正在可视化热图...")
        visualize_heatmaps(image_pil, heatmap_inv, heatmap_spec)
        
        print("完成。")
    else:
        print("请检查脚本顶部的路径配置。")
