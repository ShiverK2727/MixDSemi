import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from biomedclip_vpt import build_dual_selective_prompt_image_encoder

# --- 1. 用户配置：请修改以下路径 ---
# (这些路径基于您之前的日志)

# 您的 BiomedCLIP 模型文件夹路径
BIOMEDCLIP_PATH = "/root/models/BiomedCLIP" 

# 您训练好的 VPT 权重文件路径
VPT_WEIGHTS_PATH = "/app/MixDSemi/SynFoCLIP/model/prostate/train_MiDSS_NDC_MARK2_CLIP/vpt_basic_test_d1/vpt_final_weights.pth"

# 您想要分析的测试图像
IMAGE_PATH = "/app/MixDSemi/data/ProstateSlice/BIDMC/train/image/00_08.png"

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
    return model, preprocess, tokenizer


def generate_heatmap_grad_cam(model, preprocess, image_pil, prompt_group, device):
    """
    为单个图像生成热图 (Grad-CAM 方案)
    此方法计算最终 "image_features" 对 "patch_tokens" 的梯度归因。
    
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

    # --- 步骤 2: 前向传播，但需要保留梯度和中间特征 ---
    
    # 启用梯度计算
    processed_image.requires_grad = True 
    
    # 我们需要手动执行 model.forward 的一部分，以便获取 *head之前* 的 patch tokens
    
    # 1. 运行到 ViT trunk 的末尾
    #    (注意: model.forward 已被封装, 我们需要调用内部的 encode_image_with_prompts)
    #    encode_image_with_prompts 返回 (image_features, all_tokens)
    #    image_features 是 [B, 512] (来自 head)
    #    all_tokens 是 [B, 1+P+N, 768] (来自 trunk)
    
    # 我们需要 'all_tokens' 允许反向传播
    outputs = model.encode_image_with_prompts(
        processed_image, 
        prompt_group=prompt_group,
        pooling="cls", # 确保使用 [CLS] token
        normalize=True # 与训练时一致
    )
    
    image_features = outputs[0] # [B, 512]
    all_tokens = outputs[1]     # [B, 1+P+N, 768]
    
    # 2. 提取我们感兴趣的两个张量
    # 归因目标：最终的全局特征
    target_features = image_features # [B, 512]
    
    # 归因对象：Head 之前的 Patch Tokens
    P = model.prompt_config.num_prompts
    patch_tokens = all_tokens[:, 1 + P :, :] # [B, N, 768]
    
    # --- 步骤 3: Grad-CAM 核心逻辑 ---
    
    # 3.1 定义归因分数 (score)
    # 我们可以简单地对特征向量求和作为分数
    # 或者使用L2范数，或者选择一个特定的维度
    score = torch.sum(target_features)
    
    # 3.2 清零梯度
    model.zero_grad()
    
    # 3.3 反向传播：计算 score 对 all_tokens 的梯度
    # 修正：我们必须在 'all_tokens' (父张量) 上保留梯度，而不是在 'patch_tokens' (切片) 上
    all_tokens.retain_grad()
    score.backward(retain_graph=True) # retain_graph 可能需要，取决于后续操作
    
    # 3.4 获取梯度
    all_grads = all_tokens.grad # [B, 1+P+N, 768]
    
    if all_grads is None:
        raise RuntimeError("无法获取 all_tokens 的梯度。")
        
    # 修正：从 'all_grads' 中切片出 patch 对应的梯度
    grads = all_grads[:, 1 + P :, :] # [B, N, 768]
        
    # 3.5 计算通道权重 (Alpha_k)
    # (B, N, C) -> (B, 1, C)
    weights = torch.mean(grads, dim=1, keepdim=True) 
    
    # 3.6 计算加权激活
    # (B, N, C) * (B, 1, C) -> (B, N, C)
    weighted_activations = patch_tokens * weights
    
    # (B, N, C) -> (B, N)
    cam = torch.sum(weighted_activations, dim=2)
    
    # 3.7 ReLU (只保留正贡献)
    cam = F.relu(cam) # [B, N]
    
    similarity_map = cam.squeeze(0).detach() # [N]

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
    
    # 归一化
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
    axes[1].set_title("Domain Invariant Heatmap (Grad-CAM)")
    axes[1].axis('off')
    
    axes[2].imshow(original_image, cmap='gray')
    axes[2].imshow(heatmap_spec, cmap='jet', alpha=0.5)
    axes[2].set_title("Domain Specific Heatmap (Grad-CAM)")
    axes[2].axis('off')
    
    fig.suptitle(f"Heatmap based on Gradient Attribution (Grad-CAM)", fontsize=16)
    plt.tight_layout()
    
    save_path = "vpt_heatmap_comparison_grad_cam.png"
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

        print(f"正在生成 'invariant' 热图 (Grad-CAM)...")
        # Grad-CAM 需要在 no_grad 之外运行
        heatmap_inv = generate_heatmap_grad_cam(
            model, preprocess, image_pil, "invariant", DEVICE
        )
        
        print(f"正在生成 'specific' 热图 (Grad-CAM)...")
        heatmap_spec = generate_heatmap_grad_cam(
            model, preprocess, image_pil, "specific", DEVICE
        )
        
        print("正在可视化热图...")
        visualize_heatmaps(image_pil, heatmap_inv, heatmap_spec)
        
        print("完成。")
    else:
        print("请检查脚本顶部的路径配置。")

