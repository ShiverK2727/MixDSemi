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
IMAGE_PATH = "/app/MixDSemi/data/ProstateSlice/BIDMC/train/image/00_08.png"

# --- 2. 校准配置 (Calibration) ---
# 您需要提供一个用于校准的 图像-掩码 对
# (这里我们使用与测试图像相同的路径，仅为演示)
CALIBRATION_IMAGE_PATH = "/app/MixDSemi/data/ProstateSlice/BIDMC/train/image/00_08.png"
# (请确保掩码路径正确)
CALIBRATION_MASK_PATH = "/app/MixDSemi/data/ProstateSlice/BIDMC/train/mask/00_08.png"
# 我们将选择与掩码最相关的 Top K 个 prompts
TOP_K_PROMPTS = 5 

# --- 3. 模型和VPT配置 ---
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

def get_patch_tokens_and_all_prompts(model, preprocess, image_pil, prompt_group):
    """
    辅助函数：获取最终的 patch tokens 和 *所有* 层的 prompts
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
    
    # 1. 获取 Patch tokens (VPT-Deep 输出)
    all_tokens = outputs["tokens"] 
    patch_tokens = all_tokens[:, 1:, :] # [1, N, 768]
    
    # 2. 获取所有层的 Prompts
    if prompt_group == "invariant":
        all_prompts_tensor = model.prompt_learner.domain_invariant_prompts
    else:
        all_prompts_tensor = model.prompt_learner.domain_specific_prompts
    
    # [L, P, D] -> [L*P, D] (e.g., [48, 768])
    all_prompts = all_prompts_tensor.detach().to(device).reshape(-1, EMBED_DIM)
    
    return patch_tokens, all_prompts

def calibrate_prompts(model, preprocess, cal_image_pil, cal_mask_pil, prompt_group, top_k, device):
    """
    使用 图像-掩码 对来校准所有 (L*P) 个 prompts，找出 Top-K
    """
    print(f"正在校准 {prompt_group} prompts...")
    patch_tokens, all_prompts = get_patch_tokens_and_all_prompts(
        model, preprocess, cal_image_pil, prompt_group
    )
    
    # patch_tokens: [1, N, D], all_prompts: [L*P, D]
    num_total_prompts = all_prompts.shape[0]
    num_patches = patch_tokens.shape[1]
    num_patches_side = int(math.sqrt(num_patches) + 1e-9)
    
    # 归一化
    patch_tokens_norm = F.normalize(patch_tokens, dim=-1)
    all_prompts_norm = F.normalize(all_prompts, dim=-1)
    
    # 计算相似度矩阵: [1, N, D] @ [D, L*P] -> [1, N, L*P]
    sim_matrix = torch.einsum('bnd,pd->bnp', patch_tokens_norm, all_prompts_norm)
    sim_matrix = sim_matrix.squeeze(0) # [N, L*P]
    
    # 准备下采样掩码
    cal_mask_pil = cal_mask_pil.convert('L') # 转为灰度
    cal_mask = torch.from_numpy(np.array(cal_mask_pil)).float() / 255.0
    cal_mask = cal_mask.to(device)
    
    # 下采样到 [1, 1, 14, 14]
    mask_low_res = F.interpolate(
        cal_mask.unsqueeze(0).unsqueeze(0),
        size=(num_patches_side, num_patches_side),
        mode='bilinear',
        align_corners=False
    )
    mask_flat = (mask_low_res.squeeze() > 0.5).float().view(-1) # [N]
    
    # 计算每个 prompt (共 L*P 个) 与 前景/背景 的关联分数
    # sim_matrix: [N, L*P]
    # mask_flat: [N]
    
    # (N, L*P).T @ (N,) -> (L*P,)
    score_fg = torch.einsum('np,n->p', sim_matrix, mask_flat)
    score_bg = torch.einsum('np,n->p', sim_matrix, (1.0 - mask_flat))
    
    # 我们要找的是与前景相似度高、与背景相似度低的 prompts
    prompt_scores = score_fg - score_bg
    
    # 获取 Top K
    top_k_indices = torch.topk(prompt_scores, k=top_k).indices
    
    print(f"校准完成。Top {top_k} {prompt_group} prompts 的索引: {top_k_indices.cpu().numpy()}")
    return top_k_indices

def generate_heatmap_mask_guided(model, preprocess, image_pil, all_prompts, top_k_indices, prompt_group):
    """
    使用校准后的 Top-K prompts 生成热图
    """
    patch_tokens, _ = get_patch_tokens_and_all_prompts(
        model, preprocess, image_pil, prompt_group
    )
    
    # 归一化
    patch_tokens_norm = F.normalize(patch_tokens, dim=-1)
    all_prompts_norm = F.normalize(all_prompts, dim=-1)
    
    # 计算相似度矩阵: [1, N, L*P]
    sim_matrix = torch.einsum('bnd,pd->bnp', patch_tokens_norm, all_prompts_norm)
    sim_matrix = sim_matrix.squeeze(0) # [N, L*P]
    
    # *** 关键步骤: 只选择 Top-K prompts ***
    top_k_sim = sim_matrix[:, top_k_indices] # [N, K]
    
    # 在 Top-K prompts 中取最大值
    heatmap_scores = top_k_sim.max(dim=1).values # [N]
    
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
    
    heatmap_numpy = heatmap_high_res.squeeze().cpu().detach().numpy()
    
    min_val = heatmap_numpy.min()
    max_val = heatmap_numpy.max()
    heatmap_normalized = (heatmap_numpy - min_val) / (max_val - min_val + 1e-8)
    
    return heatmap_normalized

def visualize_heatmaps(original_image, heatmap_inv, heatmap_spec, top_k):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(original_image, cmap='gray')
    axes[1].imshow(heatmap_inv, cmap='jet', alpha=0.5)
    axes[1].set_title(f"Invariant Heatmap (Top-{top_k} Deep-PPS)")
    axes[1].axis('off')
    
    axes[2].imshow(original_image, cmap='gray')
    axes[2].imshow(heatmap_spec, cmap='jet', alpha=0.5)
    axes[2].set_title(f"Specific Heatmap (Top-{top_k} Deep-PPS)")
    axes[2].axis('off')
    
    fig.suptitle(f"Heatmap based on Mask-Guided Top-{top_k} Prompts (VPT-Deep)", fontsize=16)
    plt.tight_layout()
    
    save_path = f"vpt_heatmap_deep_mask_guided_top{top_k}.png"
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
    
    # 2. 加载校准数据
    cal_image_pil = Image.open(CALIBRATION_IMAGE_PATH).convert("RGB")
    cal_mask_pil = Image.open(CALIBRATION_MASK_PATH)
    
    # 3. 执行校准
    top_k_indices_inv = calibrate_prompts(
        model, preprocess, cal_image_pil, cal_mask_pil, "invariant", TOP_K_PROMPTS, DEVICE
    )
    top_k_indices_spec = calibrate_prompts(
        model, preprocess, cal_image_pil, cal_mask_pil, "specific", TOP_K_PROMPTS, DEVICE
    )
    
    # 4. 加载目标图像
    image_pil = Image.open(IMAGE_PATH).convert("RGB")
    
    # 5. 获取所有 prompts (只需要一次)
    _ , all_prompts_inv = get_patch_tokens_and_all_prompts(
        model, preprocess, image_pil, "invariant"
    )
    _ , all_prompts_spec = get_patch_tokens_and_all_prompts(
        model, preprocess, image_pil, "specific"
    )

    # 6. 生成热图
    print(f"正在生成 'invariant' (Mask-Guided) 热图...")
    heatmap_inv = generate_heatmap_mask_guided(
        model, preprocess, image_pil, all_prompts_inv, top_k_indices_inv, "invariant"
    )
    
    print(f"正在生成 'specific' (Mask-Guided) 热图...")
    heatmap_spec = generate_heatmap_mask_guided(
        model, preprocess, image_pil, all_prompts_spec, top_k_indices_spec, "specific"
    )
    
    # 7. 可视化
    print("正在可视化热图...")
    visualize_heatmaps(image_pil, heatmap_inv, heatmap_spec, TOP_K_PROMPTS)
    
    print("完成。")
