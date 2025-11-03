import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from biomedclip_vpt import build_dual_selective_prompt_image_encoder

# --- 1. 用户配置：请修改以下路径 ---

# 您的 BiomedCLIP 模型文件夹路径
# (在 train_MiDSS_NDC_MARK2_CLIP.py 中为 args.biomedclip_path)
BIOMEDCLIP_PATH = "/root/models/BiomedCLIP" 

# 您训练好的 VPT 权重文件路径
# (在 train_MiDSS_NDC_MARK2_CLIP.py 中保存在 snapshot_path / "vpt_final_weights.pth")
VPT_WEIGHTS_PATH = "/app/MixDSemi/SynFoCLIP/model/prostate/train_MiDSS_NDC_MARK2_CLIP/vpt_basic_test_d1/vpt_final_weights.pth"

# 您想要分析的测试图像
IMAGE_PATH = "/app/MixDSemi/data/ProstateSlice/BIDMC/train/image/00_08.png"

# 您的文本提示 (ROI 的描述)
TEXT_PROMPT = "prostate gland"

# --- 2. 模型和VPT配置 (基于您的训练脚本) ---

# 这些参数应与您的训练脚本 (train_MiDSS_NDC_MARK2_CLIP.py) 保持一致
# 来自 args.biomedclip_num_prompts
NUM_PROMPTS = 4 
# 来自 args.biomedclip_embed_dim
EMBED_DIM = 768 
# 来自 args.biomedclip_disable_scale
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
    # 注意: 您的日志中有一个 FutureWarning，建议在 torch.load 中设置 weights_only=True
    # 您可能需要修改 biomedclip_vpt.py 中的 load_prompts 函数来解决这个警告。
    # 例如: state = torch.load(path, map_location=map_location, weights_only=True)
    # (假设 config 字典可以安全加载或单独处理)
    # 由于这里无法修改 biomedclip_vpt.py，我们暂时忽略该警告。
    model.load_prompts(vpt_weights_path, map_location=device)
    model.eval()
    print("模型准备就绪 (eval mode)。")
    return model, preprocess, tokenizer


def generate_heatmap(model, preprocess, tokenizer, image_pil, text_prompt, prompt_group, device):
    """
    为单个图像和文本提示生成热图。
    
    Args:
        model (DualSelectivePromptBiomedCLIP): 已加载VPT权重的模型
        preprocess (callable): 图像预处理函数
        tokenizer (callable): 文本 tokenizer
        image_pil (PIL.Image): 输入的PIL图像
        text_prompt (str): 描述ROI的文本
        prompt_group (str): "invariant" 或 "specific"
        device (str): "cuda" 或 "cpu"

    Returns:
        np.array: 归一化后的热图 (与原图大小相同)
    """
    
    # --- 步骤 1: 准备图像和文本输入 ---
    
    # 预处理图像 (resize, center_crop, normalize)
    # processed_image 形状: [3, 224, 224]
    processed_image = preprocess(image_pil).unsqueeze(0).to(device) # [1, 3, 224, 224]
    
    # 编码文本
    text_tokens = tokenizer([text_prompt]).to(device)
    # text_features 形状: [1, 512] (已归一化)
    text_features = model.encode_text(text_tokens, normalize=True)

    # --- 步骤 2: 获取VPT调节的Patch特征 ---
    
    # 这是方案一的核心：
    # 我们调用 model.forward，它会根据 prompt_group 注入对应的VPTs。
    # VPTs 会调节ViT中所有Transformer块的自注意力计算。
    # 我们设置 return_tokens=True 来获取被VPTs调节后的 *所有* patch token。
    
    # 注意：我们设置 normalize=False，因为 forward 默认只归一化 [CLS] token 
    # (即 image_features)，而我们想要的是未归一化的原始 tokens 以便后续处理。
    with torch.no_grad():
        outputs = model.forward(
            processed_image, 
            prompt_group=prompt_group, 
            return_tokens=True, 
            normalize=False
        )

    # all_tokens 形状: [B, 1 + num_prompts + num_patches, 768]
    # (e.g., [1, 1 + 4 + 196, 768])
    all_tokens = outputs["tokens"] 
    
    # 我们跳过 [CLS] token (第0个) 和 prompt tokens (第1到num_prompts个)
    # patch_tokens 形状: [1, num_patches, 768] (e.g., [1, 196, 768])
    num_prompts = model.prompt_config.num_prompts
    patch_tokens = all_tokens[:, 1 + num_prompts :, :]

    # --- 步骤 3: 特征对齐与相似度计算 ---
    
    # 文本特征是 512-dim (来自 model.visual.head 投影)
    # Patch 特征是 768-dim (来自 ViT-L/14 主干)
    # 我们必须将 patch_tokens 通过同一个投影头，使它们维度匹配
    
    head = model.model.visual.head
    # projected_patch_tokens 形状: [1, num_patches, 512]
    projected_patch_tokens = head(patch_tokens)
    
    # 现在我们对齐了维度，在计算点积前，对 patch 特征进行归一化
    projected_patch_tokens = F.normalize(projected_patch_tokens, dim=-1)
    
    # 计算相似度 (Cosine Similarity)
    # text_features 形状: [1, 512]
    # projected_patch_tokens 形状: [1, 196, 512]
    # 我们希望得到每个 patch 与 text 的相似度
    
    # torch.einsum('bpd,bd->bp', ...)
    # b=batch, p=patches, d=dimension
    
    # --- 错误修正 ---
    # 原始代码: text_features.squeeze(0)，这会产生 [512] (1D) 张量，与 'bd' (2D) 公式不匹配
    # 修正后: 直接使用 text_features，其形状为 [1, 512]，与 'bd' (b=1, d=512) 完美匹配
    similarity_map = torch.einsum('bpd,bd->bp', projected_patch_tokens, text_features)
    # similarity_map 形状: [1, 196]
    
    similarity_map = similarity_map.squeeze(0) # 形状: [196]

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
    # 添加一个小的 epsilon 防止除以零 (如果热图所有值都相同)
    min_val = heatmap_numpy.min()
    max_val = heatmap_numpy.max()
    heatmap_normalized = (heatmap_numpy - min_val) / (max_val - min_val + 1e-8)
    
    return heatmap_normalized


def visualize_heatmaps(original_image, heatmap_inv, heatmap_spec, text_prompt):
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
    axes[1].set_title("Domain Invariant Heatmap")
    axes[1].axis('off')
    
    # 域特定热图
    axes[2].imshow(original_image, cmap='gray')
    axes[2].imshow(heatmap_spec, cmap='jet', alpha=0.5)
    axes[2].set_title("Domain Specific Heatmap")
    axes[2].axis('off')
    
    fig.suptitle(f"Text Prompt: \"{text_prompt}\"", fontsize=16)
    plt.tight_layout()
    
    # 保存或显示
    save_path = "vpt_heatmap_comparison.png"
    plt.savefig(save_path)
    print(f"热图已保存到: {save_path}")
    # plt.show() # 如果在 Jupyter 或类似环境中，取消此行注释

if __name__ == "__main__":
    if not os.path.exists(BIOMEDCLIP_PATH):
        print(f"错误: 找不到 BiomedCLIP 路径: {BIOMEDCLIP_PATH}")
        print("请修改脚本顶部的 BIOMEDCLIP_PATH 变量。")
    elif not os.path.exists(VPT_WEIGHTS_PATH):
        print(f"错误: 找不到 VPT 权重路径: {VPT_WEIGHTS_PATH}")
        print("请修改脚本顶部的 VPT_WEIGHTS_PATH 变量。")
    elif not os.path.exists(IMAGE_PATH):
        print(f"错误: 找不到图像路径: {IMAGE_PATH}")
        print("请修改脚本顶部的 IMAGE_PATH 变量。")
    else:
        # 1. 加载模型
        model, preprocess, tokenizer = setup_model(
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
        print(f"正在生成 'invariant' 热图...")
        heatmap_inv = generate_heatmap(
            model, preprocess, tokenizer, image_pil, TEXT_PROMPT, "invariant", DEVICE
        )
        
        # 4. 生成域特定热图
        print(f"正在生成 'specific' 热图...")
        heatmap_spec = generate_heatmap(
            model, preprocess, tokenizer, image_pil, TEXT_PROMPT, "specific", DEVICE
        )
        
        # 5. 可视化对比
        print("正在可视化热图...")
        visualize_heatmaps(image_pil, heatmap_inv, heatmap_spec, TEXT_PROMPT)
        
        print("完成。")

