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
# IMAGE_PATH = "/app/MixDSemi/data/ProstateSlice/BIDMC/train/image/00_08.png"
IMAGE_PATH = "/app/MixDSemi/data/ProstateSlice/UCL/train/image/00_11.png"

# --- 2. 训练时文本配置 (必须与训练脚本 train_MiDSS_NDC_MARK2_CLIP.py 匹配) ---

# 对应 args.text_root
TEXT_ROOT_PATH = "/app/MixDSemi/SynFoCLIP/code/text"
# 对应 args.dataset
DATASET_NAME = "prostate" 
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

def load_training_texts(text_root, dataset_name, llm_model, describe_nums):
    """
    根据训练脚本的逻辑加载文本描述列表。
    (这是对 train_MiDSS_NDC_MARK2_CLIP.py 中 TextSampler 逻辑的 *修正后* 的模拟)
    """
    # 来自 train_MiDSS_NDC_MARK2_CLIP.py 的 TEXT_DATASET_DEFAULTS
    TEXT_DATASET_DEFAULTS = {
        'fundus': 'Fundus',
        'prostate': 'ProstateSlice',
        'MNMS': 'MNMS',
        'BUSI': 'BUSI',
    }
    # 1. 获取数据集对应的 "key" (例如 "ProstateSlice")
    text_dataset_key = TEXT_DATASET_DEFAULTS.get(dataset_name)
    if not text_dataset_key:
        raise ValueError(f"未知的 dataset_name: {dataset_name}")
        
    # 2. 构造 *正确* 的 JSON 文件路径 (例如 .../text/ProstateSlice.json)
    json_filename = f"{text_dataset_key}.json"
    json_path = os.path.join(text_root, json_filename)
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"找不到训练文本文件: {json_path}\n"
                              f"请确保 TEXT_ROOT_PATH ({text_root}) 和 DATASET_NAME ({dataset_name}) 配置正确。")
    
    print(f"正在从 {json_path} 加载训练文本...")
    with open(json_path, 'r') as f:
        text_descriptions = json.load(f) # 加载整个 JSON 文件
        
    # 3. 根据 LLM model 和 type_key 提取文本
    if llm_model not in text_descriptions:
        raise ValueError(f"LLM model '{llm_model}' not found in {json_path}")
        
    if not isinstance(text_descriptions[llm_model], dict):
        raise ValueError(f"JSON 结构错误: 'data[{llm_model}]' 应该是一个字典 (dict)，但它不是。")

    all_texts_dict = {}
    flat_list = []
    
    # 模拟 TextSampler.load_texts 的内部逻辑
    # 遍历所有类别 (e.g., "base", "apex" for prostate)
    for type_key, texts in text_descriptions[llm_model].items():
        if isinstance(texts, list):
            selected_texts = texts[:describe_nums] # 选取前 describe_nums 个文本
            all_texts_dict[type_key] = selected_texts
            flat_list.extend(selected_texts) # 添加到扁平列表
        else:
            print(f"警告: 'data[{llm_model}][{type_key}]' 不是一个列表，已跳过。")

    if not flat_list:
        print(f"警告: 从 {json_path} (model={llm_model}, nums={describe_nums}) 加载的文本列表为空。")
    else:
        print(f"成功加载 {len(flat_list)} 条总文本描述 (来自 {len(all_texts_dict)} 个类别)。")
        
    # 返回扁平化的列表，这对应于训练中的 `all_text_descriptions`
    return flat_list

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


def generate_heatmap(model, preprocess, tokenizer, image_pil, all_training_texts, prompt_group, device):
    """
    为单个图像和 *平均文本特征* 生成热图。
    
    Args:
        model (DualSelectivePromptBiomedCLIP): 已加载VPT权重的模型
        preprocess (callable): 图像预处理函数
        tokenizer (callable): 文本 tokenizer
        image_pil (PIL.Image): 输入的PIL图像
        all_training_texts (list[str]): 训练时使用的所有文本描述
        prompt_group (str): "invariant" 或 "specific"
        device (str): "cuda" 或 "cpu"

    Returns:
        np.array: 归一化后的热图 (与原图大小相同)
    """
    
    # --- 步骤 1: 准备图像和 *平均文本* 输入 ---
    
    # 预处理图像 (resize, center_crop, normalize)
    processed_image = preprocess(image_pil).unsqueeze(0).to(device) # [1, 3, 224, 224]
    
    # 编码 *所有* 训练文本
    print(f"正在为 {len(all_training_texts)} 条文本编码 (prompt_group: {prompt_group})...")
    text_tokens = tokenizer(all_training_texts).to(device)
    
    with torch.no_grad():
        all_text_features = model.encode_text(text_tokens, normalize=True) # [N, 512]
    
    # 计算平均文本特征 (模拟训练时的 a_global_anchor)
    text_features = torch.mean(all_text_features, dim=0, keepdim=True) # [1, 512]
    text_features = F.normalize(text_features, dim=-1) # 确保归一化

    # --- 步骤 2: 获取VPT调节的Patch特征 ---
    
    # 我们调用 model.forward，它会根据 prompt_group 注入对应的VPTs。
    with torch.no_grad():
        outputs = model.forward(
            processed_image, 
            prompt_group=prompt_group, 
            return_tokens=True, 
            normalize=False
        )

    # all_tokens 形状: [B, 1 + num_prompts + num_patches, 768]
    all_tokens = outputs["tokens"] 
    
    # 我们跳过 [CLS] token (第0个) 和 prompt tokens (第1到num_prompts个)
    # patch_tokens 形状: [1, num_patches, 768]
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
    
    # b=batch, p=patches, d=dimension
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
    axes[1].set_title("Domain Invariant Heatmap")
    axes[1].axis('off')
    
    # 域特定热图
    axes[2].imshow(original_image, cmap='gray')
    axes[2].imshow(heatmap_spec, cmap='jet', alpha=0.5)
    axes[2].set_title("Domain Specific Heatmap")
    axes[2].axis('off')
    
    fig.suptitle(f"Heatmap based on Average of {DESCRIBE_NUMS} Training Texts", fontsize=16)
    plt.tight_layout()
    
    # 保存或显示
    save_path = "vpt_heatmap_comparison_avg_text.png"
    plt.savefig(save_path)
    print(f"热图已保存到: {save_path}")
    # plt.show() # 如果在 Jupyter 或类似环境中，取消此行注释

if __name__ == "__main__":
    # 检查所有路径
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
        # 1. 加载训练文本
        all_training_texts = load_training_texts(
            TEXT_ROOT_PATH, 
            DATASET_NAME, 
            LLM_MODEL, 
            DESCRIBE_NUMS
        )

        # 2. 加载模型
        model, preprocess, tokenizer = setup_model(
            BIOMEDCLIP_PATH, 
            VPT_WEIGHTS_PATH, 
            NUM_PROMPTS, 
            EMBED_DIM, 
            ENABLE_PROMPT_SCALE, 
            DEVICE
        )
        
        # 3. 加载图像 (必须转为 RGB)
        image_pil = Image.open(IMAGE_PATH).convert("RGB")

        # 4. 生成域不变热图
        print(f"正在生成 'invariant' 热图...")
        heatmap_inv = generate_heatmap(
            model, preprocess, tokenizer, image_pil, all_training_texts, "invariant", DEVICE
        )
        
        # 5. 生成域特定热图
        print(f"正在生成 'specific' 热图...")
        heatmap_spec = generate_heatmap(
            model, preprocess, tokenizer, image_pil, all_training_texts, "specific", DEVICE
        )
        
        # 6. 可视化对比
        print("正在可视化热图...")
        visualize_heatmaps(image_pil, heatmap_inv, heatmap_spec)
        
        print("完成。")
    else:
        print("请检查脚本顶部的路径配置。")

