#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 文件名：get_clip_roi.py
# 描述：
# 1. 加载BiomedCLIP和LLM生成的文本。
# 2. 将所有“目标文本”编码并平均，得到一个统一的“文本锚点”（E_task）。
# 3. 遍历所有图像，提取它们的“Patch特征”（而非[CLS]全局特征）。
# 4. 计算每个Patch特征与“文本锚点”的余弦相似度。
# 5. 将相似度分数重塑为14x14的热力图，并上采样到原图大小，保存为ROI。
#
# 这解决了 `MiDSS_CLIP_CROP_test1_cosine.py` 实验中发现的
# [CLS] token 对空间位置不敏感的问题。

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage

from utils.clip import create_biomedclip_model_and_preprocess_local
from utils.text_sampler_v2 import TextSampler

from batchgenerators.utilities.file_and_folder_operations import *

# --- 配置 (与您保持一致) ---
IMAGE_ROOT = "/app/MixDSemi/data/ProstateSlice/BIDMC/train/image"
all_images = subfiles(IMAGE_ROOT, suffix=".png", join=True)
all_masks = [i.replace("/image/", "/mask/") for i in all_images]
MAX_IMAGE_COUNT = 5
if MAX_IMAGE_COUNT is not None:
    all_images = all_images[:MAX_IMAGE_COUNT]
    all_masks = all_masks[:MAX_IMAGE_COUNT]

BIOMEDCLIP_PATH = "/root/models/BiomedCLIP" 
TEXT_ROOT_PATH = "/app/MixDSemi/SynFoCLIP/code/text"
DATASET_NAME = "ProstateSlice"
LLM_NAME = "DeepSeek"
DESCRIBE_NUMS = 80  # 80条目标文本
NUM_SAMPLES = 5     # (此脚本中未使用，但为保持一致而保留)

# 4. 定义要分析的文本组
# --- a) 新增的关键词测试组 (手工提示) ---
keyword_test_groups = {
    # 目标关键词
    'Keyword: "A photo of prostate"': ["A photo of prostate"],
    'Keyword: "A photo of prostate gland"': ["A photo of prostate gland"],

    # 风格/模态关键词
    'Keyword: "MRI"': ["MRI"],
    'Keyword: "T2-weighted MRI"': ["T2-weighted MRI"],
    'Keyword: "axial plane"': ["axial plane"],

    # 对照组
    'Keyword: "medical image"': ["medical image"],
    'Keyword: "human body"': ["human body"],
    'Keyword: "background"': ["background"]
}

# --- 输出目录 ---
OUTPUT_DIR = "/app/MixDSemi/SynFoCLIP/code/clip_roi_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 结束配置 ---


def encode_texts(model, tokenizer, texts, device, batch_size=256):
    """批量编码文本并返回归一化的特征"""
    if len(texts) == 0:
        # 获取模型的文本特征维度
        if hasattr(model, 'text_projection') and model.text_projection is not None:
            text_dim = model.text_projection.shape[1]
        else:
            # 兼容没有 text_projection 的模型
            text_dim = model.token_embedding.weight.shape[1]
        return torch.empty(0, text_dim, device=device)

    features = []
    with torch.no_grad():
        for idx in range(0, len(texts), batch_size):
            chunk = texts[idx:idx + batch_size]
            tokenized = tokenizer(chunk).to(device)
            feats = model.encode_text(tokenized)
            features.append(feats.float())
    
    if not features:
        # 如果 texts 为空列表，返回一个正确形状的空张量
        return torch.empty(0, model.text_projection.shape[1], device=device)

    features = torch.cat(features, dim=0)
    return F.normalize(features, dim=-1)


def get_patch_features(model, image_tensor, device):
    """
    获取图像的Patch特征（而非[CLS]全局特征）
    
    关键步骤：
    1. 使用 model.visual.forward_intermediates() 获取最后一层的patch特征
    2. 这些特征已经经过了投影和归一化
    """
    with torch.no_grad():
        # image_tensor 形状: (1, 3, 224, 224)
        # 获取中间层输出，indices=[-1] 表示最后一层
        intermediates = model.visual.forward_intermediates(image_tensor.to(device), indices=[-1])
        
        # 从intermediates中提取patch特征
        # patch_tokens 形状: (1, 768, 14, 14) - 已经是空间特征图格式
        patch_tokens = intermediates['image_intermediates'][0]  # 取最后一层
        
        # 重塑为 (1, 196, 768) 格式，方便后续处理
        # 14*14 = 196 patches
        batch_size, channels, height, width = patch_tokens.shape
        patch_features = patch_tokens.view(batch_size, channels, height * width).transpose(1, 2)
        # 现在形状: (1, 196, 768)
        
        # 手动应用视觉投影层将768维映射到512维
        # 使用visual.head.proj投影层（从timm模型结构中找到的）
        proj_weight = model.visual.head.proj.weight  # (512, 768)
        patch_features_proj = patch_features @ proj_weight.t()  # (1, 196, 512)
        
        # 归一化，使其与文本特征具有可比性
        patch_features_proj = F.normalize(patch_features_proj, dim=-1)
        
        return patch_features_proj  # 形状: (1, 196, 512)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. 加载模型
    print(f"Loading BioMedCLIP from: {BIOMEDCLIP_PATH}")
    model, preprocess, tokenizer = create_biomedclip_model_and_preprocess_local(BIOMEDCLIP_PATH, device)
    model.eval()

    # 2. 加载并编码"文本锚点"
    print(f"Loading texts from: {TEXT_ROOT_PATH}")
    sampler = TextSampler(TEXT_ROOT_PATH)
    targets_texts_dict, style_texts, flat_target_texts = sampler.load_texts(DATASET_NAME, LLM_NAME, DESCRIBE_NUMS)
    
    # 合并文本组：原始目标文本 + 关键词测试组
    all_text_groups = {
        'Original_Target_Texts': flat_target_texts,
        **keyword_test_groups
    }
    
    print(f"Created {len(all_text_groups)} text groups for ROI generation:")
    for group_name, texts in all_text_groups.items():
        print(f"  - {group_name}: {len(texts)} texts")
    
    # 为每个文本组创建语义锚点
    text_anchors = {}
    for group_name, texts in all_text_groups.items():
        if len(texts) > 0:
            text_features = encode_texts(model, tokenizer, texts, device)
            if text_features.numel() > 0:
                # 创建该组的语义锚点（平均所有文本特征）
                anchor = text_features.mean(dim=0, keepdim=True)  # 形状: (1, 512)
                anchor = F.normalize(anchor, dim=-1)
                text_anchors[group_name] = anchor
                print(f"  Semantic anchor for '{group_name}' created, shape: {anchor.shape}")
    
    if not text_anchors:
        print("Error: No valid text anchors created.")
        return

    # 3. 循环处理图像，生成ROI
    print(f"\nProcessing {len(all_images)} images to generate ROIs...")
    for img_idx, img_path in enumerate(all_images):
        try:
            image_name = os.path.basename(img_path).replace('.png', '')
            print(f"  [{img_idx+1}/{len(all_images)}] Processing {image_name}")

            # a. 加载和预处理图像
            orig_img_pil = Image.open(img_path).convert("RGB")
            orig_size_hw = (orig_img_pil.height, orig_img_pil.width) # (H, W)
            image_tensor = preprocess(orig_img_pil).unsqueeze(0) # (1, 3, 224, 224)

            # b. 获取 Patch 特征
            # patch_features 形状: (1, 196, 512)
            patch_features = get_patch_features(model, image_tensor, device)

            # c. 计算 Patch 与 文本锚点 的相似度
            # E_task 形状: (1, 512) -> E_task.t() 形状: (512, 1)
            # (1, 196, 512) @ (512, 1) -> (1, 196, 1)
            roi_logits = patch_features @ E_task.t()
            
            # 确定 patch 网格的边长 (例如 14)
            num_patches = patch_features.shape[1] # 196
            num_patches_side = int(num_patches ** 0.5) # 14
            
            # d. 重塑为 2D 热力图
            # (1, 196, 1) -> (14, 14)
            roi_map = roi_logits.squeeze().reshape(num_patches_side, num_patches_side)
            roi_map = roi_map.cpu().numpy()

            # e. 归一化 (0-1) 以便保存和用作“软标签”
            roi_map_norm = (roi_map - roi_map.min()) / (roi_map.max() - roi_map.min() + 1e-8)

            # f. 上采样到原始图像分辨率
            roi_upsampled = cv2.resize(
                roi_map_norm, 
                (orig_size_hw[1], orig_size_hw[0]), # (W, H) for cv2.resize
                interpolation=cv2.INTER_LINEAR
            )

            # g. 保存ROI结果
            sample_dir = os.path.join(OUTPUT_DIR, image_name)
            os.makedirs(sample_dir, exist_ok=True)
            
            # 保存为 .npy (用于后续的模型训练)
            npy_path = os.path.join(sample_dir, "roi_heatmap.npy")
            np.save(npy_path, roi_upsampled.astype(np.float32))

            # 保存为热力图 (用于可视化检查)
            heatmap_path = os.path.join(sample_dir, "roi_heatmap_visual.png")
            plt.imsave(heatmap_path, roi_upsampled, cmap='viridis')
            
            # (可选) 保存叠加图
            overlay_path = os.path.join(sample_dir, "roi_overlay.png")
            heatmap_color = cv2.applyColorMap(np.uint8(255 * roi_upsampled), cv2.COLORMAP_VIRIDIS)
            heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(np.array(orig_img_pil), 0.6, heatmap_color, 0.4, 0)
            Image.fromarray(overlay).save(overlay_path)

        except Exception as e:
            print(f"    Error processing {image_name}: {e}")
            continue
            
    print(f"\nDone. ROI heatmaps saved in {OUTPUT_DIR}")

if __name__ == "__main__":
    plt.switch_backend('Agg')
    main()