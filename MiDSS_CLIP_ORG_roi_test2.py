import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

# 导入您提供的 .py 文件
try:
    # 保持您文件中的 import 结构
    from utils.clip import create_biomedclip_model_and_preprocess_local
    from utils.text_sampler_v2 import TextSampler
except ImportError:
    print("错误：无法导入 'utils.clip' 或 'utils.text_sampler_v2'。")
    print("请确保这些文件在您的 Python 路径中。")
    # exit() # 暂时注释掉

# --- 用户配置 (来自您的文件) ---

# 您想要分析的测试图像
# IMAGE_PATH = "/app/MixDSemi/data/ProstateSlice/BIDMC/train/image/00_08.png"
IMAGE_PATH = "/app/MixDSemi/data/ProstateSlice/UCL/train/image/00_08.png"

# 您的 BiomedCLIP 模型文件夹路径
BIOMEDCLIP_PATH = "/root/models/BiomedCLIP" 

# 包含 text json 文件的根目录 (根据 text_sampler_v2.py 中的示例路径)
TEXT_ROOT_PATH = "/app/MixDSemi/SynFoCLIP/code/text"

# 文本采样器参数
DATASET_NAME = "ProstateSlice"
LLM_NAME = "GPT5" # 来自您的 MiDSS_CLIP_ORG_roi_test1.py
DESCRIBE_NUMS = 80  # 每个类别的描述数量
# NUM_SAMPLES = 5     # 在这个测试中我们不再需要 subsets

# --- 结束配置 ---

def get_patch_features(model, image_tensor):
    """
    获取 ViT 模型的 patch features (不含 CLS token)
    [此函数来自您的 MiDSS_CLIP_ORG_roi_test1.py]
    """
    with torch.no_grad():
        # open_clip 的 timm 后端没有 forward_features, 需通过 forward_intermediates 获取最后一层 patch 表示
        intermediates = model.visual.forward_intermediates(
            image_tensor,
            output_fmt='NCHW',
            output_extra_tokens=False
        ).get('image_intermediates')

    if not intermediates:
        raise RuntimeError("未能从模型中提取 patch 特征。请确认视觉骨干支持 forward_intermediates。")

    patch_grid = intermediates[-1]  # [B, C, H, W]
    if patch_grid.dim() != 4:
        raise RuntimeError(f"意外的特征维度: {patch_grid.shape}")

    # 拉平成 [B, N_Patches, Hidden_Dim] (e.g., [1, 196, 768])
    patch_features = patch_grid.flatten(2).transpose(1, 2).contiguous()

    # 通过视觉头投影到与文本特征一致的维度（通常为 512）
    if hasattr(model.visual, 'head') and model.visual.head is not None:
        patch_features = model.visual.head(patch_features) # (e.g., [1, 196, 512])

    # 归一化
    patch_features = F.normalize(patch_features, dim=-1)
    return patch_features

def get_avg_text_feature(model, tokenizer, text_list, device):
    """
    获取文本列表的平均特征向量
    [此函数来自您的 MiDSS_CLIP_ORG_roi_test1.py]
    """
    if not text_list:
        return None
    
    # 对文本进行分词
    text_tokens = tokenizer(text_list).to(device)
    
    # 编码文本并获取特征
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
    
    # 计算平均特征并归一化
    avg_feature = text_features.mean(dim=0, keepdim=True)
    avg_feature = F.normalize(avg_feature, dim=-1)
    return avg_feature

def generate_heatmap(patch_features, avg_text_feature, orig_img_size):
    """
    计算 patch 特征和文本特征之间的相似度热图
    [此函数来自您的 MiDSS_CLIP_ORG_roi_test1.py]
    """
    # patch_features: [1, N_Patches, E] (e.g., [1, 196, 512])
    # avg_text_feature: [1, E] (e.g., [1, 512])
    
    # 将文本特征转置为 [1, E, 1] 以进行矩阵乘法
    avg_text_feature_T = avg_text_feature.unsqueeze(-1)
    
    # 计算相似度: [1, N_Patches, E] @ [1, E, 1] -> [1, N_Patches, 1]
    similarity_map = patch_features @ avg_text_feature_T
    
    # 获取网格大小, e.g., 196 -> 14
    patch_count = similarity_map.shape[1]
    grid_size = int(patch_count ** 0.5)
    if grid_size * grid_size != patch_count:
        raise ValueError(f"无法将 patch 数量 {patch_count} 转换为完美的方形网格。")
        
    # Reshape to [1, H_grid, W_grid] (e.g., [1, 14, 14])
    similarity_map_grid = similarity_map.reshape(1, grid_size, grid_size)
    
    # Squeeze, detach, and move to CPU
    heatmap = similarity_map_grid.squeeze(0).cpu().detach().numpy()
    
    # 归一化到 0-1 (用于可视化)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    # 调整大小到原始图像尺寸，使用 INTER_CUBIC 插值使热图更平滑
    heatmap_resized = cv2.resize(heatmap, (orig_img_size[0], orig_img_size[1]), interpolation=cv2.INTER_CUBIC)
    
    return heatmap_resized

def plot_heatmap_on_image(ax, orig_img, heatmap, title):
    """
    在 Matplotlib Axs 上绘制热图
    [此函数来自您的 MiDSS_CLIP_ORG_roi_test1.py]
    """
    ax.imshow(orig_img)
    ax.imshow(heatmap, cmap='viridis', alpha=0.6) # alpha 控制热图透明度
    ax.set_title(title, fontsize=10)
    ax.axis('off')

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. 加载模型
    print(f"Loading BioMedCLIP from: {BIOMEDCLIP_PATH}")
    if not os.path.exists(BIOMEDCLIP_PATH):
        print(f"错误：找不到模型路径 {BIOMEDCLIP_PATH}")
        return
    try:
        model, preprocess, tokenizer = create_biomedclip_model_and_preprocess_local(BIOMEDCLIP_PATH, device)
        model.eval()
    except Exception as e:
        print(f"错误：加载模型失败。请检查 BIOMEDCLIP_PATH 是否正确。")
        print(f"详细信息: {e}")
        return

    # 2. 加载图像
    print(f"Loading image from: {IMAGE_PATH}")
    if not os.path.exists(IMAGE_PATH):
        print(f"错误：找不到图像文件 {IMAGE_PATH}")
        return
        
    orig_img = Image.open(IMAGE_PATH).convert("RGB")
    image_tensor = preprocess(orig_img).unsqueeze(0).to(device)
    
    # 3. 加载文本 (仅用于对比组)
    print(f"Loading texts from: {TEXT_ROOT_PATH}")
    flat_texts = []
    style_texts = []
    try:
        sampler = TextSampler(TEXT_ROOT_PATH)
        all_texts, style_texts, flat_texts = sampler.load_texts(DATASET_NAME, LLM_NAME, DESCRIBE_NUMS)
        # per_type_subsets, flat_subsets = sampler.sample_subsets(all_texts, NUM_SAMPLES) # <--- 修改：移除 subsets
    except Exception as e:
        print(f"警告：加载 TextSampler 文本失败。将跳过 'Avg' 组。")
        print(f"详细信息: {e}")

    # 4. 定义要分析的文本组
    text_groups = {}
    
    # --- a) 新增的关键词测试组 ---
    keyword_test_groups = {
        # 目标关键词
        'Keyword: "prostate"': ["prostate"],
        'Keyword: "prostate gland"': ["prostate gland"],
        
        # 风格/模态关键词
        'Keyword: "MRI"': ["MRI"],
        'Keyword: "T2-weighted MRI"': ["T2-weighted MRI"],
        'Keyword: "axial plane"': ["axial plane"],
        
        # 对照组
        'Keyword: "medical image"': ["medical image"],
        'Keyword: "human body"': ["human body"],
        'Keyword: "background"': ["background"]
    }
    
    # --- b) 从 TextSampler 加载的平均组 (用于对比) ---
    text_groups_from_sampler = {
        'LLM_Targets_Avg': flat_texts,
        'LLM_Styles_Avg': style_texts
    }

    # 合并字典，关键词在前
    text_groups = {**keyword_test_groups, **text_groups_from_sampler}

    # --- 修改：移除 Subsets 循环 ---
    # for i, subset in enumerate(flat_subsets):
    #     text_groups[f'Subset_{i+1}_Avg'] = subset
    
    print(f"定义了 {len(text_groups)} 个文本组进行分析: {list(text_groups.keys())}")

    # 5. 提取图像 Patch 特征 (仅一次)
    print("Extracting image patch features...")
    patch_features = get_patch_features(model, image_tensor)
    
    # 6. 准备绘图
    num_plots = len(text_groups)
    fig, axs = plt.subplots(1, num_plots, figsize=(num_plots * 4, 5)) # 调整了 figsize
    if num_plots == 1:
        axs = [axs] # 保证 axs 是一个可迭代列表
        
    # 7. 循环生成和绘制热图
    plot_index = 0
    for group_name, text_list in text_groups.items():
        # 确保 axs 索引不会越界
        if plot_index >= len(axs):
            print(f"警告：plot 索引越界 (index={plot_index})，跳过 {group_name}")
            break
            
        ax = axs[plot_index]
        
        if not text_list:
            print(f"Processing group: {group_name} (跳过，文本列表为空)")
            ax.imshow(orig_img)
            ax.set_title(f"{group_name}\n(No Texts)", fontsize=10)
            ax.axis('off')
            plot_index += 1
            continue
            
        print(f"Processing group: {group_name} ({len(text_list)} texts)")
        # 获取平均文本特征
        avg_text_feature = get_avg_text_feature(model, tokenizer, text_list, device)
        
        # 生成热图
        heatmap = generate_heatmap(patch_features, avg_text_feature, orig_img.size)
        
        # 绘制
        plot_heatmap_on_image(ax, orig_img, heatmap, group_name)
        plot_index += 1
        
    # 8. 保存图像
    output_filename = "clip_response_heatmaps_with_keywords.png" # 改了新文件名
    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"\n分析完成！热图已保存到: {output_filename}")

if __name__ == "__main__":
    # 确保 matplotlib 不会尝试在无头服务器上打开 GUI 窗口
    plt.switch_backend('Agg')
    main()
