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

# --- 结束配置 ---

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


def get_features_for_cam(model, image_tensor):
    """
    一个修改后的前向传播函数，用于获取 Grad-CAM 所需的
    CLS 特征 和 Patch Tokens。
    
    [已根据您的报错日志修复]
    """
    
    try:
        # 1. trunk 现在是 model.visual
        trunk = model.visual 
        
        # 2. 获取所有 tokens (CLS + Patches)
        #    这是 Grad-CAM 需要梯度的地方
        all_tokens = trunk.forward_features(image_tensor) # e.g., [1, 197, 768]
        all_tokens.retain_grad() # <--- 关键：为反向传播保留梯度
        
    except AttributeError as e:
        print(f"致命错误: 无法在 'model.visual' 上调用 'forward_features'。")
        print(f"报错: {e}")
        print("这表明 'model.visual' 的结构不是一个标准的 open_clip ViT。")
        print("CAM 方法失败。")
        return None, None
    except Exception as e:
        print(f"get_features_for_cam 发生未知错误: {e}")
        return None, None

    # 3. 将 CLS 和 Patch 分离
    cls_token = all_tokens[:, 0]     # [B, C] (e.g., [1, 768])
    patch_tokens = all_tokens[:, 1:, :]  # [B, N, C] (e.g., [1, 196, 768])
    
    # 4. 分别处理它们
    # CLS token -> 最终的图像特征 (用于计算 "score")
    #    a. 通过 ViT 的 head (pre-logits)
    cls_feature = trunk.forward_head(cls_token, pre_logits=True)
    
    #    b. 通过 CLIP 投影层
    #       [修复] 我们使用 'text_projection' 因为 'visual_projection' 不存在
    if hasattr(model, 'text_projection'):
        cls_feature = model.text_projection(cls_feature) # [B, E_proj] (e.g., [1, 512])
    elif hasattr(model.visual, 'head'):
        # 备用方案（如果 text_projection 不起作用）
        cls_feature = model.visual.head(cls_feature)
    else:
        print("致命错误：找不到投影层 (text_projection 或 visual.head)")
        return None, None

    # 5. 返回用于 CAM 的两个关键组件
    #    cls_feature: [1, 512] (用于计算 score)
    #    patch_tokens: [1, 196, 768] (作为特征图 A_k)
    return cls_feature, patch_tokens


def generate_heatmap_grad_cam(model, image_tensor, avg_text_feature, orig_img_size):
    """
    计算 Grad-CAM 热图
    [已根据您的报错日志修复]
    """
    
    # 1. 获取特征
    #    image_tensor 必须启用 requires_grad
    image_features, patch_tokens = get_features_for_cam(model, image_tensor)
    
    if image_features is None:
        return None # 如果 get_features_for_cam 失败

    # 2. 计算 CAM 的 "Score" (即 CLIP 相似度)
    image_features_norm = F.normalize(image_features, dim=-1)
    # avg_text_feature 已经是归一化的
    
    score = (image_features_norm @ avg_text_feature.T).sum()

    # 3. 反向传播以获取梯度
    model.zero_grad()
    score.backward() # 计算 score 对 all_tokens 的梯度
    
    # 4. 获取梯度
    #    注意：梯度是在 all_tokens 上，我们只取 patch 部分
    if patch_tokens.grad is None:
        # 如果梯度在父张量 'all_tokens' 上
        if all_tokens.grad is not None:
             grads = all_tokens.grad[:, 1:, :] # [B, N, C]
        else:
             raise RuntimeError("无法获取 patch_tokens 的梯度。")
    else:
        grads = patch_tokens.grad # [B, N, C] (e.g., [1, 196, 768])
        
    # 5. 计算 Grad-CAM 权重 (alpha_k)
    #    (B, N, C) -> (B, 1, C)
    weights = torch.mean(grads, dim=1, keepdim=True) # 空间池化

    # 6. 计算加权激活
    #    (B, N, C) * (B, 1, C) -> (B, N, C)
    #    我们使用原始的、未投影的 patch_tokens
    weighted_activations = patch_tokens * weights
    
    # (B, N, C) -> (B, N)
    cam = torch.sum(weighted_activations, dim=2) # 通道求和
    
    # 7. ReLU (只保留正贡献)
    cam = F.relu(cam) # [B, N]
    
    heatmap = cam.squeeze(0).cpu().detach().numpy() # [N]
    
    # --- 8. Reshape 和上采样 (同您的代码) ---
    patch_count = heatmap.shape[0]
    grid_size = int(patch_count ** 0.5)
    if grid_size * grid_size != patch_count:
        raise ValueError(f"无法将 patch 数量 {patch_count} 转换为完美的方形网格。")
        
    # Reshape to [H_grid, W_grid] (e.g., [14, 14])
    heatmap_grid = heatmap.reshape(grid_size, grid_size)
    
    # 归一化到 0-1 (用于可视化)
    heatmap_normalized = (heatmap_grid - heatmap_grid.min()) / (heatmap_grid.max() - heatmap_grid.min() + 1e-8)
    
    # 调整大小到原始图像尺寸
    heatmap_resized = cv2.resize(heatmap_normalized, (orig_img_size[0], orig_img_size[1]), interpolation=cv2.INTER_CUBIC)
    
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
    # --- 关键修改：为 CAM 启用梯度 ---
    image_tensor.requires_grad_(True)
    
    # 3. 加载文本 (仅用于对比组)
    print(f"Loading texts from: {TEXT_ROOT_PATH}")
    flat_texts = []
    style_texts = []
    try:
        sampler = TextSampler(TEXT_ROOT_PATH)
        all_texts, style_texts, flat_texts = sampler.load_texts(DATASET_NAME, LLM_NAME, DESCRIBE_NUMS)
    except Exception as e:
        print(f"警告：加载 TextSampler 文本失败。将跳过 'Avg' 组。")
        print(f"详细信息: {e}")

    # 4. 定义要分析的文本组 (同上一个版本)
    text_groups = {}
    keyword_test_groups = {
        'Keyword: "prostate"': ["prostate"],
        'Keyword: "prostate gland"': ["prostate gland"],
        'Keyword: "MRI"': ["MRI"],
        'Keyword: "T2-weighted MRI"': ["T2-weighted MRI"],
        'Keyword: "axial plane"': ["axial plane"],
        'Keyword: "medical image"': ["medical image"],
        'Keyword: "human body"': ["human body"],
        'Keyword: "background"': ["background"]
    }
    text_groups_from_sampler = {
        'LLM_Targets_Avg': flat_texts,
        'LLM_Styles_Avg': style_texts
    }
    text_groups = {**keyword_test_groups, **text_groups_from_sampler}
    
    print(f"定义了 {len(text_groups)} 个文本组进行分析: {list(text_groups.keys())}")
    
    # 5. 准备绘图
    num_plots = len(text_groups)
    fig, axs = plt.subplots(1, num_plots, figsize=(num_plots * 4, 5))
    if num_plots == 1:
        axs = [axs]
        
    # 6. 循环生成和绘制热图
    plot_index = 0
    for group_name, text_list in text_groups.items():
        if plot_index >= len(axs): break
        ax = axs[plot_index]
        
        if not text_list:
            print(f"Processing group: {group_name} (跳过，文本列表为空)")
            ax.imshow(orig_img)
            ax.set_title(f"{group_name}\n(No Texts)", fontsize=10)
            ax.axis('off')
            plot_index += 1
            continue
            
        print(f"Processing group: {group_name} ({len(text_list)} texts)")
        
        # --- 核心修改：使用 Grad-CAM 逻辑 ---
        
        # a. 获取平均文本特征 (和以前一样)
        avg_text_feature = get_avg_text_feature(model, tokenizer, text_list, device)
        
        # b. 生成 Grad-CAM 热图
        #    注意：我们每次循环都必须重新计算梯度
        heatmap = generate_heatmap_grad_cam(model, image_tensor, avg_text_feature, orig_img.size)
        
        if heatmap is None:
             print(f" - 生成 CAM 失败。跳过此组。")
             ax.imshow(orig_img)
             ax.set_title(f"{group_name}\n(CAM FAILED)", fontsize=10)
             ax.axis('off')
             plot_index += 1
             continue

        # c. 绘制
        plot_heatmap_on_image(ax, orig_img.copy(), heatmap, group_name)
        plot_index += 1
        
    # 7. 保存图像
    output_filename = "clip_response_heatmaps_GRAD-CAM.png" # 改了新文件名
    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"\n分析完成！CAM 热图已保存到: {output_filename}")

if __name__ == "__main__":
    # 确保 matplotlib 不会尝试在无头服务器上打开 GUI 窗口
    plt.switch_backend('Agg')
    main()

