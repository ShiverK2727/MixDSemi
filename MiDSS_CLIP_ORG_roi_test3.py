import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append("/app/MixDSemi/SynFoCLIP/refer/gScoreCAM")

from pytorch_grad_cam.gscore_cam import GScoreCAM  # type: ignore

# 导入您提供的 .py 文件
try:
    from utils.clip import create_biomedclip_model_and_preprocess_local
    from utils.text_sampler_v2 import TextSampler
except ImportError:
    print("错误：无法导入 'clip.py' 或 'text_sampler_v2.py'。")
    print("请确保这些文件与此脚本位于同一目录中。")
    exit()

# --- 用户配置 ---

# 您想要分析的测试图像
# IMAGE_PATH = "/app/MixDSemi/data/ProstateSlice/BIDMC/train/image/00_08.png"
IMAGE_PATH = "/app/MixDSemi/data/ProstateSlice/UCL/train/image/00_08.png"

# 您的 BiomedCLIP 模型文件夹路径
BIOMEDCLIP_PATH = "/root/models/BiomedCLIP" 

# 包含 text json 文件的根目录 (根据 text_sampler_v2.py 中的示例路径)
TEXT_ROOT_PATH = "/app/MixDSemi/SynFoCLIP/code/text"

# 文本采样器参数
DATASET_NAME = "ProstateSlice"
LLM_NAME = "GPT5"
DESCRIBE_NUMS = 80  # 每个类别的描述数量
NUM_SAMPLES = 5     # 采样份数

# --- 结束配置 ---

MAX_TEXTS_PER_GROUP = 20  # 每个文本组用于 CAM 的最大提示数量，避免计算量过大
GSCORECAM_TOPK = 200
GSCORECAM_BATCH = 64


def vit_reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
    """将 ViT block 输出转换为 [B, C, H, W] 形状，供 CAM 使用。"""
    if tensor.dim() != 3:
        raise ValueError(f"期望张量维度为 3，实际为 {tensor.dim()} 维。")
    patch_tokens = tensor[:, 1:, :]  # 丢弃 CLS token
    batch, num_tokens, hidden = patch_tokens.shape
    grid_size = int(num_tokens ** 0.5)
    if grid_size * grid_size != num_tokens:
        raise ValueError(f"无法将 patch 数 {num_tokens} 重塑为正方形网格。")
    patch_tokens = patch_tokens.permute(0, 2, 1).reshape(batch, hidden, grid_size, grid_size)
    return patch_tokens


class BiomedCLIPForCAM(torch.nn.Module):
    """包装 BioMedCLIP，使其前向输出与原始 CLIP 接口一致。"""

    def __init__(self, clip_model: torch.nn.Module):
        super().__init__()
        self.clip_model = clip_model
        self.visual = clip_model.visual  # 供 CAM 挂钩视觉层

    def forward(self, image: torch.Tensor, text: torch.Tensor):
        image_features, text_features, logit_scale = self.clip_model(image, text)
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        scale = logit_scale.exp()
        logits_per_image = scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text


def build_gscorecam(model: torch.nn.Module, device: str) -> GScoreCAM:
    """构造适配 BioMedCLIP 的 GScoreCAM 实例。"""
    wrapped = BiomedCLIPForCAM(model)
    target_layer = wrapped.visual.trunk.blocks[-1]
    cam = GScoreCAM(
        model=wrapped,
        target_layers=[target_layer],
        use_cuda=device == "cuda",
        reshape_transform=vit_reshape_transform,
        is_clip=True,
        batch_size=GSCORECAM_BATCH,
        topk=GSCORECAM_TOPK,
        drop=False,
    )
    return cam


def compute_group_heatmap(
    cam: GScoreCAM,
    image_tensor: torch.Tensor,
    text_list,
    tokenizer,
    device: str,
    orig_size,
    max_texts: int | None = None,
):
    """对文本组运行 gScoreCAM，并返回平均热图与有效提示数量。"""
    if not text_list:
        return None, 0

    if max_texts is not None and len(text_list) > max_texts:
        selected_texts = text_list[:max_texts]
    else:
        selected_texts = text_list

    heatmap_sum = None
    valid = 0
    for text in selected_texts:
        tokens = tokenizer([text]).to(device)
        grayscale_cam = cam((image_tensor, tokens), targets=0)
        if grayscale_cam is None:
            continue
        heatmap = grayscale_cam[0]
        if heatmap_sum is None:
            heatmap_sum = heatmap
        else:
            heatmap_sum += heatmap
        valid += 1

    if not valid or heatmap_sum is None:
        return None, 0

    avg_heatmap = heatmap_sum / valid
    avg_heatmap = (avg_heatmap - avg_heatmap.min()) / (avg_heatmap.max() - avg_heatmap.min() + 1e-8)
    heatmap_resized = cv2.resize(avg_heatmap, orig_size, interpolation=cv2.INTER_CUBIC)
    return heatmap_resized, valid

def plot_heatmap_on_image(ax, orig_img, heatmap, title):
    """
    在 Matplotlib Axs 上绘制热图
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

    print("Initializing gScoreCAM...")
    cam = build_gscorecam(model, device)

    # 2. 加载图像
    print(f"Loading image from: {IMAGE_PATH}")
    if not os.path.exists(IMAGE_PATH):
        print(f"错误：找不到图像文件 {IMAGE_PATH}")
        return
        
    orig_img = Image.open(IMAGE_PATH).convert("RGB")
    image_tensor = preprocess(orig_img).unsqueeze(0)
    cam.img_size = orig_img.size  # 避免 base_cam 在 tuple 输入上解析尺寸时报错
    
    # 3. 加载文本
    print(f"Loading texts from: {TEXT_ROOT_PATH}")
    if not os.path.exists(TEXT_ROOT_PATH):
        print(f"错误：找不到文本根目录 {TEXT_ROOT_PATH}")
        return
        
    try:
        sampler = TextSampler(TEXT_ROOT_PATH)
        all_texts, style_texts, flat_texts = sampler.load_texts(DATASET_NAME, LLM_NAME, DESCRIBE_NUMS)
        per_type_subsets, flat_subsets = sampler.sample_subsets(all_texts, NUM_SAMPLES)
    except Exception as e:
        print(f"错误：加载文本失败。请检查 TEXT_ROOT_PATH 和 text_sampler_v2.py。")
        print(f"详细信息: {e}")
        return

    # 4. 定义要分析的文本组
    text_groups = {
        'All_Targets_Avg': flat_texts,
        'All_Styles_Avg': style_texts
    }
    for i, subset in enumerate(flat_subsets):
        text_groups[f'Subset_{i+1}_Avg'] = subset
    
    print(f"定义了 {len(text_groups)} 个文本组进行分析: {list(text_groups.keys())}")

    # 5. 提取图像 Patch 特征 (仅一次)
    print("Extracting gScoreCAM heatmaps...")
    
    # 6. 准备绘图
    num_plots = len(text_groups)
    fig, axs = plt.subplots(1, num_plots, figsize=(num_plots * 5, 6))
    if num_plots == 1:
        axs = [axs] # 保证 axs 是一个可迭代列表
        
    # 7. 循环生成和绘制热图
    plot_index = 0
    for group_name, text_list in text_groups.items():
        ax = axs[plot_index]
        print(f"Processing group: {group_name} ({len(text_list)} texts)")
        
        if not text_list:
            print(f" - 跳过，因为文本列表为空。")
            ax.imshow(orig_img)
            ax.set_title(f"{group_name}\n(No Texts)", fontsize=10)
            ax.axis('off')
            plot_index += 1
            continue
            
        heatmap, used = compute_group_heatmap(
            cam,
            image_tensor,
            text_list,
            tokenizer,
            device,
            orig_img.size,
            max_texts=MAX_TEXTS_PER_GROUP,
        )

        if heatmap is None:
            ax.imshow(orig_img)
            ax.set_title(f"{group_name}\n(No Valid CAM)", fontsize=10)
            ax.axis('off')
        else:
            print(f" - 参与提示数量: {used}")
            plot_heatmap_on_image(ax, orig_img, heatmap, f"{group_name}\nused={used}")
        plot_index += 1
        
    # 8. 保存图像
    output_filename = "clip_response_heatmaps.png"
    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"\n分析完成！热图已保存到: {output_filename}")

if __name__ == "__main__":
    # 确保 matplotlib 不会尝试在无头服务器上打开 GUI 窗口
    plt.switch_backend('Agg')
    main()
