import os
import hashlib
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
from scipy import ndimage

from utils.clip import create_biomedclip_model_and_preprocess_local
from utils.text_sampler_v2 import TextSampler

from batchgenerators.utilities.file_and_folder_operations import *

IMAGE_VARIANTS = [
    ('x', 'Original (x)'),
    ('x_c', 'Masked Foreground (x_c)'),
    ('x_c2', 'Dilated Mask (x_c2)'),
    ('x_err', 'Shifted Eroded Foreground (x_err)')
]
VARIANT_KEYS = [item[0] for item in IMAGE_VARIANTS]

# Test image configuration
IMAGE_ROOT = "/app/MixDSemi/data/ProstateSlice/BIDMC/train/image"
all_images = subfiles(IMAGE_ROOT, suffix=".png", join=True)
all_masks = [i.replace("/image/", "/mask/") for i in all_images]
MAX_IMAGE_COUNT = 5  # Set to an integer for quick debugging, e.g., 5
if MAX_IMAGE_COUNT is not None:
    all_images = all_images[:MAX_IMAGE_COUNT]
    all_masks = all_masks[:MAX_IMAGE_COUNT]

# Model path
BIOMEDCLIP_PATH = "/root/models/BiomedCLIP" 

# Text root (aligned with text_sampler_v2.py example)
TEXT_ROOT_PATH = "/app/MixDSemi/SynFoCLIP/code/text2"

# Text sampler parameters
DATASET_NAME = "ProstateSlice"
LLM_NAME = "GPT5"
DESCRIBE_NUMS = 80  # Number of descriptions per class
NUM_SAMPLES = 5     # Number of sampled subsets

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

# Output directory
OUTPUT_DIR = "/app/MixDSemi/SynFoCLIP/code/crop_comparison_results2_GPT5_cosine"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- End configuration ---


def load_and_preprocess_images(image_path, mask_path, preprocess):
    """Prepare four image variants for comparison."""
    orig_img = Image.open(image_path).convert("RGB")
    orig_array = np.array(orig_img)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to load mask: {mask_path}")

    _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    mask_binary = mask_binary.astype(bool)

    x = preprocess(orig_img).unsqueeze(0)

    x_c_array = orig_array.copy()
    x_c_array[~mask_binary] = 0
    x_c_img = Image.fromarray(x_c_array)
    x_c = preprocess(x_c_img).unsqueeze(0)

    struct_elem = ndimage.generate_binary_structure(2, 1)  # 4-connectivity
    mask_dilated = ndimage.binary_dilation(mask_binary, structure=struct_elem, iterations=2)
    x_c2_array = orig_array.copy()
    x_c2_array[~mask_dilated] = 0
    x_c2_img = Image.fromarray(x_c2_array)
    x_c2 = preprocess(x_c2_img).unsqueeze(0)

    mask_eroded = ndimage.binary_erosion(mask_binary, structure=struct_elem, iterations=2)
    shifted_array = np.zeros_like(orig_array)
    if mask_eroded.any():
        ys, xs = np.where(mask_eroded)
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()

        patch = orig_array[y_min:y_max + 1, x_min:x_max + 1].copy()
        patch_mask = mask_eroded[y_min:y_max + 1, x_min:x_max + 1]
        patch[~patch_mask] = 0

        seed_hex = hashlib.md5(image_path.encode('utf-8')).hexdigest()
        seed = int(seed_hex[:8], 16)
        rng = np.random.default_rng(seed)

        max_y_offset = shifted_array.shape[0] - patch.shape[0]
        max_x_offset = shifted_array.shape[1] - patch.shape[1]
        y_offset = rng.integers(0, max_y_offset + 1) if max_y_offset > 0 else 0
        x_offset = rng.integers(0, max_x_offset + 1) if max_x_offset > 0 else 0

        dest_slice = shifted_array[y_offset:y_offset + patch.shape[0], x_offset:x_offset + patch.shape[1]]
        dest_slice[patch_mask] = patch[patch_mask]

    x_err_img = Image.fromarray(shifted_array)
    x_err = preprocess(x_err_img).unsqueeze(0)

    return x, x_c, x_c2, x_err


def encode_text_batch(model, tokenizer, texts, device, batch_size=256):
    if len(texts) == 0:
        if hasattr(model, 'text_projection') and model.text_projection is not None:
            text_dim = model.text_projection.shape[1]
        else:
            text_dim = model.token_embedding.weight.shape[1]
        return torch.empty(0, text_dim, device=device)

    features = []
    with torch.no_grad():
        for idx in range(0, len(texts), batch_size):
            chunk = texts[idx:idx + batch_size]
            tokenized = tokenizer(chunk).to(device)
            feats = model.encode_text(tokenized)
            features.append(feats.float())
    features = torch.cat(features, dim=0)
    return F.normalize(features, dim=-1)


def compute_image_features(model, images_dict, device):
    image_features = {}
    with torch.no_grad():
        for key, tensor in images_dict.items():
            feats = model.encode_image(tensor.to(device))
            image_features[key] = F.normalize(feats.float(), dim=-1)
    return image_features


def compute_text_probabilities(image_feature, text_features, scale):
    if text_features.numel() == 0:
        return torch.empty(0)
    logits = scale * image_feature @ text_features.t()
    probs = torch.softmax(logits.squeeze(0), dim=-1)
    return probs.cpu()


def compute_text_logits(image_feature, text_features, scale):
    """
    Computes the independent logits (cosine similarity * scale) for each text.
    """
    if text_features.numel() == 0:
        return torch.empty(0)
    
    # image_feature: (1, D)
    # text_features: (N, D)
    # 
    # (1, D) @ (D, N) -> (1, N)
    logits = scale * image_feature @ text_features.t()
    
    # 返回 (N,) 形状的张量
    return logits.squeeze(0).cpu()


def compute_subset_mean_features(model, tokenizer, flat_subsets, device, batch_size=256):
    subset_features = []
    for subset in flat_subsets:
        if len(subset) == 0:
            subset_features.append(None)
            continue
        feats = encode_text_batch(model, tokenizer, subset, device, batch_size)
        mean_feat = feats.mean(dim=0, keepdim=True)
        mean_feat = F.normalize(mean_feat, dim=-1)
        subset_features.append(mean_feat)
    return subset_features


def plot_text_scores(image_name, target_scores, style_scores, keyword_scores, target_texts, style_texts, keyword_groups, output_dir):
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False

    # 计算需要的子图数量
    num_plots = 1  # target_texts 始终存在
    if len(style_texts) > 0:
        num_plots += 1
    if keyword_groups:
        num_plots += 1
    
    fig, axes = plt.subplots(1, num_plots, figsize=(10 * num_plots, 6))
    axes = np.atleast_1d(axes)

    num_variants = len(IMAGE_VARIANTS)
    width = 0.8 / num_variants
    offsets = (np.arange(num_variants) - (num_variants - 1) / 2) * width

    # Plot 1: Target texts
    target_array = np.arange(len(target_texts))
    ax = axes[0]
    for idx, (variant_key, variant_label) in enumerate(IMAGE_VARIANTS):
        data = np.asarray(target_scores[variant_key])[:len(target_texts)]
        ax.bar(target_array + offsets[idx], data, width, label=variant_label, alpha=0.8)
    ax.set_xlabel('Text Index', fontsize=12)
    ax.set_ylabel('Raw Logits (cosine * scale)', fontsize=12)
    ax.set_title(f'{image_name} - Target Text Scores', fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    ax.legend()

    plot_idx = 1
    
    # Plot 2: Style texts (if available)
    if len(style_texts) > 0:
        style_array = np.arange(len(style_texts))
        ax_style = axes[plot_idx]
        for idx, (variant_key, variant_label) in enumerate(IMAGE_VARIANTS):
            data = np.asarray(style_scores[variant_key])[:len(style_texts)]
            ax_style.bar(style_array + offsets[idx], data, width, label=variant_label, alpha=0.8)
        ax_style.set_xlabel('Style Text Index', fontsize=12)
        ax_style.set_ylabel('Raw Logits (cosine * scale)', fontsize=12)
        ax_style.set_title(f'{image_name} - Style Text Scores', fontsize=14)
        ax_style.grid(axis='y', alpha=0.3)
        ax_style.legend()
        plot_idx += 1

    # Plot 3: Keyword groups (if available)
    if keyword_groups:
        keyword_array = np.arange(len(keyword_groups))
        ax_keyword = axes[plot_idx]
        for idx, (variant_key, variant_label) in enumerate(IMAGE_VARIANTS):
            values = [keyword_scores[group_name][variant_key] for group_name in keyword_groups.keys()]
            ax_keyword.bar(keyword_array + offsets[idx], values, width, label=variant_label, alpha=0.8)
        ax_keyword.set_xlabel('Keyword Group', fontsize=12)
        ax_keyword.set_ylabel('Raw Logits (cosine * scale)', fontsize=12)
        ax_keyword.set_title(f'{image_name} - Keyword Test Scores', fontsize=14)
        ax_keyword.set_xticks(keyword_array)
        ax_keyword.set_xticklabels([name.split(': ')[1] for name in keyword_groups.keys()], rotation=45, ha='right')
        ax_keyword.grid(axis='y', alpha=0.3)
        ax_keyword.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01_text_probabilities.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_subset_scores(image_name, subset_scores, output_dir):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    num_variants = len(IMAGE_VARIANTS)
    width = 0.8 / num_variants
    offsets = (np.arange(num_variants) - (num_variants - 1) / 2) * width
    subset_indices = np.arange(len(subset_scores))

    for idx, (variant_key, variant_label) in enumerate(IMAGE_VARIANTS):
        values = [subset_scores[i][variant_key] for i in range(len(subset_scores))]
        ax.bar(subset_indices + offsets[idx], values, width, label=variant_label, alpha=0.8)

    ax.set_xlabel('Subset Index', fontsize=12)
    ax.set_ylabel('Mean Text-Image Score', fontsize=12)
    ax.set_title(f'{image_name} - Subset Mean Scores', fontsize=14)
    ax.set_xticks(subset_indices)
    ax.set_xticklabels([f'Subset {i+1}' for i in range(len(subset_scores))])
    ax.grid(axis='y', alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '02_subset_mean_scores.png'), dpi=150)
    plt.close()


def save_score_details(image_name, target_scores, style_scores, keyword_scores, subset_scores, target_texts, style_texts, keyword_groups, output_dir):
    file_path = os.path.join(output_dir, 'scores.txt')
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(f"Sample: {image_name}\n")
        f.write("=" * 80 + "\n\n")

        f.write("Target Text Raw Logits (cosine * scale)\n")
        f.write("-" * 80 + "\n")
        for idx, text in enumerate(target_texts):
            f.write(f"[{idx:03d}] {text}\n")
            for key, label in IMAGE_VARIANTS:
                data = np.asarray(target_scores[key])
                value = data[idx] if idx < len(data) else 0.0
                f.write(f"  {label:<28}: {value:.6f}\n")
        f.write("\n")

        if len(style_texts) > 0:
            f.write("Style Text Raw Logits (cosine * scale)\n")
            f.write("-" * 80 + "\n")
            for idx, text in enumerate(style_texts):
                f.write(f"[{idx:03d}] {text}\n")
                for key, label in IMAGE_VARIANTS:
                    data = np.asarray(style_scores[key])
                    value = data[idx] if idx < len(data) else 0.0
                    f.write(f"  {label:<28}: {value:.6f}\n")
            f.write("\n")

        if keyword_groups:
            f.write("Keyword Test Groups Raw Logits (cosine * scale)\n")
            f.write("-" * 80 + "\n")
            for group_name, texts in keyword_groups.items():
                f.write(f"{group_name}: {texts[0]}\n")
                for key, label in IMAGE_VARIANTS:
                    value = keyword_scores[group_name][key]
                    f.write(f"  {label:<28}: {value:.6f}\n")
            f.write("\n")

        f.write("Subset Mean Scores (cosine * scale)\n")
        f.write("-" * 80 + "\n")
        for idx, scores in enumerate(subset_scores):
            f.write(f"Subset {idx + 1}\n")
            for key, label in IMAGE_VARIANTS:
                f.write(f"  {label:<28}: {scores[key]:.6f}\n")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load model
    print(f"Loading BioMedCLIP from: {BIOMEDCLIP_PATH}")
    model, preprocess, tokenizer = create_biomedclip_model_and_preprocess_local(BIOMEDCLIP_PATH, device)
    model.eval()
    scale = model.logit_scale.exp().detach().item()

    # 2. Load texts
    print(f"Loading texts from: {TEXT_ROOT_PATH}")
    sampler = TextSampler(TEXT_ROOT_PATH)
    targets_texts_dict, style_texts, flat_texts = sampler.load_texts(DATASET_NAME, LLM_NAME, DESCRIBE_NUMS)
    per_type_subsets, flat_subsets = sampler.sample_subsets(targets_texts_dict, NUM_SAMPLES)
    
    print(f"  - Target texts: {len(flat_texts)}")
    print(f"  - Style texts: {len(style_texts)}")
    print(f"  - Subsets: {len(flat_subsets)}")
    print(f"  - Keyword test groups: {len(keyword_test_groups)}")

    # Pre-encode text features for reuse
    target_text_features = encode_text_batch(model, tokenizer, flat_texts, device)
    style_text_features = encode_text_batch(model, tokenizer, style_texts, device)
    subset_mean_features = compute_subset_mean_features(model, tokenizer, flat_subsets, device)
    
    # Pre-encode keyword test group features
    keyword_text_features = {}
    for group_name, texts in keyword_test_groups.items():
        keyword_text_features[group_name] = encode_text_batch(model, tokenizer, texts, device)

    # 3. Process image-mask pairs
    print(f"\nProcessing {len(all_images)} image-mask pairs...")
    processed = 0
    
    for img_idx, (img_path, mask_path) in enumerate(zip(all_images, all_masks)):
        print(f"  [{img_idx+1}/{len(all_images)}] {os.path.basename(img_path)}")
        
        try:
            x, x_c, x_c2, x_err = load_and_preprocess_images(img_path, mask_path, preprocess)
            images_dict = {'x': x, 'x_c': x_c, 'x_c2': x_c2, 'x_err': x_err}

            # Compute image features for each variant
            image_features = compute_image_features(model, images_dict, device)

            target_scores = {}
            style_scores = {key: np.array([]) for key in VARIANT_KEYS}
            keyword_scores = {group_name: {} for group_name in keyword_test_groups.keys()}
            
            for key in images_dict.keys():
                # Target texts
                raw_scores_targets = compute_text_logits(image_features[key], target_text_features, scale)
                target_scores[key] = raw_scores_targets.numpy()
                
                # Style texts
                if style_text_features.numel() > 0:
                    raw_scores_style = compute_text_logits(image_features[key], style_text_features, scale)
                    style_scores[key] = raw_scores_style.numpy()
                
                # Keyword test groups
                for group_name, group_features in keyword_text_features.items():
                    if group_features.numel() > 0:
                        raw_scores_keyword = compute_text_logits(image_features[key], group_features, scale)
                        # 取第一个值（因为每组只有一个文本）
                        keyword_scores[group_name][key] = raw_scores_keyword[0].item()
                    else:
                        keyword_scores[group_name][key] = 0.0

            for key in VARIANT_KEYS:
                if key not in target_scores:
                    target_scores[key] = np.zeros(len(flat_texts), dtype=np.float32)
                if key not in style_scores:
                    default_len = len(style_texts)
                    style_scores[key] = np.zeros(default_len, dtype=np.float32) if default_len > 0 else np.array([])
                for group_name in keyword_test_groups.keys():
                    if key not in keyword_scores[group_name]:
                        keyword_scores[group_name][key] = 0.0

            # Compute mean scores against subset centroids
            subset_scores = []
            for subset_feat in subset_mean_features:
                if subset_feat is None:
                    subset_scores.append({key: 0.0 for key in VARIANT_KEYS})
                    continue
                scores = {}
                for key in images_dict.keys():
                    value = scale * (image_features[key] @ subset_feat.t()).item()
                    scores[key] = float(value)
                subset_scores.append(scores)

            # Export per-sample artifacts
            image_name = os.path.basename(img_path).replace('.png', '')
            sample_dir = os.path.join(OUTPUT_DIR, image_name)
            os.makedirs(sample_dir, exist_ok=True)

            plot_text_scores(image_name, target_scores, style_scores, keyword_scores, flat_texts, style_texts, keyword_test_groups, sample_dir)
            plot_subset_scores(image_name, subset_scores, sample_dir)
            save_score_details(image_name, target_scores, style_scores, keyword_scores, subset_scores, flat_texts, style_texts, keyword_test_groups, sample_dir)

            processed += 1
            
        except Exception as e:
            print(f"    Error: {e}")
            continue
    
    print(f"\nProcessed {processed} images. Results saved per sample in {OUTPUT_DIR}.")

    print("\nDone!")


if __name__ == "__main__":
    plt.switch_backend('Agg')
    main()
