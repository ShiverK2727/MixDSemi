"""
可视化训练数据加载和标签下采样过程
展示：原图、原始mask、下采样mask、softmax二值化mask、大于0二值化mask
"""
import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

import dataloaders.custom_transforms as tr
from dataloaders.dataloader_dc import (BUSISegmentation, FundusSegmentation,
                                        MNMSSegmentation, ProstateSegmentation)


def set_random_seed(seed: int) -> None:
    """设置随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_dataset_config(dataset: str) -> dict:
    """返回数据集特定配置"""
    if dataset == 'fundus':
        return {
            'train_data_path': '/app/MixDSemi/data/Fundus',
            'dataset_class': FundusSegmentation,
            'num_channels': 3,
            'patch_size': 256,
            'num_classes': 2,
            'min_v': 0.5,
            'max_v': 1.5,
            'fillcolor': 255,
            'domain_len': [50, 99, 320, 320],
            'class_text': ['background', 'optic cup', 'optic disc']
        }
    elif dataset == 'prostate':
        return {
            'train_data_path': '/app/MixDSemi/data/ProstateSlice',
            'dataset_class': ProstateSegmentation,
            'num_channels': 1,
            'patch_size': 384,
            'num_classes': 1,
            'min_v': 0.1,
            'max_v': 2,
            'fillcolor': 255,
            'domain_len': [225, 305, 136, 373, 338, 133],
            'class_text': ['background', 'prostate']
        }
    elif dataset == 'MNMS':
        return {
            'train_data_path': '/app/MixDSemi/data/mnms',
            'dataset_class': MNMSSegmentation,
            'num_channels': 1,
            'patch_size': 288,
            'num_classes': 3,
            'min_v': 0.1,
            'max_v': 2,
            'fillcolor': 0,
            'domain_len': [1030, 1342, 525, 550],
            'class_text': ['background', 'left ventricle', 'left ventricle myocardium', 'right ventricle']
        }
    elif dataset == 'BUSI':
        return {
            'train_data_path': '/app/MixDSemi/data/Dataset_BUSI_with_GT',
            'dataset_class': BUSISegmentation,
            'num_channels': 1,
            'patch_size': 256,
            'num_classes': 1,
            'min_v': 0.1,
            'max_v': 2,
            'fillcolor': 0,
            'domain_len': [350, 168],
            'class_text': ['background', 'breast tumor']
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """将Tensor转换为numpy数组用于可视化"""
    tensor = tensor.detach().cpu()
    if tensor.dim() == 3:
        if tensor.shape[0] == 1:
            return tensor[0].numpy()
        elif tensor.shape[0] == 3:
            return tensor.permute(1, 2, 0).numpy()
    elif tensor.dim() == 2:
        return tensor.numpy()
    return tensor.numpy()


def visualize_sample(
    image: torch.Tensor,
    mask_original: torch.Tensor,
    mask_downsampled: torch.Tensor,
    mask_softmax_binary: torch.Tensor,
    mask_gt_zero_binary: torch.Tensor,
    output_path: str,
    sample_name: str,
    num_classes: int,
    class_text: list
) -> None:
    """
    创建五面板可视化图并保存
    
    Args:
        image: 原始图像 [C, H, W]
        mask_original: 原始mask [H, W]
        mask_downsampled: 下采样后的mask [K, 14, 14]
        mask_softmax_binary: softmax后二值化的mask [14, 14]
        mask_gt_zero_binary: 大于0二值化的mask [14, 14]
        output_path: 输出文件路径
        sample_name: 样本名称
        num_classes: 类别数量
    """
    K_classes = num_classes + 1  # background + foreground classes
    
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))  # 增加行数以显示通道
    axes = axes.flatten()  # 展平以便索引
    
    # 1. 原始图像
    img_np = tensor_to_numpy(image)
    if img_np.ndim == 2 or (img_np.ndim == 3 and img_np.shape[-1] == 1):
        axes[0].imshow(img_np.squeeze(), cmap='gray')
    else:
        axes[0].imshow(img_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # 2. 原始Mask
    mask_orig_np = tensor_to_numpy(mask_original)
    im1 = axes[1].imshow(mask_orig_np, cmap='tab10', vmin=0, vmax=num_classes)
    axes[1].set_title(f'Original Mask\n(Shape: {mask_orig_np.shape})')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # 3. 下采样Mask (前景类的平均)
    # mask_downsampled 的形状为 [K, 14, 14]（K = num_classes + 1）。
    # 不能使用通用的 tensor_to_numpy()（它会把 C=3 的第一个维度当作 RGB 通道并做 permute），
    # 所以这里直接使用 .cpu().numpy() 保持原始维度顺序。
    mask_down_np = mask_downsampled.detach().cpu().numpy()  # [K, 14, 14]
    if num_classes == 1:
        # 单类分割：显示前景类
        mask_down_display = mask_down_np[1, :, :]
    else:
        # 多类分割：显示所有前景类的总和
        mask_down_display = mask_down_np[1:, :, :].sum(axis=0)
    
    im2 = axes[2].imshow(mask_down_display, cmap='viridis', vmin=0.0, vmax=1.0)
    axes[2].set_title(f'Downsampled Mask (AvgPool)\n(Shape: {mask_down_display.shape})')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    # 4. 每通道的 Argmax 二值化合集（以网格方式展示小方格，保持每个通道为 14x14）
    #    计算 mask_down 的 softmax (跨通道)，得到每个像素的 argmax 类别；
    #    然后为每个通道生成二值图 (argmax==c)，并以近似方形的网格拼接，便于可视化多个通道。
    mask_down_softmax_t = F.softmax(mask_downsampled, dim=0).detach().cpu().numpy()  # [K, 14, 14]
    mask_down_argmax = np.argmax(mask_down_softmax_t, axis=0)  # [14, 14]
    argmax_bins = [(mask_down_argmax == c).astype(float) for c in range(K_classes)]

    # 计算网格尺寸（尽量接近方形）
    grid_cols = int(np.ceil(np.sqrt(K_classes)))
    grid_rows = int(np.ceil(K_classes / grid_cols))
    tile_h, tile_w = argmax_bins[0].shape
    grid_h = grid_rows * tile_h
    grid_w = grid_cols * tile_w
    argmax_grid = np.zeros((grid_h, grid_w), dtype=float)
    for idx, tile in enumerate(argmax_bins):
        r = idx // grid_cols
        c = idx % grid_cols
        argmax_grid[r*tile_h:(r+1)*tile_h, c*tile_w:(c+1)*tile_w] = tile

    axes[3].imshow(argmax_grid, cmap='gray', vmin=0.0, vmax=1.0, interpolation='nearest', aspect='equal')
    axes[3].set_title('Per-channel Argmax Binaries\n(grid of 14x14 tiles)')
    axes[3].axis('off')

    # 5. 每通道的阈值二值化 (>0.5)，同样以网格方式展示
    thresh_bins = [(mask_down_softmax_t[c] > 0.5).astype(float) for c in range(K_classes)]
    thresh_grid = np.zeros((grid_h, grid_w), dtype=float)
    for idx, tile in enumerate(thresh_bins):
        r = idx // grid_cols
        c = idx % grid_cols
        thresh_grid[r*tile_h:(r+1)*tile_h, c*tile_w:(c+1)*tile_w] = tile

    axes[4].imshow(thresh_grid, cmap='gray', vmin=0.0, vmax=1.0, interpolation='nearest', aspect='equal')
    axes[4].set_title('Per-channel >0.5 Binaries\n(grid of 14x14 tiles)')
    axes[4].axis('off')
    
    # 6-10. 每个通道的下采样图像 (第二行)，以 14x14 方格清晰呈现
    for c in range(K_classes):
        channel_data = mask_down_np[c, :, :]
        ax = axes[5 + c]
        im = ax.imshow(channel_data, cmap='viridis', vmin=0.0, vmax=1.0, interpolation='nearest', aspect='equal')
        # 在标题中加入前景像素计数（>0.5）以便快速检查
        fg_count = int((mask_down_np[c] > 0.5).sum())
        ax.set_title(f'Channel {c} ({class_text[c]})\nFG>0.5: {fg_count}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 隐藏多余的轴（如果 K_classes < 5）
    for hide_idx in range(5 + K_classes, 10):
        axes[hide_idx].axis('off')
    
    plt.suptitle(f'Sample: {sample_name}', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved visualization to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="可视化训练数据加载和标签下采样过程"
    )
    parser.add_argument('--dataset', type=str, default='prostate',
                        choices=['fundus', 'prostate', 'MNMS', 'BUSI'],
                        help='数据集名称')
    parser.add_argument('--lb_domain', type=int, default=1,
                        help='有标记数据的域ID')
    parser.add_argument('--lb_num', type=int, default=20,
                        help='有标记样本数量')
    parser.add_argument('--num_samples', type=int, default=15,
                        help='可视化的样本数量')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='批量大小')
    parser.add_argument('--output_dir', type=str, default='/app/MixDSemi/SynFoCLIP/results/label_downsampling_vis',
                        help='输出可视化图像的目录')
    parser.add_argument('--seed', type=int, default=1337,
                        help='随机种子')
    parser.add_argument('--overwrite', action='store_true',
                        help='覆盖已存在的输出目录')
    return parser.parse_args()


def main():
    args = parse_args()
    set_random_seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    if len(os.listdir(args.output_dir)) > 0 and not args.overwrite:
        raise RuntimeError(
            f"输出目录 {args.output_dir} 不为空。使用 --overwrite 覆盖现有文件。"
        )
    
    # 获取数据集配置
    cfg = get_dataset_config(args.dataset)
    
    # 数据增强
    weak = transforms.Compose([
        tr.RandomScaleCrop(cfg['patch_size']),
        tr.RandomScaleRotate(fillcolor=cfg['fillcolor']),
        tr.RandomHorizontalFlip(),
        tr.elastic_transform()
    ])
    
    normal_toTensor = transforms.Compose([
        tr.Normalize_tf(dataRange=[0, 1]),
        tr.ToTensor(unet_size=cfg['patch_size'])
    ])
    
    # 构建数据集
    lb_idxs = list(range(args.lb_num))
    
    lb_dataset = cfg['dataset_class'](
        base_dir=cfg['train_data_path'],
        phase='train',
        splitid=args.lb_domain,
        domain=[args.lb_domain],
        selected_idxs=lb_idxs,
        weak_transform=weak,
        normal_toTensor=normal_toTensor,
        img_size=cfg['patch_size'],
        is_RGB=(cfg['num_channels'] == 3)
    )
    
    print(f"数据集: {args.dataset}")
    print(f"有标记样本数量: {len(lb_dataset)}")
    print(f"类别数量: {cfg['num_classes']}")
    print(f"类别文本: {cfg['class_text']}")
    
    # 构建DataLoader
    lb_loader = DataLoader(
        lb_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = cfg['num_classes']
    K_classes = num_classes + 1  # background + foreground classes
    N_patches_side = 14
    
    processed = 0
    
    print(f"\n开始可视化 {args.num_samples} 个样本...\n")
    
    for batch_idx, batch in enumerate(lb_loader):
        if processed >= args.num_samples:
            break
        
        images = batch['unet_size_img'].to(device)
        labels = batch['unet_size_label']
        
        # 处理标签
        if isinstance(labels, torch.Tensor):
            if labels.dim() == 4 and labels.size(1) == 1:
                labels = labels.squeeze(1)
        else:
            labels = torch.as_tensor(labels)
        
        # ===== 根据数据集类型重映射标签（参考 train_synfoc.py）=====
        # Prostate: 原始 mask 中 0=前景，需要映射为 mask.eq(0) -> 1 (foreground)
        if args.dataset == 'prostate':
            labels_mask = labels.eq(0).long().to(device)  # 0->1 (foreground), other->0 (background)
        elif args.dataset == 'BUSI':
            labels_mask = labels.eq(255).long().to(device)  # 255->1 (foreground)
        elif args.dataset == 'MNMS':
            labels_mask = labels.long().to(device)  # 已经是类索引
        elif args.dataset == 'fundus':
            # Fundus 有 3 个值 (0, 128, 255)，需要特殊处理
            labels_mask = (labels <= 128).long().to(device) * 2
            labels_mask[labels == 0] = 1
        else:
            labels_mask = labels.long().to(device)
        
        B, H, W = labels_mask.shape
        
        # ===== 验证标签已正确映射为类索引 {0, 1, ...} =====
        unique_labels = torch.unique(labels_mask).cpu().tolist()
        print(f"  [Batch {batch_idx}] Original labels: {torch.unique(labels).cpu().tolist()} -> Mapped: {unique_labels}")
        
        # 构建 one-hot (B, K, H, W)
        one_hot = torch.zeros((B, K_classes, H, W), dtype=torch.float32, device=device)
        for c in range(K_classes):
            one_hot[:, c, :, :] = (labels_mask == c).float()
        
        # 下采样到 14x14
        label_14 = F.adaptive_avg_pool2d(one_hot, (N_patches_side, N_patches_side))  # [B, K, 14, 14]
        
        # 对batch中的每个样本进行可视化
        for i in range(B):
            if processed >= args.num_samples:
                break
            
            # 获取图像名称
            if isinstance(batch['img_name'], list):
                img_name = batch['img_name'][i]
            else:
                img_name = batch['img_name']
                if isinstance(img_name, list):
                    img_name = img_name[i]
            
            # 提取当前样本
            img = images[i]  # [C, H, W]
            mask_orig = labels_mask[i]  # [H, W] - 已映射的 mask
            mask_down = label_14[i]  # [K, 14, 14]
            
            # 生成 softmax 二值化 mask
            # 对下采样的mask应用softmax，然后取argmax得到类别，再生成前景二值化
            mask_down_softmax = F.softmax(mask_down, dim=0)  # [K, 14, 14]
            mask_down_argmax = torch.argmax(mask_down_softmax, dim=0)  # [14, 14]
            mask_softmax_binary = (mask_down_argmax > 0).float()  # 前景为1，背景为0
            
            # 生成大于0二值化 mask
            # 将所有前景类的概率相加，大于0的位置为前景
            if num_classes == 1:
                mask_fg_sum = mask_down[1, :, :]  # 单类：只有一个前景类
            else:
                mask_fg_sum = mask_down[1:, :, :].sum(dim=0)  # 多类：所有前景类求和
            
            mask_gt_zero_binary = (mask_fg_sum > 0).float()  # [14, 14]
            
            # 生成输出文件名
            stem = os.path.splitext(str(img_name))[0]
            output_path = os.path.join(args.output_dir, f"{stem}_vis.png")
            
            # 可视化
            visualize_sample(
                image=img,
                mask_original=mask_orig,
                mask_downsampled=mask_down,
                mask_softmax_binary=mask_softmax_binary,
                mask_gt_zero_binary=mask_gt_zero_binary,
                output_path=output_path,
                sample_name=img_name,
                num_classes=num_classes,
                class_text=cfg['class_text']
            )
            
            processed += 1
            
            # 打印统计信息
            print(f"[{processed}/{args.num_samples}] {img_name}")
            print(f"  - 原始mask唯一值: {torch.unique(mask_orig).cpu().tolist()}")
            print(f"  - 下采样mask范围: [{mask_down.min():.4f}, {mask_down.max():.4f}]")
            print(f"  - Softmax二值化: 前景像素 = {mask_softmax_binary.sum().item():.0f}/{N_patches_side*N_patches_side}")
            print(f"  - GT-Zero二值化: 前景像素 = {mask_gt_zero_binary.sum().item():.0f}/{N_patches_side*N_patches_side}\n")
    
    print(f"\n完成！共可视化 {processed} 个样本。")
    print(f"结果保存在: {args.output_dir}")


if __name__ == '__main__':
    main()
