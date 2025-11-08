#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize a single batch of multi-patch dataset outputs.
Saves one PNG per sample in the batch under results/patch_batch_viz.

Usage: run from /app/MixDSemi/SynFoCLIP/code with the conda env used earlier.
"""
import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image, ImageDraw
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# All datasets with patch_sampler support are in pretrain_dataloader.py
from dataloaders.pretrain_dataloader import ProstateSegmentation, FundusSegmentation, MNMSSegmentation, BUSISegmentation
from dataloaders import custom_transforms as tr


def custom_collate(batch):
    batched = {}
    for key in batch[0].keys():
        if key in ['image', 'label', 'patch_labels', 'strong_aug']:
            batched[key] = torch.stack([sample[key] for sample in batch], dim=0)
        elif key == 'num_patches':
            batched[key] = torch.tensor([sample[key] for sample in batch])
        elif key in ['img_name', 'dc']:
            batched[key] = [sample[key] for sample in batch]
        else:
            try:
                batched[key] = torch.stack([sample[key] for sample in batch], dim=0)
            except Exception:
                batched[key] = [sample[key] for sample in batch]
    return batched


def save_batch_visualizations(batch, out_dir, denorm=True, dataset=None):
    # batch: dict from dataset collate (batch, num_patches, ...)
    os.makedirs(out_dir, exist_ok=True)
    images = batch['image']  # [B, P, C, H, W]
    masks = batch['label']   # [B, P, H, W]
    labels = batch['patch_labels']  # [B, P]
    batch_size, num_patches = images.shape[0], images.shape[1]
    has_strong = 'strong_aug' in batch.keys()
    has_orig = 'orig_image' in batch.keys()

    # number of downsampled patches per side used for the small one-hot visualization
    N_patches_side = 28

    for i in range(batch_size):
        # rows: orig(if any), weak, strong(if any), mask
        rows = 1 + 1 + (1 if has_strong else 0) + 1 if has_orig else 1 + (1 if has_strong else 0) + 1
        # simpler: if has_orig -> 4 rows when strong exists else 3; if no orig -> 3 or 2
        if has_orig:
            fig_h_base = 4 if has_strong else 3
        else:
            fig_h_base = 3 if has_strong else 2

        # Determine per-sample unique mask values across all patches to decide how many
        # one-hot channels (classes) we will visualize. We add one row per class.
        try:
            sample_masks = masks[i]  # shape: [P, H, W]
            unique_vals = np.unique(sample_masks)
            unique_vals = np.sort(unique_vals)
            K_classes = int(len(unique_vals))
        except Exception:
            unique_vals = np.array([0])
            K_classes = 1

        fig_h = fig_h_base + K_classes
        fig_w = num_patches
        fig, axes = plt.subplots(fig_h, num_patches, figsize=(fig_w * 3, fig_h * 3))
        if num_patches == 1:
            axes = axes.reshape(fig_h, 1)

        # Pre-allocate storage for per-patch hardened one-hot (downsampled) maps
        hard_maps_per_patch = np.zeros((num_patches, K_classes, N_patches_side, N_patches_side), dtype=np.float32)

        for p in range(num_patches):
            row_idx = 0
            # original (if present) — 显示归一化前的 patch (已裁剪并缩放)
            if has_orig:
                # batch['orig_image'][i, p] 已经是裁剪后缩放到 img_size 的 patch
                # 它经过了 Normalize_tf + ToTensor,值范围是 [-1, 1]
                orig_tensor = batch['orig_image'][i, p].cpu().numpy()
                
                # 转换维度: C,H,W -> H,W,C (for color) or H,W (for grayscale)
                if orig_tensor.ndim == 3 and orig_tensor.shape[0] == 1:
                    orig_img = orig_tensor[0]  # 单通道: (1,H,W) -> (H,W)
                elif orig_tensor.ndim == 3:
                    orig_img = orig_tensor.transpose(1, 2, 0)  # RGB: (3,H,W) -> (H,W,3)
                else:
                    orig_img = orig_tensor
                
                # 反归一化: [-1,1] -> [0,1]
                if denorm:
                    orig_img = (orig_img + 1.0) / 2.0
                    orig_img = np.clip(orig_img, 0, 1)
                
                axes[row_idx, p].imshow(orig_img, cmap='gray' if orig_img.ndim == 2 else None)
                axes[row_idx, p].set_title(f'Patch {p} - Orig - Label: {int(labels[i, p].item())}')
                axes[row_idx, p].axis('off')
                row_idx += 1
            # weak / main image (after Normalize_tf + ToTensor)
            img_tensor = images[i, p].cpu().numpy()
            # img_tensor: C,H,W or 1,H,W
            if img_tensor.ndim == 3 and img_tensor.shape[0] == 1:
                img = img_tensor[0]
            elif img_tensor.ndim == 3:
                img = img_tensor.transpose(1, 2, 0)
            else:
                img = img_tensor
            if denorm:
                img = np.clip(img, 0, 1)
            axes[row_idx, p].imshow(img, cmap='gray' if img.ndim == 2 else None)
            axes[row_idx, p].set_title(f'Patch {p} - Weak - Label: {int(labels[i, p].item())}')
            axes[row_idx, p].axis('off')
            row_idx += 1

            # strong augmentation (if present)
            if has_strong:
                strong_tensor = batch['strong_aug'][i, p].cpu().numpy()
                if strong_tensor.ndim == 3 and strong_tensor.shape[0] == 1:
                    strong_img = strong_tensor[0]
                elif strong_tensor.ndim == 3:
                    strong_img = strong_tensor.transpose(1, 2, 0)
                else:
                    strong_img = strong_tensor
                if denorm:
                    strong_img = np.clip(strong_img, 0, 1)
                axes[row_idx, p].imshow(strong_img, cmap='gray' if strong_img.ndim == 2 else None)
                axes[row_idx, p].set_title(f'Patch {p} - Strong')
                axes[row_idx, p].axis('off')
                row_idx += 1

            # mask row (last row)
            mask = masks[i, p].cpu().numpy()
            # --- Build per-patch one-hot using the sample-wide unique values and
            #     downsample -> harden into a one-hot 14x14 map per channel.
            try:
                Hm, Wm = mask.shape
                one_hot_np = np.zeros((K_classes, Hm, Wm), dtype=np.float32)
                for ci, v in enumerate(unique_vals):
                    one_hot_np[ci] = (mask == v).astype(np.float32)

                # Torch ops for adaptive average pooling and hardening (matches train logic)
                one_hot_t = torch.from_numpy(one_hot_np).unsqueeze(0)  # [1, K, H, W]
                with torch.no_grad():
                    blurry = F.adaptive_avg_pool2d(one_hot_t, (N_patches_side, N_patches_side))  # [1, K, 14, 14]
                    hard_idx = torch.argmax(blurry, dim=1)  # [1, 14, 14]
                    hard_one_hot = torch.zeros_like(blurry)
                    hard_one_hot.scatter_(1, hard_idx.unsqueeze(1), 1.0)

                hard_maps_per_patch[p] = hard_one_hot.squeeze(0).cpu().numpy()
            except Exception:
                # If anything fails, leave zeros (no extra visualization)
                hard_maps_per_patch[p] = np.zeros((K_classes, N_patches_side, N_patches_side), dtype=np.float32)
            # print per-patch unique mask values if provided by dataset
            if 'patch_mask_uniques' in batch:
                try:
                    uniques = batch['patch_mask_uniques'][i][p]
                    print(f'[VIS] Batch sample {i} patch {p} mask uniques: {uniques}')
                except Exception:
                    uniques = None
            else:
                uniques = None
            # Auto-adjust vmax based on actual mask values for proper visualization
            mask_max = mask.max()
            if mask_max <= 10:  # MNMS-like (0-3)
                vmax = mask_max
            else:  # Prostate/Fundus/BUSI (0-255)
                vmax = 255
            axes[row_idx, p].imshow(mask, cmap='gray', vmin=0, vmax=vmax)
            title_add = f', uniques={uniques}' if uniques is not None else ''
            axes[row_idx, p].set_title(f'Mask {p} (白=FG, max={mask_max:.0f}{title_add})')
            axes[row_idx, p].axis('off')
        # --- Now plot the per-class hardened downsampled maps as extra rows under the mask ---
        mask_row_idx = fig_h_base - 1
        for ci in range(K_classes):
            class_val = unique_vals[ci] if len(unique_vals) > 0 else ci
            for p in range(num_patches):
                ax = axes[mask_row_idx + 1 + ci, p]
                arr = hard_maps_per_patch[p, ci]
                # visualize binary one-hot channel (0/1) — use vmax=1
                ax.imshow(arr, cmap='gray', vmin=0, vmax=1)
                ax.set_title(f'Class {int(class_val)} - ch {ci}')
                ax.axis('off')

        plt.tight_layout()
        save_path = os.path.join(out_dir, f'batch_sample_{i}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Saved visualization: {save_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='prostate', choices=['fundus', 'prostate', 'MNMS', 'BUSI'])
    parser.add_argument('--out-dir', type=str, default='../../results/patch_batch_viz', help='output directory (relative to code/)')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--patch-size', type=int, default=None, help='Patch size (auto-set if None)')
    parser.add_argument('--lb-domain', type=int, default=1)
    parser.add_argument('--lb-num', type=int, default=20)
    parser.add_argument('--debug-mask-uniques', action='store_true', help='print per-patch mask unique values from dataset')
    args = parser.parse_args()

    # Create dataset-specific output directory to avoid overwriting
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), args.out_dir, args.dataset))
    os.makedirs(out_dir, exist_ok=True)

    # Dataset-specific configurations
    if args.dataset == 'fundus':
        base_dir = '/app/MixDSemi/data/Fundus'
        dataset_class = FundusSegmentation
        num_channels = 3
        patch_size = args.patch_size if args.patch_size else 256
        is_RGB = True
        # mask background for Fundus is 0 (background), so use 0 as fillcolor for mask ops
        fillcolor = 0
        min_v, max_v = 0.5, 1.5
    elif args.dataset == 'prostate':
        base_dir = '/app/MixDSemi/data/ProstateSlice'
        dataset_class = ProstateSegmentation
        num_channels = 1
        patch_size = args.patch_size if args.patch_size else 384
        is_RGB = False
        # after normalization prostate masks use 0 as background and 1 for foreground
        fillcolor = 0
        min_v, max_v = 0.1, 2.0
    elif args.dataset == 'MNMS':
        base_dir = '/app/MixDSemi/data/mnms'
        dataset_class = MNMSSegmentation
        num_channels = 1
        patch_size = args.patch_size if args.patch_size else 288
        is_RGB = False
        fillcolor = 0
        min_v, max_v = 0.1, 2.0
    elif args.dataset == 'BUSI':
        base_dir = '/app/MixDSemi/data/Dataset_BUSI_with_GT'
        dataset_class = BUSISegmentation
        num_channels = 1
        patch_size = args.patch_size if args.patch_size else 256
        is_RGB = False
        fillcolor = 0
        min_v, max_v = 0.1, 2.0
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # transforms
    weak = transforms.Compose([
        tr.RandomScaleCrop(patch_size),
        tr.RandomScaleRotate(fillcolor=fillcolor),
        tr.RandomHorizontalFlip(),
        tr.elastic_transform(),
        # tr.AdaptiveCLAHE(p=0.5, clipLimit=5.0, tileGridSize=(8, 8))
    ])
    strong = transforms.Compose([
        tr.Brightness(min_v, max_v),
        tr.Contrast(min_v, max_v),
        tr.GaussianBlur(kernel_size=int(0.1 * patch_size), num_channels=num_channels),
        tr.AdaptiveCLAHERandomized(p=0.5, clipLimit_range=(4.0, 8.0), tileGridSize_range=(4, 8)),
    ])
    normal_toTensor = transforms.Compose([
        tr.Normalize_tf(dataRange=[0,1]),
        tr.ToTensor(unet_size=patch_size)
    ])

    # patch sampler
    patch_sampler = tr.RandomPatchSamplerWithClass(
        num_patches=4,
        num_fg=2,
        min_ratio=0.3,
        fg_threshold=0.05,
        num_attempts=15,
        return_coords=True
    )

    lb_idxs = list(range(args.lb_num))

    dataset = dataset_class(
        base_dir=base_dir,
        phase='train',
        splitid=args.lb_domain,
        domain=[args.lb_domain],
        selected_idxs=lb_idxs,
        weak_transform=weak,
        strong_tranform=strong,
        normal_toTensor=normal_toTensor,
        img_size=patch_size,
        is_RGB=is_RGB,
        patch_sampler=patch_sampler,
        debug_patch_uniques=args.debug_mask_uniques
    )

    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=custom_collate)

    # take a single batch
    batch = next(iter(loader))
    print(f"Dataset: {args.dataset}, Patch size: {patch_size}, Batch size: {args.batch_size}")
    print(f"Batch keys: {batch.keys()}")
    print(f"Image shape: {batch['image'].shape}, Label shape: {batch['label'].shape}")
    save_batch_visualizations(batch, out_dir, denorm=True, dataset=dataset)


if __name__ == '__main__':
    main()
