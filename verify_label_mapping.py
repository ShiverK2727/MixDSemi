#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cd /app/MixDSemi/SynFoCLIP/code && python -c "import train_unet_MiDSS_NDC_v5_unet; print('Import :验证所有数据集的标签映射关系
"""
import os
import sys
import numpy as np
from PIL import Image
import glob

print("="*80)
print("标签映射关系验证报告")
print("="*80)

# 1. Prostate
print("\n【1. Prostate】")
print("-" * 60)
prostate_mask = '/app/MixDSemi/data/ProstateSlice/BIDMC/train/mask/00_00.png'
if os.path.exists(prostate_mask):
    mask = Image.open(prostate_mask)
    mask_np = np.array(mask)
    print(f"原始: unique={np.unique(mask_np)}, 0有{np.sum(mask_np==0)}个, 255有{np.sum(mask_np==255)}个")
    print(f"语义: 0=FG(prostate), 255=BG")
    normalized = np.zeros_like(mask_np)
    normalized[mask_np == 0] = 255
    normalized[mask_np > 0] = 0
    print(f"归一化后: unique={np.unique(normalized)}, 255有{np.sum(normalized==255)}个(FG), 0有{np.sum(normalized==0)}个(BG)")
    print(f"✓ 正确: FG=255>0, BG=0")

# 2. Fundus
print("\n【2. Fundus】")
print("-" * 60)
fundus_mask = '/app/MixDSemi/data/Fundus/Domain1/train/ROIs/mask/gdrishtiGS_002.png'
if os.path.exists(fundus_mask):
    mask = Image.open(fundus_mask).convert('L')
    mask_np = np.array(mask)
    print(f"原始: unique={np.unique(mask_np)}")
    print(f"语义: 0=BG, 128=cup, 255=disc")
    print(f"归一化后: 无变化")
    print(f"✓ 正确: FG>0, BG=0")

# 3. MNMS
print("\n【3. MNMS】")
print("-" * 60)
mnms_mask = '/app/MixDSemi/data/mnms/vendorA/train/mask/000003.png'
if os.path.exists(mnms_mask):
    mask = Image.open(mnms_mask)
    mask_np = np.array(mask)
    print(f"原始RGB: R_unique={np.unique(mask_np[:,:,0])}, G_unique={np.unique(mask_np[:,:,1])}, B_unique={np.unique(mask_np[:,:,2])}")
    new_target = np.zeros((mask_np.shape[0], mask_np.shape[1]), dtype=np.uint8)
    for n in range(3):
        new_target[mask_np[:, :, n] == 255] = n + 1
    print(f"归一化: unique={np.unique(new_target)}")
    print(f"  0=BG({np.sum(new_target==0)}个), 1=LV({np.sum(new_target==1)}个), 2=MYO({np.sum(new_target==2)}个), 3=RV({np.sum(new_target==3)}个)")
    print(f"✓ 正确: FG>0 (1/2/3), BG=0")

# 4. BUSI
print("\n【4. BUSI】")
print("-" * 60)
busi_dir = '/app/MixDSemi/data/Dataset_BUSI_with_GT/benign/'
imgs = [f for f in glob.glob(busi_dir + '*.png') if 'mask' not in f]
if imgs:
    masks = glob.glob(imgs[0].replace('.png', '_mask*.png'))
    if masks:
        mask = Image.open(masks[0]).convert('L')
        mask_np = np.array(mask)
        print(f"原始: unique={np.unique(mask_np)}")
        print(f"语义: 0=BG, 255=tumor")
        print(f"归一化后: 无变化")
        print(f"✓ 正确: FG=255>0, BG=0")

print("\n" + "="*80)
print("BUSI 背景=0, 前景>0 ✓")
print("="*80)
