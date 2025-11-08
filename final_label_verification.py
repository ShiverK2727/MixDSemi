#!/usr/bin/env python3
"""
最终验证: 确认所有数据集从 pretrain_dataloader 返回的标签是正确的
"""
import sys
sys.path.insert(0, '/app/MixDSemi/SynFoCLIP/code')

from dataloaders.pretrain_dataloader import (
    ProstateSegmentation, FundusSegmentation, 
    MNMSSegmentation, BUSISegmentation
)
from dataloaders.custom_transforms import Normalize_tf, ToTensor, RandomPatchSamplerWithClass
from torchvision import transforms as T
import numpy as np

print("="*80)
print("标签正确性最终验证")
print("="*80)

normalize_tf = T.Compose([Normalize_tf(), ToTensor()])
patch_sampler = RandomPatchSamplerWithClass(num_patches=4, num_fg=2)

# 1. Prostate
print("\n【1. Prostate】")
dataset = ProstateSegmentation(
    base_dir='/app/MixDSemi/data/ProstateSlice',
    phase='train', splitid=1, domain=[1],
    weak_transform=None, strong_tranform=None,
    normal_toTensor=normalize_tf,
    selected_idxs=list(range(2)),
    patch_sampler=patch_sampler
)
sample = dataset[0]
print(f"✓ Label shape: {sample['label'].shape}")
print(f"✓ Label unique: {sample['label'].unique()}")
print(f"✓ Label range: [{sample['label'].min():.1f}, {sample['label'].max():.1f}]")
print(f"✓ Expected: 0=BG, 255=FG (prostate)")
assert sample['label'].max() == 255.0, "Prostate FG should be 255!"
assert sample['label'].min() == 0.0, "Prostate BG should be 0!"
print("✅ PASS")

# 2. Fundus
print("\n【2. Fundus】")
dataset = FundusSegmentation(
    base_dir='/app/MixDSemi/data/Fundus',
    phase='train', splitid=1, domain=[1],
    weak_transform=None, strong_tranform=None,
    normal_toTensor=normalize_tf,
    selected_idxs=list(range(2)),
    patch_sampler=patch_sampler
)
sample = dataset[0]
print(f"✓ Label shape: {sample['label'].shape}")
print(f"✓ Label unique: {sample['label'].unique()}")
print(f"✓ Label range: [{sample['label'].min():.1f}, {sample['label'].max():.1f}]")
print(f"✓ Expected: 0=BG, 128=cup, 255=disc")
expected_values = {0.0, 128.0, 255.0}
actual_values = set(sample['label'].unique().tolist())
assert actual_values.issubset(expected_values), f"Unexpected values: {actual_values}"
print("✅ PASS")

# 3. MNMS
print("\n【3. MNMS】")
dataset = MNMSSegmentation(
    base_dir='/app/MixDSemi/data/mnms',
    phase='train', splitid=1, domain=[1],
    weak_transform=None, strong_tranform=None,
    normal_toTensor=normalize_tf,
    selected_idxs=list(range(2)),
    patch_sampler=patch_sampler
)
sample = dataset[0]
print(f"✓ Label shape: {sample['label'].shape}")
print(f"✓ Label unique: {sample['label'].unique()}")
print(f"✓ Label range: [{sample['label'].min():.1f}, {sample['label'].max():.1f}]")
print(f"✓ Expected: 0=BG, 1=LV, 2=MYO, 3=RV")
expected_values = {0.0, 1.0, 2.0, 3.0}
actual_values = set(sample['label'].unique().tolist())
assert actual_values.issubset(expected_values), f"Unexpected values: {actual_values}"
assert sample['label'].max() <= 3.0, "MNMS max should be 3!"
print("✅ PASS")

# 4. BUSI
print("\n【4. BUSI】")
dataset = BUSISegmentation(
    base_dir='/app/MixDSemi/data/Dataset_BUSI_with_GT',
    phase='train', splitid=1, domain=[1],
    weak_transform=None, strong_tranform=None,
    normal_toTensor=normalize_tf,
    selected_idxs=list(range(2)),
    patch_sampler=patch_sampler
)
sample = dataset[0]
print(f"✓ Label shape: {sample['label'].shape}")
print(f"✓ Label unique: {sample['label'].unique()}")
print(f"✓ Label range: [{sample['label'].min():.1f}, {sample['label'].max():.1f}]")
print(f"✓ Expected: 0=BG, 255=tumor")
expected_values = {0.0, 255.0}
actual_values = set(sample['label'].unique().tolist())
assert actual_values.issubset(expected_values), f"Unexpected values: {actual_values}"
print("✅ PASS")

print("\n" + "="*80)
print("✅ 所有数据集标签验证通过!")
print("="*80)
print("\n总结:")
print("1. Prostate: 255=前列腺(FG), 0=背景(BG) ✓")
print("2. Fundus: 255=视盘, 128=视杯, 0=背景 ✓")
print("3. MNMS: 1=左心室, 2=心肌, 3=右心室, 0=背景 ✓")
print("4. BUSI: 255=肿瘤, 0=背景 ✓")
print("\n所有数据集都满足: 背景=0, 前景>0")
print("="*80)
