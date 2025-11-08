#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify all 4 datasets (Prostate, Fundus, MNMS, BUSI) 
work correctly with patch_sampler when loaded from pretrain_dataloader.py
"""
import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Add code directory to path
sys.path.insert(0, '/app/MixDSemi/SynFoCLIP/code')

from dataloaders.pretrain_dataloader import (
    ProstateSegmentation, 
    FundusSegmentation, 
    MNMSSegmentation, 
    BUSISegmentation
)
from dataloaders import custom_transforms as tr
from torchvision import transforms as T

# RandomPatchSamplerWithClass definition
class RandomPatchSamplerWithClass:
    """
    Sample multiple patches from an image with class-aware sampling.
    Ensures at least one foreground patch and distributes remaining patches.
    
    Args:
        num_patches (int): Total number of patches to sample per image
        patch_size (int): Size of each square patch (default: 128)
        fillcolor (int): Fill color for padding if patch extends beyond image boundary
    
    Behavior:
        - First patch: Randomly samples from foreground (label > 0)
        - Remaining patches: 50% foreground, 50% background
        - If no foreground exists, samples background patches only
        - Returns: patches (images), patch_masks (labels), patch_labels (binary: 0=BG, 1=FG)
    """
    def __init__(self, num_patches=4, patch_size=128, fillcolor=0):
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.fillcolor = fillcolor

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        img_width, img_height = image.size
        label_np = np.array(label)
        
        # Identify foreground and background pixel coordinates
        fg_coords = np.argwhere(label_np > 0)  # Assumes foreground > 0, background = 0
        bg_coords = np.argwhere(label_np == 0)
        
        patches = []
        patch_masks = []
        patch_labels = []
        patch_coords = []
        
        for i in range(self.num_patches):
            if i == 0 and len(fg_coords) > 0:
                # First patch: always sample from foreground
                center_y, center_x = fg_coords[np.random.randint(len(fg_coords))]
                is_fg = 1
            elif i > 0 and len(fg_coords) > 0:
                # Remaining patches: 50% FG, 50% BG
                if np.random.rand() < 0.5:
                    center_y, center_x = fg_coords[np.random.randint(len(fg_coords))]
                    is_fg = 1
                else:
                    center_y, center_x = bg_coords[np.random.randint(len(bg_coords))]
                    is_fg = 0
            else:
                # No foreground available: sample from background
                center_y, center_x = bg_coords[np.random.randint(len(bg_coords))]
                is_fg = 0
            
            # Calculate patch bounding box
            half_size = self.patch_size // 2
            x1 = max(0, center_x - half_size)
            y1 = max(0, center_y - half_size)
            x2 = min(img_width, center_x + half_size)
            y2 = min(img_height, center_y + half_size)
            
            # Crop patch and mask
            patch = image.crop((x1, y1, x2, y2))
            patch_mask = label.crop((x1, y1, x2, y2))
            
            # Pad to patch_size if needed
            if patch.size != (self.patch_size, self.patch_size):
                padded_patch = Image.new(image.mode, (self.patch_size, self.patch_size), self.fillcolor)
                padded_mask = Image.new(label.mode, (self.patch_size, self.patch_size), 0)
                
                paste_x = (self.patch_size - patch.width) // 2
                paste_y = (self.patch_size - patch.height) // 2
                padded_patch.paste(patch, (paste_x, paste_y))
                padded_mask.paste(patch_mask, (paste_x, paste_y))
                
                patch = padded_patch
                patch_mask = padded_mask
            
            patches.append(patch)
            patch_masks.append(patch_mask)
            patch_labels.append(is_fg)
            patch_coords.append((center_x, center_y))
        
        sample['patches'] = patches
        sample['patch_masks'] = patch_masks
        sample['patch_labels'] = patch_labels
        sample['patch_coords'] = patch_coords
        return sample


def test_dataset(dataset_name, dataset, num_samples=2):
    """Test a dataset with patch sampling"""
    print(f"\n{'='*80}")
    print(f"Testing {dataset_name} with {len(dataset)} samples")
    print(f"{'='*80}")
    
    for i in range(min(num_samples, len(dataset))):
        try:
            sample = dataset[i]
            print(f"\n[Sample {i}] {sample['img_name']}")
            print(f"  Image shape: {sample['image'].shape}")
            print(f"  Label shape: {sample['label'].shape}")
            print(f"  Num patches: {sample['num_patches']}")
            print(f"  Patch labels: {sample['patch_labels']}")
            
            # Check label value range
            label_min = sample['label'].min().item()
            label_max = sample['label'].max().item()
            print(f"  Label value range: [{label_min:.1f}, {label_max:.1f}]")
            
            # Check if mask has foreground (expect 255 for FG after normalization)
            fg_pixels = (sample['label'] > 0).sum().item()
            total_pixels = sample['label'].numel()
            fg_ratio = fg_pixels / total_pixels * 100
            print(f"  Foreground pixels: {fg_pixels}/{total_pixels} ({fg_ratio:.2f}%)")
            
            # Verify patch_labels consistency
            for p_idx, p_label in enumerate(sample['patch_labels']):
                patch_mask = sample['label'][p_idx]
                has_fg = (patch_mask > 0).any().item()
                print(f"    Patch {p_idx}: label={p_label} (should be {1 if has_fg else 0}), has_fg={has_fg}")
            
            print(f"  ✓ Sample {i} loaded successfully")
            
        except Exception as e:
            print(f"  ✗ Error loading sample {i}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return True


def main():
    print("\n" + "="*80)
    print("Testing all datasets from pretrain_dataloader.py with patch_sampler")
    print("="*80)
    
    # Define transforms (use None for simplicity in testing)
    weak_transform = None
    
    normalize_tf = T.Compose([
        tr.Normalize_tf(),
        tr.ToTensor()
    ])
    
    # Create patch sampler
    patch_sampler = RandomPatchSamplerWithClass(
        num_patches=4,
        patch_size=128,
        fillcolor=255  # Will be adjusted per dataset
    )
    
    # Test all datasets
    results = {}
    
    # 1. Prostate (fillcolor=255, inverts: FG should be 255 after normalization)
    print("\n[1/4] Testing ProstateSegmentation...")
    patch_sampler.fillcolor = 255
    try:
        prostate_dataset = ProstateSegmentation(
            base_dir='/app/MixDSemi/data/ProstateSlice',
            phase='train',
            splitid=1,
            domain=[1,2,3,4,5,6],
            weak_transform=weak_transform,
            strong_tranform=None,
            normal_toTensor=normalize_tf,
            selected_idxs=None,
            img_size=384,
            is_RGB=False,
            patch_sampler=patch_sampler
        )
        results['Prostate'] = test_dataset('ProstateSegmentation', prostate_dataset, num_samples=2)
    except Exception as e:
        print(f"✗ Failed to initialize ProstateSegmentation: {e}")
        results['Prostate'] = False
    
    # 2. Fundus (fillcolor=255, no inversion: FG already >0)
    print("\n[2/4] Testing FundusSegmentation...")
    patch_sampler.fillcolor = 255
    try:
        fundus_dataset = FundusSegmentation(
            base_dir='/app/MixDSemi/data/Fundus',
            phase='train',
            splitid=1,
            domain=[1,2,3,4],
            weak_transform=weak_transform,
            strong_tranform=None,
            normal_toTensor=normalize_tf,
            selected_idxs=None,
            img_size=256,
            is_RGB=False,
            patch_sampler=patch_sampler
        )
        results['Fundus'] = test_dataset('FundusSegmentation', fundus_dataset, num_samples=2)
    except Exception as e:
        print(f"✗ Failed to initialize FundusSegmentation: {e}")
        results['Fundus'] = False
    
    # 3. MNMS (fillcolor=0, RGB->class indices: FG >0)
    print("\n[3/4] Testing MNMSSegmentation...")
    patch_sampler.fillcolor = 0
    try:
        mnms_dataset = MNMSSegmentation(
            base_dir='/app/MixDSemi/data/mnms',
            phase='train',
            splitid=1,
            domain=[1,2,3,4],
            weak_transform=weak_transform,
            strong_tranform=None,
            normal_toTensor=normalize_tf,
            selected_idxs=None,
            img_size=288,
            is_RGB=False,
            patch_sampler=patch_sampler
        )
        results['MNMS'] = test_dataset('MNMSSegmentation', mnms_dataset, num_samples=2)
    except Exception as e:
        print(f"✗ Failed to initialize MNMSSegmentation: {e}")
        results['MNMS'] = False
    
    # 4. BUSI (fillcolor=0, no inversion: FG=255)
    print("\n[4/4] Testing BUSISegmentation...")
    patch_sampler.fillcolor = 0
    try:
        busi_dataset = BUSISegmentation(
            base_dir='/app/MixDSemi/data/Dataset_BUSI_with_GT',
            phase='train',
            splitid=1,
            domain=[1,2],
            weak_transform=weak_transform,
            strong_tranform=None,
            normal_toTensor=normalize_tf,
            selected_idxs=None,
            img_size=256,
            is_RGB=False,
            patch_sampler=patch_sampler
        )
        results['BUSI'] = test_dataset('BUSISegmentation', busi_dataset, num_samples=2)
    except Exception as e:
        print(f"✗ Failed to initialize BUSISegmentation: {e}")
        results['BUSI'] = False
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    for dataset_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{dataset_name:20s}: {status}")
    print("="*80)
    
    # Check if all passed
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All datasets loaded successfully from pretrain_dataloader.py!")
        return 0
    else:
        print("\n⚠ Some datasets failed. See errors above.")
        return 1


if __name__ == '__main__':
    exit(main())
