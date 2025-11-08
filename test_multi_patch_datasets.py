#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick test script to verify multi-patch support for all datasets.
"""
import os
import sys
import torch
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataloaders.dataloader import FundusSegmentation, ProstateSegmentation, MNMSSegmentation, BUSISegmentation
from dataloaders import custom_transforms as tr


def test_dataset(dataset_name, dataset_class, base_dir, img_size, num_channels, fillcolor, is_RGB=False):
    """Test a dataset with multi-patch sampling."""
    print(f"\n{'='*60}")
    print(f"Testing {dataset_name} dataset")
    print(f"{'='*60}")
    
    # Simple transforms
    weak = transforms.Compose([
        tr.RandomScaleCrop(img_size),
        tr.RandomHorizontalFlip(),
    ])
    strong = transforms.Compose([
        tr.Brightness(0.5, 1.5),
        tr.Contrast(0.5, 1.5),
    ])
    normal_toTensor = transforms.Compose([
        tr.Normalize_tf(dataRange=[0,1]),
        tr.ToTensor(unet_size=img_size)
    ])
    
    # Patch sampler
    patch_sampler = tr.RandomPatchSamplerWithClass(
        num_patches=4,
        num_fg=2,
        min_ratio=0.3,
        fg_threshold=0.05,
        num_attempts=15,
        return_coords=True
    )
    
    # Test with patch sampler
    print(f"\n[1] Testing WITH patch sampler (multi-patch mode)...")
    try:
        dataset_with_patches = dataset_class(
            base_dir=base_dir,
            phase='train',
            splitid=1,
            domain=[1],
            selected_idxs=list(range(5)),  # Just first 5 images
            weak_transform=weak,
            strong_tranform=strong,
            normal_toTensor=normal_toTensor,
            img_size=img_size,
            is_RGB=is_RGB,
            patch_sampler=patch_sampler
        )
        
        print(f"   Dataset length: {len(dataset_with_patches)}")
        sample = dataset_with_patches[0]
        print(f"   Sample keys: {sample.keys()}")
        print(f"   Image shape: {sample['image'].shape}")  # Should be [num_patches, C, H, W]
        print(f"   Label shape: {sample['label'].shape}")  # Should be [num_patches, H, W]
        print(f"   Patch labels shape: {sample['patch_labels'].shape}")  # Should be [num_patches]
        print(f"   Patch labels: {sample['patch_labels']}")
        print(f"   Num patches: {sample['num_patches']}")
        if 'patch_coords' in sample:
            print(f"   Patch coords present: {len(sample['patch_coords'])} coords")
        if 'orig_image' in sample:
            print(f"   Original image shape: {sample['orig_image'].shape}")
        print(f"   ✓ Multi-patch mode works!")
    except Exception as e:
        print(f"   ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test without patch sampler (standard mode)
    print(f"\n[2] Testing WITHOUT patch sampler (standard mode)...")
    try:
        dataset_standard = dataset_class(
            base_dir=base_dir,
            phase='train',
            splitid=1,
            domain=[1],
            selected_idxs=list(range(5)),
            weak_transform=weak,
            strong_tranform=strong,
            normal_toTensor=normal_toTensor,
            img_size=img_size,
            is_RGB=is_RGB,
            patch_sampler=None  # No patch sampler
        )
        
        print(f"   Dataset length: {len(dataset_standard)}")
        sample = dataset_standard[0]
        print(f"   Sample keys: {sample.keys()}")
        print(f"   Image shape: {sample['image'].shape}")  # Should be [C, H, W]
        print(f"   Label shape: {sample['label'].shape}")  # Should be [H, W]
        print(f"   ✓ Standard mode works!")
    except Exception as e:
        print(f"   ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    print("Testing multi-patch support for all datasets...")
    
    results = {}
    
    # Test Fundus
    results['fundus'] = test_dataset(
        dataset_name='Fundus',
        dataset_class=FundusSegmentation,
        base_dir='/app/MixDSemi/data/Fundus',
        img_size=256,
        num_channels=3,
        fillcolor=255,
        is_RGB=True
    )
    
    # Test Prostate
    results['prostate'] = test_dataset(
        dataset_name='Prostate',
        dataset_class=ProstateSegmentation,
        base_dir='/app/MixDSemi/data/ProstateSlice',
        img_size=384,
        num_channels=1,
        fillcolor=255,
        is_RGB=False
    )
    
    # Test MNMS
    results['mnms'] = test_dataset(
        dataset_name='MNMS',
        dataset_class=MNMSSegmentation,
        base_dir='/app/MixDSemi/data/mnms',
        img_size=288,
        num_channels=1,
        fillcolor=0,
        is_RGB=False
    )
    
    # Test BUSI
    results['busi'] = test_dataset(
        dataset_name='BUSI',
        dataset_class=BUSISegmentation,
        base_dir='/app/MixDSemi/data/Dataset_BUSI_with_GT',
        img_size=256,
        num_channels=1,
        fillcolor=0,
        is_RGB=False
    )
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for dataset_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {dataset_name:12s}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n✓ All datasets passed!")
    else:
        print("\n✗ Some datasets failed!")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
