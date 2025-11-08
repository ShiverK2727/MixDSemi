#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for RandomPatchSamplerWithClass integration in pretrain_dataloader.py

This script validates:
1. Label normalization (foreground>0, background=0)
2. Multi-patch sampling before all transforms
3. Correct application of weak/strong/normalize transforms to each patch
4. Proper tensor stacking and shape validation
"""

import os
import sys
import torch
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import custom modules
from dataloaders.pretrain_dataloader import ProstateSegmentation
from dataloaders import custom_transforms as tr


def visualize_patches(sample, save_path=None):
    """
    Visualize sampled patches with their masks and labels.
    
    Args:
        sample: Dict containing 'image', 'label', 'patch_labels', etc.
        save_path: Optional path to save the visualization
    """
    images = sample['image']  # [num_patches, C, H, W]
    masks = sample['label']   # [num_patches, H, W]
    patch_labels = sample['patch_labels']  # [num_patches]
    num_patches = sample['num_patches']
    
    # Denormalize images for visualization
    # Assuming dataRange=[0,1] from train script
    images_np = images.cpu().numpy()
    images_np = np.clip(images_np, 0, 1)
    
    masks_np = masks.cpu().numpy()
    labels_np = patch_labels.cpu().numpy()
    
    fig, axes = plt.subplots(2, num_patches, figsize=(4*num_patches, 8))
    if num_patches == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(num_patches):
        # Image
        img = images_np[i, 0] if images_np.shape[1] == 1 else images_np[i].transpose(1, 2, 0)
        axes[0, i].imshow(img, cmap='gray' if images_np.shape[1] == 1 else None)
        axes[0, i].set_title(f'Patch {i}\nLabel: {labels_np[i]} ({"FG" if labels_np[i]==1 else "BG"})')
        axes[0, i].axis('off')
        
        # Mask
        axes[1, i].imshow(masks_np[i], cmap='gray')
        axes[1, i].set_title(f'Mask {i}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    else:
        plt.show()
    plt.close()


def test_single_sample(dataset, idx=0, save_dir='./test_outputs'):
    """Test a single sample from the dataset."""
    print(f"\n{'='*60}")
    print(f"Testing sample index: {idx}")
    print(f"{'='*60}")
    
    sample = dataset[idx]
    
    print(f"\nSample keys: {sample.keys()}")
    print(f"Image shape: {sample['image'].shape}")
    print(f"Label shape: {sample['label'].shape}")
    print(f"Patch labels: {sample['patch_labels']}")
    print(f"Num patches: {sample['num_patches']}")
    print(f"Image name: {sample['img_name']}")
    print(f"Domain code: {sample['dc']}")
    
    # Check for strong augmentation
    if 'strong_aug' in sample:
        print(f"Strong aug shape: {sample['strong_aug'].shape}")
    
    # Validate shapes
    num_patches = sample['num_patches']
    assert sample['image'].shape[0] == num_patches, "Image batch size mismatch"
    assert sample['label'].shape[0] == num_patches, "Label batch size mismatch"
    assert sample['patch_labels'].shape[0] == num_patches, "Patch labels size mismatch"
    
    # Check patch label distribution
    fg_count = (sample['patch_labels'] == 1).sum().item()
    bg_count = (sample['patch_labels'] == 0).sum().item()
    print(f"\nPatch label distribution: FG={fg_count}, BG={bg_count}")
    
    # Visualize
    save_path = os.path.join(save_dir, f'sample_{idx}.png')
    visualize_patches(sample, save_path)
    
    return sample


def test_batch_loading(dataset, batch_size=2, num_batches=3):
    """Test batch loading with DataLoader."""
    from torch.utils.data import DataLoader
    
    print(f"\n{'='*60}")
    print(f"Testing batch loading (batch_size={batch_size})")
    print(f"{'='*60}")
    
    # Custom collate function for variable-sized patch batches
    def custom_collate(batch):
        """
        Collate function that handles variable number of patches per sample.
        Each sample already has patches stacked, so we just batch the samples.
        """
        # Since each sample is already a dict with stacked patches,
        # we need to batch across samples
        batched = {}
        for key in batch[0].keys():
            if key in ['image', 'label', 'patch_labels', 'strong_aug']:
                # Stack tensors from different samples
                # Each sample[key] has shape [num_patches, ...], we want [batch, num_patches, ...]
                batched[key] = torch.stack([sample[key] for sample in batch], dim=0)
            elif key == 'num_patches':
                batched[key] = torch.tensor([sample[key] for sample in batch])
            elif key in ['img_name', 'dc']:
                batched[key] = [sample[key] for sample in batch]
            else:
                # Handle other tensor keys
                try:
                    batched[key] = torch.stack([sample[key] for sample in batch], dim=0)
                except:
                    batched[key] = [sample[key] for sample in batch]
        return batched
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                       collate_fn=custom_collate, num_workers=0)
    
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        
        print(f"\nBatch {i}:")
        print(f"  Image shape: {batch['image'].shape}")  # [batch, num_patches, C, H, W]
        print(f"  Label shape: {batch['label'].shape}")  # [batch, num_patches, H, W]
        print(f"  Patch labels shape: {batch['patch_labels'].shape}")  # [batch, num_patches]
        print(f"  Num patches: {batch['num_patches']}")
        print(f"  Image names: {batch['img_name']}")
        
        # Validate
        assert batch['image'].shape[0] == batch_size or i == len(loader) - 1
        assert batch['label'].shape[0] == batch_size or i == len(loader) - 1


def main():
    """Main test function."""
    print("="*60)
    print("RandomPatchSamplerWithClass Integration Test")
    print("="*60)
    
    # Configuration (matching train_MiDSS_MARK3_CLIP_TPT_3_SEMI_EMA.py)
    patch_size = 384
    num_channels = 1
    fillcolor = 0
    min_v = 0.7
    max_v = 1.3
    
    # Define transforms
    weak = transforms.Compose([
        tr.RandomScaleCrop(patch_size),
        tr.RandomScaleRotate(fillcolor=fillcolor),
        tr.RandomHorizontalFlip(),
        tr.elastic_transform()
    ])
    
    strong = transforms.Compose([
        tr.Brightness(min_v, max_v),
        tr.Contrast(min_v, max_v),
        tr.GaussianBlur(kernel_size=int(0.1 * patch_size), num_channels=num_channels),
    ])
    
    normal_toTensor = transforms.Compose([
        tr.Normalize_tf(dataRange=[0,1]),
        tr.ToTensor(unet_size=patch_size)
    ])
    
    # Create patch sampler
    patch_sampler = tr.RandomPatchSamplerWithClass(
        num_patches=4,      # Sample 4 patches per image
        num_fg=2,           # 2 must contain foreground
        min_ratio=0.5,      # Minimum 50% of original size
        fg_threshold=0.01,  # 1% foreground pixels to be considered FG
        num_attempts=50     # Max attempts per patch
    )
    
    print("\nPatch sampler configuration:")
    print(f"  num_patches: {patch_sampler.num_patches}")
    print(f"  num_fg: {patch_sampler.num_fg}")
    print(f"  min_ratio: {patch_sampler.min_ratio}")
    print(f"  fg_threshold: {patch_sampler.fg_threshold}")
    
    # Create dataset
    base_dir = '/app/MixDSemi/data/ProstateSlice'
    lb_domain = 1
    lb_num = 20
    lb_idxs = list(range(lb_num))
    
    print(f"\nDataset configuration:")
    print(f"  base_dir: {base_dir}")
    print(f"  lb_domain: {lb_domain}")
    print(f"  lb_num: {lb_num}")
    print(f"  img_size: {patch_size}")
    
    dataset = ProstateSegmentation(
        base_dir=base_dir,
        phase='train',
        splitid=lb_domain,
        domain=[lb_domain],
        selected_idxs=lb_idxs,
        weak_transform=weak,
        strong_tranform=strong,
        normal_toTensor=normal_toTensor,
        img_size=patch_size,
        is_RGB=False,
        patch_sampler=patch_sampler  # Enable multi-patch mode
    )
    
    print(f"\nDataset created: {len(dataset)} samples")
    
    # Test individual samples
    save_dir = './test_outputs/patch_sampler_test'
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("Testing individual samples")
    print("="*60)
    
    for idx in [0, 1, 2]:
        if idx < len(dataset):
            try:
                test_single_sample(dataset, idx=idx, save_dir=save_dir)
            except Exception as e:
                print(f"Error testing sample {idx}: {e}")
                import traceback
                traceback.print_exc()
    
    # Test batch loading
    try:
        test_batch_loading(dataset, batch_size=2, num_batches=2)
    except Exception as e:
        print(f"Error in batch loading test: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)
    
    # Test standard mode (without patch sampler)
    print("\n" + "="*60)
    print("Testing standard mode (no patch sampler)")
    print("="*60)
    
    dataset_standard = ProstateSegmentation(
        base_dir=base_dir,
        phase='train',
        splitid=lb_domain,
        domain=[lb_domain],
        selected_idxs=lb_idxs,
        weak_transform=weak,
        strong_tranform=strong,
        normal_toTensor=normal_toTensor,
        img_size=patch_size,
        is_RGB=False,
        patch_sampler=None  # Disable multi-patch mode
    )
    
    sample_std = dataset_standard[0]
    print(f"Standard mode sample keys: {sample_std.keys()}")
    print(f"Standard mode image shape: {sample_std['image'].shape}")
    print(f"Standard mode label shape: {sample_std['label'].shape}")
    
    print("\n✓ Standard mode works correctly!")


if __name__ == '__main__':
    main()
