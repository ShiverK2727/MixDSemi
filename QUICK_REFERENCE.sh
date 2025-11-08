#!/bin/bash
# Quick reference for using multi-patch datasets

echo "Multi-Patch Dataset Quick Reference"
echo "===================================="
echo ""

echo "1. TEST ALL DATASETS"
echo "-------------------"
echo "cd /app/MixDSemi/SynFoCLIP/code"
echo "python test_multi_patch_datasets.py"
echo ""

echo "2. VISUALIZE PATCHES"
echo "-------------------"
echo "# Fundus (RGB, 256x256)"
echo "python visualize_patches_batch.py --dataset fundus --batch-size 2 --lb-num 10"
echo ""
echo "# Prostate (Gray, 384x384)"
echo "python visualize_patches_batch.py --dataset prostate --batch-size 2 --lb-num 10"
echo ""
echo "# MNMS (Gray, 288x288)"
echo "python visualize_patches_batch.py --dataset MNMS --batch-size 2 --lb-num 10"
echo ""
echo "# BUSI (Gray, 256x256)"
echo "python visualize_patches_batch.py --dataset BUSI --batch-size 2 --lb-num 10"
echo ""

echo "3. PYTHON CODE EXAMPLE"
echo "---------------------"
cat << 'EOF'
from dataloaders.dataloader import FundusSegmentation
from dataloaders import custom_transforms as tr
from torchvision import transforms

# Setup transforms
weak = transforms.Compose([
    tr.RandomScaleCrop(256),
    tr.RandomHorizontalFlip(),
    tr.AdaptiveCLAHE(p=0.5, clipLimit=8.0)
])
strong = transforms.Compose([
    tr.Brightness(0.5, 1.5),
    tr.Contrast(0.5, 1.5),
])
normal_toTensor = transforms.Compose([
    tr.Normalize_tf(dataRange=[0,1]),
    tr.ToTensor(unet_size=256)
])

# Create patch sampler
patch_sampler = tr.RandomPatchSamplerWithClass(
    num_patches=4,      # Total patches to sample
    num_fg=2,           # Minimum foreground patches
    min_ratio=0.3,      # Min foreground ratio in FG patches
    fg_threshold=0.05,  # Min % of pixels to be FG
    return_coords=True  # Return crop coordinates
)

# Initialize dataset with multi-patch support
dataset = FundusSegmentation(
    base_dir='/app/MixDSemi/data/Fundus',
    phase='train',
    splitid=1,
    domain=[1],
    selected_idxs=list(range(20)),
    weak_transform=weak,
    strong_tranform=strong,
    normal_toTensor=normal_toTensor,
    img_size=256,
    is_RGB=True,
    patch_sampler=patch_sampler  # Enable multi-patch mode
)

# Load sample
sample = dataset[0]
print(f"Image shape: {sample['image'].shape}")         # [4, 3, 256, 256]
print(f"Label shape: {sample['label'].shape}")         # [4, 256, 256]
print(f"Patch labels: {sample['patch_labels']}")       # tensor([1, 1, 0, 0])
print(f"Num patches: {sample['num_patches']}")         # 4
if 'patch_coords' in sample:
    print(f"Coords: {sample['patch_coords']}")         # [(x,y,w,h), ...]
EOF
echo ""

echo "4. DATASET CONFIGURATIONS"
echo "------------------------"
echo "Fundus:   base_dir=/app/MixDSemi/data/Fundus, img_size=256, is_RGB=True, fillcolor=255"
echo "Prostate: base_dir=/app/MixDSemi/data/ProstateSlice, img_size=384, is_RGB=False, fillcolor=255"
echo "MNMS:     base_dir=/app/MixDSemi/data/mnms, img_size=288, is_RGB=False, fillcolor=0"
echo "BUSI:     base_dir=/app/MixDSemi/data/Dataset_BUSI_with_GT, img_size=256, is_RGB=False, fillcolor=0"
echo ""

echo "5. LABEL CONVENTIONS (IMPORTANT!)"
echo "---------------------------------"
echo "Fundus:   0=background, 128=cup, 255=disc (no transformation needed)"
echo "Prostate: INVERTED! Original 0=foreground → normalized 255=foreground"
echo "MNMS:     RGB mask → class indices (0=bg, 1=lv, 2=myo, 3=rv)"
echo "BUSI:     0=background, 255=tumor (no transformation needed)"
echo ""

echo "6. OUTPUT FORMAT (Multi-Patch Mode)"
echo "-----------------------------------"
echo "sample = {"
echo "    'image': torch.Size([num_patches, C, H, W]),"
echo "    'label': torch.Size([num_patches, H, W]),"
echo "    'patch_labels': torch.Size([num_patches]),  # Binary: 0=bg, 1=fg"
echo "    'img_name': str,"
echo "    'dc': int,"
echo "    'num_patches': int,"
echo "    'strong_aug': torch.Size([num_patches, C, H, W]),  # Optional"
echo "    'orig_image': torch.Size([num_patches, C, H, W]),  # Optional"
echo "    'patch_coords': List[(x, y, w, h)],  # Optional"
echo "}"
echo ""

echo "7. CHECK RESULTS"
echo "---------------"
echo "ls -lh /app/MixDSemi/results/patch_viz_*/batch_sample_*.png"
echo ""
