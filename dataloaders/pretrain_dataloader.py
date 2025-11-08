from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from glob import glob
import random
import copy
import matplotlib.pyplot as plt
import h5py
from scipy.ndimage.interpolation import zoom
import torch


class ProstateSegmentation(Dataset):
    """
    Prostate segmentation dataset with multi-patch sampling support.
    
    This dataset supports:
    1. Standard single-image mode (when patch_sampler=None)
    2. Multi-patch mode (when patch_sampler is provided):
       - Applies patch_sampler BEFORE all other transforms
       - Normalizes labels to ensure foreground>0, background=0
       - Applies weak/strong/normalize transforms to each patch independently
       - Returns batched patches, masks, and binary labels
    """

    def __init__(self,
                 base_dir='../../data/ProstateSlice',
                 phase='train',
                 splitid=2,
                 domain=[1,2,3,4,5,6],
                 weak_transform=None,
                 strong_tranform=None,
                 normal_toTensor = None,
                 selected_idxs = None,
                 img_size = 384,
                 is_RGB = False,
                 patch_sampler = None,  # NEW: RandomPatchSamplerWithClass instance
                 debug_patch_uniques=False,
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        # super().__init__()
        self._base_dir = base_dir
        self.image_list = []
        self.phase = phase
        self.domain_name = {1:'BIDMC', 2:'BMC', 3:'HK', 4:'I2CVB', 5:'RUNMC', 6:'UCL'}
        self.image_pool = []
        self.label_pool = []
        self.img_name_pool = []
        self.img_domain_code_pool = []
        self.img_size = img_size
        self.is_RGB = is_RGB

        self.splitid = splitid
        self.domain = domain
        SEED = 1212
        random.seed(SEED)
        excluded_num = 0
        for i in self.domain:
            self._image_dir = os.path.join(self._base_dir, self.domain_name[i], phase,'image/')
            print('==> Loading {} data from: {}'.format(phase, self._image_dir))

            imagelist = glob(self._image_dir + '*.png')
            imagelist.sort()
            if self.splitid == i and selected_idxs is not None:
                total = list(range(len(imagelist)))
                excluded_idxs = [x for x in total if x not in selected_idxs]
                excluded_num += len(excluded_idxs)
            else:
                excluded_idxs = []
            
            for exclude_id in reversed(sorted(excluded_idxs)):
                imagelist.pop(exclude_id)
                
            for image_path in imagelist:
                self.image_pool.append(image_path)
                gt_path = image_path.replace('image', 'mask')
                self.label_pool.append(gt_path)
                self.img_domain_code_pool.append(i)
                _img_name = image_path.split('/')[-1]
                self.img_name_pool.append(self.domain_name[i]+'_'+_img_name)

        self.weak_transform = weak_transform
        self.strong_transform = strong_tranform
        self.normal_toTensor = normal_toTensor
        self.patch_sampler = patch_sampler
        self.debug_patch_uniques = debug_patch_uniques
        
        
        print('-----Total number of images in {}: {:d}, Excluded: {:d}'.format(phase, len(self.image_pool), excluded_num))
        if self.patch_sampler is not None:
            print(f'[INFO] Multi-patch mode enabled with patch_sampler')

    def __len__(self):
        return len(self.image_pool)
    
    def _normalize_label(self, label_pil):
        """
        Convert Prostate masks from original convention (0=foreground, 255=background)
        into a training-friendly binary mask: background=0, foreground=1.

        Returns a PIL Image (mode 'L') with dtype uint8 where values are {0,1}.
        """
        label_np = np.array(label_pil)
        # map original 0 (fg) -> 1, and 255 (bg) -> 0 for training-friendly binary labels
        new = np.zeros_like(label_np, dtype=np.uint8)
        new[label_np == 0] = 1
        return Image.fromarray(new)
        
    def __getitem__(self, index):
        if self.phase != 'test':
            _img = Image.open(self.image_pool[index]).resize((self.img_size, self.img_size), Image.LANCZOS)
            _target = Image.open(self.label_pool[index]).resize((self.img_size, self.img_size), Image.NEAREST)
            if _img.mode == 'RGB':
                _img = _img.convert('L')
            if _target.mode == 'RGB':
                _target = _target.convert('L')
            if self.is_RGB:
                _img = _img.convert('RGB')
            
            # Normalize label BEFORE patch sampling to ensure correct foreground detection
            _target = self._normalize_label(_target)
            
            anco_sample = {'image': _img, 'label': _target, 'img_name':self.img_name_pool[index], 'dc': self.img_domain_code_pool[index]}
            
            # === Multi-patch mode ===
            if self.patch_sampler is not None:
                # Step 1: Apply patch sampler FIRST (before any other transforms).
                # Note: labels were normalized above so that background==0 and foreground==255,
                # therefore the patch_sampler can use its default fg detection function.
                anco_sample = self.patch_sampler(anco_sample)
                
                patches = anco_sample['patches']  # List[PIL.Image]
                patch_masks = anco_sample['patch_masks']  # List[PIL.Image]
                patch_labels = anco_sample['patch_labels']  # List[int]
                
                # 输出并记录每个 patch 的 mask 的 unique 值，便于调试与可视化问题定位
                patch_mask_uniques = []
                for pi, pm in enumerate(patch_masks):
                    arr = np.array(pm)
                    uniques = np.unique(arr)
                    patch_mask_uniques.append(uniques.tolist())
                    if getattr(self, 'debug_patch_uniques', False):
                        print(f'[DEBUG] Prostate sample {self.img_name_pool[index]} patch {pi} mask uniques: {uniques}')

                num_patches = len(patches)
                
                # Step 2: Apply weak/strong/normalize transforms to EACH patch
                processed_patches = []
                processed_strong_augs = []
                processed_masks = []
                
                for i in range(num_patches):
                    patch_sample = {
                        'image': patches[i],
                        'label': patch_masks[i],
                        'img_name': f"{self.img_name_pool[index]}_patch{i}",
                        'dc': self.img_domain_code_pool[index]
                    }
                    
                    # 保存未增强（原始）patch 的归一化张量版本，便于可视化
                    # 这里不应用 weak/strong，只应用 normal_toTensor（即 Normalize_tf + ToTensor）
                    orig_patch_tensor = None
                    if self.normal_toTensor is not None:
                        try:
                            orig_proc = self.normal_toTensor({'image': patches[i], 'label': patch_masks[i], 'img_name': patch_sample['img_name'], 'dc': patch_sample['dc']})
                            orig_patch_tensor = orig_proc['image']
                        except Exception:
                            # 若 normal_toTensor 期望不同格式则跳过
                            orig_patch_tensor = None

                    # Apply weak transform (robust: accept dict or image outputs)
                    if self.weak_transform is not None:
                        # If weak_transform is a torchvision.transforms.Compose-like object,
                        # iterate through its sub-transforms to pinpoint any that return None
                        if hasattr(self.weak_transform, 'transforms'):
                            cur = patch_sample
                            for t in self.weak_transform.transforms:
                                cur = t(cur)
                                if cur is None:
                                    raise RuntimeError(f'weak sub-transform {t.__class__.__name__} returned None for {patch_sample.get("img_name", "unknown")}')
                            # cur may be a dict or an image; normalize to dict
                            if isinstance(cur, dict):
                                patch_sample = cur
                            else:
                                patch_sample['image'] = cur
                        else:
                            out = self.weak_transform(patch_sample)
                            if out is None:
                                raise RuntimeError(f'weak_transform returned None for {patch_sample.get("img_name", "unknown")}')
                            if isinstance(out, dict):
                                patch_sample = out
                            else:
                                patch_sample['image'] = out

                    # Apply strong transform (it may accept image and return image)
                    if self.strong_transform is not None:
                        strong_out = self.strong_transform(patch_sample['image'])
                        # if transform returns None, fallback to the (possibly weak-transformed) image
                        if strong_out is None:
                            patch_sample['strong_aug'] = patch_sample['image']
                        else:
                            patch_sample['strong_aug'] = strong_out
                    
                    # Apply normalize + ToTensor
                    patch_sample = self.normal_toTensor(patch_sample)
                    
                    processed_patches.append(patch_sample['image'])  # Tensor
                    processed_masks.append(patch_sample['label'])    # Tensor
                    if 'strong_aug' in patch_sample:
                        processed_strong_augs.append(patch_sample['strong_aug'])
                    # 保存 orig tensor（如果存在）
                    if orig_patch_tensor is not None:
                        if 'orig_images' not in locals():
                            orig_images = []
                        orig_images.append(orig_patch_tensor)
                
                # Step 3: Stack all patches into batch tensors
                result = {
                    'image': torch.stack(processed_patches, dim=0),  # [num_patches, C, H, W]
                    'label': torch.stack(processed_masks, dim=0),    # [num_patches, H, W]
                    'patch_labels': torch.tensor(patch_labels, dtype=torch.long),  # [num_patches]
                    'img_name': self.img_name_pool[index],
                    'dc': self.img_domain_code_pool[index],
                    'num_patches': num_patches
                }
                
                if processed_strong_augs:
                    result['strong_aug'] = torch.stack(processed_strong_augs, dim=0)  # [num_patches, C, H, W]

                # 如果存在 orig_images（归一化后的原始 patch 张量），将其加入返回 dict
                if 'orig_images' in locals() and len(orig_images) == len(processed_patches):
                    result['orig_image'] = torch.stack(orig_images, dim=0)  # [num_patches, C, H, W]

                # include mask unique values in result for downstream inspection/visualization
                result['patch_mask_uniques'] = patch_mask_uniques
                # also compute uniques after transforms/ToTensor (processed masks)
                if getattr(self, 'debug_patch_uniques', False):
                    try:
                        proc_mask = result['label'].cpu().numpy()  # shape [P, H, W]
                        proc_uniques = [np.unique(proc_mask[p]).tolist() for p in range(proc_mask.shape[0])]
                        result['processed_patch_mask_uniques'] = proc_uniques
                        for pi, u in enumerate(proc_uniques):
                            print(f'[DEBUG] Processed mask uniques for {self.img_name_pool[index]} patch {pi}: {u}')
                    except Exception:
                        pass
                
                # Preserve additional outputs from ToTensor if present
                if 'low_res_label' in patch_sample:
                    result['low_res_label'] = torch.stack([ps['low_res_label'] for ps in [self.normal_toTensor({'image': patches[i], 'label': patch_masks[i], 'img_name': '', 'dc': 0}) for i in range(num_patches)]], dim=0)
                if 'unet_size_img' in patch_sample:
                    result['unet_size_img'] = torch.stack([self.normal_toTensor({'image': patches[i], 'label': patch_masks[i], 'img_name': '', 'dc': 0})['unet_size_img'] for i in range(num_patches)], dim=0)
                if 'unet_size_label' in patch_sample:
                    result['unet_size_label'] = torch.stack([self.normal_toTensor({'image': patches[i], 'label': patch_masks[i], 'img_name': '', 'dc': 0})['unet_size_label'] for i in range(num_patches)], dim=0)
                
                return result
            
            # === Standard single-image mode ===
            else:
                if self.weak_transform is not None:
                    anco_sample = self.weak_transform(anco_sample)
                if self.strong_transform is not None:
                    anco_sample['strong_aug'] = self.strong_transform(anco_sample['image'])
                anco_sample = self.normal_toTensor(anco_sample)
                return anco_sample
        else:
            # Test mode (no patches)
            _img = Image.open(self.image_pool[index]).resize((self.img_size, self.img_size), Image.LANCZOS)
            _target = Image.open(self.label_pool[index])
            if _img.mode == 'RGB':
                _img = _img.convert('L')
            if _target.mode == 'RGB':
                _target = _target.convert('L')
            if self.is_RGB:
                _img = _img.convert('RGB')
            
            _target = self._normalize_label(_target)
            
            anco_sample = {'image': _img, 'label': _target, 'img_name':self.img_name_pool[index], 'dc': self.img_domain_code_pool[index]}
            anco_sample = self.normal_toTensor(anco_sample)
            return anco_sample

    def __str__(self):
        return 'Prostate(phase=' + self.phase+str(self.splitid) + ')'


class FundusSegmentation(Dataset):
    """
    Fundus segmentation dataset with multi-patch sampling support.
    
    This dataset supports:
    1. Standard single-image mode (when patch_sampler=None)
    2. Multi-patch mode (when patch_sampler is provided):
       - Applies patch_sampler BEFORE all other transforms
       - Normalizes labels to ensure foreground>0, background=0
       - Applies weak/strong/normalize transforms to each patch independently
       - Returns batched patches, masks, and binary labels
    """

    def __init__(self,
                 base_dir='../../data/Fundus',
                 phase='train',
                 splitid=2,
                 domain=[1,2,3,4],
                 weak_transform=None,
                 strong_tranform=None,
                 normal_toTensor = None,
                 selected_idxs = None,
                 img_size = 256, 
                 is_RGB = False,
                 patch_sampler = None,  # NEW: RandomPatchSamplerWithClass instance
                 debug_patch_uniques=False,
                 ):
        """
        :param base_dir: path to Fundus dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        self._base_dir = base_dir
        self.image_list = []
        self.phase = phase
        self.domain_name = {1:'DGS', 2:'RIM', 3:'REF', 4:'REF_val'}
        self.image_pool = []
        self.label_pool = []
        self.img_name_pool = []
        self.img_domain_code_pool = []
        self.img_size = img_size
        self.is_RGB = is_RGB

        self.flags_DGS = ['gd', 'nd']
        self.flags_REF = ['g', 'n']
        self.flags_RIM = ['G', 'N', 'S']
        self.flags_REF_val = ['V']
        self.splitid = splitid
        self.domain = domain
        SEED = 1212
        random.seed(SEED)
        excluded_num = 0
        for i in self.domain:
            self._image_dir = os.path.join(self._base_dir, 'Domain'+str(i), phase, 'ROIs/image/')
            print('==> Loading {} data from: {}'.format(phase, self._image_dir))

            imagelist = glob(self._image_dir + '*.png')
            imagelist.sort()

            if (self.splitid == i or self.splitid == -1) and selected_idxs is not None:
                total = list(range(len(imagelist)))
                excluded_idxs = [x for x in total if x not in selected_idxs]
                excluded_num += len(excluded_idxs)
            else:
                excluded_idxs = []
            
            for exclude_id in reversed(sorted(excluded_idxs)):
                imagelist.pop(exclude_id)
                
            for image_path in imagelist:
                self.image_pool.append(image_path)
                gt_path = image_path.replace('image', 'mask')
                self.label_pool.append(gt_path)
                self.img_domain_code_pool.append(i)
                _img_name = image_path.split('/')[-1]
                self.img_name_pool.append(_img_name)
            print(f'-----Number of domain {i} images: {len(imagelist)}, Excluded: {len(excluded_idxs)}')

        self.weak_transform = weak_transform
        self.strong_transform = strong_tranform
        self.normal_toTensor = normal_toTensor
        self.patch_sampler = patch_sampler
        self.debug_patch_uniques = debug_patch_uniques

        print('-----Total number of images in {}: {:d}, Excluded: {:d}'.format(phase, len(self.image_pool), excluded_num))
        if self.patch_sampler is not None:
            print(f'[INFO] Multi-patch mode enabled with patch_sampler')

    def __len__(self):
        return len(self.image_pool)
    
    def _normalize_label(self, label_pil):
        """
        Normalize label to ensure foreground > 0 and background = 0.
        For Fundus dataset: masks have values 0 (background), 128 (cup), 255 (disc).
        We keep this mapping as-is since foreground values are already > 0.
        
        Args:
            label_pil: PIL Image in 'L' mode (grayscale)
        
        Returns:
            Normalized PIL Image where background=0, foreground>0
        """
        # Robust fundus mapping:
        # Some datasets may encode background as 255 or 0 depending on preprocessing.
        # Heuristic: inspect uniques and counts. If values are the expected {0,128,255}, map
        # 0->0, 128->1 (cup), 255->2 (disc). Otherwise choose the most frequent pixel value
        # as background and map it -> 0; map remaining distinct values to 1..K in sorted order.
        arr = np.array(label_pil)
        uniques, counts = np.unique(arr, return_counts=True)
        if getattr(self, 'debug_patch_uniques', False):
            try:
                uniq_dict = {int(u): int(c) for u, c in zip(uniques.tolist(), counts.tolist())}
                print(f'[DEBUG] Fundus original mask uniques/counts: {uniq_dict}')
            except Exception:
                pass

        # Determine background as the most frequent pixel value and map it -> 0.
        # Map remaining distinct values to 1..K in sorted order. This is a hard
        # deterministic mapping based on pixel distribution (no heuristics).
        bg_val = int(uniques[counts.argmax()])
        out = np.zeros_like(arr, dtype=np.uint8)
        other_vals = [int(u) for u in uniques.tolist() if int(u) != bg_val]
        other_vals_sorted = sorted(other_vals)
        for idx, v in enumerate(other_vals_sorted):
            out[arr == v] = idx + 1
        if getattr(self, 'debug_patch_uniques', False):
            print(f'[DEBUG] Fundus normalized mapping background={bg_val} -> 0, others={other_vals_sorted} -> 1..')
        return Image.fromarray(out)
        
    def __getitem__(self, index):
        if self.phase != 'test':
            _img = Image.open(self.image_pool[index]).convert('RGB').resize((self.img_size, self.img_size), Image.LANCZOS)
            _target = Image.open(self.label_pool[index])
            if _target.mode == 'RGB':
                _target = _target.convert('L')
            _target = _target.resize((self.img_size, self.img_size), Image.NEAREST)
            
            # Normalize label BEFORE patch sampling
            _target = self._normalize_label(_target)
            
            anco_sample = {'image': _img, 'label': _target, 'img_name':self.img_name_pool[index], 'dc': self.img_domain_code_pool[index]}
            
            # === Multi-patch mode ===
            if self.patch_sampler is not None:
                anco_sample = self.patch_sampler(anco_sample)
                
                patches = anco_sample['patches']
                patch_masks = anco_sample['patch_masks']
                patch_labels = anco_sample['patch_labels']
                # record unique mask values per patch for debugging/visualization
                patch_mask_uniques = []
                for pi, pm in enumerate(patch_masks):
                    arr = np.array(pm)
                    uniques = np.unique(arr)
                    patch_mask_uniques.append(uniques.tolist())
                    if getattr(self, 'debug_patch_uniques', False):
                        print(f'[DEBUG] Fundus sample {self.img_name_pool[index]} patch {pi} mask uniques: {uniques}')
                num_patches = len(patches)
                
                processed_patches = []
                processed_strong_augs = []
                processed_masks = []
                orig_images = []
                
                for i in range(num_patches):
                    patch_sample = {
                        'image': patches[i],
                        'label': patch_masks[i],
                        'img_name': f"{self.img_name_pool[index]}_patch{i}",
                        'dc': self.img_domain_code_pool[index]
                    }
                    
                    orig_patch_tensor = None
                    if self.normal_toTensor is not None:
                        try:
                            orig_proc = self.normal_toTensor({'image': patches[i], 'label': patch_masks[i], 'img_name': patch_sample['img_name'], 'dc': patch_sample['dc']})
                            orig_patch_tensor = orig_proc['image']
                        except Exception:
                            orig_patch_tensor = None

                    if self.weak_transform is not None:
                        if hasattr(self.weak_transform, 'transforms'):
                            cur = patch_sample
                            for t in self.weak_transform.transforms:
                                cur = t(cur)
                                if cur is None:
                                    raise RuntimeError(f'weak sub-transform {t.__class__.__name__} returned None')
                            if isinstance(cur, dict):
                                patch_sample = cur
                            else:
                                patch_sample['image'] = cur
                        else:
                            out = self.weak_transform(patch_sample)
                            if out is None:
                                raise RuntimeError(f'weak_transform returned None')
                            if isinstance(out, dict):
                                patch_sample = out
                            else:
                                patch_sample['image'] = out

                    if self.strong_transform is not None:
                        strong_out = self.strong_transform(patch_sample['image'])
                        if strong_out is None:
                            patch_sample['strong_aug'] = patch_sample['image']
                        else:
                            patch_sample['strong_aug'] = strong_out
                    
                    patch_sample = self.normal_toTensor(patch_sample)
                    
                    processed_patches.append(patch_sample['image'])
                    processed_masks.append(patch_sample['label'])
                    if 'strong_aug' in patch_sample:
                        processed_strong_augs.append(patch_sample['strong_aug'])
                    if orig_patch_tensor is not None:
                        orig_images.append(orig_patch_tensor)
                
                result = {
                    'image': torch.stack(processed_patches, dim=0),
                    'label': torch.stack(processed_masks, dim=0),
                    'patch_labels': torch.tensor(patch_labels, dtype=torch.long),
                    'img_name': self.img_name_pool[index],
                    'dc': self.img_domain_code_pool[index],
                    'num_patches': num_patches
                }
                
                if processed_strong_augs:
                    result['strong_aug'] = torch.stack(processed_strong_augs, dim=0)
                if len(orig_images) == len(processed_patches):
                    result['orig_image'] = torch.stack(orig_images, dim=0)
                if 'patch_coords' in anco_sample:
                    result['patch_coords'] = anco_sample['patch_coords']
                # include mask unique values
                result['patch_mask_uniques'] = patch_mask_uniques
                
                return result
            
            # === Standard single-image mode ===
            else:
                if self.weak_transform is not None:
                    anco_sample = self.weak_transform(anco_sample)
                if self.strong_transform is not None:
                    anco_sample['strong_aug'] = self.strong_transform(anco_sample['image'])
                anco_sample = self.normal_toTensor(anco_sample)
                return anco_sample
        else:
            _img = Image.open(self.image_pool[index]).convert('RGB').resize((self.img_size, self.img_size), Image.LANCZOS)
            _target = Image.open(self.label_pool[index])
            if _target.mode == 'RGB':
                _target = _target.convert('L')
            _target = _target.resize((256, 256), Image.NEAREST)
            _target = self._normalize_label(_target)
            anco_sample = {'image': _img, 'label': _target, 'img_name':self.img_name_pool[index], 'dc': self.img_domain_code_pool[index]}
            anco_sample = self.normal_toTensor(anco_sample)
        return anco_sample

    def __str__(self):
        return 'Fundus(phase=' + self.phase+str(self.splitid) + ')'


class MNMSSegmentation(Dataset):
    """
    MNMS segmentation dataset with multi-patch sampling support.
    
    This dataset supports:
    1. Standard single-image mode (when patch_sampler=None)
    2. Multi-patch mode (when patch_sampler is provided):
       - Applies patch_sampler BEFORE all other transforms
       - Normalizes labels to ensure foreground>0, background=0
       - Applies weak/strong/normalize transforms to each patch independently
       - Returns batched patches, masks, and binary labels
    """

    def __init__(self,
                 base_dir='../../data/mnms',
                 phase='train',
                 splitid=2,
                 domain=[1,2,3,4],
                 weak_transform=None,
                 strong_tranform=None,
                 normal_toTensor = None,
                 selected_idxs = None,
                 img_size = 288,
                 is_RGB = False,
                 patch_sampler = None,  # NEW: RandomPatchSamplerWithClass instance
                 debug_patch_uniques=False,
                 ):
        """
        :param base_dir: path to MNMS dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        self._base_dir = base_dir
        self.image_list = []
        self.phase = phase
        self.domain_name = {1:'vendorA', 2:'vendorB', 3:'vendorC', 4:'vendorD'}
        self.image_pool = []
        self.label_pool = []
        self.img_name_pool = []
        self.img_domain_code_pool = []
        self.img_size = img_size
        self.is_RGB = is_RGB

        self.splitid = splitid
        self.domain = domain
        SEED = 1212
        random.seed(SEED)
        excluded_num = 0
        for i in self.domain:
            self._image_dir = os.path.join(self._base_dir, self.domain_name[i], phase,'image/')
            print('==> Loading {} data from: {}'.format(phase, self._image_dir))

            imagelist = glob(self._image_dir + '*.png')
            imagelist.sort()
            if self.splitid == i and selected_idxs is not None:
                total = list(range(len(imagelist)))
                excluded_idxs = [x for x in total if x not in selected_idxs]
                excluded_num += len(excluded_idxs)
            else:
                excluded_idxs = []
            
            for exclude_id in reversed(sorted(excluded_idxs)):
                imagelist.pop(exclude_id)
                
            for image_path in imagelist:
                self.image_pool.append(image_path)
                gt_path = image_path.replace('image', 'mask')
                self.label_pool.append(gt_path)
                self.img_domain_code_pool.append(i)
                _img_name = image_path.split('/')[-1]
                self.img_name_pool.append(self.domain_name[i]+'_'+_img_name)

        self.weak_transform = weak_transform
        self.strong_transform = strong_tranform
        self.normal_toTensor = normal_toTensor
        self.patch_sampler = patch_sampler
        self.debug_patch_uniques = debug_patch_uniques
        
        print('-----Total number of images in {}: {:d}, Excluded: {:d}'.format(phase, len(self.image_pool), excluded_num))
        if self.patch_sampler is not None:
            print(f'[INFO] Multi-patch mode enabled with patch_sampler')

    def __len__(self):
        return len(self.image_pool)
    
    def _normalize_label(self, label_pil):
        """
        Normalize label to ensure foreground > 0 and background = 0.
        For MNMS dataset: RGB masks are converted to class indices (0=bg, 1=lv, 2=myo, 3=rv).
        This is already correct format (background=0, foreground>0).
        
        Args:
            label_pil: PIL Image (RGB or already converted to grayscale with class indices)
        
        Returns:
            Normalized PIL Image where background=0, foreground>0
        """
        # MNMS dataset: convert RGB mask to class indices
        if label_pil.mode == 'RGB':
            target_np = np.array(label_pil)
            new_target = np.zeros((target_np.shape[0], target_np.shape[1]), dtype=np.uint8)
            for n in range(3):
                new_target[target_np[:, :, n] == 255] = n + 1
            return Image.fromarray(new_target)
        else:
            # Already converted
            return label_pil
        
    def __getitem__(self, index):
        if self.phase != 'test':
            _img = Image.open(self.image_pool[index]).resize((self.img_size, self.img_size), Image.BILINEAR)
            _target = Image.open(self.label_pool[index]).resize((self.img_size, self.img_size), Image.NEAREST)
            if _img.mode == 'RGB':
                _img = _img.convert('L')
            
            # Normalize label BEFORE patch sampling
            _target = self._normalize_label(_target)
            
            anco_sample = {'image': _img, 'label': _target, 'img_name':self.img_name_pool[index], 'dc': self.img_domain_code_pool[index]}
            
            # === Multi-patch mode ===
            if self.patch_sampler is not None:
                anco_sample = self.patch_sampler(anco_sample)
                
                patches = anco_sample['patches']
                patch_masks = anco_sample['patch_masks']
                patch_labels = anco_sample['patch_labels']
                # record unique mask values per patch for debugging/visualization
                patch_mask_uniques = []
                for pi, pm in enumerate(patch_masks):
                    arr = np.array(pm)
                    uniques = np.unique(arr)
                    patch_mask_uniques.append(uniques.tolist())
                    if getattr(self, 'debug_patch_uniques', False):
                        print(f'[DEBUG] MNMS sample {self.img_name_pool[index]} patch {pi} mask uniques: {uniques}')
                num_patches = len(patches)
                
                processed_patches = []
                processed_strong_augs = []
                processed_masks = []
                orig_images = []
                
                for i in range(num_patches):
                    patch_sample = {
                        'image': patches[i],
                        'label': patch_masks[i],
                        'img_name': f"{self.img_name_pool[index]}_patch{i}",
                        'dc': self.img_domain_code_pool[index]
                    }
                    
                    orig_patch_tensor = None
                    if self.normal_toTensor is not None:
                        try:
                            orig_proc = self.normal_toTensor({'image': patches[i], 'label': patch_masks[i], 'img_name': patch_sample['img_name'], 'dc': patch_sample['dc']})
                            orig_patch_tensor = orig_proc['image']
                        except Exception:
                            orig_patch_tensor = None

                    if self.weak_transform is not None:
                        if hasattr(self.weak_transform, 'transforms'):
                            cur = patch_sample
                            for t in self.weak_transform.transforms:
                                cur = t(cur)
                                if cur is None:
                                    raise RuntimeError(f'weak sub-transform {t.__class__.__name__} returned None')
                            if isinstance(cur, dict):
                                patch_sample = cur
                            else:
                                patch_sample['image'] = cur
                        else:
                            out = self.weak_transform(patch_sample)
                            if out is None:
                                raise RuntimeError(f'weak_transform returned None')
                            if isinstance(out, dict):
                                patch_sample = out
                            else:
                                patch_sample['image'] = out

                    if self.strong_transform is not None:
                        strong_out = self.strong_transform(patch_sample['image'])
                        if strong_out is None:
                            patch_sample['strong_aug'] = patch_sample['image']
                        else:
                            patch_sample['strong_aug'] = strong_out
                    
                    patch_sample = self.normal_toTensor(patch_sample)
                    
                    processed_patches.append(patch_sample['image'])
                    processed_masks.append(patch_sample['label'])
                    if 'strong_aug' in patch_sample:
                        processed_strong_augs.append(patch_sample['strong_aug'])
                    if orig_patch_tensor is not None:
                        orig_images.append(orig_patch_tensor)
                
                result = {
                    'image': torch.stack(processed_patches, dim=0),
                    'label': torch.stack(processed_masks, dim=0),
                    'patch_labels': torch.tensor(patch_labels, dtype=torch.long),
                    'img_name': self.img_name_pool[index],
                    'dc': self.img_domain_code_pool[index],
                    'num_patches': num_patches
                }
                
                if processed_strong_augs:
                    result['strong_aug'] = torch.stack(processed_strong_augs, dim=0)
                if len(orig_images) == len(processed_patches):
                    result['orig_image'] = torch.stack(orig_images, dim=0)
                if 'patch_coords' in anco_sample:
                    result['patch_coords'] = anco_sample['patch_coords']
                # include mask uniques
                result['patch_mask_uniques'] = patch_mask_uniques
                
                return result
            
            # === Standard single-image mode ===
            else:
                if self.weak_transform is not None:
                    anco_sample = self.weak_transform(anco_sample)
                if self.strong_transform is not None:
                    anco_sample['strong_aug'] = self.strong_transform(anco_sample['image'])
                anco_sample = self.normal_toTensor(anco_sample)
                return anco_sample
        else:
            _img = Image.open(self.image_pool[index]).resize((self.img_size, self.img_size), Image.BILINEAR)
            _target = Image.open(self.label_pool[index]).resize((288,288), Image.NEAREST)
            if _img.mode == 'RGB':
                _img = _img.convert('L')
            _target = self._normalize_label(_target)
            anco_sample = {'image': _img, 'label': _target, 'img_name':self.img_name_pool[index], 'dc': self.img_domain_code_pool[index]}
            anco_sample = self.normal_toTensor(anco_sample)
        return anco_sample

    def __str__(self):
        return 'MNMS(phase=' + self.phase+str(self.splitid) + ')'


class BUSISegmentation(Dataset):
    """
    BUSI segmentation dataset with multi-patch sampling support.
    
    This dataset supports:
    1. Standard single-image mode (when patch_sampler=None)
    2. Multi-patch mode (when patch_sampler is provided):
       - Applies patch_sampler BEFORE all other transforms
       - Normalizes labels to ensure foreground>0, background=0
       - Applies weak/strong/normalize transforms to each patch independently
       - Returns batched patches, masks, and binary labels
    """
    def __init__(self, base_dir=None, phase='train', splitid=1, domain=[1,2],
                 weak_transform=None, strong_tranform=None, normal_toTensor=None,
                 selected_idxs = None,
                 img_size = 256,
                 is_RGB = False,
                 patch_sampler = None,  # NEW: RandomPatchSamplerWithClass instance
                 debug_patch_uniques=False,
                 ):
        self._base_dir = base_dir
        self.image_list = []
        self.phase = phase
        self.domain_name = {1:'benign', 2:'malignant'}
        self.sample_list = []
        self.img_name_pool = []
        self.img_domain_code_pool = []
        self.img_size = img_size
        self.is_RGB = is_RGB
        self.weak_transform = weak_transform
        self.strong_transform = strong_tranform
        self.normal_toTensor = normal_toTensor
        self.patch_sampler = patch_sampler
        self.debug_patch_uniques = debug_patch_uniques
        
        self.splitid = splitid
        self.domain = domain
        SEED = 1212
        random.seed(SEED)
        excluded_num = 0
        for i in self.domain:
            self._image_dir = os.path.join(self._base_dir, self.domain_name[i]+'/')
            print('==> Loading {} data from: {}'.format(phase, self._image_dir))

            domain_data_list = []
            imagelist = glob(self._image_dir + '*.png')
            imagelist.sort()
            for image in imagelist:
                if 'mask' not in image:
                    domain_data_list.append([image])
                else:
                    domain_data_list[-1].append(image)
            test_benign_num = int(len(domain_data_list)*0.2)
            train_benign_num = len(domain_data_list) - test_benign_num
            if self.phase == 'test':
                domain_data_list = domain_data_list[-test_benign_num:]
            elif self.phase == 'train':
                domain_data_list = domain_data_list[:train_benign_num]
            else:
                raise Exception('Unknown split...')
            if self.splitid == i and selected_idxs is not None:
                total = list(range(len(domain_data_list)))
                excluded_idxs = [x for x in total if x not in selected_idxs]
                excluded_num += len(excluded_idxs)
            else:
                excluded_idxs = []
            
            for exclude_id in reversed(sorted(excluded_idxs)):
                domain_data_list.pop(exclude_id)
                
            for image_path in domain_data_list:
                self.sample_list.append(image_path)
                self.img_domain_code_pool.append(i)
                _img_name = image_path[0].split('/')[-1]
                self.img_name_pool.append(self.domain_name[i]+'_'+_img_name)
        
        print('-----Total number of images in {}: {:d}, Excluded: {:d}'.format(phase, len(self.sample_list), excluded_num))
        if self.patch_sampler is not None:
            print(f'[INFO] Multi-patch mode enabled with patch_sampler')

    def __len__(self):
        return len(self.sample_list)
    
    def _normalize_label(self, label_pil):
        """
        Normalize label to ensure foreground > 0 and background = 0.
        For BUSI dataset: masks have values 0 (background) and 255 (tumor).
        We keep 255 for foreground (tumor) which is > 0.
        
        Args:
            label_pil: PIL Image in 'L' mode (grayscale)
        
        Returns:
            Normalized PIL Image where background=0, foreground>0
        """
        # BUSI dataset: 0=background, 255=tumor
        # convert to binary 0/1 for training: 1=tumor
        arr = np.array(label_pil)
        out = np.zeros_like(arr, dtype=np.uint8)
        out[arr != 0] = 1
        return Image.fromarray(out)

    def __getitem__(self, idx):
        _img = Image.open(self.sample_list[idx][0]).convert('L').resize((self.img_size, self.img_size), Image.LANCZOS)
        if len(self.sample_list[idx]) == 2:
            if self.phase == 'train':
                _target = Image.open(self.sample_list[idx][1]).convert('L').resize((self.img_size, self.img_size), Image.NEAREST)
            else:
                _target = Image.open(self.sample_list[idx][1]).convert('L').resize((256, 256), Image.NEAREST)
        else:
            target_list = []
            for target_path in self.sample_list[idx][1:]:
                target = Image.open(target_path).convert('L')
                target_list.append(np.array(target))
            height, width = target_list[0].shape
            combined_target = np.zeros((height, width), dtype=np.uint8)
            for target in target_list:
                combined_target = np.maximum(combined_target, target)
            if self.phase == 'train':
                _target = Image.fromarray(combined_target).convert('L').resize((self.img_size, self.img_size), Image.NEAREST)
            else:
                _target = Image.fromarray(combined_target).convert('L').resize((256, 256), Image.NEAREST)
        
        # Normalize label BEFORE patch sampling
        _target = self._normalize_label(_target)
        
        sample = {'image': _img, 'label': _target, 'img_name':self.img_name_pool[idx], 'dc': self.img_domain_code_pool[idx]}
        
        if self.phase == "train":
            # === Multi-patch mode ===
            if self.patch_sampler is not None:
                sample = self.patch_sampler(sample)
                
                patches = sample['patches']
                patch_masks = sample['patch_masks']
                patch_labels = sample['patch_labels']
                # record unique mask values per patch for debugging/visualization
                patch_mask_uniques = []
                for pi, pm in enumerate(patch_masks):
                    arr = np.array(pm)
                    uniques = np.unique(arr)
                    patch_mask_uniques.append(uniques.tolist())
                    if getattr(self, 'debug_patch_uniques', False):
                        print(f'[DEBUG] BUSI sample {self.img_name_pool[idx]} patch {pi} mask uniques: {uniques}')
                num_patches = len(patches)
                
                processed_patches = []
                processed_strong_augs = []
                processed_masks = []
                orig_images = []
                
                for i in range(num_patches):
                    patch_sample = {
                        'image': patches[i],
                        'label': patch_masks[i],
                        'img_name': f"{self.img_name_pool[idx]}_patch{i}",
                        'dc': self.img_domain_code_pool[idx]
                    }
                    
                    orig_patch_tensor = None
                    if self.normal_toTensor is not None:
                        try:
                            orig_proc = self.normal_toTensor({'image': patches[i], 'label': patch_masks[i], 'img_name': patch_sample['img_name'], 'dc': patch_sample['dc']})
                            orig_patch_tensor = orig_proc['image']
                        except Exception:
                            orig_patch_tensor = None

                    if self.weak_transform is not None:
                        if hasattr(self.weak_transform, 'transforms'):
                            cur = patch_sample
                            for t in self.weak_transform.transforms:
                                cur = t(cur)
                                if cur is None:
                                    raise RuntimeError(f'weak sub-transform {t.__class__.__name__} returned None')
                            if isinstance(cur, dict):
                                patch_sample = cur
                            else:
                                patch_sample['image'] = cur
                        else:
                            out = self.weak_transform(patch_sample)
                            if out is None:
                                raise RuntimeError(f'weak_transform returned None')
                            if isinstance(out, dict):
                                patch_sample = out
                            else:
                                patch_sample['image'] = out

                    if self.strong_transform is not None:
                        strong_out = self.strong_transform(patch_sample['image'])
                        if strong_out is None:
                            patch_sample['strong_aug'] = patch_sample['image']
                        else:
                            patch_sample['strong_aug'] = strong_out
                    
                    patch_sample = self.normal_toTensor(patch_sample)
                    
                    processed_patches.append(patch_sample['image'])
                    processed_masks.append(patch_sample['label'])
                    if 'strong_aug' in patch_sample:
                        processed_strong_augs.append(patch_sample['strong_aug'])
                    if orig_patch_tensor is not None:
                        orig_images.append(orig_patch_tensor)
                
                result = {
                    'image': torch.stack(processed_patches, dim=0),
                    'label': torch.stack(processed_masks, dim=0),
                    'patch_labels': torch.tensor(patch_labels, dtype=torch.long),
                    'img_name': self.img_name_pool[idx],
                    'dc': self.img_domain_code_pool[idx],
                    'num_patches': num_patches
                }
                
                if processed_strong_augs:
                    result['strong_aug'] = torch.stack(processed_strong_augs, dim=0)
                if len(orig_images) == len(processed_patches):
                    result['orig_image'] = torch.stack(orig_images, dim=0)
                if 'patch_coords' in sample:
                    result['patch_coords'] = sample['patch_coords']
                # include mask unique values
                result['patch_mask_uniques'] = patch_mask_uniques
                
                return result
            
            # === Standard single-image mode ===
            else:
                if self.weak_transform is not None:
                    sample = self.weak_transform(sample)
                if self.strong_transform is not None:
                    sample['strong_aug'] = self.strong_transform(sample['image'])
                sample = self.normal_toTensor(sample)
                return sample
        else:
            sample = self.normal_toTensor(sample)
        return sample

    def __str__(self):
        return 'BUSI(phase=' + self.phase+str(self.splitid) + ')'