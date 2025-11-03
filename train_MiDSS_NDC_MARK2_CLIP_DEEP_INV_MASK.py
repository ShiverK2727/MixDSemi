import argparse
import contextlib
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

# (新) 导入 cv2 和 numpy
import cv2
import numpy as np

import dataloaders.custom_transforms as tr
from dataloaders.dataloader_dc import BUSISegmentation, FundusSegmentation, MNMSSegmentation, ProstateSegmentation
# 导入新的 Invariant-Only 模型
from biomedclip_vpt_invariant_only import build_invariant_prompt_image_encoder
from utils.text_sampler import TextSampler
from torch.cuda.amp import GradScaler, autocast
from utils.training import cycle

parser = argparse.ArgumentParser()

# ==================== Basic Configuration ====================
# (与原文件相同)
parser.add_argument('--dataset', type=str, default='prostate',
                    choices=['fundus', 'prostate', 'MNMS', 'BUSI'],
                    help='Dataset to use for training')
parser.add_argument("--save_name", type=str, default="",
                    help="Experiment name for saving checkpoints and logs")
parser.add_argument("--overwrite", action='store_true',
                    help="Overwrite existing experiment directory")
parser.add_argument('--save_model', action='store_true',
                    help='Save best model checkpoints during training')
parser.add_argument("--gpu", type=str, default='0',
                    help='GPU device ID to use')
parser.add_argument("--seed", type=int, default=1337,
                    help="Random seed for reproducibility")
parser.add_argument("--deterministic", type=int, default=1,
                    help="Whether to use deterministic training")

# ==================== Training Schedule ====================
# (与原文件相同)
parser.add_argument("--max_iterations", type=int, default=None,
                    help="Maximum iterations to train (auto-set per dataset if None)")
parser.add_argument('--amp', type=int, default=1,
                    help='Use mixed precision training (1: enabled, 0: disabled)')

# ==================== Data Configuration ====================
# (与原文件相同)
parser.add_argument("--label_bs", type=int, default=None,
                    help="Labeled batch size per GPU (auto-set based on lb_num if None)")
parser.add_argument("--unlabel_bs", type=int, default=None,
                    help="Unlabeled batch size per GPU (auto-set based on lb_num if None)")
parser.add_argument('--domain_num', type=int, default=6,
                    help='Total number of domains in dataset')
parser.add_argument('--lb_domain', type=int, default=1,
                    help='Domain ID to use for labeled data')
parser.add_argument('--lb_num', type=int, default=40,
                    help='Number of labeled samples (used when lb_ratio=0)')
parser.add_argument('--lb_ratio', type=float, default=0,
                    help='Labeled data ratio of total dataset (overrides lb_num if > 0)')

# ==================== Preprocessing ====================
# (与原文件相同)
parser.add_argument('--preprocess_dir', type=str, default=None,
                    help='Override preprocessing directory for score tensors')
parser.add_argument('--llm_model', type=str, default='gemini',
                    choices=['gemini', 'GPT5', 'DeepSeek'],
                    help='LLM model used to generate score tensors')
parser.add_argument('--describe_nums', type=int, default=40,
                    choices=[20, 40, 60, 80],
                    help='Number of textual descriptions for preprocessing')

# ==================== Loss Configuration (添加 Crop Loss) ====================
clip_loss_group = parser.add_argument_group('CLIP Loss Configuration')
clip_loss_group.add_argument('--clip_loss_mv_anchor_weight', type=float, default=1.0,
                            help='Weight for (loss_mv + loss_anchor) term')
clip_loss_group.add_argument('--clip_loss_sw_reg_weight', type=float, default=1.0,
                            help='Weight for loss_sw_reg term')
# (新) 前景裁切一致性损失
clip_loss_group.add_argument('--clip_loss_crop_weight', type=float, default=0.0,
                            help='Weight for foreground crop consistency loss (default: 0.0=disabled)')
clip_loss_group.add_argument('--crop_morph_k_min', type=int, default=3,
                            help='Min kernel size for random morphology (e.g., 3)')
clip_loss_group.add_argument('--crop_morph_k_max', type=int, default=15,
                            help='Max kernel size for random morphology (e.g., 15)')


# ==================== BiomedCLIP with Visual Prompts (简化) ====================
# (与原文件相同)
parser.add_argument('--biomedclip_path', type=str, default='/root/models/BiomedCLIP',
                    help='Root directory containing BiomedCLIP weights and config JSON')
parser.add_argument('--biomedclip_num_prompts', type=int, default=4,
                    help='Number of prompts per prompt group')
parser.add_argument('--biomedclip_embed_dim', type=int, default=768,
                    help='Embedding dimension for BiomedCLIP prompts')
parser.add_argument('--biomedclip_init_std', type=float, default=0.02,
                    help='Initialization std for prompt parameters')
parser.add_argument('--biomedclip_prompt_scale_init', type=float, default=1.0,
                    help='Initial prompt scaling factor')
parser.add_argument('--biomedclip_lr', type=float, default=1e-4,
                    help='Learning rate for BiomedCLIP optimizer')
parser.add_argument('--biomedclip_weight_decay', type=float, default=1e-2,
                    help='Weight decay for BiomedCLIP optimizer')
parser.add_argument('--biomedclip_disable_scale', action='store_true',
                    help='Disable learnable prompt scaling (fix to 1.0)')

# ==================== Text Sampler ====================
# (与原文件相同)
parser.add_argument('--text_root', type=str, default='/app/MixDSemi/SynFoCLIP/code/text',
                    help='Directory containing dataset text description JSON files')
parser.add_argument('--text_num_subsets', type=int, default=4,
                    help='Number of subsets to sample per iteration (0 disables sampling)')

args = parser.parse_args()

args.biomedclip_enable_scale = not args.biomedclip_disable_scale

TEXT_DATASET_DEFAULTS = {
    'fundus': 'Fundus',
    'prostate': 'ProstateSlice',
    'MNMS': 'MNMS',
    'BUSI': 'BUSI',
}

# (新) 辅助函数：应用随机形态学操作
def apply_random_morphology(mask_batch, k_min, k_max, device):
    """
    对一个批次的 2D 掩码应用随机的腐蚀或扩张。
    
    Args:
        mask_batch (torch.Tensor): [B, H, W], 0或1的掩码
        k_min (int): 最小核大小
        k_max (int): 最大核大小
        device (torch.device): 目标设备
        
    Returns:
        torch.Tensor: [B, H, W], 经过随机形态学操作的掩码
    """
    if k_max <= k_min:
        k_max = k_min + 2
    
    output_masks = []
    # 转换为 numpy (CPU) 以使用 cv2
    mask_batch_np = mask_batch.cpu().numpy().astype(np.uint8)
    
    for mask_np in mask_batch_np:
        # 随机选择操作
        op = random.choice(['dilate', 'erode', 'none'])
        
        if op == 'none':
            output_masks.append(mask_np)
            continue
            
        # 随机选择核大小 (必须是奇数)
        k_size = random.randrange(k_min, k_max + 1, 2)
        kernel = np.ones((k_size, k_size), np.uint8)
        
        if op == 'dilate':
            # 扩张
            new_mask = cv2.dilate(mask_np, kernel, iterations=1)
        else: # op == 'erode'
            # 腐蚀
            new_mask = cv2.erode(mask_np, kernel, iterations=1)
            
        output_masks.append(new_mask)
        
    # 转换回 tensor 并移动到 GPU
    return torch.from_numpy(np.stack(output_masks, axis=0)).to(device)


def train(args, snapshot_path):
    writer = SummaryWriter(snapshot_path + '/log')
    max_iterations = args.max_iterations
    
    # ... (数据增强、数据加载等逻辑保持不变) ...
    weak = transforms.Compose([tr.RandomScaleCrop(patch_size),
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
    domain_num = args.domain_num
    domain = list(range(1,domain_num+1))
    if args.dataset == 'fundus':
        domain_len = [50, 99, 320, 320]
    elif args.dataset == 'prostate':
        domain_len = [225, 305, 136, 373, 338, 133]
    elif args.dataset == 'MNMS':
        domain_len = [1030, 1342, 525, 550]
    elif args.dataset == 'BUSI':
        domain_len = [350, 168]
    lb_domain = args.lb_domain
    data_num = domain_len[lb_domain-1]
    if args.lb_ratio > 0:
        lb_num = int(sum(domain_len) * args.lb_ratio)
    else:
        lb_num = args.lb_num
    lb_idxs = list(range(lb_num))
    unlabeled_idxs = list(range(lb_num, data_num))
    
    # (修改) lb_kwargs - 我们假设 dataloader_dc 总是会加载标签
    lb_kwargs = dict(
        base_dir=train_data_path,
        phase='train',
        splitid=lb_domain,
        domain=[lb_domain],
        selected_idxs=lb_idxs,
        weak_transform=weak,
        normal_toTensor=normal_toTensor,
        img_size=patch_size,
        preprocess_dir=dataset_preprocess_dir,
        llm_model=args.llm_model,
        describe_nums=args.describe_nums,
        return_score=True,
        allow_missing_scores=False,
    )
    ulb_kwargs = dict(
        base_dir=train_data_path,
        phase='train',
        splitid=lb_domain,
        domain=domain,
        selected_idxs=unlabeled_idxs,
        weak_transform=weak,
        strong_tranform=strong,
        normal_toTensor=normal_toTensor,
        img_size=patch_size,
        preprocess_dir=dataset_preprocess_dir,
        llm_model=args.llm_model,
        describe_nums=args.describe_nums,
        return_score=True,
        allow_missing_scores=False,
    )
    lb_dataset = dataset(**lb_kwargs)
    ulb_dataset = dataset(**ulb_kwargs)

    # ... (TextSampler 逻辑保持不变) ...
    text_dataset_key = TEXT_DATASET_DEFAULTS.get(args.dataset)
    text_sampler = TextSampler(args.text_root)
    all_texts_dict, text_all_descriptions = text_sampler.load_texts(
        dataset=text_dataset_key,
        llm=args.llm_model,
        describe_nums=args.describe_nums,
    )
    per_type_subsets, text_subsets = text_sampler.sample_subsets(
        all_texts_dict,
        num_samples=args.text_num_subsets
    )
    logging.info(
        "TextSampler loaded dataset '%s' (types=%d, subsets=%s)",
        text_dataset_key,
        len(text_all_descriptions),
        len(text_subsets),
    )

    # --- 实例化 Invariant-Only BiomedCLIP ---
    biomedclip_model = None
    biomedclip_optimizer = None
    biomedclip_preprocess = None
    biomedclip_tokenizer = None

    unet_device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    # 使用新的 builder
    biomedclip_model, biomedclip_preprocess, biomedclip_tokenizer = build_invariant_prompt_image_encoder(
        model_path=args.biomedclip_path,
        device=str(unet_device),
        num_prompts=args.biomedclip_num_prompts,
        embed_dim=args.biomedclip_embed_dim,
        init_std=args.biomedclip_init_std,
        prompt_scale_init=args.biomedclip_prompt_scale_init,
        enable_prompt_scale=args.biomedclip_enable_scale,
        freeze_backbone=True,
    )
    
    trainable_params = [p for p in biomedclip_model.prompt_learner.parameters() if p.requires_grad]
    biomedclip_optimizer = optim.AdamW(
        trainable_params,
        lr=args.biomedclip_lr,
        weight_decay=args.biomedclip_weight_decay,
    )
    logging.info(
        "BiomedCLIP Invariant-Only prompt encoder initialized on %s (trainable params=%d)",
        unet_device,
        sum(p.numel() for p in trainable_params) if trainable_params else 0,
    )

    # ... (DataLoader 逻辑保持不变) ...
    lb_loader = DataLoader(lb_dataset, batch_size=args.label_bs, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    ulb_loader = DataLoader(ulb_dataset, batch_size=args.unlabel_bs, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    logging.info("Using random sampler for unlabeled data loader.")
    lb_dataloader = cycle(lb_loader)
    ulb_dataloader = cycle(ulb_loader)
    
    logging.info(f"Total iterations: {max_iterations}")
    
    logging.info(
        "CLIP loss weights: mv_anchor=%.3f, sw_reg=%.3f, crop_cons=%.3f (Ortho removed)",
        args.clip_loss_mv_anchor_weight,
        args.clip_loss_sw_reg_weight,
        args.clip_loss_crop_weight, # (新) Log
    )
    
    # ... (AMP 逻辑保持不变) ...
    amp_enabled = bool(args.amp)
    scaler = GradScaler(enabled=amp_enabled)
    amp_cm = autocast if amp_enabled else contextlib.nullcontext
    logging.info(f"AMP (Mixed Precision) enabled: {amp_enabled}")

    # ... (CLIP text Process 逻辑保持不变) ...
    logging.info("Starting text feature pre-calculation for semantic anchors...")
    with torch.no_grad():
        all_text_tokenized = biomedclip_tokenizer(text_all_descriptions).to(unet_device)
        all_text_features = biomedclip_model.encode_text(all_text_tokenized)
    all_text_features = F.normalize(all_text_features, dim=-1)
    a_global_anchor = torch.mean(all_text_features, dim=0, keepdim=True)
    a_global_anchor = F.normalize(a_global_anchor, dim=-1) # [1, 512]

    p_k_anchors = []
    for sub_set in text_subsets:
        with torch.no_grad():
            tokens = biomedclip_tokenizer(sub_set).to(unet_device)
            sub_text_features = biomedclip_model.encode_text(tokens)
            p_k = torch.mean(sub_text_features, dim=0)
            p_k = F.normalize(p_k, dim=-1)
            p_k_anchors.append(p_k)
    p_k_anchors = torch.stack(p_k_anchors, dim=0) # [K, 512]
    print(f"p_k_anchors shape: {p_k_anchors.shape}")
    logging.info("Completed BiomedCLIP text encoding for all descriptions and subsets.")

    # --- Main training loop (添加 Crop Loss) ---
    p_bar = tqdm(range(1, max_iterations + 1), desc=f'VPT Invariant-Only Training')
    
    if biomedclip_model is not None:
        biomedclip_model.train()

    for iter_num in p_bar:
        lb_sample = next(lb_dataloader)
        ulb_sample = next(ulb_dataloader)
        
        # (修改) 加载有标签数据 (图像 + 掩码)
        lb_unet_size_x_w = lb_sample['unet_size_img']
        lb_unet_size_y = lb_sample['unet_size_label'] # <-- (新)
        
        ulb_unet_size_x_w, ulb_unet_size_x_s = ulb_sample['unet_size_img'], ulb_sample['unet_size_strong_aug']
        
        lb_unet_size_x_w, ulb_unet_size_x_w, ulb_unet_size_x_s = lb_unet_size_x_w.cuda(), ulb_unet_size_x_w.cuda(), ulb_unet_size_x_s.cuda()
        lb_unet_size_y = lb_unet_size_y.cuda() # <-- (新)

        # (新) 将 Y 标签转换为二元掩码 (0=背景, 1=前景)
        # (这个逻辑是从您之前的 U-Net 脚本中复制过来的)
        if args.dataset == 'fundus':
            lb_mask = (lb_unet_size_y <= 128) * 2
            lb_mask[lb_unet_size_y == 0] = 1
            lb_mask = lb_mask - 1 # 转换为 0=背景, 1=前景
        elif args.dataset == 'prostate':
            lb_mask = lb_unet_size_y.eq(0).long()
            lb_mask = 1 - lb_mask # 转换为 0=背景, 1=前景
        elif args.dataset == 'MNMS':
            lb_mask = (lb_unet_size_y > 0).long() # 0=背景, 1=LV/Myo/RV
        elif args.dataset == 'BUSI':
            lb_mask = lb_unet_size_y.eq(255).long() # 0=背景, 1=前景
        
        # (新) 计算前景裁切一致性损失
        loss_crop_consistency = torch.tensor(0.0).to(unet_device)
        if args.clip_loss_crop_weight > 0:
            # 1. 对掩码应用随机扩张/腐蚀
            perturbed_mask = apply_random_morphology(
                lb_mask, # [B, H, W]
                args.crop_morph_k_min,
                args.crop_morph_k_max,
                unet_device
            )
            
            # 2. 创建裁切后的图像 (只保留前景)
            lb_mask_f = perturbed_mask.unsqueeze(1).float() # [B, 1, H, W]
            lb_x_w_cropped = lb_unet_size_x_w * lb_mask_f
            
            with torch.no_grad():
                # 3. 获取“全图”特征 (作为稳定目标)
                f_lb_full = biomedclip_model.encode_image_from_tensor(
                    lb_unet_size_x_w, biomedclip_preprocess
                )['image_features'].detach()
            
            # 4. 获取“裁切图”特征
            f_lb_cropped = biomedclip_model.encode_image_from_tensor(
                lb_x_w_cropped, biomedclip_preprocess
            )['image_features']
            
            # 5. 计算一致性损失
            loss_crop_consistency = (1 - F.cosine_similarity(f_lb_full, f_lb_cropped)).mean()

        # ========== 1. CLIP 侧 (VPT 教师) 训练 (简化) ==========

        weak_images = torch.cat([lb_unet_size_x_w, ulb_unet_size_x_w], dim=0)
        strong_images_ulb = ulb_unet_size_x_s
        
        weak_images_preprocessed = biomedclip_model.encode_image_from_tensor(
            weak_images, biomedclip_preprocess, return_tokens=False
        )['image_features']
        
        strong_images_ulb_preprocessed = biomedclip_model.encode_image_from_tensor(
            strong_images_ulb, biomedclip_preprocess, return_tokens=False
        )['image_features']
        
        f_all_w = weak_images_preprocessed
        f_ulb_s = strong_images_ulb_preprocessed
        f_ulb_w = f_all_w[args.label_bs:]

        with amp_cm():
            # 1. loss_mv_anchor
            loss_mv = (1 - F.cosine_similarity(f_all_w, a_global_anchor)).mean()
            sim_matrix = f_all_w @ p_k_anchors.T 
            loss_anchor = (1 - sim_matrix.max(dim=1).values).mean()
            loss_mv_anchor = loss_mv + loss_anchor
            
            # 2. loss_sw_reg
            loss_sw_reg = (1 - F.cosine_similarity(f_ulb_w, f_ulb_s.detach())).mean()
            
            # 3. Total loss (添加 crop loss)
            loss_clip_total = (
                args.clip_loss_mv_anchor_weight * loss_mv_anchor
                + args.clip_loss_sw_reg_weight * loss_sw_reg
                + args.clip_loss_crop_weight * loss_crop_consistency # (新)
            )

        biomedclip_optimizer.zero_grad()
        
        scaler.scale(loss_clip_total).backward()
        scaler.step(biomedclip_optimizer)
        scaler.update()

        # ========== All U-Net and SSL Logic REMOVED ==========
        
        # Update Tensorboard (添加 crop loss)
        writer.add_scalar('train/clip_loss_total', loss_clip_total.item(), iter_num)
        writer.add_scalar('train/clip_loss_mv_anchor', loss_mv_anchor.item(), iter_num)
        writer.add_scalar('train/clip_loss_sw_reg', loss_sw_reg.item(), iter_num)
        writer.add_scalar('train/clip_loss_crop_cons', loss_crop_consistency.item(), iter_num) # (新)
        writer.add_scalar('train/biomedclip_lr', biomedclip_optimizer.param_groups[0]['lr'], iter_num)

        # Update tqdm description (添加 crop loss)
        p_bar.set_description(
            'iter %d: clip_loss=%.4f (mv_a:%.4f, sw_r:%.4f, crop:%.4f)'
            % (iter_num, loss_clip_total.item(), 
               loss_mv_anchor.item(),
               loss_sw_reg.item(),
               loss_crop_consistency.item()) # (新)
        )

        # Log to console (添加 crop loss)
        if iter_num % 50 == 0:
            logging.info(
                "iter %d: clip_loss=%.6f, mv_anchor=%.6f, sw_reg=%.6f, crop_cons=%.6f",
                iter_num,
                loss_clip_total.item(),
                loss_mv_anchor.item(),
                loss_sw_reg.item(),
                loss_crop_consistency.item() # (新)
            )
            
    if p_bar is not None:
        p_bar.close()

    # ========== Save Final VPT Weights ==========
    if args.save_model:
        save_text_vpt = "vpt_final_weights.pth"
        save_best_vpt = os.path.join(snapshot_path, save_text_vpt)
        if biomedclip_model is not None:
            biomedclip_model.save_prompts(save_best_vpt)
            logging.info(f"Saved final Invariant-Only VPT (prompt_learner) weights to {save_best_vpt}")

    writer.close()
    logging.info("VPT Invariant-Only training finished.")


if __name__ == "__main__":
    # (新) 添加数据集特定配置
    if args.dataset == 'fundus':
        train_data_path='/app/MixDSemi/data/Fundus'
        part = ['cup', 'disc']
        dataset = FundusSegmentation
        num_channels = 3
        patch_size = 256
        num_classes = 2 # (新) 需要这个
        min_v, max_v = 0.5, 1.5
        fillcolor = 255
        if args.max_iterations is None:
            args.max_iterations = 30000
        if args.domain_num >=4:
            args.domain_num = 4
    elif args.dataset == 'prostate':
        train_data_path="/app/MixDSemi/data/ProstateSlice"
        num_channels = 1
        patch_size = 384
        num_classes = 1 # (新) 需要这个
        part = ['base'] 
        dataset = ProstateSegmentation
        min_v, max_v = 0.1, 2
        fillcolor = 255
        if args.max_iterations is None:
            args.max_iterations = 60000
        if args.domain_num >= 6:
            args.domain_num = 6
    elif args.dataset == 'MNMS':
        train_data_path="/app/MixDSemi/data/mnms"
        num_channels = 1
        patch_size = 288
        num_classes = 3 # (新) 需要这个
        part = ['lv', 'myo', 'rv'] 
        dataset = MNMSSegmentation
        min_v, max_v = 0.1, 2
        fillcolor = 0
        if args.max_iterations is None:
            args.max_iterations = 60000
        if args.domain_num >= 4:
            args.domain_num = 4
    elif args.dataset == 'BUSI':
        train_data_path="/app/MixDSemi/data/Dataset_BUSI_with_GT"
        num_channels = 1
        patch_size = 256
        num_classes = 1 # (新) 需要这个
        part = ['base'] 
        dataset = BUSISegmentation
        min_v, max_v = 0.1, 2
        fillcolor = 0
        if args.max_iterations is None:
            args.max_iterations = 30000
        if args.domain_num >= 2:
            args.domain_num = 2
            
    if len(args.save_name) == 0:
        args.save_name = f'vpt_invariant_crop_lb{args.lb_num}_dm{args.lb_domain}' # 新的默认名称
    snapshot_path = "../model/" + args.dataset + f"/{sys.argv[0].split('.')[0]}/" + args.save_name + "/"
    
    preprocess_defaults = {
        'fundus': '/app/MixDSemi/SynFoCLIP/preprocess/Fundus',
        'prostate': '/app/MixDSemi/SynFoCLIP/preprocess/ProstateSlice',
        'MNMS': '/app/MixDSemi/SynFoCLIP/preprocess/MNMS',
        'BUSI': '/app/MixDSemi/SynFoCLIP/preprocess/BUSI',
    }
    dataset_preprocess_dir = args.preprocess_dir if args.preprocess_dir is not None else preprocess_defaults.get(args.dataset)

    if args.label_bs is None or args.unlabel_bs is None:
        if args.lb_num < 8:
            args.label_bs = 2
            args.unlabel_bs = 2
        else:
            args.label_bs = 4
            args.unlabel_bs = 4
    
    # ... (os.environ, 确定性设置 保持不变) ...
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    elif not args.overwrite:
        raise Exception('file {} is exist!'.format(snapshot_path))
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copy('./{}'.format(sys.argv[0]), snapshot_path + '/{}'.format(sys.argv[0]))

    # ... (保存配置 保持不变) ...
    import json
    config_dict = vars(args).copy()
    config_dict['snapshot_path'] = snapshot_path
    config_dict['train_data_path'] = train_data_path
    config_dict['dataset_class'] = dataset.__name__
    config_dict['num_channels'] = num_channels
    config_dict['patch_size'] = patch_size
    config_dict['num_classes'] = num_classes
    config_dict['dataset_preprocess_dir'] = dataset_preprocess_dir
    
    config_file = os.path.join(snapshot_path, 'training_config.json')
    with open(config_file, 'w') as f:
        json.dump(config_dict, f, indent=4, sort_keys=True)
    print(f"Training configuration saved to: {config_file}")

    # ... (日志 保持不变) ...
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    cmd = " ".join(["python"] + sys.argv)
    logging.info(cmd)
    logging.info(str(args))
    logging.info(f"Configuration file: {config_file}")

    train(args, snapshot_path)
