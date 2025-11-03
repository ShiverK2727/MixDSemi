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
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

import dataloaders.custom_transforms as tr
from dataloaders.dataloader import BUSISegmentation as BUSISegmentationTest
from dataloaders.dataloader import FundusSegmentation as FundusSegmentationTest
from dataloaders.dataloader import MNMSSegmentation as MNMSSegmentationTest
from dataloaders.dataloader import ProstateSegmentation as ProstateSegmentationTest
from dataloaders.dataloader_dc import BUSISegmentation, FundusSegmentation, MNMSSegmentation, ProstateSegmentation
from networks.unet_model import UNet
from torch.cuda.amp import GradScaler, autocast

from utils import losses, metrics
from utils.conf import available_conf_strategies, compute_self_consistency
from utils.domain_curriculum import DomainDistanceCurriculumSampler, build_distance_curriculum
from utils.frequency import apply_midss_frequency_augmentation
from utils.label_ops import to_2d, to_3d
from utils.training import Statistics, cycle, obtain_cutmix_box
from utils.training_helpers import (
    compute_consistency_weight,
    compute_piecewise_threshold,
    parse_alpha_schedule,
    update_ema_variables,
)
from val import ValidationConfig, run_validation

# -------------------------------------------------------------------------

CONF_STRATEGIES = available_conf_strategies()
if not CONF_STRATEGIES:
    raise RuntimeError("No confidence strategies registered. Check utils.conf module registration.")

parser = argparse.ArgumentParser()

# ==================== Basic Configuration ====================
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
parser.add_argument("--max_iterations", type=int, default=None,
                    help="Maximum iterations to train (auto-set per dataset if None)")
parser.add_argument('--num_eval_iter', type=int, default=500,
                    help='Evaluate model every N iterations')
parser.add_argument("--base_lr", type=float, default=None,
                    help="Base learning rate (default: 0.03 with polynomial decay if None)")
parser.add_argument('--warmup', action='store_true',
                    help='Enable learning rate warmup from lower lr to base_lr')
parser.add_argument('--warmup_period', type=int, default=2000,
                    help='Number of warmup iterations (only used when --warmup is enabled)')
parser.add_argument('--amp', type=int, default=1,
                    help='Use mixed precision training (1: enabled, 0: disabled)')

# ==================== Data Configuration ====================
parser.add_argument("--label_bs", type=int, default=None,
                    help="Labeled batch size per GPU (auto-set based on lb_num if None)")
parser.add_argument("--unlabel_bs", type=int, default=None,
                    help="Unlabeled batch size per GPU (auto-set based on lb_num if None)")
parser.add_argument("--test_bs", type=int, default=1,
                    help='Batch size for evaluation')
parser.add_argument('--domain_num', type=int, default=6,
                    help='Total number of domains in dataset')
parser.add_argument('--lb_domain', type=int, default=1,
                    help='Domain ID to use for labeled data')
parser.add_argument('--lb_num', type=int, default=40,
                    help='Number of labeled samples (used when lb_ratio=0)')
parser.add_argument('--lb_ratio', type=float, default=0,
                    help='Labeled data ratio of total dataset (overrides lb_num if > 0)')

# ==================== Data Augmentation ====================
parser.add_argument('--use_freq_aug', action='store_true',
                    help='Enable frequency-domain augmentation (MiDSS style)')
parser.add_argument('--LB', type=float, default=0.01,
                    help='Low-frequency band ratio for frequency augmentation')

# ==================== Semi-Supervised Learning ====================
parser.add_argument("--ema_decay", type=float, default=0.99,
                    help="EMA decay rate for teacher model")

# ==================== Pseudo-Label Threshold ====================
parser.add_argument("--threshold", type=float, default=0.95,
                    help="Fixed confidence threshold for pseudo-labels (used when enable_piecewise_tau=False)")
parser.add_argument('--enable_piecewise_tau', action='store_true',
                    help='Enable curriculum-adaptive threshold that increases with stages')
parser.add_argument('--tau_min', type=float, default=0.80,
                    help='Minimum threshold at initial curriculum stage (piecewise mode)')
parser.add_argument('--tau_max', type=float, default=0.95,
                    help='Maximum threshold at final curriculum stage (piecewise mode)')

# ==================== Symmetric Gradient Guidance (SymGD) ====================
parser.add_argument('--use_symgd', dest='use_symgd', action='store_true',
                    help='Enable Symmetric Gradient guidance (default)')
parser.add_argument('--no_symgd', dest='use_symgd', action='store_false',
                    help='Disable Symmetric Gradient guidance')
parser.set_defaults(use_symgd=True)
parser.add_argument('--symgd_mode', type=str, default='full',
                    choices=['full', 'ul_only'],
                    help='SymGD mode: full (UL+LU) or ul_only (UL only)')
parser.add_argument('--ul_weight', type=float, default=1.0,
                    help='Weight for UL CutMix pseudo-label loss')
parser.add_argument('--lu_weight', type=float, default=1.0,
                    help='Weight for LU CutMix pseudo-label loss')
parser.add_argument('--cons_weight', type=float, default=1.0,
                    help='Weight for strong-augmentation consistency loss')

# ==================== Domain Curriculum Learning ====================
parser.add_argument('--dc_parts', type=int, default=5,
                    help='Number of curriculum partitions')
parser.add_argument('--dc_distance_mode', type=str, default='prototype',
                    choices=['prototype', 'sqrt_prod'],
                    help='Distance metric for curriculum: prototype (L2) or sqrt_prod')
parser.add_argument('--dc_quantile_batching', action='store_true',
                    help='Enable stratified quantile sampling within each unlabeled batch (uses score distribution)')
parser.add_argument('--dc_quantile_bins', type=int, default=0,
                    help='Number of quantile bins per batch (default: unlabeled batch size)')
parser.add_argument('--expend_test_samples', type=int, default=32,
                    help='Number of samples to test from next partition')
parser.add_argument('--expend_test_steps_interval', type=int, default=100,
                    help='Test next partition every N steps')
parser.add_argument('--expend_max_steps', type=int, default=1000,
                    help='Maximum steps before forcing curriculum expansion')
parser.add_argument('--use_curr_conf', action='store_true',
                    help='Require current-partition confidence to meet threshold for expansion')
parser.add_argument('--curr_conf_threshold', type=float, default=0.6,
                    help='Confidence threshold for current partition samples')
parser.add_argument('--curr_conf_samples', type=int, default=32,
                    help='Number of current-partition samples to test for confidence')
parser.add_argument('--use_next_conf', action='store_true',
                    help='Require next-partition confidence to meet threshold for expansion')
parser.add_argument('--expand_conf_threshold', type=float, default=0.6,
                    help='Confidence threshold for next partition expansion')

# ==================== Preprocessing & Confidence Strategy ====================
parser.add_argument('--preprocess_dir', type=str, default=None,
                    help='Override preprocessing directory for score tensors')
parser.add_argument('--llm_model', type=str, default='gemini',
                    choices=['gemini', 'GPT5', 'DeepSeek'],
                    help='LLM model used to generate score tensors')
parser.add_argument('--describe_nums', type=int, default=40,
                    choices=[20, 40, 60, 80],
                    help='Number of textual descriptions for preprocessing')
parser.add_argument('--conf_strategy', type=str, default='robust',
                    choices=CONF_STRATEGIES,
                    help='[dice, robust, robust_no_fg, js_teacher_student]')
parser.add_argument('--conf_teacher_temp', type=float, default=1.0,
                    help='Temperature for softening teacher probabilities')

# ==================== Loss Configuration ====================
loss_group = parser.add_argument_group('Loss Configuration')
loss_group.add_argument('--consistency', type=float, default=1.0,
                        help='Consistency loss base weight')
loss_group.add_argument('--consistency_rampup', type=float, default=200.0,
                        help='Sigmoid ramp-up length (in epochs) for consistency loss')


def train(args, snapshot_path):
    writer = SummaryWriter(snapshot_path + '/log')
    validation_config = ValidationConfig(
        dataset=args.dataset,
        parts=part,
        patch_size=patch_size,
        dice_fn=dice_calcu[args.dataset],
    )
    
    # Learning rate configuration
    # Modified to match MiDSS: use 0.03 with polynomial decay (no warmup by default)
    if args.base_lr is None:
        base_lr = 0.03  # Match MiDSS default learning rate
        args.warmup = False  # Force disable warmup for default lr (MiDSS doesn't use warmup)
        logging.info("Using default learning rate: 0.03 with polynomial decay (no warmup, matching MiDSS)")
    else:
        base_lr = args.base_lr
        if args.warmup:
            logging.info(f"Using custom learning rate: {base_lr} with warmup enabled (warmup_period={args.warmup_period})")
        else:
            logging.info(f"Using custom learning rate: {base_lr} with polynomial decay (no warmup)")

    def create_model(ema=False):
        # Network definition (Simple U-Net)
        model = UNet(
            n_channels=num_channels,
            n_classes=num_classes + 1,
        )
        if ema:
            for param in model.parameters():
                param.detach_()
        return model.cuda()

    unet_model = create_model()
    ema_unet_model = create_model(ema=True)

    max_iterations = args.max_iterations
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

    # Instantiate DomainDistanceCurriculumSampler
    curriculum_partitions = args.dc_parts

    labeled_scores = lb_dataset.get_scores()
    unlabeled_scores = ulb_dataset.get_scores()
    unlabeled_indices = list(range(len(unlabeled_scores)))

    _, _, _, partitions = build_distance_curriculum(
        labeled_scores,
        unlabeled_scores,
        unlabeled_indices,
        num_partitions=curriculum_partitions,
        distance_mode=getattr(args, 'dc_distance_mode', 'prototype'),
    )

    stratified_enabled = bool(getattr(args, 'dc_quantile_batching', False))
    stratified_bins = int(getattr(args, 'dc_quantile_bins', 0) or 0)
    if stratified_enabled:
        if args.unlabel_bs is None or args.unlabel_bs <= 0:
            logging.warning(
                "Quantile batching requested but unlabeled batch size is invalid (value=%s); disabling.",
                str(args.unlabel_bs),
            )
            stratified_enabled = False
        elif stratified_bins <= 0:
            stratified_bins = int(args.unlabel_bs)
        if stratified_enabled and stratified_bins <= 1:
            logging.warning(
                "Quantile batching requested but computed bin count=%d; disabling.",
                stratified_bins,
            )
            stratified_enabled = False

    sampler_kwargs = dict(
        partitions=partitions,
        initial_stage=1,
        seed=args.seed,
        shuffle=True,
    )
    if stratified_enabled:
        sampler_kwargs.update(
            score_lookup=unlabeled_scores,
            stratified_batch_size=stratified_bins,
        )
        logging.info(
            "Curriculum sampler stratified quantile batching ENABLED (bins=%d, unlabeled_bs=%d)",
            stratified_bins,
            args.unlabel_bs,
        )
    curriculum_sampler = DomainDistanceCurriculumSampler(**sampler_kwargs)

    test_dataset = []
    test_dataloader = []
    for i in range(1, domain_num+1):
        cur_dataset = dataset_test(
            base_dir=train_data_path,
            phase='test',
            splitid=-1,
            domain=[i],
            normal_toTensor=normal_toTensor,
            img_size=patch_size
        )
        test_dataset.append(cur_dataset)
    lb_loader = DataLoader(lb_dataset, batch_size=args.label_bs, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    if curriculum_sampler is not None:
        ulb_loader = DataLoader(ulb_dataset, batch_size=args.unlabel_bs, sampler=curriculum_sampler, num_workers=2, pin_memory=True, drop_last=True)
        logging.info("Using curriculum sampler for unlabeled data loader.")
    else:
        ulb_loader = DataLoader(ulb_dataset, batch_size=args.unlabel_bs, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
        logging.info("Using random sampler for unlabeled data loader.")
    lb_dataloader = cycle(lb_loader)
    ulb_dataloader = cycle(ulb_loader)
    for i in range(0,domain_num):
        cur_dataloader = DataLoader(test_dataset[i], batch_size = args.test_bs, shuffle=False, num_workers=0, pin_memory=True)
        test_dataloader.append(cur_dataloader)

    iter_num = 0
    last_stage_update_iter = 0

    # Initialize threshold based on piecewise strategy
    if args.enable_piecewise_tau:
        threshold = compute_piecewise_threshold(
            current_stage=curriculum_sampler.stage,
            num_stages=args.dc_parts,
            tau_min=args.tau_min,
            tau_max=args.tau_max,
        )
        logging.info(
            "Piecewise threshold ENABLED: initial stage=%d, threshold=%.4f (tau_min=%.2f, tau_max=%.2f)",
            curriculum_sampler.stage,
            threshold,
            args.tau_min,
            args.tau_max
        )
    else:
        threshold = args.threshold
        logging.info("Using fixed threshold: %.4f", threshold)

    # set to train
    ce_loss = CrossEntropyLoss(reduction='none')
    softmax, sigmoid, multi = True, False, False
    dice_loss = losses.DiceLossWithMask(num_classes+1)
    
    
    # Initialize optimizer
    if args.warmup:
        b_lr = base_lr / args.warmup_period
        logging.info(f"Warmup enabled: initial lr = {b_lr:.6f}, will warmup to {base_lr} over {args.warmup_period} iterations")
    else:
        b_lr = base_lr
    
    unet_optimizer = optim.SGD(unet_model.parameters(), lr=b_lr, momentum=0.9, weight_decay=0.0001)
    
    logging.info(f"Total iterations: {max_iterations}, evaluation every {args.num_eval_iter} iterations")
    
    # Log frequency augmentation status
    if args.use_freq_aug:
        logging.info(f"Frequency-domain augmentation ENABLED (MiDSS style): LB={args.LB}, progressive degree scaling")
    else:
        logging.info("Frequency-domain augmentation DISABLED")

    if args.use_symgd:
        logging.info("Symmetric Guidance ENABLED (mode=%s)", args.symgd_mode)
    else:
        logging.info("Symmetric Guidance DISABLED; UL/LU CutMix branches skipped")

    logging.info(
        "Confidence strategy: %s (teacher temp=%.2f)",
        args.conf_strategy,
        args.conf_teacher_temp,
    )

    # Calculate number of evaluation cycles
    num_eval_cycles = max_iterations // args.num_eval_iter
    
    # Initialize best metrics
    best_dice = [0.0] * n_part
    best_dice_iter = [-1] * n_part
    best_avg_dice = 0.0
    best_avg_dice_iter = -1
    dice_of_best_avg = [0.0] * n_part

    amp_enabled = bool(args.amp)
    scaler = GradScaler(enabled=amp_enabled)
    amp_cm = autocast if amp_enabled else contextlib.nullcontext

    if curriculum_sampler.stage < len(partitions):
        next_partition_indices = partitions[curriculum_sampler.stage]
        if len(next_partition_indices) > args.expend_test_samples:
            selected_indices = torch.randperm(len(next_partition_indices))[:args.expend_test_samples].tolist()
            next_partition_indices = [next_partition_indices[i] for i in selected_indices]
        test_expend_dataset = Subset(ulb_dataset, next_partition_indices)
        test_expend_dataloader = DataLoader(test_expend_dataset, batch_size=args.test_bs, shuffle=False, num_workers=2, pin_memory=True)
        test_expend_domain_partition = curriculum_sampler.stage + 1
        logging.info(
            "Initialized expend test dataset with %d samples from partition %d",
            len(next_partition_indices),
            test_expend_domain_partition,
        )
    else:
        test_expend_dataset = None
        test_expend_dataloader = None
        test_expend_domain_partition = None

    # Main training loop - iterate by evaluation cycles
    for eval_cycle in range(num_eval_cycles):
        # Set epoch for curriculum sampler so shuffling is reproducible per-cycle
        if curriculum_sampler is not None:
            try:
                curriculum_sampler.set_epoch(eval_cycle)
            except Exception:
                # Defensive: if sampler doesn't implement set_epoch, ignore
                pass

        unet_model.train()
        ema_unet_model.train()
        p_bar = tqdm(range(args.num_eval_iter), desc=f'Cycle {eval_cycle+1}/{num_eval_cycles}')
        unet_ulb_dice_sta = [Statistics() for _ in range(n_part)]
        
        for i_batch in range(1, args.num_eval_iter+1):
            # Learning rate schedule: warmup + polynomial decay (matching MiDSS)
            if args.warmup and iter_num < args.warmup_period:
                # Warmup phase
                lr_scale = (iter_num + 1) / args.warmup_period
                for param_group in unet_optimizer.param_groups:
                    param_group['lr'] = base_lr * lr_scale
            else:
                # Polynomial decay (matching MiDSS)
                lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                for param_group in unet_optimizer.param_groups:
                    param_group['lr'] = lr_
            
            lb_sample = next(lb_dataloader)
            ulb_sample = next(ulb_dataloader)
            lb_unet_size_x_w, lb_unet_size_y = lb_sample['unet_size_img'], lb_sample['unet_size_label']
            ulb_unet_size_x_w, ulb_unet_size_x_s, ulb_unet_size_y = ulb_sample['unet_size_img'], ulb_sample['unet_size_strong_aug'], ulb_sample['unet_size_label']
            
            if args.dataset == 'fundus':
                lb_unet_size_mask = (lb_unet_size_y<=128) * 2
                lb_unet_size_mask[lb_unet_size_y==0] = 1
                ulb_unet_size_mask = (ulb_unet_size_y<=128) * 2
                ulb_unet_size_mask[ulb_unet_size_y==0] = 1
            elif args.dataset == 'prostate':
                lb_unet_size_mask = lb_unet_size_y.eq(0).long()
                ulb_unet_size_mask = ulb_unet_size_y.eq(0).long()
            elif args.dataset == 'MNMS':
                lb_unet_size_mask = lb_unet_size_y.long()
                ulb_unet_size_mask = ulb_unet_size_y.long()
            elif args.dataset == 'BUSI':
                lb_unet_size_mask = lb_unet_size_y.eq(255).long()
                ulb_unet_size_mask = ulb_unet_size_y.eq(255).long()
            lb_unet_size_x_w, ulb_unet_size_x_w, ulb_unet_size_x_s = lb_unet_size_x_w.cuda(), ulb_unet_size_x_w.cuda(), ulb_unet_size_x_s.cuda()
            lb_unet_size_mask, ulb_unet_size_mask = lb_unet_size_mask.cuda(), ulb_unet_size_mask.cuda()

            # ========== Frequency-domain augmentation (MiDSS style) ==========
            if args.use_freq_aug:
                degree = iter_num / max_iterations
                move_transx = apply_midss_frequency_augmentation(
                    labeled_images=lb_unet_size_x_w,
                    unlabeled_images=ulb_unet_size_x_w,
                    degree=degree,
                    low_band=args.LB,
                )
            else:
                move_transx = lb_unet_size_x_w

            apply_ul = args.symgd_mode in ("full", "ul_only")
            apply_lu = args.symgd_mode == "full"
            cutmix_active = apply_ul or apply_lu

            mask_teacher: torch.Tensor | None = None
            mask_ul: torch.Tensor | None = None
            mask_lu: torch.Tensor | None = None
            mask_w: torch.Tensor | None = None
            ensemble: torch.Tensor | None = None
            unet_size_img_box: torch.Tensor | None = None
            unet_size_label_box: torch.Tensor | None = None
            unet_size_pseudo_label_ul: torch.Tensor | None = None
            unet_size_pseudo_label_lu: torch.Tensor | None = None
            pseudo_label_w: torch.Tensor | None = None

            with torch.no_grad():
                # 1. Original unlabeled image pseudo-labels
                unet_logits_ulb_x_w = ema_unet_model(ulb_unet_size_x_w)
                unet_prob_ulb_x_w = torch.softmax(unet_logits_ulb_x_w, dim=1)
                unet_prob, unet_pseudo_label = torch.max(unet_prob_ulb_x_w, dim=1)
                mask_teacher = (unet_prob > threshold).unsqueeze(1).float()

                mask_w = mask_teacher.clone()
                pseudo_label_w = unet_pseudo_label.long()
                ensemble = torch.ones_like(mask_teacher)
                mask_ul = torch.zeros_like(mask_teacher)
                mask_lu = torch.zeros_like(mask_teacher)

                if cutmix_active:
                    # Generate CutMix box (only used when SymGD is active)
                    device = lb_unet_size_x_w.device
                    unet_size_label_box = torch.stack(
                        [
                            obtain_cutmix_box(
                                img_size=patch_size,
                                p=1.0,
                                device=device,
                            )
                            for _ in range(len(ulb_unet_size_x_w))
                        ],
                        dim=0,
                    )
                    unet_size_img_box = unet_size_label_box.unsqueeze(1)

                    if apply_ul:
                        # UL mixing (Unlabeled background + Labeled foreground)
                        ulb_x_w_ul = ulb_unet_size_x_w * (1 - unet_size_img_box) + lb_unet_size_x_w * unet_size_img_box
                        logits_w_ul = ema_unet_model(ulb_x_w_ul)
                        prob_w_ul = torch.softmax(logits_w_ul, dim=1)
                        conf_w_ul, pseudo_label_w_ul = torch.max(prob_w_ul, dim=1)
                        mask_w_ul = (conf_w_ul > threshold).unsqueeze(1).float()

                        mask_ul = mask_teacher.clone()
                        mask_ul[unet_size_img_box.expand_as(mask_ul) == 1] = 1
                        unet_size_pseudo_label_ul = (
                            unet_pseudo_label * (1 - unet_size_label_box)
                            + lb_unet_size_mask * unet_size_label_box
                        ).long()
                    else:
                        mask_w_ul = None

                    if apply_lu:
                        # LU mixing (Labeled background + Unlabeled foreground)
                        ulb_x_w_lu = lb_unet_size_x_w * (1 - unet_size_img_box) + ulb_unet_size_x_w * unet_size_img_box
                        logits_w_lu = ema_unet_model(ulb_x_w_lu)
                        prob_w_lu = torch.softmax(logits_w_lu, dim=1)
                        conf_w_lu, pseudo_label_w_lu = torch.max(prob_w_lu, dim=1)
                        mask_w_lu = (conf_w_lu > threshold).unsqueeze(1).float()

                        mask_lu = mask_teacher.clone()
                        mask_lu[unet_size_img_box.expand_as(mask_lu) == 0] = 1
                        unet_size_pseudo_label_lu = (
                            lb_unet_size_mask * (1 - unet_size_label_box)
                            + unet_pseudo_label * unet_size_label_box
                        ).long()
                    else:
                        mask_w_lu = None

                    if apply_ul and apply_lu:
                        mask_w = mask_w_ul * (1 - unet_size_img_box) + mask_w_lu * unet_size_img_box
                        pseudo_label_w = (
                            pseudo_label_w_ul * (1 - unet_size_label_box)
                            + pseudo_label_w_lu * unet_size_label_box
                        ).long()
                    elif apply_ul:
                        mask_w = mask_w_ul
                        pseudo_label_w = pseudo_label_w_ul.long()
                    elif apply_lu:
                        mask_w = mask_w_lu
                        pseudo_label_w = pseudo_label_w_lu.long()

                if args.use_symgd:
                    ensemble = (pseudo_label_w == unet_pseudo_label).unsqueeze(1).float() * mask_teacher
                    mask_w = mask_w.clone()
                    mask_w[ensemble == 0] = 0
                else:
                    ensemble = torch.ones_like(mask_teacher)

            mask = mask_teacher
            device = lb_unet_size_x_w.device
            zero = lb_unet_size_x_w.new_tensor(0.0)

            # Calculate dice on unlabeled data for monitoring
            if args.dataset == 'fundus':
                unet_pseudo_label_2layer = to_2d(unet_pseudo_label)
                ulb_unet_size_mask_2layer = to_2d(ulb_unet_size_mask)
                unet_ulb_dice = dice_calcu[args.dataset](np.asarray(unet_pseudo_label_2layer.cpu()), ulb_unet_size_mask_2layer.cpu())
            else:
                unet_ulb_dice = dice_calcu[args.dataset](np.asarray(unet_pseudo_label.cpu()), ulb_unet_size_mask.cpu())
            for n, p in enumerate(part):
                unet_ulb_dice_sta[n].update(unet_ulb_dice[n])

            if args.consistency_rampup > 0:
                elapsed_since_stage_update = iter_num - last_stage_update_iter
                expected_steps_base = max(1, int(getattr(args, 'expend_max_steps', 1)))

                try:
                    is_last_stage = (curriculum_sampler is not None and curriculum_sampler.stage >= len(partitions))
                except Exception:
                    is_last_stage = False

                if is_last_stage:
                    expected_steps_base = max(1, int(max_iterations - iter_num))

                if args.consistency_rampup >= 1:
                    steps_per_epoch = max(1.0, expected_steps_base / float(args.consistency_rampup))
                else:
                    steps_per_epoch = float(expected_steps_base)

                epoch_in_stage = float(elapsed_since_stage_update) / steps_per_epoch

                consistency_weight = compute_consistency_weight(
                    epoch=epoch_in_stage,
                    base_weight=args.consistency,
                    rampup_length=float(args.consistency_rampup),
                )
            else:
                consistency_weight = compute_consistency_weight(
                    epoch=0.0,
                    base_weight=args.consistency,
                    rampup_length=1.0,
                )

            with amp_cm():
                # Cache additional representations for downstream tasks
                unet_logits_lb_x_w = unet_model(lb_unet_size_x_w)
                unet_unsup_loss_ul = zero
                if apply_ul and unet_size_img_box is not None and unet_size_pseudo_label_ul is not None:
                    ulb_unet_size_x_s_ul = ulb_unet_size_x_s * (1 - unet_size_img_box) + move_transx * unet_size_img_box
                    unet_logits_ulb_x_s_ul = unet_model(ulb_unet_size_x_s_ul)
                    unet_unsup_loss_ul = (ce_loss(unet_logits_ulb_x_s_ul, unet_size_pseudo_label_ul) *
                                          mask_ul.squeeze(1)).mean() + \
                                         dice_loss(
                                             unet_logits_ulb_x_s_ul,
                                             unet_size_pseudo_label_ul.unsqueeze(1),
                                             mask=mask_ul,
                                             softmax=softmax,
                                             sigmoid=sigmoid,
                                             multi=multi,
                                         )

                unet_unsup_loss_lu = zero
                if apply_lu and unet_size_img_box is not None and unet_size_pseudo_label_lu is not None:
                    ulb_unet_size_x_s_lu = move_transx * (1 - unet_size_img_box) + ulb_unet_size_x_s * unet_size_img_box
                    unet_logits_ulb_x_s_lu = unet_model(ulb_unet_size_x_s_lu)
                    unet_unsup_loss_lu = (ce_loss(unet_logits_ulb_x_s_lu, unet_size_pseudo_label_lu) *
                                          mask_lu.squeeze(1)).mean() + \
                                         dice_loss(
                                             unet_logits_ulb_x_s_lu,
                                             unet_size_pseudo_label_lu.unsqueeze(1),
                                             mask=mask_lu,
                                             softmax=softmax,
                                             sigmoid=sigmoid,
                                             multi=multi,
                                         )

                unet_logits_ulb_x_s = unet_model(ulb_unet_size_x_s)
                unet_unsup_loss_s = (ce_loss(unet_logits_ulb_x_s, pseudo_label_w) *
                                     mask_w.squeeze(1)).mean() + \
                                    dice_loss(
                                        unet_logits_ulb_x_s,
                                        pseudo_label_w.unsqueeze(1),
                                        mask=mask_w,
                                        softmax=softmax,
                                        sigmoid=sigmoid,
                                        multi=multi,
                                    )

                unet_sup_loss = ce_loss(unet_logits_lb_x_w, lb_unet_size_mask).mean() + \
                    dice_loss(unet_logits_lb_x_w, lb_unet_size_mask.unsqueeze(1), softmax=softmax, sigmoid=sigmoid, multi=multi)

            ul_term = args.ul_weight * unet_unsup_loss_ul
            lu_term = args.lu_weight * unet_unsup_loss_lu
            cons_term = args.cons_weight * unet_unsup_loss_s
            
            loss = unet_sup_loss + consistency_weight * (ul_term + lu_term + cons_term)

            unet_optimizer.zero_grad()
            if args.amp:
                scaler.scale(loss).backward()
                scaler.step(unet_optimizer)
                scaler.update()
            else:
                loss.backward()
                unet_optimizer.step()

            update_ema_variables(unet_model, ema_unet_model, args.ema_decay, iter_num)

            iter_num = iter_num + 1
    
            if (
                iter_num % args.expend_test_steps_interval == 0
                and curriculum_sampler.stage < args.dc_parts
            ):  
                use_curr = getattr(args, 'use_curr_conf', False)
                use_next = getattr(args, 'use_next_conf', False)

                next_conf = None
                if use_next:
                    next_conf = compute_self_consistency(
                        args,
                        unet_model,
                        ema_unet_model,
                        test_expend_dataloader,
                        args.dataset,
                        dice_calcu,
                        to_2d,
                    )

                curr_conf = None
                if use_curr:
                    active_indices = curriculum_sampler.get_active_indices()
                    if active_indices:
                        sample_count = min(args.curr_conf_samples, len(active_indices))
                        if sample_count > 0:
                            sampled_indices = random.sample(active_indices, sample_count)
                            current_subset = Subset(ulb_dataset, sampled_indices)
                            current_loader = DataLoader(
                                current_subset,
                                batch_size=args.test_bs,
                                shuffle=False,
                                num_workers=2,
                                pin_memory=True,
                            )
                            curr_conf = compute_self_consistency(
                                args,
                                unet_model,
                                ema_unet_model,
                                current_loader,
                                args.dataset,
                                dice_calcu,
                                to_2d,
                            )

                if next_conf is not None:
                    logging.info(
                        "Next-partition confidence at iter %d (partition %s): %.4f",
                        iter_num,
                        str(test_expend_domain_partition) if test_expend_domain_partition is not None else "N/A",
                        next_conf,
                    )
                if curr_conf is not None:
                    logging.info(
                        "Current-partition confidence at iter %d: %.4f (active=%d)",
                        iter_num,
                        curr_conf,
                        len(curriculum_sampler.get_active_indices()),
                    )

                elapsed_since_stage_update = iter_num - last_stage_update_iter

                # ========== Curriculum Expansion Gate ==========
                # Determine whether to expand curriculum based on confidence metrics.
                # 
                # Expansion Criteria:
                #   1. Confidence-based (configurable):
                #      - If --use_curr_conf: require current partition conf >= curr_conf_threshold
                #      - If --use_next_conf: require next partition conf >= expand_conf_threshold
                #      - If both enabled: require BOTH conditions (AND logic)
                #      - If neither enabled: default to next-partition check (with warning)
                #   
                #   2. Time-based (fallback):
                #      - Force expansion if elapsed steps >= expend_max_steps
                
                # Step 1: Ensure at least one confidence criterion is enabled
                if not use_curr and not use_next:
                    use_next = True
                    logging.warning(
                        "Neither --use_curr_conf nor --use_next_conf enabled; defaulting to --use_next_conf"
                    )

                # Step 2: Evaluate individual confidence conditions
                curr_ok = True  # Default True (no constraint if not enabled)
                next_ok = True  # Default True (no constraint if not enabled)
                
                if use_curr:
                    curr_ok = (curr_conf is not None) and (curr_conf >= args.curr_conf_threshold)
                
                if use_next:
                    next_ok = (next_conf is not None) and (next_conf >= args.expand_conf_threshold)

                # Step 3: Determine expansion based on enabled criteria
                should_expand = False
                
                if use_curr and use_next:
                    # Both criteria enabled: require BOTH to be satisfied (AND logic)
                    if curr_ok and next_ok:
                        should_expand = True
                        logging.info(
                            "✓ Curriculum expansion: BOTH confidences meet thresholds "
                            "(curr=%.4f >= %.4f, next=%.4f >= %.4f)",
                            curr_conf if curr_conf is not None else float('nan'),
                            args.curr_conf_threshold,
                            next_conf if next_conf is not None else float('nan'),
                            args.expand_conf_threshold,
                        )
                
                elif use_curr:
                    # Only current-partition criterion enabled
                    if curr_ok:
                        should_expand = True
                        logging.info(
                            "✓ Curriculum expansion: current-partition confidence meets threshold "
                            "(curr=%.4f >= %.4f)",
                            curr_conf,
                            args.curr_conf_threshold,
                        )
                
                elif use_next:
                    # Only next-partition criterion enabled
                    if next_ok:
                        should_expand = True
                        logging.info(
                            "✓ Curriculum expansion: next-partition confidence meets threshold "
                            "(next=%.4f >= %.4f)",
                            next_conf,
                            args.expand_conf_threshold,
                        )

                # Step 4: Fallback expansion based on maximum elapsed steps
                if not should_expand and elapsed_since_stage_update >= args.expend_max_steps:
                    should_expand = True
                    logging.info(
                        "✓ Curriculum expansion: maximum steps reached (elapsed=%d >= %d)",
                        elapsed_since_stage_update,
                        args.expend_max_steps,
                    )

                # Step 5: Execute curriculum expansion
                if should_expand:
                    prev_stage = curriculum_sampler.stage
                    curriculum_sampler.expand()
                    last_stage_update_iter = iter_num

                    # Update threshold if piecewise strategy is enabled
                    if args.enable_piecewise_tau:
                        old_threshold = threshold
                        threshold = compute_piecewise_threshold(
                            current_stage=curriculum_sampler.stage,
                            num_stages=args.dc_parts,
                            tau_min=args.tau_min,
                            tau_max=args.tau_max,
                        )
                        logging.info(
                            "→ Curriculum stage advanced: %d→%d (iter %d), threshold: %.4f→%.4f",
                            prev_stage,
                            curriculum_sampler.stage,
                            iter_num,
                            old_threshold,
                            threshold
                        )
                    else:
                        logging.info(
                            "→ Curriculum stage advanced: %d→%d (iter %d), threshold: %.4f (fixed)",
                            prev_stage,
                            curriculum_sampler.stage,
                            iter_num,
                            threshold,
                        )

                    # Prepare next partition test dataset if available
                    if curriculum_sampler.stage < len(partitions):
                        next_partition_indices = partitions[curriculum_sampler.stage]
                        if len(next_partition_indices) > args.expend_test_samples:
                            selected_indices = torch.randperm(len(next_partition_indices))[:args.expend_test_samples].tolist()
                            next_partition_indices = [next_partition_indices[i] for i in selected_indices]
                        test_expend_dataset = Subset(ulb_dataset, next_partition_indices)
                        test_expend_dataloader = DataLoader(
                            test_expend_dataset,
                            batch_size=args.test_bs,
                            shuffle=False,
                            num_workers=2,
                            pin_memory=True,
                        )
                        test_expend_domain_partition = curriculum_sampler.stage + 1
                        logging.info(
                            "Updated expend test dataset with %d samples from partition %d",
                            len(next_partition_indices),
                            test_expend_domain_partition,
                        )
                    else:
                        test_expend_dataset = None
                        test_expend_dataloader = None
                        test_expend_domain_partition = None

            for n, p in enumerate(part):
                text = 'train/unet_ulb_{}_dice'.format(p)
                writer.add_scalar(text, unet_ulb_dice[n], iter_num)
            writer.add_scalar('train/mask_ul', mask_ul.mean(), iter_num)
            writer.add_scalar('train/mask_lu', mask_lu.mean(), iter_num)
            writer.add_scalar('train/mask_w', mask_w.mean(), iter_num)
            writer.add_scalar('train/ensemble_ratio', ensemble.mean(), iter_num)
            writer.add_scalar('train/loss', loss.item(), iter_num)
            writer.add_scalar('train/unet_sup_loss', unet_sup_loss.item(), iter_num)
            writer.add_scalar('train/unet_unsup_loss_ul', unet_unsup_loss_ul.item(), iter_num)
            writer.add_scalar('train/unet_unsup_loss_lu', unet_unsup_loss_lu.item(), iter_num)
            writer.add_scalar('train/unet_unsup_loss_s', unet_unsup_loss_s.item(), iter_num)
            writer.add_scalar('train/ul_weight_term', ul_term.item(), iter_num)
            writer.add_scalar('train/lu_weight_term', lu_term.item(), iter_num)
            writer.add_scalar('train/cons_weight_term', cons_term.item(), iter_num)
            writer.add_scalar('train/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('train/learning_rate', unet_optimizer.param_groups[0]['lr'], iter_num)
            writer.add_scalar('train/threshold', threshold, iter_num)
            writer.add_scalar('train/curriculum_stage', curriculum_sampler.stage, iter_num)
            # Every 50 iters also emit a condensed log line to the main logfile/terminal with key scalars
            if iter_num % 50 == 0:
                logging.info(
                    "iter %d: loss=%.6f, L=%.6f, UL=%.6f, LU=%.6f, S=%.6f, cons_w=%.6f",
                    iter_num,
                    float(loss.item() if hasattr(loss, 'item') else float(loss)),
                    float(unet_sup_loss.item() if hasattr(unet_sup_loss, 'item') else float(unet_sup_loss)),
                    float(unet_unsup_loss_ul.item() if hasattr(unet_unsup_loss_ul, 'item') else float(unet_unsup_loss_ul)),
                    float(unet_unsup_loss_lu.item() if hasattr(unet_unsup_loss_lu, 'item') else float(unet_unsup_loss_lu)),
                    float(unet_unsup_loss_s.item() if hasattr(unet_unsup_loss_s, 'item') else float(unet_unsup_loss_s)),
                    float(consistency_weight),
                )
            if p_bar is not None:
                p_bar.update()

            if args.dataset == 'fundus':
                p_bar.set_description('iter %d: L:%.4f, SL:%.4f, UL:%.4f, LU:%.4f, S:%.4f, cons:%.4f, mask:%.4f, ud:%.4f,%.4f' 
                                        % (iter_num, loss.item(), unet_sup_loss.item(), 
                                           unet_unsup_loss_ul.item(), unet_unsup_loss_lu.item(), unet_unsup_loss_s.item(),
                                           consistency_weight, mask_ul.mean(), unet_ulb_dice[0], unet_ulb_dice[1]))
            elif args.dataset == 'prostate' or args.dataset == 'BUSI':
                p_bar.set_description('iter %d: L:%.4f, SL:%.4f, UL:%.4f, LU:%.4f, S:%.4f, cons:%.4f, mask:%.4f, ud:%.4f' 
                                        % (iter_num, loss.item(), unet_sup_loss.item(), 
                                           unet_unsup_loss_ul.item(), unet_unsup_loss_lu.item(), unet_unsup_loss_s.item(),
                                           consistency_weight, mask_ul.mean(), unet_ulb_dice[0]))
            elif args.dataset == 'MNMS':
                p_bar.set_description('iter %d: L:%.4f, SL:%.4f, UL:%.4f, LU:%.4f, S:%.4f, cons:%.4f, mask:%.4f, ud:%.4f,%.4f,%.4f' 
                                        % (iter_num, loss.item(), unet_sup_loss.item(), 
                                           unet_unsup_loss_ul.item(), unet_unsup_loss_lu.item(), unet_unsup_loss_s.item(),
                                           consistency_weight, mask_ul.mean(), unet_ulb_dice[0], unet_ulb_dice[1], unet_ulb_dice[2]))
            if iter_num % args.num_eval_iter == 0:
                if args.dataset == 'fundus':
                    logging.info('iteration %d : loss : %f, unet_sup_loss : %f, UL : %f, LU : %f, S : %f, cons_w : %f, mask_ul : %f, ud:%.6f,%.6f' 
                                        % (iter_num, loss.item(), unet_sup_loss.item(), 
                                           unet_unsup_loss_ul.item(), unet_unsup_loss_lu.item(), unet_unsup_loss_s.item(),
                                           consistency_weight, mask_ul.mean(), unet_ulb_dice_sta[0].avg, unet_ulb_dice_sta[1].avg))
                elif args.dataset == 'prostate' or args.dataset == 'BUSI':
                    logging.info('iteration %d : loss : %f, unet_sup_loss : %f, UL : %f, LU : %f, S : %f, cons_w : %f, mask_ul : %f, ud:%.6f' 
                                        % (iter_num, loss.item(), unet_sup_loss.item(), 
                                           unet_unsup_loss_ul.item(), unet_unsup_loss_lu.item(), unet_unsup_loss_s.item(),
                                           consistency_weight, mask_ul.mean(), unet_ulb_dice_sta[0].avg))
                elif args.dataset == 'MNMS':
                    logging.info('iteration %d : loss : %f, unet_sup_loss : %f, UL : %f, LU : %f, S : %f, cons_w : %f, mask_ul : %f, ud:%.6f,%.6f,%.6f' 
                                        % (iter_num, loss.item(), unet_sup_loss.item(), 
                                           unet_unsup_loss_ul.item(), unet_unsup_loss_lu.item(), unet_unsup_loss_s.item(),
                                           consistency_weight, mask_ul.mean(), unet_ulb_dice_sta[0].avg, unet_ulb_dice_sta[1].avg, unet_ulb_dice_sta[2].avg))
                text = ''
                for n, p in enumerate(part):
                    text += 'unet_ulb_%s_dice:%f' % (p, unet_ulb_dice_sta[n].avg)
                    if n != n_part-1:
                        text += ', '
                logging.info(text)

        if p_bar is not None:
            p_bar.close()


        logging.info('test unet model')
        text = ''
        current_iter = (eval_cycle + 1) * args.num_eval_iter
        val_dice = run_validation(
            model=unet_model,
            dataloaders=test_dataloader,
            iteration=current_iter,
            writer=writer,
            config=validation_config,
        )
        
        for n, p in enumerate(part):
            if val_dice[n] > best_dice[n]:
                best_dice[n] = val_dice[n]
                best_dice_iter[n] = current_iter
            text += 'val_%s_best_dice: %f at iteration %d' % (p, best_dice[n], best_dice_iter[n])
            text += ', '
        if sum(val_dice) / len(val_dice) > best_avg_dice:
            best_avg_dice = sum(val_dice) / len(val_dice)
            best_avg_dice_iter = current_iter
            for n, p in enumerate(part):
                dice_of_best_avg[n] = val_dice[n]
            save_text = "unet_avg_dice_best_model.pth"
            save_best = os.path.join(snapshot_path, save_text)
            logging.info('save cur best avg unet model to {}'.format(save_best))
            if args.save_model:
                torch.save(unet_model.state_dict(), save_best)
        text += 'val_best_avg_dice: %f at iteration %d' % (best_avg_dice, best_avg_dice_iter)
        if n_part > 1:
            for n, p in enumerate(part):
                text += ', %s_dice: %f' % (p, dice_of_best_avg[n])
        logging.info(text)

    writer.close()


if __name__ == "__main__":
    args = parser.parse_args()

    if len(args.save_name) == 0:
        args.save_name = f'unet_only_lb{args.lb_num}_dm{args.lb_domain}'
    snapshot_path = "../model/" + args.dataset + f"/{sys.argv[0].split('.')[0]}/" + args.save_name + "/"
    
    if args.dataset == 'fundus':
        train_data_path='/app/MixDSemi/data/Fundus'
        part = ['cup', 'disc']
        dataset = FundusSegmentation
        dataset_test = FundusSegmentationTest
        num_channels = 3
        patch_size = 256
        num_classes = 2
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
        num_classes = 1
        part = ['base'] 
        dataset = ProstateSegmentation
        dataset_test = ProstateSegmentationTest
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
        num_classes = 3
        part = ['lv', 'myo', 'rv'] 
        dataset = MNMSSegmentation
        dataset_test = MNMSSegmentationTest
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
        num_classes = 1
        part = ['base'] 
        dataset = BUSISegmentation
        dataset_test = BUSISegmentationTest
        min_v, max_v = 0.1, 2
        fillcolor = 0
        if args.max_iterations is None:
            args.max_iterations = 30000
        if args.domain_num >= 2:
            args.domain_num = 2
    
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

    if num_classes > 1:
        multimask_output = True
    else:
        multimask_output = False
    n_part = len(part)
    dice_calcu = {'fundus':metrics.dice_coeff_2label, 'prostate':metrics.dice_coeff, 'MNMS':metrics.dice_coeff_3label, 'BUSI':metrics.dice_coeff}

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

    # Save training configuration to JSON file for easy inspection
    import json
    config_dict = vars(args).copy()
    config_dict['snapshot_path'] = snapshot_path
    config_dict['train_data_path'] = train_data_path
    config_dict['dataset_class'] = dataset.__name__
    config_dict['num_channels'] = num_channels
    config_dict['patch_size'] = patch_size
    config_dict['num_classes'] = num_classes
    config_dict['dataset_preprocess_dir'] = dataset_preprocess_dir
    config_dict['dataset_test_class'] = dataset_test.__name__
    
    config_file = os.path.join(snapshot_path, 'training_config.json')
    with open(config_file, 'w') as f:
        json.dump(config_dict, f, indent=4, sort_keys=True)
    print(f"Training configuration saved to: {config_file}")

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    cmd = " ".join(["python"] + sys.argv)
    logging.info(cmd)
    logging.info(str(args))
    logging.info(f"Configuration file: {config_file}")

    train(args, snapshot_path)
