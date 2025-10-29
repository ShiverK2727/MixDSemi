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
from medpy.metric import binary
from networks.unet_model import UNet
from scipy.ndimage import zoom
from torch.cuda.amp import GradScaler, autocast

from utils import losses, metrics, ramps
from utils.conf import available_conf_strategies, compute_self_consistency
from utils.domain_curriculum import DomainDistanceCurriculumSampler, build_distance_curriculum
from utils.label_ops import to_2d, to_3d
from utils.tp_ram import extract_amp_spectrum, source_to_target_freq, source_to_target_freq_midss
from utils.training import Statistics, cycle, obtain_cutmix_box

# -------------------------------------------------------------------------

CONF_STRATEGIES = available_conf_strategies()
if not CONF_STRATEGIES:
    raise RuntimeError("No confidence strategies registered. Check utils.conf module registration.")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='prostate', choices=['fundus', 'prostate', 'MNMS', 'BUSI'])
parser.add_argument("--save_name", type=str, default="", help="experiment_name")
parser.add_argument("--overwrite", action='store_true')
parser.add_argument("--max_iterations", type=int, default=None, help="maximum iterations to train, if not set, will use dataset default")
parser.add_argument('--num_eval_iter', type=int, default=500)
parser.add_argument("--deterministic", type=int, default=1, help="whether use deterministic training")
parser.add_argument("--base_lr", type=float, default=None, help="segmentation network learning rate, if not set, will use default 0.03 without warmup")
parser.add_argument("--seed", type=int, default=1337, help="random seed")
parser.add_argument("--gpu", type=str, default='0')
parser.add_argument("--threshold", type=float, default=0.95, help="confidence threshold for using pseudo-labels",)

parser.add_argument('--amp', type=int, default=1, help='use mixed precision training or not')

# Optional: enable MiDSS-style frequency-domain augmentation (kept separate from geometric/color transforms)
parser.add_argument('--use_freq_aug', action='store_true', help='Enable frequency-domain augmentation (MiDSS style)')
parser.add_argument('--LB', type=float, default=0.01, help='Low-frequency band ratio for frequency augmentation (default: 0.01)')

parser.add_argument("--label_bs", type=int, default=None, help="labeled_batch_size per gpu, auto-set based on lb_num if not provided")
parser.add_argument("--unlabel_bs", type=int, default=None, help="unlabeled_batch_size per gpu, auto-set based on lb_num if not provided")
parser.add_argument("--test_bs", type=int, default=1)
parser.add_argument('--domain_num', type=int, default=6)
parser.add_argument('--lb_domain', type=int, default=1)
parser.add_argument('--lb_num', type=int, default=40)
parser.add_argument('--lb_ratio', type=float, default=0)
# costs
parser.add_argument("--ema_decay", type=float, default=0.99, help="ema_decay")
parser.add_argument("--consistency", type=float, default=1.0, help="consistency")
parser.add_argument("--consistency_rampup", type=float, default=200.0, help="consistency_rampup")

parser.add_argument('--save_model',action='store_true')
parser.add_argument('--warmup', action='store_true', help='If activated, warp up the learning from a lower lr to the base_lr')
parser.add_argument('--warmup_period', type=int, default=5000,
                    help='Warp up iterations, only valid whrn warmup is activated')
# domain curriculum
parser.add_argument('--dc_parts', type=int, default=5, help='Number of curriculum partitions')
parser.add_argument('--expend_test_samples', type=int, default=32, help='Number of samples to test from expend partition')
parser.add_argument('--expend_test_steps_interval', type=int, default=100, help='Number of steps between testing on expend partition')
parser.add_argument('--expend_max_steps', type=int, default=1000, help='Maximum number of steps for expend partition')
parser.add_argument('--expand_conf_threshold', type=float, default=0.6, help='Confidence threshold for expanding curriculum partition')
parser.add_argument('--curr_conf_threshold', type=float, default=0.6, help='Confidence threshold for current-partition samples')
parser.add_argument('--curr_conf_samples', type=int, default=32, help='Number of active-partition samples to draw for confidence test')
parser.add_argument('--ul_weight', type=float, default=1.0, help='Weight for UL CutMix pseudo-label loss')
parser.add_argument('--lu_weight', type=float, default=1.0, help='Weight for LU CutMix pseudo-label loss')
parser.add_argument('--cons_weight', type=float, default=1.0, help='Weight for strong-augmentation consistency loss')
parser.add_argument('--preprocess_dir', type=str, default=None, help='Override preprocessing root directory for score tensors')
parser.add_argument('--llm_model', type=str, default='gemini', choices=['gemini', 'GPT5', 'DeepSeek'], help='LLM model used to generate score tensors')
parser.add_argument('--describe_nums', type=int, default=40, choices=[20, 40, 60, 80], help='Number of textual descriptions used during preprocessing')
parser.add_argument('--conf_strategy', type=str, default='robust', choices=CONF_STRATEGIES, help='Confidence strategy for self-consistency filtering')
parser.add_argument('--conf_teacher_temp', type=float, default=1.0, help='Temperature to soften teacher probabilities before confidence evaluation')
parser.add_argument('--use_symgd', dest='use_symgd', action='store_true', help='Enable Symmetric Gradient guidance (default)')
parser.add_argument('--no_symgd', dest='use_symgd', action='store_false', help='Disable Symmetric Gradient guidance')
parser.set_defaults(use_symgd=True)
parser.add_argument('--symgd_mode', type=str, default='full', choices=['full', 'ul_only'], help='full: apply SymGD to UL and LU; ul_only: constrain to unlabeled CutMix branch only')
parser.add_argument('--dc_distance_mode', type=str, default='prototype', choices=['prototype','sqrt_prod'], help='Distance mode for domain curriculum: prototype (L2 to labeled prototype) or sqrt_prod (sqrt(delta_L * delta_U))')
# Piecewise threshold (curriculum-adaptive pseudo-label threshold)
parser.add_argument('--enable_piecewise_tau', action='store_true', help='Enable piecewise threshold that increases with curriculum stages')
parser.add_argument('--tau_min', type=float, default=0.80, help='Minimum threshold at initial curriculum stage (default: 0.80)')
parser.add_argument('--tau_max', type=float, default=0.95, help='Maximum threshold at final curriculum stage (default: 0.95)')
# Curriculum expansion controls: replace numeric weights with boolean switches.
parser.add_argument('--use_curr_conf', action='store_true', help='Require current-partition confidence to meet curr_conf_threshold to allow expansion')
parser.add_argument('--use_next_conf', action='store_true', help='Require next-partition confidence to meet expand_conf_threshold to allow expansion')
args = parser.parse_args()


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def get_piecewise_threshold(current_stage, num_stages, tau_min, tau_max):
    """
    Calculate piecewise threshold based on curriculum stage.
    
    Args:
        current_stage: Current curriculum stage (0-indexed)
        num_stages: Total number of curriculum stages
        tau_min: Minimum threshold at stage 0
        tau_max: Maximum threshold at final stage
    
    Returns:
        threshold: Linearly interpolated threshold for current stage
    
    Formula: tau_k = tau_min + (k / (N-1)) * (tau_max - tau_min)
    """
    if num_stages <= 1:
        return tau_max  # Single stage: use max threshold
    
    # Clamp stage to valid range
    stage = max(0, min(current_stage, num_stages - 1))
    
    # Linear interpolation
    threshold = tau_min + (stage / (num_stages - 1)) * (tau_max - tau_min)
    
    return threshold

def update_ema_variables(model, ema_model, alpha, global_step):
    # teacher network: ema_model
    # student network: model
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

@torch.no_grad()
def test(args, model, test_dataloader, iteration, writer):
    """
    Test function for model evaluation.
    
    Args:
        iteration: Current training iteration (not epoch, since training is iteration-based)
    
    Note: This is NOT binary classification for all datasets:
        - fundus: 2 foreground classes (cup, disc) + background = 3 classes
        - MNMS: 3 foreground classes (lv, myo, rv) + background = 4 classes  
        - prostate/BUSI: 1 foreground class + background = 2 classes (truly binary)
    """
    model.eval()
    val_dice = [0.0] * n_part
    val_dc, val_jc, val_hd, val_asd = [0.0] * n_part, [0.0] * n_part, [0.0] * n_part, [0.0] * n_part
    domain_num = len(test_dataloader)
    print(f"Starting evaluation over {domain_num} domains...")
    for i in range(domain_num):
        cur_dataloader = test_dataloader[i]
        domain_val_dice = [0.0] * n_part
        domain_val_dc, domain_val_jc, domain_val_hd, domain_val_asd = [0.0] * n_part, [0.0] * n_part, [0.0] * n_part, [0.0] * n_part
        domain_code = i+1
        for batch_num,sample in enumerate(cur_dataloader):
            print(f"  Evaluating domain {domain_code}, batch {batch_num+1}/{len(cur_dataloader)}")
            print(f"    Sample dc code: {sample['dc'][0].item()}")
            assert(domain_code == sample['dc'][0].item())
            mask = sample['label']
            if args.dataset == 'fundus':
                lb_mask = (mask<=128) * 2
                lb_mask[mask==0] = 1
                mask = lb_mask
            elif args.dataset == 'prostate':
                mask = mask.eq(0).long()
            elif args.dataset == 'MNMS':
                mask = mask.long()
            elif args.dataset == 'BUSI':
                mask = mask.eq(255).long()
            data = sample['unet_size_img'].cuda()
            output = model(data)
            pred_label = torch.max(torch.softmax(output,dim=1), dim=1)[1]
            pred_label = torch.from_numpy(zoom(pred_label.cpu(), (1, patch_size / data.shape[-2], patch_size / data.shape[-1]), order=0))
            
            if args.dataset == 'fundus':
                pred_label = to_2d(pred_label)
                mask = to_2d(mask)
                pred_onehot = pred_label.clone()
                mask_onehot = mask.clone()
            elif args.dataset == 'prostate' or args.dataset == 'BUSI':
                pred_onehot = pred_label.clone().unsqueeze(1)
                mask_onehot = mask.clone().unsqueeze(1)
            elif args.dataset == 'MNMS':
                pred_onehot = to_3d(pred_label)
                mask_onehot = to_3d(mask)
            dice = dice_calcu[args.dataset](np.asarray(pred_label.cpu()),mask.cpu())
            
            dc, jc, hd, asd = [0.0] * n_part, [0.0] * n_part, [0.0] * n_part, [0.0] * n_part
            for j in range(len(data)):
                for i, p in enumerate(part):
                    dc[i] += binary.dc(np.asarray(pred_onehot[j,i], dtype=bool),
                                            np.asarray(mask_onehot[j,i], dtype=bool))
                    jc[i] += binary.jc(np.asarray(pred_onehot[j,i], dtype=bool),
                                            np.asarray(mask_onehot[j,i], dtype=bool))
                    if pred_onehot[j,i].float().sum() < 1e-4:
                        hd[i] += 100
                        asd[i] += 100
                    else:
                        hd[i] += binary.hd95(np.asarray(pred_onehot[j,i], dtype=bool),
                                            np.asarray(mask_onehot[j,i], dtype=bool))
                        asd[i] += binary.asd(np.asarray(pred_onehot[j,i], dtype=bool),
                                            np.asarray(mask_onehot[j,i], dtype=bool))
            for i, p in enumerate(part):
                dc[i] /= len(data)
                jc[i] /= len(data)
                hd[i] /= len(data)
                asd[i] /= len(data)
            for i in range(len(domain_val_dice)):
                domain_val_dice[i] += dice[i]
                domain_val_dc[i] += dc[i]
                domain_val_jc[i] += jc[i]
                domain_val_hd[i] += hd[i]
                domain_val_asd[i] += asd[i]
        
        for i in range(len(domain_val_dice)):
            domain_val_dice[i] /= len(cur_dataloader)
            val_dice[i] += domain_val_dice[i]
            domain_val_dc[i] /= len(cur_dataloader)
            val_dc[i] += domain_val_dc[i]
            domain_val_jc[i] /= len(cur_dataloader)
            val_jc[i] += domain_val_jc[i]
            domain_val_hd[i] /= len(cur_dataloader)
            val_hd[i] += domain_val_hd[i]
            domain_val_asd[i] /= len(cur_dataloader)
            val_asd[i] += domain_val_asd[i]
        for n, p in enumerate(part):
            writer.add_scalar('unet_val/domain{}/val_{}_dice'.format(domain_code, p), domain_val_dice[n], iteration)
        text = 'domain%d iter %d :' % (domain_code, iteration)
        text += '\n\t'
        for n, p in enumerate(part):
            text += 'val_%s_dice: %f, ' % (p, domain_val_dice[n])
        text += '\n\t'
        for n, p in enumerate(part):
            text += 'val_%s_dc: %f, ' % (p, domain_val_dc[n])
        text += '\t'
        for n, p in enumerate(part):
            text += 'val_%s_jc: %f, ' % (p, domain_val_jc[n])
        text += '\n\t'
        for n, p in enumerate(part):
            text += 'val_%s_hd: %f, ' % (p, domain_val_hd[n])
        text += '\t'
        for n, p in enumerate(part):
            text += 'val_%s_asd: %f, ' % (p, domain_val_asd[n])
        logging.info(text)
        
    model.train()
    for i in range(len(val_dice)):
        val_dice[i] /= domain_num
        val_dc[i] /= domain_num
        val_jc[i] /= domain_num
        val_hd[i] /= domain_num
        val_asd[i] /= domain_num
    for n, p in enumerate(part):
        writer.add_scalar('unet_val/val_{}_dice'.format(p), val_dice[n], iteration)
    text = 'iteration %d :' % (iteration)
    text += '\n\t'
    for n, p in enumerate(part):
        text += 'val_%s_dice: %f, ' % (p, val_dice[n])
    text += '\n\t'
    for n, p in enumerate(part):
        text += 'val_%s_dc: %f, ' % (p, val_dc[n])
    text += '\t'
    for n, p in enumerate(part):
        text += 'val_%s_jc: %f, ' % (p, val_jc[n])
    text += '\n\t'
    for n, p in enumerate(part):
        text += 'val_%s_hd: %f, ' % (p, val_hd[n])
    text += '\t'
    for n, p in enumerate(part):
        text += 'val_%s_asd: %f, ' % (p, val_asd[n])
    logging.info(text)
    return val_dice
    
def train(args, snapshot_path):
    writer = SummaryWriter(snapshot_path + '/log')
    
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
        # Network definition
        model = UNet(n_channels = num_channels, n_classes = num_classes+1)
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

    # If user enabled frequency augmentation, perform a small sanity check here.
    if args.use_freq_aug:
        try:
            # pick two samples (use unlabeled if available, else labeled)
            if len(ulb_dataset) >= 2:
                a = ulb_dataset[0]['unet_size_img']
                b = ulb_dataset[1]['unet_size_img']
            elif len(ulb_dataset) == 1:
                a = ulb_dataset[0]['unet_size_img']
                b = ulb_dataset[0]['unet_size_img']
            else:
                a = lb_dataset[0]['unet_size_img']
                b = lb_dataset[0]['unet_size_img']

            # convert torch tensor (C,H,W) -> numpy (H,W,C) and scale to [0,1] if needed
            a_np = a.cpu().numpy()
            b_np = b.cpu().numpy()
            if a_np.ndim == 3:
                a_np = np.transpose(a_np, (1, 2, 0))
                b_np = np.transpose(b_np, (1, 2, 0))
            # run augmentation with default L=0.1
            mutated = source_to_target_freq(a_np, b_np, L=0.1)
            if mutated.shape != a_np.shape:
                logging.warning('Freq-aug sanity check: unexpected shape mutated %s vs src %s', mutated.shape, a_np.shape)
            else:
                logging.info('Freq-aug sanity check passed (shapes match): %s', mutated.shape)
        except Exception as e:
            logging.warning('Frequency augmentation sanity check failed: %s', str(e))



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
    curriculum_sampler = DomainDistanceCurriculumSampler(
        partitions,
        initial_stage=1,
        seed=args.seed,
        shuffle=True,
    )

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
        threshold = get_piecewise_threshold(
            current_stage=curriculum_sampler.stage,
            num_stages=args.dc_parts,
            tau_min=args.tau_min,
            tau_max=args.tau_max
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

    scaler = GradScaler()
    amp_cm = autocast if args.amp else contextlib.nullcontext

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
                # Apply frequency augmentation to labeled images using unlabeled amplitude
                # Move tensors to CPU, convert to numpy, apply freq aug, convert back
                move_transx_list = []
                lb_x_w_cpu = lb_unet_size_x_w.cpu()
                ulb_x_w_cpu = ulb_unet_size_x_w.cpu()
                
                for i in range(len(lb_x_w_cpu)):
                    # Convert from tensor range [-1, 1] to [0, 255] for frequency processing
                    # Assuming input is normalized to [-1, 1] (from dataloader)
                    lb_img_255 = ((lb_x_w_cpu[i] + 1) * 127.5).numpy()  # (C, H, W) in [0, 255]
                    ulb_img_255 = ((ulb_x_w_cpu[i] + 1) * 127.5).numpy()  # (C, H, W) in [0, 255]
                    
                    # Extract amplitude spectrum from unlabeled image
                    amp_trg = extract_amp_spectrum(ulb_img_255)
                    
                    # Apply frequency augmentation with progressive degree
                    degree = iter_num / max_iterations
                    img_freq = source_to_target_freq_midss(lb_img_255, amp_trg, L=args.LB, degree=degree)
                    
                    # Clip to valid range and convert back to tensor range [-1, 1]
                    img_freq = np.clip(img_freq, 0, 255).astype(np.float32)
                    move_transx_list.append(img_freq)
                
                # Stack and convert back to tensor
                move_transx = torch.tensor(np.array(move_transx_list), dtype=torch.float32)
                move_transx = move_transx / 127.5 - 1  # Convert back to [-1, 1]
                move_transx = move_transx.cuda()
            else:
                # If frequency augmentation is disabled, use original labeled images
                move_transx = lb_unet_size_x_w

            with amp_cm():
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

                # ========== Student Model Forward Passes ==========
                unet_logits_lb_x_w = unet_model(lb_unet_size_x_w)

                unet_unsup_loss_ul = zero
                if apply_ul and unet_size_img_box is not None and unet_size_pseudo_label_ul is not None:
                    ulb_unet_size_x_s_ul = ulb_unet_size_x_s * (1 - unet_size_img_box) + move_transx * unet_size_img_box
                    unet_logits_ulb_x_s_ul = unet_model(ulb_unet_size_x_s_ul)
                    unet_unsup_loss_ul = (
                        ce_loss(unet_logits_ulb_x_s_ul, unet_size_pseudo_label_ul) * mask_ul.squeeze(1)
                    ).mean() + dice_loss(
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
                    unet_unsup_loss_lu = (
                        ce_loss(unet_logits_ulb_x_s_lu, unet_size_pseudo_label_lu) * mask_lu.squeeze(1)
                    ).mean() + dice_loss(
                        unet_logits_ulb_x_s_lu,
                        unet_size_pseudo_label_lu.unsqueeze(1),
                        mask=mask_lu,
                        softmax=softmax,
                        sigmoid=sigmoid,
                        multi=multi,
                    )

                unet_logits_ulb_x_s = unet_model(ulb_unet_size_x_s)
                unet_unsup_loss_s = (
                    ce_loss(unet_logits_ulb_x_s, pseudo_label_w) * mask_w.squeeze(1)
                ).mean() + dice_loss(
                    unet_logits_ulb_x_s,
                    pseudo_label_w.unsqueeze(1),
                    mask=mask_w,
                    softmax=softmax,
                    sigmoid=sigmoid,
                    multi=multi,
                )
                
                # Calculate dice on unlabeled data for monitoring
                if args.dataset == 'fundus':
                    unet_pseudo_label_2layer = to_2d(unet_pseudo_label)
                    ulb_unet_size_mask_2layer = to_2d(ulb_unet_size_mask)
                    unet_ulb_dice = dice_calcu[args.dataset](np.asarray(unet_pseudo_label_2layer.cpu()), ulb_unet_size_mask_2layer.cpu())
                else:
                    unet_ulb_dice = dice_calcu[args.dataset](np.asarray(unet_pseudo_label.cpu()), ulb_unet_size_mask.cpu())
                for n, p in enumerate(part):
                    unet_ulb_dice_sta[n].update(unet_ulb_dice[n])

                # Supervised loss
                unet_sup_loss = ce_loss(unet_logits_lb_x_w, lb_unet_size_mask).mean() + \
                            dice_loss(unet_logits_lb_x_w, lb_unet_size_mask.unsqueeze(1), softmax=softmax, sigmoid=sigmoid, multi=multi)
                
                consistency_weight = get_current_consistency_weight(
                    iter_num // (max_iterations/args.consistency_rampup))

                # UL unsupervised loss
                unet_unsup_loss_ul = (ce_loss(unet_logits_ulb_x_s_ul, unet_size_pseudo_label_ul) * 
                                      mask_ul.squeeze(1)).mean() + \
                                     dice_loss(unet_logits_ulb_x_s_ul, unet_size_pseudo_label_ul.unsqueeze(1), 
                                              mask=mask_ul, softmax=softmax, sigmoid=sigmoid, multi=multi)

                # LU unsupervised loss
                unet_unsup_loss_lu = (ce_loss(unet_logits_ulb_x_s_lu, unet_size_pseudo_label_lu) * 
                                      mask_lu.squeeze(1)).mean() + \
                                     dice_loss(unet_logits_ulb_x_s_lu, unet_size_pseudo_label_lu.unsqueeze(1), 
                                              mask=mask_lu, softmax=softmax, sigmoid=sigmoid, multi=multi)

                # Strong augmentation consistency loss
                unet_unsup_loss_s = (ce_loss(unet_logits_ulb_x_s, pseudo_label_w) * 
                                     mask_w.squeeze(1)).mean() + \
                                    dice_loss(unet_logits_ulb_x_s, pseudo_label_w.unsqueeze(1), 
                                             mask=mask_w, softmax=softmax, sigmoid=sigmoid, multi=multi)

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

                # New curriculum expansion gate:
                # - Use boolean switches (--use_curr_conf, --use_next_conf) to decide which
                #   confidence checks are required.
                # - If both switches are enabled, require BOTH current and next partition
                #   confidences to meet their respective thresholds to allow expansion.
                # - If only one switch is enabled, require that single metric to meet its threshold.
                # - If neither is enabled, default to using next-partition check (log a warning).
                
                if not use_curr and not use_next:
                    use_next = True
                    logging.warning(
                        "Neither --use_curr_conf nor --use_next_conf enabled; defaulting to --use_next_conf"
                    )

                curr_ok = True
                next_ok = True
                if use_curr:
                    curr_ok = (curr_conf is not None) and (curr_conf >= args.curr_conf_threshold)
                if use_next:
                    next_ok = (next_conf is not None) and (next_conf >= args.expand_conf_threshold)

                should_expand = False
                if use_curr and use_next:
                    if curr_ok and next_ok:
                        should_expand = True
                        logging.info(
                            "Curriculum expansion triggered: both current and next partition confidences meet thresholds (curr=%.4f, next=%.4f)",
                            curr_conf if curr_conf is not None else float('nan'),
                            next_conf if next_conf is not None else float('nan'),
                        )
                elif use_curr:
                    if curr_ok:
                        should_expand = True
                        logging.info(
                            "Curriculum expansion triggered by current-partition confidence (curr=%.4f >= %.4f)",
                            curr_conf,
                            args.curr_conf_threshold,
                        )
                elif use_next:
                    if next_ok:
                        should_expand = True
                        logging.info(
                            "Curriculum expansion triggered by next-partition confidence (next=%.4f >= %.4f)",
                            next_conf,
                            args.expand_conf_threshold,
                        )

                if not should_expand and elapsed_since_stage_update >= args.expend_max_steps:
                    should_expand = True
                    logging.info(
                        "Curriculum expansion triggered by max steps (elapsed=%d)",
                        elapsed_since_stage_update,
                    )

                if should_expand:
                    prev_stage = curriculum_sampler.stage
                    curriculum_sampler.expand()
                    last_stage_update_iter = iter_num

                    # Update threshold if piecewise strategy is enabled
                    if args.enable_piecewise_tau:
                        old_threshold = threshold
                        threshold = get_piecewise_threshold(
                            current_stage=curriculum_sampler.stage,
                            num_stages=args.dc_parts,
                            tau_min=args.tau_min,
                            tau_max=args.tau_max
                        )
                        logging.info(
                            "Curriculum expanded: stage %d→%d, threshold updated: %.4f→%.4f",
                            prev_stage,
                            curriculum_sampler.stage,
                            old_threshold,
                            threshold
                        )
                    else:
                        logging.info(
                            "Curriculum sampler advanced from stage %d to %d at iteration %d",
                            prev_stage,
                            curriculum_sampler.stage,
                            iter_num,
                        )

                    logging.info(
                        "Curriculum sampler advanced from stage %d to %d at iteration %d",
                        prev_stage,
                        curriculum_sampler.stage,
                        iter_num,
                    )
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
        val_dice = test(args, unet_model, test_dataloader, current_iter, writer)
        
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
