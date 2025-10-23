import argparse
import logging
import os
import random
import shutil
import sys
import time
from typing import Iterable

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from networks.unet_model import UNet
from dataloaders.dataloader import FundusSegmentation, ProstateSegmentation, MNMSSegmentation, BUSISegmentation
import dataloaders.custom_transforms as tr
from utils import losses, metrics, ramps
from torch.cuda.amp import autocast, GradScaler
import contextlib

from medpy.metric import binary
from scipy.ndimage import zoom
import cv2
from itertools import chain
from skimage.measure import label

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

parser.add_argument('--save_img',action='store_true')
parser.add_argument('--save_model',action='store_true')
parser.add_argument('--warmup', action='store_true', help='If activated, warp up the learning from a lower lr to the base_lr')
parser.add_argument('--warmup_period', type=int, default=250,
                    help='Warp up iterations, only valid whrn warmup is activated')
parser.add_argument('--eval', action='store_true', help='Only run evaluation')
args = parser.parse_args()


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    # teacher network: ema_model
    # student network: model
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

def cycle(iterable: Iterable):
    """Make an iterator returning elements from the iterable.

    .. note::
        **DO NOT** use `itertools.cycle` on `DataLoader(shuffle=True)`.\n
        Because `itertools.cycle` saves a copy of each element, batches are shuffled only at the first epoch. \n
        See https://docs.python.org/3/library/itertools.html#itertools.cycle for more details.
    """
    while True:
        for x in iterable:
            yield x

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
    for i in range(domain_num):
        cur_dataloader = test_dataloader[i]
        domain_val_dice = [0.0] * n_part
        domain_val_dc, domain_val_jc, domain_val_hd, domain_val_asd = [0.0] * n_part, [0.0] * n_part, [0.0] * n_part, [0.0] * n_part
        domain_code = i+1
        for batch_num,sample in enumerate(cur_dataloader):
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
    
def to_2d(input_tensor):
    input_tensor = input_tensor.unsqueeze(1)
    tensor_list = []
    temp_prob = input_tensor == torch.ones_like(input_tensor)
    tensor_list.append(temp_prob)
    temp_prob2 = input_tensor > torch.zeros_like(input_tensor)
    tensor_list.append(temp_prob2)
    output_tensor = torch.cat(tensor_list, dim=1)
    return output_tensor.float()

def to_3d(input_tensor):
    input_tensor = input_tensor.unsqueeze(1)
    tensor_list = []
    for i in range(1, 4):
        temp_prob = input_tensor == i * torch.ones_like(input_tensor)
        tensor_list.append(temp_prob)
    output_tensor = torch.cat(tensor_list, dim=1)
    return output_tensor.float()

def obtain_cutmix_box(img_size, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3):
    mask = torch.zeros(img_size, img_size).cuda()
    if random.random() > p:
        return mask

    size = np.random.uniform(size_min, size_max) * img_size * img_size
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.sqrt(size / ratio))
        cutmix_h = int(np.sqrt(size * ratio))
        x = np.random.randint(0, img_size)
        y = np.random.randint(0, img_size)

        if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
            break

    mask[y:y + cutmix_h, x:x + cutmix_w] = 1

    return mask

class statistics(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.record = []
        self.num = 0
        self.avg = 0

    def update(self, val):
        self.record.append(val)
        self.num += 1
        self.avg = sum(self.record) / self.num

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
    test_dataset = []
    test_dataloader = []
    lb_dataset = dataset(base_dir=train_data_path, phase='train', splitid=lb_domain, domain=[lb_domain], 
                                                selected_idxs = lb_idxs, weak_transform=weak,normal_toTensor=normal_toTensor, img_size=patch_size)
    ulb_dataset = dataset(base_dir=train_data_path, phase='train', splitid=lb_domain, domain=domain, 
                                                selected_idxs=unlabeled_idxs, weak_transform=weak, strong_tranform=strong,normal_toTensor=normal_toTensor, img_size=patch_size)
    for i in range(1, domain_num+1):
        cur_dataset = dataset(base_dir=train_data_path, phase='test', splitid=-1, domain=[i], normal_toTensor=normal_toTensor, img_size=patch_size)
        test_dataset.append(cur_dataset)
    if not args.eval:
        lb_dataloader = cycle(DataLoader(lb_dataset, batch_size = args.label_bs, shuffle=True, num_workers=2, pin_memory=True, drop_last=True))
        ulb_dataloader = cycle(DataLoader(ulb_dataset, batch_size = args.unlabel_bs, shuffle=True, num_workers=2, pin_memory=True, drop_last=True))
    for i in range(0,domain_num):
        cur_dataloader = DataLoader(test_dataset[i], batch_size = args.test_bs, shuffle=False, num_workers=0, pin_memory=True)
        test_dataloader.append(cur_dataloader)

    iter_num = 0

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

    # Calculate number of evaluation cycles
    num_eval_cycles = max_iterations // args.num_eval_iter
    
    # Initialize best metrics
    best_dice = [0.0] * n_part
    best_dice_iter = [-1] * n_part
    best_avg_dice = 0.0
    best_avg_dice_iter = -1
    dice_of_best_avg = [0.0] * n_part

    threshold = args.threshold
    scaler = GradScaler()
    amp_cm = autocast if args.amp else contextlib.nullcontext

    # Main training loop - iterate by evaluation cycles
    for eval_cycle in range(num_eval_cycles):
        unet_model.train()
        ema_unet_model.train()
        p_bar = tqdm(range(args.num_eval_iter), desc=f'Cycle {eval_cycle+1}/{num_eval_cycles}')
        unet_ulb_dice_sta = [statistics() for _ in range(n_part)]
        
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

            with amp_cm():
                # ========== Teacher Model: 3 forward passes (matching MiDSS) ==========
                with torch.no_grad():
                    # 1. Original unlabeled image pseudo-labels
                    unet_logits_ulb_x_w = ema_unet_model(ulb_unet_size_x_w)
                    unet_prob_ulb_x_w = torch.softmax(unet_logits_ulb_x_w, dim=1)
                    unet_prob, unet_pseudo_label = torch.max(unet_prob_ulb_x_w, dim=1)
                    mask = (unet_prob > threshold).unsqueeze(1).float()
                    
                    # Generate CutMix box (moved here to be used by teacher)
                    unet_size_label_box = torch.stack([obtain_cutmix_box(img_size=patch_size, p=1.0) for i in range(len(ulb_unet_size_x_w))], dim=0)
                    unet_size_img_box = unet_size_label_box.unsqueeze(1)
                    
                    # 2. UL mixing (Unlabeled background + Labeled foreground)
                    ulb_x_w_ul = ulb_unet_size_x_w * (1-unet_size_img_box) + lb_unet_size_x_w * unet_size_img_box
                    logits_w_ul = ema_unet_model(ulb_x_w_ul)
                    prob_w_ul = torch.softmax(logits_w_ul, dim=1)
                    conf_w_ul, pseudo_label_w_ul = torch.max(prob_w_ul, dim=1)
                    mask_w_ul = (conf_w_ul > threshold).unsqueeze(1).float()
                    
                    # 3. LU mixing (Labeled background + Unlabeled foreground)
                    ulb_x_w_lu = lb_unet_size_x_w * (1-unet_size_img_box) + ulb_unet_size_x_w * unet_size_img_box
                    logits_w_lu = ema_unet_model(ulb_x_w_lu)
                    prob_w_lu = torch.softmax(logits_w_lu, dim=1)
                    conf_w_lu, pseudo_label_w_lu = torch.max(prob_w_lu, dim=1)
                    mask_w_lu = (conf_w_lu > threshold).unsqueeze(1).float()
                    
                    # Merge UL and LU pseudo-labels
                    mask_w = mask_w_ul * (1-unet_size_img_box) + mask_w_lu * unet_size_img_box
                    pseudo_label_w = (pseudo_label_w_ul * (1-unet_size_label_box) + 
                                      pseudo_label_w_lu * unet_size_label_box).long()
                    
                    # Ensemble filtering: filter out inconsistent predictions
                    ensemble = (pseudo_label_w == unet_pseudo_label).unsqueeze(1).float() * mask
                    mask_w[ensemble == 0] = 0
                
                # Prepare three types of masks (matching MiDSS)
                mask_ul = mask.clone()
                mask_ul[unet_size_img_box.expand(mask_ul.shape) == 1] = 1  # CutMix region fully trusted
                
                mask_lu = mask.clone()
                mask_lu[unet_size_img_box.expand(mask_lu.shape) == 0] = 1  # Non-CutMix region fully trusted
                
                # Prepare three types of mixing
                # 1. UL mixing: unlabeled strong aug + labeled weak aug
                ulb_unet_size_x_s_ul = ulb_unet_size_x_s * (1-unet_size_img_box) + lb_unet_size_x_w * unet_size_img_box
                unet_size_pseudo_label_ul = (unet_pseudo_label * (1-unet_size_label_box) + 
                                              lb_unet_size_mask * unet_size_label_box).long()
                
                # 2. LU mixing: labeled weak aug + unlabeled strong aug
                ulb_unet_size_x_s_lu = lb_unet_size_x_w * (1-unet_size_img_box) + ulb_unet_size_x_s * unet_size_img_box
                unet_size_pseudo_label_lu = (lb_unet_size_mask * (1-unet_size_label_box) + 
                                              unet_pseudo_label * unet_size_label_box).long()
                
                # ========== Student Model: 4 forward passes (matching MiDSS) ==========
                unet_logits_lb_x_w = unet_model(lb_unet_size_x_w)
                unet_logits_ulb_x_s_ul = unet_model(ulb_unet_size_x_s_ul)
                unet_logits_ulb_x_s_lu = unet_model(ulb_unet_size_x_s_lu)
                unet_logits_ulb_x_s = unet_model(ulb_unet_size_x_s)  # Pure strong augmentation
                
                # Calculate dice on unlabeled data for monitoring
                if args.dataset == 'fundus':
                    unet_pseudo_label_2layer = to_2d(unet_pseudo_label)
                    ulb_unet_size_mask_2layer = to_2d(ulb_unet_size_mask)
                    unet_ulb_dice = dice_calcu[args.dataset](np.asarray(unet_pseudo_label_2layer.cpu()), ulb_unet_size_mask_2layer.cpu())
                else:
                    unet_ulb_dice = dice_calcu[args.dataset](np.asarray(unet_pseudo_label.cpu()), ulb_unet_size_mask.cpu())
                for n, p in enumerate(part):
                    unet_ulb_dice_sta[n].update(unet_ulb_dice[n])

                # ========== Loss Computation (matching MiDSS) ==========
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

                # Consistency loss (pure strong augmentation)
                unet_unsup_loss_s = (ce_loss(unet_logits_ulb_x_s, pseudo_label_w) * 
                                     mask_w.squeeze(1)).mean() + \
                                    dice_loss(unet_logits_ulb_x_s, pseudo_label_w.unsqueeze(1), 
                                             mask=mask_w, softmax=softmax, sigmoid=sigmoid, multi=multi)
                
                # Total loss (matching MiDSS formula)
                loss = unet_sup_loss + consistency_weight * (unet_unsup_loss_ul + unet_unsup_loss_lu + 
                                                             consistency_weight * unet_unsup_loss_s)

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
            writer.add_scalar('train/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('train/learning_rate', unet_optimizer.param_groups[0]['lr'], iter_num)
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
        min_v, max_v = 0.1, 2
        fillcolor = 0
        if args.max_iterations is None:
            args.max_iterations = 60000
        if args.domain_num >= 4:
            args.domain_num = 4
    elif args.dataset == 'BUSI':
        train_data_path="../../data/Dataset_BUSI_with_GT"
        num_channels = 1
        patch_size = 256
        num_classes = 1
        part = ['base'] 
        dataset = BUSISegmentation
        min_v, max_v = 0.1, 2
        fillcolor = 0
        if args.max_iterations is None:
            args.max_iterations = 30000
        if args.domain_num >= 2:
            args.domain_num = 2
    
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

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    cmd = " ".join(["python"] + sys.argv)
    logging.info(cmd)
    logging.info(str(args))

    train(args, snapshot_path)
