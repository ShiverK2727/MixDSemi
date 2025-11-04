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
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# 确保 dataloaders 和 utils 在 python 路径上
# (假设此脚本与您提供的 train_MiDSS_...py 位于同一目录)
import dataloaders.custom_transforms as tr
from dataloaders.dataloader_dc import (BUSISegmentation, FundusSegmentation, 
                                     MNMSSegmentation, ProstateSegmentation)
from biomedclip_vpt_RD import VPT_CLIP_RD # <-- 导入我们修改后的模型
from utils.text_sampler_v2 import TextSampler # <-- 导入您的 LLM 分组采样器
from torch.cuda.amp import GradScaler, autocast
from utils.training import cycle

# --- 1. Argument Parser ---
# (继承您 train_MiDSS...py 中的所有 argparse 配置)
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='prostate',
                    choices=['fundus', 'prostate', 'MNMS', 'BUSI'],
                    help='Dataset to use for training')
parser.add_argument("--save_name", type=str, default="vpt_anchor_pretrain",
                    help="Experiment name for saving checkpoints and logs")
parser.add_argument("--overwrite", action='store_true',
                    help="Overwrite existing experiment directory")
parser.add_argument('--save_model', action='store_true', default=True,
                    help='Save best model checkpoints during training')
parser.add_argument("--gpu", type=str, default='0',
                    help='GPU device ID to use')
parser.add_argument("--seed", type=int, default=1337,
                    help="Random seed for reproducibility")
parser.add_argument("--deterministic", type=int, default=1,
                    help="Whether to use deterministic training")
parser.add_argument("--max_iterations", type=int, default=10000,
                    help="Maximum iterations to train (auto-set per dataset if None)")
parser.add_argument('--amp', type=int, default=1,
                    help='Use mixed precision training (1: enabled, 0: disabled)')
parser.add_argument("--label_bs", type=int, default=4,
                    help="Labeled batch size per GPU")
parser.add_argument('--domain_num', type=int, default=6,
                    help='Total number of domains in dataset')
parser.add_argument('--lb_domain', type=int, default=1,
                    help='Domain ID to use for labeled data')
parser.add_argument('--lb_num', type=int, default=40,
                    help='Number of labeled samples')
parser.add_argument('--lb_ratio', type=float, default=0,
                    help='Labeled data ratio (overrides lb_num if > 0)')

# --- Preprocessing & Text ---
parser.add_argument('--llm_model', type=str, default='gemini',
                    choices=['gemini', 'GPT5', 'DeepSeek'],
                    help='LLM model used to generate score tensors')
parser.add_argument('--describe_nums', type=int, default=40,
                    choices=[20, 40, 60, 80],
                    help='Number of textual descriptions for preprocessing')
parser.add_argument('--text_root', type=str, default='/app/MixDSemi/SynFoCLIP/code/text',
                    help='Directory containing dataset text description JSON files')
parser.add_argument('--text_num_subsets', type=int, default=2,
                    help='Number of text subsets to sample for invariance loss')

# --- VPT-CLIP-RD 模型参数 ---
parser.add_argument('--biomedclip_path', type=str, default='/root/models/BiomedCLIP',
                    help='Root directory containing BiomedCLIP weights and config JSON')
parser.add_argument('--biomedclip_num_prompts', type=int, default=10,
                    help='Number of VPT prompts per layer')
parser.add_argument('--biomedclip_embed_dim', type=int, default=768,
                    help='Embedding dimension for ViT (ViT-B is 768)')
parser.add_argument('--biomedclip_init_std', type=float, default=0.02,
                    help='Initialization std for prompt parameters')
parser.add_argument('--biomedclip_prompt_scale_init', type=float, default=1.0,
                    help='Initial prompt scaling factor')
parser.add_argument('--biomedclip_lr', type=float, default=1e-4,
                    help='Learning rate for BiomedCLIP optimizer')
parser.add_argument('--biomedclip_weight_decay', type=float, default=1e-2,
                    help='Weight decay for BiomedCLIP optimizer')
parser.add_argument('--biomedclip_disable_scale', action='store_true',
                    help='Disable learnable prompt scaling')

# --- 新增：预训练损失权重 ---
parser.add_argument('--w_semantic', type=float, default=1.0, 
                    help='Weight for L_semantic (BCE loss)')
parser.add_argument('--w_invariance', type=float, default=0.5, 
                    help='Weight for L_invariance (MSE loss)')
parser.add_argument('--w_style_neg', type=float, default=0.1, 
                    help='Weight for L_style_neg (push away style)')

args = parser.parse_args()
args.biomedclip_enable_scale = not args.biomedclip_disable_scale

# (与您原文件相同的 TEXT_DATASET_DEFAULTS)
TEXT_DATASET_DEFAULTS = {
    'fundus': 'Fundus',
    'prostate': 'ProstateSlice',
    'MNMS': 'MNMS',
    'BUSI': 'BUSI',
}

def get_dataset_config(args):
    """根据 args.dataset 返回特定于数据集的配置"""
    if args.dataset == 'fundus':
        return {
            'train_data_path': '/app/MixDSemi/data/Fundus', 'dataset_class': FundusSegmentation,
            'num_channels': 3, 'patch_size': 256, 'num_classes': 2,
            'min_v': 0.5, 'max_v': 1.5, 'fillcolor': 255, 'max_iter': 30000,
            'domain_len': [50, 99, 320, 320], 'default_domain_num': 4,
        }
    elif args.dataset == 'prostate':
        return {
            'train_data_path': '/app/MixDSemi/data/ProstateSlice', 'dataset_class': ProstateSegmentation,
            'num_channels': 1, 'patch_size': 384, 'num_classes': 1,
            'min_v': 0.1, 'max_v': 2, 'fillcolor': 255, 'max_iter': 60000,
            'domain_len': [225, 305, 136, 373, 338, 133], 'default_domain_num': 6,
        }
    elif args.dataset == 'MNMS':
        return {
            'train_data_path': '/app/MixDSemi/data/mnms', 'dataset_class': MNMSSegmentation,
            'num_channels': 1, 'patch_size': 288, 'num_classes': 3,
            'min_v': 0.1, 'max_v': 2, 'fillcolor': 0, 'max_iter': 60000,
            'domain_len': [1030, 1342, 525, 550], 'default_domain_num': 4,
        }
    elif args.dataset == 'BUSI':
        return {
            'train_data_path': '/app/MixDSemi/data/Dataset_BUSI_with_GT', 'dataset_class': BUSISegmentation,
            'num_channels': 1, 'patch_size': 256, 'num_classes': 1,
            'min_v': 0.1, 'max_v': 2, 'fillcolor': 0, 'max_iter': 30000,
            'domain_len': [350, 168], 'default_domain_num': 2,
        }
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

def train_vpt_anchor(args, snapshot_path, cfg):
    writer = SummaryWriter(snapshot_path + '/log')
    max_iterations = args.max_iterations if args.max_iterations is not None else cfg['max_iter']
    
    # --- 1. 数据增强 ---
    # !! 关键: 确保空间变换对 image 和 label 一起作用
    # `tr.RandomScaleCrop` 和 `tr.RandomHorizontalFlip` 都在 sample dict 上操作
    # 您的 dataloader_dc.py 通过 weak_transform 实现了这一点
    weak_spatial = transforms.Compose([
        tr.RandomScaleCrop(cfg['patch_size']),
        tr.RandomHorizontalFlip(),
    ])
    # 强色彩/强度增强（只对image作用）
    strong_color = transforms.Compose([
        tr.Brightness(cfg['min_v'], cfg['max_v']),
        tr.Contrast(cfg['min_v'], cfg['max_v']),
        tr.GaussianBlur(kernel_size=int(0.1 * cfg['patch_size']), num_channels=cfg['num_channels']),
    ])
    # 归一化和张量转换
    normal_toTensor = transforms.Compose([
        tr.Normalize_tf(dataRange=[0,1]),
        tr.ToTensor(unet_size=cfg['patch_size']) # 确保输出为 patch_size
    ])

    # --- 2. 数据加载 ---
    # **关键：** 我们只配置 lb_dataset
    # dataloader_dc.py 会返回 'image' (弱增强), 'strong_aug' (强增强), 'label'
    
    # 确定 lb_num
    if args.lb_ratio > 0:
        lb_num = int(sum(cfg['domain_len']) * args.lb_ratio)
    else:
        lb_num = args.lb_num
        
    lb_kwargs = dict(
        base_dir=cfg['train_data_path'],
        phase='train',
        splitid=args.lb_domain,
        domain=[args.lb_domain],
        selected_idxs=list(range(lb_num)), # 只使用有标记数据
        weak_transform=weak_spatial,     # <-- 空间变换 (会应用到img和label)
        strong_tranform=strong_color,  # <-- 色彩变换 (只应用到img)
        normal_toTensor=normal_toTensor,
        img_size=cfg['patch_size'],
        is_RGB=(cfg['num_channels'] == 3),
        preprocess_dir='/app/MixDSemi/SynFoCLIP/preprocess', # (假设的路径)
        llm_model=args.llm_model,
        describe_nums=args.describe_nums,
        return_score=False, # 预训练不需要这个
        allow_missing_scores=True,
    )
    
    lb_dataset = cfg['dataset_class'](**lb_kwargs)
    
    lb_loader = DataLoader(lb_dataset, batch_size=args.label_bs, shuffle=True, 
                           num_workers=2, pin_memory=True, drop_last=True)
    lb_dataloader = cycle(lb_loader)
    
    logging.info(f"Pre-training VPT Anchor on Labeled Data only.")
    logging.info(f"Dataset: {args.dataset}, Labeled samples: {lb_num}, Domain: {args.lb_domain}")

    # --- 3. 初始化模型和优化器 ---
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    model = VPT_CLIP_RD(
        model_path=args.biomedclip_path,
        num_prompts=args.biomedclip_num_prompts,
        embed_dim=args.biomedclip_embed_dim,
        init_std=args.biomedclip_init_std,
        prompt_scale_init=args.biomedclip_prompt_scale_init,
        enable_scale=args.biomedclip_enable_scale,
        device=str(device),
    )
    
    # **关键：** 优化器只训练VPT和RD投影层
    trainable_params = list(model.prompt_learner.parameters()) + list(model.rd_query_proj.parameters())
    optimizer = optim.AdamW(
        trainable_params,
        lr=args.biomedclip_lr,
        weight_decay=args.biomedclip_weight_decay,
    )
    logging.info(
        "VPT-CLIP-RD Anchor model initialized on %s (trainable params=%d)",
        device, sum(p.numel() for p in trainable_params) if trainable_params else 0
    )

    # --- 4. 加载和编码文本锚点 ---
    text_dataset_key = TEXT_DATASET_DEFAULTS.get(args.dataset)
    text_sampler = TextSampler(args.text_root)
    targets_texts, style_texts, flat_list = text_sampler.load_texts(
        dataset=text_dataset_key,
        llm=args.llm_model,
        describe_nums=args.describe_nums,
    )
    
    # 编码 "风格" 文本，作为负类
    E_style_neg = model.encode_text(model.tokenizer(style_texts).to(device))
    E_style_neg = E_style_neg.mean(dim=0, keepdim=True) # [1, D]
    if E_style_neg.shape[1] == 0:
        logging.warning("No style texts found. L_style_neg will be disabled.")
        E_style_neg = None
        args.w_style_neg = 0

    # 编码 "背景" 文本
    bg_texts = ["background", "empty space", "other tissue"]
    E_bg_neg = model.encode_text(model.tokenizer(bg_texts).to(device))
    E_bg_neg = E_bg_neg.mean(dim=0, keepdim=True) # [1, D]
    
    # 预编码所有重叠子集的平均锚点 (您的LLM分组方案)
    _, text_subsets = text_sampler.sample_subsets(
        targets_texts,
        num_samples=args.text_num_subsets
    )
    E_subsets = []
    for subset in text_subsets:
        E_subsets.append(model.encode_text(model.tokenizer(subset).to(device)).mean(dim=0, keepdim=True))
    
    if len(E_subsets) < 2:
        logging.warning(f"Text subsets < 2 ({len(E_subsets)}). L_invariance will be disabled.")
        E_subsets = E_subsets * 2 # 复制以避免
        args.w_invariance = 0
    
    logging.info(f"Loaded {len(E_subsets)} text subset anchors, 1 style anchor, 1 background anchor.")

    # --- 5. 损失函数 ---
    bce_loss_fn = nn.BCEWithLogitsLoss()
    mse_loss_fn = nn.MSELoss()
    
    # 确定 patch token 数量 (例如 14x14)
    num_patches_side = cfg['patch_size'] // model.clip_model.visual.patch_size[0]
    num_patches = num_patches_side ** 2

    # --- 6. AMP Scaler ---
    amp_enabled = bool(args.amp)
    scaler = GradScaler(enabled=amp_enabled)
    amp_cm = autocast if amp_enabled else contextlib.nullcontext
    
    model.train() # 设置VPT为训练模式
    
    # --- 7. 主训练循环 ---
    p_bar = tqdm(range(1, max_iterations + 1), desc=f'Pre-training VPT Anchor')
    
    for iter_num in p_bar:
        # **只使用有标记数据**
        # 'image' = 弱增强(空间), 'strong_aug' = 强增强(空间+色彩)
        sample = next(lb_dataloader)
        x_weak = sample['image'].to(device) 
        x_strong = sample['strong_aug'].to(device)
        y_mask_high = sample['label'].to(device) # 可能为 (B, H, W) 或 (B, 1, H, W)
        # print(y_mask_high.shape)
        # 确保 y_mask_high 形状为 (B, 1, H, W)
        if y_mask_high.dim() == 3:
            # (B, H, W) -> (B, 1, H, W)
            y_mask_high = y_mask_high.unsqueeze(1)
        # 将掩码归一化到 [0, 1]（如果原始是 0-255）
        try:
            max_val = float(y_mask_high.max().item())
        except Exception:
            max_val = 0.0
        if max_val > 1.5:
            y_mask_high = y_mask_high.float() / 255.0
        else:
            y_mask_high = y_mask_high.float()
        
        # 注意: 我们在 forward 之后再生成与 patch 数量匹配的低分辨率真值，
        # 因为模型可能对输入做了 resize（例如 224 -> 14x14 patches），
        # 直接使用 cfg['patch_size'] 计算可能不匹配模型实际输出。

        # 从子集中随机选2个不同的锚点 (模拟文本域偏移)
        E_A, E_B = random.sample(E_subsets, 2)
        
        # 构建锚点矩阵 (K=4)
        # 0 = 锚点A, 1 = 锚点B, 2 = 背景锚点, 3 = 风格锚点
        all_anchors = torch.cat([E_A, E_B, E_bg_neg, E_style_neg], dim=0) # [4, D]

        optimizer.zero_grad()

        with amp_cm():
            # --- 前向传播 (模拟图像域偏移) ---
            # H_maps_w: [B, K, N], H_raw_w: [B, N, D]
            H_maps_w, H_raw_w = model(x_weak, all_anchors)
            # H_maps_s: [B, K, N], H_raw_s: [B, N, D]
            H_maps_s, H_raw_s = model(x_strong, all_anchors)

            # --- 在 forward 之后创建与模型 patch 数量一致的低分辨率真值 ---
            B, N, D = H_raw_w.shape
            side = int(N ** 0.5)
            # 处理非整除情况（退化到使用模型 patch 尺寸作为 pooling kernel）
            img_h = y_mask_high.shape[2]
            img_w = y_mask_high.shape[3]
            kernel_h = img_h // side
            kernel_w = img_w // side
            if kernel_h <= 0 or kernel_w <= 0:
                raise RuntimeError(f"Invalid pooling kernel computed: ({kernel_h},{kernel_w}), image size: ({img_h},{img_w}), patches side: {side}")
            y_low_res = F.avg_pool2d(y_mask_high.float(), kernel_size=(kernel_h, kernel_w))
            y_low_res_flat = y_low_res.view(-1, N) # [B, N]
            
            # --- 构造损失 ---
            
            # 1. L_semantic (语义损失)
            # 目标: H_A (来自x_weak) 和 H_B (来自x_strong) 都应匹配 y_low_res
            # H_maps_w[:, 0, :] -> 锚点A的图
            # H_maps_s[:, 1, :] -> 锚点B的图
            loss_sem_A = bce_loss_fn(H_maps_w[:, 0, :], y_low_res_flat)
            loss_sem_B = bce_loss_fn(H_maps_s[:, 1, :], y_low_res_flat)
            loss_semantic = (loss_sem_A + loss_sem_B) * 0.5

            # 2. L_invariance (域不变性损失)
            # 强迫模型在不同图像域/文本域下提取的 *原始* patch 特征一致
            loss_invariance = mse_loss_fn(H_raw_s, H_raw_w.detach())
            
            # 3. L_style_neg (负类损失)
            # 目标: 背景锚点和风格锚点应匹配 "非y_low_res"
            y_low_res_neg = 1.0 - y_low_res_flat
            # H_maps_w[:, 2, :] -> 背景锚点 (来自 x_weak)
            # H_maps_w[:, 3, :] -> 风格锚点 (来自 x_weak)
            loss_neg_bg = bce_loss_fn(H_maps_w[:, 2, :], y_low_res_neg)
            loss_neg_style = bce_loss_fn(H_maps_w[:, 3, :], y_low_res_neg)
            loss_style_neg = (loss_neg_bg + loss_neg_style) * 0.5
            
            # 4. Total Loss
            loss_total = (
                args.w_semantic * loss_semantic +
                args.w_invariance * loss_invariance +
                args.w_style_neg * loss_style_neg
            )

        scaler.scale(loss_total).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # --- 日志记录 ---
        writer.add_scalar('pretrain/loss_total', loss_total.item(), iter_num)
        writer.add_scalar('pretrain/loss_semantic', loss_semantic.item(), iter_num)
        writer.add_scalar('pretrain/loss_invariance', loss_invariance.item(), iter_num)
        writer.add_scalar('pretrain/loss_style_neg', loss_style_neg.item(), iter_num)
        writer.add_scalar('pretrain/lr', optimizer.param_groups[0]['lr'], iter_num)

        p_bar.set_description(
            'iter %d: loss=%.4f (sem:%.4f, inv:%.4f, neg:%.4f)'
            % (iter_num, loss_total.item(), 
               loss_semantic.item(),
               loss_invariance.item(),
               loss_style_neg.item())
        )

        if iter_num % 100 == 0:
            logging.info(
                "iter %d: loss=%.6f, sem=%.6f, inv=%.6f, neg=%.6f",
                iter_num,
                loss_total.item(),
                loss_semantic.item(),
                loss_invariance.item(),
                loss_style_neg.item()
            )
            
    if p_bar is not None:
        p_bar.close()

    # --- 保存最终的VPT锚点 ---
    if args.save_model:
        save_vpt_path = os.path.join(snapshot_path, "vpt_anchor_final.pth")
        model.save_prompts(save_vpt_path)
        logging.info(f"Saved final VPT Anchor (VPT + RD weights) to {save_vpt_path}")

    writer.close()
    logging.info("VPT Anchor pre-training finished.")
    return snapshot_path


if __name__ == "__main__":
    # --- 1. 参数和路径设置 (与您原文件一致) ---
    if len(args.save_name) == 0:
        args.save_name = f'vpt_pretrain_lb{args.lb_num}_dm{args.lb_domain}' # 新的默认名称
    
    dataset_cfg = get_dataset_config(args)
    
    snapshot_path = f"../model/{args.dataset}/pretrain_anchor/{args.save_name}/"
    
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    elif args.overwrite:
        shutil.rmtree(snapshot_path)
        os.makedirs(snapshot_path)
    else:
        raise Exception('file {} is exist!'.format(snapshot_path))

    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__', 'model', 'data']))

    # --- 2. 保存配置 ---
    import json
    config_dict = vars(args).copy()
    config_dict['snapshot_path'] = snapshot_path
    config_dict.update(dataset_cfg) # 合并数据集特定配置
    # (移除不需要的 SSL 和 U-Net 配置)
    # 将可能不可 JSON 序列化的对象转换为可序列化的表示
    # 例如 dataset_class 是一个类对象，json.dump 会失败；改为使用类名字符串
    for k, v in list(config_dict.items()):
        # 转换 class/type 为名称
        if isinstance(v, type):
            try:
                config_dict[k] = v.__name__
            except Exception:
                config_dict[k] = str(v)

    config_file = os.path.join(snapshot_path, 'pretrain_config.json')
    with open(config_file, 'w') as f:
        json.dump(config_dict, f, indent=4, sort_keys=True)
    print(f"Pre-training configuration saved to: {config_file}")

    # --- 3. 日志 ---
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    cmd = " ".join(["python"] + sys.argv)
    logging.info(cmd)
    logging.info(str(args))
    
    # --- 4. 启动训练 ---
    train_vpt_anchor(args, snapshot_path, dataset_cfg)
