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
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm
import inspect

import dataloaders.custom_transforms as tr
from dataloaders.dataloader_dc import BUSISegmentation, FundusSegmentation, MNMSSegmentation, ProstateSegmentation
# 导入 VPT+TPT 分割模型
# 确保这个文件在您的 python 路径上
from biomedclip_vpt_tpt_seg import build_vpt_tpt_seg_model
# 确保这个文件在您的 python 路径上
from utils.text_sampler_an import TextSampler
from torch.cuda.amp import GradScaler, autocast
# 确保这个文件在您的 python 路径上
from utils.training import cycle

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
parser.add_argument('--amp', type=int, default=1,
                    help='Use mixed precision training (1: enabled, 0: disabled)')

# ==================== Data Configuration ====================
# (数据配置参数保持不变)
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
# (预处理参数保持不变 - 'preprocess_dir' 现在不由 L_llm 使用)
parser.add_argument('--preprocess_dir', type=str, default=None,
                    help='Override preprocessing directory (if dataloader uses it)')
parser.add_argument('--llm_model', type=str, default='gemini',
                    choices=['gemini', 'GPT5', 'DeepSeek'],
                    help='LLM model used to generate score tensors')
parser.add_argument('--describe_nums', type=int, default=40,
                    choices=[20, 40, 60, 80],
                    help='Number of textual descriptions for preprocessing')

# ==================== VPT + TPT Configuration ====================
# (VPT/TPT 配置保持不变)
parser.add_argument('--biomedclip_path', type=str, default='/root/models/BiomedCLIP',
                    help='Root directory containing BiomedCLIP weights and config JSON')
parser.add_argument('--visual_num_prompts', type=int, default=4,
                    help='Number of visual prompts per layer (VPT-Deep)')
parser.add_argument('--text_num_prompts', type=int, default=4,
                    help='Number of text context prompts (TPT CoOp-style)')
parser.add_argument('--vpt_tpt_lr', type=float, default=1e-4,
                    help='Learning rate for VPT+TPT optimizer')
parser.add_argument('--vpt_tpt_weight_decay', type=float, default=1e-2,
                    help='Weight decay for VPT+TPT optimizer')
parser.add_argument('--freeze_backbone', action='store_true', default=True,
                    help='Freeze BiomedCLIP backbone (only train prompts)')

# ==================== Text Sampler ====================
parser.add_argument('--text_root', type=str, default='/app/MixDSemi/SynFoCLIP/code/text',
                    help='Directory containing dataset text description JSON files')

# ==================== (新) Loss 权重 (EMA + SCCM + Dice + VPT-Reg) ====================
parser.add_argument('--lambda_ce', type=float, default=1.0,
                    help='Weight for supervised Cross-Entropy loss (L_ce)')
parser.add_argument('--lambda_dice', type=float, default=1.0,
                    help='Weight for supervised Dice loss (L_dice)')
parser.add_argument('--lambda_consis', type=float, default=1.0,
                    help='Weight for *all* consistency losses (Map + Feat)')
parser.add_argument('--lambda_sccm', type=float, default=0.1,
                    help='Weight for SCCM text alignment loss (L_sccm)')
parser.add_argument('--lambda_reg_vpt', type=float, default=0.1,
                    help='(新) Weight for VPT feature regularization (L_reg_vpt)')
parser.add_argument('--feat_consis_scale', type=float, default=0.1,
                    help='Scaling factor for patch feature consistency loss (relative to map consistency)')
parser.add_argument('--T_consis', type=float, default=0.5,
                    help='Temperature for sharpening pseudo-labels in consistency loss')
parser.add_argument('--ema_decay', type=float, default=0.999, 
                    help='EMA decay rate for the teacher model')
parser.add_argument('--tau', type=float, default=0.9, 
                    help='Confidence threshold for FixMatch pseudo-labeling')
parser.add_argument('--save_every', type=int, default=1000,
                    help='Save student/teacher prompts every N iterations (if --save_model)')


args = parser.parse_args()

TEXT_DATASET_DEFAULTS = {
    'fundus': 'Fundus',
    'prostate': 'ProstateSlice',
    'MNMS': 'MNMS',
    'BUSI': 'BUSI',
}


# ==================== (新) Dice Loss 辅助类 ====================
class DiceLoss(nn.Module):
    def __init__(self, n_classes, smooth=1e-5):
        """
        为多类别分割计算 Dice Loss。
        假定输入 pred_probs 已经是 softmax 概率。
        假定 target_one_hot 是 one-hot 编码的硬标签。
        """
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.smooth = smooth

    def forward(self, pred_probs, target_one_hot):
        # pred_probs is [B, K, H, W] (after softmax)
        # target_one_hot is [B, K, H, W]
        
        # 排除背景类别 (class 0)
        # 这对于处理类别不平衡（背景通常占主导）非常重要
        pred_probs = pred_probs[:, 1:, ...]
        target_one_hot = target_one_hot[:, 1:, ...]

        if pred_probs.size() != target_one_hot.size():
            # 确保在排除背景后形状一致
            logging.warning(f"DiceLoss: Pred shape {pred_probs.size()} != Target shape {target_one_hot.size()}. K={self.n_classes}")
            # 可能是因为 n_classes=1 (只有前景和背景)
            if self.n_classes == 2 and pred_probs.shape[1] == 0:
                 # 如果只有背景和1个前景类，排除背景后 K-1=0，这会导致错误
                 # 回退到不排除背景
                 pred_probs = pred_probs[:, 0:, ...]
                 target_one_hot = target_one_hot[:, 0:, ...]
            else:
                # 处理 K>2 的情况
                min_k = min(pred_probs.shape[1], target_one_hot.shape[1])
                pred_probs = pred_probs[:, :min_k, ...]
                target_one_hot = target_one_hot[:, :min_k, ...]

        # 扁平化 H 和 W 维度
        dims = (2, 3) # (H, W)
        
        intersection = torch.sum(pred_probs * target_one_hot, dims)
        cardinality = torch.sum(pred_probs + target_one_hot, dims)
        
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        
        # 我们希望最小化 Loss，即 1 - DiceScore
        # 在批次和前景类别上取平均
        dice_loss = 1. - dice_score.mean()
        return dice_loss


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
    
    # (新) 修改 kwargs，不再强制要求 return_score=True
    lb_kwargs = dict(
        base_dir=train_data_path,
        phase='train',
        splitid=lb_domain,
        domain=[lb_domain],
        selected_idxs=lb_idxs,
        weak_transform=weak,
        strong_tranform=strong,
        normal_toTensor=normal_toTensor,
        img_size=patch_size,
        # (如果 dataloader_dc.py 需要这些参数，请保留)
        preprocess_dir=dataset_preprocess_dir,
        llm_model=args.llm_model,
        describe_nums=args.describe_nums,
        # return_score=True, # 移除
        # allow_missing_scores=False, # 移除
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
        # return_score=True, # 移除
        # allow_missing_scores=False, # 移除
    )
    # 尝试加载数据集，如果 Dataloader 不接受 'preprocess_dir' 等参数，会报错
    try:
        lb_dataset = dataset(**lb_kwargs)
        ulb_dataset = dataset(**ulb_kwargs)
    except TypeError as e:
        logging.warning(f"Dataloader 初始化失败: {e}")
        logging.warning("Dataloader 可能不支持 'preprocess_dir' 或 'llm_model' 等参数。")
        # 尝试通过过滤出 dataset.__init__ 接受的参数来重试初始化
        try:
            sig = inspect.signature(dataset.__init__)
            params = sig.parameters
            accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

            def _filter(kw):
                if accepts_var_kw:
                    # 如果 __init__ 接受 **kwargs，直接返回原始 dict（异常由上层处理）
                    return kw
                allowed = set(p for p in params.keys() if p != 'self')
                return {k: v for k, v in kw.items() if k in allowed}

            lb_kwargs_filtered = _filter(lb_kwargs)
            ulb_kwargs_filtered = _filter(ulb_kwargs)

            lb_dataset = dataset(**lb_kwargs_filtered)
            ulb_dataset = dataset(**ulb_kwargs_filtered)
            logging.info("Dataloader 初始化成功（使用过滤后的 kwargs）")
        except Exception as e2:
            logging.exception(f"尝试用过滤后的 kwargs 初始化 Dataloader 也失败: {e2}")
            # 重新抛出最初的错误以便上层可见
            raise e


    # --- Text sampling ---
    # (这部分现在用于 L_sccm 教师 和 L_sup 学生)
    text_dataset_key = TEXT_DATASET_DEFAULTS.get(args.dataset)
    text_sampler = TextSampler(args.text_root)
    per_class_lists, flat_list = text_sampler.load_texts(
        dataset=text_dataset_key,
        llm=args.llm_model,
        describe_nums=args.describe_nums,
    )
    logging.info(
        "TextSampler loaded dataset '%s' (num_classes=%d, total_texts=%d)",
        text_dataset_key,
        len(per_class_lists), # K-1 (前景类)
        len(flat_list),
    )

    # --- (新) 实例化 Student 和 EMA Teacher 模型 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logging.info("--- Initializing Student Model ---")
    model_student, preprocess, tokenizer = build_vpt_tpt_seg_model(
        model_path=args.biomedclip_path,
        device=str(device),
        visual_num_prompts=args.visual_num_prompts,
        text_num_prompts=args.text_num_prompts,
        freeze_backbone=args.freeze_backbone,
    )
    
    logging.info("--- Initializing EMA Teacher Model ---")
    model_teacher, _, _ = build_vpt_tpt_seg_model(
        model_path=args.biomedclip_path,
        device=str(device),
        visual_num_prompts=args.visual_num_prompts,
        text_num_prompts=args.text_num_prompts,
        freeze_backbone=args.freeze_backbone,
    )

    # 冻结 Teacher 模型，并从 Student 复制初始权重
    for param_teacher in model_teacher.parameters():
        param_teacher.requires_grad = False
    
    # 仅复制可训练的 prompt 权重
    model_teacher.visual_prompt_learner.load_from_state_dict(
        model_student.visual_prompt_learner.state_dict_for_save()
    )
    model_teacher.text_prompt_learner.load_from_state_dict(
        model_student.text_prompt_learner.state_dict_for_save()
    )
    
    # --- (新) 优化器只针对 Student 模型 ---
    trainable_params_student = []
    trainable_params_student.extend(model_student.visual_prompt_learner.parameters())
    trainable_params_student.extend(model_student.text_prompt_learner.parameters())
    trainable_params_student = [p for p in trainable_params_student if p.requires_grad]
    
    optimizer = optim.AdamW(
        trainable_params_student,
        lr=args.vpt_tpt_lr,
        weight_decay=args.vpt_tpt_weight_decay,
    )
    
    logging.info(
        "Student VPT+TPT model initialized on %s (trainable params=%d)",
        device,
        sum(p.numel() for p in trainable_params_student),
    )
    logging.info("Teacher VPT+TPT model initialized (frozen).")


    # ... (DataLoader 逻辑保持不变) ...
    lb_loader = DataLoader(lb_dataset, batch_size=args.label_bs, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    ulb_loader = DataLoader(ulb_dataset, batch_size=args.unlabel_bs, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    logging.info("Using random sampler for unlabeled data loader.")
    lb_dataloader = cycle(lb_loader)
    ulb_dataloader = cycle(ulb_loader)
    
    logging.info(f"Total iterations: {max_iterations}")
    
    # (新) 更新日志，加入 CE 和 Dice
    logging.info(
        "CLIP loss weights: L_ce=%.3f, L_dice=%.3f, L_consis=%.3f (feat_scale=%.3f, T_consis=%.2f, Tau=%.2f), L_sccm=%.3f, EMA_decay=%.4f",
        args.lambda_ce, # (新)
        args.lambda_dice, # (新)
        args.lambda_consis,
        args.feat_consis_scale,
        args.T_consis,
        args.tau, 
        args.lambda_sccm,
        args.ema_decay
    )
    
    
    # ... (AMP 逻辑保持不变) ...
    amp_enabled = bool(args.amp)
    scaler = GradScaler(enabled=amp_enabled)
    amp_cm = autocast if amp_enabled else contextlib.nullcontext
    logging.info(f"AMP (Mixed Precision) enabled: {amp_enabled}")

    # --- (新) 预编码 L_SCCM 的 LLM 教师原型 (P_g) ---
    # P_g 是使用 *原始* LLM 描述（来自 per_class_lists）
    # 和 *原始* 冻结文本编码器（无 TPT）计算的
    
    logging.info("Pre-encoding LLM Teacher Prototypes (P_g for L_sccm)...")
    K_classes = num_classes + 1
    if len(args.class_text) != K_classes:
        logging.error(f"args.class_text 长度 ({len(args.class_text)}) 与 num_classes+1 ({K_classes}) 不匹配。")
        raise ValueError("class_text 和 num_classes 配置错误")
    if len(per_class_lists) != num_classes:
         logging.error(f"per_class_lists 长度 ({len(per_class_lists)}) 与 num_classes ({num_classes}) 不匹配。")
         raise ValueError("text_sampler 加载的类别数与 num_classes 配置错误")

    llm_teacher_prototypes_list = []
    # 切换到 eval 模式以关闭 dropout (如果存在)
    original_clip_model = model_student.model.eval()
    
    with torch.no_grad():
        for i in range(K_classes):
            class_name = args.class_text[i]
            
            if i == 0:
                # 背景类：使用类名本身
                texts_for_class = [class_name]
            else:
                # 前景类：使用 per_class_lists 中对应的文本（索引 i-1）
                texts_for_class = per_class_lists[i - 1]
            
            # (新) 使用 *原始* 冻结编码器
            # 1. Tokenize
            tokens = tokenizer(texts_for_class).to(device)
            # 2. Encode
            # model_student.model 是冻结的原始 BiomedCLIP
            text_features = original_clip_model.encode_text(tokens) # [N_texts, D]
            
            # 3. Average
            text_features_mean = text_features.mean(dim=0, keepdim=True)  # [1, D]
            llm_teacher_prototypes_list.append(text_features_mean)
        
        # (新) P_g (LLM Teacher Prototypes)
        llm_teacher_prototypes_Pg = torch.cat(llm_teacher_prototypes_list, dim=0)  # [K, D]
        llm_teacher_prototypes_Pg = F.normalize(llm_teacher_prototypes_Pg, dim=-1)
    
    logging.info(f"Pre-encoded llm_teacher_prototypes_Pg: {llm_teacher_prototypes_Pg.shape}")
    
    # 确保 Student 模型回到训练模式（尽管只有 prompts 在训练）
    model_student.train()

    # --- (新) 实例化 Dice Loss ---
    dice_loss_fn = DiceLoss(n_classes=K_classes).to(device)

    # --- (新) EMA 更新辅助函数 ---
    @torch.no_grad()
    def update_ema_variables(student_model, teacher_model, alpha):
        """
        使用 EMA 更新 Teacher 模型的 *可训练* 参数 (prompts)。
        alpha 是 EMA 的 decay rate (动量)。
        """
        # 更新 Visual Prompts
        for (name_stud, param_stud), (name_teach, param_teach) in zip(
            student_model.visual_prompt_learner.named_parameters(),
            teacher_model.visual_prompt_learner.named_parameters()
        ):
            param_teach.data.mul_(alpha).add_(param_stud.data, alpha=1 - alpha)

        # 更新 Text Prompts
        for (name_stud, param_stud), (name_teach, param_teach) in zip(
            student_model.text_prompt_learner.named_parameters(),
            teacher_model.text_prompt_learner.named_parameters()
        ):
            param_teach.data.mul_(alpha).add_(param_stud.data, alpha=1 - alpha)

    
    # --- Main training loop (VPT+TPT) ---
    p_bar = tqdm(range(1, max_iterations + 1), desc=f'VPT+TPT (EMA+SCCM) Training')
    
    model_student.train()
    model_teacher.eval() # Teacher 始终处于评估模式

    for iter_num in p_bar:
        # 1. 加载数据
        lb_sample = next(lb_dataloader)
        ulb_sample = next(ulb_dataloader)
        
        # (新) 有标签数据现在也包含强增强
        lb_unet_size_x_w, lb_unet_size_x_s = lb_sample['unet_size_img'].cuda(), lb_sample['unet_size_strong_aug'].cuda()
        ulb_unet_size_x_w, ulb_unet_size_x_s = ulb_sample['unet_size_img'].cuda(), ulb_sample['unet_size_strong_aug'].cuda()
        

        # (移除了 score_tensor 加载)
        
        # 2. 处理有标签数据 (GT Label)
        if 'unet_size_label' in lb_sample:
            lb_unet_size_label = lb_sample['unet_size_label']
            if isinstance(lb_unet_size_label, torch.Tensor):
                if lb_unet_size_label.dim() == 4 and lb_unet_size_label.size(1) == 1:
                    lb_unet_size_label = lb_unet_size_label.squeeze(1)
            else:
                lb_unet_size_label = torch.as_tensor(lb_unet_size_label)
            # ===== 根据数据集类型重映射标签（参考 train_synfoc.py）=====
            # Prostate: 原始 mask 中 0=前景，需要映射为 mask.eq(0) -> 1 (foreground)
            if args.dataset == 'prostate':
                lb_unet_size_mask = lb_unet_size_label.eq(0).long().cuda()
            elif args.dataset == 'BUSI':
                lb_unet_size_mask = lb_unet_size_label.eq(255).long().cuda()
            elif args.dataset == 'MNMS':
                lb_unet_size_mask = lb_unet_size_label.long().cuda()
            elif args.dataset == 'fundus':
                lb_unet_size_mask = (lb_unet_size_label <= 128).long().cuda() * 2
                lb_unet_size_mask[lb_unet_size_label == 0] = 1
            else:
                lb_unet_size_mask = lb_unet_size_label.long().cuda()

            B_lb, H, W = lb_unet_size_mask.shape
            K_classes = num_classes + 1  # background + n foreground classes

            # ===== 验证标签映射 =====
            if iter_num % 500 == 1:
                unique_labels_orig = torch.unique(lb_unet_size_label).cpu().tolist()
                unique_labels_mapped = torch.unique(lb_unet_size_mask).cpu().tolist()
                # 额外打印原始像素值到映射索引的映射（便于审计）
                raw_to_mapped = {}
                for raw in unique_labels_orig:
                    # 找到原始标签等于 raw 的位置，获取对应映射值（因为映射是逐像素的，取第一个即可）
                    mask_positions = (lb_unet_size_label == raw)
                    if mask_positions.any():
                        mapped_val = lb_unet_size_mask[mask_positions][0].item()
                    else:
                        mapped_val = None
                    raw_to_mapped[int(raw)] = int(mapped_val) if mapped_val is not None else None

                logging.info(f"iter {iter_num}: Label mapping - Original unique values: {unique_labels_orig} -> Mapped unique: {unique_labels_mapped}")
                logging.info(f"iter {iter_num}: Raw->Mapped sample mapping: {raw_to_mapped}")

            # --- (新) 修正：使用“平均池化 + 硬化”策略 ---
            N_patches_side = 14 # 目标 14x14

            # 1. (不变) 创建 One-hot
            one_hot = torch.zeros((B_lb, K_classes, H, W), dtype=torch.float32, device=lb_unet_size_mask.device)
            for c in range(K_classes):
                one_hot[:, c, :, :] = (lb_unet_size_mask == c).float()
            
            # 2. (不变) 使用 *平均池化* 下采样 (保留小目标)
            lb_label_14_blurry = F.adaptive_avg_pool2d(one_hot, (N_patches_side, N_patches_side))

            # 3. (新) “硬化”标签：找到最可能的类别并转回 one-hot
            hard_indices = torch.argmax(lb_label_14_blurry, dim=1) # [B_lb, 14, 14]

            # 4. (新) 从“硬”索引创建新的 one-hot 标签
            lb_label_14_hard = torch.zeros_like(lb_label_14_blurry) # [B_lb, K, 14, 14]
            lb_label_14_hard.scatter_(1, hard_indices.unsqueeze(1), 1.0)
            
            # 5. (新) 使用这个“硬”标签作为监督目标
            lb_sample['unet_label_14'] = lb_label_14_hard 
        else:
            B_lb = lb_unet_size_x_w.shape[0]
            K_classes = num_classes + 1

        # 3. 准备维度
        B_ulb = ulb_unet_size_x_w.shape[0]
        N_patches_flat = 196
        N_patches_side = 14

        # 4. VPT+TPT (EMA) 前向过程
        with amp_cm():
            from biomedclip_vpt_tpt_seg import preprocess_tensor_images
            
            # === Student 前向 (接收梯度, use_vpt=True) ===
            # 1. Labeled (weak aug) -> L_sup_w 和 L_sccm
            lb_images_preprocessed_w = preprocess_tensor_images(lb_unet_size_x_w, preprocess, str(device))
            lb_outputs_w = model_student(lb_images_preprocessed_w, args.class_text, use_vpt=True)
            
            # 2. Labeled (strong aug) -> L_sup_s
            lb_images_preprocessed_s = preprocess_tensor_images(lb_unet_size_x_s, preprocess, str(device))
            lb_outputs_s = model_student(lb_images_preprocessed_s, args.class_text, use_vpt=True)
            
            # 3. Unlabeled (strong aug) -> L_consis
            ulb_images_strong_preprocessed = preprocess_tensor_images(ulb_unet_size_x_s, preprocess, str(device))
            ulb_strong_outputs = model_student(ulb_images_strong_preprocessed, args.class_text, use_vpt=True)
            
            # === Teacher 前向 (不接收梯度) ===
            with torch.no_grad():
                # 4. Unlabeled (weak aug) -> L_consis 的伪标签 (use_vpt=True)
                ulb_images_weak_preprocessed = preprocess_tensor_images(ulb_unet_size_x_w, preprocess, str(device))
                ulb_weak_outputs = model_teacher(ulb_images_weak_preprocessed, args.class_text, use_vpt=True)
                
                # 5. (新) L_reg_vpt 的教师: 原始 CLIP 特征 (use_vpt=False)
                #    我们使用 model_student (与 model_teacher 具有相同的冻结主干) 来获取
                lb_feat_orig_w = model_student(lb_images_preprocessed_w, args.class_text, use_vpt=False)["patch_features"].detach()
                lb_feat_orig_s = model_student(lb_images_preprocessed_s, args.class_text, use_vpt=False)["patch_features"].detach()
                ulb_feat_orig_s = model_student(ulb_images_strong_preprocessed, args.class_text, use_vpt=False)["patch_features"].detach()

            # ========== (新) Loss 计算 (v7 / + L_reg_vpt) ==========
            
            loss_total = torch.tensor(0.0, device=device)
            loss_ce_w = torch.tensor(0.0, device=device) 
            loss_ce_s = torch.tensor(0.0, device=device) 
            loss_dice_w = torch.tensor(0.0, device=device) 
            loss_dice_s = torch.tensor(0.0, device=device) 
            loss_consis_map = torch.tensor(0.0, device=device)
            loss_consis_feat = torch.tensor(0.0, device=device)
            loss_sccm = torch.tensor(0.0, device=device) 
            loss_reg_vpt = torch.tensor(0.0, device=device) # (新)
            mask_sum = torch.tensor(0.0, device=device) 
            
            # 1. 监督损失 (L_sup) - [Labeled Data] (CE + Dice)
            if 'unet_label_14' in lb_sample and (args.lambda_ce > 0 or args.lambda_dice > 0):
                lb_gt_maps = lb_sample['unet_label_14'] # [B_lb, K, 14, 14] (硬标签)

                # --- 弱增强监督 (Weak) ---
                lb_pred_maps_w = lb_outputs_w["H_semantic_maps"].view(B_lb, K_classes, N_patches_side, N_patches_side)
                if args.lambda_ce > 0:
                    log_pred_w = F.log_softmax(lb_pred_maps_w, dim=1)
                    loss_ce_w = F.kl_div(log_pred_w, lb_gt_maps, reduction='batchmean')
                if args.lambda_dice > 0:
                    prob_pred_w = F.softmax(lb_pred_maps_w, dim=1)
                    loss_dice_w = dice_loss_fn(prob_pred_w, lb_gt_maps)

                # --- 强增强监督 (Strong) ---
                lb_pred_maps_s = lb_outputs_s["H_semantic_maps"].view(B_lb, K_classes, N_patches_side, N_patches_side)
                if args.lambda_ce > 0:
                    log_pred_s = F.log_softmax(lb_pred_maps_s, dim=1)
                    loss_ce_s = F.kl_div(log_pred_s, lb_gt_maps, reduction='batchmean')
                if args.lambda_dice > 0:
                    prob_pred_s = F.softmax(lb_pred_maps_s, dim=1)
                    loss_dice_s = dice_loss_fn(prob_pred_s, lb_gt_maps)

                # 合并监督损失
                loss_sup_ce = (loss_ce_w + loss_ce_s) / 2.0
                loss_sup_dice = (loss_dice_w + loss_dice_s) / 2.0
                loss_sup_total = (args.lambda_ce * loss_sup_ce) + (args.lambda_dice * loss_sup_dice)
                
                loss_total += loss_sup_total
            
            # 2. 一致性损失 (L_consis) - [Unlabeled Data] FixMatch
            if args.lambda_consis > 0:
                # --- FixMatch 阈值 ---
                ulb_weak_maps_logits = ulb_weak_outputs["H_semantic_maps"].view(B_ulb, K_classes, N_patches_side, N_patches_side)
                with torch.no_grad():
                    pseudo_label_prob = F.softmax(ulb_weak_maps_logits / args.T_consis, dim=1)
                    max_prob, _ = pseudo_label_prob.max(dim=1) # [B_ulb, 14, 14]
                    mask = (max_prob >= args.tau).float() # [B_ulb, 14, 14]
                    mask_sum = mask.sum()
                
                # --- Map Consistency ---
                ulb_strong_maps_logits = ulb_strong_outputs["H_semantic_maps"].view(B_ulb, K_classes, N_patches_side, N_patches_side)
                ulb_strong_log_prob = F.log_softmax(ulb_strong_maps_logits, dim=1)
                
                kl_div_map = F.kl_div(ulb_strong_log_prob, pseudo_label_prob, reduction='none') # [B_ulb, K, 14, 14]
                loss_consis_map_per_patch = kl_div_map.sum(dim=1)
                
                if mask_sum > 0:
                    loss_consis_map = (loss_consis_map_per_patch * mask).sum() / mask_sum
                else:
                    loss_consis_map = torch.tensor(0.0, device=device) # 如果没有 patch 通过阈值
                
                # --- Feature Consistency ---
                feat_weak = ulb_weak_outputs["patch_features"]  # [B_ulb, 196, D]
                feat_strong = ulb_strong_outputs["patch_features"] # [B_ulb, 196, D]
                
                loss_consis_feat_per_patch = F.mse_loss(feat_strong, feat_weak.detach(), reduction='none').mean(dim=2) # [B_ulb, 196]
                
                mask_flat = mask.view(B_ulb, N_patches_flat) # [B_ulb, 196]
                if mask_sum > 0:
                    loss_consis_feat = (loss_consis_feat_per_patch * mask_flat).sum() / mask_sum
                else:
                    loss_consis_feat = torch.tensor(0.0, device=device)

                
                # 合并一致性损失
                loss_consis_total = loss_consis_map + (args.feat_consis_scale * loss_consis_feat)
                loss_total += args.lambda_consis * loss_consis_total
            
            # 3. SCCM 纯文本对齐损失 (L_sccm) - [Labeled Data] Student_Proto vs LLM_Proto
            if args.lambda_sccm > 0:
                # 学生原型 W_robust (来自 Labeled Data 的 *弱增强* 前向，[K, D])
                student_prototypes_W_robust = lb_outputs_w["W_robust"]
                
                # 教师原型 P_g (来自预计算，[K, D])
                teacher_prototypes_Pg = llm_teacher_prototypes_Pg.detach() 
                
                loss_sccm = F.mse_loss(student_prototypes_W_robust, teacher_prototypes_Pg)
                
                loss_total += args.lambda_sccm * loss_sccm

            # 4. (新) VPT 特征正则化损失 (L_reg_vpt) - [All Data] Prompted_Feat vs Original_Feat
            if args.lambda_reg_vpt > 0:
                # 获取学生 (prompted) 特征
                lb_feat_prompted_w = lb_outputs_w["patch_features"]
                lb_feat_prompted_s = lb_outputs_s["patch_features"]
                ulb_feat_prompted_s = ulb_strong_outputs["patch_features"]
                
                # (教师特征 lb_feat_orig_w/s, ulb_feat_orig_s 已在 with no_grad() 中获取)
                
                # 计算 MSE 损失
                loss_reg_vpt_lb_w = F.mse_loss(lb_feat_prompted_w, lb_feat_orig_w)
                loss_reg_vpt_lb_s = F.mse_loss(lb_feat_prompted_s, lb_feat_orig_s)
                loss_reg_vpt_ulb_s = F.mse_loss(ulb_feat_prompted_s, ulb_feat_orig_s)
                
                # 在所有数据上平均
                loss_reg_vpt = (loss_reg_vpt_lb_w + loss_reg_vpt_lb_s + loss_reg_vpt_ulb_s) / 3.0
                
                loss_total += args.lambda_reg_vpt * loss_reg_vpt
            
        # 5. 反向传播
        optimizer.zero_grad()
        scaler.scale(loss_total).backward()
        scaler.step(optimizer)
        scaler.update()

        # 6. 更新 EMA Teacher 模型权重
        update_ema_variables(model_student, model_teacher, args.ema_decay)

        # --- Periodic checkpointing (student + teacher prompts) ---
        if args.save_model and args.save_every > 0 and (iter_num % args.save_every == 0):
            try:
                save_path_student = os.path.join(snapshot_path, f"vpt_tpt_student_iter{iter_num:06d}.pth")
                model_student.save_all_prompts(save_path_student)

                save_path_teacher = os.path.join(snapshot_path, f"vpt_tpt_teacher_iter{iter_num:06d}.pth")
                model_teacher.save_all_prompts(save_path_teacher)

                logging.info(f"Saved periodic checkpoints at iter {iter_num}: {save_path_student}, {save_path_teacher}")
            except Exception as e:
                logging.exception(f"Failed to save periodic checkpoints at iter {iter_num}: {e}")


        # 7. 更新日志
        mask_ratio = (mask_sum / (B_ulb * N_patches_flat)) if (B_ulb * N_patches_flat) > 0 else 0.0
        writer.add_scalar('train/loss_total', loss_total.item(), iter_num)
        writer.add_scalar('train/loss_ce_w', loss_ce_w.item(), iter_num) 
        writer.add_scalar('train/loss_ce_s', loss_ce_s.item(), iter_num) 
        writer.add_scalar('train/loss_dice_w', loss_dice_w.item(), iter_num) 
        writer.add_scalar('train/loss_dice_s', loss_dice_s.item(), iter_num) 
        writer.add_scalar('train/loss_consis_map', loss_consis_map.item(), iter_num)
        writer.add_scalar('train/loss_consis_feat', loss_consis_feat.item(), iter_num)
        writer.add_scalar('train/loss_sccm', loss_sccm.item(), iter_num) 
        writer.add_scalar('train/loss_reg_vpt', loss_reg_vpt.item(), iter_num) # (新)
        writer.add_scalar('train/mask_ratio', mask_ratio.item(), iter_num) 
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], iter_num)

        # Update tqdm
        # (新) 更新 tqdm 描述
        loss_sup_avg = ((loss_ce_w + loss_ce_s)/2.0 + (loss_dice_w + loss_dice_s)/2.0).item()
        loss_consis_avg = (loss_consis_map + loss_consis_feat).item()
        p_bar.set_description(
            'iter %d: total=%.4f (sup=%.4f, consis=%.4f, sccm=%.4f, vpt_reg=%.4f, mask_ratio=%.2f)' % 
            (iter_num, loss_total.item(), loss_sup_avg, loss_consis_avg, loss_sccm.item(), loss_reg_vpt.item(), mask_ratio.item())
        )

        # Log to console
        if iter_num % 50 == 0:
            logging.info(
                "iter %d: total=%.5f (ce_w=%.5f, ce_s=%.5f, dice_w=%.5f, dice_s=%.5f, cons_map=%.5f, cons_feat=%.5f, sccm=%.5f, vpt_reg=%.5f, mask_ratio=%.3f)",
                iter_num,
                loss_total.item(),
                loss_ce_w.item(),
                loss_ce_s.item(),
                loss_dice_w.item(),
                loss_dice_s.item(),
                loss_consis_map.item(),
                loss_consis_feat.item(),
                loss_sccm.item(),
                loss_reg_vpt.item(), # (新)
                mask_ratio.item()
            )
            
    if p_bar is not None:
        p_bar.close()

    # ========== (新) Save Final Student and Teacher Weights ==========
    if args.save_model:
        save_path_student = os.path.join(snapshot_path, "vpt_tpt_student_final_weights.pth")
        model_student.save_all_prompts(save_path_student)
        
        save_path_teacher = os.path.join(snapshot_path, "vpt_tpt_teacher_final_weights.pth")
        model_teacher.save_all_prompts(save_path_teacher)
        
        logging.info(f"Saved final STUDENT prompts to {save_path_student}")
        logging.info(f"Saved final TEACHER (EMA) prompts to {save_path_teacher} (Recommended for inference)")


    writer.close()
    logging.info("VPT+TPT (EMA+SCCM) training finished.")


if __name__ == "__main__":
    # ... (snapshot_path 和数据集配置保持不变) ...
    if len(args.save_name) == 0:
        args.save_name = f'vpt_tpt_seg_lb{args.lb_num}_dm{args.lb_domain}'
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
        args.class_text = ['background', 'optic cup', 'optic disc']

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
        args.class_text = ['background', 'prostate']
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
        args.class_text = ['background', 'left ventricle', 'left ventricle myocardium', 'right ventricle']
    elif args.dataset == 'BUSI':
        train_data_path="/app/MixDSemi/data/Dataset_BUSI_with_GT"
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
        args.class_text = ['background', 'breast tumor']
    
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
