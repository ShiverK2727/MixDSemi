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

import dataloaders.custom_transforms as tr
from dataloaders.dataloader_dc import BUSISegmentation, FundusSegmentation, MNMSSegmentation, ProstateSegmentation
# 导入 VPT+TPT 分割模型
from biomedclip_vpt_tpt_seg import build_vpt_tpt_seg_model
from utils.text_sampler_an import TextSampler
from torch.cuda.amp import GradScaler, autocast
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
parser.add_argument('--preprocess_dir', type=str, default=None,
                    help='Override preprocessing directory for score tensors')
parser.add_argument('--llm_model', type=str, default='gemini',
                    choices=['gemini', 'GPT5', 'DeepSeek'],
                    help='LLM model used to generate score tensors')
parser.add_argument('--describe_nums', type=int, default=40,
                    choices=[20, 40, 60, 80],
                    help='Number of textual descriptions for preprocessing')

# ==================== VPT + TPT Configuration ====================
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

args = parser.parse_args()

TEXT_DATASET_DEFAULTS = {
    'fundus': 'Fundus',
    'prostate': 'ProstateSlice',
    'MNMS': 'MNMS',
    'BUSI': 'BUSI',
}
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

    # --- Text sampling ---
    # 使用 text_sampler_an.py，其 load_texts 返回二元组：
    #   (per_class_lists, flat_list)
    # - per_class_lists: list of lists，每项是一个类别的文本列表
    # - flat_list: 所有类别按顺序拼接的扁平列表
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
        len(per_class_lists),
        len(flat_list),
    )

    # --- 实例化 VPT+TPT 分割模型 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    vpt_tpt_model, preprocess, tokenizer = build_vpt_tpt_seg_model(
        model_path=args.biomedclip_path,
        device=str(device),
        visual_num_prompts=args.visual_num_prompts,
        text_num_prompts=args.text_num_prompts,
        freeze_backbone=args.freeze_backbone,
    )
    
    # 收集所有可训练参数（VPT + TPT）
    trainable_params = []
    trainable_params.extend(vpt_tpt_model.visual_prompt_learner.parameters())
    trainable_params.extend(vpt_tpt_model.text_prompt_learner.parameters())
    trainable_params = [p for p in trainable_params if p.requires_grad]
    
    optimizer = optim.AdamW(
        trainable_params,
        lr=args.vpt_tpt_lr,
        weight_decay=args.vpt_tpt_weight_decay,
    )
    
    logging.info(
        "VPT+TPT model initialized on %s (trainable params=%d)",
        device,
        sum(p.numel() for p in trainable_params),
    )

    # ... (DataLoader 逻辑保持不变) ...
    lb_loader = DataLoader(lb_dataset, batch_size=args.label_bs, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    ulb_loader = DataLoader(ulb_dataset, batch_size=args.unlabel_bs, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    logging.info("Using random sampler for unlabeled data loader.")
    lb_dataloader = cycle(lb_loader)
    ulb_dataloader = cycle(ulb_loader)
    
    logging.info(f"Total iterations: {max_iterations}")
    
    logging.info(
        "CLIP loss weights: mv_anchor=%.3f, sw_reg=%.3f (Ortho removed)",
        args.clip_loss_mv_anchor_weight,
        args.clip_loss_sw_reg_weight,
    )
    
    # ... (AMP 逻辑保持不变) ...
    amp_enabled = bool(args.amp)
    scaler = GradScaler(enabled=amp_enabled)
    amp_cm = autocast if amp_enabled else contextlib.nullcontext
    logging.info(f"AMP (Mixed Precision) enabled: {amp_enabled}")

    # --- 预编码类别文本特征 (使用 args.class_text 和 per_class_lists) ---
    # 为每个类别（包括背景）获取其所有文本描述并预编码
    logging.info("Pre-encoding class text features for K=%d classes...", len(args.class_text))
    
    # 为每个类别收集文本描述
    # args.class_text = ['background', 'class1', 'class2', ...]
    # per_class_lists[i] 对应 args.class_text[i+1]（因为 per_class_lists 通常不含 background）
    # 但为了安全，我们直接使用 per_class_lists 的长度匹配
    
    class_text_features_list = []
    with torch.no_grad():
        for i, class_name in enumerate(args.class_text):
            if i == 0:
                # 背景类：使用类名本身
                texts_for_class = [class_name]
            else:
                # 前景类：使用 per_class_lists 中对应的文本（索引 i-1）
                if i - 1 < len(per_class_lists):
                    texts_for_class = per_class_lists[i - 1]
                else:
                    # 如果 per_class_lists 不够长，回退到类名
                    logging.warning(f"per_class_lists 索引 {i-1} 越界，使用类名 '{class_name}'")
                    texts_for_class = [class_name]
            
            # 使用 VPT_TPT 模型的 encode_text_with_prompts 预编码
            # 注意：这会应用 TPT (CoOp-style) prompts
            W_class = vpt_tpt_model.encode_text_with_prompts(texts_for_class)  # [len(texts), D]
            # 对该类的所有文本取平均得到类原型
            W_class_mean = W_class.mean(dim=0, keepdim=True)  # [1, D]
            class_text_features_list.append(W_class_mean)
        
        # 堆叠为 [K, D]
        class_text_features = torch.cat(class_text_features_list, dim=0)  # [K, D]
        class_text_features = F.normalize(class_text_features, dim=-1)
    
    logging.info(f"Pre-encoded class_text_features: {class_text_features.shape}")
    # class_text_features 现在可以在训练循环中直接使用

    # --- Main training loop (VPT+TPT) ---
    p_bar = tqdm(range(1, max_iterations + 1), desc=f'VPT+TPT Training')
    
    vpt_tpt_model.train()

    for iter_num in p_bar:
        lb_sample = next(lb_dataloader)
        ulb_sample = next(ulb_dataloader)
        lb_unet_size_x_w = lb_sample['unet_size_img']
        ulb_unet_size_x_w, ulb_unet_size_x_s = ulb_sample['unet_size_img'], ulb_sample['unet_size_strong_aug']
        
        lb_unet_size_x_w, ulb_unet_size_x_w, ulb_unet_size_x_s = lb_unet_size_x_w.cuda(), ulb_unet_size_x_w.cuda(), ulb_unet_size_x_s.cuda()

        # ====== 下采样标签到 14x14 并构建多通道（背景, 类1, 类2, ...） ======
        if 'unet_size_label' in lb_sample:
            lb_unet_size_label = lb_sample['unet_size_label']
            # 规范形状 -> (B, H, W)
            if isinstance(lb_unet_size_label, torch.Tensor):
                if lb_unet_size_label.dim() == 4 and lb_unet_size_label.size(1) == 1:
                    lb_unet_size_label = lb_unet_size_label.squeeze(1)
            else:
                lb_unet_size_label = torch.as_tensor(lb_unet_size_label)
            lb_unet_size_label = lb_unet_size_label.long().cuda()

            B, H, W = lb_unet_size_label.shape
            channels = num_classes + 1  # background + n foreground classes

            # 简单检查
            max_label_val = int(lb_unet_size_label.max().item())
            if max_label_val >= channels:
                logging.warning(f"Label max value ({max_label_val}) >= expected channels ({channels}).")

            # 构建 one-hot (B, C, H, W)
            one_hot = torch.zeros((B, channels, H, W), dtype=torch.float32, device=lb_unet_size_label.device)
            for c in range(channels):
                one_hot[:, c, :, :] = (lb_unet_size_label == c).float()

            # 使用自适应平均池化下采样到 14x14
            lb_label_14 = F.adaptive_avg_pool2d(one_hot, (14, 14))
            lb_sample['unet_label_14'] = lb_label_14

        # ========== VPT+TPT 前向过程 ==========
        # 对 labeled 和 unlabeled weak 图像进行前向
        with amp_cm():
            # 1. 对 labeled 图像前向
            from biomedclip_vpt_tpt_seg import preprocess_tensor_images
            lb_images_preprocessed = preprocess_tensor_images(lb_unet_size_x_w, preprocess, str(device))
            lb_outputs = vpt_tpt_model(lb_images_preprocessed, args.class_text)
            # lb_outputs: {
            #   "H_semantic_maps": [B_lb, K, 196],
            #   "patch_features": [B_lb, 196, D],
            #   "W_robust": [K, D]
            # }
            
            # 2. 对 unlabeled weak 图像前向
            ulb_images_weak_preprocessed = preprocess_tensor_images(ulb_unet_size_x_w, preprocess, str(device))
            ulb_weak_outputs = vpt_tpt_model(ulb_images_weak_preprocessed, args.class_text)
            
            # 3. 对 unlabeled strong 图像前向
            ulb_images_strong_preprocessed = preprocess_tensor_images(ulb_unet_size_x_s, preprocess, str(device))
            ulb_strong_outputs = vpt_tpt_model(ulb_images_strong_preprocessed, args.class_text)
            
            # ========== Loss 计算（留空，交由用户设计） ==========
            # 可用的变量：
            # - lb_outputs["H_semantic_maps"]: [B_lb, K, 196] labeled 语义图
            # - lb_outputs["patch_features"]: [B_lb, 196, D] labeled patch 特征
            # - lb_sample['unet_label_14']: [B_lb, K, 14, 14] 下采样的标签（如果存在）
            # - ulb_weak_outputs, ulb_strong_outputs: unlabeled 的输出
            # - class_text_features: [K, D] 预编码的类别文本特征
            
            # TODO: 在这里添加你的损失函数
            loss_total = torch.tensor(0.0, device=device)  # 占位符
            
        optimizer.zero_grad()
        scaler.scale(loss_total).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update Tensorboard
        writer.add_scalar('train/loss_total', loss_total.item(), iter_num)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], iter_num)

        # Update tqdm
        p_bar.set_description(
            'iter %d: loss=%.4f' % (iter_num, loss_total.item())
        )

        # Log to console
        if iter_num % 50 == 0:
            logging.info(
                "iter %d: loss=%.6f",
                iter_num,
                loss_total.item(),
            )
            
    if p_bar is not None:
        p_bar.close()

    # ========== Save Final VPT+TPT Weights ==========
    if args.save_model:
        save_path = os.path.join(snapshot_path, "vpt_tpt_final_weights.pth")
        vpt_tpt_model.save_all_prompts(save_path)
        logging.info(f"Saved final VPT+TPT weights to {save_path}")

    writer.close()
    logging.info("VPT+TPT training finished.")


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
    # (移除了 dataset_test_class 因为这个脚本不包含测试)
    
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
