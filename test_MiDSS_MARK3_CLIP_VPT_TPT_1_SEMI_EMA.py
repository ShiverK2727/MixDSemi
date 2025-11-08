import argparse
import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import dataloaders.custom_transforms as tr
from dataloaders.dataloader import (BUSISegmentation, FundusSegmentation,
									   MNMSSegmentation, ProstateSegmentation)
from biomedclip_vpt_tpt_seg import build_vpt_tpt_seg_model, preprocess_tensor_images
from utils.text_sampler_an import TextSampler


TEXT_DATASET_DEFAULTS = {
	'fundus': 'Fundus',
	'prostate': 'ProstateSlice',
	'MNMS': 'MNMS',
	'BUSI': 'BUSI',
}


def set_random_seed(seed: int) -> None:
	"""Set python, numpy, and torch seeds for reproducibility."""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def get_dataset_config(dataset: str) -> dict:
	"""Return dataset-specific configuration matching the training script."""
	if dataset == 'fundus':
		return {
			'train_data_path': '/app/MixDSemi/data/Fundus',
			'dataset_class': FundusSegmentation,
			'num_channels': 3,
			'patch_size': 256,
			'num_classes': 2,
			'min_v': 0.5,
			'max_v': 1.5,
			'fillcolor': 255,
			'max_iter': 30000,
			'domain_len': [50, 99, 320, 320],
			'default_domain_num': 4,
		}
	if dataset == 'prostate':
		return {
			'train_data_path': '/app/MixDSemi/data/ProstateSlice',
			'dataset_class': ProstateSegmentation,
			'num_channels': 1,
			'patch_size': 384,
			'num_classes': 1,
			'min_v': 0.1,
			'max_v': 2,
			'fillcolor': 255,
			'max_iter': 60000,
			'domain_len': [225, 305, 136, 373, 338, 133],
			'default_domain_num': 6,
		}
	if dataset == 'MNMS':
		return {
			'train_data_path': '/app/MixDSemi/data/mnms',
			'dataset_class': MNMSSegmentation,
			'num_channels': 1,
			'patch_size': 288,
			'num_classes': 3,
			'min_v': 0.1,
			'max_v': 2,
			'fillcolor': 0,
			'max_iter': 60000,
			'domain_len': [1030, 1342, 525, 550],
			'default_domain_num': 4,
		}
	if dataset == 'BUSI':
		return {
			'train_data_path': '/app/MixDSemi/data/Dataset_BUSI_with_GT',
			'dataset_class': BUSISegmentation,
			'num_channels': 1,
			'patch_size': 256,
			'num_classes': 1,
			'min_v': 0.1,
			'max_v': 2,
			'fillcolor': 0,
			'max_iter': 30000,
			'domain_len': [350, 168],
			'default_domain_num': 2,
		}
	raise ValueError(f"Unknown dataset: {dataset}")


def build_test_loader(cfg: dict, args, domains: list[int]) -> DataLoader:
	"""Construct a deterministic test DataLoader matching training preprocessing."""
	base_transform = transforms.Compose([
		tr.Normalize_tf(dataRange=[0, 1]),
		tr.ToTensor(unet_size=cfg['patch_size'])
	])

	dataset = cfg['dataset_class'](
		base_dir=cfg['train_data_path'],
		phase='test',
		splitid=-1,
		domain=domains,
		weak_transform=None,
		strong_tranform=None,
		normal_toTensor=base_transform,
		img_size=cfg['patch_size'],
		is_RGB=(cfg['num_channels'] == 3),
	)

	loader = DataLoader(
		dataset,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.num_workers,
		pin_memory=True,
		drop_last=False,
	)
	return loader


def get_class_text_list(dataset: str, num_classes: int) -> list:
	"""Return class text list matching the training script."""
	if dataset == 'fundus':
		return ['background', 'optic cup', 'optic disc']
	elif dataset == 'prostate':
		return ['background', 'prostate']
	elif dataset == 'MNMS':
		return ['background', 'left ventricle', 'left ventricle myocardium', 'right ventricle']
	elif dataset == 'BUSI':
		return ['background', 'breast tumor']
	else:
		# 默认：background + num_classes 个前景类
		return ['background'] + [f'class_{i}' for i in range(1, num_classes + 1)]


def ensure_mask_shape(mask: torch.Tensor) -> torch.Tensor:
	"""Ensure mask tensor has shape (B, 1, H, W) and values in [0, 1]."""
	if mask.dim() == 3:
		mask = mask.unsqueeze(1)
	mask = mask.float()
	if mask.max() > 1.5:
		mask = mask / 255.0
	return mask.clamp(0.0, 1.0)


def tensor_to_display_image(tensor: torch.Tensor) -> np.ndarray:
	"""Convert a tensor in CHW format (0..1) to numpy HxW[x3] for matplotlib."""
	tensor = tensor.detach().cpu()
	if tensor.dim() == 3:
		if tensor.shape[0] == 1:
			return tensor[0].numpy()
		return tensor.permute(1, 2, 0).numpy()
	if tensor.dim() == 2:
		return tensor.numpy()
	raise ValueError(f"Unexpected tensor shape for display: {tuple(tensor.shape)}")


def visualize_sample(
	image_tensor: torch.Tensor,
	mask_high: torch.Tensor,
	mask_down_avg: torch.Tensor,
	model_prediction: torch.Tensor,
	output_path: str,
) -> None:
	"""Create a four-panel visualization and save it to disk."""
	img_np = tensor_to_display_image(image_tensor)
	mask_high_np = tensor_to_display_image(mask_high.squeeze(0))
	mask_avg_np = tensor_to_display_image(mask_down_avg.squeeze(0))
	pred_np = model_prediction.detach().cpu().numpy()

	fig, axes = plt.subplots(1, 4, figsize=(16, 4))

	cmap_image = 'gray' if (img_np.ndim == 2 or img_np.shape[-1] == 1) else None
	axes[0].imshow(img_np, cmap=cmap_image)
	axes[0].set_title('Original Image')
	axes[0].axis('off')

	axes[1].imshow(mask_high_np, cmap='gray', vmin=0.0, vmax=1.0)
	axes[1].set_title('Original Mask')
	axes[1].axis('off')

	axes[2].imshow(mask_avg_np, cmap='gray', vmin=0.0, vmax=1.0, interpolation='bilinear')
	axes[2].set_title('Mask (AvgPool 14x14)')
	axes[2].axis('off')

	im = axes[3].imshow(pred_np, cmap='viridis', vmin=0.0, vmax=1.0, interpolation='nearest')
	axes[3].set_title('Model Prediction (14x14)')
	axes[3].axis('off')
	fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

	fig.tight_layout()
	fig.savefig(output_path, dpi=150)
	plt.close(fig)


def parse_args():
	parser = argparse.ArgumentParser(
		description="Run VPT+TPT+EMA inference on the test set and visualize predictions."
	)
	parser.add_argument('--dataset', type=str, default='prostate',
						choices=['fundus', 'prostate', 'MNMS', 'BUSI'])
	parser.add_argument('--biomedclip_path', type=str, default='/root/models/BiomedCLIP',
						help='Root directory containing BiomedCLIP weights and config JSON')
	parser.add_argument('--visual_num_prompts', type=int, default=4,
						help='Number of visual prompts per layer (VPT-Deep)')
	parser.add_argument('--text_num_prompts', type=int, default=4,
						help='Number of text context prompts (TPT CoOp-style)')
	parser.add_argument('--freeze_backbone', action='store_true', default=True,
						help='Freeze BiomedCLIP backbone (only train prompts)')
	parser.add_argument('--prompts_path', type=str, required=True,
						help='Path to the saved VPT+TPT prompts (e.g., vpt_tpt_teacher_final_weights.pth).')
	parser.add_argument('--output_dir', type=str, required=True,
						help='Directory to store visualization figures.')
	parser.add_argument('--text_root', type=str, default='/app/MixDSemi/SynFoCLIP/code/text-an',
						help='Directory containing dataset text description JSON files')
	parser.add_argument('--preprocess_root', type=str, default='/app/MixDSemi/SynFoCLIP/preprocess')
	parser.add_argument('--llm_model', type=str, default='GPT5', choices=['gemini', 'GPT5', 'DeepSeek'])
	parser.add_argument('--describe_nums', type=int, default=20, choices=[20, 40, 60, 80])
	parser.add_argument('--gpu', type=str, default='0')
	parser.add_argument('--batch_size', type=int, default=1)
	parser.add_argument('--num_workers', type=int, default=2)
	parser.add_argument('--seed', type=int, default=1337)
	parser.add_argument('--max_samples', type=int, default=None,
						help='If set, limit the number of processed samples.')
	parser.add_argument('--overwrite', action='store_true',
						help='Overwrite existing files in output_dir if necessary.')
	parser.add_argument('--test_domains', type=int, nargs='*', default=None,
						help='Optional list of domain indices to evaluate (defaults to all domains).')
	return parser.parse_args()


def main():
	args = parse_args()
	set_random_seed(args.seed)

	device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

	os.makedirs(args.output_dir, exist_ok=True)
	if (len(os.listdir(args.output_dir)) > 0) and not args.overwrite:
		raise RuntimeError(
			f"Output directory {args.output_dir} is not empty. Pass --overwrite to replace existing files."
		)

	cfg = get_dataset_config(args.dataset)
	if args.test_domains is None:
		domains = list(range(1, cfg['default_domain_num'] + 1))
	else:
		domains = args.test_domains

	test_loader = build_test_loader(cfg, args, domains)

	# 实例化 VPT+TPT 模型
	print(f"Loading VPT+TPT model from {args.biomedclip_path}...")
	model, preprocess, tokenizer = build_vpt_tpt_seg_model(
		model_path=args.biomedclip_path,
		device=str(device),
		visual_num_prompts=args.visual_num_prompts,
		text_num_prompts=args.text_num_prompts,
		freeze_backbone=args.freeze_backbone,
	)
	
	# 加载训练好的 prompts
	print(f"Loading prompts from {args.prompts_path}...")
	model.load_all_prompts(args.prompts_path)
	model.eval()
	
	# 获取类别文本列表
	class_text = get_class_text_list(args.dataset, cfg['num_classes'])
	print(f"Class text list: {class_text}")
	
	processed = 0
	with torch.no_grad():
		for batch in tqdm(test_loader, desc='Inference', total=len(test_loader)):
			images = batch['image'].to(device)
			masks_high = ensure_mask_shape(batch['label'].to(device))

			# 预处理图像以适配 CLIP
			images_preprocessed = preprocess_tensor_images(images, preprocess, str(device))
			
			# VPT+TPT 前向传播
			outputs = model(images_preprocessed, class_text)
			H_semantic_maps = outputs["H_semantic_maps"]  # [B, K_classes, 196]
			
			# 重塑为 14x14 grid
			B = images.shape[0]
			K_classes = cfg['num_classes'] + 1  # background + foreground classes
			N_patches_side = 14
			
			# 将 H_semantic_maps reshape 为 [B, K_classes, 14, 14]
			semantic_maps_grid = H_semantic_maps.view(B, K_classes, N_patches_side, N_patches_side)
			
			# 对前景类取平均作为预测（如果是二分类：背景 vs 前景）
			if cfg['num_classes'] == 1:
				# 单类分割：使用 sigmoid(foreground_logits)
				prediction_map = torch.sigmoid(semantic_maps_grid[:, 1, :, :])  # [B, 14, 14]
			else:
				# 多类分割：对所有前景类使用 softmax，然后合并
				semantic_probs = F.softmax(semantic_maps_grid, dim=1)  # [B, K, 14, 14]
				# 合并所有前景类的概率
				prediction_map = semantic_probs[:, 1:, :, :].sum(dim=1)  # [B, 14, 14]
			
			# 使用 avgpool2d 下采样 mask
			mask_down_avg = F.adaptive_avg_pool2d(masks_high, (N_patches_side, N_patches_side))

			for idx in range(B):
				img_tensor = images[idx]
				mask_high_tensor = masks_high[idx]
				mask_avg_tensor = mask_down_avg[idx]
				pred_tensor = prediction_map[idx]

				if isinstance(batch['img_name'], list):
					img_name = batch['img_name'][idx]
				else:
					img_name = batch['img_name']
					if isinstance(img_name, list):
						img_name = img_name[idx]

				stem = os.path.splitext(str(img_name))[0]
				output_path = os.path.join(args.output_dir, f"{stem}.png")

				visualize_sample(
					image_tensor=img_tensor,
					mask_high=mask_high_tensor,
					mask_down_avg=mask_avg_tensor,
					model_prediction=pred_tensor,
					output_path=output_path,
				)

				processed += 1
				if args.max_samples is not None and processed >= args.max_samples:
					return
	
	print(f"Inference complete. Processed {processed} samples. Results saved to {args.output_dir}")


if __name__ == '__main__':
	main()

