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
from dataloaders.dataloader_dc import (BUSISegmentation, FundusSegmentation,
									   MNMSSegmentation, ProstateSegmentation)
from biomedclip_vpt_RD import VPT_CLIP_RD
from utils.text_sampler_v2 import TextSampler


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
		preprocess_dir=args.preprocess_root,
		llm_model=args.llm_model,
		describe_nums=args.describe_nums,
		return_score=False,
		allow_missing_scores=True,
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


def prepare_text_anchors(model: VPT_CLIP_RD, args, dataset_key: str, device: torch.device) -> torch.Tensor:
	"""Encode text anchors (two target subsets + background + style)."""
	text_sampler = TextSampler(args.text_root)
	targets_texts, style_texts, _ = text_sampler.load_texts(
		dataset=dataset_key,
		llm=args.llm_model,
		describe_nums=args.describe_nums,
	)

	_, text_subsets = text_sampler.sample_subsets(targets_texts, num_samples=args.text_num_subsets)
	if len(text_subsets) == 0:
		raise RuntimeError("No text subsets available; please check text_root / describe_nums configuration.")
	if len(text_subsets) == 1:
		text_subsets = text_subsets * 2

	anchors = []
	for subset in text_subsets[:2]:
		tokens = model.tokenizer(subset).to(device)
		enc = model.encode_text(tokens).mean(dim=0, keepdim=True)
		anchors.append(enc)

	bg_texts = ["background", "empty space", "other tissue"]
	bg_tokens = model.tokenizer(bg_texts).to(device)
	bg_anchor = model.encode_text(bg_tokens).mean(dim=0, keepdim=True)
	anchors.append(bg_anchor)

	if len(style_texts) == 0:
		style_texts = ["artifact", "style"]
	style_tokens = model.tokenizer(style_texts).to(device)
	style_anchor = model.encode_text(style_tokens).mean(dim=0, keepdim=True)
	anchors.append(style_anchor)

	anchor_tensor = torch.cat(anchors, dim=0).to(device)
	return anchor_tensor


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
	mask_down_nearest: torch.Tensor,
	model_prediction: torch.Tensor,
	output_path: str,
) -> None:
	"""Create a four-panel visualization and save it to disk."""
	img_np = tensor_to_display_image(image_tensor)
	mask_high_np = tensor_to_display_image(mask_high.squeeze(0))
	mask_nearest_np = tensor_to_display_image(mask_down_nearest.squeeze(0))
	pred_np = model_prediction.detach().cpu().numpy()

	fig, axes = plt.subplots(1, 4, figsize=(16, 4))

	cmap_image = 'gray' if (img_np.ndim == 2 or img_np.shape[-1] == 1) else None
	axes[0].imshow(img_np, cmap=cmap_image)
	axes[0].set_title('Original Image')
	axes[0].axis('off')

	axes[1].imshow(mask_high_np, cmap='gray', vmin=0.0, vmax=1.0)
	axes[1].set_title('Original Mask')
	axes[1].axis('off')

	axes[2].imshow(mask_nearest_np, cmap='gray', vmin=0.0, vmax=1.0, interpolation='nearest')
	axes[2].set_title('Mask (Nearest 14x14)')
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
		description="Run VPT-CLIP-RD inference on the test set and visualize predictions."
	)
	parser.add_argument('--dataset', type=str, default='prostate',
						choices=['fundus', 'prostate', 'MNMS', 'BUSI'])
	parser.add_argument('--biomedclip_path', type=str, default='/root/models/BiomedCLIP')
	parser.add_argument('--biomedclip_num_prompts', type=int, default=4)
	parser.add_argument('--biomedclip_embed_dim', type=int, default=768)
	parser.add_argument('--biomedclip_init_std', type=float, default=0.02)
	parser.add_argument('--biomedclip_prompt_scale_init', type=float, default=1.0)
	parser.add_argument('--biomedclip_disable_scale', action='store_true')
	parser.add_argument('--prompts_path', type=str, required=True,
						help='Path to the saved VPT prompts (e.g., vpt_anchor_final.pth).')
	parser.add_argument('--output_dir', type=str, required=True,
						help='Directory to store visualization figures.')
	parser.add_argument('--text_root', type=str, default='/app/MixDSemi/SynFoCLIP/code/text')
	parser.add_argument('--preprocess_root', type=str, default='/app/MixDSemi/SynFoCLIP/preprocess')
	parser.add_argument('--llm_model', type=str, default='GPT5', choices=['gemini', 'GPT5', 'DeepSeek'])
	parser.add_argument('--describe_nums', type=int, default=80, choices=[20, 40, 60, 80])
	parser.add_argument('--text_num_subsets', type=int, default=4)
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

	model = VPT_CLIP_RD(
		model_path=args.biomedclip_path,
		num_prompts=args.biomedclip_num_prompts,
		embed_dim=args.biomedclip_embed_dim,
		init_std=args.biomedclip_init_std,
		prompt_scale_init=args.biomedclip_prompt_scale_init,
		enable_scale=not args.biomedclip_disable_scale,
		device=str(device),
	)
	model.load_prompts(args.prompts_path)
	model.eval()

	dataset_key = TEXT_DATASET_DEFAULTS.get(args.dataset)
	if dataset_key is None:
		raise ValueError(f"Unsupported dataset for text anchors: {args.dataset}")
	anchors = prepare_text_anchors(model, args, dataset_key, device)

	processed = 0
	with torch.no_grad():
		for batch in tqdm(test_loader, desc='Inference', total=len(test_loader)):
			images = batch['image'].to(device)
			masks_high = ensure_mask_shape(batch['label'].to(device))

			logits_maps, patch_features = model(images, anchors)
			positive_logits = torch.stack([logits_maps[:, 0, :], logits_maps[:, 1, :]], dim=0).mean(dim=0)
			side = int(math.sqrt(positive_logits.shape[-1]))
			if side * side != positive_logits.shape[-1]:
				raise RuntimeError("Patch sequence length is not a perfect square; cannot reshape to grid.")

			prediction_map = torch.sigmoid(positive_logits).view(images.shape[0], side, side)
			mask_down_nearest = F.interpolate(masks_high, size=(side, side), mode='nearest')
			mask_down_avg = F.adaptive_avg_pool2d(masks_high, (side, side))

			for idx in range(images.shape[0]):
				img_tensor = images[idx]
				mask_high_tensor = masks_high[idx]
				mask_nearest_tensor = mask_down_nearest[idx]
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
					mask_down_nearest=mask_nearest_tensor,
					model_prediction=pred_tensor,
					output_path=output_path,
				)

				processed += 1
				if args.max_samples is not None and processed >= args.max_samples:
					return


if __name__ == '__main__':
	main()

