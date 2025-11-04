#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 文件名：MiDSS_CLIP_PATCH_test1_fixed.py
# 描述：
# 1. 加载BiomedCLIP和LLM生成的文本及关键词测试组。
# 2. 为每个文本组创建语义锚点（平均文本特征）。
# 3. 遍历所有图像，提取它们的"Patch特征"（而非[CLS]全局特征）。
# 4. 计算每个Patch特征与每个文本锚点的余弦相似度。
# 5. 将相似度分数重塑为14x14的热力图，并上采样到原图大小，保存为ROI。

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

from batchgenerators.utilities.file_and_folder_operations import *

# --- 配置 ---
IMAGE_ROOT = "/app/MixDSemi/data/mnms/vendorC/train/image"
all_images = subfiles(IMAGE_ROOT, suffix=".png", join=True)
all_masks = [i.replace("/image/", "/mask/") for i in all_images]


def visualize_and_save_masks(images, masks, output_dir, max_samples=None):
	"""为每个样本保存一张拼接图：原图 | 原始 mask | mask 下采样到 14x14 后按 nearest 放大以可视化格子效果。

	输出文件命名为 <原文件名>_viz.png，保存在 output_dir 中。
	"""
	os.makedirs(output_dir, exist_ok=True)
	cnt = 0
	for img_path, mask_path in zip(images, masks):
		if max_samples is not None and cnt >= max_samples:
			break
		if not os.path.exists(img_path):
			continue
		if not os.path.exists(mask_path):
			# 如果没有对应 mask，跳过
			continue

		img = cv2.imread(img_path, cv2.IMREAD_COLOR)
		mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
		if img is None or mask is None:
			continue

		# 先把原图和 mask 都调整到 224x224（图像用双线性，mask 用最近邻以保留标签）
		target_size = (224, 224)
		img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
		mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
		h, w = target_size[1], target_size[0]

		# 下采样到 14x14：改用平均池化（平均下采样）替代最近邻
		# 使用 PyTorch 的 adaptive avg pool 保证输出精确为 14x14
		# mask 为单通道 uint8，先转为 tensor，再池化，最后四舍五入为整数以便可视化
		mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
		small_tensor = F.adaptive_avg_pool2d(mask_tensor, (14, 14))
		small = small_tensor.squeeze().cpu().numpy()
		# 将平均值四舍五入回整数标签，方便后续用最近邻上采样可视化格子效果
		small_rounded = np.rint(small).astype(np.uint8)
		# 为了直观显示，将 14x14 最近邻上采样回 224x224（保持方块效果）
		small_up = cv2.resize(small_rounded, target_size, interpolation=cv2.INTER_NEAREST)

		# 把单通道 mask 转为 BGR，便于拼接显示
		mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
		small_color = cv2.cvtColor(small_up, cv2.COLOR_GRAY2BGR)

		# 横向拼接：原图 | 原始 mask | 下采样可视化
		try:
			concat = np.hstack([img, mask_color, small_color])
		except Exception:
			# 保险处理：如果尺寸不匹配，先调整到相同高度
			target_h = h
			def ensure_bgr(im):
				if im is None:
					return np.zeros((target_h, w, 3), dtype=np.uint8)
				if im.ndim == 2:
					return cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
				return im
			img2 = ensure_bgr(img)
			mask_color2 = ensure_bgr(mask)
			small_color2 = ensure_bgr(small_up)
			concat = np.hstack([img2, mask_color2, small_color2])

		# 添加简单标签（可选），坐标基于图像宽度
		try:
			cv2.putText(concat, "Image", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
			cv2.putText(concat, "Mask", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
			cv2.putText(concat, "Mask 14x14", (2 * w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
		except Exception:
			pass

		base = os.path.splitext(os.path.basename(img_path))[0]
		out_path = os.path.join(output_dir, f"{base}_viz.png")
		cv2.imwrite(out_path, concat)
		cnt += 1


if __name__ == "__main__":
	# 输出目录放在脚本同级目录下的 test_visuals
	script_dir = os.path.dirname(os.path.abspath(__file__))
	outdir = os.path.join(script_dir, "test_visuals2_mnms")
	# 为了快速验证，默认处理前 20 个样本；如果想处理全部，把 max_samples 设置为 None
	visualize_and_save_masks(all_images, all_masks, outdir, max_samples=20)