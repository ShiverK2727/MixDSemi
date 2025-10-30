"""Validation utilities decoupled from the training loop."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch
from medpy.metric import binary
from scipy.ndimage import zoom
from torch.utils.data import DataLoader

from utils.label_ops import to_2d, to_3d


@dataclass
class ValidationConfig:
    dataset: str
    parts: Sequence[str]
    patch_size: int
    dice_fn: Callable[[np.ndarray, torch.Tensor], Sequence[float]]
    writer_prefix: str = "unet_val"


@torch.no_grad()
def run_validation(
    model: torch.nn.Module,
    dataloaders: Sequence[DataLoader],
    iteration: int,
    writer,
    config: ValidationConfig,
) -> list[float]:
    """Evaluate model across domains and log metrics to TensorBoard."""

    model.eval()
    num_parts = len(config.parts)
    domain_num = len(dataloaders)

    val_dice = [0.0] * num_parts
    val_dc = [0.0] * num_parts
    val_jc = [0.0] * num_parts
    val_hd = [0.0] * num_parts
    val_asd = [0.0] * num_parts

    logging.info("Starting evaluation over %d domains...", domain_num)

    for domain_idx, cur_dataloader in enumerate(dataloaders, start=1):
        domain_val_dice = [0.0] * num_parts
        domain_val_dc = [0.0] * num_parts
        domain_val_jc = [0.0] * num_parts
        domain_val_hd = [0.0] * num_parts
        domain_val_asd = [0.0] * num_parts

        for sample in cur_dataloader:
            assert domain_idx == sample["dc"][0].item()

            mask = sample["label"]
            if config.dataset == "fundus":
                fundus_mask = (mask <= 128) * 2
                fundus_mask[mask == 0] = 1
                mask = fundus_mask
            elif config.dataset == "prostate":
                mask = mask.eq(0).long()
            elif config.dataset == "MNMS":
                mask = mask.long()
            elif config.dataset == "BUSI":
                mask = mask.eq(255).long()

            data = sample["unet_size_img"].cuda()
            output = model(data)
            pred_label = torch.max(torch.softmax(output, dim=1), dim=1)[1]
            scale_factor = config.patch_size / data.shape[-2]
            pred_label = torch.from_numpy(
                zoom(pred_label.cpu(), (1, scale_factor, scale_factor), order=0)
            )

            if config.dataset == "fundus":
                pred_label = to_2d(pred_label)
                mask = to_2d(mask)
                pred_onehot = pred_label.clone()
                mask_onehot = mask.clone()
            elif config.dataset in {"prostate", "BUSI"}:
                pred_onehot = pred_label.clone().unsqueeze(1)
                mask_onehot = mask.clone().unsqueeze(1)
            elif config.dataset == "MNMS":
                pred_onehot = to_3d(pred_label)
                mask_onehot = to_3d(mask)

            dice = config.dice_fn(np.asarray(pred_label.cpu()), mask.cpu())

            dc = [0.0] * num_parts
            jc = [0.0] * num_parts
            hd = [0.0] * num_parts
            asd = [0.0] * num_parts

            batch_size = len(data)
            for sample_idx in range(batch_size):
                for part_idx, _ in enumerate(config.parts):
                    dc[part_idx] += binary.dc(
                        np.asarray(pred_onehot[sample_idx, part_idx], dtype=bool),
                        np.asarray(mask_onehot[sample_idx, part_idx], dtype=bool),
                    )
                    jc[part_idx] += binary.jc(
                        np.asarray(pred_onehot[sample_idx, part_idx], dtype=bool),
                        np.asarray(mask_onehot[sample_idx, part_idx], dtype=bool),
                    )
                    if pred_onehot[sample_idx, part_idx].float().sum() < 1e-4:
                        hd[part_idx] += 100
                        asd[part_idx] += 100
                    else:
                        hd[part_idx] += binary.hd95(
                            np.asarray(pred_onehot[sample_idx, part_idx], dtype=bool),
                            np.asarray(mask_onehot[sample_idx, part_idx], dtype=bool),
                        )
                        asd[part_idx] += binary.asd(
                            np.asarray(pred_onehot[sample_idx, part_idx], dtype=bool),
                            np.asarray(mask_onehot[sample_idx, part_idx], dtype=bool),
                        )

            for part_idx in range(num_parts):
                dc[part_idx] /= batch_size
                jc[part_idx] /= batch_size
                hd[part_idx] /= batch_size
                asd[part_idx] /= batch_size

                domain_val_dice[part_idx] += dice[part_idx]
                domain_val_dc[part_idx] += dc[part_idx]
                domain_val_jc[part_idx] += jc[part_idx]
                domain_val_hd[part_idx] += hd[part_idx]
                domain_val_asd[part_idx] += asd[part_idx]

        num_batches = len(cur_dataloader)
        for part_idx in range(num_parts):
            domain_val_dice[part_idx] /= num_batches
            val_dice[part_idx] += domain_val_dice[part_idx]
            domain_val_dc[part_idx] /= num_batches
            val_dc[part_idx] += domain_val_dc[part_idx]
            domain_val_jc[part_idx] /= num_batches
            val_jc[part_idx] += domain_val_jc[part_idx]
            domain_val_hd[part_idx] /= num_batches
            val_hd[part_idx] += domain_val_hd[part_idx]
            domain_val_asd[part_idx] /= num_batches
            val_asd[part_idx] += domain_val_asd[part_idx]

            writer.add_scalar(
                f"{config.writer_prefix}/domain{domain_idx}/val_{config.parts[part_idx]}_dice",
                domain_val_dice[part_idx],
                iteration,
            )

        domain_summary: list[str] = []
        domain_summary.extend(
            f"val_{config.parts[idx]}_dice: {domain_val_dice[idx]:.6f}" for idx in range(num_parts)
        )
        domain_summary.extend(
            f"val_{config.parts[idx]}_dc: {domain_val_dc[idx]:.6f}" for idx in range(num_parts)
        )
        domain_summary.extend(
            f"val_{config.parts[idx]}_jc: {domain_val_jc[idx]:.6f}" for idx in range(num_parts)
        )
        domain_summary.extend(
            f"val_{config.parts[idx]}_hd: {domain_val_hd[idx]:.6f}" for idx in range(num_parts)
        )
        domain_summary.extend(
            f"val_{config.parts[idx]}_asd: {domain_val_asd[idx]:.6f}" for idx in range(num_parts)
        )
        logging.info("domain%d iter %d :\n\t%s", domain_idx, iteration, "\n\t".join(domain_summary))

    model.train()

    for part_idx in range(num_parts):
        val_dice[part_idx] /= domain_num
        val_dc[part_idx] /= domain_num
        val_jc[part_idx] /= domain_num
        val_hd[part_idx] /= domain_num
        val_asd[part_idx] /= domain_num

        writer.add_scalar(
            f"{config.writer_prefix}/val_{config.parts[part_idx]}_dice",
            val_dice[part_idx],
            iteration,
        )

    summary_lines: list[str] = []
    summary_lines.extend(
        f"val_{config.parts[idx]}_dice: {val_dice[idx]:.6f}" for idx in range(num_parts)
    )
    summary_lines.extend(
        f"val_{config.parts[idx]}_dc: {val_dc[idx]:.6f}" for idx in range(num_parts)
    )
    summary_lines.extend(
        f"val_{config.parts[idx]}_jc: {val_jc[idx]:.6f}" for idx in range(num_parts)
    )
    summary_lines.extend(
        f"val_{config.parts[idx]}_hd: {val_hd[idx]:.6f}" for idx in range(num_parts)
    )
    summary_lines.extend(
        f"val_{config.parts[idx]}_asd: {val_asd[idx]:.6f}" for idx in range(num_parts)
    )
    logging.info("iteration %d :\n\t%s", iteration, "\n\t".join(summary_lines))

    return val_dice