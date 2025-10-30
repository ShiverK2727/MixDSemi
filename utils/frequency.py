"""Frequency-domain augmentation helpers."""

from __future__ import annotations

import numpy as np
import torch

from .tp_ram import extract_amp_spectrum, source_to_target_freq_midss


def apply_midss_frequency_augmentation(
    labeled_images: torch.Tensor,
    unlabeled_images: torch.Tensor,
    degree: float,
    low_band: float,
) -> torch.Tensor:
    """Apply MiDSS-style frequency augmentation to labeled images."""

    labeled_cpu = labeled_images.detach().cpu()
    unlabeled_cpu = unlabeled_images.detach().cpu()

    augmented = []
    for lb_tensor, ulb_tensor in zip(labeled_cpu, unlabeled_cpu):
        lb_img = ((lb_tensor + 1) * 127.5).numpy()
        ulb_img = ((ulb_tensor + 1) * 127.5).numpy()

        amp_trg = extract_amp_spectrum(ulb_img)
        freq_img = source_to_target_freq_midss(lb_img, amp_trg, L=low_band, degree=degree)
        freq_img = np.clip(freq_img, 0, 255).astype(np.float32)
        augmented.append(freq_img)

    augmented_np = np.stack(augmented)
    augmented_tensor = torch.from_numpy(augmented_np) / 127.5 - 1
    return augmented_tensor.to(labeled_images.device)
