"""Frequency-domain augmentation utilities (TP-RAM style).

This module centralizes the MiDSS low-frequency mixing helpers so that
training scripts can re-use them without duplicating FFT logic.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _extract_amp_phase(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
	"""Return amplitude and phase of image using FFT.

	Supports grayscale (H, W) or multi-channel (H, W, C) arrays.
	"""

	if image.ndim == 2:
		fft = np.fft.fft2(image)
		return np.abs(fft), np.angle(fft)

	if image.ndim == 3:
		amps, phases = [], []
		for channel in range(image.shape[2]):
			fft = np.fft.fft2(image[:, :, channel])
			amps.append(np.abs(fft))
			phases.append(np.angle(fft))
		return np.stack(amps, axis=2), np.stack(phases, axis=2)

	raise ValueError(f"Unsupported image ndim for FFT: {image.ndim}")


def _low_freq_mutate_np(amp_src: np.ndarray, amp_trg: np.ndarray, L: float = 0.1) -> np.ndarray:
	"""Replace low-frequency region of ``amp_src`` with that from ``amp_trg``."""

	a_src = amp_src.copy()
	h, w = a_src.shape[:2]
	b = max(int(np.amin((h, w)) * L), 1)
	c_h, c_w = h // 2, w // 2

	h1 = max(0, c_h - b)
	h2 = min(h, c_h + b)
	w1 = max(0, c_w - b)
	w2 = min(w, c_w + b)

	if amp_trg.shape[:2] != (h, w):
		tr_h, tr_w = amp_trg.shape[:2]
		sh = max((tr_h - h) // 2, 0)
		sw = max((tr_w - w) // 2, 0)
		amp_trg = amp_trg[sh:sh + h, sw:sw + w]

	a_src[h1:h2, w1:w2] = amp_trg[h1:h2, w1:w2]
	return a_src


def source_to_target_freq(src_img: np.ndarray, tgt_img: np.ndarray, L: float = 0.1) -> np.ndarray:
	"""Mutate ``src_img`` low-frequency amplitude with ``tgt_img`` and return result."""

	src = src_img.astype(np.float32)
	tgt = tgt_img.astype(np.float32)

	if src.ndim == 2:
		amp_s, pha_s = _extract_amp_phase(src)
		amp_t, _ = _extract_amp_phase(tgt)
		amp_mut = _low_freq_mutate_np(amp_s, amp_t, L=L)
		fft_mut = amp_mut * np.exp(1j * pha_s)
		return np.real(np.fft.ifft2(fft_mut))

	if src.ndim == 3:
		out = np.zeros_like(src)
		for channel in range(src.shape[2]):
			amp_s, pha_s = _extract_amp_phase(src[:, :, channel])
			amp_t, _ = _extract_amp_phase(tgt[:, :, channel])
			amp_mut = _low_freq_mutate_np(amp_s, amp_t, L=L)
			out[:, :, channel] = np.real(np.fft.ifft2(amp_mut * np.exp(1j * pha_s)))
		return out

	raise ValueError(f"Unsupported image ndim for source_to_target_freq: {src.ndim}")


def extract_amp_spectrum(img_np: np.ndarray) -> np.ndarray:
	"""Extract amplitude spectrum from image (MiDSS style)."""

	return np.abs(np.fft.fft2(img_np, axes=(-2, -1)))


def low_freq_mutate_np(amp_src: np.ndarray, amp_trg: np.ndarray, L: float = 0.1, degree: float = 1.0) -> np.ndarray:
	"""Mutate low-frequency components of source amplitude with target (MiDSS style)."""

	a_src = np.fft.fftshift(amp_src, axes=(-2, -1))
	a_trg = np.fft.fftshift(amp_trg, axes=(-2, -1))

	_, h, w = a_src.shape
	b = int(np.floor(np.amin((h, w)) * L))
	c_h = int(np.floor(h / 2.0))
	c_w = int(np.floor(w / 2.0))

	h1, h2 = c_h - b, c_h + b + 1
	w1, w2 = c_w - b, c_w + b + 1

	ratio = np.random.uniform(0.0, degree)
	a_src[:, h1:h2, w1:w2] = a_src[:, h1:h2, w1:w2] * (1 - ratio) + a_trg[:, h1:h2, w1:w2] * ratio
	return np.fft.ifftshift(a_src, axes=(-2, -1))


def source_to_target_freq_midss(src_img: np.ndarray, amp_trg: np.ndarray, L: float = 0.1, degree: float = 1.0) -> np.ndarray:
	"""Apply frequency-domain augmentation (MiDSS style)."""

	fft_src = np.fft.fft2(src_img, axes=(-2, -1))
	amp_src = np.abs(fft_src)
	pha_src = np.angle(fft_src)

	amp_mut = low_freq_mutate_np(amp_src, amp_trg, L=L, degree=degree)
	fft_mut = amp_mut * np.exp(1j * pha_src)
	return np.real(np.fft.ifft2(fft_mut, axes=(-2, -1)))
