"""Common training-time utility helpers."""

from __future__ import annotations

import random
from typing import Iterable, Iterator, TypeVar

import numpy as np
import torch

T = TypeVar("T")


def cycle(iterable: Iterable[T]) -> Iterator[T]:
    """Yield from ``iterable`` forever without storing the entire stream."""

    while True:
        for item in iterable:
            yield item


def obtain_cutmix_box(
    img_size: int,
    p: float = 0.5,
    size_min: float = 0.02,
    size_max: float = 0.4,
    ratio_1: float = 0.3,
    ratio_2: float = 1 / 0.3,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Create a binary CutMix mask with random rectangle."""

    mask = torch.zeros(img_size, img_size, device=device)

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

    mask[y : y + cutmix_h, x : x + cutmix_w] = 1
    return mask


class Statistics:
    """Track running mean of scalar values."""

    def __init__(self) -> None:
        self._values: list[float] = []

    @property
    def avg(self) -> float:
        if not self._values:
            return 0.0
        return float(sum(self._values) / len(self._values))

    def update(self, value: float) -> None:
        self._values.append(float(value))
