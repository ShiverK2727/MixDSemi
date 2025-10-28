import logging
import math
import time
from typing import Iterator, List, Sequence, Tuple

import torch
from torch.utils.data import Sampler


def _stack_scores(score_list: Sequence[torch.Tensor]) -> torch.Tensor:
    """将任意形状的得分张量展平并堆叠成二维矩阵。"""
    flattened = []
    for score in score_list:
        tensor = torch.as_tensor(score, dtype=torch.float32).flatten()
        flattened.append(tensor)
    if not flattened:
        return torch.zeros((0, 1), dtype=torch.float32)
    return torch.stack(flattened, dim=0)


def compute_prototype(score_list: Sequence[torch.Tensor]) -> torch.Tensor:
    """求有标注样本得分的原型向量。"""
    stacked = _stack_scores(score_list)
    if stacked.numel() == 0:
        raise ValueError("No scores provided to compute prototype.")
    return stacked.mean(dim=0)


def compute_l2_distances(score_list: Sequence[torch.Tensor], prototype: torch.Tensor) -> torch.Tensor:
    """计算每个样本得分与原型之间的 L2 距离。"""
    stacked = _stack_scores(score_list)
    if stacked.numel() == 0:
        return torch.zeros(0)
    if prototype.dim() != 1:
        prototype = prototype.flatten()
    return torch.norm(stacked - prototype.unsqueeze(0), dim=1)


def partition_by_distance(sorted_indices: Sequence[int], num_partitions: int) -> List[List[int]]:
    """按均分策略将排序后索引切分成指定份数。"""
    if num_partitions <= 0:
        raise ValueError("num_partitions must be positive")
    if not sorted_indices:
        return [[] for _ in range(num_partitions)]
    partition_size = math.ceil(len(sorted_indices) / num_partitions)
    partitions: List[List[int]] = []
    for start in range(0, len(sorted_indices), partition_size):
        partitions.append(list(sorted_indices[start:start + partition_size]))
    # 如果不满 K 份，填充空列表
    while len(partitions) < num_partitions:
        partitions.append([])
    return partitions


def build_distance_curriculum(
    labeled_scores: Sequence[torch.Tensor],
    unlabeled_scores: Sequence[torch.Tensor],
    unlabeled_indices: Sequence[int],
    num_partitions: int,
    distance_mode: str = "prototype",
) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[List[int]]]:
    """根据得分构建距离排序和分段结果。

    distance_mode controls how distances are computed for unlabeled samples:
      - "prototype" (default): distance = L2(unlabeled_score, prototype_of_labeled)
      - "sqrt_prod": distance = sqrt( delta_L * delta_U ), where
            delta_L = L2(unlabeled_score, prototype_of_labeled)
            delta_U = L2(unlabeled_score, prototype_of_unlabeled)

    Returns: (prototype_labeled, distances, sorted_indices, partitions)
    """
    prototype = compute_prototype(labeled_scores)

    if distance_mode == "prototype":
        distances = compute_l2_distances(unlabeled_scores, prototype)
    elif distance_mode == "sqrt_prod":
        # compute prototype of unlabeled scores (center of unlabeled distribution)
        unlabeled_proto = compute_prototype(unlabeled_scores) if len(unlabeled_scores) > 0 else None
        # delta_L: distance to labeled prototype
        delta_L = compute_l2_distances(unlabeled_scores, prototype)
        # delta_U: distance to unlabeled prototype (if available)
        if unlabeled_proto is None:
            # fallback to delta_L if unlabeled proto cannot be computed
            delta_U = delta_L.clone()
        else:
            delta_U = compute_l2_distances(unlabeled_scores, unlabeled_proto)

        # element-wise sqrt(prod). ensure non-negative and same shape
        prod = delta_L * delta_U
        # numerical safety: clamp to >= 0
        prod = torch.clamp(prod, min=0.0)
        distances = torch.sqrt(prod)
    else:
        raise ValueError(f"Unknown distance_mode: {distance_mode}")

    if len(unlabeled_indices) != len(distances):
        raise ValueError("Length mismatch between indices and distances")

    # 根据距离排序索引 (ascending)
    sorted_pairs = sorted(zip(unlabeled_indices, distances.tolist()), key=lambda x: x[1])
    sorted_indices = [idx for idx, _ in sorted_pairs]
    partitions = partition_by_distance(sorted_indices, num_partitions)
    return prototype, distances, sorted_indices, partitions


class DomainDistanceCurriculumSampler(Sampler[int]):
    """根据距离分组逐步扩充训练样本的采样器。"""

    def __init__(
        self,
        partitions: Sequence[Sequence[int]],
        initial_stage: int = 1,
        seed: int = 42,
        shuffle: bool = True,
    ) -> None:
        if not partitions:
            raise ValueError("partitions must not be empty")
        self.partitions = [list(part) for part in partitions]
        self.total_indices = sum(len(part) for part in self.partitions)
        self.shuffle = shuffle
        self.seed = seed
        # epoch used to vary the RNG seed in a reproducible way
        self._epoch = 0
        self.stage = 0
        self._active_indices: List[int] = []
        self.set_stage(initial_stage)

    def set_epoch(self, epoch: int) -> None:
        """Set epoch index used to derive RNG seed for shuffling.

        This allows reproducible but epoch-varying permutations by seeding the
        RNG with self.seed + epoch. Call this from the training loop at the
        start of each epoch/cycle.
        """
        try:
            self._epoch = int(epoch)
        except Exception:
            self._epoch = 0

    def set_stage(self, stage: int) -> None:
        stage = max(1, min(stage, len(self.partitions)))
        if stage == self.stage and self._active_indices:
            return
        self.stage = stage
        active = []
        for part in self.partitions[:stage]:
            active.extend(part)
        self._active_indices = active
        logging.info(
            "DistanceCurriculumSampler stage=%d active=%d/%d",
            self.stage,
            len(self._active_indices),
            self.total_indices,
        )

    def expand(self, additional_stages: int = 1) -> None:
        self.set_stage(self.stage + additional_stages)

    def get_active_indices(self) -> List[int]:
        """Return a copy of currently active indices."""
        return list(self._active_indices)

    def __iter__(self) -> Iterator[int]:
        if not self._active_indices:
            return iter([])
        if self.shuffle:
            generator = torch.Generator()
            # Use seed + epoch for reproducible, epoch-varying permutations.
            generator.manual_seed(self.seed + int(getattr(self, '_epoch', 0)))
            order = torch.randperm(len(self._active_indices), generator=generator)
            shuffled = [self._active_indices[i] for i in order]
            return iter(shuffled)
        return iter(self._active_indices)

    def __len__(self) -> int:
        return len(self._active_indices)
    


