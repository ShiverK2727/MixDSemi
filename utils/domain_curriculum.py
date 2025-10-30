import logging
import math
import time
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

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
        score_lookup: Optional[Union[Sequence[torch.Tensor], Dict[int, torch.Tensor]]] = None,
        stratified_batch_size: Optional[int] = None,
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
        self._score_lookup = self._normalize_scores(score_lookup)
        self.stratified_batch_size = int(stratified_batch_size) if stratified_batch_size and stratified_batch_size > 0 else 0
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
        generator = None
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + int(getattr(self, '_epoch', 0)))

        if self.stratified_batch_size > 1 and self._score_lookup is not None:
            ordered = self._build_stratified_order(self._active_indices, generator)
        elif self.shuffle:
            ordered = self._permute_indices(self._active_indices, generator)
        else:
            ordered = list(self._active_indices)
        return iter(ordered)

    def __len__(self) -> int:
        return len(self._active_indices)

    # ------------------------------------------------------------------
    def _normalize_scores(
        self,
        score_lookup: Optional[Union[Sequence[torch.Tensor], Dict[int, torch.Tensor]]],
    ) -> Optional[Union[List[Optional[float]], Dict[int, Optional[float]]]]:
        if score_lookup is None:
            return None
        if isinstance(score_lookup, dict):
            normalized: Dict[int, Optional[float]] = {}
            for key, value in score_lookup.items():
                normalized[key] = self._score_to_scalar(value)
            return normalized
        normalized_list: List[Optional[float]] = []
        for value in score_lookup:
            normalized_list.append(self._score_to_scalar(value))
        return normalized_list

    @staticmethod
    def _score_to_scalar(score: Optional[Union[torch.Tensor, Sequence[float], float]]) -> Optional[float]:
        if score is None:
            return None
        try:
            tensor = torch.as_tensor(score, dtype=torch.float32)
        except Exception:
            return None
        if tensor.numel() == 0:
            return None
        return float(tensor.float().mean().item())

    def _get_score_value(self, index: int) -> Optional[float]:
        if self._score_lookup is None:
            return None
        if isinstance(self._score_lookup, dict):
            return self._score_lookup.get(index)
        if 0 <= index < len(self._score_lookup):
            return self._score_lookup[index]
        return None

    def _permute_indices(self, indices: Sequence[int], generator: Optional[torch.Generator]) -> List[int]:
        if generator is None or len(indices) <= 1:
            return list(indices)
        order = torch.randperm(len(indices), generator=generator).tolist()
        return [indices[i] for i in order]

    def _build_stratified_order(self, indices: Sequence[int], generator: Optional[torch.Generator]) -> List[int]:
        if not indices:
            return []

        scored: List[tuple[int, float]] = []
        missing: List[int] = []
        for idx in indices:
            value = self._get_score_value(idx)
            if value is None:
                missing.append(idx)
            else:
                scored.append((idx, value))

        if not scored:
            # fallback to random order if scores unavailable
            return self._permute_indices(indices, generator)

        scored.sort(key=lambda x: x[1])
        num_bins = min(self.stratified_batch_size, len(scored))
        if num_bins <= 1:
            ordered = [idx for idx, _ in scored]
        else:
            bins: List[List[int]] = [[] for _ in range(num_bins)]
            total = len(scored)
            for rank, (idx, _) in enumerate(scored):
                bin_id = min(num_bins - 1, int(rank * num_bins / total))
                bins[bin_id].append(idx)

            if generator is not None:
                for i in range(num_bins):
                    if len(bins[i]) > 1:
                        bins[i] = self._permute_indices(bins[i], generator)

            max_bin_len = max(len(bin_list) for bin_list in bins)
            ordered = []
            for offset in range(max_bin_len):
                for bin_id in range(num_bins):
                    bin_list = bins[bin_id]
                    if offset < len(bin_list):
                        ordered.append(bin_list[offset])

        if missing:
            tail = self._permute_indices(missing, generator) if generator is not None else missing
            ordered.extend(tail)
        return ordered
    


