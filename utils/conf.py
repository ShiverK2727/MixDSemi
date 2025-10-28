"""Confidence utility helpers used by training scripts."""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import torch


ConfFn = Callable[
    [object, torch.Tensor, torch.Tensor, str, Dict[str, Callable], Optional[Callable]],
    torch.Tensor,
]


CONF_REGISTRY: Dict[str, ConfFn] = {}


def register_conf(name: str) -> Callable[[ConfFn], ConfFn]:
    """Decorator to register a confidence aggregation strategy."""

    def decorator(fn: ConfFn) -> ConfFn:
        CONF_REGISTRY[name] = fn
        return fn

    return decorator


def available_conf_strategies() -> List[str]:
    """Return a list of registered confidence strategy names."""

    return sorted(CONF_REGISTRY.keys())


def _mean_from_metric_output(metric_output) -> np.ndarray:
    """Convert various metric return shapes into per-sample means."""

    if isinstance(metric_output, list):
        arrays = [np.asarray(arr) for arr in metric_output]
        stacked = np.stack(arrays, axis=-1)
        return stacked.mean(axis=-1)

    arr = np.asarray(metric_output)
    if arr.ndim > 1:
        return arr.mean(axis=tuple(range(1, arr.ndim)))
    return arr


def _normalized_entropy(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Return per-sample normalized entropy of probability maps."""

    _, C, _, _ = probs.shape
    log_c = float(np.log(C))
    entropy = -(probs * probs.clamp_min(eps).log()).sum(dim=1)  # [B, H, W]
    entropy = entropy.mean(dim=(1, 2)) / log_c
    return entropy


def _fg_ratio_from_argmax(argmax_label: torch.Tensor) -> torch.Tensor:
    """Compute foreground pixel ratio for each sample given argmax labels."""

    return (argmax_label != 0).float().mean(dim=(1, 2))


@register_conf("dice")
def dice_confidence(
    args,  # noqa: D417 - pass-through for extensibility
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    dataset_name: str,
    dice_calcu: Dict[str, Callable],
    to_2d_fn: Optional[Callable],
    **_,
) -> torch.Tensor:
    """Baseline Dice-based consistency used in MiDSS."""

    student_label = torch.softmax(student_logits, dim=1).argmax(dim=1)
    teacher_label = torch.softmax(teacher_logits, dim=1).argmax(dim=1)

    if dataset_name == "fundus" and to_2d_fn is not None:
        student_metric = to_2d_fn(student_label).cpu()
        teacher_metric = to_2d_fn(teacher_label).cpu()
    else:
        student_metric = student_label.detach().cpu()
        teacher_metric = teacher_label.detach().cpu()

    dice_values = dice_calcu[dataset_name](
        np.asarray(student_metric),
        teacher_metric,
        ret_arr=True,
    )
    per_sample = _mean_from_metric_output(dice_values)
    return torch.as_tensor(per_sample, device=student_logits.device, dtype=torch.float32)


@register_conf("robust")
def robust_confidence(
    args,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    dataset_name: str,
    dice_calcu: Dict[str, Callable],
    to_2d_fn: Optional[Callable],
    **_,
) -> torch.Tensor:
    """Dice * (1 - entropy) * sqrt(fg_ratio) used for confidence-aware ramps."""

    dice_scores = dice_confidence(
        args,
        student_logits,
        teacher_logits,
        dataset_name,
        dice_calcu,
        to_2d_fn,
    )

    temperature = getattr(args, "conf_teacher_temp", 1.0)
    teacher_prob = torch.softmax(teacher_logits / temperature, dim=1)
    teacher_label = teacher_prob.argmax(dim=1)

    entropy = _normalized_entropy(teacher_prob)
    fg_ratio = _fg_ratio_from_argmax(teacher_label)

    robust = dice_scores * (1.0 - entropy) * torch.sqrt(fg_ratio.clamp_min(1e-6))
    return robust


def compute_self_consistency(
    args,
    student_model: torch.nn.Module,
    ema_model: torch.nn.Module,
    dataloader,
    dataset_name: str,
    dice_calcu: Dict[str, Callable],
    to_2d_fn: Optional[Callable],
) -> Optional[float]:
    """Compute EMA/student self-consistency metric over ``dataloader``."""

    if dataloader is None:
        return None

    strategy = getattr(args, "conf_strategy", "dice")
    conf_fn = CONF_REGISTRY.get(strategy)
    if conf_fn is None:
        raise ValueError(f"Unknown confidence strategy: {strategy}")

    student_model.eval()
    ema_model.eval()
    values: List[float] = []

    with torch.no_grad():
        for sample in dataloader:
            data = sample["unet_size_img"].cuda(non_blocking=True)
            teacher_logits = ema_model(data)
            student_logits = student_model(data)

            conf_scores = conf_fn(
                args,
                student_logits,
                teacher_logits,
                dataset_name,
                dice_calcu,
                to_2d_fn,
            )
            values.append(conf_scores.mean().item())

    student_model.train()
    ema_model.train()

    if not values:
        return None
    return float(np.mean(values))

