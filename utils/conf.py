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


def _kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """計算 KL 散度 KL(p || q) per sample."""
    # p, q shape: [B, C, H, W]
    # 確保 p, q 總和為 1 (如果輸入是 logits 需要先 softmax)
    kl_pixel = (p * (p.clamp_min(eps).log() - q.clamp_min(eps).log())).sum(dim=1) # [B, H, W]
    # 在空間維度上取平均得到每個樣本的 KL
    kl_sample = kl_pixel.mean(dim=(1, 2)) # [B]
    return kl_sample

def _js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """計算 Jensen-Shannon 散度 JS(p || q) per sample."""
    # 計算平均分佈 M
    m = 0.5 * (p + q)
    # 計算 JS 散度
    js = 0.5 * _kl_divergence(p, m, eps) + 0.5 * _kl_divergence(q, m, eps) # [B]
    return js

@register_conf("js_teacher_student") # 註冊新策略
def js_teacher_student_confidence(
    args,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    dataset_name: str,  # 添加以匹配签名
    dice_calcu: Dict[str, Callable],  # 添加以匹配签名
    to_2d_fn: Optional[Callable],  # 添加以匹配签名
    **_,
) -> torch.Tensor:
    """
    基於教師-學生 Softmax 輸出的 JS 散度計算置信度分數。
    Confidence = 1 - JS / ln(2)
    """
    # 1. 獲取教師和學生的 Softmax 概率，應用溫度
    #    注意：同時對學生應用溫度是可選的，但可以使比較更“公平”
    temperature = getattr(args, "conf_teacher_temp", 1.0) # 沿用之前的溫度參數
    student_prob = torch.softmax(student_logits / temperature, dim=1)
    teacher_prob = torch.softmax(teacher_logits / temperature, dim=1)

    # 2. 計算 JS 散度
    js_div = _js_divergence(teacher_prob, student_prob) # shape [B]

    # 3. 將 JS 散度 (範圍 [0, ln2]) 轉換為置信度 (範圍 [0, 1])
    #    JS = 0 (完全一致) => Confidence = 1
    #    JS = ln2 (最大差異) => Confidence = 0
    max_js = float(np.log(2.0))
    # 使用 clamp_max 防止因數值誤差導致 js > ln(2)
    confidence = 1.0 - js_div.clamp_max(max_js) / max_js # shape [B]

    return confidence

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


@register_conf("robust_no_fg")
def robust_no_fg_confidence(
    args,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    dataset_name: str,
    dice_calcu: Dict[str, Callable],
    to_2d_fn: Optional[Callable],
    **_,
) -> torch.Tensor:
    """Dice * (1 - entropy) used for confidence-aware ramps (without fg_ratio)."""

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

    entropy = _normalized_entropy(teacher_prob)

    robust_no_fg = dice_scores * (1.0 - entropy)
    return robust_no_fg





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

