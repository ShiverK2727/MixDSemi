"""Utility helpers for BiomedCLIP teacher forward and loss computation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class ClipTeacherOutputs:
    """Container for CLIP teacher embeddings."""

    v_inv_w: torch.Tensor
    v_spec_w: torch.Tensor
    v_inv_s: torch.Tensor


@dataclass
class ClipLossComponents:
    """Individual CLIP loss components for readability and configurability."""

    mv_anchor: torch.Tensor
    ortho: torch.Tensor
    sw_reg: torch.Tensor


@dataclass
class ClipTeacherTargets:
    """Detached teacher embeddings split into labeled and unlabeled subsets."""

    v_inv_lb: torch.Tensor
    v_spec_lb: torch.Tensor
    v_inv_ulb: torch.Tensor
    v_spec_ulb: torch.Tensor


def run_clip_teacher_step(
    biomedclip_model,
    preprocess,
    weak_images: torch.Tensor,
    strong_images_ulb: torch.Tensor,
) -> ClipTeacherOutputs:
    """Run BiomedCLIP forward pass for weak and strong views."""

    dual_outputs_w = biomedclip_model.forward_dual_from_tensor(
        images=weak_images,
        preprocess=preprocess,
    )
    v_inv_w = dual_outputs_w["invariant"]["image_features"]
    v_spec_w = dual_outputs_w["specific"]["image_features"]

    outputs_s = biomedclip_model.encode_image_from_tensor(
        images=strong_images_ulb,
        preprocess=preprocess,
        prompt_group="invariant",
    )
    v_inv_s = outputs_s["image_features"]

    return ClipTeacherOutputs(v_inv_w=v_inv_w, v_spec_w=v_spec_w, v_inv_s=v_inv_s)


def compute_clip_loss_components(
    outputs: ClipTeacherOutputs,
    anchors: torch.Tensor,
    global_anchor: torch.Tensor,
    label_batch_size: int,
) -> ClipLossComponents:
    """Compute grouped CLIP losses: (mv+anchor), ortho, sw_reg."""

    sim_matrix = torch.matmul(outputs.v_inv_w, anchors.T)
    loss_mv = torch.var(sim_matrix, dim=1).mean()

    sim_global = torch.matmul(outputs.v_inv_w, global_anchor.T).squeeze(-1)
    loss_anchor = (1 - sim_global).mean()

    loss_ortho = torch.sum(outputs.v_inv_w * outputs.v_spec_w, dim=1).pow(2).mean()

    v_inv_ulb_w = outputs.v_inv_w[label_batch_size:]
    loss_sw_reg = (1 - torch.sum(outputs.v_inv_s * v_inv_ulb_w.detach(), dim=1)).mean()

    return ClipLossComponents(
        mv_anchor=loss_mv + loss_anchor,
        ortho=loss_ortho,
        sw_reg=loss_sw_reg,
    )


def split_teacher_outputs(outputs: ClipTeacherOutputs, label_batch_size: int) -> ClipTeacherTargets:
    """Detach teacher embeddings for distillation."""

    v_inv_w_detached = outputs.v_inv_w.detach()
    v_spec_w_detached = outputs.v_spec_w.detach()

    return ClipTeacherTargets(
        v_inv_lb=v_inv_w_detached[:label_batch_size],
        v_spec_lb=v_spec_w_detached[:label_batch_size],
        v_inv_ulb=v_inv_w_detached[label_batch_size:],
        v_spec_ulb=v_spec_w_detached[label_batch_size:],
    )
