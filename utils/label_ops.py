"""Label conversion helpers shared across training scripts."""

from __future__ import annotations

import torch


def to_2d(input_tensor: torch.Tensor) -> torch.Tensor:
    """Convert 1-channel label map to two-channel one-hot (background/foreground)."""

    input_tensor = input_tensor.unsqueeze(1)
    tensor_list = []
    temp_prob = input_tensor == torch.ones_like(input_tensor)
    tensor_list.append(temp_prob)
    temp_prob2 = input_tensor > torch.zeros_like(input_tensor)
    tensor_list.append(temp_prob2)
    output_tensor = torch.cat(tensor_list, dim=1)
    return output_tensor.float()


def to_3d(input_tensor: torch.Tensor) -> torch.Tensor:
    """Convert single-channel labels to three-class one-hot encoding."""

    input_tensor = input_tensor.unsqueeze(1)
    tensor_list = []
    for i in range(1, 4):
        temp_prob = input_tensor == i * torch.ones_like(input_tensor)
        tensor_list.append(temp_prob)
    output_tensor = torch.cat(tensor_list, dim=1)
    return output_tensor.float()
