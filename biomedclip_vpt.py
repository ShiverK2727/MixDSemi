"""BiomedCLIP 双域可选 VPT 实现（聚焦图像编码器）

本实现基于单个 BiomedCLIP 图像编码器，在 patch embedding 之后
一次性注入两组可学习的 prompt tokens（域不变 / 域特定），并允许
在推理 / 训练时按需选择其中一组进行前向传播。

特性：
- 仅在 patch embedding 之后插入 prompts，其余 ViT 结构不变
- 保持单个冻结的 BiomedCLIP 模型，节省显存
- 域不变 / 域特定两个 prompt 组可分别训练
- 前向函数仅输出图像侧特征，文本编码由调用方自行控制
- 支持 prompt 缩放开关、外部注入配置、以及权重保存 / 加载
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS
import torchvision.transforms as T


@dataclass
class DualPromptConfig:
    """双域 VPT 基础配置."""

    embed_dim: int = 768
    num_prompts: int = 4
    init_std: float = 0.02
    prompt_scale_init: float = 1.0
    enable_prompt_scale: bool = True


class DualPromptLearner(nn.Module):
    """管理域不变 / 域特定两组 prompts 的模块."""

    def __init__(self, config: DualPromptConfig):
        super().__init__()
        self.config = config

        # 域不变、域特定两组 prompts，尺寸均为 [num_prompts, embed_dim]
        self.domain_invariant_prompts = nn.Parameter(
            torch.randn(config.num_prompts, config.embed_dim) * config.init_std
        )
        self.domain_specific_prompts = nn.Parameter(
            torch.randn(config.num_prompts, config.embed_dim) * config.init_std
        )

        # 全局缩放因子，方便调节 prompt 注入强度
        self.prompt_scale = nn.Parameter(torch.tensor(config.prompt_scale_init))

    def get(self, group: str, batch_size: int, device: torch.device) -> torch.Tensor:
        """根据 group 返回对应的一批 prompts."""
        if group == "invariant":
            prompts = self.domain_invariant_prompts
        elif group == "specific":
            prompts = self.domain_specific_prompts
        else:
            raise ValueError(f"不支持的 prompt 组: {group}")

        prompts = prompts.to(device)
        prompts = prompts.unsqueeze(0).expand(batch_size, -1, -1)
        if self.config.enable_prompt_scale:
            return prompts * self.prompt_scale
        return prompts

    def state_dict_for_save(self) -> Dict[str, torch.Tensor]:
        """导出保存所需的权重."""
        return {
            "domain_invariant_prompts": self.domain_invariant_prompts.detach().cpu(),
            "domain_specific_prompts": self.domain_specific_prompts.detach().cpu(),
            "prompt_scale": self.prompt_scale.detach().cpu(),
            "config": self.config.__dict__,
        }

    def load_from_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        """从 state 中恢复权重."""
        self.domain_invariant_prompts.data.copy_(state["domain_invariant_prompts"].to(self.domain_invariant_prompts.device))
        self.domain_specific_prompts.data.copy_(state["domain_specific_prompts"].to(self.domain_specific_prompts.device))
        if "prompt_scale" in state:
            self.prompt_scale.data.copy_(state["prompt_scale"].to(self.prompt_scale.device))


def create_biomedclip_model_and_preprocess_local(model_path: str, device: str):
    """从本地路径加载 BiomedCLIP 模型及预处理配置."""
    model_name = "biomedclip_local"
    config_file = os.path.join(model_path, "open_clip_config.json")
    with open(config_file, "r") as f:
        config = json.load(f)
        model_cfg = config["model_cfg"]
        preprocess_cfg = config["preprocess_cfg"]

    if (
        not model_name.startswith(HF_HUB_PREFIX)
        and model_name not in _MODEL_CONFIGS
        and config is not None
    ):
        _MODEL_CONFIGS[model_name] = model_cfg

    tokenizer = get_tokenizer(model_name)

    model, _, preprocess = create_model_and_transforms(
        model_name=model_name,
        pretrained=os.path.join(model_path, "open_clip_pytorch_model.bin"),
        **{f"image_{k}": v for k, v in preprocess_cfg.items()},
        device=device,
    )

    return model, preprocess, tokenizer


def build_dual_selective_prompt_image_encoder(
    model_path: str,
    device: str,
    num_prompts: int = 4,
    embed_dim: int = 768,
    init_std: float = 0.02,
    prompt_scale_init: float = 1.0,
    enable_prompt_scale: bool = True,
    freeze_backbone: bool = True,
):
    """构建带双域 prompt 的 BiomedCLIP 图像编码器，便于训练脚本动态配置."""

    base_model, preprocess, tokenizer = create_biomedclip_model_and_preprocess_local(
        model_path, device
    )

    prompt_config = DualPromptConfig(
        embed_dim=embed_dim,
        num_prompts=num_prompts,
        init_std=init_std,
        prompt_scale_init=prompt_scale_init,
        enable_prompt_scale=enable_prompt_scale,
    )

    image_encoder = DualSelectivePromptBiomedCLIP(
        base_model,
        prompt_config,
        freeze_backbone=freeze_backbone,
    ).to(device)

    return image_encoder, preprocess, tokenizer


class DualSelectivePromptBiomedCLIP(nn.Module):
    """BiomedCLIP 双域可选 VPT 图像编码器."""

    def __init__(
        self,
        biomedclip_model: nn.Module,
        prompt_config: DualPromptConfig,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.model = biomedclip_model
        self.prompt_learner = DualPromptLearner(prompt_config)
        self.prompt_config = prompt_config

        # 冻结除 prompts 之外的全部参数
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        # prompt learner 内部的参数保持可训练
        for param in self.prompt_learner.parameters():
            param.requires_grad = True

        # 缓存扩展后的 position embedding，以避免重复构造
        self._cached_pos_embed: Optional[torch.Tensor] = None

        # 仅提示信息
        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("=" * 70)
        print("✓ DualSelectivePromptBiomedCLIP 初始化成功")
        print(f"  - 域不变 prompt 数量: {prompt_config.num_prompts}")
        print(f"  - 域特定 prompt 数量: {prompt_config.num_prompts}")
        print(f"  - 可训练参数(仅 prompts): {total_trainable:,}")
        print(f"  - Prompt 缩放启用: {prompt_config.enable_prompt_scale}")
        print("=" * 70)

    # ------------------------------------------------------------------
    # 位置编码相关
    # ------------------------------------------------------------------
    def _get_extended_pos_embed(self, device: torch.device) -> torch.Tensor:
        """构建 [CLS + prompts + patches] 的位置编码."""
        base_pos_embed = self.model.visual.trunk.pos_embed  # [1, 197, 768]
        expected_len = 1 + self.prompt_config.num_prompts + (base_pos_embed.shape[1] - 1)

        if (
            self._cached_pos_embed is None
            or self._cached_pos_embed.shape[1] != expected_len
            or self._cached_pos_embed.device != device
        ):
            prompt_pos = torch.zeros(
                1,
                self.prompt_config.num_prompts,
                base_pos_embed.shape[-1],
                device=base_pos_embed.device,
            )
            extended = torch.cat(
                [base_pos_embed[:, :1, :], prompt_pos, base_pos_embed[:, 1:, :]], dim=1
            )
            self._cached_pos_embed = extended.to(device)

        return self._cached_pos_embed

    # ------------------------------------------------------------------
    # 前向主流程
    # ------------------------------------------------------------------
    def encode_image_with_prompts(
        self,
        images: torch.Tensor,
        prompt_group: str = "invariant",
        pooling: str = "cls",
        normalize: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用指定 prompt 组对图像进行编码，返回图像特征与完整 token 序列。

        Args:
            images: [B, 3, 224, 224]
            prompt_group: "invariant" 或 "specific"
            pooling: token 聚合策略，目前支持 "cls" / "mean_patch"
            normalize: 是否对最终 512 维特征进行 L2 标准化
        Returns:
            image_features: [B, 512]
            tokens: [B, 1 + num_prompts + num_patches, 768]
        """
        trunk = self.model.visual.trunk
        device = images.device
        batch_size = images.shape[0]

        # patch embedding
        x = trunk.patch_embed(images)  # [B, num_patches, 768]

        # 拼接 CLS + prompts + patch tokens
        cls_token = trunk.cls_token.expand(batch_size, -1, -1)
        prompt_tokens = self.prompt_learner.get(prompt_group, batch_size, device)
        x = torch.cat([cls_token, prompt_tokens, x], dim=1)

        # 添加扩展后的位置编码，并通过与原网络一致的预处理流程
        pos_embed = self._get_extended_pos_embed(device)
        x = x + pos_embed[:, : x.size(1), :]
        x = trunk.pos_drop(x)
        x = trunk.patch_drop(x)

        norm_pre = getattr(trunk, "norm_pre", None)
        if norm_pre is not None and not isinstance(norm_pre, nn.Identity):
            x = norm_pre(x)

        for block in trunk.blocks:
            x = block(x)

        x = trunk.norm(x)

        if pooling == "cls":
            pooled = x[:, 0, :]
        elif pooling == "mean_patch":
            pooled = x[:, 1 + self.prompt_config.num_prompts :, :].mean(dim=1)
        else:
            raise ValueError(f"不支持的 pooling 策略: {pooling}")

        fc_norm = getattr(trunk, "fc_norm", None)
        if fc_norm is not None and not isinstance(fc_norm, nn.Identity):
            pooled = fc_norm(pooled)

        image_features = self.model.visual.head(pooled)
        if normalize:
            image_features = F.normalize(image_features, dim=-1)

        return image_features, x

    def forward(
        self,
        images: torch.Tensor,
        prompt_group: str = "invariant",
        pooling: str = "cls",
        normalize: bool = True,
        return_tokens: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """单次前向，仅输出图像特征，可选返回 token 序列."""
        image_features, tokens = self.encode_image_with_prompts(
            images, prompt_group=prompt_group, pooling=pooling, normalize=normalize
        )

        output = {"image_features": image_features}
        if return_tokens:
            output["tokens"] = tokens

        return output

    def encode_image_from_tensor(
        self,
        images: torch.Tensor,
        preprocess,
        prompt_group: str = "invariant",
        pooling: str = "cls",
        normalize: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        便捷方法：直接接受 Tensor 图像（NCHW），使用传入的 `preprocess` 补齐 resize/normalize
        并返回与 `forward` 相同的字典结构。
        """
        device = next(self.parameters()).device
        imgs = preprocess_tensor_images(images, preprocess, device)
        return self.forward(imgs, prompt_group=prompt_group, pooling=pooling, normalize=normalize, return_tokens=True)

    def forward_dual(
        self,
        images: torch.Tensor,
        pooling: str = "cls",
        normalize: bool = True,
        return_tokens: bool = False,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """同时获取域不变 / 域特定两次前向的结果（共用同一套权重）。"""
        outputs = {}
        outputs["invariant"] = self.forward(
            images,
            prompt_group="invariant",
            pooling=pooling,
            normalize=normalize,
            return_tokens=return_tokens,
        )
        outputs["specific"] = self.forward(
            images,
            prompt_group="specific",
            pooling=pooling,
            normalize=normalize,
            return_tokens=return_tokens,
        )
        return outputs

    def forward_dual_from_tensor(
        self,
        images: torch.Tensor,
        preprocess,
        pooling: str = "cls",
        normalize: bool = True,
        return_tokens: bool = False,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        便捷方法：接受原始 Tensor 图像（[B,C,H,W] 或 [C,H,W]），使用 `preprocess`
        进行 resize/normalize，再对两组 prompts 执行前向。

        返回结构与 `forward_dual` 相同；当 `return_tokens=True` 时，每个分支会包含 tokens。
        """
        device = next(self.parameters()).device
        imgs = preprocess_tensor_images(images, preprocess, device)
        return self.forward_dual(imgs, pooling=pooling, normalize=normalize, return_tokens=return_tokens)

    def encode_text(self, texts: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """封装文本编码，方便调用方按需使用."""
        text_features = self.model.encode_text(texts)
        if normalize:
            text_features = F.normalize(text_features, dim=-1)
        return text_features

    def compute_image_text_logits(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """基于共享的 logit_scale 计算匹配分数."""
        logit_scale = self.model.logit_scale.exp()
        return logit_scale * image_features @ text_features.t()

    # ------------------------------------------------------------------
    # 权重保存 / 加载
    # ------------------------------------------------------------------
    def save_prompts(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.prompt_learner.state_dict_for_save(), path)
        print(f"✓ Prompt 权重已保存到: {path}")

    def load_prompts(self, path: str, map_location: Optional[str] = None) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"未找到 prompt 权重文件: {path}")
        state = torch.load(path, map_location=map_location)
        self.prompt_learner.load_from_state_dict(state)
        print(f"✓ Prompt 权重已从 {path} 加载")


def _extract_preprocess_params(preprocess):
    """
    从 open_clip 返回的 preprocess (通常为 torchvision.transforms.Compose)
    中提取 target size 和 Normalize 的 mean/std（如果存在）。
    返回 (target_size, mean_tensor, std_tensor)
    """
    target = 224
    mean = None
    std = None
    if hasattr(preprocess, 'transforms'):
        for t in preprocess.transforms:
            # adapt to torchvision transforms types
            if isinstance(t, T.Resize):
                # t.size can be int or sequence
                if isinstance(t.size, int):
                    target = t.size
                elif isinstance(t.size, (list, tuple)):
                    target = t.size[0]
            if isinstance(t, T.CenterCrop):
                if isinstance(t.size, int):
                    target = t.size
                elif isinstance(t.size, (list, tuple)):
                    target = t.size[0]
            if isinstance(t, T.Normalize):
                mean = torch.tensor(t.mean, dtype=torch.float32)
                std = torch.tensor(t.std, dtype=torch.float32)
    return target, mean, std


def preprocess_tensor_images(images: torch.Tensor, preprocess, device: str):
    """
    将 Tensor 图像批处理为 BiomedCLIP 可接受的输入格式（已归一化并 resize）。

    支持输入形状：[C,H,W] 或 [B,C,H,W]，支持范围：[-1,1], [0,1], [0,255]。
    如果为单通道（C==1），会复制为 3 通道；最后结果位于指定 device 上。
    """
    if images.dim() == 3:
        images = images.unsqueeze(0)

    assert images.dim() == 4, 'images must be [B,C,H,W] or [C,H,W]'
    b, c, h, w = images.shape

    # to float
    imgs = images.float()

    # handle channel: if grayscale -> repeat
    if c == 1:
        imgs = imgs.repeat(1, 3, 1, 1)
        c = 3

    # detect dynamic range and convert to [0,1]
    mn = float(imgs.min().item())
    mx = float(imgs.max().item())
    if mn >= -1.1 and mx <= 1.1:
        # assume [-1,1]
        imgs = (imgs + 1.0) / 2.0
    elif mx > 2.0:
        # assume [0,255]
        imgs = imgs / 255.0
    # else assume already [0,1]

    # extract target size and normalize params from preprocess
    target, mean, std = _extract_preprocess_params(preprocess)

    # resize using interpolate (expects NCHW)
    if imgs.shape[-1] != target or imgs.shape[-2] != target:
        imgs = torch.nn.functional.interpolate(imgs, size=(target, target), mode='bilinear', align_corners=False)

    # normalize
    if mean is not None and std is not None:
        mean = mean.to(imgs.device).view(1, -1, 1, 1)
        std = std.to(imgs.device).view(1, -1, 1, 1)
        imgs = (imgs - mean) / std

    return imgs.to(device)


# ----------------------------------------------------------------------
# 简单测试流程
# ----------------------------------------------------------------------
def test_dual_selective_prompts():
    print("\n" + "=" * 70)
    print("测试 DualSelectivePromptBiomedCLIP")
    print("=" * 70)

    device = "cuda:3"
    model_path = "/root/models/BiomedCLIP"

    model, preprocess, tokenizer = build_dual_selective_prompt_image_encoder(
        model_path=model_path,
        device=device,
        num_prompts=4,
        embed_dim=768,
        init_std=0.02,
        prompt_scale_init=1.0,
        enable_prompt_scale=True,
        freeze_backbone=True,
    )

    # 构造随机输入
    images = torch.randn(2, 3, 224, 224, device=device)
    texts = tokenizer(["lung nodule", "tissue sample"]).to(device)
    text_features = model.encode_text(texts)
    print(f"文本特征: {text_features.shape}")

    # ------ Tensor 预处理测试样例（不同范围与通道） ------
    print('\nTensor preprocessing tests:')
    # 1) 浮点 [0,1]
    t01 = torch.rand(2, 3, 300, 300)
    out01 = model.encode_image_from_tensor(t01, preprocess)
    print(f"  [0,1] -> image_features: {out01['image_features'].shape}, tokens: {out01['tokens'].shape}")

    # 2) 浮点 [-1,1]
    tneg = torch.rand(2, 3, 200, 200) * 2.0 - 1.0
    outneg = model.encode_image_from_tensor(tneg, preprocess)
    print(f"  [-1,1] -> image_features: {outneg['image_features'].shape}")

    # 3) uint8 [0,255]
    t255 = (torch.randint(0, 256, (2, 3, 224, 224), dtype=torch.uint8)).float()
    out255 = model.encode_image_from_tensor(t255, preprocess)
    print(f"  [0,255] uint8 -> image_features: {out255['image_features'].shape}")

    # 4) 灰度单通道，范围 [0,1]
    tgray = torch.rand(1, 1, 180, 180)
    outgray = model.encode_image_from_tensor(tgray, preprocess)
    print(f"  grayscale -> image_features: {outgray['image_features'].shape}")


    # 单独前向
    out_inv = model.forward(images, prompt_group="invariant", return_tokens=True)
    out_spec = model.forward(images, prompt_group="specific")
    logits_inv = model.compute_image_text_logits(out_inv["image_features"], text_features)
    logits_spec = model.compute_image_text_logits(out_spec["image_features"], text_features)

    print("--- 单独前向 ---")
    print(f"域不变图像特征: {out_inv['image_features'].shape}")
    print(f"域特定图像特征: {out_spec['image_features'].shape}")
    print(f"域不变 logits: {logits_inv.shape}")
    print(f"域特定 logits: {logits_spec.shape}")

    # 双前向
    dual_out = model.forward_dual(images)
    diff = torch.mean(
        torch.abs(
            dual_out["invariant"]["image_features"]
            - dual_out["specific"]["image_features"]
        )
    )
    print("--- 双前向 ---")
    print(f"特征差异(平均绝对值): {diff.item():.6f}")

    # 保存 / 加载测试
    save_path = "/tmp/dual_prompt_weights.pth"
    model.save_prompts(save_path)

    new_model, _, _ = build_dual_selective_prompt_image_encoder(
        model_path=model_path,
        device=device,
        num_prompts=4,
        embed_dim=768,
        init_std=0.02,
        prompt_scale_init=1.0,
        enable_prompt_scale=True,
        freeze_backbone=True,
    )
    new_model.load_prompts(save_path, map_location=device)

    new_out = new_model.forward(images, prompt_group="invariant")
    recon_diff = torch.mean(torch.abs(out_inv["image_features"] - new_out["image_features"]))
    print(f"--- 重新加载验证差异: {recon_diff.item():.6f}")

    # 训练流程示例
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    optimizer.zero_grad()

    dual_outputs = model.forward_dual(images)
    labels = torch.arange(images.size(0), device=device)
    logits_inv = model.compute_image_text_logits(
        dual_outputs["invariant"]["image_features"], text_features
    )
    logits_spec = model.compute_image_text_logits(
        dual_outputs["specific"]["image_features"], text_features
    )
    loss_invariant = F.cross_entropy(logits_inv, labels)
    loss_specific = F.cross_entropy(logits_spec, labels)
    total_loss = loss_invariant + loss_specific
    total_loss.backward()
    optimizer.step()

    print("--- 简单训练步 ---")
    print(f"Loss (inv/spec): {loss_invariant.item():.4f} / {loss_specific.item():.4f}")
    print("✓ 测试完成")


if __name__ == "__main__":
    test_dual_selective_prompts()
