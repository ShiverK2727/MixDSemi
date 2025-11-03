"""BiomedCLIP 域不变 VPT-Deep 实现 (Invariant-Only)

本实现基于VPT-Deep思想，在 *每一层* Transformer 
注入 *一组* 可学习的 prompt tokens（域不变），
以实现更强的特征调节。

特性：
- 架构简化：移除了所有 "domain-specific" 逻辑。
- 目标专注：VPTs 100% 专注于学习域不变特征。
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
class PromptConfig:
    """VPT-Deep 基础配置."""
    embed_dim: int = 768
    num_prompts: int = 4
    init_std: float = 0.02
    prompt_scale_init: float = 1.0
    enable_prompt_scale: bool = True
    num_layers: int = 12  # 将由 build 函数动态设置


class PromptLearner(nn.Module):
    """管理 L x P x D 维度 prompts 的模块 (VPT-Deep)."""

    def __init__(self, config: PromptConfig):
        super().__init__()
        self.config = config

        # 域不变 prompts
        # 尺寸: [num_layers, num_prompts, embed_dim]
        self.prompts = nn.Parameter(
            torch.randn(
                config.num_layers, config.num_prompts, config.embed_dim
            ) * config.init_std
        )

        # 全局缩放因子
        self.prompt_scale = nn.Parameter(torch.tensor(config.prompt_scale_init))

    def get(
        self, layer_idx: int, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        """根据 layer_idx 返回对应的一批 prompts."""
        prompts = self.prompts[layer_idx]
        prompts = prompts.to(device)
        prompts = prompts.unsqueeze(0).expand(batch_size, -1, -1)
        if self.config.enable_prompt_scale:
            return prompts * self.prompt_scale
        return prompts

    def state_dict_for_save(self) -> Dict[str, torch.Tensor]:
        """导出保存所需的权重."""
        return {
            "prompts": self.prompts.detach().cpu(),
            "prompt_scale": self.prompt_scale.detach().cpu(),
            "config": self.config.__dict__,
        }

    def load_from_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        """从 state 中恢复权重."""
        self.prompts.data.copy_(
            state["prompts"].to(self.prompts.device)
        )
        if "prompt_scale" in state:
            self.prompt_scale.data.copy_(
                state["prompt_scale"].to(self.prompt_scale.device)
            )

# ... (create_biomedclip_model_and_preprocess_local 保持不变) ...
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


def build_invariant_prompt_image_encoder( # 重命名
    model_path: str,
    device: str,
    num_prompts: int = 4,
    embed_dim: int = 768,
    init_std: float = 0.02,
    prompt_scale_init: float = 1.0,
    enable_prompt_scale: bool = True,
    freeze_backbone: bool = True,
):
    """构建带 Invariant-Only VPT-Deep 的 BiomedCLIP 图像编码器."""

    base_model, preprocess, tokenizer = create_biomedclip_model_and_preprocess_local(
        model_path, device
    )
    
    # 动态获取 Transformer 的层数
    num_layers = len(base_model.visual.trunk.blocks)

    prompt_config = PromptConfig( # 使用新的 Config
        embed_dim=embed_dim,
        num_prompts=num_prompts,
        init_std=init_std,
        prompt_scale_init=prompt_scale_init,
        enable_prompt_scale=enable_prompt_scale,
        num_layers=num_layers, 
    )

    image_encoder = InvariantPromptBiomedCLIP( # 使用新的模型类
        base_model,
        prompt_config,
        freeze_backbone=freeze_backbone,
    ).to(device)

    return image_encoder, preprocess, tokenizer


class InvariantPromptBiomedCLIP(nn.Module): # 重命名
    """BiomedCLIP 域不变 VPT-Deep 图像编码器."""

    def __init__(
        self,
        biomedclip_model: nn.Module,
        prompt_config: PromptConfig, # 使用新的 Config
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.model = biomedclip_model
        self.prompt_learner = PromptLearner(prompt_config) # 使用新的 Learner
        self.prompt_config = prompt_config
        self.num_layers = prompt_config.num_layers

        # 冻结
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        for param in self.prompt_learner.parameters():
            param.requires_grad = True

        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("=" * 70)
        print("✓ InvariantPromptBiomedCLIP (VPT-Deep) 初始化成功")
        print(f"  - Transformer 层数 (L): {self.num_layers}")
        print(f"  - 每层 prompt 数量 (P): {prompt_config.num_prompts}")
        print(f"  - 总 prompts: {self.num_layers * prompt_config.num_prompts}")
        print(f"  - 可训练参数(仅 prompts): {total_trainable:,}")
        print(f"  - Prompt 缩放启用: {prompt_config.enable_prompt_scale}")
        print("=" * 70)

    # ------------------------------------------------------------------
    # 前向主流程 (VPT-Deep)
    # ------------------------------------------------------------------
    def encode_image_with_prompts(
        self,
        images: torch.Tensor,
        pooling: str = "cls",
        normalize: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        VPT-Deep 图像编码 (已移除 prompt_group)
        """
        trunk = self.model.visual.trunk
        device = images.device
        batch_size = images.shape[0]

        # 1. 初始 Patch 和 CLS
        x_patches = trunk.patch_embed(images)  # [B, num_patches(N), 768]
        cls_token = trunk.cls_token.expand(batch_size, -1, -1) # [B, 1, 768]
        
        # 2. 拼接 CLS 和 Patches
        x = torch.cat([cls_token, x_patches], dim=1) # [B, 1+N, 768]
        
        # 3. 添加 *原始* 位置编码
        x = x + trunk.pos_embed[:, : x.size(1), :]
        x = trunk.pos_drop(x)
        x = trunk.patch_drop(x)

        norm_pre = getattr(trunk, "norm_pre", None)
        if norm_pre is not None and not isinstance(norm_pre, nn.Identity):
            x = norm_pre(x)

        # 4. 迭代 Transformer 层 (VPT-Deep 核心)
        for i, block in enumerate(trunk.blocks):
            # 4.1 获取第 i 层的 prompts (不再需要 group)
            prompt_tokens = self.prompt_learner.get(
                i, batch_size, device
            ) # [B, P, 768]
            
            # 4.2 注入: 拼接 [CLS, Patches] 和 [Prompts]
            x_with_prompts = torch.cat([x, prompt_tokens], dim=1)
            
            # 4.3 通过 Transformer 块
            x_with_prompts = block(x_with_prompts)
            
            # 4.4 丢弃: 只保留 [CLS, Patches] 的输出
            x = x_with_prompts[:, : -(self.prompt_config.num_prompts), :]

        # 5. 最终 Norm
        x = trunk.norm(x)
        tokens_output = x 

        # 7. 池化
        if pooling == "cls":
            pooled = x[:, 0, :]
        elif pooling == "mean_patch":
            pooled = x[:, 1:, :].mean(dim=1) 
        else:
            raise ValueError(f"不支持的 pooling 策略: {pooling}")

        fc_norm = getattr(trunk, "fc_norm", None)
        if fc_norm is not None and not isinstance(fc_norm, nn.Identity):
            pooled = fc_norm(pooled)

        # 8. 投影 Head
        image_features = self.model.visual.head(pooled)
        if normalize:
            image_features = F.normalize(image_features, dim=-1)

        return image_features, tokens_output

    def forward(
        self,
        images: torch.Tensor,
        pooling: str = "cls",
        normalize: bool = True,
        return_tokens: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """单次前向 (已移除 prompt_group)"""
        image_features, tokens = self.encode_image_with_prompts(
            images, pooling=pooling, normalize=normalize
        )

        output = {"image_features": image_features}
        if return_tokens:
            output["tokens"] = tokens 

        return output

    # ... (移除 forward_dual 和 forward_dual_from_tensor) ...
    
    # --- 修复: 匹配训练脚本的调用 ---
    def encode_image_from_tensor(
        self,
        images: torch.Tensor,
        preprocess,
        pooling: str = "cls",
        normalize: bool = True,
        return_tokens: bool = False, # <--- 1. 添加 'return_tokens' 参数
    ) -> Dict[str, torch.Tensor]:
        """
        便捷方法 (已移除 prompt_group)
        """
        device = next(self.parameters()).device
        imgs = preprocess_tensor_images(images, preprocess, device)
        # <--- 2. 将 'return_tokens' 传递给 self.forward
        return self.forward(imgs, pooling=pooling, normalize=normalize, return_tokens=return_tokens)
    # --- 修复结束 ---

    def encode_text(self, texts: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        text_features = self.model.encode_text(texts)
        if normalize:
            text_features = F.normalize(text_features, dim=-1)
        return text_features

    def compute_image_text_logits(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        logit_scale = self.model.logit_scale.exp()
        return logit_scale * image_features @ text_features.t()

    # ------------------------------------------------------------------
    # 权重保存 / 加载
    # ------------------------------------------------------------------
    def save_prompts(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # 使用 state_dict_for_save
        torch.save(self.prompt_learner.state_dict_for_save(), path)
        print(f"✓ VPT-Deep (Invariant-Only) 权重已保存到: {path}")

    def load_prompts(self, path: str, map_location: Optional[str] = None) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"未找到 prompt 权重文件: {path}")
        
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            # 确保加载的 state 字典键与 load_from_state_dict 匹配
            state = torch.load(path, map_location=map_location)
            
        self.prompt_learner.load_from_state_dict(state)
        print(f"✓ VPT-Deep (Invariant-Only) 权重已从 {path} 加载")

# ... (preprocess_tensor_images 和 _extract_preprocess_params 保持不变) ...
def _extract_preprocess_params(preprocess):
    target = 224
    mean = None
    std = None
    if hasattr(preprocess, 'transforms'):
        for t in preprocess.transforms:
            if isinstance(t, T.Resize):
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
    if images.dim() == 3:
        images = images.unsqueeze(0)
    assert images.dim() == 4, 'images must be [B,C,H,W] or [C,H,W]'
    b, c, h, w = images.shape
    imgs = images.float()
    if c == 1:
        imgs = imgs.repeat(1, 3, 1, 1)
        c = 3
    mn = float(imgs.min().item())
    mx = float(imgs.max().item())
    if mn >= -1.1 and mx <= 1.1:
        imgs = (imgs + 1.0) / 2.0
    elif mx > 2.0:
        imgs = imgs / 2.0 # 修正：应该是 imgs / 255.0
    
    # --- 修复：从 `preprocess_tensor_images` 中复制的笔误 ---
    # 之前这里是 imgs / 2.0，已修正为 255.0
    elif mx > 2.0:
        # assume [0,255]
        imgs = imgs / 255.0
    # --- 修复结束 ---

    target, mean, std = _extract_preprocess_params(preprocess)
    if imgs.shape[-1] != target or imgs.shape[-2] != target:
        imgs = torch.nn.functional.interpolate(imgs, size=(target, target), mode='bilinear', align_corners=False)
    if mean is not None and std is not None:
        mean = mean.to(imgs.device).view(1, -1, 1, 1)
        std = std.to(imgs.device).view(1, -1, 1, 1)
        imgs = (imgs - mean) / std
    return imgs.to(device)

