"""
BiomedCLIP 域不变 VPT + TPT 分割模型 (v3 - CoOp Style)

本实现融合了多种思想：
1.  VPT-Deep (视觉提示): 在 ViT 每一层注入 $V_l$ 学习域不变 patch 特征。
2.  TPT (CoOp-style): 在文本输入端注入一组 C_l (learnable context)，
    构建 [SOT, C_l, Class, EOT] 序列，以学习语义原型 $W_{robust}$。
3.  Patch 匹配: 直接使用 $W_{robust}$ 和 $H_{patch}$ 进行点积。

* 变更 (v3):
- TPT 从 "TPT-Deep" (逐层注入) 架构更改为 "CoOp-style" (浅层拼接) 架构。
- `encode_text_with_prompts` 被重写，`TextPromptLearner` 被简化。
- `forward` 函数现在接受 `text_list: list[str]`。
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS


# --------------------------------------------------------------------------
# 1. PEFT 模块 (VPT 和 TPT)
# --------------------------------------------------------------------------

@dataclass
class VisualPromptConfig:
    """VPT-Deep 基础配置 (与 v2 相同)."""
    embed_dim: int = 768  # ViT 内部维度
    num_prompts: int = 4
    init_std: float = 0.02
    num_layers: int = 12  # 将由 build 函数动态设置

@dataclass
class TextPromptConfig:
    """TPT (CoOp-style) 基础配置 (已修改)."""
    embed_dim: int = 512  # 文本编码器内部维度
    num_prompts: int = 4
    init_std: float = 0.02
    # (已移除 num_layers)


class VisualPromptLearner(nn.Module):
    """管理视觉 prompts (V_l) 的模块 (VPT-Deep). (与 v2 相同)"""

    def __init__(self, config: VisualPromptConfig):
        super().__init__()
        self.config = config

        self.prompts = nn.Parameter(
            torch.randn(
                config.num_layers, config.num_prompts, config.embed_dim
            ) * config.init_std
        )
        print(f"✓ 初始化 VisualPromptLearner (V_l, Deep): {self.prompts.shape}")

    def get(
        self, layer_idx: int, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        prompts = self.prompts[layer_idx]
        prompts = prompts.to(device)
        return prompts.unsqueeze(0).expand(batch_size, -1, -1)

    def state_dict_for_save(self) -> Dict[str, torch.Tensor]:
        return {
            "prompts": self.prompts.detach().cpu(),
            "config": self.config.__dict__,
        }

    def load_from_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.prompts.data.copy_(
            state["prompts"].to(self.prompts.device)
        )


class TextPromptLearner(nn.Module):
    """管理文本 prompts (C_l) 的模块 (CoOp-style, 已修改)."""

    def __init__(self, config: TextPromptConfig):
        super().__init__()
        self.config = config

        # 域无关的可学习上下文 (C_l)
        # 尺寸: [num_prompts, embed_dim]
        self.prompts = nn.Parameter(
            torch.randn(
                config.num_prompts, config.embed_dim
            ) * config.init_std
        )
        print(f"✓ 初始化 TextPromptLearner (C_l, CoOp-style): {self.prompts.shape}")

    # (已移除 get 方法)

    def state_dict_for_save(self) -> Dict[str, torch.Tensor]:
        return {
            "prompts": self.prompts.detach().cpu(),
            "config": self.config.__dict__,
        }

    def load_from_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.prompts.data.copy_(
            state["prompts"].to(self.prompts.device)
        )

# --------------------------------------------------------------------------
# 2. 本地 BiomedCLIP 加载器 (与 v2 相同)
# --------------------------------------------------------------------------

def create_biomedclip_model_and_preprocess_local(model_path: str, device: str):
    """从本地路径加载 BiomedCLIP 模型及预处理配置."""
    model_name = "biomedclip_local"
    config_file = os.path.join(model_path, "open_clip_config.json")
    if not os.path.exists(config_file):
         raise FileNotFoundError(f"未找到配置文件: {config_file}。请确保 {model_path} 包含 open_clip_config.json")

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

    model_file = os.path.join(model_path, "open_clip_pytorch_model.bin")
    if not os.path.exists(model_file):
        model_file_alt = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(model_file_alt):
            model_file = model_file_alt
        else:
            raise FileNotFoundError(f"未找到权重文件: {model_file} 或 {model_file_alt}")

    model, _, preprocess = create_model_and_transforms(
        model_name=model_name,
        pretrained=model_file,
        **{f"image_{k}": v for k, v in preprocess_cfg.items()},
        device=device,
    )

    return model, preprocess, tokenizer

# --------------------------------------------------------------------------
# 3. 完整的主模型 (已修改)
# --------------------------------------------------------------------------

class VPT_TPT_CLIP_Seg(nn.Module):
    """
    VPT (Deep) + TPT (CoOp-style) + 直接 Patch-Matching 的分割模型
    """

    def __init__(
        self,
        biomedclip_model: nn.Module,
        tokenizer,
        visual_prompt_config: VisualPromptConfig,
        text_prompt_config: TextPromptConfig,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.model = biomedclip_model
        self.tokenizer = tokenizer
        
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
        
        self.visual_prompt_learner = VisualPromptLearner(visual_prompt_config)
        self.text_prompt_learner = TextPromptLearner(text_prompt_config)
        
        self.clip_embed_dim = self.model.text.proj.shape[1]
        
        for param in self.visual_prompt_learner.parameters():
            param.requires_grad = True
        for param in self.text_prompt_learner.parameters():
            param.requires_grad = True
        
        self.to(self.model.device)

        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("=" * 70)
        print("✓ VPT_TPT_CLIP_Seg (VPT-Deep + TPT-CoOp) 初始化成功")
        print(f"  - 视觉 Prompts (V_l, Deep): {self.visual_prompt_learner.prompts.numel():,}")
        print(f"  - 文本 Prompts (C_l, CoOp): {self.text_prompt_learner.prompts.numel():,}")
        print(f"  - 总可训练参数: {total_trainable:,}")
        print("=" * 70)

    # ------------------------------------------------------------------
    # 视觉编码 (VPT-Deep) - (与 v2 相同)
    # ------------------------------------------------------------------
    def _visual_forward_with_prompts(self, images: torch.Tensor) -> torch.Tensor:
        """在 BiomedCLIP 图像分支中注入 VPT-Deep prompts (V_l)"""
        trunk = self.model.visual.trunk
        device = images.device
        batch_size = images.shape[0]

        x_patches = trunk.patch_embed(images)
        cls_token = trunk.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_token, x_patches], dim=1)
        
        x = x + trunk.pos_embed[:, : x.size(1), :]
        x = x.pos_drop(x)
        x = x.patch_drop(x)

        norm_pre = getattr(trunk, "norm_pre", None)
        if norm_pre is not None and not isinstance(norm_pre, nn.Identity):
            x = norm_pre(x)

        for i, block in enumerate(trunk.blocks):
            prompt_tokens = self.visual_prompt_learner.get(
                i, batch_size, device
            ) # [B, P_v, 768]
            x_with_prompts = torch.cat([x, prompt_tokens], dim=1)
            x_with_prompts = block(x_with_prompts)
            x = x_with_prompts[:, : -(self.visual_prompt_learner.config.num_prompts), :]

        x = trunk.norm(x)
        patch_tokens = x[:, 1:, :] # [B, N, 768] (N=196)
        patch_features = self.model.visual.head(patch_tokens) # [B, N, D] (D=512)
        patch_features = F.normalize(patch_features, dim=-1)

        return patch_features

    # ------------------------------------------------------------------
    # 文本编码 (TPT - CoOp-style) - (已重写)
    # ------------------------------------------------------------------
    def encode_text_with_prompts(self, text_list: List[str]) -> torch.Tensor:
        """
        在 BiomedCLIP 文本分支中注入 CoOp-style prompts (C_l)
        构建 [SOT, C_l, Class, EOT] 序列
        
        Args:
            text_list (list[str]): K 个类别的原始字符串 (例如 ["prostate", "background"])
        """
        
        text_transformer = self.model.text
        device = self.model.device
        K = len(text_list) # 类别数

        # 1. 获取可学习的上下文 (C_l)
        # self.text_prompt_learner.prompts 的形状是 [P_t, D_text]
        context_vectors = self.text_prompt_learner.prompts # [P_t, 512]
        P_t = context_vectors.shape[0]

        # 2. 获取 SOT (Start) 和 EOT (End) 嵌入
        sot_token = torch.tensor([[self.tokenizer.sot_token_id]], device=device).long()
        eot_token = torch.tensor([[self.tokenizer.eot_token_id]], device=device).long()
        
        sot_embed = text_transformer.token_embedding(sot_token).squeeze(0) # [1, 512]
        eot_embed = text_transformer.token_embedding(eot_token).squeeze(0) # [1, 512]

        # 3. 准备 K 个类别的嵌入
        all_text_features = []
        for class_name in text_list:
            # 3.1 Tokenize 单个类名
            # open_clip tokenizer 会自动添加 SOT 和 EOT，我们需要移除它们
            class_tokens = self.tokenizer(class_name).to(device) # [1, 77]
            
            # 找到 EOT token 的索引
            eot_idx = (class_tokens == self.tokenizer.eot_token_id).nonzero()[0, 1]
            # 只保留 SOT 和 EOT 之间的 class token id (即索引 1 到 eot_idx-1)
            class_tokens_clean = class_tokens[0, 1:eot_idx] # [L_k]
            
            # 3.2 嵌入类名
            class_embed = text_transformer.token_embedding(class_tokens_clean) # [L_k, 512]
            L_k = class_embed.shape[0]
            
            # 3.3 构建完整的序列: [SOT, C_l, Class, EOT]
            # [1, D], [P_t, D], [L_k, D], [1, D]
            full_sequence = torch.cat(
                [
                    sot_embed,
                    context_vectors,
                    class_embed,
                    eot_embed
                ],
                dim=0
            ) # [1 + P_t + L_k + 1, D]
            
            # 3.4 添加位置编码
            seq_len = full_sequence.shape[0]
            if seq_len > text_transformer.positional_embedding.shape[0]:
                raise ValueError(f"序列长度 {seq_len} 超过了 CLIP 的最大位置编码 {text_transformer.positional_embedding.shape[0]}")
                
            full_sequence = full_sequence + text_transformer.positional_embedding[:seq_len]
            
            # 3.5 准备 Transformer 输入
            # (Batch size = 1, Seq_len, Dim)
            full_sequence = full_sequence.unsqueeze(0) 
            # LND -> NLD (Transformer 期望的输入)
            x = full_sequence.permute(1, 0, 2) # [Seq, 1, D]
            
            # 4. (新) 单次前向传播
            # (注意：我们不再逐层循环，而是完整运行)
            x = text_transformer.transformer(x) 
            
            x = x.permute(1, 0, 2)  # NLD -> LND [1, Seq, D]
            x = text_transformer.ln_final(x)
            
            # 5. 池化
            # 我们使用 EOT token (最后一个 token) 的输出来池化
            # 它的索引是 seq_len - 1
            # open_clip 标准做法是使用 token id 列表中的 argmax，我们这里简化
            # 为与 open_clip 行为一致，应使用 argmax 索引 EOT
            # [SOT, C_l(P_t), Class(L_k), EOT]
            # EOT 索引是 1 + P_t + L_k
            eot_index_in_sequence = 1 + P_t + L_k
            
            # pooled = x[0, eot_index_in_sequence, :] @ text_transformer.text_projection
            
            # 使用 open_clip 的标准池化方式更安全
            # 构造一个假的 token id 列表来匹配
            fake_token_ids = torch.zeros(1, seq_len, dtype=torch.long, device=device)
            fake_token_ids[0, eot_index_in_sequence] = self.tokenizer.eot_token_id # 放置 EOT id
            
            pooled = x[
                torch.arange(x.shape[0]), fake_token_ids.argmax(dim=-1)
            ] @ text_transformer.text_projection

            all_text_features.append(pooled.squeeze(0))

        # 6. 堆叠 K 个类别的原型
        W_robust = torch.stack(all_text_features, dim=0) # [K, D]
        W_robust = F.normalize(W_robust, dim=-1)
        
        return W_robust

    # ------------------------------------------------------------------
    # 主前向过程 (与 v2 相同)
    # ------------------------------------------------------------------
    def forward(
        self, 
        images: torch.Tensor, 
        text_list: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        模型的主 forward pass (VPT + TPT-CoOp + 直接 Patch-Matching)
        
        Args:
            images (torch.Tensor): [B, 3, H, W] 预处理后的图像
            text_list (List[str]): K 个类别的字符串列表
        
        Returns:
            Dict[str, torch.Tensor]:
                "H_semantic_maps": [B, K, N] 最终的 14x14 语义图 (N=196)
                "patch_features": [B, N, D] 域不变的 patch 特征 (用于一致性损失)
        """
        device = self.model.device
        images = images.to(device)
        # (text_list 已经是字符串列表)

        # 1. 提取视觉 patch 特征 (V_l 生效)
        patch_features = self._visual_forward_with_prompts(images) # [B, N, D]
        
        B, N, D_visual = patch_features.shape
        if D_visual != self.clip_embed_dim:
             raise ValueError(f"视觉输出维度 {D_visual} 与 CLIP 嵌入维度 {self.clip_embed_dim} 不匹配")
        
        # 2. 提取文本原型 (C_l CoOp 生效)
        W_robust = self.encode_text_with_prompts(text_list) # [K, D]
        K, D_text = W_robust.shape
        
        # 3. 计算最终语义图 (直接点积)
        W_robust_expanded = W_robust.unsqueeze(0).expand(B, -1, -1) # [B, K, D]
        H_semantic_maps = W_robust_expanded @ patch_features.transpose(1, 2) # [B, K, N]
        
        # 4. 返回您需要的特征
        return {
            "H_semantic_maps": H_semantic_maps,    # [B, K, 196]
            "patch_features": patch_features,      # [B, 196, D]
            "W_robust": W_robust                   # [K, D]
        }

    # ------------------------------------------------------------------
    # 辅助函数 (与 v2 相同)
    # ------------------------------------------------------------------

    def save_all_prompts(self, filepath: str) -> None:
        """保存所有可学习的参数 (V_l, C_l)"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        save_dict = {
            'visual_prompt_learner': self.visual_prompt_learner.state_dict_for_save(),
            'text_prompt_learner': self.text_prompt_learner.state_dict_for_save(),
        }
        torch.save(save_dict, filepath)
        print(f"✓ 所有 PEFT 权重已保存到: {filepath}")

    def load_all_prompts(self, filepath: str, map_location: Optional[str] = None) -> None:
        """加载已训练的 PEFT 参数"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"未找到 PEFT 权重文件: {filepath}")
        
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            state_dict = torch.load(filepath, map_location=map_location or self.model.device)
            
        self.visual_prompt_learner.load_from_state_dict(state_dict['visual_prompt_learner'])
        self.text_prompt_learner.load_from_state_dict(state_dict['text_prompt_learner'])
        print(f"✓ 所有 PEFT 权重已从 {filepath} 加载")

# --------------------------------------------------------------------------
# 4. 图像预处理 (与 v2 相同)
# --------------------------------------------------------------------------

def _extract_preprocess_params(preprocess):
    """从 open_clip 预处理中提取标准化参数"""
    target = 224
    mean = None
    std = None
    if hasattr(preprocess, 'transforms'):
        for t in preprocess.transforms:
            if isinstance(t, (T.Resize, T.CenterCrop)):
                size = getattr(t, 'size', target)
                if isinstance(size, int):
                    target = size
                elif isinstance(size, (list, tuple)):
                    target = size[0]
            if isinstance(t, T.Normalize):
                mean = torch.tensor(t.mean, dtype=torch.float32)
                std = torch.tensor(t.std, dtype=torch.float32)
    return target, mean, std

def preprocess_tensor_images(images: torch.Tensor, preprocess, device: str):
    """对已在 [0, 1] 或 [-1, 1] 范围内的 Tensor 进行 CLIP 标准化。"""
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
        imgs = imgs / 255.0
    
    target, mean, std = _extract_preprocess_params(preprocess)
    
    if imgs.shape[-1] != target or imgs.shape[-2] != target:
        imgs = torch.nn.functional.interpolate(
            imgs, size=(target, target), mode='bilinear', align_corners=False
        )
    
    if mean is not None and std is not None:
        mean = mean.to(imgs.device).view(1, -1, 1, 1)
        std = std.to(imgs.device).view(1, -1, 1, 1)
        imgs = (imgs - mean) / std
        
    return imgs.to(device)


# --------------------------------------------------------------------------
# 5. Builder 函数 (已更新)
# --------------------------------------------------------------------------

def build_vpt_tpt_seg_model(
    model_path: str,
    device: str,
    visual_num_prompts: int = 4,
    text_num_prompts: int = 4,
    freeze_backbone: bool = True,
):
    """
    构建完整的 VPT-TPT-Seg 模型 (v3 CoOp-style)
    """
    
    base_model, preprocess, tokenizer = create_biomedclip_model_and_preprocess_local(
        model_path, device
    )
    
    # 动态获取 Transformer 的层数和维度
    try:
        num_visual_layers = len(base_model.visual.trunk.blocks)
        visual_embed_dim = base_model.visual.trunk.embed_dim
    except Exception:
        num_visual_layers = 12
        visual_embed_dim = 768
        
    try:
        # (文本层数不再需要)
        text_embed_dim = base_model.text.token_embedding.embedding_dim
    except Exception:
        text_embed_dim = 512

    visual_prompt_config = VisualPromptConfig(
        embed_dim=visual_embed_dim,
        num_prompts=visual_num_prompts,
        num_layers=num_visual_layers,
    )
    
    text_prompt_config = TextPromptConfig(
        embed_dim=text_embed_dim,
        num_prompts=text_num_prompts,
        # (已移除 num_layers)
    )

    # 实例化主模型
    model = VPT_TPT_CLIP_Seg(
        base_model,
        tokenizer,
        visual_prompt_config,
        text_prompt_config,
        freeze_backbone=freeze_backbone,
    )

    return model, preprocess, tokenizer

# --------------------------------------------------------------------------
# 6. 示例用法 (已更新)
# --------------------------------------------------------------------------
if __name__ == "__main__":
    """
    这是一个示例，展示如何构建模型并执行一次完整的前向传播。
    你需要一个本地的 BiomedCLIP 模型目录 (包含 config 和权重文件)。
    """
    
    # --- 配置 ---
    BIOMEDCLIP_MODEL_PATH = "/root/models/BiomedCLIP" # <--- 修改为您的本地 BiomedCLIP 路径
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    B = 2  # Batch size
    K = 2  # 类别数 (例如: LV, RV, Background)
    N = 196 # Patch 数量 (14x14)
    
    # 检查模型路径是否存在
    if not os.path.exists(BIOMEDCLIP_MODEL_PATH):
        print("="*70)
        print(f"警告: 示例模型路径 {BIOMEDCLIP_MODEL_PATH} 不存在。")
        print("此 main 函数将跳过执行。")
        print("请下载 BiomedCLIP (open_clip 格式) 并解压到该目录，或修改路径。")
        print("="*70)
    else:
        print(f"正在从 {BIOMEDCLIP_MODEL_PATH} 加载模型...")
        
        # 1. 构建模型
        model, preprocess, tokenizer = build_vpt_tpt_seg_model(
            model_path=BIOMEDCLIP_MODEL_PATH,
            device=DEVICE,
            visual_num_prompts=4,
            text_num_prompts=4, # CoOp-style 提示
        )
        model.to(DEVICE)
        model.train() 
        
        # 2. 准备输入数据 (已修改)
        images = torch.rand(B, 3, 224, 224).to(DEVICE) 
        
        # (新) CoOp-style 需要字符串列表
        text_list = ["prostate", "background"] 
        if len(text_list) != K:
            raise ValueError(f"示例类别数 K={K} 与 text_list 长度 {len(text_list)} 不匹配")
        
        # (不再需要 text_tokens)
        
        print("\n--- 准备输入 ---")
        print(f"图像张量 shape: {images.shape}")
        print(f"文本列表: {text_list} (K={K})")
        
        # 3. 执行前向传播和反向传播 (已修改)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer.zero_grad()
        
        # (新) 传递字符串列表
        outputs = model(images, text_list) 
        
        H_maps = outputs["H_semantic_maps"] # [B, K, N]
        patches = outputs["patch_features"] # [B, N, D]
        
        # 模拟损失
        target_maps = torch.rand_like(H_maps).softmax(dim=1)
        loss1 = F.kl_div(H_maps.log_softmax(dim=1), target_maps, reduction='batchmean')
        loss2 = patches.mean()
        total_loss = loss1 + loss2
        total_loss.backward()
        
        print("\n--- 执行前向和反向传播 ---")
        print(f"输出 'H_semantic_maps' (分割图) shape: {H_maps.shape}")
        print(f"输出 'patch_features' (用于一致性) shape: {patches.shape}")
        print(f"计算总损失: {total_loss.item():.4f}")

        # 4. 验证梯度
        print("\n--- 验证 PEFT 模块梯度 ---")
        
        vpt_grad = model.visual_prompt_learner.prompts.grad
        if vpt_grad is not None:
            print(f"✓ 视觉提示 (V_l) 梯度已计算 (grad.norm = {vpt_grad.norm():.2f})")
        else:
            print("✗ 警告: 视觉提示 (V_l) 没有梯度!")

        tpt_grad = model.text_prompt_learner.prompts.grad
        if tpt_grad is not None:
            print(f"✓ 文本提示 (C_l) 梯度已计算 (grad.norm = {tpt_grad.norm():.2f})")
        else:
            print("✗ 警告: 文本提示 (C_l) 没有梯度!")

        backbone_grad = model.model.visual.trunk.blocks[0].attn.qkv.weight.grad
        if backbone_grad is None:
            print("✓ CLIP 主干已冻结 (无梯度)")
        else:
            print("✗ 警告: CLIP 主干未冻结 (有梯度)!")
            
        # 5. 保存和加载 PEFT 权重
        save_path = "./temp_peft_weights_v3.pth"
        model.save_all_prompts(save_path)
        
        model_new, _, _ = build_vpt_tpt_seg_model(
            model_path=BIOMEDCLIP_MODEL_PATH,
            device=DEVICE,
        )
        model_new.load_all_prompts(save_path)
        os.remove(save_path)
