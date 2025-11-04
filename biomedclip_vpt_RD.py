import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip

from utils.clip import create_biomedclip_model_and_preprocess_local
from biomedclip_vpt_invariant_only import preprocess_tensor_images

# PromptLearner 保持不变 (与您提供的文件一致)
class PromptLearner(nn.Module):
    """
    VPT (Visual Prompt Tuning) 模块
    在ViT的每一层都插入可学习的prompt
    """
    def __init__(self, num_prompts, embed_dim, init_std, 
                 num_layers, prompt_scale_init, enable_scale):
        super().__init__()
        # ... (此处代码与您提供的 biomedclip_vpt_patch.py 完全相同) ...
        self.num_prompts = num_prompts
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # 初始化 prompts
        self.prompts = nn.Parameter(torch.empty(
            num_layers, num_prompts, embed_dim
        ))
        nn.init.normal_(self.prompts, std=init_std)

        # 可学习的缩放因子
        self.enable_scale = enable_scale
        if self.enable_scale:
            self.prompt_scale = nn.Parameter(torch.full(
                (num_layers, 1, 1), prompt_scale_init
            ))
        else:
            self.register_buffer('prompt_scale', torch.full(
                (num_layers, 1, 1), prompt_scale_init
            ))
            
    def forward(self, x, layer_idx):
        """返回第 layer_idx 层的 prompts"""
        prompts = self.prompts[layer_idx] * self.prompt_scale[layer_idx]
        prompts = prompts.to(device=x.device, dtype=x.dtype)
        # 扩展到 batch size
        return prompts.unsqueeze(0).expand(x.shape[0], -1, -1)


class VPT_CLIP_RD(nn.Module):
    """
    融合了VPT和RD (ZegCLIP [1] Sec 3.5, Eq 8) 的锚点生成器
    [1] Zhou et al. - ZegCLIP (CVPR 2023)
    """
    def __init__(self, model_path, num_prompts, embed_dim, init_std, 
                 prompt_scale_init, enable_scale, device):
        super().__init__()

        # 统一处理 device 表达方式
        self.device = device

        # 1. 加载并冻结 BiomedCLIP
        self.clip_model = None
        self.preprocess = None
        self.tokenizer = None

        model_path_is_dir = isinstance(model_path, str) and os.path.isdir(model_path)
        config_file = os.path.join(model_path, "open_clip_config.json") if isinstance(model_path, str) else None

        if model_path_is_dir and os.path.exists(config_file):
            # 使用本地 BiomedCLIP 权重
            self.clip_model, self.preprocess, self.tokenizer = create_biomedclip_model_and_preprocess_local(
                model_path, device
            )
        else:
            # 兼容旧流程：接受 open_clip 的预训练标签或权重路径
            self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
                'ViT-B-16',
                pretrained=model_path,
                precision='amp' if 'cuda' in str(device) else 'fp32'
            )
            self.tokenizer = open_clip.get_tokenizer('ViT-B-16')

        if self.clip_model is None:
            raise RuntimeError(f"无法根据 model_path={model_path} 加载 BiomedCLIP 模型，请检查路径/配置是否正确")

        # 将模型移动到目标设备，并冻结 CLIP 主干
        self.clip_model.eval().to(self.device)
        for param in self.clip_model.parameters():
            param.requires_grad = False

        visual = self.clip_model.visual
        # open_clip 的 BiomedCLIP 使用 TimmModel 封装，核心在 trunk
        self.visual_trunk = getattr(visual, 'trunk', visual)

        # 记录 patch_size，供训练脚本调用
        patch_size = getattr(visual, 'patch_size', None)
        if patch_size is None and hasattr(self.visual_trunk, 'patch_embed'):
            patch_size = getattr(self.visual_trunk.patch_embed, 'patch_size', None)
            if patch_size is not None:
                setattr(self.clip_model.visual, 'patch_size', patch_size)
        self.patch_size = patch_size

        # 2. 注入 VPT (PromptLearner)
        if hasattr(self.visual_trunk, 'blocks'):
            num_layers = len(self.visual_trunk.blocks)
        else:
            raise AssertionError("BiomedCLIP 图像编码器缺少 transformer blocks，无法注入 VPT")

        self.prompt_learner = PromptLearner(
            num_prompts, embed_dim, init_std,
            num_layers, prompt_scale_init, enable_scale
        ).to(self.device)

        # 3. 覆盖视觉分支的前向过程（闭包持有 VPT_CLIP_RD 实例）
        def forward_vpt_with_prompts(_visual, images):
            return self._visual_forward_with_prompts(images)

        self.clip_model.visual.forward_vpt = forward_vpt_with_prompts.__get__(
            self.clip_model.visual, type(self.clip_model.visual)
        )

        # 4. **新增**: 可学习的RD查询投影层 (ZegCLIP [1] Fig 2)
        # ZegCLIP [1] 使用 [t ⊙ g, t] (2*D 维) -> 线性层 -> D 维
        # 我们获取CLIP的特征维度 (例如 512)
        clip_embed_dim = self._infer_clip_embed_dim()
        self.clip_embed_dim = clip_embed_dim

        # 我们的RD查询将是 [t ⊙ g, t]，所以输入是 2 * clip_embed_dim
        self.rd_query_proj = nn.Linear(2 * clip_embed_dim, clip_embed_dim).to(self.device)

    def _visual_forward_with_prompts(self, images: torch.Tensor):
        """在 BiomedCLIP 图像分支中注入 VPT-Deep prompts."""
        visual = self.clip_model.visual
        trunk = self.visual_trunk

        images = preprocess_tensor_images(images, self.preprocess, self.device)
        dtype = next(trunk.parameters()).dtype
        images = images.to(dtype=dtype)

        # 1. Patch + CLS
        x = trunk.patch_embed(images)
        if hasattr(trunk, 'cls_token') and trunk.cls_token is not None:
            cls_tokens = trunk.cls_token.to(x.dtype).expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        if getattr(trunk, 'pos_embed', None) is not None:
            x = x + trunk.pos_embed[:, : x.size(1), :]
        x = trunk.pos_drop(x)
        if hasattr(trunk, 'patch_drop'):
            x = trunk.patch_drop(x)

        norm_pre = getattr(trunk, 'norm_pre', None)
        if norm_pre is not None and not isinstance(norm_pre, nn.Identity):
            x = norm_pre(x)

        # 2. Transformer 层逐层注入 prompts
        for i, block in enumerate(trunk.blocks):
            prompts = self.prompt_learner(x, i)
            x_with_prompts = torch.cat(
                (x[:, :1, :], prompts, x[:, 1:, :]), dim=1
            )
            x_with_prompts = block(x_with_prompts)
            x = torch.cat(
                (
                    x_with_prompts[:, :1, :],
                    x_with_prompts[:, 1 + self.prompt_learner.num_prompts :, :],
                ),
                dim=1,
            )

        x = trunk.norm(x)

        tokens = x  # [B, 1+N, embed_dim]
        pooled = tokens[:, 0, :]
        if hasattr(trunk, 'fc_norm') and trunk.fc_norm is not None and not isinstance(trunk.fc_norm, nn.Identity):
            pooled = trunk.fc_norm(pooled)

        patch_tokens = tokens[:, 1:, :]

        # 3. 使用视觉头映射到 CLIP 公共嵌入空间
        g_features = visual.head(pooled)  # [B, D]
        patch_features = visual.head(patch_tokens)  # [B, N, D]

        g_features = F.normalize(g_features, dim=-1)
        patch_features = F.normalize(patch_features, dim=-1)

        return g_features, patch_features

    def _infer_clip_embed_dim(self) -> int:
        """推断 CLIP 公共嵌入维度 (通常为 512)."""
        visual = self.clip_model.visual
        visual_head = getattr(visual, 'head', None)
        embed_dim = None

        # 优先从视觉头中获取 out_features
        if isinstance(visual_head, nn.Sequential):
            for module in reversed(visual_head):
                if hasattr(module, 'out_features'):
                    embed_dim = module.out_features
                    break
        elif hasattr(visual_head, 'out_features'):
            embed_dim = visual_head.out_features

        # 退而求其次，从文本投影层中获取
        if embed_dim is None and hasattr(self.clip_model, 'text'):
            text_module = self.clip_model.text
            text_proj = getattr(text_module, 'proj', None)
            if isinstance(text_proj, nn.Sequential):
                for module in reversed(text_proj):
                    if hasattr(module, 'out_features'):
                        embed_dim = module.out_features
                        break
            elif hasattr(text_proj, 'out_features'):
                embed_dim = text_proj.out_features

        if embed_dim is None:
            raise RuntimeError("无法推断 BiomedCLIP 嵌入维度，检查视觉头/文本投影结构是否符合预期")

        return int(embed_dim)

    def encode_text(self, text_tokens):
        # 包装器，用于编码文本
        with torch.no_grad(): # 文本编码器始终冻结
            text_feats = self.clip_model.encode_text(text_tokens)
            text_feats_norm = F.normalize(text_feats, dim=-1)
        return text_feats_norm

    def forward(self, image_tensor, text_anchors):
        """
        模型的主 forward pass
        
        Args:
            image_tensor (torch.Tensor): [B, 3, H, W] 预处理后的图像
            text_anchors (torch.Tensor): [K, D] K个文本锚点 (例如 E_subset_A, E_bg)
        
        Returns:
            H_semantic_maps (torch.Tensor): [B, K, N] 最终的语义图 (N=196)
            patch_features (torch.Tensor): [B, N, D] 原始 patch 特征 (用于 L_invariance)
        """
        # 1. 提取图像特征
        # g_features: [B, D] (全局 [CLS] 特征)
        # patch_features: [B, N, D] (N=196)
        g_features, patch_features = self.clip_model.visual.forward_vpt(image_tensor)
        
        B, N, D = patch_features.shape
        K, _ = text_anchors.shape
        
        # 2. 实现 ZegCLIP [1] 的 RD (关系描述符)
        # g: [B, D] -> [B, 1, D] -> [B, K, D]
        g_expanded = g_features.unsqueeze(1).expand(-1, K, -1)
        # t: [K, D] -> [1, K, D] -> [B, K, D]
        t_expanded = text_anchors.unsqueeze(0).expand(B, -1, -1)
        
        # 关系 r = t ⊙ g
        r = t_expanded * g_expanded # [B, K, D]
        
        # T_hat = concat[r, t]
        T_hat_concat = torch.cat([r, t_expanded], dim=-1) # [B, K, 2*D]
        
        # 3. RD 投影
        # [B, K, 2*D] -> [B, K, D]
        # **rd_query_proj 是我们新的可学习参数**
        Q_dynamic = self.rd_query_proj(T_hat_concat) # 图像感知的动态查询
        Q_dynamic_norm = F.normalize(Q_dynamic, dim=-1) # [B, K, D]
        
        # 4. 计算最终语义图 (交叉注意力)
        # Q @ K^T
        # Q_dynamic_norm: [B, K, D]
        # patch_features: [B, N, D] -> transpose -> [B, D, N]
        # H_semantic_maps: [B, K, N]
        H_semantic_maps = Q_dynamic_norm @ patch_features.transpose(1, 2)
        
        # 返回语义图 (用于 L_semantic) 和原始 patch 特征 (用于 L_invariance)
        return H_semantic_maps, patch_features

    def save_prompts(self, filepath):
        """保存所有可学习的参数 (VPT + RD投影层)"""
        save_dict = {
            'prompt_learner': self.prompt_learner.state_dict(),
            'rd_query_proj': self.rd_query_proj.state_dict()
        }
        torch.save(save_dict, filepath)

    def load_prompts(self, filepath):
        """加载已训练的参数"""
        state_dict = torch.load(filepath, map_location=self.device)
        self.prompt_learner.load_state_dict(state_dict['prompt_learner'])
        self.rd_query_proj.load_state_dict(state_dict['rd_query_proj'])
        print(f"Loaded VPT prompts and RD projector from {filepath}")
