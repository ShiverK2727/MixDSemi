""" 带有 FiLM 调制的 U-Net 模型部分 """

import torch
import torch.nn as nn
import torch.nn.functional as F
# 支持将此文件作为模块（包）或脚本直接运行。
# 当以脚本方式执行（python d_unet_model.py）时，相对导入如
# `from .unet_parts import *` 可能会因“attempted relative import with no
# known parent package”而失败。先尝试相对导入，失败时回退到绝对导入。
try:
    from .unet_parts import *
except Exception:
    # Fallback for direct script execution
    from unet_parts import *
class AffineGen(nn.Module): # 重命名以反映通用性
    """
    从全局 u_spec 生成仿射参数 (δγ, δβ) 或 (γ, β)。
    支持低秩投影或直接映射两种实现。
    """
    def __init__(self, d: int, channels: int, 
                 use_low_rank: bool = True, # 新增開關
                 r: int = 8, alpha: float = 0.1, output_delta: bool = True):
        super().__init__()
        self.use_low_rank = use_low_rank
        self.output_delta = output_delta
        self.alpha = alpha

        if use_low_rank:
            # 低秩版本：d -> r -> 2C
            self.proj = nn.Linear(d, r, bias=False)
            self.head = nn.Linear(r, 2 * channels, bias=True)
            print(f"  AffineGen: Using Low-Rank (d={d}, r={r}, C={channels})")
        else:
            # 直接映射版本：d -> 2C
            self.direct_map = nn.Linear(d, 2 * channels, bias=True)
            print(f"  AffineGen: Using Direct Mapping (d={d}, C={channels})")

    def forward(self, u_spec):  # u_spec: [B, d]
        if self.use_low_rank:
            z = self.proj(u_spec)           # [B, r]
            affine_params = self.head(torch.tanh(z)) # [B, 2C]
        else:
            affine_params = self.direct_map(u_spec) # [B, 2C]

        gamma, beta = affine_params.chunk(2, dim=1) # 各 [B, C]

        if self.output_delta:
            # 输出增量（用于 Conditional Affine 和 CBN）
            gamma = torch.tanh(gamma).unsqueeze(-1).unsqueeze(-1) * self.alpha
            beta  = torch.tanh(beta ).unsqueeze(-1).unsqueeze(-1) * self.alpha
        else:
            # 输出完整参数（用于 FiLM）
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)
            beta = beta.unsqueeze(-1).unsqueeze(-1)
             
        return gamma, beta

# ===================================================================
# 调制类型 1: FiLM（替换式）
# ===================================================================
# FiLMLayer, FiLMDoubleConv, FiLMDown 保持不变（它们不使用 AffineGen）
class FiLMLayer(nn.Module):
    def __init__(self, style_dim, num_channels):
        super().__init__()
        self.generator = nn.Linear(style_dim, num_channels * 2)
    def forward(self, style_code):
        gamma_beta = self.generator(style_code)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        return gamma.unsqueeze(-1).unsqueeze(-1), beta.unsqueeze(-1).unsqueeze(-1)

class FiLMDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, style_dim=512):
        super().__init__()
        if not mid_channels: mid_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels, affine=False)
        self.film1 = FiLMLayer(style_dim, mid_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, affine=False)
        self.film2 = FiLMLayer(style_dim, out_channels)
        self.relu2 = nn.ReLU(inplace=True)
    def forward(self, x, u_spec):
        x = self.conv1(x); x_norm = self.bn1(x)
        gamma1, beta1 = self.film1(u_spec); x = (1 + gamma1) * x_norm + beta1
        x = self.relu1(x); x = self.conv2(x); x_norm = self.bn2(x)
        gamma2, beta2 = self.film2(u_spec); x = (1 + gamma2) * x_norm + beta2
        x = self.relu2(x); return x

class FiLMDown(nn.Module):
    def __init__(self, in_channels, out_channels, style_dim=512):
        super().__init__(); self.maxpool = nn.MaxPool2d(2)
        self.conv = FiLMDoubleConv(in_channels, out_channels, style_dim=style_dim)
    def forward(self, x, u_spec): return self.conv(self.maxpool(x), u_spec)


# ===================================================================
# 调制类型 2: Conditional Affine（叠加式）
# ===================================================================
class ConditionalAffineDoubleConv(nn.Module):
    # 这个模块现在接收 AffineGen 实例
    def __init__(self, in_channels, out_channels, gen1: AffineGen, gen2: AffineGen, mid_channels=None):
        super().__init__()
        if not mid_channels: mid_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels, affine=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, affine=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.gen1 = gen1
        self.gen2 = gen2
    def forward(self, x, u_spec):
        x = self.conv1(x); y_bn1 = self.bn1(x)
        dgamma1, dbeta1 = self.gen1(u_spec); x = y_bn1 * (1 + dgamma1) + dbeta1
        x = self.relu1(x); x = self.conv2(x); y_bn2 = self.bn2(x)
        dgamma2, dbeta2 = self.gen2(u_spec); x = y_bn2 * (1 + dgamma2) + dbeta2
        x = self.relu2(x); return x

class ConditionalAffineDown(nn.Module):
    def __init__(self, in_channels, out_channels, gen_pair_factory):
        super().__init__(); self.pool = nn.MaxPool2d(2)
        gen1, gen2 = gen_pair_factory(out_channels) # 工廠現在創建 AffineGen
        self.block = ConditionalAffineDoubleConv(in_channels, out_channels, gen1, gen2)
    def forward(self, x, u_spec): return self.block(self.pool(x), u_spec)

# ===================================================================
# 调制类型 3: CBN（加性参数调制）
# ===================================================================
class CBNDoubleConv(nn.Module):
    # 这个模块现在接收 AffineGen 实例
    def __init__(self, in_channels, out_channels, gen1: AffineGen, gen2: AffineGen, mid_channels=None):
        super().__init__()
        if not mid_channels: mid_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels, affine=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, affine=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.gen1 = gen1
        self.gen2 = gen2
    def forward(self, x, u_spec):
        # Block 1
        x = self.conv1(x)
        gamma_bn1 = self.bn1.weight; beta_bn1 = self.bn1.bias
        dgamma1, dbeta1 = self.gen1(u_spec)
        gamma_final1 = gamma_bn1.view(1, -1, 1, 1) + dgamma1 
        beta_final1 = beta_bn1.view(1, -1, 1, 1) + dbeta1
        x_normalized1 = F.batch_norm(x, self.bn1.running_mean, self.bn1.running_var, 
                                     weight=None, bias=None, training=self.bn1.training, 
                                     momentum=self.bn1.momentum, eps=self.bn1.eps)
        x = gamma_final1 * x_normalized1 + beta_final1; x = self.relu1(x)
        # Block 2
        x = self.conv2(x)
        gamma_bn2 = self.bn2.weight; beta_bn2 = self.bn2.bias
        dgamma2, dbeta2 = self.gen2(u_spec)
        gamma_final2 = gamma_bn2.view(1, -1, 1, 1) + dgamma2
        beta_final2 = beta_bn2.view(1, -1, 1, 1) + dbeta2
        x_normalized2 = F.batch_norm(x, self.bn2.running_mean, self.bn2.running_var, 
                                     weight=None, bias=None, training=self.bn2.training, 
                                     momentum=self.bn2.momentum, eps=self.bn2.eps)
        x = gamma_final2 * x_normalized2 + beta_final2; x = self.relu2(x)
        return x

class CBNDown(nn.Module):
    def __init__(self, in_channels, out_channels, gen_pair_factory):
        super().__init__(); self.pool = nn.MaxPool2d(2)
        gen1, gen2 = gen_pair_factory(out_channels) # 工廠現在創建 AffineGen
        self.block = CBNDoubleConv(in_channels, out_channels, gen1, gen2)
    def forward(self, x, u_spec): return self.block(self.pool(x), u_spec)


class DistillationUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False,
                 style_dim=512,
                 modulation_type='conditional',  # 'none', 'film', 'conditional', 'cbn'
                 # 控制头
                 use_invariance_head=True,
                 use_style_head=True,
                 ):
        super(DistillationUNet, self).__init__()

        # --- 参数设置 ---
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.style_dim = style_dim if use_style_head else 0
        self.modulation_type = modulation_type if use_style_head else 'none'
        self.use_invariance_head = use_invariance_head
        self.use_style_head = use_style_head

        # --- 构建编码器 ---
        self.inc = DoubleConv(n_channels, 64)

        # 直接使用原始 Down（不使用任何 FiLM/Conditional/CBN 调制）
        DownBlock = lambda in_c, out_c, *args: Down(in_c, out_c)

        factor = 2 if bilinear else 1

        # ==================== 修改点 1 ====================
        # 将调制应用于 down1, down2, down3
        self.down1 = DownBlock(64, 128, 0)           # <-- 使用 alphas[0] 调制
        self.down2 = DownBlock(128, 256, 1)          # <-- 使用 alphas[1] 调制
        self.down3 = DownBlock(256, 512, 2)          # <-- 使用 alphas[2] 调制
        self.down4 = Down(512, 1024 // factor)       # <-- 恢复为标准 Down，不调制
        # ================================================

        # --- 知识蒸馏头 ---
        self.style_head = None
        if self.use_style_head:
            self.style_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                # ==================== 修改点 2 ====================
                # inc 的输出 (x1) 是 64 通道，而不是 128
                nn.Linear(64, style_dim), nn.ReLU(inplace=True),
                # ================================================
                nn.Linear(style_dim, style_dim)
            )

        self.invariance_head = None
        clip_inv_dim = style_dim 
        if self.use_invariance_head:
            self.invariance_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Linear(1024 // factor, clip_inv_dim), nn.ReLU(inplace=True),
                nn.Linear(clip_inv_dim, clip_inv_dim)
            )

    # --- 解码器 ---（保持不变）
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        # 编码器部分：包括初始卷积和所有下采样块
        self.encoder_modules = nn.ModuleList([
            self.inc, self.down1, self.down2, self.down3, self.down4
        ])

        # 解码器部分：包括所有上采样块和最终输出卷积
        self.decoder_modules = nn.ModuleList([
            self.up1, self.up2, self.up3, self.up4, self.outc
        ])

        # 蒸馏头部分：它们在概念上与编码器相关
        self.head_modules = nn.ModuleList()
        if self.use_invariance_head and self.invariance_head is not None:
            self.head_modules.append(self.invariance_head)
        if self.use_style_head and self.style_head is not None:
            self.head_modules.append(self.style_head)

    def forward(self, x, training=False):
        # --- 提取用于风格头的浅层特征 ---
        x1 = self.inc(x)
        
        # ==================== 修改点 3 ====================
        # --- 从 x1（inc 的输出）生成 u_spec（若启用） ---
        u_spec = None
        if self.use_style_head and self.style_head is not None:
            u_spec = self.style_head(x1) # <-- 从 x1 生成
            u_spec = F.normalize(u_spec, dim=1)
        
        # --- 运行编码器 ---
        # 编码器（不带调制）
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # ================================================

        # --- 生成 u_inv（若启用） ---
        u_inv = None
        if self.use_invariance_head and self.invariance_head is not None:
            u_inv = self.invariance_head(x5)
            u_inv = F.normalize(u_inv, dim=1)

    # --- 解码器 ---
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        if training:
            return logits, u_inv, u_spec
        else:
            return logits
# --- 示例用法 ---
if __name__ == '__main__':
    
    print("--- 测试 Conditional Affine (Low Rank) ---")
    model_cond_lr = DistillationUNet(
        n_channels=3, n_classes=1, style_dim=512, 
        modulation_type='conditional', use_low_rank=True, r_rank=8, 
        use_invariance_head=True, use_style_head=True
    )
    dummy_input = torch.randn(2, 3, 224, 224)
    out_lr, inv_lr, spec_lr = model_cond_lr(dummy_input, training=True)
    print(f"Logits: {out_lr.shape}, u_inv: {inv_lr.shape}, u_spec: {spec_lr.shape}")
    
    print("\n--- 测试 Conditional Affine (Direct Map) ---")
    model_cond_direct = DistillationUNet(
        n_channels=3, n_classes=1, style_dim=512, 
        modulation_type='conditional', use_low_rank=False, # <-- 设为 False
        use_invariance_head=True, use_style_head=True
    )
    out_direct, inv_direct, spec_direct = model_cond_direct(dummy_input, training=True)
    print(f"Logits: {out_direct.shape}, u_inv: {inv_direct.shape}, u_spec: {spec_direct.shape}")

    print("\n--- 测试 FiLM ---")
    model_film = DistillationUNet(
        n_channels=3, n_classes=1, style_dim=512, 
        modulation_type='film', 
        use_invariance_head=True, use_style_head=True
    )
    out_film, inv_film, spec_film = model_film(dummy_input, training=True)
    print(f"Logits: {out_film.shape}, u_inv: {inv_film.shape}, u_spec: {spec_film.shape}")

    print("\n--- 测试 CBN (Low Rank) ---")
    model_cbn_lr = DistillationUNet(
        n_channels=3, n_classes=1, style_dim=512, 
        modulation_type='cbn', use_low_rank=True, r_rank=8, 
        use_invariance_head=True, use_style_head=True
    )
    out_cbn_lr, inv_cbn_lr, spec_cbn_lr = model_cbn_lr(dummy_input, training=True)
    print(f"Logits: {out_cbn_lr.shape}, u_inv: {inv_cbn_lr.shape}, u_spec: {spec_cbn_lr.shape}")

    print("\n--- 测试原始 U-Net ---")
    model_orig = DistillationUNet(
        n_channels=3, n_classes=1, 
        modulation_type='none', 
        use_invariance_head=False, use_style_head=False
    )
    out_orig, inv_orig, spec_orig = model_orig(dummy_input, training=True)
    print(f"Logits: {out_orig.shape}, u_inv: {inv_orig}, u_spec: {spec_orig}")