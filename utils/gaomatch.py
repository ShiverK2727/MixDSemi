import torch
from torch.nn import functional as F
import torch.nn as nn


class gapmatch_targeted_v1:
    """
    一个基于 GAPv4 逻辑的定向 GapMatch 版本。
    它接收两组不同的参数（编码器/解码器），并对它们分别应用
    GapMatch 扰动和梯度合并。
    
    它依赖调用者（训练循环）通过梯度累积来正确填充
    param.grad（即 g1 和 g2）。
    """
    def __init__(self, encoder_params, decoder_params, gamma=0.5):
        # 将参数列表转换为集合，以便快速查找（如果需要）
        # 但我们这里直接迭代列表
        self.encoder_params = list(encoder_params)
        self.decoder_params = list(decoder_params)
        self.gamma = gamma

        # g1 (velocity) 和 theta (backup) 分开存储
        self.velocity_enc = {}
        self.velocity_dec = {}
        self.backup = {} # backup 可以是全局的，因为键是唯一的 param 对象

    def _get_norm(self, g):
        # 辅助函数，用于计算范数，避免除零
        return torch.norm(g, p=2) + 1e-8

    def perturb(self, epsilon=0.1):
        """
        在 g1 已经计算并存储在 .grad 中之后调用。
        保存 g1，备份 theta，并应用 theta* = theta + r(g1)。
        """
        # --- 处理解码器 ---
        for param in self.decoder_params:
            if param.grad is None:
                continue
            
            g1_dec = param.grad.data.clone()
            self.velocity_dec[param] = g1_dec # 保存 g1
            self.backup[param] = param.data.clone() # 保存 theta
            
            r_at = epsilon * g1_dec / self._get_norm(g1_dec)
            param.data.add_(r_at) # 应用 theta*

        # --- 处理编码器 ---
        for param in self.encoder_params:
            if param.grad is None:
                continue
            
            g1_enc = param.grad.data.clone()
            self.velocity_enc[param] = g1_enc # 保存 g1
            self.backup[param] = param.data.clone() # 保存 theta
            
            # 编码器的扰动 r(g1) 是基于其 *自己的* g1 (可能是复合梯度)
            r_at = epsilon * g1_enc / self._get_norm(g1_enc)
            param.data.add_(r_at) # 应用 theta*

    def restore(self):
        """
        在 g2 已经计算并存储在 .grad 中之后调用。
        合并 g_u = (1-g)*g1 + g*g2，并恢复 theta。
        """
        # --- 处理解码器 ---
        for param, g1 in self.velocity_dec.items():
            if param.grad is None:
                # 如果 g2 由于某种原因不存在，则跳过
                if param in self.backup:
                    param.data = self.backup[param] # 仍需恢复 theta
                continue

            g2 = param.grad.data.clone()
            
            # GAPv4 合并逻辑: g_u = gamma*g2 + (1-gamma)*g1
            # (注意：我们在 train.py 中使用的是 (1-g)*g1 + g*g2，这里保持一致)
            param.grad.data = self.gamma * g2 + (1 - self.gamma) * g1
            
            # 恢复 theta
            param.data = self.backup[param]

        # --- 处理编码器 ---
        for param, g1 in self.velocity_enc.items():
            if param.grad is None:
                if param in self.backup:
                    param.data = self.backup[param]
                continue
                
            g2 = param.grad.data.clone()
            param.grad.data = self.gamma * g2 + (1 - self.gamma) * g1
            param.data = self.backup[param]

        # 清理状态
        self.velocity_enc.clear()
        self.velocity_dec.clear()
        self.backup.clear()

