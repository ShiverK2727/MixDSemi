# VPT+TPT 训练脚本使用说明

## 概述
此脚本集成了 `biomedclip_vpt_tpt_seg.py` 中的 VPT+TPT 模型到训练循环中，实现了：
1. 使用 `text_sampler_an.py` 加载文本描述
2. 预编码类别文本特征（使用 TPT prompts）
3. 在训练循环中前向传播并下采样标签到 14×14

## 主要修改

### 1. TextSampler 替换
- **旧**: `from utils.text_sampler_v2 import TextSampler`
- **新**: `from utils.text_sampler_an import TextSampler`
- **返回值**: `(per_class_lists, flat_list)` 而非三元组
- **移除**: `text_subsets` 和相关的子集采样逻辑

### 2. 配置参数更新

#### 删除的参数
- `--clip_loss_mv_anchor_weight`
- `--clip_loss_sw_reg_weight`
- `--biomedclip_num_prompts`
- `--biomedclip_embed_dim`
- `--biomedclip_init_std`
- `--biomedclip_prompt_scale_init`
- `--biomedclip_disable_scale`
- `--text_num_subsets`

#### 新增的参数
```bash
--visual_num_prompts 4        # VPT-Deep 每层的 prompt 数量
--text_num_prompts 4          # TPT CoOp-style context prompt 数量
--vpt_tpt_lr 1e-4            # VPT+TPT 优化器学习率
--vpt_tpt_weight_decay 1e-2  # 权重衰减
--freeze_backbone            # 冻结 BiomedCLIP 主干（默认启用）
```

### 3. 预编码文本特征

在训练开始前，脚本会：
```python
# 为每个类别（K个）预编码文本特征
class_text_features: Tensor  # [K, D]
# 其中：
# - K = len(args.class_text)（包括背景）
# - D = 512 (BiomedCLIP 特征维度)
# - 背景类使用类名本身
# - 前景类使用 per_class_lists[i-1] 的所有文本取平均
```

### 4. 训练循环前向过程

每次迭代会进行三次前向传播：
```python
# 1. Labeled weak 增强图像
lb_outputs = vpt_tpt_model(lb_images_preprocessed, args.class_text)
# 输出:
# - H_semantic_maps: [B_lb, K, 196]  # 14×14 语义图
# - patch_features: [B_lb, 196, D]   # patch 特征
# - W_robust: [K, D]                  # 文本原型

# 2. Unlabeled weak 增强图像
ulb_weak_outputs = vpt_tpt_model(ulb_images_weak_preprocessed, args.class_text)

# 3. Unlabeled strong 增强图像
ulb_strong_outputs = vpt_tpt_model(ulb_images_strong_preprocessed, args.class_text)
```

### 5. 可用的变量（用于设计损失）

在训练循环中，以下变量可用于损失计算：
```python
# Labeled 数据
lb_outputs["H_semantic_maps"]     # [B_lb, K, 196] 语义图
lb_outputs["patch_features"]      # [B_lb, 196, D] patch 特征
lb_sample['unet_label_14']        # [B_lb, K, 14, 14] 下采样标签（soft）

# Unlabeled 数据
ulb_weak_outputs                  # weak 增强的输出
ulb_strong_outputs                # strong 增强的输出

# 预编码特征
class_text_features               # [K, D] 类别文本特征
```

## 使用示例

### 基本运行
```bash
python train_MiDSS_MARK3_CLIP_VPT_TPT_1_SEMI.py \
    --dataset prostate \
    --lb_num 40 \
    --lb_domain 1 \
    --visual_num_prompts 4 \
    --text_num_prompts 4 \
    --vpt_tpt_lr 1e-4 \
    --max_iterations 60000 \
    --save_model
```

### 自定义配置
```bash
python train_MiDSS_MARK3_CLIP_VPT_TPT_1_SEMI.py \
    --dataset MNMS \
    --lb_num 80 \
    --lb_domain 1 \
    --visual_num_prompts 8 \
    --text_num_prompts 8 \
    --vpt_tpt_lr 5e-5 \
    --vpt_tpt_weight_decay 1e-3 \
    --label_bs 4 \
    --unlabel_bs 4 \
    --amp 1 \
    --save_name my_experiment
```

## 下一步：设计损失函数

在训练循环中的 `# TODO: 在这里添加你的损失函数` 位置，你可以设计损失，例如：

```python
# 示例：语义图与下采样标签的交叉熵损失
if 'unet_label_14' in lb_sample:
    # lb_outputs["H_semantic_maps"]: [B, K, 196]
    # lb_sample['unet_label_14']: [B, K, 14, 14] -> reshape to [B, K, 196]
    lb_label_flat = lb_sample['unet_label_14'].view(B, K, -1)  # [B, K, 196]
    
    # 计算每个 patch 的损失
    loss_seg = F.cross_entropy(
        lb_outputs["H_semantic_maps"].permute(0, 2, 1),  # [B, 196, K]
        lb_label_flat.argmax(dim=1),  # [B, 196]
        reduction='mean'
    )
    
    loss_total = loss_seg

# 或者使用 soft labels 的 KL 散度
# loss_kl = F.kl_div(
#     F.log_softmax(lb_outputs["H_semantic_maps"], dim=1),
#     lb_label_flat,
#     reduction='batchmean'
# )
```

## 输出和保存

训练完成后，脚本会保存：
- `vpt_tpt_final_weights.pth`: VPT 和 TPT 的 prompt 参数
- `log.txt`: 训练日志
- `training_config.json`: 完整的训练配置

## 注意事项

1. **标签映射**: 确保数据集的标签已正确映射到 [0, K-1]，避免 255 等原始值
2. **内存使用**: VPT-Deep 会在每层注入 prompts，增加少量内存开销
3. **预编码特征**: `class_text_features` 在训练开始前固定，如需动态更新需修改代码
4. **TPT prompts**: 文本 prompts 在每次前向时都会被应用，确保模型学习域不变的语义原型

## 类别文本配置

每个数据集的 `args.class_text` 预设如下：
- **Fundus**: `['background', 'optic cup', 'optic disc']`
- **Prostate**: `['background', 'prostate']`
- **MNMS**: `['background', 'left ventricle', 'left ventricle myocardium', 'right ventricle']`
- **BUSI**: `['background', 'breast tumor']`

这些类别文本将用于：
1. 预编码类别特征
2. 每次前向传播时通过 TPT 编码
