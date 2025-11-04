# BiomedCLIP VPT+TPT 分割模型 - 代码分析与修复总结

## ✅ 任务完成情况

您的代码 `/app/MixDSemi/SynFoCLIP/code/biomedclip_vpt_tpt_seg.py` **已成功完成所有三个任务**：

### 1. ✅ 文本部分加入 Learnable Token
- **实现方式**: `TextPromptLearner` 类管理可学习上下文向量 (C_l)
- **结构**: [CLS, **C_l**, Class, SEP] (CoOp风格)
- **维度**: 4个prompts × 768维 (BERT内部维度)
- **位置**: 在 `encode_text_with_prompts()` 方法中与类别字符串共同编码

### 2. ✅ 视觉编码器每层加入 Learnable Token  
- **实现方式**: `VisualPromptLearner` 类管理视觉prompts (V_l)
- **结构**: VPT-Deep 架构，每层 Transformer 注入独立的prompts
- **维度**: 12层 × 4个prompts × 768维
- **位置**: 在 `_visual_forward_with_prompts()` 方法中逐层注入

### 3. ✅ 支持 Tensor 图像输入
- **实现方式**: `preprocess_tensor_images()` 函数
- **功能**: 
  - 自动检测输入范围 ([0,1], [-1,1], 或 [0,255])
  - 自动调整尺寸到 224×224
  - 应用 CLIP 标准化 (mean/std)
  - 支持单通道自动扩展到3通道

---

## 🐛 主要修复问题

### 问题1: `text.proj` 不是 Tensor 而是 Sequential
**报错**: `AttributeError: 'Sequential' object has no attribute 'shape'`

**原因**: BiomedCLIP 的 `text.proj` 是一个 Sequential 容器:
```
Sequential(
  Linear(768 → 640),
  GELU(),
  Linear(640 → 512)
)
```

**修复**: 使用 `model.text.output_dim` 或 `proj[-1].out_features` 获取维度

### 问题2: 模型对象没有 `.device` 属性
**报错**: `AttributeError: 'CustomTextCLIP' object has no attribute 'device'`

**修复**: 添加 `@property` 方法从参数获取设备:
```python
@property
def device(self) -> torch.device:
    return next(self.model.parameters()).device
```

### 问题3: `pos_drop` 和 `patch_drop` 是模块方法，不是 Tensor 方法
**报错**: `AttributeError: 'Tensor' object has no attribute 'pos_drop'`

**修复**: 改为 `trunk.pos_drop(x)` 而非 `x.pos_drop(x)`

### 问题4: BiomedCLIP 使用 BERT Tokenizer，没有 `sot_token_id`
**报错**: `AttributeError: 'HFTokenizer' object has no attribute 'sot_token_id'`

**关键发现**: 
- BiomedCLIP 使用 **HFTokenizer (BERT风格)**
- 使用 **CLS token (ID=2)** 而非 SOT
- 使用 **SEP token (ID=3)** 而非 EOT
- 内部维度是 **768** (BERT hidden size)，而非 512

**修复**: 完全重写 `encode_text_with_prompts()`:
1. 使用 `transformer.embeddings.word_embeddings` 获取嵌入
2. 构建 [CLS, C_l, Class, SEP] 序列
3. 添加 BERT 的 position_embeddings 和 token_type_embeddings
4. 通过 `transformer.encoder` 前向传播
5. 使用 `pooler` 池化（传递完整的 encoder_outputs）
6. 通过 `proj` 投影到 CLIP 空间 (512维)

### 问题5: TextPrompt 维度不匹配
**报错**: `RuntimeError: Expected size 768 but got size 512`

**修复**: 将 `TextPromptConfig.embed_dim` 从 512 改为 768

---

## 📊 最终运行结果

```
✓ VPT_TPT_CLIP_Seg (VPT-Deep + TPT-CoOp) 初始化成功
  - 视觉 Prompts (V_l, Deep): 36,864 参数
  - 文本 Prompts (C_l, CoOp): 3,072 参数
  - 总可训练参数: 39,936 参数

输出 'H_semantic_maps' (分割图) shape: torch.Size([2, 2, 196])
输出 'patch_features' (用于一致性) shape: torch.Size([2, 196, 512])
计算总损失: 3.9998

✓ 视觉提示 (V_l) 梯度已计算 (grad.norm = 2.56)
✓ 文本提示 (C_l) 梯度已计算 (grad.norm = 2.60)
✓ CLIP 主干已冻结 (无梯度)
```

---

## 🔑 关键技术点

### BiomedCLIP 架构特点
1. **视觉编码器**: ViT-B (Timm 实现)
   - 12层 Transformer
   - 768 维内部特征
   - 输出投影到 512 维 CLIP 空间

2. **文本编码器**: BERT (HuggingFace 实现)
   - 768 维内部特征  
   - Sequential 投影层: 768→640→512
   - 使用 CLS/SEP 而非 SOT/EOT

### Prompt 设计
- **VPT-Deep**: 每层独立的可学习 prompts，注入到 Transformer 的 sequence 末尾
- **TPT-CoOp**: 单层可学习上下文，插入到 [CLS] 和 [Class] 之间
- **总参数**: 仅 ~40K，相比完整模型 (~100M+) 极轻量

### 训练策略
- **冻结主干**: 仅训练 prompts，实现 PEFT (Parameter-Efficient Fine-Tuning)
- **梯度验证**: 确认只有 prompts 有梯度，主干无梯度
- **权重保存**: 支持单独保存/加载 prompts

---

## 📝 使用示例

```python
# 1. 构建模型
from biomedclip_vpt_tpt_seg import build_vpt_tpt_seg_model

model, preprocess, tokenizer = build_vpt_tpt_seg_model(
    model_path="/root/models/BiomedCLIP",
    device="cuda",
    visual_num_prompts=4,
    text_num_prompts=4,
)

# 2. 准备输入
images = torch.rand(2, 3, 224, 224).cuda()  # 支持 tensor 输入
text_list = ["prostate", "background"]      # 字符串列表

# 3. 前向传播
outputs = model(images, text_list)
H_maps = outputs["H_semantic_maps"]  # [2, 2, 196] 分割图
patches = outputs["patch_features"]   # [2, 196, 512] patch特征

# 4. 保存 prompts
model.save_all_prompts("./my_prompts.pth")

# 5. 加载 prompts
model.load_all_prompts("./my_prompts.pth")
```

---

## ✨ 代码优势

1. **参数高效**: 仅训练 40K 参数 (~0.04% 的完整模型)
2. **域不变性**: VPT 学习域不变的视觉特征，TPT 学习鲁棒的语义原型
3. **即插即用**: 可以轻松切换不同的 prompts 权重
4. **GPU 友好**: 小参数量，训练速度快，显存占用低

---

## 📚 参考

- **VPT**: Visual Prompt Tuning (Deep variant)
- **CoOp**: Learning to Prompt for Vision-Language Models
- **BiomedCLIP**: A Multimodal Biomedical Foundation Model
- **PEFT**: Parameter-Efficient Fine-Tuning

测试通过时间: 2025-11-04
